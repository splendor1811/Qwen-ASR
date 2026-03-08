#!/usr/bin/env python3
"""Main training entry point for Qwen3-ASR Vietnamese finetuning."""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import TrainingArguments

from src.config import load_config
from src.data import load_jsonl_dataset, DataCollatorForQwen3ASRFinetune
from src.evaluation.metrics import compute_wer
from src.evaluation.normalize_vi import normalize_vietnamese
from src.model.loader import load_model_and_processor
from src.model.freezing import apply_freezing
from src.model.lora import apply_lora
from src.training.trainer import Qwen3ASRTrainer
from src.training.callbacks import MakeCheckpointInferableCallback, WandbAudioCallback
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-ASR for Vietnamese ASR")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None, help="Override DeepSpeed config")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    return parser.parse_args()


def preprocess_logits_for_metrics(logits, labels):
    """Argmax logits to token IDs before storing (avoids OOM from full vocab logits)."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def build_compute_metrics(processor):
    """Build a compute_metrics function for the Trainer."""
    tokenizer = processor.tokenizer

    def compute_metrics_fn(eval_preds):
        pred_ids, label_ids = eval_preds

        # Replace -100 padding with pad token for decoding
        pad_id = tokenizer.pad_token_id or 0
        pred_ids[pred_ids == -100] = pad_id
        label_ids[label_ids == -100] = pad_id

        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        predictions = [normalize_vietnamese(p) for p in predictions]
        references = [normalize_vietnamese(r) for r in references]

        pairs = [(r, h) for r, h in zip(references, predictions) if r.strip()]
        if not pairs:
            return {"wer": 0.0}

        from jiwer import wer
        refs, hyps = zip(*pairs)
        wer_score = wer(list(refs), list(hyps))
        return {"wer": wer_score}

    return compute_metrics_fn


def main():
    args = parse_args()
    setup_logging()

    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override deepspeed if provided via CLI
    if args.deepspeed:
        config.training.deepspeed = args.deepspeed

    # Load model and processor
    logger.info("Loading model and processor...")
    device_map = None if config.training.deepspeed else "auto"
    model, processor = load_model_and_processor(config.model, device_map=device_map)

    # Apply freezing before LoRA
    logger.info("Applying freezing strategy...")
    apply_freezing(model, config.freeze)

    # Apply LoRA
    logger.info("Applying LoRA...")
    model = apply_lora(model, config.lora)

    # Tell Trainer to ignore rope_deltas in eval predictions (otherwise it breaks metric collection)
    model.config.keys_to_ignore_at_inference = ["past_key_values", "rope_deltas"]

    # Enable gradient checkpointing (inputs need requires_grad=True for LoRA)
    if config.training.gradient_checkpointing:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_base_model().thinker.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = load_jsonl_dataset(config.data.train_jsonl)
    val_dataset = None
    if os.path.exists(config.data.val_jsonl):
        val_dataset = load_jsonl_dataset(config.data.val_jsonl)

    # Build data collator
    collator = DataCollatorForQwen3ASRFinetune(
        processor=processor,
        sample_rate=config.data.sample_rate,
        max_text_length=config.data.max_text_length,
        language_prefix=config.data.language_prefix,
    )

    # Set up W&B
    if config.training.report_to == "wandb":
        import wandb
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config={
                "model": config.model.__dict__,
                "lora": config.lora.__dict__,
                "training": config.training.__dict__,
            },
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        eval_steps=config.training.eval_steps if val_dataset else None,
        eval_strategy=config.training.eval_strategy if val_dataset else "no",
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end if val_dataset else False,
        metric_for_best_model=config.training.metric_for_best_model if val_dataset else None,
        greater_is_better=config.training.greater_is_better if val_dataset else None,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=config.training.report_to,
        seed=config.training.seed,
        max_grad_norm=config.training.max_grad_norm,
        deepspeed=config.training.deepspeed,
        ddp_find_unused_parameters=config.training.ddp_find_unused_parameters,
        label_names=["labels"],
    )

    # Callbacks
    callbacks = [
        MakeCheckpointInferableCallback(config.model.name),
        WandbAudioCallback(log_every_n_steps=config.training.logging_steps * 50),
    ]

    # Create trainer
    compute_metrics_fn = build_compute_metrics(processor) if val_dataset else None
    trainer = Qwen3ASRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if val_dataset else None,
        callbacks=callbacks,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(config.training.output_dir)

    logger.info(f"Training complete. Model saved to {config.training.output_dir}")


if __name__ == "__main__":
    main()
