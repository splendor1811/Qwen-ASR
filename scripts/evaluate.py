#!/usr/bin/env python3
"""Evaluate a checkpoint on Vietnamese ASR benchmarks."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.data.utils import load_audio
from src.evaluation.benchmarks import load_benchmark
from src.evaluation.metrics import compute_wer, compute_cer
from src.evaluation.normalize_vi import normalize_vietnamese
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-ASR on benchmarks")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Override benchmarks")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, config, device: str):
    """Load model from checkpoint (LoRA or merged)."""
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    logger.info(f"Loading base model: {config.model.name}")
    qwen3_asr = Qwen3ASRModel.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen3ASRProcessor.from_pretrained(config.model.name)

    model = qwen3_asr.model

    # Check if checkpoint has LoRA adapter
    adapter_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    adapter_path_bin = os.path.join(checkpoint_path, "adapter_model.bin")

    if os.path.exists(adapter_path) or os.path.exists(adapter_path_bin):
        logger.info(f"Loading LoRA adapter from {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

    # Note: do NOT set model.forward = model.thinker.forward here.
    # That hack is only for HF Trainer compatibility during training.
    # For inference, model.generate() properly routes to thinker.generate().
    model = model.to(device).eval()

    return model, processor


def transcribe_sample(model, processor, audio, sr: int, device: str) -> str:
    """Transcribe a single audio sample."""
    if isinstance(audio, str):
        audio = load_audio(audio, target_sr=16000)
        sr = 16000

    # Build conversation for processor
    conversation = [
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]

    # Apply chat template to get the text string the processor expects
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device=device, dtype=torch.bfloat16) if hasattr(v, "is_floating_point") and v.is_floating_point()
        else v.to(device=device) if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Decode only the generated tokens (after input)
    input_len = inputs["input_ids"].shape[1]
    sequences = generated_ids.sequences if hasattr(generated_ids, "sequences") else generated_ids
    output_ids = sequences[:, input_len:]
    text = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Strip model language prefix (e.g. "language Vietnamese<asr_text>")
    text = re.sub(r"^language\s+\w+<asr_text>", "", text)
    return text.strip()


def main():
    args = parse_args()
    setup_logging()

    config = load_config(args.config)
    benchmarks = args.benchmarks or config.eval.benchmarks

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model, processor = load_checkpoint(args.checkpoint, config, device)

    results = {}
    for bench_name in benchmarks:
        logger.info(f"Evaluating on {bench_name}...")

        try:
            samples = load_benchmark(bench_name)
        except Exception as e:
            logger.error(f"Failed to load benchmark {bench_name}: {e}")
            continue

        references = []
        hypotheses = []
        sample_details = []

        for i, sample in enumerate(tqdm(samples, desc=bench_name)):
            audio = sample.get("audio")
            text = sample["text"]
            sr = sample.get("sr", 16000)

            try:
                prediction = transcribe_sample(model, processor, audio, sr, device)
                references.append(text)
                hypotheses.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to transcribe sample: {e}")
                prediction = ""
                references.append(text)
                hypotheses.append(prediction)

            sample_details.append({
                "raw": prediction,
                "ref": normalize_vietnamese(text),
                "hyp": normalize_vietnamese(prediction),
            })
            if text != prediction:
                logger.info(f"[{bench_name}#{i}] REF: {text}")
                logger.info(f"[{bench_name}#{i}] HYP: {prediction}")

        wer_score = compute_wer(references, hypotheses)
        cer_score = compute_cer(references, hypotheses)

        results[bench_name] = {
            "wer": wer_score,
            "cer": cer_score,
            "n_samples": len(references),
            "samples": sample_details,
        }
        logger.info(f"{bench_name}: WER={wer_score:.4f} ({wer_score*100:.2f}%), "
                     f"CER={cer_score:.4f} ({cer_score*100:.2f}%), n={len(references)}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for name, metrics in results.items():
        print(f"  {name:20s}  WER: {metrics['wer']*100:6.2f}%  "
              f"CER: {metrics['cer']*100:6.2f}%  (n={metrics['n_samples']})")
    print("=" * 60)

    # Save results
    if args.output_file:
        import json
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
