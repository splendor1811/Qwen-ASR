"""LoRA configuration and application."""

from __future__ import annotations

import logging

from peft import LoraConfig, get_peft_model, TaskType

from src.config import LoRAConfig

logger = logging.getLogger(__name__)

TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
}


def apply_lora(model, config: LoRAConfig):
    """Apply LoRA adapters to the model.

    LoRA is applied only to the LLM decoder layers (target_modules match
    the attention and MLP projections in the language model).
    """
    if not config.enabled:
        logger.info("LoRA disabled, returning model as-is")
        return model

    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TASK_TYPE_MAP.get(config.task_type, TaskType.CAUSAL_LM),
    )

    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0
    logger.info(f"LoRA applied: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")
    model.print_trainable_parameters()

    return model
