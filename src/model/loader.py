"""Load Qwen3-ASR model and processor."""

from __future__ import annotations

import logging

import torch
from qwen_asr import Qwen3ASRModel, Qwen3ASRProcessor

from src.config import ModelConfig

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model_and_processor(
    config: ModelConfig,
    device_map: str | None = None,
) -> tuple:
    """Load Qwen3-ASR model and processor.

    Returns:
        (model, processor) tuple. model.forward is patched for HF Trainer compatibility.
    """
    dtype = DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

    logger.info(f"Loading model: {config.name}")

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation and config.attn_implementation != "eager":
        model_kwargs["attn_implementation"] = config.attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    qwen3_asr = Qwen3ASRModel.from_pretrained(config.name, **model_kwargs)
    processor = Qwen3ASRProcessor.from_pretrained(config.name)

    # Extract the inner thinker model for HF Trainer compatibility
    model = qwen3_asr.model
    model.forward = model.thinker.forward

    # Store processor config path for checkpoint saving
    model.config._name_or_path = config.name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total:,} params, {trainable:,} trainable")

    return model, processor
