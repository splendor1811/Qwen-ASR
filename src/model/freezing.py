"""Freeze model components (audio encoder, embeddings)."""

from __future__ import annotations

import logging

from src.config import FreezeConfig

logger = logging.getLogger(__name__)


def apply_freezing(model, config: FreezeConfig) -> None:
    """Freeze specified model components.

    For Qwen3-ASR, the audio encoder is at model.thinker.audio.*
    and embeddings are at model.thinker.embed_tokens.
    """
    frozen_count = 0

    if config.freeze_audio_encoder:
        for name, param in model.named_parameters():
            if "audio" in name:
                param.requires_grad = False
                frozen_count += 1

    if config.freeze_embeddings:
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                param.requires_grad = False
                frozen_count += 1

    if config.freeze_lm_head:
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = False
                frozen_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Freezing applied: {frozen_count} param groups frozen, "
        f"{trainable:,} trainable / {total:,} total"
    )
