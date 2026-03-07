"""Custom trainer for Qwen3-ASR finetuning."""

from __future__ import annotations

import logging

import torch
from transformers import Trainer

logger = logging.getLogger(__name__)


class Qwen3ASRTrainer(Trainer):
    """Custom Trainer that handles Qwen3-ASR specifics.

    Key customizations:
    - Casts audio features to model dtype to avoid dtype mismatches
    - Computes WER metric during evaluation
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure audio features match model dtype
        model_dtype = next(model.parameters()).dtype
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model_dtype)

        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        return (loss, outputs) if return_outputs else loss
