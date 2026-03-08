#!/usr/bin/env python3
"""Merge LoRA adapter weights into the base model."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--checkpoint", type=str, required=True, help="LoRA checkpoint path")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    logger.info(f"Loading base model: {args.base_model}")
    qwen3_asr = Qwen3ASRModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen3ASRProcessor.from_pretrained(args.base_model)

    model = qwen3_asr.model

    logger.info(f"Loading LoRA adapter from: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    logger.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)

    logger.info("Merge complete!")


if __name__ == "__main__":
    main()
