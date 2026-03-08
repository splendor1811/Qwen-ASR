#!/usr/bin/env python3
"""Interactive inference demo for Qwen3-ASR Vietnamese."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

from src.data.utils import load_audio
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive ASR inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="LoRA or merged checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--audio", type=str, default=None, help="Audio file path (non-interactive)")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(base_model: str, checkpoint: str | None, device: str):
    """Load model, optionally with LoRA adapter."""
    qwen3_asr = Qwen3ASRModel.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen3ASRProcessor.from_pretrained(base_model)
    model = qwen3_asr.model

    if checkpoint:
        adapter_path = os.path.join(checkpoint, "adapter_model.safetensors")
        adapter_path_bin = os.path.join(checkpoint, "adapter_model.bin")
        if os.path.exists(adapter_path) or os.path.exists(adapter_path_bin):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint)
            model = model.merge_and_unload()
            logger.info(f"Loaded LoRA adapter from {checkpoint}")

    # Note: do NOT set model.forward = model.thinker.forward here.
    # That hack is only for HF Trainer compatibility during training.
    # For inference, model.generate() properly routes to thinker.generate().
    model = model.to(device).eval()
    return model, processor


def transcribe(model, processor, audio_path: str, device: str) -> str:
    """Transcribe a single audio file."""
    audio = load_audio(audio_path, target_sr=16000)

    conversation = [
        {"role": "user", "content": [{"type": "audio", "audio": audio}]},
    ]

    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=text_prompt,
        audio=[audio],
        sampling_rate=16000,
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

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model, processor = load_model(args.base_model, args.checkpoint, device)

    if args.audio:
        # Single file mode
        result = transcribe(model, processor, args.audio, device)
        print(f"\nTranscription: {result}")
        return

    # Interactive mode
    print("\nQwen3-ASR Vietnamese - Interactive Mode")
    print("Enter audio file path (or 'quit' to exit):\n")

    while True:
        try:
            audio_path = input("Audio file: ").strip()
            if audio_path.lower() in ("quit", "exit", "q"):
                break
            if not os.path.exists(audio_path):
                print(f"File not found: {audio_path}")
                continue

            result = transcribe(model, processor, audio_path, device)
            print(f"Transcription: {result}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
