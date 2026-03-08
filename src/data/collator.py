"""Data collator for Qwen3-ASR finetuning.

This is the most critical component of the data pipeline. It handles:
1. Loading audio at 16kHz
2. Building the chat template prefix
3. Tokenizing prefix + target + EOS
4. Processing through Qwen3ASR processor for audio features
5. Masking prefix tokens with -100 in labels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from src.data.utils import load_audio

logger = logging.getLogger(__name__)

# Qwen3-ASR chat template for ASR task
ASR_CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class DataCollatorForQwen3ASRFinetune:
    """Collator that builds training batches for Qwen3-ASR finetuning.

    For each sample:
    - Loads audio and extracts features via the processor
    - Builds input_ids: [prefix_tokens] + [target_tokens] + [eos]
    - Builds labels: [-100]*len(prefix) + [target_tokens] + [eos]
    """

    processor: Any
    sample_rate: int = 16000
    max_text_length: int = 512
    language_prefix: str = "Vietnamese"

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        batch_audio_features = []
        batch_audio_feature_lengths = []

        tokenizer = self.processor.tokenizer

        for sample in features:
            audio_path = sample["audio"]
            text = sample["text"]

            # Load audio
            audio = load_audio(audio_path, self.sample_rate)

            # Process audio through the processor to get audio features
            audio_inputs = self.processor.audio_processor(
                [audio],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            audio_feats = audio_inputs["input_features"].squeeze(0)
            audio_feat_len = audio_inputs["feature_attention_mask"].sum(-1).squeeze(0)

            # Tokenize the prefix (chat template)
            prefix_tokens = tokenizer.encode(ASR_CHAT_TEMPLATE, add_special_tokens=False)

            # Add Qwen3-ASR language prefix: "language Vietnamese<asr_text>..."
            target_text = f"language {self.language_prefix}<asr_text>{text}"
            target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
            eos_token = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

            # Build full sequence
            input_ids = prefix_tokens + target_tokens + eos_token

            # Truncate if needed
            if len(input_ids) > self.max_text_length:
                input_ids = input_ids[: self.max_text_length]
                # Ensure EOS at end
                if eos_token:
                    input_ids[-1] = eos_token[0]

            # Build labels: mask prefix with -100
            labels = [-100] * len(prefix_tokens) + target_tokens + eos_token
            if len(labels) > self.max_text_length:
                labels = labels[: self.max_text_length]
                if eos_token:
                    labels[-1] = eos_token[0]

            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            batch_audio_features.append(audio_feats)
            batch_audio_feature_lengths.append(audio_feat_len)

        # Pad sequences
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )

        # Pad audio features
        max_audio_len = max(f.shape[0] for f in batch_audio_features)
        audio_dim = batch_audio_features[0].shape[-1]
        padded_audio = torch.zeros(len(batch_audio_features), max_audio_len, audio_dim)
        audio_mask = torch.zeros(len(batch_audio_features), max_audio_len, dtype=torch.long)
        for i, feats in enumerate(batch_audio_features):
            padded_audio[i, : feats.shape[0]] = feats
            audio_mask[i, : feats.shape[0]] = 1

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
            "input_features": padded_audio,
            "feature_attention_mask": audio_mask,
        }
