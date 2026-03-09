"""Tests for the data collator (requires model/processor - integration test)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration test - set RUN_INTEGRATION_TESTS=1 to run",
)
def test_collator_output_shapes():
    """Test that collator produces correctly shaped tensors."""
    import torch
    import numpy as np
    import tempfile
    import soundfile as sf

    from src.data.collator import DataCollatorForQwen3ASRFinetune
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    processor = Qwen3ASRProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B")
    collator = DataCollatorForQwen3ASRFinetune(processor=processor)

    # Create a dummy audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds
        sf.write(f.name, audio, 16000)
        audio_path = f.name

    try:
        samples = [
            {"audio": audio_path, "text": "xin chào thế giới"},
            {"audio": audio_path, "text": "tôi là ai"},
        ]

        batch = collator(samples)

        assert "input_ids" in batch
        assert "labels" in batch
        assert "attention_mask" in batch
        assert "input_features" in batch
        assert "feature_attention_mask" in batch

        assert batch["input_ids"].shape[0] == 2
        assert batch["labels"].shape == batch["input_ids"].shape
        assert batch["attention_mask"].shape == batch["input_ids"].shape

        # Check that labels have -100 masking for prefix
        assert (batch["labels"][0] == -100).any(), "Labels should have -100 prefix masking"
        assert (batch["labels"][0] != -100).any(), "Labels should have non-masked target tokens"

    finally:
        os.unlink(audio_path)


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration test - set RUN_INTEGRATION_TESTS=1 to run",
)
def test_collator_normalizes_text():
    """Test that collator normalizes uppercase text to lowercase in labels."""
    import torch
    import numpy as np
    import tempfile
    import soundfile as sf

    from src.data.collator import DataCollatorForQwen3ASRFinetune
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    processor = Qwen3ASRProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B")
    tokenizer = processor.tokenizer

    # Collator with normalization enabled (default)
    collator_norm = DataCollatorForQwen3ASRFinetune(processor=processor, normalize_text=True)
    # Collator with normalization disabled
    collator_raw = DataCollatorForQwen3ASRFinetune(processor=processor, normalize_text=False)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = np.random.randn(16000 * 2).astype(np.float32)
        sf.write(f.name, audio, 16000)
        audio_path = f.name

    try:
        samples = [{"audio": audio_path, "text": "KHÁCH SẠN NHÀ HÀNG"}]

        batch_norm = collator_norm(samples)
        batch_raw = collator_raw(samples)

        # Decode the non-masked labels (target tokens)
        labels_norm = batch_norm["labels"][0]
        labels_raw = batch_raw["labels"][0]

        target_norm = tokenizer.decode(labels_norm[labels_norm != -100], skip_special_tokens=True)
        target_raw = tokenizer.decode(labels_raw[labels_raw != -100], skip_special_tokens=True)

        # Normalized should be lowercase
        assert "khách sạn nhà hàng" in target_norm
        # Raw should preserve uppercase
        assert "KHÁCH SẠN NHÀ HÀNG" in target_raw

    finally:
        os.unlink(audio_path)


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration test - set RUN_INTEGRATION_TESTS=1 to run",
)
def test_collator_label_masking():
    """Test that prefix tokens are properly masked with -100."""
    import torch
    import numpy as np
    import tempfile
    import soundfile as sf

    from src.data.collator import DataCollatorForQwen3ASRFinetune, ASR_CHAT_TEMPLATE
    from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    processor = Qwen3ASRProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B")
    collator = DataCollatorForQwen3ASRFinetune(processor=processor)
    tokenizer = processor.tokenizer

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio = np.random.randn(16000 * 2).astype(np.float32)
        sf.write(f.name, audio, 16000)
        audio_path = f.name

    try:
        samples = [{"audio": audio_path, "text": "xin chào"}]
        batch = collator(samples)

        labels = batch["labels"][0]
        prefix_len = len(tokenizer.encode(ASR_CHAT_TEMPLATE, add_special_tokens=False))

        # First prefix_len tokens (chat template) should be -100
        assert (labels[:prefix_len] == -100).all()
        # The full target includes "language Vietnamese<asr_text>xin chào" + eos
        # so there should be non-masked tokens after the prefix
        assert (labels[prefix_len:] != -100).any()

    finally:
        os.unlink(audio_path)
