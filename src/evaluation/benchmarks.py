"""Load benchmark test sets for evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_benchmark(name: str, data_dir: str = "data") -> list[dict]:
    """Load a benchmark test set.

    Returns list of {"audio": path_or_array, "text": reference, "sr": sample_rate}
    """
    loaders = {
        "vivos": _load_vivos_test,
        "fleurs_vi": _load_fleurs_test,
        "vlsp2020": _load_vlsp2020_test,
    }

    if name not in loaders:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(loaders.keys())}")

    return loaders[name](data_dir)


def _load_vivos_test(data_dir: str) -> list[dict]:
    """Load VIVOS test set."""
    # Try local JSONL first
    jsonl_path = Path(data_dir) / "processed" / "vivos_test.jsonl"
    if jsonl_path.exists():
        return _load_from_jsonl(jsonl_path)

    # Fall back to HuggingFace
    logger.info("Loading VIVOS test from HuggingFace...")
    ds = load_dataset("AILAB-VNUHCM/vivos", split="test", trust_remote_code=True)
    return [
        {
            "audio": sample["audio"]["array"],
            "text": sample["sentence"].strip(),
            "sr": sample["audio"]["sampling_rate"],
        }
        for sample in ds
        if sample["sentence"].strip()
    ]


def _load_fleurs_test(data_dir: str) -> list[dict]:
    """Load FLEURS Vietnamese test set."""
    jsonl_path = Path(data_dir) / "processed" / "fleurs_test.jsonl"
    if jsonl_path.exists():
        return _load_from_jsonl(jsonl_path)

    logger.info("Loading FLEURS Vietnamese test from HuggingFace...")
    ds = load_dataset("google/fleurs", "vi_vn", split="test", trust_remote_code=True)
    return [
        {
            "audio": sample["audio"]["array"],
            "text": sample["transcription"].strip(),
            "sr": sample["audio"]["sampling_rate"],
        }
        for sample in ds
        if sample["transcription"].strip()
    ]


def _load_vlsp2020_test(data_dir: str) -> list[dict]:
    """Load VLSP2020 test set (unofficial)."""
    jsonl_path = Path(data_dir) / "processed" / "vlsp_test.jsonl"
    if jsonl_path.exists():
        return _load_from_jsonl(jsonl_path)

    logger.info("Loading VLSP2020 test from HuggingFace...")
    ds = load_dataset("doof-ferb/vlsp2020_vinai_100h", split="test", trust_remote_code=True)
    return [
        {
            "audio": sample["audio"]["array"],
            "text": sample.get("transcription", sample.get("sentence", "")).strip(),
            "sr": sample["audio"]["sampling_rate"],
        }
        for sample in ds
        if sample.get("transcription", sample.get("sentence", "")).strip()
    ]


def _load_from_jsonl(jsonl_path: Path) -> list[dict]:
    """Load benchmark from local JSONL (audio as file paths)."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
