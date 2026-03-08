"""Load JSONL datasets as HuggingFace Datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_jsonl_dataset(jsonl_path: str | Path) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset.

    Expected format per line: {"audio": "/path/to/file.wav", "text": "transcription"}
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    records = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "audio" not in record or "text" not in record:
                    logger.warning(f"Line {line_num}: missing 'audio' or 'text' field, skipping")
                    continue
                records.append(record)
            except json.JSONDecodeError:
                logger.warning(f"Line {line_num}: invalid JSON, skipping")

    logger.info(f"Loaded {len(records)} samples from {jsonl_path}")
    return Dataset.from_list(records)
