"""Abstract base class for dataset processors."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class StreamingJsonlWriter:
    """Write JSONL records incrementally (for streaming large datasets)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "w")
        return self

    def __exit__(self, *args):
        self._file.close()
        logger.info(f"Wrote {self.count} records to {self.path}")

    def write(self, record: dict):
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1


class BaseProcessor(ABC):
    """Base class for processing raw datasets into unified JSONL format.

    Output format per line: {"audio": "/absolute/path.wav", "text": "transcription"}
    """

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / self.name
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name used for directory and file naming."""
        ...

    @abstractmethod
    def download(self) -> None:
        """Download the raw dataset."""
        ...

    @abstractmethod
    def process(self, max_samples: int | None = None) -> dict[str, Path]:
        """Process raw data into JSONL files.

        Args:
            max_samples: Maximum number of samples to process (None = all).

        Returns:
            Dict mapping split names to JSONL file paths,
            e.g. {"train": Path("data/processed/vivos_train.jsonl")}
        """
        ...

    def write_jsonl(self, records: list[dict], output_path: Path) -> int:
        """Write records to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        logger.info(f"Wrote {count} records to {output_path}")
        return count
