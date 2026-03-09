"""Abstract base class for dataset processors."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


class StreamingJsonlWriter:
    """Write JSONL records incrementally with automatic resume support."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.count = 0
        self._file = None
        self._start_time = None
        self._new_count = 0

    def __enter__(self):
        if self.path.exists() and self.path.stat().st_size > 0:
            # Resume: validate existing lines, truncate incomplete last line
            valid_lines = []
            with open(self.path, "r") as f:
                for line in f:
                    stripped = line.rstrip("\n")
                    if stripped:
                        try:
                            json.loads(stripped)
                            valid_lines.append(stripped)
                        except json.JSONDecodeError:
                            break
            with open(self.path, "w") as f:
                for line in valid_lines:
                    f.write(line + "\n")
            self.count = len(valid_lines)
            self._file = open(self.path, "a")
            logger.info(f"Resuming: {self.count} existing records in {self.path}")
        else:
            self._file = open(self.path, "w")
        self._start_time = time.monotonic()
        return self

    def __exit__(self, *args):
        self._file.close()
        elapsed = time.monotonic() - self._start_time
        rate = self._new_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"Total {self.count} records in {self.path} "
            f"(+{self._new_count} new, {elapsed:.0f}s, {rate:.0f} samples/s)"
        )

    def write(self, record: dict):
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
        self._new_count += 1


class ParallelWavWriter:
    """Write WAV files in parallel using a thread pool.

    Overlaps disk I/O with network streaming for ~2-4x speedup.
    """

    def __init__(self, max_workers: int = 8):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Wait for all pending writes to complete
        for future in self._futures:
            future.result()  # raises if any write failed
        self._executor.shutdown(wait=True)

    def submit(self, audio_path: Path, audio_array, sr: int):
        """Submit a WAV write to the thread pool."""
        future = self._executor.submit(
            sf.write, str(audio_path), audio_array, sr
        )
        self._futures.append(future)
        # Periodically clean up completed futures to avoid memory buildup
        if len(self._futures) > 1000:
            self._futures = [f for f in self._futures if not f.done()]


def find_resume_idx(audio_dir: Path, prefix: str) -> int:
    """Find the next stream index to process from existing WAV filenames.

    Scans for files matching ``{prefix}*.wav`` and extracts the max trailing
    integer.  Returns ``max_idx + 1`` so callers can ``ds.skip(resume_idx)``.
    Returns 0 if no files exist.
    """
    import re

    max_idx = -1
    for wav_file in audio_dir.glob(f"{prefix}*.wav"):
        match = re.search(r"(\d+)\.wav$", wav_file.name)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


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
