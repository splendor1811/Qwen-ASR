"""FLEURS Vietnamese dataset processor (~12h).

Uses huggingface_hub to download raw files (TSV + audio tars) since
datasets v4.x no longer supports loading scripts.
"""

from __future__ import annotations

import csv
import logging
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

HF_REPO = "google/fleurs"
LANG = "vi_vn"

# FLEURS split name mapping: TSV "dev" -> HF "validation"
_SPLIT_MAP = {"train": "train", "dev": "validation", "test": "test"}


class FLEURSProcessor(BaseProcessor):
    name = "fleurs"

    def download(self) -> None:
        logger.info("Downloading FLEURS Vietnamese...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        for tsv_split in _SPLIT_MAP:
            hf_hub_download(
                HF_REPO, f"data/{LANG}/audio/{tsv_split}.tar.gz", repo_type="dataset"
            )
            hf_hub_download(
                HF_REPO, f"data/{LANG}/{tsv_split}.tsv", repo_type="dataset"
            )
        logger.info("FLEURS Vietnamese downloaded successfully")

    def process(self) -> dict[str, Path]:
        logger.info("Processing FLEURS Vietnamese dataset...")

        results = {}
        for tsv_split, output_split in _SPLIT_MAP.items():
            tsv_path = hf_hub_download(
                HF_REPO, f"data/{LANG}/{tsv_split}.tsv", repo_type="dataset"
            )
            tar_path = hf_hub_download(
                HF_REPO, f"data/{LANG}/audio/{tsv_split}.tar.gz", repo_type="dataset"
            )

            # Extract audio tar
            audio_dir = self.raw_dir / output_split
            audio_dir.mkdir(parents=True, exist_ok=True)

            if not any(audio_dir.glob("*.wav")):
                logger.info(f"Extracting {tsv_split} audio...")
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(audio_dir)

            # Parse TSV to get filename -> transcription mapping
            # TSV columns: id, file_name, raw_transcription, transcription, ...
            transcriptions: dict[str, str] = {}
            with open(tsv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    file_name = row.get("file_name", row.get("filename", ""))
                    text = row.get("transcription", row.get("raw_transcription", ""))
                    if file_name and text:
                        transcriptions[file_name] = text.strip()

            # Match audio files to transcriptions
            records = []
            for wav_file in sorted(audio_dir.rglob("*.wav")):
                text = transcriptions.get(wav_file.name)
                if not text:
                    continue
                records.append({
                    "audio": str(wav_file.resolve()),
                    "text": text,
                })

            output_path = self.processed_dir / f"fleurs_{output_split}.jsonl"
            self.write_jsonl(records, output_path)
            results[output_split] = output_path

        return results
