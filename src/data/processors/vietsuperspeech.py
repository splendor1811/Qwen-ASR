"""VietSuperSpeech dataset processor (~103h Vietnamese).

The HF repo contains JSON manifests (train.json, dev.json) with metadata
and an audio/ directory with WAV files (~62GB). The 'audio' column contains
string paths (not Audio objects), so we use snapshot_download.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class VietSuperSpeechProcessor(BaseProcessor):
    name = "vietsuperspeech"
    hf_repo = "thanhnew2001/VietSuperSpeech"

    def download(self) -> None:
        logger.info(f"Downloading VietSuperSpeech from {self.hf_repo} (this is ~62GB)...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(self.hf_repo, repo_type="dataset")
        logger.info("VietSuperSpeech downloaded successfully")

    def process(self) -> dict[str, Path]:
        logger.info("Processing VietSuperSpeech...")

        # Get the snapshot directory
        snapshot_dir = Path(
            snapshot_download(self.hf_repo, repo_type="dataset")
        )

        # Map JSON manifest names to output split names
        split_files = {"train": "train.json", "validation": "dev.json"}

        results = {}
        for split, manifest_name in split_files.items():
            manifest_path = snapshot_dir / manifest_name
            if not manifest_path.exists():
                logger.warning(f"Manifest not found: {manifest_path}")
                continue

            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            records = []
            for idx, entry in enumerate(manifest):
                # audio column is a relative path like "audio/asr_segments_.../filename.wav"
                audio_rel = entry.get("audio", "")
                text = entry.get("text", "").strip()

                if not text or not audio_rel:
                    continue

                audio_path = snapshot_dir / audio_rel
                if not audio_path.exists():
                    continue

                records.append({
                    "audio": str(audio_path.resolve()),
                    "text": text,
                })

                if (idx + 1) % 10000 == 0:
                    logger.info(f"Processed {idx + 1} samples from {split}...")

            output_path = self.processed_dir / f"vietsuperspeech_{split}.jsonl"
            self.write_jsonl(records, output_path)
            results[split] = output_path

        return results
