"""VIET_BUD500 dataset processor (~500h Vietnamese)."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
import soundfile as sf

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class VietBud500Processor(BaseProcessor):
    name = "vietbud500"
    hf_repo = "linhtran92/viet_bud500"

    def download(self) -> None:
        logger.info(f"Downloading VIET_BUD500 from {self.hf_repo}...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(self.hf_repo, split="train[:1]")
        logger.info("VIET_BUD500 download initiated")

    def process(self) -> dict[str, Path]:
        logger.info("Processing VIET_BUD500...")
        ds = load_dataset(self.hf_repo)

        results = {}
        for split in ds.keys():
            records = []
            audio_dir = self.raw_dir / split
            audio_dir.mkdir(parents=True, exist_ok=True)

            for idx, sample in enumerate(ds[split]):
                audio_data = sample.get("audio", {})
                text = sample.get("transcription", sample.get("sentence", "")).strip()

                if not text or not audio_data:
                    continue

                audio_array = audio_data["array"]
                sr = audio_data["sampling_rate"]
                audio_path = audio_dir / f"vb500_{split}_{idx:06d}.wav"
                sf.write(str(audio_path), audio_array, sr)

                records.append({
                    "audio": str(audio_path.resolve()),
                    "text": text,
                })

                if (idx + 1) % 10000 == 0:
                    logger.info(f"Processed {idx + 1} samples from {split}...")

            output_path = self.processed_dir / f"vietbud500_{split}.jsonl"
            self.write_jsonl(records, output_path)
            results[split] = output_path

        return results
