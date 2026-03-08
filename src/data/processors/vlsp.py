"""VLSP dataset processor (unofficial VLSP2020 from HuggingFace)."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
import soundfile as sf

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class VLSPProcessor(BaseProcessor):
    name = "vlsp"
    hf_repo = "doof-ferb/vlsp2020_vinai_100h"

    def download(self) -> None:
        logger.info(f"Downloading VLSP2020 from {self.hf_repo}...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(self.hf_repo)
        logger.info("VLSP2020 downloaded successfully")

    def process(self) -> dict[str, Path]:
        logger.info("Processing VLSP2020 dataset...")
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
                audio_path = audio_dir / f"vlsp_{split}_{idx:06d}.wav"
                sf.write(str(audio_path), audio_array, sr)

                records.append({
                    "audio": str(audio_path.resolve()),
                    "text": text,
                })

            output_name = "train" if "train" in split else split
            output_path = self.processed_dir / f"vlsp_{output_name}.jsonl"
            self.write_jsonl(records, output_path)
            results[output_name] = output_path

        return results
