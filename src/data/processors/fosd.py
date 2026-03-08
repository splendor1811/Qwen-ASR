"""FOSD (Free Online Speech Dataset) Vietnamese processor."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
import soundfile as sf

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class FOSDProcessor(BaseProcessor):
    name = "fosd"
    hf_repo = "doof-ferb/fpt_fosd"

    def download(self) -> None:
        logger.info(f"Downloading FOSD from {self.hf_repo}...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(self.hf_repo, split="train[:1]")
        logger.info("FOSD download initiated")

    def process(self) -> dict[str, Path]:
        logger.info("Processing FOSD...")
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
                audio_path = audio_dir / f"fosd_{split}_{idx:06d}.wav"
                sf.write(str(audio_path), audio_array, sr)

                records.append({
                    "audio": str(audio_path.resolve()),
                    "text": text,
                })

            output_path = self.processed_dir / f"fosd_{split}.jsonl"
            self.write_jsonl(records, output_path)
            results[split] = output_path

        return results
