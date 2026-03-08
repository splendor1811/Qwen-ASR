"""GigaSpeech2 Vietnamese subset processor (~2000-3000h)."""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
import soundfile as sf

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class GigaSpeech2Processor(BaseProcessor):
    name = "gigaspeech2"
    hf_repo = "speechcolab/gigaspeech2"

    def download(self) -> None:
        logger.info("Downloading GigaSpeech2 Vietnamese subset (gated, requires HF_TOKEN)...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        # This is a gated dataset - user needs to accept terms on HF
        load_dataset(self.hf_repo, "vi", split="train_refined[:1]")
        logger.info("GigaSpeech2 Vietnamese access confirmed")

    def process(self) -> dict[str, Path]:
        logger.info("Processing GigaSpeech2 Vietnamese (this may take a long time)...")

        ds = load_dataset(self.hf_repo, "vi", split="train_refined")

        records = []
        audio_dir = self.raw_dir / "train"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for idx, sample in enumerate(ds):
            audio_data = sample.get("audio", {})
            text = sample.get("text", sample.get("transcription", "")).strip()

            if not text or not audio_data:
                continue

            audio_array = audio_data["array"]
            sr = audio_data["sampling_rate"]
            audio_path = audio_dir / f"gs2_{idx:08d}.wav"
            sf.write(str(audio_path), audio_array, sr)

            records.append({
                "audio": str(audio_path.resolve()),
                "text": text,
            })

            if (idx + 1) % 10000 == 0:
                logger.info(f"Processed {idx + 1} samples...")

        output_path = self.processed_dir / "gigaspeech2_train.jsonl"
        self.write_jsonl(records, output_path)
        return {"train": output_path}
