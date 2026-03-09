"""GigaSpeech2 Vietnamese subset processor (~2000-3000h).

Uses streaming to avoid HF cache duplication for this ~1TB dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset
import soundfile as sf

from src.data.processors.base import BaseProcessor, StreamingJsonlWriter

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

    def process(self, max_samples: int | None = None) -> dict[str, Path]:
        logger.info("Processing GigaSpeech2 Vietnamese via streaming...")
        if max_samples:
            logger.info(f"Will stop after {max_samples} samples")

        ds = load_dataset(self.hf_repo, "vi", split="train_refined", streaming=True)

        audio_dir = self.raw_dir / "train"
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / "gigaspeech2_train.jsonl"

        with StreamingJsonlWriter(output_path) as writer:
            for idx, sample in enumerate(ds):
                if max_samples and idx >= max_samples:
                    break

                audio_data = sample.get("audio", {})
                text = sample.get("text", sample.get("transcription", "")).strip()

                if not text or not audio_data:
                    continue

                audio_array = audio_data["array"]
                sr = audio_data["sampling_rate"]
                audio_path = audio_dir / f"gs2_{idx:08d}.wav"
                sf.write(str(audio_path), audio_array, sr)

                writer.write({
                    "audio": str(audio_path.resolve()),
                    "text": text,
                })

                if (idx + 1) % 10000 == 0:
                    logger.info(f"Processed {idx + 1} samples...")

        return {"train": output_path}
