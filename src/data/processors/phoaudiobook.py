"""PhoAudioBook Vietnamese dataset processor (~1500h).

Uses streaming to avoid HF cache duplication.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset
import soundfile as sf

from src.data.processors.base import BaseProcessor, StreamingJsonlWriter

logger = logging.getLogger(__name__)


class PhoAudioBookProcessor(BaseProcessor):
    name = "phoaudiobook"
    hf_repo = "thivux/phoaudiobook"

    def download(self) -> None:
        logger.info(f"Downloading PhoAudioBook from {self.hf_repo} (gated, requires HF agreement)...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(self.hf_repo, split="train[:1]")
        logger.info("PhoAudioBook download initiated")

    def process(self, max_samples: int | None = None) -> dict[str, Path]:
        logger.info("Processing PhoAudioBook via streaming...")
        if max_samples:
            logger.info(f"Will stop after {max_samples} samples per split")

        splits = get_dataset_split_names(self.hf_repo)
        results = {}

        for split in splits:
            ds = load_dataset(self.hf_repo, split=split, streaming=True)
            audio_dir = self.raw_dir / split
            audio_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.processed_dir / f"phoaudiobook_{split}.jsonl"

            with StreamingJsonlWriter(output_path) as writer:
                for idx, sample in enumerate(ds):
                    if max_samples and idx >= max_samples:
                        break

                    audio_data = sample.get("audio", {})
                    text = sample.get("transcription", sample.get("text", "")).strip()

                    if not text or not audio_data:
                        continue

                    audio_array = audio_data["array"]
                    sr = audio_data["sampling_rate"]
                    audio_path = audio_dir / f"pab_{split}_{idx:06d}.wav"
                    sf.write(str(audio_path), audio_array, sr)

                    writer.write({
                        "audio": str(audio_path.resolve()),
                        "text": text,
                    })

                    if (idx + 1) % 10000 == 0:
                        logger.info(f"Processed {idx + 1} samples from {split}...")

            results[split] = output_path

        return results
