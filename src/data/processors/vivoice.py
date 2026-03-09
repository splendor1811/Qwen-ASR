"""viVoice Vietnamese dataset processor (~1017h, 887k samples).

Uses streaming to avoid HF cache duplication for this 169GB dataset.
Audio is 24kHz — resampled to 16kHz at process time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
from datasets import load_dataset

from src.data.processors.base import BaseProcessor, ParallelWavWriter, StreamingJsonlWriter, find_resume_idx

logger = logging.getLogger(__name__)


class ViVoiceProcessor(BaseProcessor):
    name = "vivoice"
    hf_repo = "capleaf/viVoice"

    def download(self) -> None:
        logger.info(f"Checking viVoice access at {self.hf_repo} (gated, requires HF agreement)...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        # Lightweight access check — stream one sample
        ds = load_dataset(self.hf_repo, split="train", streaming=True)
        next(iter(ds))
        logger.info("viVoice access confirmed")

    def process(self, max_samples: int | None = None) -> dict[str, Path]:
        logger.info("Processing viVoice via streaming...")
        if max_samples:
            logger.info(f"Will stop after {max_samples} samples")

        ds = load_dataset(self.hf_repo, split="train", streaming=True)

        audio_dir = self.raw_dir / "train"
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / "vivoice_train.jsonl"

        resume_idx = find_resume_idx(audio_dir, "vivoice_")
        if resume_idx > 0:
            logger.info(f"Resuming from stream index {resume_idx}")
            ds = ds.skip(resume_idx)

        with StreamingJsonlWriter(output_path) as writer, ParallelWavWriter() as wav_writer:
            for idx, sample in enumerate(ds, start=resume_idx):
                if max_samples and idx >= max_samples:
                    break

                text = sample["text"].strip()
                if not text:
                    continue

                audio_data = sample["audio"]
                audio_array = librosa.resample(
                    audio_data["array"],
                    orig_sr=audio_data["sampling_rate"],
                    target_sr=16000,
                )
                audio_path = audio_dir / f"vivoice_{idx:08d}.wav"
                wav_writer.submit(audio_path, audio_array, 16000)

                writer.write({
                    "audio": str(audio_path.resolve()),
                    "text": text,
                })

                if (idx + 1) % 10000 == 0:
                    logger.info(f"Processed {idx + 1} samples...")

        return {"train": output_path}
