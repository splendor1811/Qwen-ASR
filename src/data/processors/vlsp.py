"""VLSP dataset processor (unofficial VLSP2020 from HuggingFace).

Uses streaming to avoid HF cache duplication.
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset

from src.data.processors.base import BaseProcessor, ParallelWavWriter, StreamingJsonlWriter, find_resume_idx

logger = logging.getLogger(__name__)


class VLSPProcessor(BaseProcessor):
    name = "vlsp"
    hf_repo = "doof-ferb/vlsp2020_vinai_100h"

    def download(self) -> None:
        logger.info(f"Downloading VLSP2020 from {self.hf_repo}...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        load_dataset(self.hf_repo, split="train[:1]")
        logger.info("VLSP2020 download initiated")

    def process(self, max_samples: int | None = None) -> dict[str, Path]:
        logger.info("Processing VLSP2020 dataset via streaming...")
        if max_samples:
            logger.info(f"Will stop after {max_samples} samples per split")

        splits = get_dataset_split_names(self.hf_repo)
        results = {}

        for split in splits:
            ds = load_dataset(self.hf_repo, split=split, streaming=True)
            audio_dir = self.raw_dir / split
            audio_dir.mkdir(parents=True, exist_ok=True)

            output_name = "train" if "train" in split else split
            output_path = self.processed_dir / f"vlsp_{output_name}.jsonl"

            resume_idx = find_resume_idx(audio_dir, f"vlsp_{split}_")
            if resume_idx > 0:
                logger.info(f"Resuming {split} from stream index {resume_idx}")
                ds_split = ds.skip(resume_idx)
            else:
                ds_split = ds

            with StreamingJsonlWriter(output_path) as writer, ParallelWavWriter() as wav_writer:
                for idx, sample in enumerate(ds_split, start=resume_idx):
                    if max_samples and idx >= max_samples:
                        break

                    audio_data = sample.get("audio", {})
                    text = sample.get("transcription", sample.get("sentence", "")).strip()

                    if not text or not audio_data:
                        continue

                    audio_array = audio_data["array"]
                    sr = audio_data["sampling_rate"]
                    audio_path = audio_dir / f"vlsp_{split}_{idx:06d}.wav"
                    wav_writer.submit(audio_path, audio_array, sr)

                    writer.write({
                        "audio": str(audio_path.resolve()),
                        "text": text,
                    })

            results[output_name] = output_path

        return results
