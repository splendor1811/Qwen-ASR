"""VIVOS dataset processor (~15h Vietnamese read speech).

Uses huggingface_hub to download raw files (tar.gz + prompts) since
datasets v4.x no longer supports loading scripts.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.data.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

HF_REPO = "AILAB-VNUHCM/vivos"


class VIVOSProcessor(BaseProcessor):
    name = "vivos"

    def download(self) -> None:
        logger.info("Downloading VIVOS from HuggingFace...")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Download tar.gz and prompt files
        hf_hub_download(HF_REPO, "data/vivos.tar.gz", repo_type="dataset")
        hf_hub_download(HF_REPO, "data/prompts-train.txt.gz", repo_type="dataset")
        hf_hub_download(HF_REPO, "data/prompts-test.txt.gz", repo_type="dataset")
        logger.info("VIVOS downloaded successfully")

    def process(self) -> dict[str, Path]:
        logger.info("Processing VIVOS dataset...")

        # Resolve cached file paths
        tar_path = hf_hub_download(HF_REPO, "data/vivos.tar.gz", repo_type="dataset")

        # Extract tar.gz to raw_dir
        extract_dir = self.raw_dir / "extracted"
        if not extract_dir.exists():
            logger.info(f"Extracting {tar_path} ...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_dir)

        results = {}
        for split in ["train", "test"]:
            # Parse prompts file: each line is "SPEAKER_UTTID transcription text"
            prompts_path = hf_hub_download(
                HF_REPO, f"data/prompts-{split}.txt.gz", repo_type="dataset"
            )
            transcriptions: dict[str, str] = {}
            with gzip.open(prompts_path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, text = line.split(" ", 1)
                    transcriptions[utt_id] = text.strip()

            # Find WAV files: vivos/{split}/waves/SPEAKER/SPEAKER_UTTID.wav
            waves_dir = extract_dir / "vivos" / split / "waves"
            if not waves_dir.exists():
                logger.warning(f"Waves dir not found: {waves_dir}")
                continue

            # Copy WAV files to raw_dir/{split}/ and build records
            audio_out_dir = self.raw_dir / split
            audio_out_dir.mkdir(parents=True, exist_ok=True)

            records = []
            for wav_file in sorted(waves_dir.rglob("*.wav")):
                utt_id = wav_file.stem  # e.g. VIVOSSPK01_001
                text = transcriptions.get(utt_id)
                if not text:
                    continue

                dest = audio_out_dir / wav_file.name
                if not dest.exists():
                    shutil.copy2(wav_file, dest)

                records.append({
                    "audio": str(dest.resolve()),
                    "text": text,
                })

            output_path = self.processed_dir / f"vivos_{split}.jsonl"
            self.write_jsonl(records, output_path)
            results[split] = output_path

        return results
