#!/usr/bin/env python3
"""Process raw datasets into unified JSONL format."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processors import PROCESSOR_REGISTRY
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ALL_DATASETS = list(PROCESSOR_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Vietnamese ASR data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vivos"],
        choices=ALL_DATASETS + ["all"],
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all train/val splits into unified train.jsonl/val.jsonl",
    )
    return parser.parse_args()


def merge_jsonl_files(data_dir: str, datasets: list[str]) -> None:
    """Merge individual dataset JSONLs into unified train/val files."""
    processed_dir = Path(data_dir) / "processed"

    for split in ["train", "val", "validation", "test"]:
        all_records = []
        for ds_name in datasets:
            jsonl_path = processed_dir / f"{ds_name}_{split}.jsonl"
            if not jsonl_path.exists():
                continue
            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        all_records.append(line.strip())

        if not all_records:
            continue

        output_name = "val" if split == "validation" else split
        output_path = processed_dir / f"{output_name}.jsonl"
        with open(output_path, "w") as f:
            for record in all_records:
                f.write(record + "\n")
        logger.info(f"Merged {len(all_records)} records into {output_path}")


def main():
    args = parse_args()
    setup_logging()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    processed_datasets = []
    for name in datasets:
        if name not in PROCESSOR_REGISTRY:
            logger.warning(f"Unknown dataset: {name}, skipping")
            continue

        logger.info(f"Processing {name}...")
        try:
            processor = PROCESSOR_REGISTRY[name](data_dir=args.data_dir)
            result = processor.process()
            logger.info(f"Successfully processed {name}: {list(result.keys())}")
            processed_datasets.append(name)
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")

    if args.merge and processed_datasets:
        logger.info("Merging datasets...")
        merge_jsonl_files(args.data_dir, processed_datasets)

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
