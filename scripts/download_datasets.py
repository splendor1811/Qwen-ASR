#!/usr/bin/env python3
"""Download Vietnamese ASR datasets from HuggingFace."""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.processors import PROCESSOR_REGISTRY
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

ALL_DATASETS = list(PROCESSOR_REGISTRY.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Download Vietnamese ASR datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["vivos"],
        choices=ALL_DATASETS + ["all"],
        help=f"Datasets to download. Available: {ALL_DATASETS}",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    for name in datasets:
        if name not in PROCESSOR_REGISTRY:
            logger.warning(f"Unknown dataset: {name}, skipping")
            continue

        logger.info(f"Downloading {name}...")
        try:
            processor = PROCESSOR_REGISTRY[name](data_dir=args.data_dir)
            processor.download()
            logger.info(f"Successfully downloaded {name}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")


if __name__ == "__main__":
    main()
