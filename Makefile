.PHONY: install train eval setup-cloud download prepare test lint

install:
	uv sync --extra dev

install-train:
	uv sync --extra train

download:
	uv run python scripts/download_datasets.py --datasets $(DATASETS)

prepare:
	uv run python scripts/prepare_data.py --datasets $(DATASETS)

train:
	uv run python scripts/train.py --config $(CONFIG)

train-ds:
	uv run deepspeed scripts/train.py --config $(CONFIG) --deepspeed src/training/deepspeed_configs/zero2.json

eval:
	uv run python scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

merge:
	uv run python scripts/merge_lora.py --checkpoint $(CHECKPOINT) --output $(OUTPUT)

infer:
	uv run python scripts/inference.py --checkpoint $(CHECKPOINT) --audio $(AUDIO)

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ scripts/ tests/
	uv run ruff format --check src/ scripts/ tests/

format:
	uv run ruff check --fix src/ scripts/ tests/
	uv run ruff format src/ scripts/ tests/

setup-cloud:
	bash scripts/setup_cloud.sh
