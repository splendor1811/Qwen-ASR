# Qwen-ASR Vietnamese Finetuning

## Project
Vietnamese ASR finetuning of Qwen3-ASR-1.7B using LoRA. Targets VLSP/FLEURS/VIVOS benchmarks.

## Architecture
- Model: `Qwen/Qwen3-ASR-1.7B` loaded via `qwen_asr` package
- Critical: `model.forward = model.thinker.forward` for HF Trainer compatibility
- LoRA applied to LLM decoder only (audio encoder frozen)
- Data collator handles chat template prefix + audio feature extraction

## Environment
- Uses `uv` for Python environment management
- Install: `uv sync --extra dev` (dev) or `uv sync --extra train` (GPU training)
- All commands run via `uv run` or `make`

## Commands
- Train: `uv run python scripts/train.py --config configs/base.yaml`
- Evaluate: `uv run python scripts/evaluate.py --config configs/base.yaml --checkpoint <path>`
- Inference: `uv run python scripts/inference.py --checkpoint <path> --audio <file>`
- Prepare data: `uv run python scripts/prepare_data.py --datasets vivos`
- Merge LoRA: `uv run python scripts/merge_lora.py --checkpoint <path> --output <dir>`
- Tests: `uv run pytest tests/ -v`

## Code Style
- Python 3.10+, type hints on public APIs
- Use dataclasses for configs, YAML for experiment configs
- All audio at 16kHz mono

## Key Patterns
- Data format: JSONL with `{"audio": "/path.wav", "text": "transcription"}`
- Chat template: `<|audio_bos|><|AUDIO|><|audio_eos|>` prefix for ASR
- Labels: prefix tokens masked with -100, only transcription tokens supervised
- ZeRO-2 (not ZeRO-3) with LoRA
