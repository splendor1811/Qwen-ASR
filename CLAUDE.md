# Qwen-ASR Vietnamese Finetuning

## Project
Vietnamese ASR finetuning of Qwen3-ASR-1.7B using LoRA. Targets VLSP/FLEURS/VIVOS benchmarks.

## Architecture
- Model: `Qwen/Qwen3-ASR-1.7B` loaded via `qwen_asr` package
- Critical: `model.forward = model.thinker.forward` for HF Trainer compatibility
- LoRA applied to LLM decoder only (audio encoder frozen)
- Data collator handles chat template prefix + language prefix + audio feature extraction

## Environment
- Uses `uv` for Python environment management
- Install: `uv sync --extra dev` (dev) or `uv sync --extra train` (GPU training)
- All commands run via `uv run` or `make`
- flash-attn uses prebuilt wheel via `[tool.uv.sources]` (avoids 30-60min CUDA build on rented GPUs)
- Prebuilt wheel configured for: CUDA 12.4 + PyTorch 2.5 + Python 3.11 (RunPod default)
- To change wheel: find URL at https://mjunya.com/flash-attention-prebuild-wheels/ and update `[tool.uv.sources]` in pyproject.toml

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

## Data Loading
- datasets v4.x dropped `trust_remote_code` and loading scripts
- VIVOS/FLEURS: use `huggingface_hub.hf_hub_download()` for manual download (no `load_dataset`)
- VietSuperSpeech: uses `snapshot_download` (audio is separate from JSON metadata)
- Correct HF repo IDs: FOSD=`doof-ferb/fpt_fosd`, PhoAudioBook=`thivux/phoaudiobook`
- Common Voice removed (empty on HF, moved to Mozilla Data Collective)
- GigaSpeech2 default split: `train_refined` (not `train`)

## Key Patterns
- Data format: JSONL with `{"audio": "/path.wav", "text": "transcription"}` (plain text, no prefix)
- Collator adds language prefix: `language Vietnamese<asr_text>` before transcription at training time
- Chat template: `<|audio_bos|><|AUDIO|><|audio_eos|>` prefix for ASR
- Labels: chat template masked with -100; language prefix + transcription are supervised
- Configurable via `data.language_prefix` in YAML (default: `"Vietnamese"`, use `"None"` to skip language detection)
- ZeRO-2 (not ZeRO-3) with LoRA
