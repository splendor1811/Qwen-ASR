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
- **torch version MUST match the prebuilt wheel's torch version** (ABI coupling — mismatch causes `undefined symbol` crash)
- To change wheel: find URL at https://mjunya.com/flash-attention-prebuild-wheels/, update `[tool.uv.sources]` URL AND `torch>=X.Y,<X.Z` pin in pyproject.toml

## Commands
- Train: `uv run python scripts/train.py --config configs/base.yaml`
- Evaluate: `uv run python scripts/evaluate.py --config configs/base.yaml --checkpoint <path>`
- Inference: `uv run python scripts/inference.py --checkpoint <path> --audio <file>`
- Prepare data: `uv run python scripts/prepare_data.py --datasets vivos`
- Prepare subset: `uv run python scripts/prepare_data.py --datasets gigaspeech2 --max_samples 500000`
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
- viVoice: `capleaf/viVoice`, gated (CC-BY-NC-SA-4.0), 887k samples, 1017h, train-only, 24kHz→16kHz via librosa
- Large datasets (GigaSpeech2, PhoAudioBook, VietBud500, VLSP, viVoice) use `load_dataset(..., streaming=True)` to avoid HF cache duplication
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

## Text Normalization
- Collator normalizes training labels via `normalize_vietnamese()` from `src/evaluation/normalize_vi.py`
- Normalization: Unicode NFC → lowercase → remove punctuation → collapse whitespace
- Same function used in training AND evaluation (ensures consistency)
- Configurable via `data.normalize_text` in YAML config (default: `true`)
- VIVOS uses UPPERCASE labels; other datasets have mixed case — normalization resolves this
- Raw JSONL data stays untouched; normalization applied at collator time only

## Training Strategy
- Phase 1 (Large SFT): GigaSpeech2-Vi (2000h) + VietBud500 (500h) + VLSP2020 (100h), LR=2e-4
- Phase 2 (Domain Adapt): VietSuperSpeech (103h) + VIVOS (15h) + FLEURS (12h), LR=5e-5, resume from Phase 1
- Phase 3 (Competition): Curated subset, LR=1e-5, resume from Phase 2
- Each phase resumes from previous best checkpoint via `--resume_from_checkpoint`
- VIVOS smoke test config: `configs/finetune_vivos.yaml` (batch=4, grad_accum=4, LR=5e-5, 10 epochs)
- Effective batch size = per_device_batch * gradient_accumulation_steps * n_gpus
- eval_steps must be small enough to actually evaluate during training (not larger than total steps)
