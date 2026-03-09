# Qwen3-ASR Vietnamese Finetuning

Finetune [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for Vietnamese speech recognition using LoRA. Built to beat current Vietnamese ASR SOTA on public benchmarks.

**Current Vietnamese SOTA baselines (targets to beat):**

| Benchmark | Best WER | Model |
|---|---|---|
| VLSP2020 Test-T1 | 12.29% | Zipformer-30M |
| VLSP2023 PublicTest | 10.40% | Zipformer-30M |
| FLEURS-Vi | ~4.39% | Qwen3-ASR-0.6B (zero-shot) |
| VIVOS | ~5.80% | PhoWhisper-large |

**Why Qwen3-ASR?** The base model already achieves 4.39% WER on FLEURS-Vi *without any Vietnamese finetuning*. With targeted LoRA training on 2000+ hours of Vietnamese data, we expect significant improvements across all benchmarks.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Step 1: Environment Setup](#step-1-environment-setup)
- [Step 2: Get API Tokens](#step-2-get-api-tokens)
- [Step 3: Download Datasets](#step-3-download-datasets)
- [Step 4: Preprocess Data](#step-4-preprocess-data)
- [Step 5: Understand the Config](#step-5-understand-the-config)
- [Step 6: Local Sanity Check](#step-6-local-sanity-check)
- [Step 7: Cloud GPU Setup (RunPod)](#step-7-cloud-gpu-setup-runpod)
- [Step 8: Start Training](#step-8-start-training)
- [Step 9: Monitor Training](#step-9-monitor-training)
- [Step 10: Evaluate](#step-10-evaluate)
- [Step 11: Merge LoRA & Deploy](#step-11-merge-lora--deploy)
- [3-Phase Training Strategy](#3-phase-training-strategy)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## How It Works

Understanding the architecture will help you make better decisions during training. Here's what happens under the hood:

### Model Architecture

Qwen3-ASR-1.7B is a speech-language model with two main components:

```
Audio Input (16kHz WAV)
    |
    v
[Audio Encoder] -----> Frozen during training (pretrained Whisper-style encoder)
    |
    v
Audio Features (mel spectrogram -> transformer features)
    |
    v
[LLM Decoder] -------> This is what we finetune with LoRA
    |                   (Qwen3 language model, ~1.7B params)
    v
Text Output ("xin chao the gioi")
```

**LoRA (Low-Rank Adaptation)** adds small trainable matrices to the LLM decoder's attention and MLP layers. Instead of updating all 1.7B parameters, we only train ~50M parameters (about 3% of the model). This means:
- Much less GPU memory needed (fits on a single A100-80GB)
- Faster training (2-3x vs full finetuning)
- Less risk of catastrophic forgetting (the model keeps its multilingual knowledge)

### Data Flow

```
Raw datasets (HuggingFace)
    |
    v  [download_datasets.py]
Cached on disk
    |
    v  [prepare_data.py]
JSONL files: {"audio": "/path/to/file.wav", "text": "transcription"}
    |
    v  [DataCollatorForQwen3ASRFinetune]
Training batches:
  - input_ids:      [<chat_prefix_tokens>, <lang_prefix + transcription_tokens>, <eos>]
  - labels:         [-100, -100, ..., -100,  <lang_prefix + transcription_tokens>, <eos>]
  - input_features: [audio mel spectrogram features]
    |
    v  [Qwen3ASRTrainer]
Model updates via LoRA
```

The key insight: the chat prefix tokens (system prompt + audio placeholder) are masked with `-100` in the labels. The language prefix (`language Vietnamese<asr_text>`) and the transcription are both supervised — the model learns to predict the language tag and the transcription, but not the chat template.

**Language prefix** (per Qwen3-ASR finetuning guide):
- The collator automatically wraps your text as: `language Vietnamese<asr_text>your transcription`
- Configurable via `data.language_prefix` in your config YAML (default: `"Vietnamese"`)
- Set to `"None"` if you don't have language info (model won't learn language detection)

**Text normalization** (added after VIVOS experiment):
- The collator automatically normalizes training labels: Unicode NFC → lowercase → remove punctuation → collapse spaces
- This uses the same `normalize_vietnamese()` function as evaluation, ensuring train/eval consistency
- Critical because datasets have inconsistent casing (VIVOS=UPPERCASE, others=mixed)
- Configurable via `data.normalize_text` in config YAML (default: `true`)
- Raw JSONL data is NOT modified — normalization happens at training time only

---

## Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** (Python package manager - like pip but faster)
- **Git**
- **GPU for training**: NVIDIA A100-80GB recommended (we'll rent one via RunPod)
- **macOS/Linux for local development** (preprocessing, config editing, evaluation analysis)

Install uv if you haven't:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 1: Environment Setup

Clone the repo and install all dependencies:

```bash
cd /path/to/your/workspace
git clone <your-repo-url> Qwen-ASR
cd Qwen-ASR

# Install all dependencies (dev mode, includes pytest + ruff)
uv sync --extra dev
```

This creates a `.venv/` virtual environment and installs everything defined in `pyproject.toml`. You never need to manually activate the venv - just prefix commands with `uv run` and it handles everything.

Verify the installation:

```bash
# Run the test suite to confirm everything works
uv run pytest tests/ -v
```

You should see all 13 tests pass (config loading, Vietnamese normalization, etc.).

---

## Step 2: Get API Tokens

You need two tokens. Create the `.env` file from the template:

```bash
cp .env.example .env
```

Then edit `.env` with your tokens:

### HuggingFace Token (required)

1. Go to https://huggingface.co/settings/tokens
2. Click "New token" -> name it "qwen-asr" -> select "Read" access
3. Copy the token (starts with `hf_`)
4. **Important**: Go to https://huggingface.co/Qwen/Qwen3-ASR-1.7B and click "Agree" to accept the model's terms
5. If using GigaSpeech2: Go to https://huggingface.co/datasets/speechcolab/gigaspeech2 and accept the dataset terms too

### Weights & Biases Token (recommended, for training monitoring)

1. Go to https://wandb.ai/authorize
2. Copy your API key
3. If you don't want W&B, change `report_to: "wandb"` to `report_to: "none"` in your config YAML

Your `.env` should look like:
```
HF_TOKEN=hf_abcdefghijklmnop
WANDB_API_KEY=1234567890abcdef
WANDB_PROJECT=qwen-asr-vi
```

Login to HuggingFace (needed for dataset downloads):
```bash
uv run huggingface-cli login --token $(grep HF_TOKEN .env | cut -d= -f2)
```

---

## Step 3: Download Datasets

### Available Datasets

| Dataset | Hours | Source | Access | Best For |
|---|---|---|---|---|
| **VIVOS** | ~15h | `AILAB-VNUHCM/vivos` | Open | Testing pipeline, evaluation |
| **FLEURS-Vi** | ~12h | `google/fleurs` (vi_vn) | Open | Evaluation benchmark |
| **VLSP2020** | ~100h | `doof-ferb/vlsp2020_vinai_100h` | Open (unofficial) | Vietnamese broadcast |
| **VietBud500** | ~500h | `linhtran92/viet_bud500` | Open | Large-scale training |
| **VietSuperSpeech** | ~103h | `thanhnew2001/VietSuperSpeech` | Open | Domain adaptation |
| **GigaSpeech2-Vi** | ~2000h | `speechcolab/gigaspeech2` | Gated (needs approval) | Phase 1 backbone |
| **FOSD** | ~100h | `doof-ferb/fpt_fosd` | Open | Additional data (train only) |
| **PhoAudioBook** | ~1500h | `thivux/phoaudiobook` | Gated (needs agreement) | Audiobook domain |

### Start Small: Download VIVOS First

Always start with VIVOS (~15h, smallest dataset) to validate your pipeline end-to-end before downloading larger datasets:

```bash
uv run python scripts/download_datasets.py --datasets vivos
```

This downloads from HuggingFace and caches locally. First download may take a few minutes.

### Download Multiple Datasets

```bash
# Download several datasets at once
uv run python scripts/download_datasets.py --datasets vivos fleurs vlsp

# Download everything (warning: GigaSpeech2 is ~2TB, takes hours)
uv run python scripts/download_datasets.py --datasets all
```

### What Happens During Download

Each dataset processor:
1. Calls `datasets.load_dataset()` from HuggingFace, which caches the data in `~/.cache/huggingface/`
2. The actual audio extraction to WAV files happens in the next step (preprocessing)

---

## Step 4: Preprocess Data

Preprocessing converts raw datasets into a unified JSONL format that our training pipeline expects.

### Process VIVOS (quickest, do this first)

```bash
uv run python scripts/prepare_data.py --datasets vivos
```

This will:
1. Load VIVOS from HuggingFace cache
2. Extract each audio sample as a 16kHz WAV file to `data/raw/vivos/{train,test}/`
3. Write JSONL metadata to `data/processed/vivos_train.jsonl` and `data/processed/vivos_test.jsonl`

### Verify the Output

Check what was created:

```bash
# See the JSONL format
head -3 data/processed/vivos_train.jsonl
```

Each line looks like:
```json
{"audio": "/absolute/path/to/data/raw/vivos/train/VIVOSSPK01_001.wav", "text": "MỘT CÂU NÓI TIẾNG VIỆT"}
```

> **Note**: The JSONL stores plain transcription text. The language prefix (`language Vietnamese<asr_text>`) is added automatically by the data collator at training time.

Check file counts:
```bash
wc -l data/processed/vivos_*.jsonl
```

You should see ~11,660 training samples and ~760 test samples.

### Process Multiple Datasets and Merge

When you're ready for real training, process multiple datasets and merge them into a single training file:

```bash
# Process and merge into data/processed/train.jsonl and data/processed/test.jsonl
uv run python scripts/prepare_data.py --datasets vivos vlsp fleurs --merge
```

The `--merge` flag combines all individual `*_train.jsonl` files into a unified `train.jsonl` (and same for test/val splits). This merged file is what the training config points to by default.

### For Phase 1 Large-Scale Training

```bash
# Process the big datasets (do this on the cloud machine for speed)
uv run python scripts/prepare_data.py --datasets gigaspeech2 vietbud500 vlsp --merge
# Then rename/copy to match the phase1 config:
cp data/processed/train.jsonl data/processed/phase1_train.jsonl
cp data/processed/val.jsonl data/processed/phase1_val.jsonl
```

---

## Step 5: Understand the Config

All training hyperparameters live in YAML files under `configs/`. The system supports inheritance via `_base_`, so you only override what changes.

### Config Structure

```
configs/
  base.yaml                  # All defaults live here
  phase1_large_sft.yaml      # Inherits base, overrides for large-scale training
  phase2_domain_adapt.yaml   # Inherits base, lower LR for domain adaptation
  phase3_competition.yaml    # Inherits base, aggressive tuning for benchmarks
```

### Key Settings in `base.yaml` (annotated)

```yaml
model:
  name: "Qwen/Qwen3-ASR-1.7B"       # Base model from HuggingFace
  attn_implementation: "flash_attention_2"  # Requires flash-attn package + Ampere GPU

lora:
  enabled: true
  rank: 64          # Higher = more capacity but more memory. 64 is a good start.
  alpha: 128        # Usually 2x rank. Controls learning rate scaling for LoRA.
  dropout: 0.05     # Regularization. Increase if overfitting.
  target_modules:   # Which layers get LoRA adapters:
    - q_proj        #   Query projection (attention)
    - k_proj        #   Key projection (attention)
    - v_proj        #   Value projection (attention)
    - o_proj        #   Output projection (attention)
    - gate_proj     #   Gate projection (MLP)
    - up_proj       #   Up projection (MLP)
    - down_proj     #   Down projection (MLP)

freeze:
  freeze_audio_encoder: true   # ALWAYS true - don't touch the audio encoder
  freeze_embeddings: true      # Freeze token embeddings (saves memory)

training:
  per_device_train_batch_size: 1     # Batch size per GPU (1 because audio is large)
  gradient_accumulation_steps: 16    # Effective batch size = 1 * 16 = 16
  learning_rate: 2e-4                # Standard for LoRA finetuning
  lr_scheduler_type: "cosine"        # Cosine annealing (smooth decay)
  warmup_ratio: 0.02                 # 2% of steps for LR warmup
  bf16: true                         # BFloat16 (requires Ampere+ GPU)
  gradient_checkpointing: true       # Trade compute for memory (essential for 80GB)
  save_steps: 500                    # Save checkpoint every 500 steps
  eval_steps: 500                    # Evaluate every 500 steps
  report_to: "wandb"                 # Log to Weights & Biases
```

### How Inheritance Works

`phase1_large_sft.yaml` starts with:
```yaml
_base_: base.yaml    # Inherit everything from base.yaml

training:
  learning_rate: 2e-4              # Same as base
  gradient_accumulation_steps: 32  # Override: larger effective batch
  deepspeed: "src/training/deepspeed_configs/zero2.json"  # Add DeepSpeed
```

Only the fields you specify are overridden. Everything else comes from `base.yaml`.

### Creating Your Own Config

For a quick experiment, create `configs/my_experiment.yaml`:

```yaml
_base_: base.yaml

data:
  train_jsonl: "data/processed/vivos_train.jsonl"
  val_jsonl: "data/processed/vivos_test.jsonl"

training:
  output_dir: "outputs/my_experiment"
  num_train_epochs: 5
  save_steps: 100
  eval_steps: 100

wandb:
  name: "my-first-experiment"
```

---

## Step 6: Local Sanity Check

Before renting a GPU, verify the full pipeline works locally on CPU. This catches config errors, data format issues, and import problems.

### Quick Smoke Test

Create a tiny dataset with just 2 samples:

```bash
# Take the first 2 lines from VIVOS
head -2 data/processed/vivos_train.jsonl > data/processed/smoke_train.jsonl
head -2 data/processed/vivos_test.jsonl > data/processed/smoke_val.jsonl
```

Create a minimal config `configs/smoke_test.yaml`:

```yaml
_base_: base.yaml

model:
  attn_implementation: "eager"  # No flash attention on CPU

data:
  train_jsonl: "data/processed/smoke_train.jsonl"
  val_jsonl: "data/processed/smoke_val.jsonl"

training:
  output_dir: "outputs/smoke_test"
  num_train_epochs: 1
  bf16: false             # CPU doesn't support bf16
  gradient_checkpointing: false
  logging_steps: 1
  save_steps: 999999      # Don't save during smoke test
  eval_steps: 999999
  report_to: "none"       # No W&B for smoke test
  dataloader_num_workers: 0
```

> **Note**: This requires downloading the Qwen3-ASR-1.7B model (~3.4GB). The smoke test is just to verify the pipeline loads and runs one forward pass without errors. It will be extremely slow on CPU - that's expected. Kill it after you see the first training log line.

```bash
uv run python scripts/train.py --config configs/smoke_test.yaml
```

**What success looks like:**
```
Loading model: Qwen/Qwen3-ASR-1.7B
Model loaded: 1,700,000,000 params, 1,700,000,000 trainable
Applying freezing strategy...
Applying LoRA...
LoRA applied: 50,331,648 trainable / 1,750,331,648 total (2.88%)
Loading datasets...
Loaded 2 samples from data/processed/smoke_train.jsonl
Starting training...
{'loss': 8.234, 'learning_rate': 0.0002, 'epoch': 0.5}   <-- SUCCESS! Pipeline works
```

If you see a training loss printed, the pipeline is working. Press Ctrl+C to stop.

---

## Step 7: Cloud GPU Setup (RunPod)

### Renting a GPU

1. Go to [RunPod.io](https://www.runpod.io/) and create an account
2. Add credits ($50 is enough for initial experiments)
3. Go to **GPU Cloud** -> **Deploy**
4. Select:
   - **GPU**: 1x A100 80GB SXM (~$1.74/hr) or 1x A100 80GB PCIe (~$1.44/hr)
   - **Template**: `RunPod Pytorch 2.4` (or similar CUDA 12.x template)
   - **Container Disk**: 100 GB
   - **Volume Disk**: 200 GB (persists between sessions, mount at `/workspace`)
5. Click **Deploy**

### Setting Up the Instance

Once your pod is running, open a terminal (Web Terminal or SSH) and run:

```bash
# Option A: Clone your repo
cd /workspace
git clone <your-repo-url> Qwen-ASR
cd Qwen-ASR

# Option B: If you didn't push to git, use rsync from your local machine:
# (run this on your LOCAL machine, not the pod)
# rsync -avz --exclude '.venv' --exclude 'data' --exclude 'outputs' \
#   /path/to/Qwen-ASR/ root@<pod-ip>:/workspace/Qwen-ASR/
```

Install uv and dependencies on the pod:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # or restart shell

# Install system deps
apt-get update && apt-get install -y ffmpeg libsndfile1

# Install Python dependencies + flash-attn (uses prebuilt wheel, ~30s instead of ~45min)
cd /workspace/Qwen-ASR
uv sync --extra train
```

> **Note on flash-attn**: The `pyproject.toml` is configured to download a prebuilt flash-attn wheel matching the default RunPod template (CUDA 12.4 + PyTorch 2.5 + Python 3.11). This avoids the 30-60 minute CUDA compilation that costs real money on rented GPUs. If your environment differs, find the right wheel at https://mjunya.com/flash-attention-prebuild-wheels/ and update the URL in `[tool.uv.sources]`.

Set up your environment:

```bash
# Create .env with your tokens
cp .env.example .env
nano .env  # Edit with your HF_TOKEN and WANDB_API_KEY

# Login to HuggingFace
source .env
uv run huggingface-cli login --token $HF_TOKEN

# Login to W&B
uv run wandb login $WANDB_API_KEY

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
# Expected: GPU: NVIDIA A100-SXM4-80GB, VRAM: 80.0GB
```

### Download and Prepare Data on the Cloud

Do this on the cloud machine (faster network):

```bash
# Start with VIVOS for a quick validation run
uv run python scripts/download_datasets.py --datasets vivos
uv run python scripts/prepare_data.py --datasets vivos --merge

# Then download the big datasets for real training
uv run python scripts/download_datasets.py --datasets vivos vlsp vietbud500 fleurs
uv run python scripts/prepare_data.py --datasets vivos vlsp vietbud500 fleurs --merge
```

---

## Step 8: Start Training

### Quick Validation Run (VIVOS only, ~30 minutes)

Before committing to a long training run, do a quick 100-step validation:

```bash
uv run python scripts/train.py --config configs/base.yaml
```

With the default config pointing to VIVOS data, this trains on ~15h of data. Watch the loss curve in W&B - it should decrease steadily.

### Full Training with DeepSpeed

For larger datasets, use DeepSpeed ZeRO-2 for memory efficiency:

```bash
# Single GPU with DeepSpeed
uv run python scripts/train.py \
  --config configs/phase1_large_sft.yaml \
  --deepspeed src/training/deepspeed_configs/zero2.json
```

### Multi-GPU Training

If you have multiple GPUs (e.g., 2x A100):

```bash
uv run torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/phase1_large_sft.yaml \
  --deepspeed src/training/deepspeed_configs/zero2.json
```

### Resume from Checkpoint

If training gets interrupted (pod preempted, crash, etc.):

```bash
uv run python scripts/train.py \
  --config configs/phase1_large_sft.yaml \
  --resume_from_checkpoint outputs/phase1_large_sft/checkpoint-5000
```

The trainer automatically finds the latest checkpoint if you point to the output directory:

```bash
uv run python scripts/train.py \
  --config configs/phase1_large_sft.yaml \
  --resume_from_checkpoint outputs/phase1_large_sft
```

### Expected Training Times

| Dataset Size | GPU | Epochs | Approx. Time | Approx. Cost |
|---|---|---|---|---|
| VIVOS (15h) | 1x A100-80GB | 3 | ~1 hour | ~$2 |
| VLSP+VIVOS (115h) | 1x A100-80GB | 3 | ~8 hours | ~$14 |
| Phase 1 (2000h+) | 1x A100-80GB | 2 | ~72 hours | ~$125 |

---

## Step 9: Monitor Training

### Weights & Biases Dashboard

If you configured W&B, go to https://wandb.ai/ -> your project (`qwen-asr-vi`). You'll see:

- **train/loss**: Should decrease steadily. If it plateaus, training is converging.
- **eval/wer**: Word Error Rate on validation set. Lower is better. This is your primary metric.
- **learning_rate**: Should follow a cosine curve (warm up, then decay).

**What to watch for:**
- Loss not decreasing after 500 steps -> Check your data, learning rate may be too low
- Loss oscillating wildly -> Learning rate too high, try halving it
- eval/wer not improving but loss is decreasing -> Possible overfitting, increase dropout or reduce epochs
- NaN loss -> Learning rate too high, or data issue (corrupted audio files)

### Without W&B

Training logs are printed to stdout. You can also check the training state:

```bash
# Check the latest checkpoint
ls -la outputs/base/checkpoint-*/

# View training log
cat outputs/base/trainer_state.json | python -m json.tool | grep -A2 "log_history" | head -20
```

---

## Step 10: Evaluate

### Evaluate a Checkpoint

After training (or at any checkpoint), run evaluation on benchmarks:

```bash
# Evaluate on VIVOS and FLEURS (default benchmarks)
uv run python scripts/evaluate.py \
  --config configs/base.yaml \
  --checkpoint outputs/base/checkpoint-500

# Evaluate on specific benchmarks
uv run python scripts/evaluate.py \
  --config configs/base.yaml \
  --checkpoint outputs/base/checkpoint-500 \
  --benchmarks vivos fleurs_vi vlsp2020

# Save results to a file
uv run python scripts/evaluate.py \
  --config configs/base.yaml \
  --checkpoint outputs/base/checkpoint-500 \
  --output_file results.json
```

### Output Format

```
============================================================
EVALUATION RESULTS
============================================================
  vivos                 WER:   4.52%  CER:   1.23%  (n=760)
  fleurs_vi             WER:   3.87%  CER:   0.95%  (n=423)
============================================================
```

### Evaluate the Base Model (Baseline)

To know how much your finetuning improved things, first get the base model's scores:

```bash
# Evaluate without any finetuning (use the original model)
uv run python scripts/evaluate.py \
  --config configs/base.yaml \
  --checkpoint Qwen/Qwen3-ASR-1.7B \
  --output_file results_baseline.json
```

### Understanding WER

**Word Error Rate (WER)** = (Substitutions + Insertions + Deletions) / Total Reference Words

- **5% WER** = Excellent, near-human performance
- **10% WER** = Good, usable for most applications
- **20% WER** = Fair, needs improvement
- **50%+ WER** = Poor, something is wrong

Our Vietnamese text normalization (`src/evaluation/normalize_vi.py`) handles:
- Unicode NFC normalization (critical for Vietnamese diacritics)
- Lowercasing
- Punctuation removal (keeps Vietnamese diacritics like a, e, o, u)
- Whitespace collapsing

This ensures fair comparison: "Xin Chao!" and "xin chao" are treated as identical.

---

## Step 11: Merge LoRA & Deploy

### Merge LoRA Weights

LoRA checkpoints are small (~200MB) but require the base model at inference time. Merging "bakes in" the LoRA weights permanently:

```bash
uv run python scripts/merge_lora.py \
  --checkpoint outputs/base/checkpoint-500 \
  --output outputs/merged_model
```

The merged model at `outputs/merged_model/` is a standalone model - no PEFT dependency needed.

### Interactive Inference

Test your model on individual audio files:

```bash
# With LoRA checkpoint (not merged)
uv run python scripts/inference.py \
  --checkpoint outputs/base/checkpoint-500 \
  --audio /path/to/test.wav

# With merged model
uv run python scripts/inference.py \
  --checkpoint outputs/merged_model \
  --audio /path/to/test.wav

# Interactive mode (enter file paths one at a time)
uv run python scripts/inference.py \
  --checkpoint outputs/merged_model
```

### Using the Merged Model in Your Own Code

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "outputs/merged_model",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(audio="path/to/vietnamese_audio.wav")
print(results[0].text)
```

---

## 3-Phase Training Strategy

This codebase is designed for a progressive training approach. Each phase **continues from the previous phase's best checkpoint** via `--resume_from_checkpoint`. Text normalization is automatic in the collator (controlled by `data.normalize_text`, default `true`).

### Phase 1: Large-Scale SFT (config: `phase1_large_sft.yaml`)

**Goal**: Teach the model Vietnamese speech patterns at scale.

| Dataset | Hours | HuggingFace Repo | Access |
|---|---|---|---|
| GigaSpeech2-Vi | ~2000h | `speechcolab/gigaspeech2` | Gated (needs approval) |
| VietBud500 | ~500h | `linhtran92/viet_bud500` | Open |
| VLSP2020 | ~100h | `doof-ferb/vlsp2020_vinai_100h` | Open |

- **Total**: ~2600h
- **Duration**: ~72h on 1x A100-80GB | **Cost**: ~$125
- **LR**: 2e-4, cosine decay | **LoRA**: rank=64, alpha=128
- **Expected**: Initial loss ~8-9, WER should drop below 10% on VIVOS by end of training

**Data preparation:**
```bash
# Download datasets (do this on the cloud machine for speed)
uv run python scripts/download_datasets.py --datasets gigaspeech2 vietbud500 vlsp

# Process and merge into unified JSONL
uv run python scripts/prepare_data.py --datasets gigaspeech2 vietbud500 vlsp --merge

# Rename to match phase1 config paths
cp data/processed/train.jsonl data/processed/phase1_train.jsonl

# Create a validation set (sample 1000 from train, or use VIVOS test)
shuf data/processed/phase1_train.jsonl | head -1000 > data/processed/phase1_val.jsonl
```

**Start training:**
```bash
uv run python scripts/train.py --config configs/phase1_large_sft.yaml
```

### Phase 2: Domain Adaptation (config: `phase2_domain_adapt.yaml`)

**Goal**: Adapt to specific Vietnamese speech domains (conversational, read speech). **Continues from Phase 1 checkpoint** — the lower LR (5e-5 vs 2e-4) refines without forgetting Phase 1 knowledge.

| Dataset | Hours | HuggingFace Repo | Access |
|---|---|---|---|
| VietSuperSpeech | ~103h | `thanhnew2001/VietSuperSpeech` | Open |
| VIVOS | ~15h | `AILAB-VNUHCM/vivos` | Open |
| FLEURS-Vi | ~12h | `google/fleurs` (vi_vn) | Open |

- **Total**: ~130h
- **Duration**: ~12h | **Cost**: ~$21
- **LR**: 5e-5 (lower, to refine without forgetting Phase 1)
- **Start from**: Best Phase 1 checkpoint

**Data preparation:**
```bash
uv run python scripts/download_datasets.py --datasets vietsuperspeech vivos fleurs
uv run python scripts/prepare_data.py --datasets vietsuperspeech vivos fleurs --merge
cp data/processed/train.jsonl data/processed/phase2_train.jsonl
cp data/processed/val.jsonl data/processed/phase2_val.jsonl
```

**Start training (resume from Phase 1):**
```bash
uv run python scripts/train.py --config configs/phase2_domain_adapt.yaml \
  --resume_from_checkpoint outputs/phase1_large_sft/checkpoint-BEST
```

### Phase 3: Competition Tuning (config: `phase3_competition.yaml`)

**Goal**: Squeeze out the last WER improvements on target benchmarks.

- **Data**: Curated subset of highest-quality data (clean labels only)
- **Duration**: ~4h
- **LR**: 1e-5 (very low, surgical refinement)
- **LoRA**: rank=128, alpha=256 (more capacity for fine details)
- **Start from**: Best Phase 2 checkpoint

```bash
uv run python scripts/train.py --config configs/phase3_competition.yaml \
  --resume_from_checkpoint outputs/phase2_domain_adapt/checkpoint-BEST
```

### VIVOS Smoke Test (config: `finetune_vivos.yaml`)

For quick pipeline validation on a single small dataset:

```bash
uv run python scripts/train.py --config configs/finetune_vivos.yaml
```

Uses conservative hyperparameters (batch=4, grad_accum=4, LR=5e-5, 10 epochs) to avoid the WER degradation seen with aggressive settings.

### Recommended Order

```
Beginner path (quick results):
  VIVOS only (15h) -> finetune_vivos.yaml -> evaluate
  Total: ~1h, ~$2

Full pipeline:
  Phase 1 (2600h) -> Phase 2 (130h) -> Phase 3 (curated) -> evaluate all benchmarks
  Total: ~90h, ~$160
```

---

## Configuration Reference

### All Config Fields

| Section | Field | Default | Description |
|---|---|---|---|
| `model` | `name` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID |
| `model` | `torch_dtype` | `bfloat16` | Model precision (`bfloat16`, `float16`, `float32`) |
| `model` | `attn_implementation` | `flash_attention_2` | Use `eager` for CPU/non-Ampere GPUs |
| `lora` | `enabled` | `true` | Set `false` for full finetuning (not recommended) |
| `lora` | `rank` | `64` | LoRA rank (higher = more params, more capacity) |
| `lora` | `alpha` | `128` | LoRA alpha (usually 2x rank) |
| `lora` | `dropout` | `0.05` | LoRA dropout (increase if overfitting) |
| `freeze` | `freeze_audio_encoder` | `true` | Always keep `true` |
| `freeze` | `freeze_embeddings` | `true` | Saves memory, usually keep `true` |
| `data` | `train_jsonl` | `data/processed/train.jsonl` | Training data path |
| `data` | `val_jsonl` | `data/processed/val.jsonl` | Validation data path |
| `data` | `max_audio_duration` | `30.0` | Max audio length in seconds |
| `data` | `sample_rate` | `16000` | Audio sample rate (always 16kHz) |
| `data` | `language_prefix` | `Vietnamese` | Language tag for Qwen3-ASR (`Vietnamese`, `None`, etc.) |
| `data` | `normalize_text` | `true` | Normalize training text (lowercase, remove punctuation). Must match eval normalization. |
| `training` | `per_device_train_batch_size` | `1` | Batch size per GPU |
| `training` | `gradient_accumulation_steps` | `16` | Effective batch = this * batch_size * n_gpus |
| `training` | `learning_rate` | `2e-4` | Peak learning rate |
| `training` | `num_train_epochs` | `3` | Number of training epochs |
| `training` | `bf16` | `true` | BFloat16 training (requires Ampere+ GPU) |
| `training` | `gradient_checkpointing` | `true` | Saves VRAM, slight speed penalty |
| `training` | `deepspeed` | `null` | Path to DeepSpeed config (for large training) |
| `training` | `report_to` | `wandb` | Logging backend (`wandb` or `none`) |

---

## Project Structure

```
Qwen-ASR/
├── CLAUDE.md                          # AI assistant context
├── README.md                          # This file
├── pyproject.toml                     # Dependencies and project config
├── Makefile                           # Convenience commands
├── .gitignore
├── .env.example                       # Template for API tokens
│
├── configs/
│   ├── base.yaml                      # Default config (all settings)
│   ├── phase1_large_sft.yaml          # Phase 1: Large-scale SFT
│   ├── phase2_domain_adapt.yaml       # Phase 2: Domain adaptation
│   └── phase3_competition.yaml        # Phase 3: Competition tuning
│
├── scripts/
│   ├── train.py                       # Main training script
│   ├── evaluate.py                    # Benchmark evaluation
│   ├── inference.py                   # Interactive transcription
│   ├── merge_lora.py                  # Merge LoRA into base model
│   ├── download_datasets.py           # Download datasets from HuggingFace
│   ├── prepare_data.py                # Convert to JSONL format
│   └── setup_cloud.sh                 # Cloud instance setup
│
├── src/
│   ├── config.py                      # Dataclass configs + YAML loader
│   ├── data/
│   │   ├── collator.py                # DataCollator (audio -> training batch)
│   │   ├── datasets.py                # JSONL -> HuggingFace Dataset
│   │   ├── utils.py                   # Audio loading, resampling
│   │   └── processors/               # One file per dataset
│   │       ├── base.py                # Abstract processor interface
│   │       ├── vivos.py               # VIVOS (~15h)
│   │       ├── vlsp.py                # VLSP2020 (~100h)
│   │       ├── fleurs.py              # FLEURS-Vi (~12h)
│   │       ├── gigaspeech2.py         # GigaSpeech2-Vi (~2000h)
│   │       ├── vietbud500.py          # VietBud500 (~500h)
│   │       ├── vietsuperspeech.py     # VietSuperSpeech (~103h)
│   │       ├── fosd.py                # FOSD (~100h)
│   │       └── phoaudiobook.py        # PhoAudioBook (~1500h)
│   ├── model/
│   │   ├── loader.py                  # Load Qwen3-ASR model + processor
│   │   ├── lora.py                    # Apply LoRA adapters
│   │   └── freezing.py                # Freeze encoder/embeddings
│   ├── training/
│   │   ├── trainer.py                 # Custom HF Trainer subclass
│   │   ├── callbacks.py               # Checkpoint + W&B callbacks
│   │   └── deepspeed_configs/
│   │       └── zero2.json             # DeepSpeed ZeRO-2 config
│   ├── evaluation/
│   │   ├── metrics.py                 # WER/CER computation
│   │   ├── normalize_vi.py            # Vietnamese text normalization
│   │   └── benchmarks.py              # Load test sets
│   └── utils/
│       └── logging.py                 # Structured logging setup
│
└── tests/
    ├── test_config.py                 # Config loading tests
    ├── test_collator.py               # Data collator tests (integration)
    └── test_normalize_vi.py           # Vietnamese normalization tests
```

---

## Troubleshooting

### "No module named 'flash_attn'"

Flash Attention requires a compatible GPU (Ampere/A100+). On CPU or older GPUs:
```yaml
# In your config YAML:
model:
  attn_implementation: "eager"
```

If you're on a GPU machine but flash-attn failed to install, check that the prebuilt wheel in `[tool.uv.sources]` matches your environment (CUDA version, PyTorch version, Python version). Find the right wheel at https://mjunya.com/flash-attention-prebuild-wheels/.

### "CUDA out of memory"

Reduce memory usage (try in this order):
1. Ensure `gradient_checkpointing: true` in config
2. Reduce `per_device_train_batch_size` to 1
3. Reduce `max_audio_duration` to 20 or 15 seconds
4. Use DeepSpeed ZeRO-2: add `--deepspeed src/training/deepspeed_configs/zero2.json`
5. Reduce LoRA `rank` from 64 to 32

### "FileNotFoundError: JSONL file not found"

You need to preprocess data before training:
```bash
uv run python scripts/prepare_data.py --datasets vivos --merge
```

### Loss is NaN

- Check for corrupted audio files in your dataset
- Try reducing learning rate to `1e-4`
- Ensure `max_grad_norm: 1.0` is set (gradient clipping)

### Training is extremely slow

- Verify GPU is being used: `nvidia-smi` should show GPU utilization
- Ensure flash-attn is installed: `uv sync --extra train` (uses prebuilt wheel)
- Ensure `bf16: true` (not `float32`)
- Check `dataloader_num_workers: 4` (not 0)

### "GigaSpeech2 access denied"

GigaSpeech2 is a gated dataset. You need to:
1. Go to https://huggingface.co/datasets/speechcolab/gigaspeech2
2. Click "Agree" to accept the terms
3. Wait for approval (usually instant)
4. Make sure your `HF_TOKEN` is set in `.env`

### W&B not logging

Check that:
1. `WANDB_API_KEY` is set in `.env`
2. `report_to: "wandb"` in your config (not `"none"`)
3. Run `uv run wandb login` to verify authentication

### Pod got preempted / training interrupted

Resume from the latest checkpoint:
```bash
uv run python scripts/train.py \
  --config configs/phase1_large_sft.yaml \
  --resume_from_checkpoint outputs/phase1_large_sft
```

Checkpoints include the optimizer state, so training continues exactly where it left off.
