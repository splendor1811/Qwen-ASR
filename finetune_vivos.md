# VIVOS Finetuning Tutorial

End-to-end guide for finetuning Qwen3-ASR-1.7B on the VIVOS Vietnamese speech dataset using LoRA.

**Dataset**: VIVOS (~15 hours, ~11.6k train / ~760 test utterances)
**GPU**: 1x A100 40GB (or equivalent with >=24GB VRAM)
**Training time**: ~1-2 hours for 5 epochs on a single A100

## Prerequisites

```bash
# Install dependencies (GPU machine) — flash-attn uses prebuilt wheel (~30s, not ~45min build)
uv sync --extra train

# Verify installation
uv run python -c "from qwen_asr import Qwen3ASRModel; print('OK')"
uv run python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')"
```

> **Note**: The prebuilt flash-attn wheel in `pyproject.toml` is configured for the default RunPod template (CUDA 12.4 + PyTorch 2.5 + Python 3.11). If your environment differs, update the URL in `[tool.uv.sources]` — find your wheel at https://mjunya.com/flash-attention-prebuild-wheels/.

## Step 1: Download and Prepare VIVOS Data

Download the raw dataset from HuggingFace and convert it to JSONL format.

```bash
# Download VIVOS
uv run python scripts/download_datasets.py --datasets vivos

# Process into JSONL format
uv run python scripts/prepare_data.py --datasets vivos
```

Verify the output files exist:

```bash
wc -l data/processed/vivos_train.jsonl data/processed/vivos_test.jsonl
```

Expected output (~11,660 train lines, ~760 test lines). Each line is a JSON object with `audio` (path) and `text` (transcription) fields.

## Step 2: Baseline Evaluation

Evaluate the base Qwen3-ASR model **before** finetuning to establish a baseline WER.

```bash
uv run python scripts/evaluate.py \
    --config configs/finetune_vivos.yaml \
    --checkpoint Qwen/Qwen3-ASR-1.7B \
    --benchmarks vivos \
    --output_file outputs/baseline_vivos.json
```

Note the baseline WER and CER — you'll compare against these after training.

Example output:
```
============================================================
EVALUATION RESULTS
============================================================
  vivos                WER:  XX.XX%  CER:  XX.XX%  (n=760)
============================================================
```

## Step 3: Train

Run finetuning with the VIVOS-specific config. This uses LoRA (rank=64) on the LLM decoder while keeping the audio encoder frozen.

```bash
uv run python scripts/train.py --config configs/finetune_vivos.yaml
```

**What to monitor:**
- Training loss should decrease steadily over epochs
- `eval_wer` is computed every 200 steps (~3-4 times per epoch)
- Best checkpoint is saved based on lowest `eval_wer`
- Logs are sent to W&B under project `qwen-asr-vi`, run name `finetune-vivos`

**To resume from a checkpoint** (if training is interrupted):

```bash
uv run python scripts/train.py \
    --config configs/finetune_vivos.yaml \
    --resume_from_checkpoint outputs/finetune_vivos/checkpoint-XXXX
```

Training saves up to 3 checkpoints (controlled by `save_total_limit`). The best checkpoint is loaded at the end of training.

## Step 4: Evaluate Finetuned Model

Evaluate the best checkpoint on the VIVOS test set.

```bash
uv run python scripts/evaluate.py \
    --config configs/finetune_vivos.yaml \
    --checkpoint outputs/finetune_vivos \
    --benchmarks vivos \
    --output_file outputs/finetune_vivos_eval.json
```

## Step 5: Compare Results

Compare baseline vs finetuned performance:

| Model | WER | CER |
|-------|-----|-----|
| Qwen3-ASR-1.7B (baseline) | _fill from Step 2_ | _fill from Step 2_ |
| + LoRA finetune (VIVOS, 5 epochs) | _fill from Step 4_ | _fill from Step 4_ |

You can also compare the JSON files directly:

```bash
cat outputs/baseline_vivos.json
cat outputs/finetune_vivos_eval.json
```

## Step 6: Inference

Test the finetuned model on a real audio file:

```bash
uv run python scripts/inference.py \
    --checkpoint outputs/finetune_vivos \
    --audio path/to/your/audio.wav
```

For interactive mode (enter file paths one by one):

```bash
uv run python scripts/inference.py --checkpoint outputs/finetune_vivos
```

### Optional: Merge LoRA Weights

To create a standalone model without the LoRA adapter (for deployment):

```bash
uv run python scripts/merge_lora.py \
    --checkpoint outputs/finetune_vivos \
    --output outputs/finetune_vivos_merged
```

Then use the merged model directly:

```bash
uv run python scripts/inference.py \
    --checkpoint outputs/finetune_vivos_merged \
    --audio path/to/your/audio.wav
```

## Step 7: Next Steps

Once you've validated the pipeline on VIVOS:

1. **Add more datasets**: Download and process additional datasets (FLEURS, FOSD, PhoAudioBook, etc.):
   ```bash
   uv run python scripts/download_datasets.py --datasets vivos fleurs_vi fosd
   uv run python scripts/prepare_data.py --datasets vivos fleurs_vi fosd --merge
   ```
   The `--merge` flag creates unified `train.jsonl` / `val.jsonl` from all datasets.

2. **Train on combined data**: Use `configs/base.yaml` which points to the merged files:
   ```bash
   uv run python scripts/train.py --config configs/base.yaml
   ```

3. **Multi-GPU training** with DeepSpeed ZeRO-2:
   ```bash
   uv run torchrun --nproc_per_node=4 scripts/train.py \
       --config configs/base.yaml \
       --deepspeed configs/ds_zero2.json
   ```

4. **Evaluate on all benchmarks**:
   ```bash
   uv run python scripts/evaluate.py \
       --config configs/base.yaml \
       --checkpoint outputs/base \
       --benchmarks vivos fleurs_vi
   ```
