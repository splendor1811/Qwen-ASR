# DeepSpeed Study Guide for Qwen-ASR Training

A practical guide to understand DeepSpeed in the context of this project. Written for someone who has never used DeepSpeed before.

---

## Table of Contents

1. [What Problem Does DeepSpeed Solve?](#1-what-problem-does-deepspeed-solve)
2. [GPU Memory Breakdown](#2-gpu-memory-breakdown)
3. [ZeRO Stages Explained](#3-zero-stages-explained)
4. [How DeepSpeed Works in This Project](#4-how-deepspeed-works-in-this-project)
5. [The ZeRO-2 Config Explained Line by Line](#5-the-zero-2-config-explained-line-by-line)
6. [How to Run Training With/Without DeepSpeed](#6-how-to-run-training-withwithout-deepspeed)
7. [Single GPU vs Multi-GPU](#7-single-gpu-vs-multi-gpu)
8. [When to Use DeepSpeed](#8-when-to-use-deepspeed)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Quick Reference](#10-quick-reference)

---

## 1. What Problem Does DeepSpeed Solve?

Training a large model requires storing **much more than just the model weights** in GPU memory. For a 1.7B parameter model like Qwen3-ASR:

```
Model weights:          ~3.4 GB  (1.7B params × 2 bytes in bf16)
Optimizer states:       ~13.6 GB (Adam stores 2 copies in fp32 = 4× model size)
Gradients:              ~3.4 GB  (same size as model weights)
Activations:            ~5-20 GB (depends on batch size and sequence length)
─────────────────────────────────
Total:                  ~25-40 GB
```

On a single A100-80GB, this fits. But when you have 2600+ hours of data with gradient accumulation, batch sizes, and audio features, memory gets tight.

**DeepSpeed ZeRO** (Zero Redundancy Optimizer) reduces memory usage by partitioning optimizer states, gradients, and optionally model parameters across GPUs — or even to CPU RAM.

**Key insight**: Even on a **single GPU**, DeepSpeed helps because it manages memory more efficiently than vanilla PyTorch (better memory fragmentation, gradient management, etc.).

---

## 2. GPU Memory Breakdown

Here's what lives in GPU memory during training:

```
┌─────────────────────────────────────────┐
│              GPU Memory (80GB)          │
│                                         │
│  ┌──────────────────────┐               │
│  │  Model Parameters    │  ~3.4 GB      │  The actual neural network weights
│  │  (bf16)              │               │  (frozen encoder + LoRA adapters)
│  └──────────────────────┘               │
│                                         │
│  ┌──────────────────────┐               │
│  │  Optimizer States    │  ~13.6 GB     │  Adam keeps:
│  │  (fp32)              │               │    - 1st moment (mean of gradients)
│  │                      │               │    - 2nd moment (variance of gradients)
│  │                      │               │    - fp32 copy of parameters
│  └──────────────────────┘               │
│                                         │
│  ┌──────────────────────┐               │
│  │  Gradients           │  ~3.4 GB      │  ∂Loss/∂parameters for each
│  │  (bf16)              │               │  trainable parameter
│  └──────────────────────┘               │
│                                         │
│  ┌──────────────────────┐               │
│  │  Activations         │  ~5-20 GB     │  Intermediate values saved for
│  │  (varies)            │               │  backward pass (reduced by
│  │                      │               │  gradient_checkpointing)
│  └──────────────────────┘               │
│                                         │
│  ┌──────────────────────┐               │
│  │  Audio Features      │  ~1-5 GB      │  Mel spectrograms, encoder outputs
│  └──────────────────────┘               │
└─────────────────────────────────────────┘
```

**The optimizer states are the biggest memory hog** — 4× the model size! This is what DeepSpeed ZeRO primarily targets.

---

## 3. ZeRO Stages Explained

DeepSpeed ZeRO has 3 stages. Each stage partitions more data across GPUs:

### ZeRO Stage 1: Partition Optimizer States
```
Without ZeRO:
  GPU 0: [All optimizer states] [All gradients] [All params]
  GPU 1: [All optimizer states] [All gradients] [All params]  ← duplicated!

ZeRO-1:
  GPU 0: [Optimizer states 1/2] [All gradients] [All params]
  GPU 1: [Optimizer states 2/2] [All gradients] [All params]
  → Saves ~50% optimizer memory
```

### ZeRO Stage 2: + Partition Gradients ← **We use this**
```
ZeRO-2:
  GPU 0: [Optimizer states 1/2] [Gradients 1/2] [All params]
  GPU 1: [Optimizer states 2/2] [Gradients 2/2] [All params]
  → Saves ~50% optimizer + ~50% gradient memory
```

### ZeRO Stage 3: + Partition Parameters
```
ZeRO-3:
  GPU 0: [Optimizer 1/2] [Gradients 1/2] [Params 1/2]
  GPU 1: [Optimizer 1/2] [Gradients 1/2] [Params 2/2]
  → Maximum memory savings, but slower (needs all-gather for each forward pass)
```

### Why ZeRO-2 (not ZeRO-3) for LoRA?

With LoRA, only ~3% of parameters are trainable. ZeRO-3 would partition the **full model parameters** across GPUs, requiring expensive all-gather operations during every forward pass. Since LoRA already makes the trainable parameters small, ZeRO-2 gives the best **memory savings vs speed** tradeoff:

```
LoRA trainable params:    ~50M  (small gradients + optimizer states)
Frozen params:            ~1.65B (just need one copy, no gradients/optimizer)
→ ZeRO-3's param partitioning adds overhead for minimal benefit
→ ZeRO-2 is the sweet spot
```

---

## 4. How DeepSpeed Works in This Project

### The Flow

```
YAML Config (e.g., phase1_large_sft.yaml)
    │
    │  training:
    │    deepspeed: "src/training/deepspeed_configs/zero2.json"
    │
    ▼
scripts/train.py
    │
    │  1. Loads config → config.training.deepspeed = "src/.../zero2.json"
    │  2. Sets device_map = None (DeepSpeed handles device placement)
    │  3. Passes to TrainingArguments(deepspeed=config.training.deepspeed)
    │
    ▼
HuggingFace Trainer
    │
    │  Reads zero2.json, initializes DeepSpeed engine internally
    │  Wraps model, optimizer, scheduler with DeepSpeed
    │
    ▼
Training loop runs with DeepSpeed managing memory
```

### Key Code in `scripts/train.py`

```python
# Line 101: DeepSpeed handles device placement — don't use device_map="auto"
device_map = None if config.training.deepspeed else "auto"

# Line 183: Pass DeepSpeed config path to HuggingFace TrainingArguments
training_args = TrainingArguments(
    ...
    deepspeed=config.training.deepspeed,  # path to zero2.json or None
    ...
)
```

When `deepspeed` is `None`, training runs with vanilla PyTorch. When it's a path to a JSON file, HuggingFace Trainer automatically initializes DeepSpeed.

### The `"auto"` Fields

In `zero2.json`, several fields are set to `"auto"`:

```json
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto"
```

`"auto"` means **read the value from HuggingFace TrainingArguments**. This avoids having to keep two configs in sync — you set `gradient_accumulation_steps: 16` in your YAML config, and DeepSpeed reads it from TrainingArguments automatically.

### Environment Variables for Single-GPU

```python
def _ensure_distributed_env():
    """Set env vars for single-GPU DeepSpeed when launched via plain `python`."""
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
```

DeepSpeed normally expects to be launched via `torchrun` or `deepspeed` launcher, which sets these environment variables. This helper lets you run `python scripts/train.py` directly on a single GPU without a launcher.

---

## 5. The ZeRO-2 Config Explained Line by Line

File: `src/training/deepspeed_configs/zero2.json`

```json
{
  "bf16": {
    "enabled": true                          // Use BFloat16 mixed precision
  },                                         // (matches training.bf16 in YAML)

  "zero_optimization": {
    "stage": 2,                              // ZeRO Stage 2: partition optimizer + gradients

    "offload_optimizer": {
      "device": "none"                       // Keep optimizer on GPU (not CPU)
    },                                       // "cpu" would save GPU RAM but 10-50x slower

    "allgather_partitions": true,            // After backward pass, gather updated params
                                             // from all GPUs (needed for next forward pass)

    "allgather_bucket_size": 2e8,            // 200MB buckets for all-gather operations
                                             // Larger = fewer operations, more memory
                                             // Smaller = less memory, more overhead

    "overlap_comm": true,                    // Overlap GPU computation with network
                                             // communication (pipeline parallelism)
                                             // ~10-15% speedup on multi-GPU

    "reduce_scatter": true,                  // Use reduce-scatter instead of all-reduce
                                             // for gradient reduction (more memory efficient)

    "reduce_bucket_size": 2e8,               // 200MB buckets for gradient reduction
                                             // Same tradeoff as allgather_bucket_size

    "contiguous_gradients": true             // Store gradients in contiguous memory block
                                             // Reduces memory fragmentation
  },

  "gradient_accumulation_steps": "auto",     // Read from TrainingArguments
  "gradient_clipping": "auto",              // Read from max_grad_norm in TrainingArguments
  "train_batch_size": "auto",               // = per_device_batch × grad_accum × num_gpus
  "train_micro_batch_size_per_gpu": "auto", // = per_device_train_batch_size
  "wall_clock_breakdown": false              // Don't profile timing (set true to debug perf)
}
```

---

## 6. How to Run Training With/Without DeepSpeed

### Without DeepSpeed (small datasets, single GPU)

```bash
# VIVOS smoke test — no DeepSpeed needed
uv run python scripts/train.py --config configs/finetune_vivos.yaml
```

The YAML has no `deepspeed` field → defaults to `None` → vanilla PyTorch training.

### With DeepSpeed via YAML config

```bash
# Phase 1 — deepspeed is set in the YAML file
uv run python scripts/train.py --config configs/phase1_large_sft.yaml
```

`phase1_large_sft.yaml` contains:
```yaml
training:
  deepspeed: "src/training/deepspeed_configs/zero2.json"
```

### With DeepSpeed via CLI override

```bash
# Add DeepSpeed to ANY config, even base.yaml
uv run python scripts/train.py \
  --config configs/base.yaml \
  --deepspeed src/training/deepspeed_configs/zero2.json
```

The `--deepspeed` CLI flag overrides whatever is in the YAML.

### Multi-GPU with DeepSpeed

```bash
# 2 GPUs
uv run torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/phase1_large_sft.yaml

# Or use the Makefile shortcut
make train-ds CONFIG=configs/phase1_large_sft.yaml
```

`torchrun` sets the distributed environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`) automatically.

---

## 7. Single GPU vs Multi-GPU

### Single GPU

```
┌──────────────────────────────┐
│         GPU 0 (80GB)         │
│                              │
│  ZeRO-2 still helps:        │
│  - Better memory management │
│  - Contiguous gradients     │
│  - Efficient grad clipping  │
│                              │
│  No partitioning benefit    │
│  (only 1 GPU to partition   │
│   across)                    │
└──────────────────────────────┘
```

On a single GPU, ZeRO-2 helps with memory management but can't partition across GPUs. Still useful for:
- Gradient accumulation efficiency
- Mixed precision management
- Memory fragmentation reduction

### Multi-GPU (e.g., 2× A100)

```
┌──────────────────────┐     ┌──────────────────────┐
│      GPU 0 (80GB)    │     │      GPU 1 (80GB)    │
│                      │     │                      │
│  Optimizer states 1/2│     │  Optimizer states 2/2│
│  Gradients 1/2       │     │  Gradients 2/2       │
│  All params          │◄───►│  All params          │
│                      │comm │                      │
│  ~50% less memory    │     │  ~50% less memory    │
│  per GPU             │     │  per GPU             │
└──────────────────────┘     └──────────────────────┘
```

The real power of ZeRO-2 shines with multiple GPUs — each GPU only stores half the optimizer states and gradients.

### Effective Batch Size with Multi-GPU

```
effective_batch = per_device_batch × gradient_accumulation × num_gpus

Single GPU:   1 × 16 × 1 = 16
Two GPUs:     1 × 16 × 2 = 32  ← doubled!
```

If you add a second GPU, halve `gradient_accumulation_steps` to keep the same effective batch size:
```yaml
# 2 GPUs: reduce grad_accum to maintain batch size of 16
training:
  gradient_accumulation_steps: 8  # was 16 for single GPU
```

---

## 8. When to Use DeepSpeed

| Scenario | DeepSpeed? | Why |
|---|---|---|
| VIVOS smoke test (15h) | No | Small data, fits easily without it |
| Single A100, base.yaml | Optional | Helps with memory management |
| Phase 1 (2600h+), single GPU | **Yes** | Long training, need stable memory |
| Phase 1, multi-GPU | **Yes** | Partitioning saves significant memory |
| CPU smoke test | No | DeepSpeed requires CUDA |
| Inference / Evaluation | No | No optimizer states, no gradients |

**Rule of thumb**: Use DeepSpeed for any training run longer than 1 hour or with more than 100h of data.

---

## 9. Common Pitfalls

### 1. `device_map` conflict

```python
# WRONG — DeepSpeed and device_map="auto" will fight over device placement
model = load_model(device_map="auto")
TrainingArguments(deepspeed="zero2.json")

# CORRECT — let DeepSpeed handle device placement
model = load_model(device_map=None)  # our code does this automatically
TrainingArguments(deepspeed="zero2.json")
```

Our `train.py` handles this at line 101:
```python
device_map = None if config.training.deepspeed else "auto"
```

### 2. Mismatched batch sizes

If you set `train_batch_size` in the DeepSpeed JSON AND in TrainingArguments, they might conflict. That's why we use `"auto"` — DeepSpeed reads from TrainingArguments.

### 3. ZeRO-3 with LoRA

Don't use ZeRO-3 with LoRA. It partitions ALL parameters (including frozen ones) across GPUs, adding communication overhead for parameters that don't even need gradients. ZeRO-2 is better for LoRA.

### 4. CPU offloading too aggressive

```json
// This will make training 10-50x slower:
"offload_optimizer": { "device": "cpu" }

// Keep optimizer on GPU unless you're truly out of VRAM:
"offload_optimizer": { "device": "none" }
```

Only use CPU offloading as a last resort if you're getting OOM errors after trying everything else.

### 5. Saving/Loading checkpoints

DeepSpeed checkpoints have a different format than vanilla PyTorch. HuggingFace Trainer handles this transparently — `--resume_from_checkpoint` works the same way with or without DeepSpeed.

---

## 10. Quick Reference

### Memory savings summary

| Strategy | GPU Memory Saved | Speed Impact |
|---|---|---|
| bf16 (vs fp32) | ~50% model memory | Neutral/faster |
| Gradient checkpointing | ~60-70% activation memory | ~20-30% slower |
| LoRA (vs full finetune) | ~97% optimizer memory | Faster |
| ZeRO-2 (single GPU) | ~10-15% (fragmentation) | Neutral |
| ZeRO-2 (2 GPUs) | ~50% optimizer + gradients | ~5% comm overhead |
| ZeRO-3 | Maximum savings | ~15-30% slower |
| CPU offloading | Moves optimizer to RAM | 10-50x slower |

### Our training stack

```
Qwen3-ASR-1.7B (1.7B params)
    + LoRA (rank=64, only ~50M trainable = 3%)
    + Frozen audio encoder
    + bf16 mixed precision
    + Gradient checkpointing
    + ZeRO-2 (for Phase 1/2/3)
    = Fits on 1× A100-80GB comfortably
```

### Files involved

| File | Role |
|---|---|
| `src/training/deepspeed_configs/zero2.json` | DeepSpeed config (ZeRO-2 settings) |
| `src/config.py` | `TrainingConfig.deepspeed: Optional[str]` field |
| `scripts/train.py` | Reads config, passes to `TrainingArguments(deepspeed=...)` |
| `configs/phase1_large_sft.yaml` | Sets `training.deepspeed` path |
| `configs/phase2_domain_adapt.yaml` | Sets `training.deepspeed` path |
| `configs/phase3_competition.yaml` | Sets `training.deepspeed` path |
| `configs/base.yaml` | No DeepSpeed (default) |
| `configs/finetune_vivos.yaml` | No DeepSpeed (small dataset) |

### Commands cheat sheet

```bash
# Without DeepSpeed
uv run python scripts/train.py --config configs/finetune_vivos.yaml

# With DeepSpeed (from YAML)
uv run python scripts/train.py --config configs/phase1_large_sft.yaml

# With DeepSpeed (CLI override)
uv run python scripts/train.py --config configs/base.yaml \
  --deepspeed src/training/deepspeed_configs/zero2.json

# Multi-GPU with DeepSpeed
uv run torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/phase1_large_sft.yaml

# Makefile shortcut
make train-ds CONFIG=configs/phase1_large_sft.yaml
```

---

## Further Reading

- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054) — the original research
- [HuggingFace DeepSpeed integration docs](https://huggingface.co/docs/transformers/main/en/deepspeed)
- [DeepSpeed configuration docs](https://www.deepspeed.ai/docs/config-json/)
