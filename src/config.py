"""Configuration dataclasses and YAML loader with inheritance support."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-ASR-1.7B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class FreezeConfig:
    freeze_audio_encoder: bool = True
    freeze_embeddings: bool = True
    freeze_lm_head: bool = False


@dataclass
class DataConfig:
    train_jsonl: str = "data/processed/train.jsonl"
    val_jsonl: str = "data/processed/val.jsonl"
    max_audio_duration: float = 30.0
    min_audio_duration: float = 0.5
    sample_rate: int = 16000
    max_text_length: int = 512
    num_workers: int = 4
    language_prefix: str = "Vietnamese"


@dataclass
class TrainingConfig:
    output_dir: str = "outputs/default"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.02
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_wer"
    greater_is_better: bool = False
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    report_to: str = "wandb"
    seed: int = 42
    max_grad_norm: float = 1.0
    deepspeed: Optional[str] = None
    ddp_find_unused_parameters: bool = False


@dataclass
class WandbConfig:
    project: str = "qwen-asr-vi"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    benchmarks: list[str] = field(
        default_factory=lambda: ["vivos", "fleurs_vi"]
    )
    batch_size: int = 1
    num_beams: int = 1
    max_new_tokens: int = 512


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    freeze: FreezeConfig = field(default_factory=FreezeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_env_vars(data: dict) -> dict:
    """Resolve ${ENV_VAR} placeholders in string values."""
    for key, value in data.items():
        if isinstance(value, dict):
            _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_key = value[2:-1]
            data[key] = os.environ.get(env_key, value)
    return data


def _dataclass_from_dict(cls, data: dict):
    """Create a dataclass instance from a dict, ignoring extra keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment config from YAML with _base_ inheritance support."""
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    # Handle _base_ inheritance
    if "_base_" in raw:
        base_path = config_path.parent / raw.pop("_base_")
        with open(base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        raw = _deep_update(base_raw, raw)

    raw = _resolve_env_vars(raw)

    return ExperimentConfig(
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        lora=_dataclass_from_dict(LoRAConfig, raw.get("lora", {})),
        freeze=_dataclass_from_dict(FreezeConfig, raw.get("freeze", {})),
        data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
        wandb=_dataclass_from_dict(WandbConfig, raw.get("wandb", {})),
        eval=_dataclass_from_dict(EvalConfig, raw.get("eval", {})),
    )
