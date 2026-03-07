"""Training callbacks for checkpoint management and W&B logging."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logger = logging.getLogger(__name__)


class MakeCheckpointInferableCallback(TrainerCallback):
    """Copy processor config files into each checkpoint directory.

    This ensures checkpoints can be used for inference without
    needing the original model directory.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return

        # Try to copy processor files from the model cache
        try:
            from huggingface_hub import snapshot_download

            cache_dir = snapshot_download(self.model_name, local_files_only=True)
            for fname in [
                "preprocessor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "chat_template.json",
            ]:
                src = Path(cache_dir) / fname
                if src.exists():
                    shutil.copy2(src, checkpoint_dir / fname)
            logger.info(f"Copied processor files to {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"Could not copy processor files: {e}")


class WandbAudioCallback(TrainerCallback):
    """Log audio samples and predictions to Weights & Biases."""

    def __init__(self, log_every_n_steps: int = 500):
        self.log_every_n_steps = log_every_n_steps

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        if state.global_step % self.log_every_n_steps != 0:
            return

        try:
            import wandb

            if wandb.run is None:
                return

            # Log training metrics with step
            if logs:
                wandb.log(
                    {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                    step=state.global_step,
                )
        except ImportError:
            pass
