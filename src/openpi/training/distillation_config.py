"""Configuration for One-Step π0.5 distillation training."""

import dataclasses
from typing import Literal

import flax.nnx as nnx

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.shared import nnx_utils as nnx_utils


@dataclasses.dataclass(frozen=True)
class DistillConfig(_config.TrainConfig):
    """Training config for distilling π0.5 into a single-step model.

    The student model reuses the teacher's architecture but only trains the action
    projection layers (action_in_proj, action_out_proj, time_mlp_in/out).
    All other parameters (vision encoder, LLM) are frozen.
    """

    # Teacher model checkpoint path
    teacher_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params"

    # Distillation loss configuration
    loss_type: Literal["mse", "weighted_mse", "consistency"] = "mse"
    use_multi_step: bool = False
    multi_step_weight: float = 0.3
    use_cosine_similarity: bool = False
    cosine_weight: float = 0.1

    # Override defaults for distillation
    num_train_steps: int = 50_000
    batch_size: int = 16
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=50_000,
            decay_lr=1e-5,
        )
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(clip_gradient_norm=1.0)
    )
    ema_decay: float | None = 0.999
    wandb_enabled: bool = False

    # FSDP devices for model parallelism across GPUs
    fsdp_devices: int = 2

    @property
    def distill_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze all parameters except action projection layers.

        Only action_in_proj, action_out_proj, and time_mlp layers are trainable.
        All vision (img) and language (llm) parameters are frozen.
        """
        # Match action projection layers that should be trainable
        action_proj_filter = nnx_utils.PathRegex(".*action_(in|out)_proj.*")
        time_mlp_filter = nnx_utils.PathRegex(".*time_mlp.*")

        # Freeze everything EXCEPT action projection layers
        return nnx.All(
            nnx.Param,
            nnx.Not(action_proj_filter),
            nnx.Not(time_mlp_filter),
        )

    @property
    def distill_trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for trainable parameters during distillation."""
        return nnx.All(nnx.Param, nnx.Not(self.distill_freeze_filter))

    def __post_init__(self) -> None:
        # Override the parent's freeze_filter with distillation-specific filter
        object.__setattr__(self, "freeze_filter", self.distill_freeze_filter)
