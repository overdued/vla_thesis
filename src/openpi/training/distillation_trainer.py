"""Distillation trainer for One-Step π0.5.

Trains a student model to predict clean actions in a single forward pass,
using a frozen teacher model's 10-step ODE output as the supervision signal.
"""

import dataclasses
import functools
import logging
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

logger = logging.getLogger("openpi")


def init_teacher_model(
    config: _config.TrainConfig,
    checkpoint_path: str,
    mesh: jax.sharding.Mesh,
) -> _model.BaseModel:
    """Load and freeze the teacher model from a checkpoint.

    Args:
        config: The training config used to create the model architecture.
        checkpoint_path: Path to the teacher checkpoint.
        mesh: JAX mesh for device placement.

    Returns:
        The loaded teacher model with all parameters frozen.
    """
    logger.info(f"Loading teacher model from {checkpoint_path}")

    # Create the model architecture
    rng = jax.random.key(0)
    model = config.model.create(rng)

    # Load checkpoint params
    params = _model.restore_params(checkpoint_path, dtype=jnp.bfloat16)

    # Merge params into model
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)

    # Set to eval mode
    model.eval()

    logger.info("Teacher model loaded successfully")
    return model


def teacher_generate(
    teacher_model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    num_steps: int = 10,
) -> _model.Actions:
    """Generate teacher distillation targets OUTSIDE of JIT.

    This must NOT be inside a jit-compiled function because sample_actions
    uses jax.lax.while_loop which can cause infinite compilation when nested.
    """
    return teacher_model.sample_actions(rng, observation, num_steps=num_steps)


@at.typecheck
def student_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    student_state: training_utils.TrainState,
    observation: _model.Observation,
    teacher_actions: _model.Actions,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Student training step (JIT-compiled).

    Only the student forward/backward pass is compiled. Teacher targets are
    pre-computed and passed in as arguments.

    Args:
        config: Training configuration.
        rng: Random key.
        student_state: Current student training state.
        observation: Preprocessed observation.
        teacher_actions: Pre-computed teacher targets.

    Returns:
        Updated student state and metrics dict.
    """
    student_model = nnx.merge(student_state.model_def, student_state.params)
    student_model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        teacher_actions: _model.Actions,
    ):
        return model.compute_distillation_loss(rng, observation, teacher_actions)

    # Compute gradients only for trainable params (action projection layers)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
        student_model, rng, observation, teacher_actions
    )

    # Update only trainable parameters
    params = student_state.params.filter(config.trainable_filter)
    updates, new_opt_state = student_state.tx.update(grads, student_state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update model in place and extract new full state
    nnx.update(student_model, new_params)
    new_params = nnx.state(student_model)

    new_state = dataclasses.replace(
        student_state,
        step=student_state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )

    # EMA update
    if student_state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: student_state.ema_decay * old + (1 - student_state.ema_decay) * new,
                student_state.ema_params,
                new_params,
            ),
        )

    # Compute gradient norm for monitoring
    grad_norm = optax.global_norm(grads)

    info = {
        "distill_loss": loss,
        "grad_norm": grad_norm,
    }
    return new_state, info


def init_student_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    teacher_checkpoint: str | None = None,
    *,
    resume: bool = False,
) -> tuple[training_utils.TrainState, Any]:
    """Initialize the student training state.

    Optionally loads teacher weights as initialization.

    Args:
        config: Training configuration.
        init_rng: Random key for initialization.
        mesh: JAX mesh for device placement.
        teacher_checkpoint: Optional teacher checkpoint to initialize from.
        resume: Whether to resume from existing checkpoint.

    Returns:
        Initialized student state and its sharding spec.
    """
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(
        rng: at.KeyArrayLike,
        partial_params: at.Params | None = None,
    ) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        # Merge partial params (from teacher checkpoint) into model
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)

        # Convert frozen params to bfloat16 to save memory
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    # Load teacher weights as initialization if provided
    if teacher_checkpoint is not None:
        partial_params = _model.restore_params(
            teacher_checkpoint, dtype=jnp.bfloat16
        )
    else:
        partial_params = None

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def run_distillation(config: _config.TrainConfig):
    """Main distillation training loop.

    Args:
        config: Distillation training configuration. Must have a `teacher_checkpoint` attribute.
    """
    logger.info(f"Starting distillation training on {jax.devices()}")

    # Initialize mesh for FSDP
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Setup checkpointing
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    # Setup wandb
    if config.wandb_enabled:
        import openpi.scripts.train as train_script
        train_script.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Data loader
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logger.info(f"Data loader initialized")

    # Random keys
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # Load teacher model (frozen)
    teacher_checkpoint = getattr(config, "teacher_checkpoint", "gs://openpi-assets/checkpoints/pi05_base/params")
    teacher_model = init_teacher_model(config, teacher_checkpoint, mesh)

    # Initialize student state (optionally initialized from teacher weights)
    student_state, student_sharding = init_student_state(
        config, init_rng, mesh, teacher_checkpoint=teacher_checkpoint, resume=resuming
    )
    jax.block_until_ready(student_state)
    logger.info(f"Student state initialized")

    if resuming:
        student_state = _checkpoints.restore_state(checkpoint_manager, student_state, data_loader)

    # Compile ONLY the student training step (teacher generation is outside JIT)
    pstudent_step = jax.jit(
        functools.partial(student_train_step, config),
        in_shardings=(replicated_sharding, student_sharding, data_sharding, data_sharding),
        out_shardings=(student_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # Training loop
    start_step = int(student_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        observation, _ = batch

        # Step 1: Teacher generates distillation target (OUTSIDE of JIT)
        teacher_rng = jax.random.fold_in(train_rng, step)
        teacher_actions = teacher_generate(teacher_model, teacher_rng, observation, num_steps=10)

        # Step 2: Student training (JIT-compiled)
        student_rng = jax.random.fold_in(train_rng, step + 1000000)  # Different subkey
        with sharding.set_mesh(mesh):
            student_state, info = pstudent_step(student_rng, student_state, observation, teacher_actions)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if config.wandb_enabled:
                wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, student_state, data_loader, step)
            logger.info(f"Checkpoint saved at step {step}")

    logger.info("Distillation training complete!")
    checkpoint_manager.wait_until_finished()
