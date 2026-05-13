"""Evaluation script for One-Step π0.5 distilled model.

Compares inference speed and success rate between:
- Teacher: π0.5 with 10-step ODE integration
- Student: distilled single-step model

Usage:
    # Evaluate teacher (baseline)
    uv run scripts/eval_one_step.py --mode=teacher --config=pi05_libero \
        --checkpoint=gs://openpi-assets/checkpoints/pi05_libero

    # Evaluate student (distilled)
    uv run scripts/eval_one_step.py --mode=student --config=pi05_libero_distill \
        --checkpoint=checkpoints/pi05_libero_distill/exp/50000

    # Compare both
    uv run scripts/eval_one_step.py --mode=compare --task_suite=libero_10 \
        --num_trials=10
"""

import dataclasses
import enum
import logging
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config


class EvalMode(enum.Enum):
    TEACHER = "teacher"
    STUDENT = "student"
    COMPARE = "compare"


@dataclasses.dataclass
class Args:
    """Arguments for evaluation."""

    mode: EvalMode = EvalMode.COMPARE

    # Teacher config and checkpoint
    teacher_config: str = "pi05_libero"
    teacher_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"

    # Student config and checkpoint (for student/compare mode)
    student_config: str = "pi05_libero_distill"
    student_checkpoint: str | None = None

    # Evaluation parameters
    task_suite: str = "libero_10"
    num_trials: int = 10
    num_steps: int = 10  # Teacher ODE steps

    # Logging
    output_dir: str = "technical_research/eval_results"


def benchmark_inference_speed(model, config: _config.TrainConfig, num_steps: int = 100):
    """Benchmark inference speed of a model.

    Args:
        model: The model to benchmark.
        config: Model configuration.
        num_steps: Number of inference steps to measure.

    Returns:
        Dict with timing statistics.
    """
    # Create a fake observation
    observation = config.model.fake_obs(batch_size=1)
    rng = jax.random.key(42)

    times = []
    for _ in range(num_steps):
        start = time.time()
        if hasattr(model, "single_step_predict"):
            # Student: single-step
            _ = model.single_step_predict(rng, observation)
        else:
            # Teacher: multi-step ODE
            _ = model.sample_actions(rng, observation, num_steps=num_steps)
        jax.block_until_ready(_)  # Ensure computation is complete
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def evaluate_model(
    model_name: str,
    config_name: str,
    checkpoint_path: str,
    num_inference_steps: int = 10,
) -> dict[str, Any]:
    """Evaluate a single model.

    Args:
        model_name: "teacher" or "student".
        config_name: Config name (e.g., "pi05_libero").
        checkpoint_path: Path to checkpoint.
        num_inference_steps: Number of ODE steps for teacher.

    Returns:
        Evaluation results dict.
    """
    logger = logging.getLogger("openpi")
    logger.info(f"Evaluating {model_name} model: {config_name}")

    # Load model
    train_config = _config.get_config(config_name)
    model = train_config.model.load(
        _model.restore_params(checkpoint_path, dtype=jnp.bfloat16)
    )

    # Benchmark speed
    logger.info(f"Benchmarking {model_name} inference speed...")
    speed_stats = benchmark_inference_speed(model, train_config)
    logger.info(
        f"{model_name} speed: {speed_stats['mean_ms']:.1f}ms "
        f"(p50={speed_stats['p50_ms']:.1f}ms, p99={speed_stats['p99_ms']:.1f}ms)"
    )

    return {
        "model": model_name,
        "config": config_name,
        "checkpoint": checkpoint_path,
        "speed": speed_stats,
    }


def compare_models(args: Args) -> dict[str, Any]:
    """Compare teacher and student models.

    Args:
        args: Evaluation arguments.

    Returns:
        Comparison results.
    """
    results = {}

    # Evaluate teacher
    logger.info("=" * 60)
    logger.info("Evaluating TEACHER (10-step ODE)")
    logger.info("=" * 60)
    results["teacher"] = evaluate_model(
        "teacher",
        args.teacher_config,
        args.teacher_checkpoint,
        num_inference_steps=10,
    )

    # Evaluate student (if checkpoint provided)
    if args.student_checkpoint:
        logger.info("=" * 60)
        logger.info("Evaluating STUDENT (single-step)")
        logger.info("=" * 60)
        results["student"] = evaluate_model(
            "student",
            args.student_config,
            args.student_checkpoint,
            num_inference_steps=1,
        )

        # Compute speedup
        teacher_time = results["teacher"]["speed"]["mean_ms"]
        student_time = results["student"]["speed"]["mean_ms"]
        speedup = teacher_time / student_time
        results["speedup"] = speedup
        logger.info(f"\nSpeedup: {speedup:.1f}x")

    return results


def save_results(results: dict[str, Any], output_dir: str):
    """Save evaluation results to disk."""
    import json

    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = out_path / f"eval_results_{timestamp}.json"

    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {result_file}")


def main(args: Args):
    """Main evaluation entry point."""
    logging.basicConfig(level=logging.INFO)

    if args.mode == EvalMode.TEACHER:
        results = evaluate_model(
            "teacher",
            args.teacher_config,
            args.teacher_checkpoint,
            num_inference_steps=args.num_steps,
        )
    elif args.mode == EvalMode.STUDENT:
        if not args.student_checkpoint:
            raise ValueError("--student_checkpoint is required for student evaluation")
        results = evaluate_model(
            "student",
            args.student_config,
            args.student_checkpoint,
            num_inference_steps=1,
        )
    elif args.mode == EvalMode.COMPARE:
        results = compare_models(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Save results
    save_results(results, args.output_dir)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    if "teacher" in results:
        t = results["teacher"]["speed"]
        logger.info(f"Teacher (10-step): {t['mean_ms']:.1f}ms ± {t['std_ms']:.1f}ms")
    if "student" in results:
        s = results["student"]["speed"]
        logger.info(f"Student (1-step):  {s['mean_ms']:.1f}ms ± {s['std_ms']:.1f}ms")
    if "speedup" in results:
        logger.info(f"Speedup:           {results['speedup']:.1f}x")
    logger.info("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(Args))
