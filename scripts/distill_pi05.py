"""Training script for One-Step π0.5 distillation.

Distills a frozen teacher π0.5 model (10-step ODE) into a student model
that predicts clean actions in a single forward pass.

Usage:
    uv run scripts/distill_pi05.py pi05_libero_distill --exp-name=distill_run1
"""

import logging
import platform

import tyro

import openpi.training.config as _config
import openpi.training.distillation_trainer as _distill_trainer


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running distillation on: {platform.node()}")
    logging.info(f"Teacher checkpoint: {getattr(config, 'teacher_checkpoint', 'N/A')}")
    logging.info(f"Trainable filter: {config.trainable_filter}")
    logging.info(f"Freeze filter: {config.freeze_filter}")

    _distill_trainer.run_distillation(config)


if __name__ == "__main__":
    main(_config.cli())
