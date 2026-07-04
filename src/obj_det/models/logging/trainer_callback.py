from __future__ import annotations

from obj_det.models.logging.base import BaseExperimentLogger
from obj_det.models.logging.metrics import flatten_prefixed_scalar_mapping


def make_transformers_logging_callback(trainer_callback_cls, logger: BaseExperimentLogger, prefix: str):
    class LoggingCallback(trainer_callback_cls):
        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics, step = flatten_prefixed_scalar_mapping(prefix, logs or {})
            if metrics:
                logger.log_metrics(metrics, step=step if step is not None else state.global_step)

    return LoggingCallback()
