#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from .._training import TensorBoardLogger
from ..logging import emit


def term_label(term: Any, /) -> str:
    """Return the stable display label for one scalar objective term."""
    return term.label or type(term).__name__




def best_display_value(
    best_value: int | float | None,
    loss: float,
    /,
    *,
    keep_best: bool,
) -> float:
    """Choose the scalar displayed as the run's current best value."""
    if keep_best and best_value is not None:
        return float(best_value)
    return loss


def _clean_tag_part(value: str, /) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)
    ).strip("_")
    return cleaned or "term"


def _term_tag(index: int, name: str, /) -> str:
    return f"terms/{index:03d}_{_clean_tag_part(name)}"


def _model_loss_tag(index: int, name: str, /) -> str:
    return f"model_losses/{index:03d}_{_clean_tag_part(name)}"


def _as_scalar(value: Any, /) -> float:
    return float(jnp.asarray(value, dtype=float).reshape(()))


def training_scalars(
    *,
    loss: Any,
    best_loss: float,
    evaluation_loss: Any | None,
    iter_time_s: float,
    train_term_names: tuple[str, ...],
    train_terms: Any,
    train_data_metrics: tuple[dict[str, Any], ...],
    train_model_loss_names: tuple[str, ...],
    train_model_loss_terms: Any,
    evaluation_term_names: tuple[str, ...],
    eval_terms: Any,
    eval_data_metrics: tuple[dict[str, Any], ...],
    log_terms: bool,
    optimizer_metrics: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Materialize the backend-independent scalar schema exactly once."""

    scalars = {
        "train/best_loss": float(best_loss),
        "train/iter_time_s": float(iter_time_s),
        "train/loss": _as_scalar(loss),
    }
    if evaluation_loss is not None:
        scalars["eval/loss"] = _as_scalar(evaluation_loss)
    if optimizer_metrics is not None:
        for name, value in optimizer_metrics.items():
            scalars[str(name)] = _as_scalar(value)
    if not log_terms:
        return scalars

    train_values = list(map(float, jnp.asarray(train_terms, dtype=float)))
    for index, (name, value) in enumerate(
        zip(train_term_names, train_values, strict=True)
    ):
        prefix = f"train/{_term_tag(index, name)}"
        scalars[f"{prefix}/value"] = value
        for metric_name, metric_value in train_data_metrics[index].items():
            scalars[f"{prefix}/{metric_name}"] = _as_scalar(metric_value)

    model_loss_values = list(map(float, jnp.asarray(train_model_loss_terms, dtype=float)))
    for index, (name, value) in enumerate(
        zip(train_model_loss_names, model_loss_values, strict=True)
    ):
        scalars[f"train/{_model_loss_tag(index, name)}/loss"] = value

    evaluation_values = list(map(float, jnp.asarray(eval_terms, dtype=float)))
    for index, (name, value) in enumerate(
        zip(evaluation_term_names, evaluation_values, strict=True)
    ):
        prefix = f"eval/{_term_tag(index, name)}"
        scalars[f"{prefix}/value"] = value
        for metric_name, metric_value in eval_data_metrics[index].items():
            scalars[f"{prefix}/{metric_name}"] = _as_scalar(metric_value)
    return scalars


def emit_training_scalars(
    scalars: Mapping[str, float],
    /,
    *,
    backend: str,
    step: int,
    total_steps: int,
) -> None:
    """Emit one structured training-step event."""

    emit(
        "INFO",
        "training.step.completed",
        "Training step completed",
        backend=backend,
        metrics=tuple(
            {"name": name, "value": value} for name, value in scalars.items()
        ),
        step=int(step),
        total_steps=int(total_steps),
    )


def write_tensorboard_scalars(
    writer: TensorBoardLogger,
    scalars: Mapping[str, float],
    /,
    *,
    step: int,
) -> None:
    """Write one already-materialized scalar report to TensorBoard."""

    for name, value in scalars.items():
        writer.scalar(name, value, step)


__all__ = [
    "best_display_value",
    "emit_training_scalars",
    "term_label",
    "training_scalars",
    "write_tensorboard_scalars",
]
