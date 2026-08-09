#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .._training import TensorBoardLogger


def term_label(term: Any, /) -> str:
    """Return the stable display label for one scalar objective term."""
    return term.label or type(term).__name__


def metric_suffix(metrics: dict[str, Any], /) -> str:
    """Format scalar term diagnostics for one console log line."""
    if not metrics:
        return ""
    parts = []
    for name, value in metrics.items():
        value_f = float(jnp.asarray(value, dtype=float).reshape(()))
        parts.append(f"{name}={value_f:.6e}")
    return " " + " ".join(parts)


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


def _write_term_scalars(
    writer: TensorBoardLogger,
    *,
    step: int,
    namespace: str,
    term_names: tuple[str, ...],
    terms: Any,
    data_metrics: tuple[dict[str, Any], ...],
) -> None:
    terms_arr = jnp.asarray(terms, dtype=float)
    for index, (name, value) in enumerate(
        zip(term_names, list(map(float, terms_arr)), strict=True)
    ):
        prefix = f"{namespace}/{_term_tag(index, name)}"
        writer.scalar(f"{prefix}/value", value, step)
        for metric_name, metric_value in data_metrics[index].items():
            writer.scalar(f"{prefix}/{metric_name}", metric_value, step)


def _write_model_loss_scalars(
    writer: TensorBoardLogger,
    *,
    step: int,
    model_loss_names: tuple[str, ...],
    terms: Any,
) -> None:
    terms_arr = jnp.asarray(terms, dtype=float)
    for index, (name, value) in enumerate(
        zip(model_loss_names, list(map(float, terms_arr)), strict=True)
    ):
        writer.scalar(f"train/{_model_loss_tag(index, name)}/loss", value, step)


def log_tensorboard_scalars(
    writer: TensorBoardLogger,
    *,
    step: int,
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
) -> None:
    """Emit the backend-independent scalar schema for one training step."""
    writer.scalar("train/loss", loss, step)
    writer.scalar("train/best_loss", best_loss, step)
    if evaluation_loss is not None:
        writer.scalar("eval/loss", evaluation_loss, step)
    writer.scalar("train/iter_time_s", iter_time_s, step)

    if not log_terms:
        return

    _write_term_scalars(
        writer,
        step=step,
        namespace="train",
        term_names=train_term_names,
        terms=train_terms,
        data_metrics=train_data_metrics,
    )
    _write_model_loss_scalars(
        writer,
        step=step,
        model_loss_names=train_model_loss_names,
        terms=train_model_loss_terms,
    )
    _write_term_scalars(
        writer,
        step=step,
        namespace="eval",
        term_names=evaluation_term_names,
        terms=eval_terms,
        data_metrics=eval_data_metrics,
    )


__all__ = [
    "best_display_value",
    "log_tensorboard_scalars",
    "metric_suffix",
    "term_label",
]
