#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from ..domain._model_function import _ConcatenatedModelCallable
from ..nn.models.core._base import _AbstractBaseModel
from ..nn.models.core._loss import (
    _has_custom_model_loss,
    _model_loss_labels,
    _model_loss_values,
    ModelWithLoss,
)


def function_model_loss_labels(functions: Any, /) -> tuple[str, ...]:
    """Return static labels for all model objective terms in a function tree."""
    labels: list[str] = []
    seen_models: set[int] = set()
    for model in _iter_model_roots(functions, seen_nodes=set()):
        labels.extend(_model_loss_labels(model, seen=seen_models))
    return tuple(labels)


def function_model_loss_values(
    functions: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    iter_: Array | None = None,
) -> tuple[Array, ...]:
    """Evaluate all model objective terms in a function tree."""
    values: list[Array] = []
    seen_models: set[int] = set()
    counter = [0]
    for model in _iter_model_roots(functions, seen_nodes=set()):
        values.extend(
            _model_loss_values(
                model,
                key=key,
                iter_=iter_,
                seen=seen_models,
                counter=counter,
            )
        )
    return tuple(jnp.asarray(v, dtype=float).reshape(()) for v in values)


def _iter_model_roots(value: Any, /, *, seen_nodes: set[int]):
    node_id = id(value)
    if node_id in seen_nodes:
        return
    seen_nodes.add(node_id)

    if isinstance(value, _ConcatenatedModelCallable):
        yield value.raw_model
        return

    if isinstance(value, (ModelWithLoss, _AbstractBaseModel)) or _has_custom_model_loss(
        value
    ):
        yield value
        return

    if value is None or isinstance(value, (str, bytes, int, float, complex, bool)):
        return

    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            if field.name == "_strict_initialized":
                continue
            yield from _iter_model_roots(getattr(value, field.name), seen_nodes=seen_nodes)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_model_roots(item, seen_nodes=seen_nodes)
        return

    if isinstance(value, tuple | list):
        for item in value:
            yield from _iter_model_roots(item, seen_nodes=seen_nodes)


__all__ = [
    "function_model_loss_labels",
    "function_model_loss_values",
]
