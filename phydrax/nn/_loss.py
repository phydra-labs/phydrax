#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._model import (
    AxisModelEvaluator,
    model_objective_labels as _model_objective_labels,
    model_objective_values as _model_objective_values,
    ModelBinding,
    ModelEvaluator,
    ModelObjectiveProvider,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._keys import EvalKey


class ModelLossTerm(StrictModule, NonTrainableState):
    """A fixed scalar penalty attached to a model."""

    penalty: Callable
    weight: Array
    label: str | None

    def __init__(
        self,
        penalty: Callable[..., Any],
        /,
        *,
        weight: Any = 1.0,
        label: str | None = None,
    ):
        if not callable(penalty):
            raise TypeError("Model loss penalty must be callable.")
        self.penalty = _ensure_special_kwonly_args(penalty)
        self.weight = jnp.asarray(weight, dtype=float)
        self.label = None if label is None else str(label)

    def __call__(
        self,
        model: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> Array:
        value = self.penalty(model, key=key, iter_=iter_)
        return self.weight * jnp.asarray(value, dtype=float).reshape(())


class ModelWithLoss(
    StrictModule,
    ModelEvaluator,
    AxisModelEvaluator,
    ModelObjectiveProvider,
):
    """Callable model wrapper carrying additional scalar objective terms."""

    model: Any
    loss_terms: tuple[ModelLossTerm, ...]
    loss_identity: int
    in_size: Any
    out_size: Any
    binding: ModelBinding

    def __init__(
        self,
        model: Any,
        /,
        *,
        loss_terms: Sequence[ModelLossTerm] = (),
        loss_identity: int | None = None,
    ):
        if not isinstance(model, ModelEvaluator):
            raise TypeError(
                "ModelWithLoss requires a model with an explicit input binding."
            )
        terms = tuple(loss_terms)
        bad = tuple(t for t in terms if not isinstance(t, ModelLossTerm))
        if bad:
            raise TypeError(
                "loss_terms must contain ModelLossTerm instances; got "
                f"{tuple(type(t).__name__ for t in bad)!r}."
            )
        self.model = model
        self.loss_terms = terms
        self.loss_identity = int(id(terms) if loss_identity is None else loss_identity)
        self.in_size = model.in_size
        self.out_size = model.out_size
        self.binding = model.input_binding()

    def __call__(
        self,
        x: Any,
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Array:
        return self.binding.call(
            self.model,
            x,
            key=key,
            iter_=iter_,
            kwargs=dict(kwargs),
        )

    def add_model_loss(
        self,
        penalty: Callable[..., Any],
        /,
        *,
        weight: Any = 1.0,
        label: str | None = None,
    ) -> "ModelWithLoss":
        term = ModelLossTerm(penalty, weight=weight, label=label)
        return ModelWithLoss(self.model, loss_terms=self.loss_terms + (term,))

    def input_binding(self) -> ModelBinding:
        return self.binding

    def model_objective_identity(self) -> int:
        return self.loss_identity

    def model_objective_children_first(self) -> bool:
        return True

    def local_model_objective_labels(self) -> tuple[str, ...]:
        return tuple(
            term.label
            if term.label is not None
            else f"{type(self.model).__name__}.model_loss_{index}"
            for index, term in enumerate(self.loss_terms)
        )

    def local_model_objective_values(
        self,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
    ) -> tuple[Array, ...]:
        return tuple(
            jnp.asarray(
                term(self.model, key=jr.fold_in(key, index), iter_=iter_),
                dtype=float,
            ).reshape(())
            for index, term in enumerate(self.loss_terms)
        )

    def __call_axis_batch__(
        self,
        batch: Any,
        deps: tuple[str, ...],
        /,
        *,
        key: EvalKey = DOC_KEY0,
        iter_: Array | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(self.model, AxisModelEvaluator):
            return self.model.__call_axis_batch__(
                batch, deps, key=key, iter_=iter_, **kwargs
            )
        raise TypeError("Wrapped model does not support axis-batch execution.")


def add_model_loss(
    model: Any,
    penalty: Callable[..., Any],
    /,
    *,
    weight: Any = 1.0,
    label: str | None = None,
) -> ModelWithLoss:
    """Return `model` wrapped with an additional scalar objective term.

    `penalty` is called as `penalty(model, key=..., iter_=...)` and must return a
    scalar JAX-compatible value.
    """
    term = ModelLossTerm(penalty, weight=weight, label=label)
    if isinstance(model, ModelWithLoss):
        return ModelWithLoss(model.model, loss_terms=model.loss_terms + (term,))
    return ModelWithLoss(model, loss_terms=(term,))


def model_loss_labels(model: Any, /) -> tuple[str, ...]:
    """Return static labels for all objective terms attached to `model`."""
    return _model_objective_labels(model)


def model_loss_values(
    model: Any,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    iter_: Array | None = None,
) -> tuple[Array, ...]:
    """Evaluate all scalar objective terms attached to `model`."""
    return _model_objective_values(model, key=key, iter_=iter_)


__all__ = [
    "ModelLossTerm",
    "ModelWithLoss",
    "add_model_loss",
    "model_loss_labels",
    "model_loss_values",
]
