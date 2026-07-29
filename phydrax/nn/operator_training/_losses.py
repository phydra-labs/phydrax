#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array, Key

from ..models.core._operator import (
    OperatorBatch,
    OperatorPrediction,
    OperatorTargetBatch,
)
from ..models.core._operator_metrics import operator_l2_loss


@dataclass(frozen=True)
class OperatorLossContext:
    """Paired execution-space and physical-space views for one loss evaluation."""

    execution_prediction: OperatorPrediction
    execution_batch: OperatorBatch
    execution_targets: OperatorTargetBatch
    physical_prediction: OperatorPrediction
    physical_batch: OperatorBatch
    physical_targets: OperatorTargetBatch
    normalization: Any = None
    task: Any = None

    def view(
        self,
        space: Literal["execution", "physical"],
        /,
    ) -> tuple[OperatorPrediction, OperatorBatch, OperatorTargetBatch]:
        if space == "execution":
            return (
                self.execution_prediction,
                self.execution_batch,
                self.execution_targets,
            )
        if space == "physical":
            return (
                self.physical_prediction,
                self.physical_batch,
                self.physical_targets,
            )
        raise ValueError("Loss space must be 'execution' or 'physical'.")


class AbstractOperatorLossTerm(ABC):
    """One named scalar objective evaluated against a rich operator batch."""

    name: str
    weight: float

    @abstractmethod
    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Stable identity included in exact-resume compatibility checks."""
        raise NotImplementedError


@dataclass(frozen=True)
class OperatorLossTerm(AbstractOperatorLossTerm):
    """Adapt a custom scalar callable with an explicit coordinate/value space."""

    name: str
    fn: Callable[..., Array]
    weight: float = 1.0
    identity: str | None = None
    space: Literal["execution", "physical"] = "physical"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Operator loss term names must be non-empty.")
        if not callable(self.fn):
            raise TypeError("Operator loss term fn must be callable.")
        if not jnp.isfinite(self.weight):
            raise ValueError("Operator loss term weight must be finite.")
        if self.space not in ("execution", "physical"):
            raise ValueError("Loss space must be 'execution' or 'physical'.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del prediction, batch, targets
        selected_prediction, selected_batch, selected_targets = context.view(self.space)
        value = self.fn(
            selected_prediction,
            selected_batch,
            selected_targets,
            model=model,
            key=key,
            step=step,
            training=training,
            context=context,
        )
        return jnp.asarray(self.weight, dtype=jnp.asarray(value).dtype) * value

    @property
    def fingerprint(self) -> str:
        identity = self.identity
        if identity is None:
            if inspect.isfunction(self.fn) or inspect.ismethod(self.fn):
                identity = f"{self.fn.__module__}.{self.fn.__qualname__}"
            else:
                function_type = type(self.fn)
                identity = f"{function_type.__module__}.{function_type.__qualname__}"
        payload = json.dumps(
            {
                "kind": "custom",
                "name": self.name,
                "weight": self.weight,
                "identity": identity,
                "space": self.space,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupervisedOperatorLoss(AbstractOperatorLossTerm):
    """Named supervised L² objective in physical or execution space."""

    name: str = "supervised_l2"
    weight: float = 1.0
    prediction_field: str | None = None
    target_field: str | None = None
    relative: bool = False
    squared: bool = True
    reduction: Literal["none", "mean", "sum"] = "mean"
    epsilon: float = 1e-12
    space: Literal["execution", "physical"] = "physical"

    def __post_init__(self):
        if not self.name:
            raise ValueError("Operator loss term names must be non-empty.")
        if not jnp.isfinite(self.weight):
            raise ValueError("Operator loss term weight must be finite.")
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        if self.reduction == "none":
            raise ValueError("Training loss terms must reduce to a scalar.")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if self.space not in ("execution", "physical"):
            raise ValueError("Loss space must be 'execution' or 'physical'.")

    def __call__(
        self,
        model: Any,
        prediction: OperatorPrediction,
        batch: OperatorBatch,
        targets: OperatorTargetBatch,
        /,
        *,
        key: Key[Array, ""],
        step: Array,
        training: bool,
        context: OperatorLossContext,
    ) -> Array:
        del model, prediction, batch, targets, key, step, training
        selected_prediction, selected_batch, selected_targets = context.view(self.space)
        prediction_name = self.prediction_field
        if prediction_name is None:
            if len(selected_prediction.fields) != 1:
                raise ValueError(
                    "prediction_field is required for multi-output predictions."
                )
            prediction_name = next(iter(selected_prediction.fields))
        target_name = self.target_field
        if target_name is None:
            if prediction_name in selected_targets.fields:
                target_name = prediction_name
            elif len(selected_targets.fields) == 1:
                target_name = next(iter(selected_targets.fields))
            else:
                raise ValueError("target_field is required for multi-target batches.")
        predicted = selected_prediction.field(prediction_name)
        truth = selected_targets.field(target_name)
        if predicted.query_name != truth.query_name:
            raise ValueError(
                f"Prediction {prediction_name!r} and target {target_name!r} "
                "must use the same query."
            )
        value = operator_l2_loss(
            predicted.values,
            truth.values,
            selected_batch.query(predicted.query_name),
            relative=self.relative,
            squared=self.squared,
            reduction=self.reduction,
            eps=self.epsilon,
        )
        return jnp.asarray(self.weight, dtype=jnp.asarray(value).dtype) * value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "supervised_l2",
                "name": self.name,
                "weight": self.weight,
                "prediction_field": self.prediction_field,
                "target_field": self.target_field,
                "relative": self.relative,
                "squared": self.squared,
                "reduction": self.reduction,
                "epsilon": self.epsilon,
                "space": self.space,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AbstractOperatorLossTerm",
    "OperatorLossContext",
    "OperatorLossTerm",
    "SupervisedOperatorLoss",
]
