#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..architectures import ChemicalConditionalAffineOperator
from ..data import OperatorBatch, OperatorPrediction, OperatorTargetBatch
from ..metrics import operator_l2_loss
from ._losses import (
    _weighted_case_reduction,
    AbstractOperatorLossTerm,
    OperatorLossContext,
)


Reduction = Literal["mean", "sum"]


def _loss_fingerprint(kind: str, payload: dict[str, Any], /) -> str:
    encoded = json.dumps(
        {"kind": kind, **payload},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _scaled_case_loss(
    prediction: Array,
    target: Array,
    scale: Array,
    query,
    context: OperatorLossContext,
    reduction: Reduction,
    /,
) -> Array:
    mask = query.mask_array(case_shape=context.physical_batch.case_shape)
    expanded_mask = mask.reshape(mask.shape + (1,) * (prediction.ndim - mask.ndim))
    predicted = jnp.where(expanded_mask, prediction / scale, 0.0)
    truth = jnp.where(expanded_mask, target / scale, 0.0)
    case_values = operator_l2_loss(
        predicted,
        truth,
        query,
        squared=True,
        reduction="none",
    )
    return _weighted_case_reduction(case_values, context, reduction)


def _driver_source(
    batch: OperatorBatch,
    name: str,
    query_name: str,
    driver_size: int,
    /,
) -> Array:
    samples = batch.input(name)
    query = batch.query(query_name)
    if samples.values is None or samples.sample_shape != query.sample_shape:
        raise ValueError(
            "Driver supervision source must share the conditional-affine query shape."
        )
    values = jnp.asarray(samples.values)
    expected = batch.case_shape + query.sample_shape + (driver_size,)
    if values.shape != expected:
        raise ValueError(f"Driver supervision source must have shape {expected}.")
    return values


@dataclass(frozen=True)
class ChemicalConditionalAffineDriverLoss(AbstractOperatorLossTerm):
    name: str = "conditional_affine_drivers"
    weight: float = 1.0
    driver_source: str = "driver_targets"
    reduction: Reduction = "mean"

    def __post_init__(self):
        if not self.name or not self.driver_source:
            raise ValueError("Driver loss names must be non-empty.")
        if not jnp.isfinite(self.weight):
            raise ValueError("Driver loss weight must be finite.")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("Driver loss reduction must be 'mean' or 'sum'.")

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
        del prediction, batch, targets, step, training
        if not isinstance(model, ChemicalConditionalAffineOperator):
            raise TypeError(
                "ChemicalConditionalAffineDriverLoss requires "
                "ChemicalConditionalAffineOperator."
            )
        _, physical_batch, _ = context.view("physical")
        truth = _driver_source(
            physical_batch,
            self.driver_source,
            model.query_name,
            model.chemistry.driver_size,
        )
        predicted = model.predict_drivers(physical_batch, key=key)
        if predicted.shape != truth.shape:
            raise ValueError("Predicted and supervised driver shapes must match exactly.")
        value = _scaled_case_loss(
            predicted,
            truth,
            model.scaling.driver_scale,
            physical_batch.query(model.query_name),
            context,
            self.reduction,
        )
        return jnp.asarray(self.weight, dtype=value.dtype) * value

    @property
    def fingerprint(self) -> str:
        return _loss_fingerprint(
            "chemical-conditional-affine-driver-loss",
            {
                "name": self.name,
                "weight": self.weight,
                "driver_source": self.driver_source,
                "reduction": self.reduction,
            },
        )


@dataclass(frozen=True)
class ChemicalConditionalAffineTeacherForcedLoss(AbstractOperatorLossTerm):
    name: str = "conditional_affine_teacher_forced"
    weight: float = 1.0
    state_target_field: str = "state"
    driver_source: str = "driver_targets"
    reduction: Reduction = "mean"

    def __post_init__(self):
        if not self.name or not self.state_target_field or not self.driver_source:
            raise ValueError("Teacher-forced loss names must be non-empty.")
        if not jnp.isfinite(self.weight):
            raise ValueError("Teacher-forced loss weight must be finite.")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("Teacher-forced loss reduction must be 'mean' or 'sum'.")

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
        del prediction, batch, targets, step, training
        if not isinstance(model, ChemicalConditionalAffineOperator):
            raise TypeError(
                "ChemicalConditionalAffineTeacherForcedLoss requires "
                "ChemicalConditionalAffineOperator."
            )
        _, physical_batch, physical_targets = context.view("physical")
        state_truth = physical_targets.field(self.state_target_field)
        driver_truth = _driver_source(
            physical_batch,
            self.driver_source,
            model.query_name,
            model.chemistry.driver_size,
        )
        if state_truth.query_name != model.query_name:
            raise ValueError("Teacher-forced state target must use the model query.")
        result = model.transition_with_drivers(
            physical_batch,
            driver_truth,
            key=key,
        )
        query = physical_batch.query(model.query_name)
        mask = query.mask_array(case_shape=physical_batch.case_shape)
        candidate = eqx.error_if(
            result.candidate_state,
            jnp.any(mask & ~result.successful),
            "Teacher-forced conditional-affine transition failed.",
        )
        if candidate.shape != state_truth.values.shape:
            raise ValueError("Predicted and target state shapes must match exactly.")
        value = _scaled_case_loss(
            candidate,
            state_truth.values,
            model.scaling.state_scale,
            query,
            context,
            self.reduction,
        )
        return jnp.asarray(self.weight, dtype=value.dtype) * value

    @property
    def fingerprint(self) -> str:
        return _loss_fingerprint(
            "chemical-conditional-affine-teacher-forced-loss",
            {
                "name": self.name,
                "weight": self.weight,
                "state_target_field": self.state_target_field,
                "driver_source": self.driver_source,
                "reduction": self.reduction,
            },
        )


__all__ = [
    "ChemicalConditionalAffineDriverLoss",
    "ChemicalConditionalAffineTeacherForcedLoss",
]
