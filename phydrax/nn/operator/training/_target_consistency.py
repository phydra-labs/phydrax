# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Key

from ..data import OperatorBatch, OperatorPrediction, OperatorTargetBatch
from ..metrics import operator_l2_loss
from ._losses import AbstractOperatorLossTerm, OperatorLossContext


@dataclass(frozen=True)
class TargetOperatorConsistencyLoss(AbstractOperatorLossTerm):
    """Measured physical-output consistency against a stopped target model."""

    field_name: str
    name: str = "target_operator_consistency"
    weight: float = 1.0

    def __post_init__(self):
        if not self.field_name or not self.name or not jnp.isfinite(self.weight):
            raise ValueError("Target operator consistency configuration is invalid.")

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
        del model, batch, targets, key, step, training
        target_prediction = context.target_physical_prediction
        if target_prediction is None:
            raise ValueError(
                "TargetOperatorConsistencyLoss requires target model prediction."
            )
        current = prediction.field(self.field_name)
        target = target_prediction.field(self.field_name)
        if current.query_name != target.query_name:
            raise ValueError("Target/current operator fields use different queries.")
        query = context.physical_batch.query(current.query_name)
        value = operator_l2_loss(
            current.values,
            jnp.asarray(target.values),
            query,
            squared=True,
            reduction="mean",
        )
        return jnp.asarray(self.weight, dtype=value.dtype) * value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "kind": "target_operator_consistency",
                "field": self.field_name,
                "weight": self.weight,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["TargetOperatorConsistencyLoss"]
