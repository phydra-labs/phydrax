#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..data import OperatorBatch, OperatorPrediction, OperatorTargetBatch
from ..distribution import AbstractProbabilisticOperatorModel
from ._losses import (
    _weighted_case_reduction,
    AbstractOperatorLossTerm,
    OperatorAccumulationKind,
    OperatorLossContext,
)


DistributionReduction = Literal["none", "mean", "sum"]


def operator_distribution_nll(
    model: AbstractProbabilisticOperatorModel,
    batch: OperatorBatch,
    target: Array,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    reduction: DistributionReduction = "mean",
) -> Array:
    """Evaluate a complete-field negative log likelihood in model coordinates."""
    if not isinstance(model, AbstractProbabilisticOperatorModel):
        raise TypeError("operator_distribution_nll requires a probabilistic operator.")
    return model.distribution(batch, key=key).negative_log_likelihood(
        jnp.asarray(target),
        reduction=reduction,
    )


@dataclass(frozen=True)
class OperatorDistributionNLL(AbstractOperatorLossTerm):
    """Supervised complete-field NLL evaluated in normalized execution space."""

    name: str = "distribution_nll"
    target_field: str = "output"
    weight: float = 1.0
    reduction: DistributionReduction = "mean"

    def __post_init__(self):
        if not self.name:
            raise ValueError("OperatorDistributionNLL name must be non-empty.")
        if not self.target_field:
            raise ValueError("OperatorDistributionNLL target_field must be non-empty.")
        if not isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError(
                "OperatorDistributionNLL weight must be finite and nonnegative."
            )
        if self.reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        if self.reduction == "none":
            raise ValueError(
                "OperatorDistributionNLL used as a training term requires a scalar reduction."
            )

    def __call__(
        self,
        model,
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
        if not isinstance(model, AbstractProbabilisticOperatorModel):
            raise TypeError(
                "OperatorDistributionNLL requires an AbstractProbabilisticOperatorModel."
            )
        target = context.execution_targets.field(self.target_field)
        execution_batch = context.execution_batch
        distribution = model.distribution(execution_batch, key=key)
        if target.query_name != execution_batch.single_query_name():
            raise ValueError(
                "OperatorDistributionNLL target must use the model's single query branch."
            )
        if target.spec.channels != distribution.output_spec.channels:
            raise ValueError(
                "OperatorDistributionNLL target channels do not match the distribution."
            )
        if self.reduction == "mean":
            case_values = distribution.negative_log_likelihood(
                target.values,
                reduction="none",
            )
            value = _weighted_case_reduction(case_values, context, "mean")
        else:
            value = distribution.negative_log_likelihood(
                target.values,
                reduction=self.reduction,
            )
        return jnp.asarray(self.weight, dtype=jnp.asarray(value).dtype) * value

    @property
    def accumulation_kind(self) -> OperatorAccumulationKind:
        return "case_mean" if self.reduction == "mean" else "single_batch"

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "type": type(self).__name__,
                "name": self.name,
                "target_field": self.target_field,
                "weight": float(self.weight),
                "reduction": self.reduction,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DistributionReduction",
    "OperatorDistributionNLL",
    "operator_distribution_nll",
]
