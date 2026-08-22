#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from ._operators import (
    _materialize_by_basis,
    AbstractLinearOperator,
    DenseLinearOperator,
)
from ._properties import OperatorCapabilities, OperatorProperties
from ._spaces import ArraySpace


class EmpiricalGramLinearOperator(AbstractLinearOperator):
    """Weighted feature Gram action with optional sample centering and damping."""

    features: AbstractLinearOperator
    normalized_weights: Array
    weight_ess: Array
    active_samples: Array
    damping: float = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    rank_upper_bound: int = eqx.field(static=True)

    def __init__(
        self,
        features: AbstractLinearOperator,
        weights: ArrayLike,
        /,
        *,
        centered: bool = True,
        damping: float = 0.0,
        operator_id: str | None = None,
    ):
        if not isinstance(features, AbstractLinearOperator):
            raise TypeError("features must be an AbstractLinearOperator.")
        if features.batch_shape:
            raise ValueError("Empirical feature operators must be unbatched.")
        if not isinstance(features.target, ArraySpace):
            raise TypeError("Empirical feature targets must be an ArraySpace.")
        if not features.target.shape:
            raise ValueError("Empirical feature targets need a leading sample axis.")
        sample_count = int(features.target.shape[0])
        if sample_count < 1:
            raise ValueError("Empirical feature targets require at least one sample.")
        values = jnp.asarray(weights)
        if values.shape != (sample_count,):
            raise ValueError(
                f"weights must have shape ({sample_count},); got {values.shape}."
            )
        if jnp.iscomplexobj(values):
            raise TypeError("Empirical weights must be real-valued.")
        invalid_weights = (
            jnp.any(~jnp.isfinite(values))
            | jnp.any(values < 0.0)
            | (jnp.sum(values) <= 0.0)
        )
        if not isinstance(values, jax.core.Tracer) and bool(invalid_weights):
            raise ValueError(
                "Empirical weights must be finite, non-negative, and have positive mass."
            )
        active_bound = (
            sample_count
            if isinstance(values, jax.core.Tracer)
            else int(jnp.count_nonzero(values > 0.0))
        )
        values = eqx.error_if(
            values,
            invalid_weights,
            "Empirical weights must be finite, non-negative, and have positive mass.",
        )
        damping_ = float(damping)
        if not isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        normalized = values.astype(float) / jnp.sum(values.astype(float))
        event_size = prod(features.target.shape[1:]) if features.target.shape[1:] else 1
        rank_bound = active_bound * event_size
        if centered:
            rank_bound = max(active_bound - 1, 0) * event_size
        rank_bound = min(features.source.size, rank_bound)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "empirical-gram",
                    "features": features.operator_id,
                    "weight_shape": list(values.shape),
                    "centered": bool(centered),
                    "damping": damping_,
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        positive = damping_ > 0.0
        features_ = features
        if isinstance(features, DenseLinearOperator):
            sample_mask = (normalized > 0.0).reshape(
                (sample_count,) + (1,) * (len(features.target.shape) - 1)
            )
            target_mask = jnp.broadcast_to(sample_mask, features.target.shape).reshape(
                (-1, 1)
            )
            safe_matrix = jnp.where(
                target_mask,
                features.matrix,
                jnp.zeros((), dtype=features.matrix.dtype),
            )
            features_ = DenseLinearOperator(
                safe_matrix,
                source=features.source,
                target=features.target,
                properties=features.properties,
                operator_id=f"{features.operator_id}:sample-masked",
            )
        self.features = features_
        self.normalized_weights = normalized
        self.weight_ess = jnp.reciprocal(jnp.sum(normalized**2))
        self.active_samples = jnp.sum(normalized > 0.0, dtype=jnp.int32)
        self.damping = damping_
        self.centered = bool(centered)
        self.rank_upper_bound = rank_bound
        self.source = features.source
        self.target = features.source
        self.properties = OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            positive_definite=positive,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
                **({"positive_definite": "construction"} if positive else {}),
            },
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = identifier

    def _sample_weights(self, output_ndim: int, /) -> Array:
        return self.normalized_weights.reshape(
            self.normalized_weights.shape + (1,) * output_ndim
        )

    def _center(self, values: Array, /) -> Array:
        weights = self._sample_weights(values.ndim - 1)
        active = weights > 0.0
        safe_values = jnp.where(active, values, jnp.zeros((), dtype=values.dtype))
        if not self.centered:
            return safe_values
        mean = jnp.sum(weights * safe_values, axis=0, keepdims=True)
        return jnp.where(active, safe_values - mean, jnp.zeros((), dtype=values.dtype))

    def _center_adjoint(self, covector: Array, /) -> Array:
        weights = self._sample_weights(covector.ndim - 1)
        active = weights > 0.0
        safe_covector = jnp.where(active, covector, jnp.zeros((), dtype=covector.dtype))
        if not self.centered:
            return safe_covector
        total = jnp.sum(safe_covector, axis=0, keepdims=True)
        return safe_covector - weights * total

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        tangent = self.source.validate(vector)
        features = jnp.asarray(self.features.mv(tangent))
        centered = self._center(features)
        weights = self._sample_weights(centered.ndim - 1)
        covector = self._center_adjoint(weights * centered)
        action = self.features.adjoint_mv(covector)
        if self.damping == 0.0:
            return self.source.validate(action)
        return self.source.validate(
            jax.tree_util.tree_map(
                lambda value, direction: value + self.damping * direction,
                action,
                tangent,
            )
        )

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        value = self.target.validate(vector)
        transpose = jax.linear_transpose(self.mv, self.source.zeros())
        return self.source.validate(transpose(value)[0])

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return self.mv(vector)

    def _materialize(self, /) -> Array:
        return _materialize_by_basis(self)


__all__ = ["EmpiricalGramLinearOperator"]
