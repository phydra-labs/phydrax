#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...sparse import RowRelation


IMPLICIT_OBSERVED_ZERO = 0
IMPLICIT_STRUCTURAL_ABSENCE = 1
IMPLICIT_MISSING = 2


def _shape2(name: str, value: Array, /) -> tuple[int, int]:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, features).")
    samples, features = (int(size) for size in value.shape)
    if samples < 1 or features < 1:
        raise ValueError(f"{name} must contain at least one sample and feature.")
    return samples, features


def _mask(name: str, value: ArrayLike | None, shape: tuple[int, int], /) -> Array:
    if value is None:
        return jnp.zeros(shape, dtype=bool)
    result = jnp.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}.")
    return result


def _implicit_state(value: int, /) -> int:
    state = int(value)
    if state not in (
        IMPLICIT_OBSERVED_ZERO,
        IMPLICIT_STRUCTURAL_ABSENCE,
        IMPLICIT_MISSING,
    ):
        raise ValueError("implicit_state must be an IMPLICIT_* constant.")
    return state


def _sparse_inputs(
    feature_indices: ArrayLike,
    values: ArrayLike,
    /,
    *,
    num_features: int,
    route_valid: ArrayLike | None,
    missing: ArrayLike | None,
    structural_absence: ArrayLike | None,
) -> tuple[Array, Array, Array, Array, Array, int, int]:
    indices = jnp.asarray(feature_indices)
    payload = jnp.asarray(values)
    if indices.ndim != 2 or payload.shape != indices.shape:
        raise ValueError(
            "fixed-sparse feature_indices and values must have the same "
            "(samples, width) shape."
        )
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError("feature_indices must have an integer dtype.")
    shape = (int(indices.shape[0]), int(indices.shape[1]))
    samples, width = shape
    features = int(num_features)
    if samples < 1 or width < 1 or features < 1:
        raise ValueError("fixed-sparse assay dimensions must be positive.")
    valid = (
        jnp.ones(shape, dtype=bool)
        if route_valid is None
        else jnp.asarray(route_valid, dtype=bool)
    )
    if valid.shape != indices.shape:
        raise ValueError(
            f"route_valid must have shape {indices.shape}; got {valid.shape}."
        )
    route_missing = _mask("missing", missing, shape)
    route_structural = _mask("structural_absence", structural_absence, shape)
    route_missing = route_missing & valid
    route_structural = route_structural & valid
    if bool(jnp.any(route_missing & route_structural)):
        raise ValueError("missing and structural_absence routes must be disjoint.")
    if bool(jnp.any(valid & ((indices < 0) | (indices >= features)))):
        raise ValueError("A valid feature index lies outside the assay feature space.")
    safe_indices = jnp.where(valid, indices, features).astype(jnp.int32)
    ordered = jnp.sort(safe_indices, axis=1)
    if width > 1:
        duplicate = (ordered[:, 1:] < features) & (ordered[:, 1:] == ordered[:, :-1])
        if bool(jnp.any(duplicate)):
            raise ValueError("Fixed-sparse routes must be unique within each sample.")
    return (
        indices.astype(jnp.int32),
        payload,
        valid,
        route_missing,
        route_structural,
        samples,
        features,
    )


def _dense_masks(
    values: Array,
    missing: ArrayLike | None,
    structural_absence: ArrayLike | None,
    /,
) -> tuple[Array, Array, Array]:
    shape = _shape2("values", values)
    structural = _mask("structural_absence", structural_absence, shape)
    explicit_missing = _mask("missing", missing, shape)
    inferred_missing = ~jnp.isfinite(values)
    missing_mask = explicit_missing | inferred_missing
    if bool(jnp.any(missing_mask & structural)):
        raise ValueError("missing and structural_absence cells must be disjoint.")
    observed = ~(missing_mask | structural)
    return observed, structural, missing_mask


def _sparse_dense(
    values: Array,
    indices: Array,
    valid: Array,
    route_missing: Array,
    route_structural: Array,
    /,
    *,
    num_features: int,
    implicit_state: int,
) -> tuple[Array, Array, Array, Array]:
    samples, width = values.shape
    rows = jnp.broadcast_to(
        jnp.arange(samples, dtype=jnp.int32)[:, None], (samples, width)
    )
    safe_indices = jnp.where(valid, indices, 0)
    present_count = (
        jnp.zeros((samples, num_features), dtype=jnp.int32)
        .at[rows, safe_indices]
        .add(valid.astype(jnp.int32))
    )
    missing_count = (
        jnp.zeros((samples, num_features), dtype=jnp.int32)
        .at[rows, safe_indices]
        .add((valid & route_missing).astype(jnp.int32))
    )
    structural_count = (
        jnp.zeros((samples, num_features), dtype=jnp.int32)
        .at[rows, safe_indices]
        .add((valid & route_structural).astype(jnp.int32))
    )
    dense = (
        jnp.zeros((samples, num_features), dtype=values.dtype)
        .at[rows, safe_indices]
        .add(jnp.where(valid & ~route_missing & ~route_structural, values, 0))
    )
    present = present_count > 0
    route_missing_dense = missing_count > 0
    route_structural_dense = structural_count > 0
    route_observed_dense = present & ~route_missing_dense & ~route_structural_dense
    observed = jnp.where(
        present, route_observed_dense, implicit_state == IMPLICIT_OBSERVED_ZERO
    )
    structural = jnp.where(
        present,
        route_structural_dense,
        implicit_state == IMPLICIT_STRUCTURAL_ABSENCE,
    )
    missing = jnp.where(present, route_missing_dense, implicit_state == IMPLICIT_MISSING)
    return dense, observed, structural, missing


class CountAssay(StrictModule):
    """Samples-by-features counts with explicit measurement-state semantics.

    Dense assays store all cells. Fixed-sparse assays store a bounded number of
    routes per sample; unlisted cells have the declared ``implicit_state``.
    Structural absence and missingness are never interpreted as biological zero.
    """

    values: Array
    observed: Array
    structural_absence: Array
    missing: Array
    relation: RowRelation | None
    sparse: bool = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    implicit_state: int = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        missing: ArrayLike | None = None,
        structural_absence: ArrayLike | None = None,
    ):
        counts = jnp.asarray(values)
        samples, features = _shape2("values", counts)
        observed, structural, missing_mask = _dense_masks(
            counts, missing, structural_absence
        )
        if bool(jnp.any(observed & ((counts < 0) | (counts != jnp.floor(counts))))):
            raise ValueError("Observed counts must be finite nonnegative integers.")
        self.values = jnp.where(observed, counts, 0)
        self.observed = observed
        self.structural_absence = structural
        self.missing = missing_mask
        self.relation = None
        self.sparse = False
        self.num_samples = samples
        self.num_features = features
        self.implicit_state = IMPLICIT_OBSERVED_ZERO

    @classmethod
    def from_fixed_sparse(
        cls,
        feature_indices: ArrayLike,
        values: ArrayLike,
        /,
        *,
        num_features: int,
        route_valid: ArrayLike | None = None,
        missing: ArrayLike | None = None,
        structural_absence: ArrayLike | None = None,
        implicit_state: int = IMPLICIT_OBSERVED_ZERO,
    ) -> "CountAssay":
        (
            indices,
            counts,
            valid,
            route_missing,
            route_structural,
            samples,
            features,
        ) = _sparse_inputs(
            feature_indices,
            values,
            num_features=num_features,
            route_valid=route_valid,
            missing=missing,
            structural_absence=structural_absence,
        )
        observed_routes = valid & ~route_missing & ~route_structural
        if bool(
            jnp.any(
                observed_routes
                & ((counts < 0) | ~jnp.isfinite(counts) | (counts != jnp.floor(counts)))
            )
        ):
            raise ValueError("Observed counts must be finite nonnegative integers.")
        self = object.__new__(cls)
        object.__setattr__(self, "values", jnp.where(observed_routes, counts, 0))
        object.__setattr__(self, "observed", observed_routes)
        object.__setattr__(self, "structural_absence", route_structural)
        object.__setattr__(self, "missing", route_missing)
        relation = RowRelation(
            indices,
            source_size=features,
            valid=valid,
            case_shape=(samples,),
        )
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "sparse", True)
        object.__setattr__(self, "num_samples", samples)
        object.__setattr__(self, "num_features", features)
        object.__setattr__(self, "implicit_state", _implicit_state(implicit_state))
        return self

    def dense_components(self) -> tuple[Array, Array, Array, Array]:
        if not self.sparse:
            return self.values, self.observed, self.structural_absence, self.missing
        relation = self.relation
        if relation is None:
            raise RuntimeError("A sparse count assay requires a row relation.")
        return _sparse_dense(
            self.values,
            relation.source_indices,
            relation.valid,
            self.missing,
            self.structural_absence,
            num_features=self.num_features,
            implicit_state=self.implicit_state,
        )

    @property
    def feature_indices(self) -> Array:
        if self.relation is None:
            return jnp.empty((self.num_samples, 0), dtype=jnp.int32)
        return self.relation.source_indices

    @property
    def route_valid(self) -> Array:
        if self.relation is None:
            return jnp.empty((self.num_samples, 0), dtype=bool)
        return self.relation.valid

    @property
    def dense_values(self) -> Array:
        return self.dense_components()[0]

    @property
    def observed_mask(self) -> Array:
        return self.dense_components()[1]

    @property
    def structural_absence_mask(self) -> Array:
        return self.dense_components()[2]

    @property
    def missing_mask(self) -> Array:
        return self.dense_components()[3]

    @property
    def zero_mask(self) -> Array:
        values, observed, _, _ = self.dense_components()
        return observed & (values == 0)


class ContinuousAssay(StrictModule):
    """Continuous measurements with dense or bounded fixed-sparse storage."""

    values: Array
    observed: Array
    structural_absence: Array
    missing: Array
    relation: RowRelation | None
    sparse: bool = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    implicit_state: int = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        missing: ArrayLike | None = None,
        structural_absence: ArrayLike | None = None,
    ):
        measurements = jnp.asarray(values)
        if not jnp.issubdtype(measurements.dtype, jnp.inexact):
            measurements = measurements.astype(float)
        samples, features = _shape2("values", measurements)
        observed, structural, missing_mask = _dense_masks(
            measurements, missing, structural_absence
        )
        self.values = jnp.where(observed, measurements, 0)
        self.observed = observed
        self.structural_absence = structural
        self.missing = missing_mask
        self.relation = None
        self.sparse = False
        self.num_samples = samples
        self.num_features = features
        self.implicit_state = IMPLICIT_OBSERVED_ZERO

    @classmethod
    def from_fixed_sparse(
        cls,
        feature_indices: ArrayLike,
        values: ArrayLike,
        /,
        *,
        num_features: int,
        route_valid: ArrayLike | None = None,
        missing: ArrayLike | None = None,
        structural_absence: ArrayLike | None = None,
        implicit_state: int = IMPLICIT_OBSERVED_ZERO,
    ) -> "ContinuousAssay":
        (
            indices,
            measurements,
            valid,
            route_missing,
            route_structural,
            samples,
            features,
        ) = _sparse_inputs(
            feature_indices,
            values,
            num_features=num_features,
            route_valid=route_valid,
            missing=missing,
            structural_absence=structural_absence,
        )
        if not jnp.issubdtype(measurements.dtype, jnp.inexact):
            measurements = measurements.astype(float)
        observed_routes = valid & ~route_missing & ~route_structural
        if bool(jnp.any(observed_routes & ~jnp.isfinite(measurements))):
            raise ValueError("Observed continuous measurements must be finite.")
        self = object.__new__(cls)
        object.__setattr__(self, "values", jnp.where(observed_routes, measurements, 0))
        object.__setattr__(self, "observed", observed_routes)
        object.__setattr__(self, "structural_absence", route_structural)
        object.__setattr__(self, "missing", route_missing)
        relation = RowRelation(
            indices,
            source_size=features,
            valid=valid,
            case_shape=(samples,),
        )
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "sparse", True)
        object.__setattr__(self, "num_samples", samples)
        object.__setattr__(self, "num_features", features)
        object.__setattr__(self, "implicit_state", _implicit_state(implicit_state))
        return self

    def dense_components(self) -> tuple[Array, Array, Array, Array]:
        if not self.sparse:
            return self.values, self.observed, self.structural_absence, self.missing
        relation = self.relation
        if relation is None:
            raise RuntimeError("A sparse continuous assay requires a row relation.")
        return _sparse_dense(
            self.values,
            relation.source_indices,
            relation.valid,
            self.missing,
            self.structural_absence,
            num_features=self.num_features,
            implicit_state=self.implicit_state,
        )

    @property
    def feature_indices(self) -> Array:
        if self.relation is None:
            return jnp.empty((self.num_samples, 0), dtype=jnp.int32)
        return self.relation.source_indices

    @property
    def route_valid(self) -> Array:
        if self.relation is None:
            return jnp.empty((self.num_samples, 0), dtype=bool)
        return self.relation.valid

    @property
    def dense_values(self) -> Array:
        return self.dense_components()[0]

    @property
    def observed_mask(self) -> Array:
        return self.dense_components()[1]

    @property
    def structural_absence_mask(self) -> Array:
        return self.dense_components()[2]

    @property
    def missing_mask(self) -> Array:
        return self.dense_components()[3]

    @property
    def zero_mask(self) -> Array:
        values, observed, _, _ = self.dense_components()
        return observed & (values == 0)


__all__ = [
    "ContinuousAssay",
    "CountAssay",
    "IMPLICIT_MISSING",
    "IMPLICIT_OBSERVED_ZERO",
    "IMPLICIT_STRUCTURAL_ABSENCE",
]
