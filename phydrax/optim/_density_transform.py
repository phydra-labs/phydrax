#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.spatial import cKDTree

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, SparseLinearMap


_DEFAULT_MAXIMUM_CONNECTIONS = 5_000_000


def _unit_interval_array(name: str, value: ArrayLike, /) -> np.ndarray:
    array = np.asarray(value)
    if not (
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must be real-valued.")
    array = np.asarray(array, dtype=float)
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be finite and lie in [0, 1].")
    return array


class ConicDensityFilterPlan(StrictModule, NonTrainableState):
    """A sparse conic filter over physical sample coordinates.

    ``design_mask`` partitions the samples into design and fixed regions. Filter rows
    see both regions, while values outside the design region are always supplied by
    ``fixed_density`` and remain fixed in the result. Omitting ``fixed_density``
    explicitly selects fixed void (zero density) outside the design region.

    Positive ``measures`` account for nonuniform physical sample volumes. The weight
    from source ``j`` to target ``i`` is
    ``measure[j] * max(radius - distance(i, j), 0)``, normalized by the exact target
    row sum. A zero radius is the identity transform on the design region.
    """

    coordinates: Array
    measures: Array
    design_mask: Array
    fixed_density: Array
    radius: float = eqx.field(static=True)
    maximum_connections: int = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        radius: float,
        design_mask: ArrayLike,
        fixed_density: ArrayLike | None = None,
        measures: ArrayLike | None = None,
        /,
        *,
        maximum_connections: int = _DEFAULT_MAXIMUM_CONNECTIONS,
    ):
        points_value = np.asarray(coordinates)
        if not (
            np.issubdtype(points_value.dtype, np.number)
            and not np.issubdtype(points_value.dtype, np.complexfloating)
        ):
            raise TypeError("coordinates must be real-valued.")
        points = np.asarray(points_value, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("coordinates must have shape (sample_count, dimension).")
        if np.any(~np.isfinite(points)):
            raise ValueError("coordinates must be finite.")

        radius_ = float(radius)
        if not isfinite(radius_) or radius_ < 0.0:
            raise ValueError("radius must be finite and non-negative.")

        mask_value = np.asarray(design_mask)
        if mask_value.dtype != np.dtype(bool):
            raise TypeError("design_mask must have boolean dtype.")
        mask = np.asarray(mask_value, dtype=bool)
        sample_count = int(points.shape[0])
        if mask.shape != (sample_count,):
            raise ValueError(
                f"design_mask must have shape ({sample_count},); got {mask.shape}."
            )

        if measures is None:
            volumes = np.ones((sample_count,), dtype=float)
        else:
            measures_value = np.asarray(measures)
            if not (
                np.issubdtype(measures_value.dtype, np.number)
                and not np.issubdtype(measures_value.dtype, np.complexfloating)
            ):
                raise TypeError("measures must be real-valued.")
            volumes = np.asarray(measures_value, dtype=float)
            if volumes.shape != (sample_count,):
                raise ValueError(
                    f"measures must have shape ({sample_count},); got {volumes.shape}."
                )
            if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
                raise ValueError("measures must be finite and strictly positive.")

        if fixed_density is None:
            fixed = np.zeros((sample_count,), dtype=float)
        else:
            fixed = _unit_interval_array("fixed_density", fixed_density)
            if fixed.shape != (sample_count,):
                raise ValueError(
                    f"fixed_density must have shape ({sample_count},); got {fixed.shape}."
                )

        if isinstance(maximum_connections, bool):
            raise TypeError("maximum_connections must be an integer.")
        connection_limit = int(maximum_connections)
        if connection_limit != maximum_connections or connection_limit <= 0:
            raise ValueError("maximum_connections must be a positive integer.")

        self.coordinates = jnp.asarray(points)
        self.measures = jnp.asarray(volumes)
        self.design_mask = jnp.asarray(mask)
        self.fixed_density = jnp.asarray(fixed)
        self.radius = radius_
        self.maximum_connections = connection_limit

    def prepare(self, /) -> "PreparedConicDensityFilter":
        """Materialize the bounded sparse physical-radius operator."""
        return PreparedConicDensityFilter(self)


class PreparedConicDensityFilter(StrictModule, NonTrainableState):
    """Prepared sparse conic operator with explicit fixed-region context."""

    plan: ConicDensityFilterPlan
    operator: SparseLinearMap

    def __init__(self, plan: ConicDensityFilterPlan, /):
        if not isinstance(plan, ConicDensityFilterPlan):
            raise TypeError("plan must be a ConicDensityFilterPlan.")
        points = np.asarray(plan.coordinates, dtype=float)
        measures = np.asarray(plan.measures, dtype=float)
        sample_count = int(points.shape[0])

        if plan.radius == 0.0:
            candidate_count = sample_count
            if candidate_count > plan.maximum_connections:
                raise ValueError(
                    "The zero-radius identity requires "
                    f"{candidate_count} connections, exceeding maximum_connections="
                    f"{plan.maximum_connections}."
                )
            sources = np.arange(sample_count, dtype=np.int32)
            targets = np.arange(sample_count, dtype=np.int32)
            coefficients = np.ones((sample_count,), dtype=float)
        else:
            tree = cKDTree(points)
            counts = np.asarray(
                tree.query_ball_point(points, plan.radius, return_length=True),
                dtype=np.int64,
            )
            candidate_count = int(np.sum(counts, dtype=np.int64))
            if candidate_count > plan.maximum_connections:
                raise ValueError(
                    "The physical-radius search requires at most "
                    f"{candidate_count} candidate connections, exceeding "
                    f"maximum_connections={plan.maximum_connections}."
                )

            sources_buffer = np.empty((candidate_count,), dtype=np.int32)
            targets_buffer = np.empty((candidate_count,), dtype=np.int32)
            weights_buffer = np.empty((candidate_count,), dtype=float)
            row_sums = np.zeros((sample_count,), dtype=float)
            route_count = 0
            neighbourhoods = tree.query_ball_point(points, plan.radius)
            for target, neighbours in enumerate(neighbourhoods):
                for source in sorted(int(index) for index in neighbours):
                    distance = float(np.linalg.norm(points[target] - points[source]))
                    weight = (plan.radius - distance) * measures[source]
                    if weight > 0.0:
                        if not isfinite(weight):
                            raise ValueError(
                                "Conic weights must remain finite; rescale the physical "
                                "coordinates, radius, or measures."
                            )
                        sources_buffer[route_count] = source
                        targets_buffer[route_count] = target
                        weights_buffer[route_count] = weight
                        row_sums[target] += weight
                        route_count += 1

            if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0.0):
                raise ValueError(
                    "Every conic-filter row must have finite positive support."
                )
            sources = sources_buffer[:route_count]
            targets = targets_buffer[:route_count]
            coefficients = weights_buffer[:route_count] / row_sums[targets]

        relation = EdgeRelation(
            sources,
            targets,
            source_size=sample_count,
            target_size=sample_count,
        )
        self.plan = plan
        self.operator = SparseLinearMap(relation, coefficients)

    def apply(self, density: ArrayLike, /) -> Array:
        """Filter design values while ignoring and restoring fixed-region inputs."""
        value = jnp.asarray(density)
        expected_shape = self.operator.input_shape
        if value.shape != expected_shape:
            raise ValueError(
                f"density must have shape {expected_shape}; got {value.shape}."
            )
        invalid_design = self.plan.design_mask & (
            ~jnp.isfinite(value) | (value < 0.0) | (value > 1.0)
        )
        value = eqx.error_if(
            value,
            jnp.any(invalid_design),
            "Design density must be finite and lie in [0, 1].",
        )
        contextual_density = jnp.where(
            self.plan.design_mask, value, self.plan.fixed_density
        )
        filtered = self.operator.mv(contextual_density)
        return jnp.where(self.plan.design_mask, filtered, self.plan.fixed_density)


class TanhDensityProjectionPlan(StrictModule, NonTrainableState):
    """A monotone finite-beta tanh projection with dynamic scalar parameters."""

    eta: Array

    def __init__(self, eta: ArrayLike, /):
        eta_value = np.asarray(eta)
        if eta_value.shape != ():
            raise ValueError("eta must be a scalar array.")
        if not (
            np.issubdtype(eta_value.dtype, np.number)
            and not np.issubdtype(eta_value.dtype, np.complexfloating)
        ):
            raise TypeError("eta must be real-valued.")
        eta_ = float(eta_value)
        if not isfinite(eta_) or not 0.0 < eta_ < 1.0:
            raise ValueError("eta must be finite and lie strictly between zero and one.")
        self.eta = jnp.asarray(eta_, dtype=float)

    def apply(self, filtered_density: ArrayLike, beta: ArrayLike, /) -> Array:
        """Project a density in [0, 1] using a finite positive dynamic ``beta``."""
        value = jnp.asarray(filtered_density)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value < 0.0) | (value > 1.0)),
            "Filtered density must be finite and lie in [0, 1].",
        )
        dtype = jnp.result_type(value, self.eta)
        beta_ = jnp.asarray(beta, dtype=dtype)
        if beta_.shape != ():
            raise ValueError("beta must be a scalar array.")
        beta_ = eqx.error_if(
            beta_,
            ~jnp.isfinite(beta_) | (beta_ <= 0.0),
            "beta must be finite and strictly positive.",
        )
        eta = self.eta.astype(dtype)
        lower = jnp.tanh(beta_ * eta)
        numerator = lower + jnp.tanh(beta_ * (value - eta))
        denominator = lower + jnp.tanh(beta_ * (1.0 - eta))
        return numerator / denominator


class DensityTransformPlan(StrictModule, NonTrainableState):
    """Compose one conic filter and one differentiable density projection."""

    filter: ConicDensityFilterPlan
    projection: TanhDensityProjectionPlan

    def __init__(
        self,
        filter: ConicDensityFilterPlan,
        projection: TanhDensityProjectionPlan,
        /,
    ):
        if not isinstance(filter, ConicDensityFilterPlan):
            raise TypeError("filter must be a ConicDensityFilterPlan.")
        if not isinstance(projection, TanhDensityProjectionPlan):
            raise TypeError("projection must be a TanhDensityProjectionPlan.")
        self.filter = filter
        self.projection = projection

    def prepare(self, /) -> "PreparedDensityTransform":
        """Prepare the sparse filtering stage for repeated differentiable application."""
        return PreparedDensityTransform(self, self.filter.prepare())


class PreparedDensityTransform(StrictModule, NonTrainableState):
    """Prepared filter-project chain with dynamic beta continuation."""

    plan: DensityTransformPlan
    filter: PreparedConicDensityFilter

    def __init__(
        self,
        plan: DensityTransformPlan,
        filter: PreparedConicDensityFilter,
        /,
    ):
        if not isinstance(plan, DensityTransformPlan):
            raise TypeError("plan must be a DensityTransformPlan.")
        if not isinstance(filter, PreparedConicDensityFilter):
            raise TypeError("filter must be a PreparedConicDensityFilter.")
        if filter.plan is not plan.filter:
            raise ValueError("Prepared filter and density-transform plan must match.")
        self.plan = plan
        self.filter = filter

    def apply(self, density: ArrayLike, beta: ArrayLike, /) -> Array:
        """Return physical density while preserving fixed values exactly."""
        filtered = self.filter.apply(density)
        projected = self.plan.projection.apply(filtered, beta)
        filter_plan = self.plan.filter
        return jnp.where(filter_plan.design_mask, projected, filter_plan.fixed_density)


def threshold_density(density: ArrayLike, eta: ArrayLike = 0.5, /) -> Array:
    """Return a hard binary density with no surrogate derivative.

    Values equal to ``eta`` are classified as solid. This operation is deliberately
    separate from :class:`TanhDensityProjectionPlan` and stops all reverse-mode
    sensitivity through the threshold.
    """
    value = jnp.asarray(density)
    value = eqx.error_if(
        value,
        jnp.any(~jnp.isfinite(value) | (value < 0.0) | (value > 1.0)),
        "Density must be finite and lie in [0, 1].",
    )
    cutoff = jnp.asarray(eta, dtype=jnp.result_type(value, float))
    if cutoff.shape != ():
        raise ValueError("eta must be a scalar array.")
    cutoff = eqx.error_if(
        cutoff,
        ~jnp.isfinite(cutoff) | (cutoff < 0.0) | (cutoff > 1.0),
        "eta must be finite and lie in [0, 1].",
    )
    binary = jnp.where(value >= cutoff, jnp.ones_like(value), jnp.zeros_like(value))
    return jax.lax.stop_gradient(binary)


__all__ = [
    "ConicDensityFilterPlan",
    "DensityTransformPlan",
    "PreparedConicDensityFilter",
    "PreparedDensityTransform",
    "TanhDensityProjectionPlan",
    "threshold_density",
]
