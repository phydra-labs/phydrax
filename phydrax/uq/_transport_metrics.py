#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..integration import weighted
from ..transport import (
    AbstractBalancedTransportSolver,
    AbstractGroundCost,
    discrete_problem,
    Sinkhorn,
    sinkhorn_divergence,
    SinkhornDivergenceResult,
    sliced_wasserstein_distance,
    SlicedWassersteinResult,
    SquaredEuclideanCost,
)
from ._operator import _queries_equal, OperatorPredictiveField
from ._operator_event import (
    case_count,
    event_weights,
    Measure,
    OperatorReduction,
    reduce_cases,
    require_predictive,
    sample_case_event,
)


class PredictiveTransportMetricResult(StrictModule):
    """Case-preserving predictive transport metric with native diagnostics."""

    value: Array
    per_case: Array
    transport: SinkhornDivergenceResult | None
    sliced: SlicedWassersteinResult | None
    method: str = eqx.field(static=True)
    reduction: str = eqx.field(static=True)


def predictive_sinkhorn_divergence(
    source_samples: ArrayLike,
    target_samples: ArrayLike,
    /,
    *,
    source_weights: ArrayLike | None = None,
    target_weights: ArrayLike | None = None,
    cost: AbstractGroundCost | None = None,
    solver: AbstractBalancedTransportSolver | None = None,
    epsilon: float = 0.1,
) -> SinkhornDivergenceResult:
    """Compare two empirical predictive laws of complete vector events."""
    source = _event_samples(source_samples, name="source_samples")
    target = _event_samples(target_samples, name="target_samples")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Predictive event sizes must agree.")
    configured_cost = SquaredEuclideanCost() if cost is None else cost
    configured_solver = _solver(epsilon, solver)
    source_target = weighted(
        source,
        _log_sample_weights(source_weights, source.shape[0], dtype=source.dtype),
        normalized=True,
        sample_axes=0,
        provenance="predictive-transport-source",
    )
    target_target = weighted(
        target,
        _log_sample_weights(target_weights, target.shape[0], dtype=target.dtype),
        normalized=True,
        sample_axes=0,
        provenance="predictive-transport-target",
    )
    problem = discrete_problem(source_target, target_target, cost=configured_cost)
    return sinkhorn_divergence(problem, configured_solver)


def operator_ensemble_sinkhorn_divergence(
    left: OperatorPredictiveField,
    right: OperatorPredictiveField,
    /,
    *,
    left_weights: ArrayLike | None = None,
    right_weights: ArrayLike | None = None,
    measure: Measure = "quadrature",
    reduction: OperatorReduction = "mean",
    cost: AbstractGroundCost | None = None,
    solver: AbstractBalancedTransportSolver | None = None,
    epsilon: float = 0.1,
) -> PredictiveTransportMetricResult:
    """Compare complete operator-output laws independently for every case."""
    left_samples, right_samples, coordinate_scale = _operator_events(
        left,
        right,
        measure=measure,
    )
    configured_cost = SquaredEuclideanCost() if cost is None else cost
    configured_solver = _solver(epsilon, solver)
    left_cases = jnp.swapaxes(left_samples * coordinate_scale[None, ...], 0, 1)
    right_cases = jnp.swapaxes(right_samples * coordinate_scale[None, ...], 0, 1)

    def solve_case(source, target):
        return predictive_sinkhorn_divergence(
            source,
            target,
            source_weights=left_weights,
            target_weights=right_weights,
            cost=configured_cost,
            solver=configured_solver,
        )

    transport = jax.vmap(solve_case)(left_cases, right_cases)
    per_case = transport.value.reshape(left.case_shape)
    value = reduce_cases(per_case, left.case_shape, reduction=reduction)
    return PredictiveTransportMetricResult(
        value=value,
        per_case=per_case,
        transport=transport,
        sliced=None,
        method="sinkhorn-divergence",
        reduction=reduction,
    )


def operator_ensemble_sliced_wasserstein(
    left: OperatorPredictiveField,
    right: OperatorPredictiveField,
    /,
    *,
    left_weights: ArrayLike | None = None,
    right_weights: ArrayLike | None = None,
    measure: Measure = "quadrature",
    reduction: OperatorReduction = "mean",
    p: float = 2.0,
    num_projections: int = 128,
    key: Array | None = None,
    projections: ArrayLike | None = None,
) -> PredictiveTransportMetricResult:
    """Estimate sliced transport between complete operator-output laws per case."""
    left_samples, right_samples, coordinate_scale = _operator_events(
        left,
        right,
        measure=measure,
    )
    left_cases = jnp.swapaxes(left_samples * coordinate_scale[None, ...], 0, 1)
    right_cases = jnp.swapaxes(right_samples * coordinate_scale[None, ...], 0, 1)
    count = case_count(left.case_shape)
    if projections is None:
        if key is None:
            raise ValueError("key is required when projections are not supplied.")
        keys = jax.vmap(lambda index: jr.fold_in(key, index))(
            jnp.arange(count, dtype=jnp.uint32)
        )

        def solve_case(source, target, case_key):
            return sliced_wasserstein_distance(
                source,
                target,
                source_weights=left_weights,
                target_weights=right_weights,
                p=p,
                num_projections=num_projections,
                key=case_key,
            )

        sliced = jax.vmap(solve_case)(left_cases, right_cases, keys)
    else:
        directions = jnp.asarray(projections, dtype=left_cases.dtype)

        def solve_case(source, target):
            return sliced_wasserstein_distance(
                source,
                target,
                source_weights=left_weights,
                target_weights=right_weights,
                p=p,
                projections=directions,
            )

        sliced = jax.vmap(solve_case)(left_cases, right_cases)
    per_case = sliced.value.reshape(left.case_shape)
    value = reduce_cases(per_case, left.case_shape, reduction=reduction)
    return PredictiveTransportMetricResult(
        value=value,
        per_case=per_case,
        transport=None,
        sliced=sliced,
        method="sliced-wasserstein",
        reduction=reduction,
    )


def _operator_events(
    left: OperatorPredictiveField,
    right: OperatorPredictiveField,
    /,
    *,
    measure: Measure,
) -> tuple[Array, Array, Array]:
    require_predictive(left)
    require_predictive(right)
    if (
        left.case_axes != right.case_axes
        or left.case_shape != right.case_shape
        or left.field_name != right.field_name
        or left.output_spec.channels != right.output_spec.channels
        or left.output_spec.component_names != right.output_spec.component_names
        or not _queries_equal(left.query, right.query)
    ):
        raise ValueError("Operator ensembles must share one physical output contract.")
    left_samples = sample_case_event(left)
    right_samples = sample_case_event(right)
    weights = event_weights(
        left.query,
        left.output_spec,
        left.case_shape,
        measure=measure,
    ).reshape((case_count(left.case_shape), -1))
    return left_samples, right_samples, jnp.sqrt(weights)


def _event_samples(value: ArrayLike, /, *, name: str) -> Array:
    result = jnp.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must have shape (sample, event) with nonempty axes.")
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite events.",
    )


def _log_sample_weights(
    weights: ArrayLike | None,
    count: int,
    /,
    *,
    dtype,
) -> Array:
    if weights is None:
        return jnp.zeros((count,), dtype=dtype)
    values = jnp.asarray(weights, dtype=dtype)
    if values.shape != (count,):
        raise ValueError(f"Sample weights must have shape {(count,)}.")
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | jnp.any(values < 0.0) | (jnp.sum(values) <= 0.0),
        "Sample weights must be finite, nonnegative, and have positive total mass.",
    )
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


def _solver(
    epsilon: float, solver: AbstractBalancedTransportSolver | None, /
) -> AbstractBalancedTransportSolver:
    if solver is not None:
        if not isinstance(solver, AbstractBalancedTransportSolver):
            raise TypeError(
                "solver must implement the balanced transport solver contract or be None."
            )
        return solver
    return Sinkhorn(
        epsilon,
        max_iterations=500,
        min_iterations=1,
        tolerance=1e-7,
        check_every=5,
        early_stop=False,
    )


__all__ = [
    "PredictiveTransportMetricResult",
    "operator_ensemble_sinkhorn_divergence",
    "operator_ensemble_sliced_wasserstein",
    "predictive_sinkhorn_divergence",
]
