#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed and sequential finite-design Bayesian quadrature preparation."""

from __future__ import annotations

from numbers import Integral
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
)
from ._kernel_means import AbstractKernelMean


class _PositiveFactor(NamedTuple):
    prepared: PreparedLinearSolve


class FixedBayesianQuadratureDesign(StrictModule):
    """Explicit fixed point design with no implicit target transport."""

    points: Array
    source_indices: Array

    def __init__(self, points: ArrayLike, /, *, source_indices: ArrayLike | None = None):
        values = jnp.asarray(points)
        if values.ndim < 2 or int(values.shape[0]) <= 0:
            raise ValueError("points must have shape (point,) + kernel_input_shape.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Bayesian-quadrature points must be finite.",
        )
        indices = (
            jnp.arange(values.shape[0], dtype=jnp.int32)
            if source_indices is None
            else jnp.asarray(source_indices, dtype=jnp.int32)
        )
        if indices.shape != (values.shape[0],):
            raise ValueError("source_indices must align with points.")
        self.points = values
        self.source_indices = indices

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


class SequentialBayesianQuadratureDesign(StrictModule):
    """Finite candidate pool and deterministic exact variance-reduction policy."""

    candidates: Array
    initial_indices: Array
    initial_count: int = eqx.field(static=True)
    total_count: int = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)

    def __init__(
        self,
        candidates: ArrayLike,
        /,
        *,
        initial_count: int,
        total_count: int,
        block_size: int = 256,
        initial_indices: ArrayLike | None = None,
    ):
        values = jnp.asarray(candidates)
        if values.ndim < 2 or int(values.shape[0]) <= 0:
            raise ValueError("candidates must have a nonempty candidate axis.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Sequential BQ candidates must be finite.",
        )
        candidate_count = int(values.shape[0])
        initial = _positive_integer(initial_count, name="initial_count")
        total = _positive_integer(total_count, name="total_count")
        block = _positive_integer(block_size, name="block_size")
        if initial > total or total > candidate_count:
            raise ValueError(
                "Sequential BQ requires initial_count <= total_count <= candidate_count."
            )
        indices = (
            jnp.arange(initial, dtype=jnp.int32)
            if initial_indices is None
            else jnp.asarray(initial_indices, dtype=jnp.int32)
        )
        host = np.asarray(jax.device_get(indices))
        if (
            indices.shape != (initial,)
            or np.any(host < 0)
            or np.any(host >= candidate_count)
            or np.unique(host).size != initial
        ):
            raise ValueError("initial_indices must be distinct in-range candidates.")
        self.candidates = values
        self.initial_indices = indices
        self.initial_count = initial
        self.total_count = total
        self.candidate_count = candidate_count
        self.block_size = block


class PreparedKernelMeanBayesianQuadrature(StrictModule):
    """Prepared weights, selected support, and posterior-variance evidence."""

    points: Array
    source_indices: Array
    mask: Array
    weights: Array
    kernel_mean: Array
    kernel_double_mean: Array
    posterior_variance: Array
    variance_reduction_history: Array
    posterior_variance_history: Array
    target_mass: Array
    observation_noise: Array
    solve_regularization: Array
    solve_result: LinearSolveResult
    target_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    embedding_exactness: str = eqx.field(static=True)
    embedding_hypotheses: str = eqx.field(static=True)
    proposal_kind: str = eqx.field(static=True)


def prepare_kernel_mean_bayesian_quadrature(
    plan: Any, /
) -> PreparedKernelMeanBayesianQuadrature:
    """Prepare explicit or sequential design weights before integrand evaluation."""
    from ._bayesian_quadrature import BayesianQuadraturePlan

    if not isinstance(plan, BayesianQuadraturePlan):
        raise TypeError("plan must be a BayesianQuadraturePlan.")
    kernel_mean = plan.kernel_mean
    if not isinstance(kernel_mean, AbstractKernelMean):
        raise TypeError("plan.kernel_mean must implement AbstractKernelMean.")
    if isinstance(plan.design, FixedBayesianQuadratureDesign):
        points = plan.design.points
        indices = plan.design.source_indices
        reductions = jnp.empty((0,), dtype=points.dtype)
        selection_variances = jnp.empty((0,), dtype=points.dtype)
        proposal_kind = "fixed"
    elif isinstance(plan.design, SequentialBayesianQuadratureDesign):
        indices, reductions, selection_variances = _select_sequential(
            kernel_mean,
            plan.design,
            observation_noise=plan.observation_noise,
            solve_regularization=plan.solve_regularization,
            solve_policy=plan.solve_policy,
        )
        points = plan.design.candidates[indices]
        proposal_kind = "exact-posterior-variance-reduction"
    else:
        raise TypeError(
            "Kernel-mean preparation requires FixedBayesianQuadratureDesign or SequentialBayesianQuadratureDesign."
        )
    if int(points.shape[0]) > plan.max_points:
        raise ValueError("Prepared BQ support exceeds plan.max_points.")
    matrix = kernel_mean.matrix(points, points)
    kernel_vector = kernel_mean.mean(points)
    double_mean = kernel_mean.double_mean()
    system = matrix + (plan.observation_noise + plan.solve_regularization) * jnp.eye(
        points.shape[0], dtype=matrix.dtype
    )
    factor = _prepare_factor(system, plan.solve_policy)
    solve_result = solve(factor.prepared, kernel_vector)
    weights = _solve_vector(factor, kernel_vector)
    posterior_variance = double_mean - oe.contract("i,i->", kernel_vector, weights)
    variance_history = (
        jnp.asarray((posterior_variance,), dtype=points.dtype)
        if selection_variances.shape[0] == 0
        else selection_variances
    )
    return PreparedKernelMeanBayesianQuadrature(
        points=points,
        source_indices=indices,
        mask=jnp.ones((points.shape[0],), dtype=bool),
        weights=weights,
        kernel_mean=kernel_vector,
        kernel_double_mean=double_mean,
        posterior_variance=posterior_variance,
        variance_reduction_history=reductions,
        posterior_variance_history=variance_history,
        target_mass=kernel_mean.target_mass,
        observation_noise=plan.observation_noise,
        solve_regularization=plan.solve_regularization,
        solve_result=solve_result,
        target_id=kernel_mean.target_id,
        kernel_id=kernel_mean.kernel.kernel_id,
        embedding_exactness=kernel_mean.exactness,
        embedding_hypotheses=kernel_mean.hypotheses,
        proposal_kind=proposal_kind,
    )


def reduce_kernel_mean_bayesian_quadrature(
    values: Any, prepared: PreparedKernelMeanBayesianQuadrature, /
) -> Any:
    """Contract real BQ weights with array/PyTree/complex integrand values."""
    if not isinstance(prepared, PreparedKernelMeanBayesianQuadrature):
        raise TypeError("prepared must be PreparedKernelMeanBayesianQuadrature.")

    def reduce_leaf(leaf: Any) -> Array:
        array = jnp.asarray(leaf)
        if array.ndim == 0 or array.shape[0] != prepared.points.shape[0]:
            raise ValueError("Every integrand leaf must start with the BQ point axis.")
        return oe.contract("i,i...->...", prepared.weights, array)

    return jax.tree_util.tree_map(reduce_leaf, values)


def _select_sequential(
    kernel_mean: AbstractKernelMean,
    design: SequentialBayesianQuadratureDesign,
    /,
    *,
    observation_noise: Array,
    solve_regularization: Array,
    solve_policy: LinearSolvePolicy,
) -> tuple[Array, Array, Array]:
    selected = [
        int(index) for index in np.asarray(jax.device_get(design.initial_indices))
    ]
    reductions: list[Array] = []
    variance_history: list[Array] = []
    candidates = design.candidates
    diagonal = kernel_mean.kernel.diagonal(candidates)
    means = kernel_mean.mean(candidates)
    used = np.zeros((design.candidate_count,), dtype=bool)
    used[selected] = True
    while True:
        points = candidates[jnp.asarray(selected, dtype=jnp.int32)]
        matrix = kernel_mean.matrix(points, points)
        system = matrix + (observation_noise + solve_regularization) * jnp.eye(
            len(selected), dtype=matrix.dtype
        )
        factor = _prepare_factor(system, solve_policy)
        selected_mean = kernel_mean.mean(points)
        solved_mean = _solve_vector(factor, selected_mean)
        current_variance = kernel_mean.double_mean() - oe.contract(
            "i,i->", selected_mean, solved_mean
        )
        variance_history.append(current_variance)
        if len(selected) == design.total_count:
            break
        cross = kernel_mean.matrix(candidates, points)
        solved_cross = jax.vmap(lambda row: _solve_vector(factor, row))(cross)
        numerator = means - cross @ solved_mean
        denominator = (
            diagonal + observation_noise - oe.contract("ij,ij->i", cross, solved_cross)
        )
        valid = (
            ~jnp.asarray(used)
            & jnp.isfinite(numerator)
            & jnp.isfinite(denominator)
            & (denominator > 0.0)
        )
        reduction = jnp.where(valid, numerator * numerator / denominator, -jnp.inf)
        if not bool(jnp.any(jnp.isfinite(reduction))):
            raise ValueError(
                "Sequential BQ exhausted unique candidates with positive conditional variance."
            )
        index = int(jnp.argmax(reduction))
        selected.append(index)
        used[index] = True
        reductions.append(reduction[index])
    return (
        jnp.asarray(selected, dtype=jnp.int32),
        jnp.stack(tuple(reductions))
        if reductions
        else jnp.empty((0,), dtype=candidates.dtype),
        jnp.stack(tuple(variance_history)),
    )


def _prepare_factor(matrix: Array, policy: LinearSolvePolicy, /) -> _PositiveFactor:
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    return _PositiveFactor(prepare(LinearSystem(operator), policy))


def _solve_vector(factor: _PositiveFactor, right: Array, /) -> Array:
    result = solve(factor.prepared, right)
    return eqx.error_if(
        result.value,
        ~result.successful,
        "Bayesian-quadrature positive-definite solve failed.",
    )


def _positive_integer(value: int, /, *, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


__all__ = [
    "FixedBayesianQuadratureDesign",
    "PreparedKernelMeanBayesianQuadrature",
    "SequentialBayesianQuadratureDesign",
    "prepare_kernel_mean_bayesian_quadrature",
    "reduce_kernel_mean_bayesian_quadrature",
]
