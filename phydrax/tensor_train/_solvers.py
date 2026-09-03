#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import TensorTrain, TensorTrainOperator, tt_svd
from ._local import regularized_least_squares


TensorTrainSolveMethod = Literal["als", "amen"]


class TensorTrainSolvePlan(StrictModule, NonTrainableState):
    """Static ALS/AMEn resources and convergence policy."""

    method: TensorTrainSolveMethod = eqx.field(static=True)
    mode_sizes: tuple[int, ...] = eqx.field(static=True)
    max_rank: int = eqx.field(static=True)
    enrichment_rank: int = eqx.field(static=True)
    sweeps: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    local_regularization: float = eqx.field(static=True)
    max_dense_entries: int = eqx.field(static=True)
    max_local_unknowns: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: TensorTrainSolveMethod,
        mode_sizes: Sequence[int],
        /,
        *,
        max_rank: int,
        enrichment_rank: int,
        sweeps: int,
        relative_tolerance: float,
        local_regularization: float,
        max_dense_entries: int,
        max_local_unknowns: int,
    ):
        modes = tuple(int(size) for size in mode_sizes)
        rank = int(max_rank)
        enrich = int(enrichment_rank)
        sweep_count = int(sweeps)
        tolerance = float(relative_tolerance)
        ridge = float(local_regularization)
        dense_limit = int(max_dense_entries)
        local_limit = int(max_local_unknowns)
        if method not in ("als", "amen"):
            raise ValueError("TensorTrain solve method must be 'als' or 'amen'.")
        if not modes or any(size <= 0 for size in modes):
            raise ValueError("TensorTrain solve modes must be nonempty and positive.")
        if rank <= 0 or sweep_count <= 0 or tolerance < 0.0 or ridge <= 0.0:
            raise ValueError(
                "TensorTrain solve ranks, sweeps, tolerance, or ridge are invalid."
            )
        if enrich <= 0 or enrich > rank:
            raise ValueError(
                "enrichment_rank must be positive and no greater than max_rank."
            )
        if dense_limit < prod(modes) ** 2 or local_limit <= 0:
            raise ValueError(
                "TensorTrain solve dense or local resource budget is infeasible."
            )
        self.method = method
        self.mode_sizes = modes
        self.max_rank = rank
        self.enrichment_rank = enrich
        self.sweeps = sweep_count
        self.relative_tolerance = tolerance
        self.local_regularization = ridge
        self.max_dense_entries = dense_limit
        self.max_local_unknowns = local_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "tensor-train-solve-plan",
                "method": method,
                "modes": modes,
                "max_rank": rank,
                "enrichment_rank": enrich,
                "sweeps": sweep_count,
                "relative_tolerance": tolerance,
                "local_regularization": ridge,
                "max_dense_entries": dense_limit,
                "max_local_unknowns": local_limit,
            }
        )


class PreparedTensorTrainSolve(StrictModule, NonTrainableState):
    plan: TensorTrainSolvePlan
    operator: TensorTrainOperator
    right_hand_side: TensorTrain
    initial: TensorTrain
    numeric_version: Array

    def __init__(
        self,
        plan: TensorTrainSolvePlan,
        operator: TensorTrainOperator,
        right_hand_side: TensorTrain,
        initial: TensorTrain,
        /,
        *,
        numeric_version: ArrayLike = 0,
    ):
        if (
            operator.input_mode_sizes != plan.mode_sizes
            or operator.output_mode_sizes != plan.mode_sizes
        ):
            raise ValueError("Prepared TT solve operator does not match the plan modes.")
        if (
            right_hand_side.mode_sizes != plan.mode_sizes
            or initial.mode_sizes != plan.mode_sizes
        ):
            raise ValueError("Prepared TT solve vectors do not match the plan modes.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.shape != () or int(np.asarray(version)) < 0:
            raise ValueError("numeric_version must be a non-negative scalar.")
        self.plan = plan
        self.operator = operator
        self.right_hand_side = right_hand_side
        self.initial = initial
        self.numeric_version = version


class TensorTrainSolveEvidence(StrictModule):
    """True global residual history and any residual-enrichment compression bounds."""

    true_global_residual_norms: Array
    relative_global_residual_norms: Array
    enrichment_frobenius_bounds: Array
    local_solve_count: int = eqx.field(static=True)
    sweep_count: int = eqx.field(static=True)

    def __init__(
        self,
        true_global_residual_norms: ArrayLike,
        relative_global_residual_norms: ArrayLike,
        enrichment_frobenius_bounds: ArrayLike,
        /,
        *,
        local_solve_count: int,
        sweep_count: int,
    ):
        residuals = jnp.asarray(true_global_residual_norms)
        relative = jnp.asarray(relative_global_residual_norms)
        bounds = jnp.asarray(enrichment_frobenius_bounds)
        if (
            residuals.ndim != 1
            or relative.shape != residuals.shape
            or bounds.shape != (sweep_count,)
        ):
            raise ValueError("TT solve evidence histories have inconsistent shapes.")
        self.true_global_residual_norms = residuals
        self.relative_global_residual_norms = relative
        self.enrichment_frobenius_bounds = bounds
        self.local_solve_count = int(local_solve_count)
        self.sweep_count = int(sweep_count)


class TensorTrainSolveResult(StrictModule):
    solution: TensorTrain
    evidence: TensorTrainSolveEvidence
    converged: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        solution: TensorTrain,
        evidence: TensorTrainSolveEvidence,
        converged: bool,
        /,
    ):
        self.solution = solution
        self.evidence = evidence
        self.converged = bool(converged)
        self.status = "converged" if self.converged else "iteration_budget_exhausted"


def plan_tensor_train_solve(
    operator: TensorTrainOperator,
    /,
    *,
    method: TensorTrainSolveMethod,
    max_rank: int,
    enrichment_rank: int,
    sweeps: int,
    relative_tolerance: float,
    local_regularization: float,
    max_dense_entries: int,
    max_local_unknowns: int,
) -> TensorTrainSolvePlan:
    if operator.input_mode_sizes != operator.output_mode_sizes:
        raise ValueError("ALS and AMEn-like solves require a square TT operator.")
    return TensorTrainSolvePlan(
        method,
        operator.input_mode_sizes,
        max_rank=max_rank,
        enrichment_rank=enrichment_rank,
        sweeps=sweeps,
        relative_tolerance=relative_tolerance,
        local_regularization=local_regularization,
        max_dense_entries=max_dense_entries,
        max_local_unknowns=max_local_unknowns,
    )


def plan_als(
    operator: TensorTrainOperator,
    /,
    **resources,
) -> TensorTrainSolvePlan:
    return plan_tensor_train_solve(operator, method="als", **resources)


def plan_amen(
    operator: TensorTrainOperator,
    /,
    **resources,
) -> TensorTrainSolvePlan:
    return plan_tensor_train_solve(operator, method="amen", **resources)


def prepare_tensor_train_solve(
    plan: TensorTrainSolvePlan,
    operator: TensorTrainOperator,
    right_hand_side: TensorTrain,
    initial: TensorTrain,
    /,
) -> PreparedTensorTrainSolve:
    return PreparedTensorTrainSolve(plan, operator, right_hand_side, initial)


def refresh_tensor_train_solve(
    prepared: PreparedTensorTrainSolve,
    /,
    *,
    operator: TensorTrainOperator,
    right_hand_side: TensorTrain,
    initial: TensorTrain,
) -> PreparedTensorTrainSolve:
    return PreparedTensorTrainSolve(
        prepared.plan,
        operator,
        right_hand_side,
        initial,
        numeric_version=prepared.numeric_version + 1,
    )


def _left_interface(tensor: TensorTrain, stop: int, /) -> Array:
    interface = jnp.ones((1, 1), dtype=tensor.dtype)
    for axis in range(stop):
        core = tensor.cores[axis]
        interface = ein.contract("pa,aib->pib", interface, core).reshape(
            (interface.shape[0] * core.shape[1], core.shape[2])
        )
    return interface


def _right_interface(tensor: TensorTrain, start: int, /) -> Array:
    interface = jnp.ones((1, 1), dtype=tensor.dtype)
    for axis in range(tensor.order - 1, start - 1, -1):
        core = tensor.cores[axis]
        interface = ein.contract("aib,bq->aiq", core, interface).reshape(
            (core.shape[0], core.shape[1] * interface.shape[1])
        )
    return interface


def _core_frame(tensor: TensorTrain, axis: int, /) -> Array:
    core = tensor.cores[axis]
    unknowns = prod(core.shape)
    basis = jnp.eye(unknowns, dtype=tensor.dtype).reshape(core.shape + (unknowns,))
    left = _left_interface(tensor, axis)
    right = _right_interface(tensor, axis + 1)
    return ein.contract("pa,aibk,bq->piqk", left, basis, right).reshape((-1, unknowns))


def _update_core(
    tensor: TensorTrain,
    matrix: Array,
    right_hand_side: Array,
    axis: int,
    plan: TensorTrainSolvePlan,
    /,
) -> TensorTrain:
    frame = _core_frame(tensor, axis)
    if frame.shape[1] > plan.max_local_unknowns:
        raise ValueError(
            f"ALS core needs {frame.shape[1]} local unknowns, exceeding budget "
            f"{plan.max_local_unknowns}."
        )
    design = ein.contract("ij,jk->ik", matrix, frame)
    local = regularized_least_squares(design, right_hand_side, plan.local_regularization)
    cores = list(tensor.cores)
    cores[axis] = local.reshape(cores[axis].shape)
    return TensorTrain(tuple(cores))


def _global_residual(
    matrix: Array, solution: TensorTrain, right: Array, plan: TensorTrainSolvePlan, /
) -> Array:
    dense = solution.to_dense(max_entries=plan.max_dense_entries).reshape((-1,))
    return right - ein.contract("ij,j->i", matrix, dense)


def solve_tensor_train(prepared: PreparedTensorTrainSolve, /) -> TensorTrainSolveResult:
    """Execute bounded ALS or SPD residual-enriched AMEn-like sweeps."""
    plan = prepared.plan
    matrix = prepared.operator.to_matrix(max_entries=plan.max_dense_entries)
    if plan.method == "amen":
        host_matrix = np.asarray(matrix)
        if (
            not np.allclose(
                host_matrix,
                np.conj(host_matrix.T),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            or float(np.min(np.linalg.eigvalsh(host_matrix))) <= 0.0
        ):
            raise ValueError("AMEn-like residual enrichment requires an SPD operator.")
    right = prepared.right_hand_side.to_dense(max_entries=plan.max_dense_entries).reshape(
        (-1,)
    )
    solution = prepared.initial
    right_norm = jnp.sqrt(jnp.sum(jnp.abs(right) ** 2))
    safe_right_norm = jnp.where(right_norm > 0, right_norm, 1)
    residual = _global_residual(matrix, solution, right, plan)
    residual_norms = [jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2))]
    enrichment_bounds: list[Array] = []
    for _ in range(plan.sweeps):
        for axis in range(solution.order):
            solution = _update_core(solution, matrix, right, axis, plan)
        for axis in range(solution.order - 1, -1, -1):
            solution = _update_core(solution, matrix, right, axis, plan)
        bound = jnp.asarray(0, dtype=right_norm.dtype)
        residual = _global_residual(matrix, solution, right, plan)
        if plan.method == "amen":
            direction_decomposition = tt_svd(
                residual.reshape(plan.mode_sizes),
                max_ranks=plan.enrichment_rank,
                relative_tolerance=0.0,
            )
            direction = direction_decomposition.tensor
            direction_dense = direction.to_dense(
                max_entries=plan.max_dense_entries
            ).reshape((-1,))
            applied_direction = ein.contract("ij,j->i", matrix, direction_dense)
            denominator = jnp.real(
                ein.contract("i,i->", jnp.conj(applied_direction), applied_direction)
            )
            numerator = jnp.real(
                ein.contract("i,i->", jnp.conj(applied_direction), residual)
            )
            step = jnp.where(denominator > 0, numerator / denominator, 0)
            enriched = solution + step * direction
            rounded = enriched.round(max_ranks=plan.max_rank, relative_tolerance=0.0)
            solution = rounded.tensor
            bound = (
                jnp.abs(step) * direction_decomposition.evidence.frobenius_error_bound
                + rounded.evidence.frobenius_error_bound
            )
            residual = _global_residual(matrix, solution, right, plan)
        enrichment_bounds.append(bound)
        residual_norms.append(jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2)))
    residual_history = jnp.stack(residual_norms)
    relative_history = residual_history / safe_right_norm
    converged = bool(np.asarray(relative_history[-1] <= plan.relative_tolerance))
    evidence = TensorTrainSolveEvidence(
        residual_history,
        relative_history,
        jnp.stack(enrichment_bounds),
        local_solve_count=2 * solution.order * plan.sweeps,
        sweep_count=plan.sweeps,
    )
    return TensorTrainSolveResult(solution, evidence, converged)


__all__ = [
    "PreparedTensorTrainSolve",
    "TensorTrainSolveEvidence",
    "TensorTrainSolveMethod",
    "TensorTrainSolvePlan",
    "TensorTrainSolveResult",
    "plan_als",
    "plan_amen",
    "plan_tensor_train_solve",
    "prepare_tensor_train_solve",
    "refresh_tensor_train_solve",
    "solve_tensor_train",
]
