#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule
from ..linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ._peps import PEPS


class PEPSUpdatePolicy(StrictModule):
    maximum_bond_dimension: int = eqx.field(static=True)
    singular_value_cutoff: float = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    maximum_tensor_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_bond_dimension: int,
        /,
        *,
        singular_value_cutoff: float = 0.0,
        regularization: float = 1e-10,
        maximum_tensor_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
    ):
        values = (
            int(maximum_bond_dimension),
            float(singular_value_cutoff),
            float(regularization),
            int(maximum_tensor_elements),
            int(maximum_workspace_bytes),
        )
        if (
            values[0] < 1
            or values[1] < 0.0
            or values[2] < 0.0
            or values[3] < 1
            or values[4] < 1
        ):
            raise ValueError("PEPS update policy values are outside their finite ranges.")
        self.maximum_bond_dimension = values[0]
        self.singular_value_cutoff = values[1]
        self.regularization = values[2]
        self.maximum_tensor_elements = values[3]
        self.maximum_workspace_bytes = values[4]
        self.policy_id = canonical_fingerprint(
            {
                "kind": "peps-update-policy",
                "maximum_bond_dimension": values[0],
                "singular_value_cutoff": values[1],
                "regularization": values[2],
                "maximum_tensor_elements": values[3],
                "maximum_workspace_bytes": values[4],
            }
        )


class PEPSUpdateEvidence(StrictModule):
    source_state_id: str = eqx.field(static=True)
    result_state_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    route: str = eqx.field(static=True)
    bond: tuple[int, int, str] = eqx.field(static=True)
    truncation_error: Array
    local_objective_residual: Array
    gauge_residual: Array
    solver_status: Array
    solver_successful: Array
    solver_plan_id: str = eqx.field(static=True)
    solver_backend: str = eqx.field(static=True)
    solver_method: str = eqx.field(static=True)
    finite: Array
    accepted: Array
    exact_local_update: Array
    claim: str = eqx.field(static=True)
    global_error_bound_claimed: bool = eqx.field(static=True)
    admitted_peak_elements: int = eqx.field(static=True)
    admitted_peak_bytes: int = eqx.field(static=True)


class PEPSUpdateResult(StrictModule):
    state: PEPS
    bond_singular_values: Array
    evidence: PEPSUpdateEvidence


def _bond_sites(state: PEPS, row: int, column: int, direction: str, /) -> tuple[int, int]:
    row_ = int(row)
    column_ = int(column)
    if not 0 <= row_ < state.rows or not 0 <= column_ < state.columns:
        raise ValueError("PEPS update site lies outside the lattice.")
    if direction == "right":
        if column_ + 1 >= state.columns:
            raise ValueError("Right PEPS update crosses the OBC boundary.")
        return row_ * state.columns + column_, row_ * state.columns + column_ + 1
    if direction == "down":
        if row_ + 1 >= state.rows:
            raise ValueError("Down PEPS update crosses the OBC boundary.")
        return row_ * state.columns + column_, (row_ + 1) * state.columns + column_
    raise ValueError("direction must be 'right' or 'down'.")


def _two_site_matrix(
    first: Array, second: Array, gate: Array, direction: str, /
) -> tuple[Array, tuple[int, ...], tuple[int, ...]]:
    if gate.ndim != 4 or gate.shape[2:] != (first.shape[4], second.shape[4]):
        raise ValueError(
            "Two-site gate axes must be (out-left, out-right, in-left, in-right)."
        )
    if direction == "right":
        if first.shape[1] != second.shape[3]:
            raise ValueError("Horizontal PEPS update bond dimensions differ.")
        tensor = oe.contract(
            "axbcp,defxq,PQpq->abcPdefQ",
            first,
            second,
            gate,
            optimize="greedy",
        )
        left_shape = (first.shape[0], first.shape[2], first.shape[3], gate.shape[0])
        right_shape = (second.shape[0], second.shape[1], second.shape[2], gate.shape[1])
    else:
        if first.shape[2] != second.shape[0]:
            raise ValueError("Vertical PEPS update bond dimensions differ.")
        tensor = oe.contract(
            "abxcp,xdefq,PQpq->abcPdefQ",
            first,
            second,
            gate,
            optimize="greedy",
        )
        left_shape = (first.shape[0], first.shape[1], first.shape[3], gate.shape[0])
        right_shape = (second.shape[1], second.shape[2], second.shape[3], gate.shape[1])
    return tensor.reshape((prod(left_shape), prod(right_shape))), left_shape, right_shape


def _admit_update(
    first: Array,
    second: Array,
    gate: Array,
    direction: str,
    policy: PEPSUpdatePolicy,
    metric_elements: int,
    /,
) -> tuple[int, int]:
    if gate.dtype != first.dtype or second.dtype != first.dtype:
        raise TypeError("PEPS tensors and gate must use one storage dtype.")
    if direction == "right":
        left_elements = first.shape[0] * first.shape[2] * first.shape[3] * gate.shape[0]
        right_elements = (
            second.shape[0] * second.shape[1] * second.shape[2] * gate.shape[1]
        )
    else:
        left_elements = first.shape[0] * first.shape[1] * first.shape[3] * gate.shape[0]
        right_elements = (
            second.shape[1] * second.shape[2] * second.shape[3] * gate.shape[1]
        )
    matrix_elements = left_elements * right_elements
    peak = matrix_elements * 3 + metric_elements
    if (
        matrix_elements > policy.maximum_tensor_elements
        or peak > policy.maximum_tensor_elements
    ):
        raise MemoryError(
            "PEPS update exceeds maximum_tensor_elements before allocation."
        )
    bytes_ = peak * precision_itemsize(str(first.dtype))
    if bytes_ > policy.maximum_workspace_bytes:
        raise MemoryError(
            "PEPS update exceeds maximum_workspace_bytes before allocation."
        )
    return peak, bytes_


def _factor_update(
    matrix: Array,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    direction: str,
    policy: PEPSUpdatePolicy,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    decomposition = factorize(DenseLinearOperator(matrix), FactorizationPolicy("svd"))
    svd = decomposition.prepared_solve.state
    u, singular, vh = svd.u, svd.singular_values, svd.vh
    retained = min(policy.maximum_bond_dimension, singular.shape[0])
    kept = singular[:retained]
    mask = kept > policy.singular_value_cutoff
    discarded = jnp.sum(jnp.square(jnp.abs(singular[retained:])))
    discarded = discarded + jnp.sum(jnp.where(mask, 0.0, jnp.square(jnp.abs(kept))))
    left_matrix = u[:, :retained]
    right_matrix = jnp.where(mask, kept, 0.0)[:, None] * vh[:retained, :]
    gauge = jnp.max(
        jnp.abs(
            jnp.conj(left_matrix.T) @ left_matrix
            - jnp.eye(retained, dtype=left_matrix.dtype)
        )
    )
    reconstructed = left_matrix @ right_matrix
    residual = jnp.linalg.norm(matrix - reconstructed)
    if direction == "right":
        left = left_matrix.reshape(left_shape + (retained,)).transpose((0, 4, 1, 2, 3))
        right = right_matrix.reshape((retained,) + right_shape).transpose((1, 2, 3, 0, 4))
    else:
        left = left_matrix.reshape(left_shape + (retained,)).transpose((0, 1, 4, 2, 3))
        right = right_matrix.reshape((retained,) + right_shape)
    return left, right, kept, discarded, jnp.stack((residual, gauge))


def _result(
    state: PEPS,
    tensors: list[Array],
    singular: Array,
    row: int,
    column: int,
    direction: str,
    policy: PEPSUpdatePolicy,
    route: str,
    discarded: Array,
    residual_gauge: Array,
    solver_status: Array,
    solver_successful: Array,
    solver_plan_id: str,
    solver_backend: str,
    solver_method: str,
    peak_elements: int,
    peak_bytes: int,
    /,
) -> PEPSUpdateResult:
    updated = PEPS(
        tuple(tensors),
        state.rows,
        state.columns,
        precision=state.precision,
        numeric_version=state.numeric_version + 1,
    )
    finite = (
        jnp.all(jnp.isfinite(singular))
        & jnp.isfinite(discarded)
        & jnp.all(jnp.isfinite(residual_gauge))
    )
    accepted = finite & solver_successful
    exact = accepted & (discarded == 0.0) & jnp.asarray(route == "simple")
    replay_id = canonical_fingerprint(
        {
            "kind": "peps-local-update",
            "source": state.state_id,
            "result": updated.state_id,
            "bond": (int(row), int(column), direction),
            "route": route,
            "policy": policy.policy_id,
        }
    )
    evidence = PEPSUpdateEvidence(
        state.state_id,
        updated.state_id,
        replay_id,
        route,
        (int(row), int(column), direction),
        discarded,
        residual_gauge[0],
        residual_gauge[1],
        solver_status,
        solver_successful,
        solver_plan_id,
        solver_backend,
        solver_method,
        finite,
        accepted,
        exact,
        (
            "simple local SVD update; no global error bound"
            if route == "simple"
            else "full environment-weighted regularized solve update; no global error bound"
        ),
        False,
        peak_elements,
        peak_bytes,
    )
    return PEPSUpdateResult(updated, singular, evidence)


def simple_update_peps(
    state: PEPS,
    row: int,
    column: int,
    direction: str,
    gate: ArrayLike,
    policy: PEPSUpdatePolicy,
    /,
) -> PEPSUpdateResult:
    """Apply a two-site gate and truncate by the unweighted local Schmidt spectrum."""

    if not isinstance(state, PEPS) or not isinstance(policy, PEPSUpdatePolicy):
        raise TypeError("state and policy have invalid types.")
    first_index, second_index = _bond_sites(state, row, column, direction)
    gate_ = jnp.asarray(gate)
    if gate_.ndim != 4:
        raise ValueError("Two-site gate must have rank four.")
    first = state.tensors[first_index]
    second = state.tensors[second_index]
    peak_elements, peak_bytes = _admit_update(first, second, gate_, direction, policy, 0)
    matrix, left_shape, right_shape = _two_site_matrix(first, second, gate_, direction)
    left, right, singular, discarded, residual_gauge = _factor_update(
        matrix, left_shape, right_shape, direction, policy
    )
    tensors = list(state.tensors)
    tensors[first_index] = left
    tensors[second_index] = right
    return _result(
        state,
        tensors,
        singular,
        row,
        column,
        direction,
        policy,
        "simple",
        discarded,
        residual_gauge,
        jnp.asarray(-1, dtype=jnp.int32),
        jnp.asarray(True),
        "",
        "",
        "",
        peak_elements,
        peak_bytes,
    )


def full_update_peps(
    state: PEPS,
    row: int,
    column: int,
    direction: str,
    gate: ArrayLike,
    environment_metric: ArrayLike,
    policy: PEPSUpdatePolicy,
    /,
) -> PEPSUpdateResult:
    """Apply an environment-weighted update using the native linear-solve substrate."""

    if not isinstance(state, PEPS) or not isinstance(policy, PEPSUpdatePolicy):
        raise TypeError("state and policy have invalid types.")
    first_index, second_index = _bond_sites(state, row, column, direction)
    gate_ = jnp.asarray(gate)
    metric = jnp.asarray(environment_metric)
    first = state.tensors[first_index]
    second = state.tensors[second_index]
    if gate_.ndim != 4:
        raise ValueError("Two-site gate must have rank four.")
    if direction == "right":
        left_size = first.shape[0] * first.shape[2] * first.shape[3] * gate_.shape[0]
    else:
        left_size = first.shape[0] * first.shape[1] * first.shape[3] * gate_.shape[0]
    if metric.shape != (left_size, left_size):
        raise ValueError(
            "environment_metric must act on the updated left composite space."
        )
    if metric.dtype != first.dtype:
        raise TypeError("environment_metric dtype must match PEPS storage dtype.")
    peak_elements, peak_bytes = _admit_update(
        first, second, gate_, direction, policy, metric.size
    )
    matrix, left_shape, right_shape = _two_site_matrix(first, second, gate_, direction)
    if policy.regularization <= 0.0:
        raise ValueError("Full PEPS update requires strictly positive regularization.")
    gram = jnp.conj(metric.T) @ metric
    regularized = gram + policy.regularization * jnp.eye(left_size, dtype=metric.dtype)
    operator = DenseLinearOperator(
        regularized,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    solved = solve(
        LinearSystem(operator),
        gram @ matrix,
        policy=LinearSolvePolicy(DenseCholesky(), failure=FailurePolicy("status")),
    )
    weighted_target = solved.value
    left, right, singular, discarded, residual_gauge = _factor_update(
        weighted_target, left_shape, right_shape, direction, policy
    )
    tensors = list(state.tensors)
    tensors[first_index] = left
    tensors[second_index] = right
    weighted_residual = jnp.linalg.norm(regularized @ weighted_target - gram @ matrix)
    residual_gauge = residual_gauge.at[0].set(weighted_residual)
    return _result(
        state,
        tensors,
        singular,
        row,
        column,
        direction,
        policy,
        "full-environment-weighted",
        discarded,
        residual_gauge,
        jnp.max(solved.status),
        jnp.all(solved.successful),
        solved.provenance.plan_id,
        solved.provenance.backend,
        solved.provenance.method,
        peak_elements,
        peak_bytes,
    )


__all__ = [
    "PEPSUpdateEvidence",
    "PEPSUpdatePolicy",
    "PEPSUpdateResult",
    "full_update_peps",
    "simple_update_peps",
]
