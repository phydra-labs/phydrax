#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, LinearSystem, solve


ShadowingMemoryMode: TypeAlias = Literal["store", "recompute"]


class ShadowingSolveStatus(IntEnum):
    SUCCESS = 0
    TRAJECTORY_INVALID = 1
    BASIS_RANK_LOST = 2
    LINEAR_SOLVE_FAILED = 3
    NONFINITE = 4
    CONTINUITY_FAILED = 5
    NEUTRAL_CONSTRAINT_FAILED = 6


class ShadowingSolveCost(StrictModule, NonTrainableState):
    method: str = eqx.field(static=True)
    state_dimension: int = eqx.field(static=True)
    unstable_dimension: int = eqx.field(static=True)
    basis_dimension: int = eqx.field(static=True)
    horizon_steps: int = eqx.field(static=True)
    segment_count: int = eqx.field(static=True)
    input_trajectory_bytes: int = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)


class AbstractShadowingSolvePlan(StrictModule, NonTrainableState):
    method: AbstractAttribute[str]
    state_dimension: AbstractAttribute[int]
    unstable_dimension: AbstractAttribute[int]
    basis_dimension: AbstractAttribute[int]
    segment_steps: AbstractAttribute[int]
    segment_count: AbstractAttribute[int]
    regularization: AbstractAttribute[float]
    rank_tolerance: AbstractAttribute[float]
    memory_mode: AbstractAttribute[ShadowingMemoryMode]
    maximum_retained_bytes: AbstractAttribute[int]
    maximum_workspace_bytes: AbstractAttribute[int]
    plan_id: AbstractAttribute[str]

    @property
    def horizon_steps(self) -> int:
        return self.segment_steps * self.segment_count


def validate_shadowing_plan(
    method: str,
    state_dimension: int,
    unstable_dimension: int,
    basis_dimension: int,
    segment_steps: int,
    segment_count: int,
    regularization: float,
    rank_tolerance: float,
    memory_mode: ShadowingMemoryMode,
    maximum_retained_bytes: int,
    maximum_workspace_bytes: int,
    /,
) -> tuple[int, int, int, int, int, float, float, int, int]:
    dimension = int(state_dimension)
    unstable = int(unstable_dimension)
    basis = int(basis_dimension)
    length = int(segment_steps)
    count = int(segment_count)
    penalty = float(regularization)
    rank = float(rank_tolerance)
    retained = int(maximum_retained_bytes)
    workspace = int(maximum_workspace_bytes)
    if (
        not method
        or dimension < 1
        or unstable < 0
        or basis < unstable
        or basis > dimension
        or length < 1
        or count < 1
        or not isfinite(penalty)
        or penalty < 0.0
        or not isfinite(rank)
        or rank <= 0.0
        or memory_mode not in ("store", "recompute")
        or retained <= 0
        or workspace <= 0
    ):
        raise ValueError("Shadowing dimensions, tolerances, or resources are invalid.")
    return dimension, unstable, basis, length, count, penalty, rank, retained, workspace


def shadowing_plan_id(
    method: str,
    values: tuple[int, int, int, int, int, float, float, int, int],
    memory_mode: ShadowingMemoryMode,
    /,
) -> str:
    (
        dimension,
        unstable,
        basis,
        length,
        count,
        regularization,
        rank,
        retained,
        workspace,
    ) = values
    return canonical_fingerprint(
        {
            "kind": "shadowing-solve-plan",
            "method": method,
            "state_dimension": dimension,
            "unstable_dimension": unstable,
            "basis_dimension": basis,
            "segment_steps": length,
            "segment_count": count,
            "regularization": regularization,
            "rank_tolerance": rank,
            "memory_mode": memory_mode,
            "maximum_retained_bytes": retained,
            "maximum_workspace_bytes": workspace,
        }
    )


def orthonormalize_shadowing_basis(
    basis: Array,
    /,
    *,
    rank_tolerance: float,
) -> tuple[Array, Array, Array, Array]:
    columns = int(basis.shape[1])
    if columns == 0:
        zero = jnp.asarray(0.0, dtype=basis.real.dtype)
        return basis, jnp.zeros((0, 0), dtype=basis.dtype), zero, jnp.asarray(True)
    orthonormal, relation = jnp.linalg.qr(basis, mode="reduced")
    diagonal = jnp.diag(relation)
    signs = jnp.where(diagonal < 0.0, -1.0, 1.0).astype(basis.dtype)
    orthonormal = orthonormal * signs[None, :]
    relation = signs[:, None] * relation
    gram = jnp.conj(orthonormal).T @ orthonormal
    defect = jnp.max(
        jnp.abs(gram - jnp.eye(columns, dtype=basis.dtype)),
        initial=0.0,
    )
    singular_values = jnp.linalg.svd(relation, compute_uv=False)
    scale = jnp.maximum(jnp.max(singular_values, initial=0.0), 1.0)
    real_dtype = np.dtype(jnp.empty((), dtype=basis.dtype).real.dtype)
    threshold = (
        max(
            rank_tolerance,
            10.0 * np.finfo(real_dtype).eps * max(basis.shape),
        )
        * scale
    )
    valid = (
        jnp.all(jnp.isfinite(orthonormal))
        & jnp.all(jnp.isfinite(relation))
        & jnp.all(singular_values > threshold)
    )
    return orthonormal, relation, defect, valid


class ReducedShadowingResult(StrictModule):
    coefficients: Array
    multipliers: Array
    continuity_residual: Array
    neutral_residual: Array
    rank: Array
    condition_estimate: Array
    linear_status: Array
    successful: Array


def solve_reduced_shadowing(
    covariances: Array,
    linear_terms: Array,
    relations: Array,
    offsets: Array,
    /,
    *,
    regularization: float,
    neutral_basis: Array | None = None,
    neutral_offset: Array | None = None,
    relation_orientation: Literal["forward", "backward"] = "forward",
    solve_id: str,
) -> ReducedShadowingResult:
    segments, basis_dimension = linear_terms.shape
    coefficient_count = segments * basis_dimension
    continuity_count = max(segments - 1, 0) * basis_dimension
    neutral_count = 0 if neutral_basis is None else 1
    constraint_count = continuity_count + neutral_count
    dtype = linear_terms.dtype
    if relation_orientation not in ("forward", "backward"):
        raise ValueError("Unknown reduced shadowing relation orientation.")
    if coefficient_count == 0:
        return ReducedShadowingResult(
            coefficients=jnp.zeros((segments, 0), dtype=dtype),
            multipliers=jnp.zeros((0,), dtype=dtype),
            continuity_residual=jnp.asarray(0.0, dtype=dtype),
            neutral_residual=jnp.asarray(0.0, dtype=dtype),
            rank=jnp.asarray(0, dtype=jnp.int32),
            condition_estimate=jnp.asarray(1.0, dtype=dtype),
            linear_status=jnp.asarray(0, dtype=jnp.int32),
            successful=jnp.asarray(True),
        )

    hessian = jnp.zeros((coefficient_count, coefficient_count), dtype=dtype)
    gradient = linear_terms.reshape((-1,))
    identity = jnp.eye(basis_dimension, dtype=dtype)
    for segment in range(segments):
        block = slice(segment * basis_dimension, (segment + 1) * basis_dimension)
        hessian = hessian.at[block, block].set(
            covariances[segment] + regularization * identity
        )
    constraints = jnp.zeros((constraint_count, coefficient_count), dtype=dtype)
    constraint_rhs = jnp.zeros((constraint_count,), dtype=dtype)
    for boundary in range(segments - 1):
        row = slice(boundary * basis_dimension, (boundary + 1) * basis_dimension)
        left = slice(boundary * basis_dimension, (boundary + 1) * basis_dimension)
        right = slice(
            (boundary + 1) * basis_dimension,
            (boundary + 2) * basis_dimension,
        )
        if relation_orientation == "forward":
            constraints = constraints.at[row, left].set(-relations[boundary])
            constraints = constraints.at[row, right].set(identity)
        else:
            constraints = constraints.at[row, left].set(identity)
            constraints = constraints.at[row, right].set(-relations[boundary])
        constraint_rhs = constraint_rhs.at[row].set(offsets[boundary])
    if neutral_basis is not None:
        if neutral_basis.shape != (segments, basis_dimension):
            raise ValueError("neutral_basis has an incompatible shape.")
        if neutral_offset is None or jnp.asarray(neutral_offset).shape != ():
            raise ValueError("neutral_offset must be scalar when a constraint is used.")
        constraints = constraints.at[-1].set(neutral_basis.reshape((-1,)))
        constraint_rhs = constraint_rhs.at[-1].set(-jnp.asarray(neutral_offset))

    if constraint_count:
        kkt = jnp.block(
            [
                [hessian, constraints.T],
                [
                    constraints,
                    jnp.zeros((constraint_count, constraint_count), dtype=dtype),
                ],
            ]
        )
        rhs = jnp.concatenate((-gradient, constraint_rhs))
    else:
        kkt = hessian
        rhs = -gradient
    operator = DenseLinearOperator(
        kkt,
        operator_id=canonical_fingerprint(
            {
                "kind": "shadowing-reduced-kkt",
                "solve": solve_id,
                "size": kkt.shape[0],
            }
        ),
    )
    linear = solve(LinearSystem(operator), rhs)
    raw = linear.value[:coefficient_count].reshape((segments, basis_dimension))
    coefficients = jnp.where(linear.successful, raw, jnp.full_like(raw, jnp.nan))
    continuity = (
        jnp.max(
            jnp.abs(
                constraints[:continuity_count] @ coefficients.reshape((-1,))
                - constraint_rhs[:continuity_count]
            ),
            initial=0.0,
        )
        if continuity_count
        else jnp.asarray(0.0, dtype=dtype)
    )
    neutral = (
        jnp.abs(constraints[-1] @ coefficients.reshape((-1,)) - constraint_rhs[-1])
        if neutral_count
        else jnp.asarray(0.0, dtype=dtype)
    )
    return ReducedShadowingResult(
        coefficients=coefficients,
        multipliers=linear.value[coefficient_count:],
        continuity_residual=continuity,
        neutral_residual=neutral,
        rank=linear.diagnostics.rank,
        condition_estimate=linear.diagnostics.condition_estimate,
        linear_status=linear.status,
        successful=linear.successful,
    )


__all__ = [
    "AbstractShadowingSolvePlan",
    "ReducedShadowingResult",
    "ShadowingMemoryMode",
    "ShadowingSolveCost",
    "ShadowingSolveStatus",
]
