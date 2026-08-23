#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class DynamicalMapSeriesPhysicality(StrictModule):
    cp_margins: Array
    trace_preservation_residuals: Array
    intermediate_cp_margins: Array
    intermediate_condition_numbers: Array
    intermediate_solve_residuals: Array
    cp_valid: Array
    cp_divisible: Array
    valid: Array

    def __init__(
        self,
        cp_margins: ArrayLike,
        trace_preservation_residuals: ArrayLike,
        intermediate_cp_margins: ArrayLike,
        intermediate_condition_numbers: ArrayLike,
        intermediate_solve_residuals: ArrayLike,
        /,
    ):
        self.cp_margins = jnp.asarray(cp_margins)
        self.trace_preservation_residuals = jnp.asarray(trace_preservation_residuals)
        self.intermediate_cp_margins = jnp.asarray(intermediate_cp_margins)
        self.intermediate_condition_numbers = jnp.asarray(intermediate_condition_numbers)
        self.intermediate_solve_residuals = jnp.asarray(intermediate_solve_residuals)
        self.cp_valid = jnp.all(self.cp_margins >= -1e-8) & jnp.all(
            self.trace_preservation_residuals <= 1e-8
        )
        self.cp_divisible = jnp.all(self.intermediate_cp_margins >= -1e-8)
        self.valid = (
            jnp.all(jnp.isfinite(self.cp_margins))
            & jnp.all(jnp.isfinite(self.trace_preservation_residuals))
            & jnp.all(jnp.isfinite(self.intermediate_condition_numbers))
            & jnp.all(jnp.isfinite(self.intermediate_solve_residuals))
        )


def _choi_evidence(superoperator: Array, dimension: int, /) -> tuple[Array, Array]:
    size = int(dimension)
    choi = jnp.zeros((size, size, size, size), dtype=superoperator.dtype)
    for row in range(size):
        for column in range(size):
            basis = (
                jnp.zeros((size, size), dtype=superoperator.dtype)
                .at[row, column]
                .set(1.0)
            )
            output = (superoperator @ basis.reshape(-1)).reshape((size, size))
            choi = choi.at[row, :, column, :].set(output)
    flat = choi.reshape((size * size, size * size))
    flat = 0.5 * (flat + jnp.conj(flat.T))
    partial = jnp.trace(choi, axis1=1, axis2=3)
    return (
        jnp.min(jnp.linalg.eigvalsh(flat)),
        jnp.max(jnp.abs(partial - jnp.eye(size, dtype=superoperator.dtype))),
    )


def analyze_dynamical_map_series(
    superoperators: ArrayLike,
    dimension: int,
    /,
    *,
    condition_limit: float = 1e12,
) -> DynamicalMapSeriesPhysicality:
    maps = jnp.asarray(superoperators)
    size = int(dimension) ** 2
    if maps.ndim != 3 or maps.shape[-2:] != (size, size):
        raise ValueError("Dynamical maps require shape (time,d²,d²).")
    direct = [_choi_evidence(mapping, dimension) for mapping in maps]
    intermediate_margins = []
    conditions = []
    solve_residuals = []
    for previous, current in zip(maps[:-1], maps[1:], strict=True):
        condition = jnp.linalg.cond(previous)
        conditions.append(condition)
        intermediate = jnp.linalg.solve(previous.T, current.T).T
        solve_residual = jnp.linalg.norm(intermediate @ previous - current)
        solve_residuals.append(
            jnp.where(condition <= condition_limit, solve_residual, jnp.nan)
        )
        margin, _ = _choi_evidence(intermediate, dimension)
        intermediate_margins.append(
            jnp.where(condition <= condition_limit, margin, jnp.nan)
        )
    return DynamicalMapSeriesPhysicality(
        jnp.stack([item[0] for item in direct]),
        jnp.stack([item[1] for item in direct]),
        jnp.stack(intermediate_margins) if intermediate_margins else jnp.zeros((0,)),
        jnp.stack(conditions) if conditions else jnp.zeros((0,)),
        jnp.stack(solve_residuals) if solve_residuals else jnp.zeros((0,)),
    )


def trace_distance(left: ArrayLike, right: ArrayLike, /) -> Array:
    difference = jnp.asarray(left) - jnp.asarray(right)
    singular_values = jnp.linalg.svd(difference, compute_uv=False)
    return 0.5 * jnp.sum(singular_values)


def blp_information_backflow(
    left_trajectory: ArrayLike,
    right_trajectory: ArrayLike,
    /,
) -> Array:
    left = jnp.asarray(left_trajectory)
    right = jnp.asarray(right_trajectory)
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("BLP trajectories must share shape (time,n,n).")
    distances = jnp.stack(
        [
            trace_distance(left_state, right_state)
            for left_state, right_state in zip(left, right, strict=True)
        ]
    )
    increments = distances[1:] - distances[:-1]
    return jnp.sum(jnp.maximum(increments, 0.0))


__all__ = [
    "DynamicalMapSeriesPhysicality",
    "analyze_dynamical_map_series",
    "blp_information_backflow",
    "trace_distance",
]
