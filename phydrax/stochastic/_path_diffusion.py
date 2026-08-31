#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._gaussian_diffusion import AbstractGaussianDiffusion
from ._subspace_diffusion import AffineSubspaceLayout, SubspaceGaussianDiffusion


PathScoreDependency: TypeAlias = Literal["global", "causal"]


class TrajectoryEventLayout(StrictModule):
    """Fixed-grid trajectory event represented by explicit temporal coefficients."""

    times: Array
    valid_time: Array
    coefficient_layout: AffineSubspaceLayout
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_times: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        state_shape,
        basis: ArrayLike,
        /,
        *,
        origin: ArrayLike | None = None,
        valid_time: ArrayLike | None = None,
        layout_id: str | None = None,
    ):
        grid = jnp.asarray(times, dtype=float)
        shape = tuple(int(size) for size in state_shape)
        if grid.ndim != 1 or int(grid.size) < 2 or bool(jnp.any(jnp.diff(grid) <= 0.0)):
            raise ValueError("Trajectory times must be a strictly increasing vector.")
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("state_shape must contain positive dimensions.")
        state_size = prod(shape)
        event_shape = (int(grid.size),) + shape
        raw_matrix = jnp.asarray(basis)
        raw_center = (
            jnp.zeros(event_shape, dtype=raw_matrix.dtype)
            if origin is None
            else jnp.asarray(origin).reshape(event_shape)
        )
        if jnp.iscomplexobj(raw_center) or jnp.iscomplexobj(raw_matrix):
            raise TypeError("Trajectory coefficient layouts require real coordinates.")
        dtype = jnp.result_type(raw_center.dtype, raw_matrix.dtype)
        if not jnp.issubdtype(dtype, jnp.inexact):
            dtype = jnp.dtype(float)
        center = raw_center.astype(dtype)
        matrix = raw_matrix.astype(dtype)
        if matrix.ndim != 2 or matrix.shape[0] != int(grid.size) * state_size:
            raise ValueError("Trajectory basis has an incompatible flattened event size.")
        mask = (
            jnp.ones(grid.shape, dtype=bool)
            if valid_time is None
            else jnp.asarray(valid_time, dtype=bool)
        )
        if mask.shape != grid.shape or not bool(mask[0]):
            raise ValueError("valid_time must match times and include the initial node.")
        if bool(jnp.any(mask & (jnp.cumsum(~mask) > 0))):
            raise ValueError("valid_time must be one contiguous prefix.")
        active_rows = jnp.repeat(mask, state_size)
        if bool(jnp.any(matrix[~active_rows] != 0.0)):
            raise ValueError("Trajectory basis must vanish at padded time nodes.")
        weights = jnp.repeat(
            jnp.where(mask, _time_weights(grid), 1.0),
            state_size,
        )
        identifier = layout_id or canonical_fingerprint(
            {
                "kind": "trajectory-event-layout",
                "times": grid.tolist(),
                "state_shape": list(shape),
                "rank": int(matrix.shape[1]),
            }
        )
        self.times = grid
        self.valid_time = mask
        self.coefficient_layout = AffineSubspaceLayout(
            center,
            matrix,
            event_shape=event_shape,
            quadrature_weights=weights.reshape(event_shape),
            layout_id=identifier,
        )
        self.state_shape = shape
        self.num_times = int(grid.size)
        self.state_size = state_size
        self.layout_id = identifier

    @classmethod
    def from_increments(cls, times: ArrayLike, state_shape, /):
        grid = jnp.asarray(times, dtype=float)
        if (
            grid.ndim != 1
            or int(grid.size) < 2
            or bool(jnp.any(jnp.diff(grid) <= 0.0))
        ):
            raise ValueError("Trajectory times must be a strictly increasing vector.")
        size = prod(tuple(state_shape))
        intervals = int(grid.size) - 1
        cumulative = jnp.tril(jnp.ones((int(grid.size), intervals)), k=-1)
        scale = jnp.sqrt(jnp.diff(grid))
        temporal = cumulative * scale[None, :]
        basis = jnp.kron(temporal, jnp.eye(size))
        return cls(grid, state_shape, basis)

    def coefficients(self, trajectory: ArrayLike, /):
        return self.coefficient_layout.project(trajectory)

    def synthesize(self, coefficients: ArrayLike, /):
        return self.coefficient_layout.synthesize(coefficients)


def _time_weights(times: Array) -> Array:
    intervals = jnp.diff(times)
    weights = jnp.zeros_like(times)
    weights = weights.at[0].set(0.5 * intervals[0])
    weights = weights.at[-1].set(0.5 * intervals[-1])
    if int(times.size) > 2:
        weights = weights.at[1:-1].set(0.5 * (intervals[:-1] + intervals[1:]))
    return weights


class PathCoefficientDiffusion(StrictModule):
    """Gaussian diffusion in a trajectory basis or innovation coordinate space."""

    layout: TrajectoryEventLayout
    coefficient_process: AbstractGaussianDiffusion
    subspace_process: SubspaceGaussianDiffusion
    score_dependency: PathScoreDependency = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: TrajectoryEventLayout,
        coefficient_process: AbstractGaussianDiffusion,
        /,
        *,
        score_dependency: PathScoreDependency = "global",
    ):
        if not isinstance(layout, TrajectoryEventLayout):
            raise TypeError("layout must be a TrajectoryEventLayout.")
        if coefficient_process.state_shape != (layout.coefficient_layout.rank,):
            raise ValueError("Coefficient process dimension must equal trajectory basis rank.")
        if score_dependency not in ("global", "causal"):
            raise ValueError("score_dependency must be 'global' or 'causal'.")
        identifier = canonical_fingerprint(
            {
                "kind": "path-coefficient-diffusion",
                "layout_id": layout.layout_id,
                "coefficient_process_id": coefficient_process.process_id,
                "score_dependency": score_dependency,
            }
        )
        self.layout = layout
        self.coefficient_process = coefficient_process
        self.subspace_process = SubspaceGaussianDiffusion(
            layout.coefficient_layout,
            coefficient_process,
            process_id=identifier,
        )
        self.score_dependency = score_dependency
        self.process_id = identifier

    def perturb(self, key: Key[Array, ""], trajectory: ArrayLike, /, *, time):
        return self.subspace_process.perturb(key, trajectory, time=time)

    def conditional_coefficient_score(self, perturbed, clean, /, *, time):
        return self.subspace_process.conditional_coefficient_score(
            perturbed,
            clean,
            time=time,
        )

    def require_causal_mask(self, dependency_mask: ArrayLike, /) -> None:
        mask = jnp.asarray(dependency_mask, dtype=bool)
        expected = (self.layout.num_times, self.layout.num_times)
        if mask.shape != expected:
            raise ValueError(f"dependency_mask must have shape {expected}.")
        if self.score_dependency == "causal" and bool(jnp.any(jnp.triu(mask, k=1))):
            raise ValueError("Causal path scores cannot depend on future trajectory nodes.")


__all__ = ["PathCoefficientDiffusion", "PathScoreDependency", "TrajectoryEventLayout"]
