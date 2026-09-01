#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._population import VortexPopulationState


class CompleteVortexRemeshEvidence(StrictModule):
    circulation_residual: Array
    first_moment_residual: Array
    second_moment_residual: Array
    dropped_strength: Array
    obstacle_violation_count: Array
    active_target_count: Array
    finite: Array


class CompleteVortexRemeshResult(StrictModule):
    candidate: VortexPopulationState
    accepted: VortexPopulationState
    evidence: CompleteVortexRemeshEvidence
    successful: Array
    remesh_id: str = eqx.field(static=True)


class CompleteVortexRemeshPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    grid_position: Array
    obstacle_clearance: Callable[[Array], Array] | None
    obstacle_id: str | None = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    remesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        shape: tuple[int, ...],
        /,
        *,
        degree: int = 3,
        boundary: str = "reject",
        periodic: tuple[bool, ...] | None = None,
        obstacle_clearance=None,
        obstacle_id: str | None = None,
    ):
        lower_, upper_ = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        shape_, degree_, dimension = (
            tuple(int(value) for value in shape),
            int(degree),
            int(lower_.size),
        )
        periodic_ = (
            (False,) * dimension
            if periodic is None
            else tuple(bool(value) for value in periodic)
        )
        if (
            dimension not in (2, 3)
            or upper_.shape != lower_.shape
            or len(shape_) != dimension
            or any(value < degree_ + 1 for value in shape_)
            or degree_ not in (1, 2, 3)
            or boundary not in ("reject", "drop")
            or len(periodic_) != dimension
            or np.any(upper_ <= lower_)
        ):
            raise ValueError(
                "Complete remesh geometry/degree/boundary controls are invalid."
            )
        if (obstacle_clearance is None) != (obstacle_id is None) or (
            obstacle_clearance is not None and not callable(obstacle_clearance)
        ):
            raise ValueError(
                "Obstacle clearance requires callable and stable ID together."
            )
        axes = tuple(
            jnp.linspace(
                lower_[axis], upper_[axis], shape_[axis], endpoint=not periodic_[axis]
            )
            for axis in range(dimension)
        )
        mesh = jnp.meshgrid(*axes, indexing="ij")
        self.lower, self.upper, self.shape, self.degree, self.boundary, self.periodic = (
            jnp.asarray(lower_),
            jnp.asarray(upper_),
            shape_,
            degree_,
            boundary,
            periodic_,
        )
        self.grid_position = jnp.stack(
            tuple(component.reshape(-1) for component in mesh), axis=-1
        )
        self.obstacle_clearance, self.obstacle_id, self.dimension = (
            obstacle_clearance,
            obstacle_id,
            dimension,
        )
        self.remesh_id = canonical_fingerprint(
            {
                "kind": "complete-vortex-remesh",
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
                "shape": shape_,
                "degree": degree_,
                "boundary": boundary,
                "periodic": periodic_,
                "obstacle_id": obstacle_id,
            }
        )

    @property
    def capacity(self) -> int:
        return int(np.prod(self.shape))

    def _axis_stencil(self, coordinate: Array, axis: int, /) -> tuple[Array, Array]:
        count = self.shape[axis]
        spacing = (self.upper[axis] - self.lower[axis]) / (
            count if self.periodic[axis] else count - 1
        )
        value = (coordinate - self.lower[axis]) / spacing
        if self.degree == 1:
            base = jnp.floor(value).astype(jnp.int32)
            fraction = value - base
            indices = jnp.stack((base, base + 1), axis=-1)
            weights = jnp.stack((1.0 - fraction, fraction), axis=-1)
        elif self.degree == 2:
            base = jnp.floor(value - 0.5).astype(jnp.int32)
            fraction = value - base
            indices = jnp.stack((base, base + 1, base + 2), axis=-1)
            weights = jnp.stack(
                (
                    0.5 * (1.5 - fraction) ** 2,
                    0.75 - (fraction - 1.0) ** 2,
                    0.5 * (fraction - 0.5) ** 2,
                ),
                axis=-1,
            )
        else:
            integer = jnp.floor(value).astype(jnp.int32)
            fraction = value - integer
            base = integer - 1
            indices = jnp.stack((base, base + 1, base + 2, base + 3), axis=-1)
            weights = jnp.stack(
                (
                    (1.0 - fraction) ** 3 / 6.0,
                    (3.0 * fraction**3 - 6.0 * fraction**2 + 4.0) / 6.0,
                    (-3.0 * fraction**3 + 3.0 * fraction**2 + 3.0 * fraction + 1.0) / 6.0,
                    fraction**3 / 6.0,
                ),
                axis=-1,
            )
        if self.periodic[axis]:
            indices = jnp.mod(indices, count)
            support = jnp.ones(indices.shape, dtype=bool)
        else:
            support = (indices >= 0) & (indices < count)
            indices = jnp.clip(indices, 0, count - 1)
        return indices, jnp.where(support, weights, 0.0)

    def apply(
        self,
        source: VortexPopulationState,
        previous_target: VortexPopulationState | None = None,
        /,
    ) -> CompleteVortexRemeshResult:
        if source.positions.shape[1] != self.dimension:
            raise ValueError("Remesh source dimension is incompatible.")
        if previous_target is not None and previous_target.positions.shape != (
            self.capacity,
            self.dimension,
        ):
            raise ValueError("Previous remesh target capacity is incompatible.")
        stencils = tuple(
            self._axis_stencil(source.positions[:, axis], axis)
            for axis in range(self.dimension)
        )
        offsets = tuple(itertools.product(range(self.degree + 1), repeat=self.dimension))
        target_strength = jnp.zeros(
            (self.capacity,) + source.strength.shape[1:], dtype=source.strength.dtype
        )
        target_volume = jnp.zeros((self.capacity,), dtype=source.volume.dtype)
        target_core_second = jnp.zeros((self.capacity,), dtype=source.core_radius.dtype)
        captured = jnp.zeros((source.positions.shape[0],), dtype=source.positions.dtype)
        for offset in offsets:
            indices = tuple(
                stencils[axis][0][:, offset[axis]] for axis in range(self.dimension)
            )
            weight = jnp.ones((source.positions.shape[0],), dtype=source.positions.dtype)
            for axis in range(self.dimension):
                weight = weight * stencils[axis][1][:, offset[axis]]
            flat = jnp.ravel_multi_index(indices, self.shape, mode="clip")
            active_weight = jnp.where(source.active_mask, weight, 0.0)
            target_strength = target_strength.at[flat].add(
                active_weight.reshape((-1,) + (1,) * (source.strength.ndim - 1))
                * source.strength
            )
            target_volume = target_volume.at[flat].add(active_weight * source.volume)
            target_core_second = target_core_second.at[flat].add(
                active_weight * source.volume * source.core_radius**2
            )
            captured = captured + active_weight
        dropped_mask = source.active_mask & (captured < 1.0 - 1.0e-10)
        target_active = (
            jnp.linalg.norm(target_strength, axis=-1)
            if target_strength.ndim == 2
            else jnp.abs(target_strength)
        ) > 0.0
        target_core = jnp.sqrt(
            target_core_second
            / jnp.maximum(target_volume, jnp.finfo(target_volume.dtype).tiny)
        )
        obstacle_violation = (
            jnp.zeros((self.capacity,), dtype=bool)
            if self.obstacle_clearance is None
            else target_active
            & (jax.vmap(self.obstacle_clearance)(self.grid_position) <= 0.0)
        )
        target_active = target_active & ~obstacle_violation
        target_strength = jnp.where(
            target_active.reshape((-1,) + (1,) * (target_strength.ndim - 1)),
            target_strength,
            0.0,
        )
        target_volume = jnp.where(target_active, target_volume, 1.0)
        target_core = jnp.where(target_active, target_core, 1.0)
        stable_ids = jnp.where(
            target_active, jnp.arange(self.capacity, dtype=jnp.int64), -1
        )
        candidate = VortexPopulationState(
            self.grid_position,
            target_strength,
            target_core,
            target_volume,
            target_active,
            stable_ids,
            jnp.full((self.capacity,), -1, dtype=jnp.int64),
            jnp.zeros((self.capacity,), dtype=jnp.int32),
            jnp.zeros((self.capacity,), dtype=source.age.dtype),
            jnp.asarray(self.capacity, dtype=jnp.int64),
        )
        source_total = jnp.sum(
            jnp.where(
                source.active_mask.reshape((-1,) + (1,) * (source.strength.ndim - 1)),
                source.strength,
                0.0,
            ),
            axis=0,
        )
        target_total = jnp.sum(target_strength, axis=0)
        source_first = jnp.sum(
            source.strength.reshape((source.strength.shape[0], -1))[:, :, None]
            * source.positions[:, None, :],
            axis=0,
        )
        target_first = jnp.sum(
            target_strength.reshape((self.capacity, -1))[:, :, None]
            * self.grid_position[:, None, :],
            axis=0,
        )
        source_second = jnp.sum(
            source.strength.reshape((source.strength.shape[0], -1))[:, :, None]
            * source.positions[:, None, :] ** 2,
            axis=0,
        )
        target_second = jnp.sum(
            target_strength.reshape((self.capacity, -1))[:, :, None]
            * self.grid_position[:, None, :] ** 2,
            axis=0,
        )
        dropped = jnp.sum(
            jnp.where(
                dropped_mask.reshape((-1,) + (1,) * (source.strength.ndim - 1)),
                source.strength,
                0.0,
            ),
            axis=0,
        )
        circulation_residual = target_total + dropped - source_total
        first_residual, second_residual = (
            target_first - source_first,
            target_second - source_second,
        )
        scale = jnp.maximum(jnp.max(jnp.abs(source_total)), 1.0)
        tolerance = 1.0e-8 * scale
        finite = jnp.all(jnp.isfinite(target_strength)) & jnp.all(
            jnp.isfinite(target_core)
        )
        boundary_ok = (
            jnp.asarray(True) if self.boundary == "drop" else ~jnp.any(dropped_mask)
        )
        nonperiodic_axis = jnp.asarray(
            tuple(not value for value in self.periodic),
            dtype=bool,
        )
        first_moment_defect = jnp.max(
            jnp.abs(
                jnp.where(
                    nonperiodic_axis[None, :],
                    first_residual,
                    0.0,
                )
            )
        )
        successful = (
            finite
            & boundary_ok
            & ~jnp.any(obstacle_violation)
            & (jnp.max(jnp.abs(circulation_residual)) <= tolerance)
            & (first_moment_defect <= 10.0 * tolerance)
        )
        accepted = (
            candidate
            if previous_target is None
            else jax.tree_util.tree_map(
                lambda current, previous: jnp.where(successful, current, previous),
                candidate,
                previous_target,
            )
        )
        evidence = CompleteVortexRemeshEvidence(
            circulation_residual,
            first_residual,
            second_residual,
            dropped,
            jnp.sum(obstacle_violation, dtype=jnp.int32),
            jnp.sum(target_active, dtype=jnp.int32),
            finite,
        )
        return CompleteVortexRemeshResult(
            candidate, accepted, evidence, successful, self.remesh_id
        )


__all__ = [
    "CompleteVortexRemeshEvidence",
    "CompleteVortexRemeshPlan",
    "CompleteVortexRemeshResult",
]
