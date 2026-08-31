#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import PreparedTensorGrid
from ..finite_volume._mac_interface_state import MACFreeSurfaceGeometryState


def _central(value, axis, spacing, periodic):
    if periodic:
        return (jnp.roll(value, -1, axis=axis) - jnp.roll(value, 1, axis=axis)) / (
            2.0 * spacing
        )
    forward = jnp.roll(value, -1, axis=axis)
    backward = jnp.roll(value, 1, axis=axis)
    derivative = (forward - backward) / (2.0 * spacing)
    lower = [slice(None)] * value.ndim
    upper = [slice(None)] * value.ndim
    lower[axis], upper[axis] = 0, value.shape[axis] - 1
    derivative = derivative.at[tuple(lower)].set(
        (jnp.take(value, 1, axis=axis) - jnp.take(value, 0, axis=axis)) / spacing
    )
    derivative = derivative.at[tuple(upper)].set(
        (jnp.take(value, -1, axis=axis) - jnp.take(value, -2, axis=axis)) / spacing
    )
    return derivative


class ParticleLevelSetPlan(StrictModule, NonTrainableState):
    """Fixed-band union-of-particle-spheres level-set reconstruction."""

    grid: PreparedTensorGrid
    particle_radius: float = eqx.field(static=True)
    narrow_band_cells: int = eqx.field(static=True)
    minimum_ghost_fraction: float = eqx.field(static=True)
    points: Array
    spacing: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        particle_radius: float,
        /,
        *,
        narrow_band_cells: int = 4,
        minimum_ghost_fraction: float = 1.0e-2,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be PreparedTensorGrid.")
        radius = float(particle_radius)
        band = int(narrow_band_cells)
        minimum = float(minimum_ghost_fraction)
        if radius <= 0.0 or band <= 0 or not 0.0 < minimum <= 1.0:
            raise ValueError("Particle level-set policy is invalid.")
        cells = grid.cells()
        mesh = jnp.meshgrid(*cells.coordinates_by_axis, indexing="ij")
        points = jnp.stack(tuple(value.reshape((-1,)) for value in mesh), axis=-1)
        spacing = tuple(
            float(jnp.min(axis.interval_widths)) for axis in grid.structured_axes
        )
        self.grid = grid
        self.particle_radius = radius
        self.narrow_band_cells = band
        self.minimum_ghost_fraction = minimum
        self.points = points
        self.spacing = spacing
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-level-set",
                "grid": grid.prepared_id,
                "radius": radius,
                "band": band,
                "minimum_ghost_fraction": minimum,
            }
        )

    def evaluate(
        self, position: ArrayLike, active_mask: ArrayLike, /
    ) -> MACFreeSurfaceGeometryState:
        particles = jnp.asarray(position, dtype=self.points.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        if particles.ndim != 2 or particles.shape[1] != len(self.grid.shape):
            raise ValueError("Particle level-set positions have incompatible dimension.")
        if active.shape != (particles.shape[0],):
            raise ValueError("Particle level-set activity must preserve capacity.")
        delta = self.points[:, None, :] - particles[None, :, :]
        for axis, structured_axis in enumerate(self.grid.structured_axes):
            if structured_axis.periodic:
                length = structured_axis.bounds[1] - structured_axis.bounds[0]
                delta = delta.at[..., axis].set(
                    delta[..., axis] - length * jnp.round(delta[..., axis] / length)
                )
        distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1)) - self.particle_radius
        phi_flat = jnp.min(jnp.where(active[None, :], distance, jnp.inf), axis=1)
        empty = ~jnp.any(active)
        maximum_spacing = max(self.spacing)
        phi_flat = jnp.where(
            empty,
            self.narrow_band_cells * maximum_spacing,
            phi_flat,
        )
        shape = self.grid.cells().shape
        phi = phi_flat.reshape(shape)
        band_width = self.narrow_band_cells * maximum_spacing
        valid_band = jnp.abs(phi) <= band_width
        liquid = phi <= 0.0
        cell_fraction = jnp.clip(0.5 - phi / (2.0 * maximum_spacing), 0.0, 1.0)
        gradients = tuple(
            _central(phi, axis, self.spacing[axis], grid_axis.periodic)
            for axis, grid_axis in enumerate(self.grid.structured_axes)
        )
        gradient = jnp.stack(gradients, axis=-1)
        norm = jnp.sqrt(jnp.sum(gradient * gradient, axis=-1))
        normal = gradient / jnp.maximum(norm[..., None], 1.0e-30)
        curvature = sum(
            _central(normal[..., axis], axis, self.spacing[axis], grid_axis.periodic)
            for axis, grid_axis in enumerate(self.grid.structured_axes)
        )
        face_fraction = []
        ghost_fraction = []
        interfaces = []
        clamped = jnp.asarray(0, dtype=jnp.int32)
        minimum_unclamped = jnp.asarray(jnp.inf, dtype=phi.dtype)
        for axis in range(len(shape)):
            neighbor_phi = jnp.roll(phi, -1, axis=axis)
            neighbor_fraction = jnp.roll(cell_fraction, -1, axis=axis)
            face_fraction.append(0.5 * (cell_fraction + neighbor_fraction))
            crossing = (phi <= 0.0) != (neighbor_phi <= 0.0)
            raw = jnp.abs(phi) / jnp.maximum(
                jnp.abs(phi) + jnp.abs(neighbor_phi), 1.0e-30
            )
            minimum_unclamped = jnp.minimum(
                minimum_unclamped,
                jnp.min(jnp.where(crossing, raw, jnp.inf)),
            )
            clamped = clamped + jnp.sum(
                crossing & (raw < self.minimum_ghost_fraction), dtype=jnp.int32
            )
            ghost_fraction.append(
                jnp.where(
                    crossing,
                    jnp.maximum(raw, self.minimum_ghost_fraction),
                    1.0,
                )
            )
            interfaces.append(crossing)
        finite = (
            jnp.all(jnp.isfinite(phi))
            & jnp.all(jnp.isfinite(normal))
            & jnp.all(jnp.isfinite(curvature))
        )
        return MACFreeSurfaceGeometryState(
            phi,
            liquid,
            valid_band,
            cell_fraction,
            tuple(face_fraction),
            tuple(ghost_fraction),
            tuple(interfaces),
            normal,
            curvature,
            minimum_unclamped,
            clamped,
            finite,
            finite & ~empty,
            self.plan_id,
        )


__all__ = ["MACFreeSurfaceGeometryState", "ParticleLevelSetPlan"]
