#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparedTensorGrid


class IsolatedGravityDiagnostics(StrictModule):
    potential: Array
    acceleration: Array
    source_mass: Array
    force_mean: Array
    finite: Array


class IsolatedCartesianGravityPlan(StrictModule, NonTrainableState):
    grid: PreparedTensorGrid
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    kernel_transform: Array
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        gravitational_constant: float = 1.0,
        softening: float = 1e-3,
    ):
        coupling = float(gravitational_constant)
        epsilon = float(softening)
        if (
            not isinstance(grid, PreparedTensorGrid)
            or any(axis.periodic for axis in grid.structured_axes)
            or coupling <= 0.0
            or epsilon <= 0.0
        ):
            raise ValueError("Isolated Cartesian gravity requires a bounded tensor grid.")
        padded_shape = tuple(2 * count for count in grid.shape)
        coordinates = tuple(
            jnp.where(
                jnp.arange(size) <= size // 2,
                jnp.arange(size),
                jnp.arange(size) - size,
            )
            * float(np.mean(axis.interval_widths))
            for size, axis in zip(padded_shape, grid.structured_axes, strict=True)
        )
        mesh = jnp.meshgrid(*coordinates, indexing="ij")
        radius_squared = sum(component**2 for component in mesh) + epsilon**2
        kernel = -coupling / jnp.sqrt(radius_squared)
        kernel = kernel.at[(0,) * len(padded_shape)].set(0.0)
        self.grid = grid
        self.gravitational_constant = coupling
        self.softening = epsilon
        self.kernel_transform = jnp.fft.fftn(kernel)
        self.padded_shape = padded_shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isolated-cartesian-gravity",
                "grid": grid.prepared_id,
                "gravitational_constant": coupling,
                "softening": epsilon,
            }
        )

    def solve(
        self,
        density: ArrayLike,
        /,
    ) -> tuple[Array, Array, IsolatedGravityDiagnostics]:
        source = jnp.asarray(density)
        if source.shape != self.grid.shape:
            raise ValueError("Isolated gravity density must match the grid shape.")
        padded = jnp.zeros(self.padded_shape, dtype=source.dtype)
        slices = tuple(slice(0, count) for count in self.grid.shape)
        padded = padded.at[slices].set(source * self.grid.quadrature_weights)
        potential_padded = jnp.fft.ifftn(
            jnp.fft.fftn(padded) * self.kernel_transform
        ).real
        potential = potential_padded[slices]
        acceleration_components = []
        for axis, structured_axis in enumerate(self.grid.structured_axes):
            spacing = jnp.asarray(structured_axis.interval_widths)
            mean_spacing = jnp.mean(spacing)
            gradient = (
                jnp.roll(potential, -1, axis=axis) - jnp.roll(potential, 1, axis=axis)
            ) / (2.0 * mean_spacing)
            acceleration_components.append(-gradient)
        acceleration = jnp.stack(acceleration_components, axis=-1)
        mass = jnp.sum(source * self.grid.quadrature_weights)
        force_mean = (
            jnp.sum(
                source[..., None]
                * acceleration
                * self.grid.quadrature_weights[..., None],
                axis=tuple(range(source.ndim)),
            )
            / mass
        )
        diagnostics = IsolatedGravityDiagnostics(
            potential=potential,
            acceleration=acceleration,
            source_mass=mass,
            force_mean=force_mean,
            finite=jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(acceleration)),
        )
        return potential, acceleration, diagnostics


__all__ = ["IsolatedCartesianGravityPlan", "IsolatedGravityDiagnostics"]
