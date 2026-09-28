#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparedTensorGrid
from ..operators.integral._free_space_convolution import FreeSpaceConvolutionPlan


class IsolatedGravityDiagnostics(StrictModule):
    potential: Array
    acceleration: Array
    source_mass: Array
    force_mean: Array
    finite: Array


class IsolatedCartesianGravityPlan(StrictModule, NonTrainableState):
    """Softened isolated (open-boundary) Newtonian gravity on a bounded grid.

    The potential is ``G`` times the Hockney doubled-grid convolution of the
    density with the point-sampled softened kernel ``−1/√(r² + softening²)``
    (:class:`FreeSpaceConvolutionPlan` with ``"newton-softened"``); the
    acceleration is its centered difference.
    """

    grid: PreparedTensorGrid
    gravitational_constant: float = eqx.field(static=True)
    softening: float = eqx.field(static=True)
    convolution: FreeSpaceConvolutionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        gravitational_constant: float = 1.0,
        softening: float = 1e-3,
    ) -> None:
        coupling = float(gravitational_constant)
        epsilon = float(softening)
        if (
            not isinstance(grid, PreparedTensorGrid)
            or any(axis.periodic for axis in grid.structured_axes)
            or coupling <= 0.0
            or epsilon <= 0.0
        ):
            raise ValueError("Isolated Cartesian gravity requires a bounded tensor grid.")
        self.grid = grid
        self.gravitational_constant = coupling
        self.softening = epsilon
        self.convolution = FreeSpaceConvolutionPlan(
            "newton-softened", grid, softening=epsilon
        )
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
        potential = self.gravitational_constant * self.convolution.convolve(source).field
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
