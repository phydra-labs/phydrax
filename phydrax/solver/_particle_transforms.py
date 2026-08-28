#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..discretization.particle import PreparedWeaklyCompressibleSPHDynamics
from ..discretization.particle._pairwise import particle_pair_geometry
from ..discretization.particle._stabilization import shepard_renormalized_density
from ._fixed_step import AbstractAcceptedStepTransform, AcceptedStepTransformResult


class ShepardDensityRenormalizationTransform(
    AbstractAcceptedStepTransform, NonTrainableState
):
    dynamics: PreparedWeaklyCompressibleSPHDynamics
    apply_every_steps: int = eqx.field(static=True)
    first_step: int = eqx.field(static=True)
    maximum_relative_correction: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedWeaklyCompressibleSPHDynamics,
        /,
        *,
        apply_every_steps: int,
        first_step: int | None = None,
        maximum_relative_correction: float = 0.5,
    ):
        if not isinstance(dynamics, PreparedWeaklyCompressibleSPHDynamics):
            raise TypeError("dynamics must be PreparedWeaklyCompressibleSPHDynamics.")
        if not dynamics.state_layout.density_evolved:
            raise ValueError("Density renormalization requires evolved density.")
        every = int(apply_every_steps)
        first = every if first_step is None else int(first_step)
        maximum = float(maximum_relative_correction)
        if every <= 0 or first <= 0 or maximum <= 0.0:
            raise ValueError("Renormalization schedule and correction bound are invalid.")
        self.dynamics = dynamics
        self.apply_every_steps = every
        self.first_step = first
        self.maximum_relative_correction = maximum
        self.transform_id = canonical_fingerprint(
            {
                "kind": "shepard-density-renormalization-transform",
                "dynamics": dynamics.prepared_id,
                "apply_every_steps": every,
                "first_step": first,
                "maximum_relative_correction": maximum,
            }
        )

    def apply(
        self,
        step_index: Array,
        time: Array,
        previous_state: Array,
        candidate_state: Array,
        args: Any,
        /,
    ) -> AcceptedStepTransformResult:
        del time, previous_state, args
        one_based = step_index + 1
        scheduled = (one_based >= self.first_step) & (
            (one_based - self.first_step) % self.apply_every_steps == 0
        )

        def apply_transform(state):
            position, velocity, density = self.dynamics.state_layout.unpack(state)
            neighborhood = self.dynamics.neighborhood.build(position)
            position = neighborhood.require_success(position)
            geometry = particle_pair_geometry(
                position,
                neighborhood.pair_relation,
                box=self.dynamics.neighborhood.box,
            )
            physical = self.dynamics._physical_pair_mask(geometry)
            candidate, local_success = shepard_renormalized_density(
                self.dynamics.particles,
                density,
                neighborhood.pair_relation,
                geometry,
                physical,
                self.dynamics.method.kernel,
                self.dynamics.method.smoothing_length,
                self.dynamics.execution,
            )
            relative = jnp.abs(candidate - density) / jnp.maximum(jnp.abs(density), 1e-14)
            within_bound = jnp.all(
                jnp.where(
                    self.dynamics.particles.active_mask,
                    relative <= self.maximum_relative_correction,
                    True,
                )
            )
            successful = (
                jnp.all(
                    jnp.where(self.dynamics.particles.active_mask, local_success, True)
                )
                & within_bound
            )
            transformed = self.dynamics.state_layout.pack(position, velocity, candidate)
            norm = jnp.sqrt(jnp.sum((candidate - density) ** 2))
            return AcceptedStepTransformResult(
                transformed, jnp.asarray(True), successful, norm
            )

        def skip(state):
            return AcceptedStepTransformResult(
                state,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.zeros((), dtype=state.dtype),
            )

        return jax.lax.cond(scheduled, apply_transform, skip, candidate_state)


__all__ = ["ShepardDensityRenormalizationTransform"]
