#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._assembly import (
    ParticleAssemblyPlan,
    ParticleInteractionLedger,
    ParticlePopulation,
)
from ._bipartite_neighborhood import (
    BipartiteNeighborhoodState,
    PreparedDenseBipartiteParticleNeighborhood,
)
from ._pairwise import ParticleBox
from ._wcsph import PreparedWeaklyCompressibleSPHDynamics


class PhaseDefinition(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    dynamics: PreparedWeaklyCompressibleSPHDynamics
    phase_id: str = eqx.field(static=True)

    def __init__(self, name: str, dynamics: PreparedWeaklyCompressibleSPHDynamics, /):
        name_ = str(name)
        if not name_ or not isinstance(dynamics, PreparedWeaklyCompressibleSPHDynamics):
            raise ValueError("PhaseDefinition requires a name and WCSPH dynamics.")
        self.name = name_
        self.dynamics = dynamics
        self.phase_id = canonical_fingerprint(
            {"kind": "sph-phase", "name": name_, "dynamics": dynamics.prepared_id}
        )


class MultiphaseWCSPHPlan(StrictModule, NonTrainableState):
    surface_tension: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, surface_tension: float = 0.0):
        tension = float(surface_tension)
        if not np.isfinite(tension) or tension < 0.0:
            raise ValueError("surface_tension must be finite and non-negative.")
        self.surface_tension = tension
        self.plan_id = canonical_fingerprint(
            {"kind": "multiphase-wcsph", "surface_tension": tension}
        )


class MultiphaseInteractionResult(StrictModule):
    target_force: Array
    source_force: Array
    pressure_force: Array
    viscous_force: Array
    surface_tension_force: Array
    ledger: ParticleInteractionLedger


def multiphase_interface_interaction(
    plan: MultiphaseWCSPHPlan,
    target: PhaseDefinition,
    source: PhaseDefinition,
    relation_state: BipartiteNeighborhoodState,
    target_state: ArrayLike,
    source_state: ArrayLike,
    /,
    *,
    box: ParticleBox | None = None,
) -> MultiphaseInteractionResult:
    if not relation_state.successful:
        raise ValueError("Multiphase relation is unsuccessful.")
    target_position, target_velocity, target_density_state = (
        target.dynamics.state_layout.unpack(target_state)
    )
    source_position, source_velocity, source_density_state = (
        source.dynamics.state_layout.unpack(source_state)
    )
    if target_density_state is None or source_density_state is None:
        raise ValueError("Multiphase WCSPH requires continuity-density phases.")
    target_density = target_density_state
    source_density = source_density_state
    target_pressure = target.dynamics.material.pressure(target_density)
    source_pressure = source.dynamics.material.pressure(source_density)
    relation = relation_state.relation
    ti = relation.target_indices
    sj = relation.source_indices
    displacement = target_position[ti] - source_position[sj]
    if box is not None:
        displacement = box.minimum_image(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    h = 0.5 * (
        target.dynamics.method.smoothing_length + source.dynamics.method.smoothing_length
    )
    kernel = target.dynamics.method.kernel
    valid = relation.valid & (distance < kernel.support_radius(h))
    gradient = kernel.gradient(displacement, distance, h)
    target_volume = target.dynamics.particles.safe_masses / target_density
    source_volume = source.dynamics.particles.safe_masses / source_density
    pressure_average = (
        source_density[sj] * target_pressure[ti]
        + target_density[ti] * source_pressure[sj]
    ) / (target_density[ti] + source_density[sj])
    pressure_pair = (
        -(target_volume[ti] ** 2 + source_volume[sj] ** 2)[:, None]
        * pressure_average[:, None]
        * gradient
    )
    target_nu = (
        0.0
        if target.dynamics.method.physical_viscosity is None
        else target.dynamics.method.physical_viscosity.kinematic_viscosity
    )
    source_nu = (
        0.0
        if source.dynamics.method.physical_viscosity is None
        else source.dynamics.method.physical_viscosity.kinematic_viscosity
    )
    eta_target = target_density[ti] * target_nu
    eta_source = source_density[sj] * source_nu
    eta_harmonic = (
        2.0
        * eta_target
        * eta_source
        / jnp.where(eta_target + eta_source > 0.0, eta_target + eta_source, 1.0)
    )
    velocity_difference = target_velocity[ti] - source_velocity[sj]
    radial = jnp.sum(displacement * gradient, axis=-1)
    viscous_pair = (
        (target_volume[ti] ** 2 + source_volume[sj] ** 2)
        * eta_harmonic
        * radial
        / (distance**2 + 0.01 * h**2)
    )[:, None] * velocity_difference
    surface_pair = (
        -plan.surface_tension
        * (target_volume[ti] * source_volume[sj])[:, None]
        * gradient
    )
    total_pair = jnp.where(
        valid[:, None], pressure_pair + viscous_pair + surface_pair, 0.0
    )
    pressure_pair = jnp.where(valid[:, None], pressure_pair, 0.0)
    viscous_pair = jnp.where(valid[:, None], viscous_pair, 0.0)
    surface_pair = jnp.where(valid[:, None], surface_pair, 0.0)
    target_force = jnp.zeros_like(target_position).at[ti].add(total_pair)
    source_force = jnp.zeros_like(source_position).at[sj].add(-total_pair)
    ledger = ParticleInteractionLedger.from_forces(
        target_force,
        source_force,
        target_velocity,
        source_velocity,
        jnp.sum(valid),
    )
    return MultiphaseInteractionResult(
        target_force,
        source_force,
        jnp.zeros_like(target_position).at[ti].add(pressure_pair),
        jnp.zeros_like(target_position).at[ti].add(viscous_pair),
        jnp.zeros_like(target_position).at[ti].add(surface_pair),
        ledger,
    )


class MultiphaseSPHDiagnostics(StrictModule):
    interaction: ParticleInteractionLedger
    total_momentum_rate: Array
    interface_pair_count: Array


class PreparedMultiphaseWCSPHDynamics(StrictModule, NonTrainableState):
    target: PhaseDefinition
    source: PhaseDefinition
    interaction_plan: MultiphaseWCSPHPlan
    relation: PreparedDenseBipartiteParticleNeighborhood
    assembly: ParticleAssemblyPlan
    box: ParticleBox | None
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        target: PhaseDefinition,
        source: PhaseDefinition,
        interaction_plan: MultiphaseWCSPHPlan,
        relation: PreparedDenseBipartiteParticleNeighborhood,
        /,
        *,
        box: ParticleBox | None = None,
    ):
        self.target = target
        self.source = source
        self.interaction_plan = interaction_plan
        self.relation = relation
        self.box = box
        target_population = ParticlePopulation(
            target.name,
            target.dynamics.particles,
            role="material-phase",
            state_shape=target.dynamics.state_layout.shape,
            population_id=target.phase_id,
        )
        source_population = ParticlePopulation(
            source.name,
            source.dynamics.particles,
            role="material-phase",
            state_shape=source.dynamics.state_layout.shape,
            population_id=source.phase_id,
        )
        self.assembly = ParticleAssemblyPlan((target_population, source_population))
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-multiphase-wcsph",
                "target": target.phase_id,
                "source": source.phase_id,
                "interaction": interaction_plan.plan_id,
                "relation": relation.prepared_id,
            }
        )

    def pack(self, target_state: ArrayLike, source_state: ArrayLike, /) -> Array:
        return self.assembly.state_layout.pack((target_state, source_state))

    def unpack(self, state: ArrayLike, /) -> tuple[Array, Array]:
        target, source = self.assembly.state_layout.unpack(state)
        return target, source

    def __call__(self, time: Array, state: Array, args=None, /) -> Array:
        target_state, source_state = self.unpack(state)
        target_rate = self.target.dynamics(time, target_state, args)
        source_rate = self.source.dynamics(time, source_state, args)
        target_position, _, _ = self.target.dynamics.state_layout.unpack(target_state)
        source_position, _, _ = self.source.dynamics.state_layout.unpack(source_state)
        relation_state = self.relation.build(target_position, source_position)
        interaction = multiphase_interface_interaction(
            self.interaction_plan,
            self.target,
            self.source,
            relation_state,
            target_state,
            source_state,
            box=self.box,
        )
        tq, tv, trho = self.target.dynamics.state_layout.unpack_rate(target_rate)
        sq, sv, srho = self.source.dynamics.state_layout.unpack_rate(source_rate)
        tv = (
            tv
            + interaction.target_force
            / self.target.dynamics.particles.safe_masses[:, None]
        )
        sv = (
            sv
            + interaction.source_force
            / self.source.dynamics.particles.safe_masses[:, None]
        )
        return self.pack(
            self.target.dynamics.state_layout.pack_rate(tq, tv, trho),
            self.source.dynamics.state_layout.pack_rate(sq, sv, srho),
        )

    def diagnostics(
        self, time: Array, state: Array, args=None, /
    ) -> MultiphaseSPHDiagnostics:
        target_state, source_state = self.unpack(state)
        target_position, _, _ = self.target.dynamics.state_layout.unpack(target_state)
        source_position, _, _ = self.source.dynamics.state_layout.unpack(source_state)
        relation_state = self.relation.build(target_position, source_position)
        interaction = multiphase_interface_interaction(
            self.interaction_plan,
            self.target,
            self.source,
            relation_state,
            target_state,
            source_state,
            box=self.box,
        )
        return MultiphaseSPHDiagnostics(
            interaction.ledger,
            interaction.ledger.action_reaction_defect,
            interaction.ledger.pair_count,
        )


__all__ = [
    "MultiphaseInteractionResult",
    "MultiphaseSPHDiagnostics",
    "MultiphaseWCSPHPlan",
    "PhaseDefinition",
    "PreparedMultiphaseWCSPHDynamics",
    "multiphase_interface_interaction",
]
