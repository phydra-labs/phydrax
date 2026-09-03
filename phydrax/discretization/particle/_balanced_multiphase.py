#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._bipartite_neighborhood import BipartiteNeighborhoodState
from ._multiphase import (
    multiphase_interface_interaction,
    MultiphaseInteractionResult,
    MultiphaseWCSPHPlan,
    PhaseDefinition,
)
from ._pairwise import ParticleBox
from ._production_boundaries import ContactAnglePlan


class PhaseInterfaceGeometryState(StrictModule):
    target_color: Array
    source_color: Array
    target_normal: Array
    source_normal: Array
    target_curvature: Array
    source_curvature: Array
    target_delta: Array
    source_delta: Array
    confidence: Array


def corrected_phase_interface_geometry(
    target: PhaseDefinition,
    source: PhaseDefinition,
    relation_state: BipartiteNeighborhoodState,
    target_state: ArrayLike,
    source_state: ArrayLike,
    /,
    *,
    box: ParticleBox | None = None,
) -> PhaseInterfaceGeometryState:
    tq, _, trho = target.dynamics.state_layout.unpack(target_state)
    sq, _, srho = source.dynamics.state_layout.unpack(source_state)
    relation = relation_state.relation
    ti = relation.target_indices
    sj = relation.source_indices
    displacement = tq[ti] - sq[sj]
    if box is not None:
        displacement = box.minimum_image(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    h = 0.5 * (
        target.dynamics.method.smoothing_length + source.dynamics.method.smoothing_length
    )
    kernel = target.dynamics.method.kernel
    valid = relation.valid & (distance < kernel.support_radius(h))
    gradient = kernel.gradient(displacement, distance, h)
    target_volume = target.dynamics.particles.safe_masses / trho
    source_volume = source.dynamics.particles.safe_masses / srho
    target_gradient = (
        jnp.zeros_like(tq)
        .at[ti]
        .add(jnp.where(valid[:, None], source_volume[sj, None] * gradient, 0.0))
    )
    source_gradient = (
        jnp.zeros_like(sq)
        .at[sj]
        .add(jnp.where(valid[:, None], target_volume[ti, None] * gradient, 0.0))
    )
    target_delta = jnp.sqrt(jnp.sum(target_gradient**2, axis=-1))
    source_delta = jnp.sqrt(jnp.sum(source_gradient**2, axis=-1))
    target_normal = (
        target_gradient / jnp.where(target_delta > 0.0, target_delta, 1.0)[:, None]
    )
    source_normal = (
        -source_gradient / jnp.where(source_delta > 0.0, source_delta, 1.0)[:, None]
    )
    target_color = jnp.clip(target_delta * h, 0.0, 1.0)
    source_color = jnp.clip(source_delta * h, 0.0, 1.0)
    confidence = jnp.minimum(
        jnp.max(target_color, initial=0.0), jnp.max(source_color, initial=0.0)
    )
    return PhaseInterfaceGeometryState(
        target_color,
        source_color,
        target_normal,
        source_normal,
        jnp.zeros_like(target_delta),
        jnp.zeros_like(source_delta),
        target_delta,
        source_delta,
        confidence,
    )


class ContinuumSurfaceStressPlan(StrictModule, NonTrainableState):
    surface_tension: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, surface_tension: float, /):
        tension = float(surface_tension)
        if tension < 0.0 or not np.isfinite(tension):
            raise ValueError("surface_tension must be finite and non-negative.")
        self.surface_tension = tension
        self.plan_id = canonical_fingerprint(
            {"kind": "continuum-surface-stress", "surface_tension": tension}
        )


class BalancedInterfaceForcePlan(StrictModule, NonTrainableState):
    pressure_plan: MultiphaseWCSPHPlan
    surface_stress: ContinuumSurfaceStressPlan
    contact_angle: ContactAnglePlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pressure_plan: MultiphaseWCSPHPlan,
        surface_stress: ContinuumSurfaceStressPlan,
        /,
        *,
        contact_angle: ContactAnglePlan | None = None,
    ):
        self.pressure_plan = pressure_plan
        self.surface_stress = surface_stress
        self.contact_angle = contact_angle
        self.plan_id = canonical_fingerprint(
            {
                "kind": "balanced-interface-force",
                "pressure": pressure_plan.plan_id,
                "surface_stress": surface_stress.plan_id,
                "contact_angle": None if contact_angle is None else contact_angle.plan_id,
            }
        )


class BalancedInterfaceResult(StrictModule):
    base: MultiphaseInteractionResult
    target_surface_force: Array
    source_surface_force: Array
    total_action_reaction_defect: Array
    surface_power: Array


def balanced_interface_force(
    plan: BalancedInterfaceForcePlan,
    target: PhaseDefinition,
    source: PhaseDefinition,
    relation_state: BipartiteNeighborhoodState,
    target_state: ArrayLike,
    source_state: ArrayLike,
    /,
    *,
    box: ParticleBox | None = None,
) -> BalancedInterfaceResult:
    base = multiphase_interface_interaction(
        plan.pressure_plan,
        target,
        source,
        relation_state,
        target_state,
        source_state,
        box=box,
    )
    geometry = corrected_phase_interface_geometry(
        target, source, relation_state, target_state, source_state, box=box
    )
    tq, tv, trho = target.dynamics.state_layout.unpack(target_state)
    sq, sv, srho = source.dynamics.state_layout.unpack(source_state)
    relation = relation_state.relation
    ti = relation.target_indices
    sj = relation.source_indices
    displacement = tq[ti] - sq[sj]
    if box is not None:
        displacement = box.minimum_image(displacement)
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    h = 0.5 * (
        target.dynamics.method.smoothing_length + source.dynamics.method.smoothing_length
    )
    kernel = target.dynamics.method.kernel
    valid = relation.valid & (distance < kernel.support_radius(h))
    gradient = kernel.gradient(displacement, distance, h)
    dimension = tq.shape[-1]
    identity = jnp.eye(dimension, dtype=tq.dtype)
    target_tensor = (
        plan.surface_stress.surface_tension
        * (
            identity
            - contract("ni,nj->nij", geometry.target_normal, geometry.target_normal)
        )
        * geometry.target_delta[:, None, None]
    )
    source_tensor = (
        plan.surface_stress.surface_tension
        * (
            identity
            - contract("ni,nj->nij", geometry.source_normal, geometry.source_normal)
        )
        * geometry.source_delta[:, None, None]
    )
    pair_tensor = 0.5 * (target_tensor[ti] + source_tensor[sj])
    pair_force = contract("eij,ej->ei", pair_tensor, gradient)
    pair_force = jnp.where(valid[:, None], pair_force, 0.0)
    target_force = jnp.zeros_like(tq).at[ti].add(pair_force)
    source_force = jnp.zeros_like(sq).at[sj].add(-pair_force)
    defect = jnp.sum(target_force, axis=0) + jnp.sum(source_force, axis=0)
    power = jnp.sum(target_force * tv) + jnp.sum(source_force * sv)
    return BalancedInterfaceResult(base, target_force, source_force, defect, power)


__all__ = [
    "BalancedInterfaceForcePlan",
    "BalancedInterfaceResult",
    "ContinuumSurfaceStressPlan",
    "PhaseInterfaceGeometryState",
    "balanced_interface_force",
    "corrected_phase_interface_geometry",
]
