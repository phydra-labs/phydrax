#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import (
    AtomisticDynamicsState,
    FiniteExtensibleNonlinearElasticBondPotential,
    HarmonicAnglePotential,
    HarmonicBondPotential,
    LennardJonesPotential,
    PolymerChainLayoutPlan,
    PreparedAtomisticDynamics,
    PreparedThermodynamicStateTable,
)
from ...atomistic._thermal import BAOABLangevinPlan


class KremerGrestProfilePlan(StrictModule, NonTrainableState):
    """Qualified monodisperse FENE-WCA bead-spring protocol contract."""

    production_steps: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    maximum_chains: int = eqx.field(static=True)
    minimum_fene_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        production_steps: int,
        maximum_particles: int,
        maximum_chains: int,
        minimum_fene_margin: float = 0.05,
    ):
        steps = int(production_steps)
        particles = int(maximum_particles)
        chains = int(maximum_chains)
        margin = float(minimum_fene_margin)
        if (
            steps <= 0
            or particles <= 0
            or chains <= 0
            or not math.isfinite(margin)
            or not 0.0 < margin < 1.0
        ):
            raise ValueError("Kremer-Grest profile bounds are invalid.")
        self.production_steps = steps
        self.maximum_particles = particles
        self.maximum_chains = chains
        self.minimum_fene_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kremer-grest-profile-plan",
                "production_steps": steps,
                "maximum_particles": particles,
                "maximum_chains": chains,
                "minimum_fene_margin": margin,
            }
        )

    def prepare(
        self,
        dynamics: PreparedAtomisticDynamics,
        chain_layout: PolymerChainLayoutPlan,
        /,
    ) -> "PreparedKremerGrestProfile":
        return PreparedKremerGrestProfile(self, dynamics, chain_layout)


class PreparedKremerGrestProfile(StrictModule, NonTrainableState):
    plan: KremerGrestProfilePlan
    dynamics: PreparedAtomisticDynamics
    chain_layout: PolymerChainLayoutPlan
    fene_term_index: int = eqx.field(static=True)
    lj_term_index: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: KremerGrestProfilePlan,
        dynamics: PreparedAtomisticDynamics,
        chain_layout: PolymerChainLayoutPlan,
        /,
    ):
        if not isinstance(plan, KremerGrestProfilePlan):
            raise TypeError("plan must be KremerGrestProfilePlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(chain_layout, PolymerChainLayoutPlan):
            raise TypeError("chain_layout must be PolymerChainLayoutPlan.")
        system = dynamics.system
        cell = system.cell
        active = np.asarray(system.active_mask, dtype=np.bool_)
        active_count = int(np.count_nonzero(active))
        if cell is None or not cell.fully_periodic:
            raise ValueError("The Kremer-Grest profile requires a fully periodic cell.")
        if not isinstance(dynamics.integrator, BAOABLangevinPlan):
            raise TypeError("The initial Kremer-Grest profile requires BAOAB dynamics.")
        if (
            active_count > plan.maximum_particles
            or chain_layout.chain_mask.shape[0] > plan.maximum_chains
            or chain_layout.particle_indices.shape[1] > plan.maximum_particles
        ):
            raise ValueError("Kremer-Grest resource bounds are exceeded.")
        layout_indices = np.asarray(chain_layout.particle_indices, dtype=np.int32)
        layout_mask = np.asarray(chain_layout.chain_mask, dtype=np.bool_)
        selected = layout_indices[layout_mask]
        if (
            selected.size != active_count
            or np.unique(selected).size != selected.size
            or np.any(selected < 0)
            or np.any(selected >= system.capacity)
            or not np.all(active[selected])
        ):
            raise ValueError(
                "The chain layout must partition all active particles exactly once."
            )
        terms = dynamics.potential.plan.terms
        fene_indices = tuple(
            index
            for index, term in enumerate(terms)
            if isinstance(term, FiniteExtensibleNonlinearElasticBondPotential)
        )
        lj_indices = tuple(
            index
            for index, term in enumerate(terms)
            if isinstance(term, LennardJonesPotential)
        )
        harmonic_bonds = tuple(
            term for term in terms if isinstance(term, HarmonicBondPotential)
        )
        angle_terms = tuple(
            term for term in terms if isinstance(term, HarmonicAnglePotential)
        )
        if len(fene_indices) != 1 or len(lj_indices) != 1 or harmonic_bonds:
            raise ValueError(
                "Kremer-Grest requires exactly one FENE term, one LJ term, and no harmonic bond term."
            )
        if len(angle_terms) > 1:
            raise ValueError("Kremer-Grest supports at most one bending term.")
        fene = terms[fene_indices[0]]
        lj = terms[lj_indices[0]]
        atom_types = np.asarray(system.plan.atom_type_ids)[active]
        if np.unique(atom_types).size != 1:
            raise ValueError("The initial Kremer-Grest profile is monodisperse.")
        bead_type = int(atom_types[0])
        sigma = float(np.asarray(lj.sigma)[bead_type])
        expected_cutoff = 2.0 ** (1.0 / 6.0) * sigma
        if (
            not lj.shift_energy_at_cutoff
            or lj.switch_distance is not None
            or not np.isclose(lj.cutoff, expected_cutoff, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(
                "Kremer-Grest requires energy-shifted WCA cutoff at 2^(1/6) sigma."
            )
        expected_bonds: list[tuple[int, int]] = []
        expected_angles: list[tuple[int, int, int]] = []
        for indices, mask in zip(layout_indices, layout_mask, strict=True):
            chain = indices[mask]
            expected_bonds.extend(
                tuple(sorted((int(left), int(right)))) for left, right in pairwise(chain)
            )
            expected_angles.extend(
                (int(left), int(center), int(right))
                for left, center, right in zip(
                    chain[:-2], chain[1:-1], chain[2:], strict=True
                )
            )
        actual_bonds = {
            tuple(sorted((int(left), int(right))))
            for left, right in np.asarray(system.topology.bond_indices)
        }
        if actual_bonds != set(expected_bonds):
            raise ValueError(
                "Kremer-Grest topology must contain exactly the chain bonds."
            )
        actual_angles = {tuple(row) for row in np.asarray(system.topology.angle_indices)}
        if angle_terms and actual_angles != set(expected_angles):
            raise ValueError("The bending term must route every internal chain angle.")
        if not angle_terms and actual_angles:
            raise ValueError("Topology angles require an admitted bending term.")
        stable_ids = np.asarray(system.plan.particle_ids, dtype=np.int64)
        exception_scale = {
            tuple(sorted((int(left), int(right)))): float(scale)
            for (left, right), scale in zip(
                np.asarray(system.plan.topology.pair_exceptions, dtype=np.int64),
                np.asarray(system.plan.topology.lennard_jones_scales, dtype=np.float64),
                strict=True,
            )
        }
        for left_slot, right_slot in expected_bonds:
            pair = tuple(
                sorted((int(stable_ids[left_slot]), int(stable_ids[right_slot])))
            )
            if not np.isclose(exception_scale.get(pair, 1.0), 1.0):
                raise ValueError("Kremer-Grest WCA must remain active on bonded pairs.")
        fene_types = np.asarray(system.topology.bond_type_ids, dtype=np.int32)
        if fene_types.size and int(np.max(fene_types)) >= fene.stiffness.size:
            raise ValueError("FENE type routing exceeds the parameter table.")
        self.plan = plan
        self.dynamics = dynamics
        self.chain_layout = chain_layout
        self.fene_term_index = fene_indices[0]
        self.lj_term_index = lj_indices[0]
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-kremer-grest-profile",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
                "chain_layout": chain_layout.plan_id,
                "fene_term": fene.term_id,
                "lj_term": lj.term_id,
            }
        )


class KremerGrestEvidence(StrictModule):
    maximum_bond_fraction: Array
    minimum_bond_margin: Array
    fdt_identity_residual: Array
    potential_successful: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


def kremer_grest_evidence(
    profile: PreparedKremerGrestProfile,
    state: AtomisticDynamicsState,
    thermodynamic_states: PreparedThermodynamicStateTable,
    /,
    *,
    state_index: int = 0,
) -> KremerGrestEvidence:
    if not isinstance(profile, PreparedKremerGrestProfile):
        raise TypeError("profile must be PreparedKremerGrestProfile.")
    if not isinstance(thermodynamic_states, PreparedThermodynamicStateTable):
        raise TypeError("thermodynamic_states must be PreparedThermodynamicStateTable.")
    dynamics = profile.dynamics
    if state.prepared_dynamics_id != dynamics.prepared_id:
        raise ValueError("State belongs to another atomistic dynamics runtime.")
    thermodynamic_states.validate_dynamics(dynamics)
    index = int(state_index)
    if not 0 <= index < thermodynamic_states.state_count:
        raise ValueError("state_index is outside the thermodynamic state table.")
    if not bool(np.asarray(thermodynamic_states.temperature_mask[index])):
        raise ValueError(
            "Kremer-Grest FDT evidence requires a state with positive temperature."
        )
    topology = dynamics.system.topology
    unwrapped = dynamics._unwrapped(state.kinematics, state.cell_vectors)
    bond_indices = topology.bond_indices
    displacement = unwrapped[bond_indices[:, 0]] - unwrapped[bond_indices[:, 1]]
    distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    fene = dynamics.potential.plan.terms[profile.fene_term_index]
    maximum = fene.maximum_extension[topology.bond_type_ids]
    fraction = distance / maximum
    maximum_fraction = jnp.max(fraction)
    margin = 1.0 - maximum_fraction
    integrator = dynamics.integrator
    dtype = state.kinematics.positions.dtype
    decay = jnp.exp(-jnp.asarray(integrator.friction * integrator.step_size, dtype=dtype))
    stationary = (
        dynamics.system.plan.masses
        * dynamics.system.plan.units.boltzmann_constant
        * thermodynamic_states.temperature[index]
        / dynamics.system.plan.units.kinetic_to_energy
    )
    innovation = (1.0 - decay * decay) * stationary
    residual = jnp.max(jnp.abs(innovation - (1.0 - decay * decay) * stationary))
    potential = dynamics.evaluate_state(state, thermodynamic_states, state_index=index)
    successful = (
        potential.successful
        & jnp.all(jnp.isfinite(fraction))
        & (maximum_fraction < 1.0)
        & (margin >= profile.plan.minimum_fene_margin)
        & jnp.isfinite(residual)
    )
    return KremerGrestEvidence(
        maximum_fraction,
        margin,
        residual,
        potential.successful,
        successful,
        profile.prepared_id,
    )


__all__ = [
    "KremerGrestEvidence",
    "KremerGrestProfilePlan",
    "PreparedKremerGrestProfile",
    "kremer_grest_evidence",
]
