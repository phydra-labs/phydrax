#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded soft-matter protocols composed from the atomistic runtime."""

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._classical import HarmonicBondPotential, LennardJonesPotential
from ._dynamics import PreparedAtomisticDynamics
from ._thermal import BAOABLangevinPlan
from ._thermodynamic import PreparedThermodynamicStateTable


class SoftMatterProtocolKind(StrEnum):
    LENNARD_JONES_LIQUID = "lennard-jones-liquid"
    FIXED_BINARY_GLASS = "fixed-binary-glass"
    LANGEVIN_COLLOID = "langevin-colloid"
    FIXED_LINEAR_POLYMER = "fixed-linear-polymer"


class SoftMatterAtomisticProtocol(StrictModule, NonTrainableState):
    """A validated protocol binding existing atomistic execution owners."""

    dynamics: PreparedAtomisticDynamics
    kind: SoftMatterProtocolKind = eqx.field(static=True)
    production_steps: int = eqx.field(static=True)
    maximum_particles: int = eqx.field(static=True)
    protocol_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedAtomisticDynamics,
        kind: SoftMatterProtocolKind,
        /,
        *,
        production_steps: int,
        maximum_particles: int,
    ):
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(kind, SoftMatterProtocolKind):
            raise TypeError("kind must be SoftMatterProtocolKind.")
        steps = int(production_steps)
        capacity = int(maximum_particles)
        active_count = int(np.count_nonzero(np.asarray(dynamics.system.active_mask)))
        if steps <= 0 or capacity <= 0 or active_count > capacity:
            raise ValueError(
                "Soft-matter production steps or particle capacity are invalid."
            )
        cell = dynamics.system.cell
        if cell is None or not cell.fully_periodic:
            raise ValueError(
                "These soft-matter protocols require one fully periodic cell."
            )
        terms = dynamics.potential.plan.terms
        lennard_jones = tuple(
            term for term in terms if isinstance(term, LennardJonesPotential)
        )
        bonds = tuple(term for term in terms if isinstance(term, HarmonicBondPotential))
        if len(lennard_jones) != 1:
            raise ValueError(
                "A soft-matter protocol requires exactly one Lennard-Jones term."
            )
        lj = lennard_jones[0]
        active = np.asarray(dynamics.system.active_mask, dtype=np.bool_)
        types = np.asarray(dynamics.system.plan.atom_type_ids)[active]
        topology = dynamics.system.plan.topology
        if kind is SoftMatterProtocolKind.LENNARD_JONES_LIQUID:
            if (
                np.unique(types).size != 1
                or len(bonds) != 0
                or topology.bonds.shape[0] != 0
            ):
                raise ValueError("The Lennard-Jones liquid is monodisperse and unbonded.")
        elif kind is SoftMatterProtocolKind.FIXED_BINARY_GLASS:
            expected_epsilon = np.asarray(((1.0, 1.5), (1.5, 0.5)))
            expected_sigma = np.asarray(((1.0, 0.8), (0.8, 0.88)))
            counts = np.asarray([np.count_nonzero(types == index) for index in range(2)])
            if (
                lj.combining_rule != "explicit"
                or lj.explicit_epsilon is None
                or lj.explicit_sigma is None
                or np.unique(types).size != 2
                or not np.array_equal(np.unique(types), np.asarray((0, 1)))
                or counts[0] != 4 * counts[1]
                or not np.array_equal(np.asarray(lj.explicit_epsilon), expected_epsilon)
                or not np.array_equal(np.asarray(lj.explicit_sigma), expected_sigma)
                or lj.cutoff != 2.5
                or len(bonds) != 0
            ):
                raise ValueError(
                    "The fixed binary glass requires the 80:20 explicit A/B parameter set."
                )
        elif kind is SoftMatterProtocolKind.LANGEVIN_COLLOID:
            sigma = float(np.asarray(lj.sigma)[int(types[0])])
            wca_cutoff = 2.0 ** (1.0 / 6.0) * sigma
            if (
                not isinstance(dynamics.integrator, BAOABLangevinPlan)
                or np.unique(types).size != 1
                or np.any(np.asarray(dynamics.system.plan.element_mask)[active])
                or not np.isclose(lj.cutoff, wca_cutoff, rtol=0.0, atol=1.0e-12)
                or not lj.shift_energy_at_cutoff
                or len(bonds) != 0
            ):
                raise ValueError(
                    "The Langevin colloid requires non-element WCA beads and BAOAB forcing."
                )
        else:
            particle_ids = np.asarray(dynamics.system.plan.particle_ids)[active]
            expected_bonds = np.sort(
                np.stack((particle_ids[:-1], particle_ids[1:]), axis=1), axis=1
            )
            actual_bonds = np.asarray(topology.bonds)
            exception_pairs = np.asarray(topology.pair_exceptions)
            exception_scales = np.asarray(topology.lennard_jones_scales)
            excluded = {
                tuple(pair): scale
                for pair, scale in zip(exception_pairs, exception_scales, strict=True)
            }
            bonded_excluded = all(
                excluded.get(tuple(pair), 1.0) == 0.0 for pair in expected_bonds
            )
            molecule_ids = np.asarray(dynamics.system.plan.molecule_ids)[active]
            if (
                not isinstance(dynamics.integrator, BAOABLangevinPlan)
                or len(bonds) != 1
                or actual_bonds.shape != expected_bonds.shape
                or not np.array_equal(actual_bonds, expected_bonds)
                or not bonded_excluded
                or np.unique(molecule_ids).size != 1
            ):
                raise ValueError(
                    "The fixed linear polymer requires one ordered chain, harmonic bonds, "
                    "excluded bonded LJ pairs, and BAOAB forcing."
                )
        self.dynamics = dynamics
        self.kind = kind
        self.production_steps = steps
        self.maximum_particles = capacity
        self.protocol_id = canonical_fingerprint(
            {
                "kind": "soft-matter-atomistic-protocol",
                "profile": kind.value,
                "dynamics": dynamics.prepared_id,
                "production_steps": steps,
                "maximum_particles": capacity,
            }
        )


class LangevinFDTReport(StrictModule, NonTrainableState):
    stationary_momentum_variance: Array
    innovation_variance: Array
    identity_residual: Array
    successful: Array
    noise_addressing: str = eqx.field(static=True)
    integrator_id: str = eqx.field(static=True)


def langevin_fdt_report(
    protocol: SoftMatterAtomisticProtocol,
    thermodynamic_states: PreparedThermodynamicStateTable,
    /,
    *,
    state_index: int = 0,
) -> LangevinFDTReport:
    """Evidence for the BAOAB discrete Ornstein--Uhlenbeck FDT identity."""

    if not isinstance(protocol, SoftMatterAtomisticProtocol):
        raise TypeError("protocol must be SoftMatterAtomisticProtocol.")
    if not isinstance(thermodynamic_states, PreparedThermodynamicStateTable):
        raise TypeError("thermodynamic_states must be PreparedThermodynamicStateTable.")
    thermodynamic_states.validate_dynamics(protocol.dynamics)
    index = int(state_index)
    if not 0 <= index < thermodynamic_states.state_count:
        raise ValueError("state_index is outside the thermodynamic state table.")
    if not bool(np.asarray(thermodynamic_states.temperature_mask[index])):
        raise ValueError("FDT evidence requires a state with positive temperature.")
    integrator = protocol.dynamics.integrator
    if not isinstance(integrator, BAOABLangevinPlan):
        raise ValueError("FDT evidence requires a BAOABLangevinPlan.")
    system = protocol.dynamics.system
    dtype = system.plan.masses.dtype
    temperature = thermodynamic_states.temperature[index]
    decay = jnp.exp(-jnp.asarray(integrator.friction * integrator.step_size, dtype=dtype))
    stationary = (
        system.plan.masses
        * system.plan.units.boltzmann_constant
        * temperature
        / system.plan.units.kinetic_to_energy
    )
    innovation = (1.0 - decay * decay) * stationary
    residual = jnp.max(jnp.abs(innovation - (1.0 - decay * decay) * stationary))
    successful = (
        jnp.all(jnp.isfinite(stationary))
        & jnp.all(stationary > 0.0)
        & jnp.all(jnp.isfinite(innovation))
        & jnp.all(innovation > 0.0)
        & jnp.isfinite(residual)
    )
    return LangevinFDTReport(
        stationary,
        innovation,
        residual,
        successful,
        "stable-particle-id/step/operator/realization",
        integrator.plan_id,
    )


__all__ = [
    "LangevinFDTReport",
    "SoftMatterAtomisticProtocol",
    "SoftMatterProtocolKind",
    "langevin_fdt_report",
]
