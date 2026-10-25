#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small explicit dynamics fixtures shared by the thermostat, NPT and rank tests.

Terms here are deliberately tiny analytic Hamiltonians in reduced units:

* ``DirectedPairTerm`` sums a pair profile over the directed image routes of
  its own ``DIRECTED_IMAGE_GRAPH`` requirement, read only through the bundle
  view of the potential context (``exponential``: ``sum exp(-|d|)`` inside the
  cutoff; ``smooth``: ``strength * sum_directed (1 - |d|/r_c)^4 / 2``, C3 at
  the cutoff so energy-conservation tests see no truncation jump).
* ``ExternalFieldTerm`` needs no neighborhood: ``zero`` (an ideal gas whose
  homogeneous-strain response is exactly zero), ``harmonic`` tethers
  ``k/2 |x - a|^2`` and a ``linear`` field ``g . x``.

Every bundle resource is explicit: one dense image search with a stated
deformation margin, skin, search capacity ladder and role capacity ladder
bound to ``program.requirements.neighborhoods`` via the shared bundle helper.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._identity import IdentityWords
from phydrax.atomistic._dynamics import AtomisticDynamicsPlan
from phydrax.atomistic._graph import AtomisticGraphExecutionPlan
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticDenseImageSearch,
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
    AtomisticRoleCapacity,
)
from phydrax.atomistic._potential import (
    AtomisticPotentialCapabilities,
    AtomisticPotentialRequirements,
)
from phydrax.atomistic._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialProgram,
    AtomisticTermEvaluation,
)
from phydrax.atomistic._thermodynamic import (
    AtomisticPhaseSpaceMeasurePlan,
    AtomisticThermodynamicStatePlan,
    PreparedThermodynamicStateTable,
)
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


SKIN = 0.3
# Dense image reference for these few-atom cells; the 25% deformation margin
# widens the image stencil so Monte Carlo and MTK cell moves stay certified.
SEARCH = AtomisticDenseImageSearch(maximum_dense_routes=200_000, deformation_margin=0.25)
SEARCH_CAPACITIES = (search_capacity(512, particles_per_cell=16, images=1024),)
ROLE_CAPACITIES = (role_capacity(4096, 512),)


def _periodic_capabilities() -> AtomisticPotentialCapabilities:
    return AtomisticPotentialCapabilities(
        orthorhombic_periodic=True, triclinic_periodic=True, cell_derivative=True
    )


class DirectedPairTerm(AbstractAtomisticEnergyTerm):
    """Pair profile summed over the directed image routes of one requirement."""

    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    profile: str = eqx.field(static=True)
    strength: jax.Array
    requirement: AtomisticNeighborhoodRequirement
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self, name: str, cutoff: float, /, *, profile: str, strength: float = 1.0
    ) -> None:
        if profile not in ("exponential", "smooth"):
            raise ValueError("profile must be 'exponential' or 'smooth'.")
        self.name = name
        self.force_group = 0
        self.profile = profile
        self.strength = jnp.asarray(float(strength), dtype=jnp.float64)
        self.requirement = AtomisticNeighborhoodRequirement(
            name, AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH, float(cutoff)
        )
        self.term_identity = IdentityWords.of(
            f"directed-pair-{profile}-{name}-{float(cutoff)}-{float(strength)}"
        )
        self.capabilities = _periodic_capabilities()
        self.requirements = AtomisticPotentialRequirements(
            neighborhoods=(self.requirement,)
        )

    def prepare(self, system: Any, /) -> "PreparedDirectedPairTerm":
        del system
        return PreparedDirectedPairTerm(self)


class PreparedDirectedPairTerm(AbstractPreparedAtomisticEnergyTerm):
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    profile: str = eqx.field(static=True)
    strength: jax.Array
    requirement: AtomisticNeighborhoodRequirement
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: DirectedPairTerm, /) -> None:
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.prepared_identity = IdentityWords.of(f"prepared-{plan.term_id}")
        self.profile = plan.profile
        self.strength = plan.strength
        self.requirement = plan.requirement
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements

    def energy(self, context: Any, /) -> AtomisticTermEvaluation:
        view = context.neighborhood(self.requirement)
        positions = context.positions
        vectors = jnp.asarray(context.cell_vectors).astype(positions.dtype)
        displacement = view.displacement(positions, vectors)
        inside = view.cutoff_mask(positions, vectors)
        squared = jnp.sum(displacement * displacement, axis=-1)
        distance = jnp.sqrt(jnp.where(inside, squared, 1.0))
        cutoff = self.requirement.bounded_cutoff()
        if self.profile == "exponential":
            pair = jnp.exp(-distance)
        else:
            pair = 0.5 * (1.0 - distance / cutoff) ** 4
        edge = self.strength * jnp.where(inside, pair, 0.0)
        atom_energy = (
            jnp.zeros((positions.shape[0],), dtype=positions.dtype)
            .at[view.relation.receiver_indices]
            .add(edge)
        )
        return AtomisticTermEvaluation(
            jnp.sum(edge), atom_energy, jnp.all(view.evidence.successful)
        )


class ExternalFieldTerm(AbstractAtomisticEnergyTerm):
    """Neighborhood-free one-body field: zero, harmonic tether or linear."""

    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    profile: str = eqx.field(static=True)
    stiffness: jax.Array
    coefficients: jax.Array
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        name: str,
        profile: str,
        coefficients: Any,
        /,
        *,
        stiffness: float = 0.0,
    ) -> None:
        if profile not in ("zero", "harmonic", "linear"):
            raise ValueError("profile must be 'zero', 'harmonic' or 'linear'.")
        values = np.asarray(coefficients, dtype=np.float64)
        self.name = name
        self.force_group = 0
        self.profile = profile
        self.stiffness = jnp.asarray(float(stiffness), dtype=jnp.float64)
        self.coefficients = jnp.asarray(values)
        self.term_identity = IdentityWords.of(
            f"external-{profile}-{name}-{float(stiffness)}-{values.round(15).tolist()}"
        )
        # Only the zero field is a lattice-invariant (periodic) Hamiltonian.
        self.capabilities = (
            _periodic_capabilities()
            if profile == "zero"
            else AtomisticPotentialCapabilities()
        )
        self.requirements = AtomisticPotentialRequirements()

    def prepare(self, system: Any, /) -> "PreparedExternalFieldTerm":
        del system
        return PreparedExternalFieldTerm(self)


class PreparedExternalFieldTerm(AbstractPreparedAtomisticEnergyTerm):
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    profile: str = eqx.field(static=True)
    stiffness: jax.Array
    coefficients: jax.Array
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(self, plan: ExternalFieldTerm, /) -> None:
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.prepared_identity = IdentityWords.of(f"prepared-{plan.term_id}")
        self.profile = plan.profile
        self.stiffness = plan.stiffness
        self.coefficients = plan.coefficients
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements

    def energy(self, context: Any, /) -> AtomisticTermEvaluation:
        positions = context.positions
        coefficients = self.coefficients.astype(positions.dtype)
        if self.profile == "zero":
            atom_energy = 0.0 * jnp.sum(positions, axis=-1)
        elif self.profile == "harmonic":
            offset = positions - coefficients
            atom_energy = 0.5 * self.stiffness * jnp.sum(offset * offset, axis=-1)
        else:
            atom_energy = jnp.sum(coefficients * positions, axis=-1)
        return AtomisticTermEvaluation(
            jnp.sum(atom_energy), atom_energy, jnp.asarray(True)
        )


def atomistic_system(
    particle_ids: Any,
    masses: Any,
    /,
    *,
    cell: Any = None,
    topology: Any = None,
    mobile_mask: Any = None,
) -> Any:
    ids = np.asarray(particle_ids, dtype=np.int64)
    return phx.atomistic.AtomisticSystemPlan(
        ids,
        np.ones(ids.shape, dtype=np.int32),
        np.asarray(masses, dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(ids.shape, dtype=np.int32),
        mobile_mask=mobile_mask,
        topology=topology,
        cell=cell,
    ).prepare()


def periodic_cell(vectors: Any, /) -> Any:
    return phx.discretization.PeriodicCell(jnp.asarray(vectors, dtype=jnp.float64))


def prepare_potential(
    system: Any,
    terms: tuple[AbstractAtomisticEnergyTerm, ...],
    /,
    *,
    role_capacities: tuple[AtomisticRoleCapacity, ...] = ROLE_CAPACITIES,
) -> Any:
    """Program over ``terms`` bound to the explicit bundle of its requirements."""
    program = AtomisticPotentialProgram(terms)
    if not program.requirements.neighborhoods:
        return prepare_program(program, system)
    graph_execution = (
        AtomisticGraphExecutionPlan(
            max(capacity.maximum_degree for capacity in role_capacities),
            backend="particle",
        )
        if any(
            requirement.role is AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
            for requirement in program.requirements.neighborhoods
        )
        else None
    )
    return prepare_program(
        program,
        system,
        search=SEARCH,
        skin=SKIN,
        search_capacities=SEARCH_CAPACITIES,
        role_capacities=role_capacities,
        graph_execution=graph_execution,
    )


def prepare_dynamics(
    system: Any,
    terms: tuple[AbstractAtomisticEnergyTerm, ...],
    integrator: Any,
    /,
    *,
    constraints: Any = None,
    role_capacities: tuple[AtomisticRoleCapacity, ...] = ROLE_CAPACITIES,
) -> Any:
    potential = prepare_potential(system, terms, role_capacities=role_capacities)
    return AtomisticDynamicsPlan(
        system, potential, integrator, constraints=constraints
    ).prepare()


def measure(
    system: Any,
    /,
    *,
    convention: str = "atomic",
    restriction: str = "none",
) -> AtomisticPhaseSpaceMeasurePlan:
    return AtomisticPhaseSpaceMeasurePlan(
        system,
        volume_coordinate_convention=convention,  # ty: ignore[invalid-argument-type]
        momentum_restriction=restriction,  # ty: ignore[invalid-argument-type]
    )


def row(
    phase_space: AtomisticPhaseSpaceMeasurePlan,
    ensemble: str,
    /,
    *,
    temperature: float | None = None,
    pressure: float | None = None,
) -> AtomisticThermodynamicStatePlan:
    return AtomisticThermodynamicStatePlan(
        phase_space,
        ensemble=ensemble,  # ty: ignore[invalid-argument-type]
        temperature=temperature,
        pressure=pressure,
    )


def table(dynamics: Any, *rows: AtomisticThermodynamicStatePlan) -> Any:
    return PreparedThermodynamicStateTable(dynamics, rows)


@eqx.filter_jit
def step_detailed(dynamics: Any, state: Any, thermodynamic: Any) -> Any:
    """One compiled transaction; identical to the eager ``step_detailed``."""
    return dynamics.step_detailed(state, thermodynamic)


def assert_tree_equal(first: Any, second: Any, /) -> None:
    """Bitwise equality of two pytrees (structure, static fields and leaves)."""
    first_leaves, first_structure = jax.tree.flatten(first)
    second_leaves, second_structure = jax.tree.flatten(second)
    assert first_structure == second_structure
    for left, right in zip(first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def observed_role_storage(state: Any, /) -> AtomisticRoleCapacity:
    """Smallest role storage holding the single role view of an accepted state.

    Used to declare a first ladder entry that fits the initial geometry exactly,
    so a later, denser geometry overflows it (a pure capacity failure).
    """
    (view,) = state.neighborhoods.views
    routes = int(np.sum(np.asarray(view.evidence.required_routes)))
    degree = int(np.max(np.asarray(view.evidence.required_degree), initial=0))
    return role_capacity(max(routes, 1), max(degree, 1))
