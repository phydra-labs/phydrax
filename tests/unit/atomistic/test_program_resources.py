#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.atomistic._classical import LennardJonesPotential
from phydrax.atomistic._electrostatics import DirectCoulombPotential
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticCompletePairSource,
    AtomisticNeighborhoodBundlePlan,
    AtomisticNeighborhoodBundleState,
    AtomisticNeighborhoodSource,
    AtomisticRoleCapacity,
    AtomisticRoleCapacityLadder,
)
from phydrax.atomistic._potential_program import (
    AtomisticPotentialProgram,
    LearnedGraphPotentialTerm,
    PreparedAtomisticPotentialProgram,
)
from phydrax.atomistic._sites import AtomisticSiteDomain
from phydrax.atomistic._system import AtomisticSystemPlan
from phydrax.atomistic._units import AtomisticUnitSystem
from phydrax.discretization import (
    ParticleImageCapacity,
    ParticleImageCapacityLadder,
    PeriodicCell,
)
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_prepare import prepare_mace_potential
from phydrax.nn.atomistic._radial_projection import RadialTableDeclaration


type _CoulombCase = tuple[
    PreparedAtomisticPotentialProgram, Array, AtomisticNeighborhoodBundleState
]


@pytest.fixture(scope="module")
def coulomb_case() -> _CoulombCase:
    system = AtomisticSystemPlan(
        np.asarray([13, 29], dtype=np.int64),
        np.ones(2, dtype=np.int32),
        np.ones(2, dtype=np.float64),
        AtomisticUnitSystem.reduced(),
        charges=np.asarray([1.0, -1.0], dtype=np.float64),
    ).prepare()
    plan = AtomisticPotentialProgram((DirectCoulombPotential(),))
    requirements = plan.requirements.neighborhoods
    resources = AtomisticNeighborhoodBundlePlan(
        requirements,
        (
            AtomisticCompletePairSource(
                "complete",
                AtomisticSiteDomain.PHYSICAL_ATOMS,
                requirements,
                maximum_pairs=1,
            ),
        ),
        (),
    )
    program = plan.prepare(system, neighborhood_bundle=resources)
    position = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    bundle = program.neighborhood_bundle
    state = bundle.initialize(bundle.geometry(position[None]))
    return program, position, state


@eqx.filter_jit
def _group_gradient(
    program: PreparedAtomisticPotentialProgram,
    positions: Array,
    neighborhoods: AtomisticNeighborhoodBundleState,
) -> Array:
    def scalar(value: Array) -> Array:
        return program.force_group_energy(0, value, neighborhoods)[0]

    return jax.grad(scalar)(positions)


def test_program_refuses_a_resource_support_that_omits_physical_atoms(
    coulomb_case: _CoulombCase,
) -> None:
    program, position, _ = coulomb_case
    bundle = program.neighborhood_bundle
    reduced = bundle.initialize(
        bundle.geometry(
            position[None],
            active=jnp.asarray([[True, False]], dtype=jnp.bool_),
        )
    )
    assert bool(jnp.all(reduced.successful))
    result = program.evaluate(position, reduced)
    assert not bool(result.successful)
    assert bool(jnp.isnan(result.energy))
    assert bool(jnp.all(jnp.isnan(result.forces)))


def test_failed_force_group_has_invalid_coordinate_derivatives(
    coulomb_case: _CoulombCase,
) -> None:
    program, position, state = coulomb_case
    moved = position.at[1, 0].set(2.1)
    energy, auxiliary = program.force_group_energy(0, moved, state)
    assert not bool(auxiliary.successful)
    assert bool(jnp.isnan(energy))
    assert bool(jnp.all(jnp.isnan(_group_gradient(program, moved, state))))


def test_force_group_uses_only_its_required_resource_evidence() -> None:
    system = AtomisticSystemPlan(
        np.asarray([2, 4, 8], dtype=np.int64),
        np.ones(3, dtype=np.int32),
        np.ones(3, dtype=np.float64),
        AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros(3, dtype=np.int32),
    ).prepare()
    epsilon = np.asarray([1.0], dtype=np.float64)
    sigma = np.asarray([0.2], dtype=np.float64)
    plan = AtomisticPotentialProgram(
        (
            LennardJonesPotential(epsilon, sigma, 0.8, name="short", force_group=0),
            LennardJonesPotential(epsilon, sigma, 2.0, name="long", force_group=1),
        )
    )
    short = plan.terms[0].requirements.neighborhoods[0]
    long = plan.terms[1].requirements.neighborhoods[0]
    capacity = ParticleImageCapacity(
        maximum_particles_per_cell=4,
        maximum_edges=32,
        maximum_case_edges=32,
        maximum_degree=16,
        maximum_images=1,
    )
    search = AtomisticCellListImageSearch(maximum_candidate_slots=4096)
    sources = tuple(
        AtomisticNeighborhoodSource(
            name,
            AtomisticSiteDomain.PHYSICAL_ATOMS,
            (requirement,),
            search,
            ParticleImageCapacityLadder((capacity,)),
            skin=0.1,
        )
        for name, requirement in (("short", short), ("long", long))
    )
    resources = AtomisticNeighborhoodBundlePlan(
        plan.requirements.neighborhoods,
        sources,
        (
            AtomisticRoleCapacityLadder(
                short, (AtomisticRoleCapacity(maximum_routes=4, maximum_degree=16),)
            ),
            AtomisticRoleCapacityLadder(
                long, (AtomisticRoleCapacity(maximum_routes=1, maximum_degree=16),)
            ),
        ),
        frame=PeriodicCell(
            np.eye(3, dtype=np.float64) * 8.0,
            periodic_axes=(False, False, False),
        ),
    )
    program = plan.prepare(system, neighborhood_bundle=resources)
    point = jnp.asarray(
        [[0.2, 0.2, 0.2], [0.7, 0.2, 0.2], [1.9, 0.2, 0.2]],
        dtype=jnp.float64,
    )
    bundle = program.neighborhood_bundle
    state = bundle.initialize(bundle.geometry(point[None]))
    assert bool(jnp.all(bundle.view(state, short).evidence.successful))
    assert not bool(jnp.all(bundle.view(state, long).evidence.successful))
    energy, auxiliary = program.force_group_energy(0, point, state)
    expected = 4.0 * ((0.2 / 0.5) ** 12 - (0.2 / 0.5) ** 6)
    assert bool(auxiliary.successful)
    np.testing.assert_allclose(energy, expected, rtol=1e-12, atol=1e-14)
    complete_energy, complete = program.energy(point, state)
    assert not bool(complete.successful)
    assert bool(jnp.isnan(complete_energy))


def test_prepared_learned_identity_distinguishes_numerical_realizations() -> None:
    system = AtomisticSystemPlan(
        np.asarray([13, 29], dtype=np.int64),
        np.ones(2, dtype=np.int32),
        np.ones(2, dtype=np.float64),
        AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond(),
    ).prepare()
    model = MACEPotential(
        system.plan.units.scale,
        MACEArchitecture(
            species=(1,),
            cutoff=2.5,
            radial_basis_count=2,
            cutoff_power=5,
            channel_count=1,
            hidden_degree=0,
            edge_degree=0,
            interactions=("real-agnostic",),
            correlations=(1,),
            radial_widths=(2,),
            readout_width=None,
            average_neighbor_count=1.0,
        ),
        atomic_energies=np.zeros((1, 1), dtype=np.float64),
        key=jax.random.key(4),
    )
    exact = prepare_mace_potential(model)
    tabulated = prepare_mace_potential(
        model,
        radial="tabulated",
        tables=RadialTableDeclaration(0.2, 64, layout="projected-width"),
    )
    exact_term = LearnedGraphPotentialTerm(exact).prepare(system)
    tabulated_term = LearnedGraphPotentialTerm(tabulated).prepare(system)
    assert exact_term.term_id == tabulated_term.term_id
    assert exact_term.prepared_id != tabulated_term.prepared_id
