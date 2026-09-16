#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.fuzzy_space import (
    fuzzy_sphere_spectrum,
    FuzzySphereTwoParticlePlan,
    prepare_fuzzy_sphere_two_particle,
)
from phydrax.operators.quantum.lattice import (
    prepare_su2_sector_basis,
    project_product_operator_to_su2_sector,
    SU2CouplingTreePlan,
    SU2SectorResourcePolicy,
)


def _resources():
    return SU2SectorResourcePolicy(
        maximum_product_dimension=128,
        maximum_sector_dimension=128,
        maximum_matrix_elements=16_384,
    )


def test_three_spin_half_total_half_sector_has_two_exact_multiplets():
    basis = prepare_su2_sector_basis(
        SU2CouplingTreePlan(
            ("a", "b", "c"),
            (1, 1, 1),
            1,
            _resources(),
        )
    )
    assert basis.plan.multiplicity_dimension == 2
    assert basis.plan.sector_dimension == 4
    assert bool(basis.evidence.accepted)
    np.testing.assert_allclose(
        basis.transform @ basis.transform.T,
        jnp.eye(4),
        rtol=1e-12,
        atol=1e-12,
    )
    vector = jnp.arange(1.0, 5.0)
    np.testing.assert_allclose(
        basis.from_product(basis.to_product(vector)), vector, atol=1e-12
    )


def test_su2_projected_swap_preserves_total_spin_sector():
    basis = prepare_su2_sector_basis(
        SU2CouplingTreePlan(("a", "b"), (1, 1), 0, _resources())
    )
    swap = np.zeros((4, 4))
    coordinates = tuple(np.ndindex(2, 2))
    lookup = {coordinate: index for index, coordinate in enumerate(coordinates)}
    for index, coordinate in enumerate(coordinates):
        swap[lookup[(coordinate[1], coordinate[0])], index] = 1.0
    projected = project_product_operator_to_su2_sector(
        swap,
        basis,
        operator_source_id="two-spin-swap",
        hermitian=True,
    )
    np.testing.assert_allclose(projected.matrix, ((-1.0,),), atol=1e-12)
    assert bool(projected.evidence.accepted)


def test_fuzzy_sphere_statistics_select_exact_pair_spin_sectors():
    fermion = prepare_fuzzy_sphere_two_particle(
        FuzzySphereTwoParticlePlan(
            1,
            "fermion",
            {0: 2.5},
            _resources(),
        )
    )
    fermion_spectrum = fuzzy_sphere_spectrum(fermion)
    assert fermion.plan.allowed_total_twice_spins == (0,)
    assert fermion.physical_hamiltonian.shape == (1, 1)
    np.testing.assert_allclose(fermion_spectrum.energies, (2.5,))
    assert bool(fermion.evidence.accepted)

    boson = prepare_fuzzy_sphere_two_particle(
        FuzzySphereTwoParticlePlan(
            1,
            "boson",
            {2: -0.75},
            _resources(),
        )
    )
    boson_spectrum = fuzzy_sphere_spectrum(boson)
    assert boson.plan.allowed_total_twice_spins == (2,)
    assert boson.physical_hamiltonian.shape == (3, 3)
    np.testing.assert_allclose(boson_spectrum.energies, -0.75 * jnp.ones((3,)))
    np.testing.assert_array_equal(boson_spectrum.degeneracies, (3, 3, 3))
    assert bool(boson.evidence.accepted)
    assert "no-continuum" in boson_spectrum.claim


def test_spin_one_fermions_keep_only_antisymmetric_total_spin_one():
    model = prepare_fuzzy_sphere_two_particle(
        FuzzySphereTwoParticlePlan(
            2,
            "fermion",
            {2: 1.25},
            _resources(),
        )
    )
    assert model.plan.allowed_total_twice_spins == (2,)
    assert model.physical_hamiltonian.shape == (3, 3)
    np.testing.assert_allclose(jnp.diag(model.total_spin_squared), 2.0)
    assert bool(model.evidence.exchange_sector_complete)
