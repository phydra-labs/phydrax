#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _fine_system(*, cell=None):
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    return phx.atomistic.AtomisticSystemPlan(
        [10, 20, 30, 40],
        [1, 1, 1, 1],
        [1.0, 3.0, 2.0, 2.0],
        units,
        atom_type_ids=[1, 1, 1, 1],
        charges=[0.1, -0.1, 0.2, -0.2],
        molecule_ids=[0, 0, 1, 1],
        cell=cell,
    ).prepare()


def _mapping(system):
    return phx.atomistic.MolecularCoarseMapPlan([100, 200], [0, 1], [0, 0, 1, 1]).prepare(
        system
    )


def test_center_of_mass_map_conserves_mass_charge_momentum_and_force():
    system = _fine_system()
    mapping = _mapping(system)
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]]
    )
    forces = jnp.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]
    )
    momenta = 0.5 * forces

    result = mapping.evaluate(positions, forces=forces, momenta=momenta)

    assert bool(result.successful)
    np.testing.assert_allclose(result.positions, [[1.5, 0.0, 0.0], [0.0, 4.0, 0.0]])
    np.testing.assert_allclose(result.forces, [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    np.testing.assert_allclose(result.momenta, 0.5 * np.asarray(result.forces))
    assert result.mass_residual == 0.0
    assert result.charge_residual == 0.0
    assert not bool(jnp.any(mapping.coarse_system.plan.element_mask))
    assert jnp.array_equal(mapping.coarse_system.plan.atom_type_ids, jnp.asarray([0, 1]))


def test_periodic_map_uses_image_counts_and_rejects_invalid_partition():
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0)
    system = _fine_system(cell=cell)
    mapping = _mapping(system)
    positions = jnp.asarray(
        [[9.8, 0.0, 0.0], [0.2, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 2.0, 0.0]]
    )
    images = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]])

    mapped = mapping.evaluate(positions, image_counts=images)

    assert bool(mapped.successful)
    assert jnp.allclose(mapped.positions[0, 0], 0.1)
    with pytest.raises(ValueError, match="Every active"):
        phx.atomistic.MolecularCoarseMapPlan([100, 200], [0, 1], [0, -1, 1, 1]).prepare(
            system
        )


def test_type_id_potential_accepts_coarse_particles_and_atomic_model_rejects():
    system = _fine_system()
    mapping = _mapping(system)
    positions = jnp.asarray(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]]]
    )
    fine = phx.atomistic.AtomisticBatch(
        jnp.ones((1, 4), dtype=jnp.int32),
        positions,
        jnp.asarray([[1.0, 3.0, 2.0, 2.0]]),
        system.plan.units.scale,
        particle_ids=jnp.asarray([[10, 20, 30, 40]]),
        atom_type_ids=jnp.ones((1, 4), dtype=jnp.int32),
        structure_ids=("fine-frame",),
    )
    forces = jnp.asarray(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]]
    )
    graph = phx.atomistic.AtomisticGraphExecutionPlan(
        1, backend="dense", maximum_dense_atoms=2
    )
    problem = phx.atomistic.CoarseForceMatchingProblem(mapping, fine, forces, graph)
    atomic = phx.nn.atomistic.PaiNNPotential(
        system.plan.units.scale,
        cutoff=6.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        maximum_species_id=1,
        key=jax.random.key(1),
    )
    typed = phx.nn.atomistic.PaiNNPotential(
        system.plan.units.scale,
        cutoff=6.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        maximum_species_id=1,
        species_kind=phx.atomistic.AtomisticSpeciesKind.ATOM_TYPE_ID,
        key=jax.random.key(2),
    )

    with pytest.raises(ValueError, match="atom-type-ID"):
        phx.atomistic.fit_coarse_potential(
            atomic,
            problem,
            phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0, energy_weight=0.0),
            jax.random.key(3),
        )
    fitted = phx.atomistic.fit_coarse_potential(
        typed,
        problem,
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0, energy_weight=0.0),
        jax.random.key(4),
    )
    assert bool(fitted.valid)
    assert jnp.isfinite(fitted.projected_force_rms)
    mapped = mapping.evaluate(positions[0], forces=forces[0])
    qualification = phx.atomistic.qualify_molecular_coarse_model(
        mapped, fitted, 0.01, True
    )
    assert bool(qualification.claims_satisfied)
