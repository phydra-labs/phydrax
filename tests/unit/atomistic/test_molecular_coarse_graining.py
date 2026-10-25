#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _fine_system(*, cell: Any = None) -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    return phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20, 30, 40],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 3.0, 2.0, 2.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[1, 1, 1, 1],
        # ty: ignore[invalid-argument-type]
        charges=[0.1, -0.1, 0.2, -0.2],
        # ty: ignore[invalid-argument-type]
        molecule_ids=[0, 0, 1, 1],
        cell=cell,
    ).prepare()


def _mapping(system: Any) -> Any:
    # ty: ignore[invalid-argument-type]
    return phx.atomistic.MolecularCoarseMapPlan([100, 200], [0, 1], [0, 0, 1, 1]).prepare(
        system
    )


def test_molecular_coarse_graining_scenario_1() -> None:
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
        # ty: ignore[invalid-argument-type]
        phx.atomistic.MolecularCoarseMapPlan([100, 200], [0, 1], [0, -1, 1, 1]).prepare(
            system
        )
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
    problem = phx.atomistic.CoarseForceMatchingProblem(
        mapping, fine, forces, graph, cutoff=6.0
    )
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
            phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0),
            jax.random.key(3),
        )
    fitted = phx.atomistic.fit_coarse_potential(
        typed,
        problem,
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=0),
        jax.random.key(4),
    )
    assert bool(fitted.valid)
    assert jnp.isfinite(fitted.projected_force_rms)
    mapped = mapping.evaluate(positions[0], forces=forces[0])
    qualification = phx.atomistic.qualify_molecular_coarse_model(
        mapped, fitted, 0.01, True
    )
    assert bool(qualification.claims_satisfied)


def _fine_batch(system: Any, positions: Any, structure_id: str) -> Any:
    return phx.atomistic.AtomisticBatch(
        jnp.ones((1, 4), dtype=jnp.int32),
        jnp.asarray([positions]),
        jnp.asarray([[1.0, 3.0, 2.0, 2.0]]),
        system.plan.units.scale,
        particle_ids=jnp.asarray([[10, 20, 30, 40]]),
        atom_type_ids=jnp.ones((1, 4), dtype=jnp.int32),
        structure_ids=(structure_id,),
    )


def test_coarse_force_matching_lowers_force_only_canonical_problem() -> None:
    system = _fine_system()
    mapping = _mapping(system)
    fine = _fine_batch(
        system,
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.0, 0.0]],
        "fine-frame",
    )
    forces = jnp.asarray(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]]
    )
    prior = jnp.asarray([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    validation = _fine_batch(
        system,
        [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.4, 0.0]],
        "fine-validation",
    )
    graph = phx.atomistic.AtomisticGraphExecutionPlan(
        1, backend="dense", maximum_dense_atoms=2
    )
    problem = phx.atomistic.CoarseForceMatchingProblem(
        mapping,
        fine,
        forces,
        graph,
        cutoff=6.0,
        prior_forces=prior,
        prior_id="harmonic-prior",
        validation_batch=validation,
        validation_fine_forces=forces,
        validation_prior_forces=prior,
    )

    # Canonical source: atom-type-ID beads, coarse masses, residual force labels.
    source = problem.training_source
    schema = source.schema
    assert schema.field("species").convention == "atom-type-id"
    assert "energy" not in schema.field_names
    assert "stress" not in schema.field_names
    assert source.lineage_ids == ("fine-frame",)
    rows = source.read([0])
    np.testing.assert_array_equal(rows.values["species"], [0, 1])
    np.testing.assert_allclose(rows.values["masses"], [4.0, 4.0])
    np.testing.assert_allclose(
        rows.values["positions"], [[1.5, 0.0, 0.0], [0.0, 4.0, 0.0]]
    )
    np.testing.assert_allclose(rows.values["forces"], [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    np.testing.assert_array_equal(rows.stable_ids, [100, 200])
    assert problem.validation_source is not None
    assert problem.validation_source.lineage_ids == ("fine-validation",)
    # The lowered problem is force-only over the bead species kind.
    training = problem.training
    assert training.label_slots == ("forces",)
    assert len(training.objectives) == 1
    assert isinstance(training.objectives[0], phx.atomistic.AtomisticForceObjective)
    assert training.fields.species_kind is phx.atomistic.AtomisticSpeciesKind.ATOM_TYPE_ID
    assert training.validation is not None
    assert training.split_evidence is not None
    # Projected and prior-corrected force-scale evidence.
    np.testing.assert_allclose(
        problem.projected_forces, [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]]
    )
    np.testing.assert_allclose(
        problem.residual_forces, [[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]
    )

    typed = phx.nn.atomistic.PaiNNPotential(
        system.plan.units.scale,
        cutoff=6.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        maximum_species_id=1,
        species_kind=phx.atomistic.AtomisticSpeciesKind.ATOM_TYPE_ID,
        key=jax.random.key(5),
    )
    fitted = phx.atomistic.fit_coarse_potential(
        typed,
        problem,
        phx.atomistic.AtomisticTrainingPolicy(maximum_steps=1, learning_rate=1e-3),
        jax.random.key(6),
    )
    assert bool(fitted.valid)
    assert fitted.problem_id == problem.problem_id
    assert fitted.training.problem_id == training.problem_id
    assert fitted.training.training_objectives.shape == (1, 1)
    np.testing.assert_allclose(fitted.projected_force_rms, np.sqrt(3.0))
    np.testing.assert_allclose(fitted.residual_force_rms, np.sqrt(4.0 / 3.0))

    # Validation configurations must not share the training lineage.
    leaked = _fine_batch(
        system,
        [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 5.4, 0.0]],
        "fine-frame",
    )
    with pytest.raises(ValueError, match="lineage"):
        phx.atomistic.CoarseForceMatchingProblem(
            mapping,
            fine,
            forces,
            graph,
            cutoff=6.0,
            validation_batch=leaked,
            validation_fine_forces=forces,
        )
