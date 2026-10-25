from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._identity import IdentityWords
from phydrax.atomistic._dynamics import (
    AtomisticDynamicsPlan,
    AtomisticStepRejectionReason,
    VelocityVerletPlan,
)
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_dynamics_fixtures import (
    assert_tree_equal,
    measure,
    row,
    table,
)


# Finite two-atom pair: explicit nonperiodic search frame and the one-pair
# storage of the former dense neighborhood.
_RESOURCES: dict[str, Any] = {
    "search": phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
    "skin": 0.5,
    "search_capacities": (search_capacity(8, particles_per_cell=2, images=27),),
    "role_capacities": (role_capacity(1, 1),),
    "frame": phx.discretization.PeriodicCell(
        20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
    ),
}


def _program() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [10, 20],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 2.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 1],
        # ty: ignore[invalid-argument-type]
        region_ids=[1, 0],
    ).prepare()
    lj = phx.atomistic.LennardJonesPotential(
        # ty: ignore[invalid-argument-type]
        [0.5, 1.0],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.2],
        2.5,
        name="lj",
        force_group=0,
    )
    program = prepare_program(
        phx.atomistic.AtomisticPotentialProgram([lj]), system, **_RESOURCES
    )
    return system, lj, program


def test_dynamics_hybrid_scenario_1() -> None:
    system, lj, program = _program()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    masked = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [phx.atomistic.RegionMaskedPotential(lj, 1)]
        ),
        system,
        **_RESOURCES,
    )
    masked_value = masked.evaluate(
        positions, bundle_state(masked, positions), species=system.plan.atom_type_ids
    )
    assert bool(masked_value.successful)
    assert bool(jnp.isfinite(masked_value.energy))
    system, _, program = _program()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    neighborhoods = bundle_state(program, positions)
    group = phx.atomistic.evaluate_force_group(
        program,
        0,
        positions,
        neighborhoods,
        species=system.plan.atom_type_ids,
    )
    assert bool(group.successful)
    transition = phx.atomistic.variance_constrained_semigrand_step(
        program,
        positions,
        neighborhoods,
        system.plan.atom_type_ids,
        jax.random.key_data(jax.random.key(12)),
        0,
        # ty: ignore[invalid-argument-type]
        phx.atomistic.VarianceConstrainedSemiGrandPlan(1.0, [0.0, 0.0], [0.5, 0.5], 1.0),
    )
    assert bool(transition.successful)
    assert transition.species.shape == (2,)
    system, _, program = _program()
    ring = phx.atomistic.PreparedRingPolymerDynamics(
        phx.atomistic.RingPolymerPlan(1, 1.0, 1.0e-4),
        program,
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    state = ring.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(13)
    )
    np.testing.assert_allclose(state.spring_energy, 0.0, atol=0.0)
    step = ring.step(state)
    assert bool(step.successful)
    assert bool(jnp.all(jnp.isfinite(step.estimators.centroid)))
    assert float(step.estimators.radius_of_gyration) == 0.0


def test_born_oppenheimer_provider_boundary_advances_conservative_state() -> None:
    system, _, _ = _program()
    evaluations: list[None] = []

    def evaluator(prepared: Any, positions: Any, cell_vectors: Any) -> Any:
        del prepared, cell_vectors
        evaluations.append(None)
        energy = 0.5 * jnp.sum(positions * positions)
        return phx.atomistic.ExternalAtomisticEvaluation(
            energy,
            -positions,
            None,
            jnp.asarray(True),
            provider_label="harmonic-electronic-surface",
        )

    provider = phx.atomistic.CallableBornOppenheimerProvider(
        evaluator, "harmonic-electronic-surface"
    )
    dynamics = phx.atomistic.BornOppenheimerVelocityVerletPlan(system, provider, 1.0e-3)
    positions = jnp.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    state = dynamics.initialize(positions, velocity=jnp.zeros_like(positions))
    step = dynamics.step(state)
    assert bool(step.successful)
    assert int(step.state.step_index) == 1
    assert bool(jnp.isfinite(step.state.energy))
    # A labelled provider keeps its exact declared label everywhere.
    assert provider.provider_identity is None
    assert provider.provider_id == "harmonic-electronic-surface"
    for value in (step.state, step.initial_evaluation, step.final_evaluation):
        assert value.provider_identity is None
        assert value.provider_label == "harmonic-electronic-surface"
        assert value.provider_id == "harmonic-electronic-surface"
    # Foreign labels are rejected before invoking an external evaluator.
    other = phx.atomistic.BornOppenheimerVelocityVerletPlan(
        system,
        phx.atomistic.CallableBornOppenheimerProvider(evaluator, "other-surface"),
        1.0e-3,
    )
    with pytest.raises(ValueError):
        other.initialize(positions, velocity=jnp.zeros_like(positions))
    previous_evaluations = len(evaluations)
    with pytest.raises(ValueError):
        other.step(state)
    assert len(evaluations) == previous_evaluations


def test_external_evaluation_binds_exactly_one_provider_kind() -> None:
    forces = jnp.zeros((2, 3))
    digest = "0123456789abcdef" * 4
    with pytest.raises(ValueError, match="exactly one"):
        phx.atomistic.ExternalAtomisticEvaluation(0.0, forces, None, jnp.asarray(True))
    with pytest.raises(ValueError, match="exactly one"):
        phx.atomistic.ExternalAtomisticEvaluation(
            0.0,
            forces,
            None,
            jnp.asarray(True),
            provider_identity=IdentityWords.from_fingerprint(digest),
            provider_label="label",
        )
    # A digest-shaped label stays a label: the kind is declared, never inferred.
    labelled = phx.atomistic.ExternalAtomisticEvaluation(
        0.0, forces, None, jnp.asarray(True), provider_label=digest
    )
    assert labelled.provider_identity is None and labelled.provider_label == digest
    digested = phx.atomistic.ExternalAtomisticEvaluation(
        0.0,
        forces,
        None,
        jnp.asarray(True),
        provider_identity=IdentityWords.from_fingerprint(digest),
    )
    assert digested.provider_label is None and digested.provider_id == digest
    assert jax.tree.structure(labelled) != jax.tree.structure(digested)


_RESPA_REVISIONS = {"1": ([0.5, 1.0], [1.0, 1.2]), "2": ([0.6, 0.9], [1.05, 1.15])}


def _respa_runtime(revision: str) -> tuple[Any, Any]:
    """NVE runtime of one parameter revision; fast and slow LJ force groups."""
    system, _, _ = _program()
    epsilon, sigma = _RESPA_REVISIONS[revision]
    terms = tuple(
        phx.atomistic.LennardJonesPotential(
            np.asarray(epsilon), np.asarray(sigma), 2.5, name=name, force_group=group
        )
        for name, group in (("fast", 0), ("slow", 1))
    )
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(terms), system, **_RESOURCES
    )
    dynamics = AtomisticDynamicsPlan(
        system, potential, VelocityVerletPlan(1.0e-3)
    ).prepare()
    return dynamics, table(dynamics, row(measure(system), "nve"))


def test_respa_refuses_a_foreign_runtime_revision_by_binding_status() -> None:
    owner, owner_table = _respa_runtime("1")
    foreign, foreign_table = _respa_runtime("2")
    assert owner.layout_id == foreign.layout_id
    assert owner.prepared_id != foreign.prepared_id
    assert owner_table.table_id != foreign_table.table_id
    plan = phx.atomistic.RESPAPlan(1.0e-3, 2)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]])
    velocity = jnp.zeros_like(positions)
    state = owner.initialize_state(
        positions, owner_table, velocity=velocity, key=jax.random.key(0)
    )
    accepted = phx.atomistic.respa_step(owner, state, owner_table, plan)
    assert bool(accepted.successful), int(accepted.rejection_reasons)
    assert int(accepted.state.step_index) == 1
    foreign_state = foreign.initialize_state(
        positions, foreign_table, velocity=velocity, key=jax.random.key(0)
    )
    # A state of another revision, and a thermodynamic table of another revision.
    for candidate, thermodynamic in (
        (foreign_state, owner_table),
        (state, foreign_table),
    ):
        evaluation = phx.atomistic.respa_step(owner, candidate, thermodynamic, plan)
        assert not bool(evaluation.successful)
        assert int(evaluation.rejection_reasons) & int(
            AtomisticStepRejectionReason.BINDING
        )
        assert_tree_equal(evaluation.state, candidate)
