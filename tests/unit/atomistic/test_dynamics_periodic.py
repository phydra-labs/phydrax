from itertools import product
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.units import COULOMB, KELVIN, KILOGRAM, SECOND


def _cell() -> Any:
    return phx.discretization.PeriodicCell(
        # ty: ignore[invalid-argument-type]
        [[3.0, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.1, 3.1]]
    )


def test_dynamics_periodic_scenario_1() -> None:
    cell = _cell()
    displacement = jnp.asarray([2.7, 1.9, -2.2])
    observed = cell.minimum_image(displacement)
    vectors = np.asarray(cell.vectors)
    candidates = np.asarray(
        [
            np.asarray(displacement) - np.asarray(shift) @ vectors
            for shift in product(range(-3, 4), repeat=3)
        ]
    )
    expected = candidates[np.argmin(np.sum(candidates * candidates, axis=1))]
    np.testing.assert_allclose(observed, expected, atol=1.0e-12)
    cell = _cell()
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1, 2, 3],
        # ty: ignore[invalid-argument-type]
        [1, 1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0, 0, 0],
        cell=cell,
    ).prepare()
    fractional = jnp.asarray(
        [[0.05, 0.05, 0.05], [0.95, 0.05, 0.05], [0.5, 0.5, 0.5], [0.6, 0.5, 0.5]]
    )
    positions = cell.cartesian(fractional)
    metric = phx.discretization.MetricCellListParticleNeighborhoodPlan(
        0.7, 4, 6, cell
    ).prepare(system.particles)
    dense = phx.discretization.DenseParticleNeighborhoodPlan(6, box=cell).prepare(
        system.particles
    )
    metric_state = metric.build(positions)
    dense_state = dense.build(positions)
    dense_geometry = phx.discretization.particle_pair_geometry(
        positions, dense_state.pair_relation, box=cell
    )
    dense_pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right, distance in zip(
            np.asarray(dense_state.pair_relation.left_particle_ids),
            np.asarray(dense_state.pair_relation.right_particle_ids),
            np.asarray(dense_geometry.distance),
            strict=True,
        )
        if distance < 0.7
    }
    metric_pairs = {
        tuple(sorted((int(left), int(right))))
        for left, right, valid in zip(
            np.asarray(metric_state.pair_relation.left_particle_ids),
            np.asarray(metric_state.pair_relation.right_particle_ids),
            np.asarray(metric_state.pair_relation.valid),
            strict=True,
        )
        if valid
    }
    assert metric_pairs == dense_pairs
    cell = _cell()
    particles = phx.discretization.ParticleSetPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        ambient_dimension=3,
    ).prepare()
    base = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell)
    verlet = phx.discretization.VerletParticleNeighborhoodPlan(base, 0.8, 0.2).prepare(
        particles
    )
    positions = cell.cartesian(jnp.asarray([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]))
    state = verlet.initialize(positions, cell_vectors=cell.vectors)
    deformed = cell.vectors.at[0, 0].add(0.15)
    updated = verlet.update(positions, state, cell_vectors=deformed)
    assert bool(updated.rebuilt)
    assert float(updated.maximum_cell_deformation) > 0.1


def test_dynamics_periodic_scenario_2() -> None:
    cell = _cell()
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        cell=cell,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 1.2)]
    ).prepare(system)
    fractional = jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]])
    positions = cell.cartesian(fractional)
    relation = neighborhood.build(positions)
    result = phx.atomistic.atomistic_cell_energy_and_stress(
        potential, fractional, relation
    )
    assert bool(result.successful)
    assert bool(jnp.all(jnp.isfinite(result.stress)))
    np.testing.assert_allclose(result.stress, result.stress.T, atol=1.0e-12)
    model_units = (
        phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    )
    units = phx.atomistic.AtomisticUnitSystem(
        model_units.scale,
        mass_unit=KILOGRAM,
        time_unit=SECOND,
        charge_unit=COULOMB,
        temperature_unit=KELVIN,
        constant_set_id="codata-2018",
    )
    cell = phx.discretization.PeriodicCell(5.0 * jnp.eye(3))
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[1, 1],
        cell=cell,
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1, box=cell).prepare(
        system.particles
    )
    model = phx.nn.atomistic.PaiNNPotential(
        model_units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(77),
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LearnedGraphPotentialTerm(model, allow_periodic=True)]
    ).prepare(
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(1, backend="particle"),
    )
    positions = jnp.asarray([[0.2, 0.2, 0.2], [4.4, 0.2, 0.2]])
    relation = neighborhood.build(positions)
    result = program.evaluate(positions, relation, species=system.plan.atomic_numbers)
    assert bool(result.successful)
    assert bool(jnp.isfinite(result.energy))
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[1, 1],
    ).prepare()
    model = phx.nn.atomistic.PaiNNPotential(
        units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(77),
    )
    program = phx.atomistic.AtomisticPotentialProgram(
        [_CutoffFreeGraphTerm(phx.atomistic.LearnedGraphPotentialTerm(model))]
    )
    with pytest.raises(ValueError, match="require a cutoff"):
        program.prepare(
            system,
            graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(
                1, backend="particle"
            ),
        )


class _CutoffFreeGraphTerm(phx.atomistic.AbstractAtomisticEnergyTerm):
    """Consumer graph term that declares a directed graph but no cutoff."""

    learned: phx.atomistic.LearnedGraphPotentialTerm
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: phx.atomistic.AtomisticPotentialCapabilities
    requirements: phx.atomistic.AtomisticPotentialRequirements

    def __init__(self, learned: Any) -> None:
        self.learned = learned
        self.name = learned.name
        self.force_group = learned.force_group
        self.term_id = f"cutoff-free-{learned.term_id}"
        self.capabilities = learned.capabilities
        self.requirements = phx.atomistic.AtomisticPotentialRequirements(
            pair_geometry=True, directed_graph=True
        )

    def prepare(self, system: Any, /) -> Any:
        return self.learned.prepare(system)


class _EdgeExponentialTerm(phx.atomistic.AbstractAtomisticEnergyTerm):
    """Pure directed-graph energy ``sum_e exp(-|d_e|)`` over image routes."""

    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: phx.atomistic.AtomisticPotentialCapabilities
    requirements: phx.atomistic.AtomisticPotentialRequirements

    def __init__(self, cutoff: float) -> None:
        self.name = "edge-exponential"
        self.force_group = 0
        self.term_id = f"edge-exponential-{cutoff}"
        self.capabilities = phx.atomistic.AtomisticPotentialCapabilities(
            orthorhombic_periodic=True, triclinic_periodic=True, cell_derivative=True
        )
        self.requirements = phx.atomistic.AtomisticPotentialRequirements(
            cutoff=cutoff, directed_graph=True
        )

    def prepare(self, system: Any, /) -> Any:
        return _PreparedEdgeExponentialTerm(self)


class _PreparedEdgeExponentialTerm(phx.atomistic.AbstractPreparedAtomisticEnergyTerm):
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: phx.atomistic.AtomisticPotentialCapabilities
    requirements: phx.atomistic.AtomisticPotentialRequirements

    def __init__(self, plan: _EdgeExponentialTerm) -> None:
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.prepared_id = f"prepared-{plan.term_id}"
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements

    def energy(self, context: Any, /) -> Any:
        graph = context.graph
        distance = graph.graph.edges["distance"][:, 0]
        edge = jnp.where(graph.graph.edge_mask, jnp.exp(-distance), 0.0)
        atoms = context.positions.shape[0]
        atom_energy = (
            jnp.zeros((atoms,), dtype=distance.dtype).at[graph.graph.receivers].add(edge)
        )
        return phx.atomistic.AtomisticTermEvaluation(
            jnp.sum(edge), atom_energy, ~jnp.any(graph.overflow)
        )


def _image_runtime(capacity: Any, cutoff: float = 2.4, skin: float = 0.3) -> Any:
    cell = phx.discretization.PeriodicCell(
        # ty: ignore[invalid-argument-type]
        [[2.0, 0.0, 0.0], [0.3, 2.1, 0.0], [0.1, -0.2, 2.2]]
    )
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
        cell=cell,
    ).prepare()
    program = phx.atomistic.AtomisticPotentialProgram(
        [_EdgeExponentialTerm(cutoff)]
    ).prepare(
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(
            256, backend="particle"
        ),
    )
    base = phx.discretization.CellListParticleImageNeighborhoodPlan(
        cutoff + skin, cell, capacity
    )
    neighborhood = phx.discretization.ImageVerletParticleNeighborhoodPlan(
        base, cutoff, skin
    ).prepare(system.particles)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, program, neighborhood, phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0)
    ).prepare()
    states = (
        phx.atomistic.AtomisticThermodynamicStatePlan(
            phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system),
            ensemble="nvt",
            temperature=0.5,
        ),
    )
    return (
        cell,
        dynamics,
        phx.atomistic.PreparedThermodynamicStateTable(dynamics, states),
        states,
    )


_LARGE = phx.discretization.ParticleImageCapacity(
    maximum_particles_per_cell=4,
    maximum_edges=512,
    maximum_degree=256,
    maximum_images=512,
)
_SMALL = phx.discretization.ParticleImageCapacity(
    maximum_particles_per_cell=4,
    maximum_edges=4,
    maximum_degree=256,
    maximum_images=512,
)


def test_image_dynamics_beyond_unique_radius_matches_independent_lattice_energy() -> None:
    cell, dynamics, thermodynamic, _ = _image_runtime(_LARGE)
    assert 2.4 > cell.unique_image_radius
    vectors = np.asarray(cell.vectors)
    unwrapped = jnp.asarray([[0.2, 0.3, 0.1], [2.1, 1.9, 2.4]])
    state = dynamics.initialize_state(
        unwrapped,
        thermodynamic,
        velocity=jnp.zeros_like(unwrapped),
        key=jr.key(0),
    )
    positions = np.asarray(state.kinematics.positions)
    expected = 0.0
    for shift in product(range(-4, 5), repeat=3):
        translation = np.asarray(shift) @ vectors
        for source, receiver in product(range(2), repeat=2):
            if source == receiver and not any(shift):
                continue
            distance = np.linalg.norm(
                positions[receiver] - positions[source] + translation
            )
            if distance < 2.4:
                expected += np.exp(-distance)
    np.testing.assert_allclose(state.force.potential_energy, expected, rtol=1e-12)
    np.testing.assert_allclose(jnp.sum(state.force.forces, axis=0), 0.0, atol=1e-12)
    diagnostics = dynamics.diagnostics(state, thermodynamic)
    assert float(diagnostics.image_uniqueness_margin) == float("inf")
    step = eqx.filter_jit(dynamics.step_detailed)(state, thermodynamic)
    assert bool(step.successful)


def test_capacity_failure_retries_same_attempt_without_advancing_state() -> None:
    _, dynamics, thermodynamic, states = _image_runtime(_LARGE)
    unwrapped = jnp.asarray([[0.2, 0.3, 0.1], [1.1, 1.0, 1.2]])
    state = dynamics.initialize_state(
        unwrapped,
        thermodynamic,
        velocity=jnp.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
        key=jr.key(4),
    )
    reference = dynamics.step_detailed(state, thermodynamic)
    assert bool(reference.successful)
    small_neighborhood = dynamics.neighborhood.plan.with_capacity(_SMALL).prepare(
        dynamics.system.particles
    )
    small, small_table, small_state = dynamics.rebind_neighborhood(
        small_neighborhood, state, thermodynamic, states
    )
    np.testing.assert_array_equal(small_state.random_key, state.random_key)
    np.testing.assert_array_equal(small_state.force.forces, state.force.forces)
    assert int(small_state.step_index) == int(state.step_index)
    rejected = small.step_detailed(small_state, small_table)
    assert not bool(rejected.successful)
    assert int(rejected.rejection_reasons) & int(
        phx.atomistic.AtomisticStepRejectionReason.PAIR_CAPACITY
    )
    assert int(rejected.accepted_state.step_index) == int(small_state.step_index)
    np.testing.assert_array_equal(
        rejected.accepted_state.kinematics.positions, small_state.kinematics.positions
    )
    ladder = phx.discretization.ParticleImageCapacityLadder((_SMALL, _LARGE))
    grown, grown_table, evaluation = phx.atomistic.retry_atomistic_step_with_capacity(
        small, small_state, small_table, states, ladder
    )
    assert bool(evaluation.successful)
    grown_neighborhood = grown.neighborhood
    assert isinstance(
        grown_neighborhood, phx.discretization.PreparedImageVerletParticleNeighborhood
    )
    assert grown_neighborhood.capacity.capacity_id == _LARGE.capacity_id
    assert grown_table.dynamics_id == grown.prepared_id
    np.testing.assert_allclose(
        evaluation.accepted_state.kinematics.positions,
        reference.accepted_state.kinematics.positions,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        evaluation.accepted_state.kinematics.momenta,
        reference.accepted_state.kinematics.momenta,
        rtol=1e-12,
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        phx.atomistic.retry_atomistic_step_with_capacity(
            small,
            small_state,
            small_table,
            states,
            phx.discretization.ParticleImageCapacityLadder((_SMALL,)),
        )


def test_image_neighborhood_refuses_classical_pair_programs() -> None:
    cell, dynamics, _, _ = _image_runtime(_LARGE)
    classical = phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 0.9)]
    ).prepare(dynamics.system)
    with pytest.raises(ValueError, match="pure directed-graph"):
        phx.atomistic.AtomisticDynamicsPlan(
            dynamics.system,
            classical,
            dynamics.neighborhood,
            phx.atomistic.VelocityVerletPlan(1.0e-3),
        )


def test_reused_schedule_must_match_topology_routes_not_only_owner_and_epoch() -> None:
    _, dynamics, _, _ = _image_runtime(_LARGE)
    prepared = dynamics.neighborhood.base
    system = dynamics.system
    plan = phx.atomistic.AtomisticGraphExecutionPlan(256, backend="particle").streamed
    near = prepared.build(jnp.asarray([[0.2, 0.3, 0.1], [0.9, 0.6, 0.5]]))
    far = prepared.build(jnp.asarray([[0.2, 0.3, 0.1], [1.2, 1.1, 1.3]]))
    assert near.relation_schema_id == far.relation_schema_id
    assert not np.array_equal(
        np.asarray(near.relation.source_indices)[np.asarray(near.relation.valid)],
        np.asarray(far.relation.source_indices)[np.asarray(far.relation.valid)],
    ) or not np.array_equal(
        np.asarray(near.relation.image_shifts), np.asarray(far.relation.image_shifts)
    )
    first = phx.atomistic.particle_atomistic_graph_topology(system, near, streamed=plan)
    reused = phx.atomistic.particle_atomistic_graph_topology(
        system, near, streamed=first.streamed
    )
    np.testing.assert_array_equal(reused.overflow, [False])
    stale = phx.atomistic.particle_atomistic_graph_topology(
        system, far, streamed=first.streamed
    )
    np.testing.assert_array_equal(stale.overflow, [True])
    fresh = phx.atomistic.particle_atomistic_graph_topology(system, far, streamed=plan)
    np.testing.assert_array_equal(fresh.overflow, [False])


def test_supplied_topology_binds_current_routes_and_execution_plan() -> None:
    _, dynamics, _, _ = _image_runtime(_LARGE)
    program = dynamics.potential
    prepared = dynamics.neighborhood.base
    system = dynamics.system
    near = prepared.build(jnp.asarray([[0.2, 0.3, 0.1], [0.9, 0.6, 0.5]]))
    far_positions = jnp.asarray([[0.2, 0.3, 0.1], [1.2, 1.1, 1.3]])
    far = prepared.build(far_positions)
    fresh = program.evaluate(far_positions, far)
    assert bool(fresh.successful)
    streamed = program.graph_execution.streamed
    own = phx.atomistic.particle_atomistic_graph_topology(system, far, streamed=streamed)
    reused = program.evaluate(far_positions, far, topology=own)
    np.testing.assert_array_equal(reused.energy, fresh.energy)
    np.testing.assert_array_equal(reused.forces, fresh.forces)
    # Same relation schema and shapes, routes of another search: refused.
    stale = phx.atomistic.particle_atomistic_graph_topology(
        system, near, streamed=streamed
    )
    rejected = program.evaluate(far_positions, far, topology=stale)
    assert not bool(rejected.successful)
    assert bool(jnp.isnan(rejected.energy))
    other_budget = phx.atomistic.particle_atomistic_graph_topology(
        system,
        far,
        streamed=phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=8),
    )
    with pytest.raises(ValueError, match="another streamed plan"):
        program.evaluate(far_positions, far, topology=other_budget)


def test_image_state_lattice_is_default_and_overrides_need_its_certificate() -> None:
    cell, dynamics, _, _ = _image_runtime(_LARGE)
    program = dynamics.potential
    prepared = dynamics.neighborhood.base
    positions = jnp.asarray([[0.2, 0.3, 0.1], [1.2, 1.1, 1.3]])
    deformed_vectors = 0.99 * cell.vectors
    deformed = prepared.build(positions, cell_vectors=deformed_vectors)
    default = program.evaluate(positions, deformed, compute_stress=True)
    explicit = program.evaluate(
        positions, deformed, compute_stress=True, cell_vectors=deformed_vectors
    )
    assert bool(default.successful)
    np.testing.assert_array_equal(default.energy, explicit.energy)
    np.testing.assert_array_equal(default.forces, explicit.forces)
    np.testing.assert_array_equal(default.stress, explicit.stress)
    state = prepared.build(positions)
    # Inside the search skin the cached routes are complete: exact parity.
    nearby_vectors = 1.001 * cell.vectors
    certified = program.evaluate(positions, state, cell_vectors=nearby_vectors)
    rebuilt = program.evaluate(
        positions, prepared.build(positions, cell_vectors=nearby_vectors)
    )
    assert bool(certified.successful)
    np.testing.assert_allclose(certified.energy, rebuilt.energy, rtol=1e-12)
    np.testing.assert_allclose(certified.forces, rebuilt.forces, atol=1e-12)
    # A compressed lattice or moved atom admits images the state never searched.
    compressed = program.evaluate(positions, state, cell_vectors=0.8 * cell.vectors)
    moved = program.evaluate(positions.at[1].add(0.6), state)
    for refused in (compressed, moved):
        assert not bool(refused.successful)
        assert bool(jnp.isnan(refused.energy))


def test_capacity_retry_preserves_independent_thermodynamic_rejection() -> None:
    _, dynamics, table, states = _image_runtime(_LARGE)
    positions = jnp.asarray([[0.2, 0.3, 0.1], [1.1, 1.0, 1.2]], dtype=jnp.float64)
    state = dynamics.initialize_state(
        positions, table, velocity=jnp.zeros_like(positions), key=jr.key(4)
    )
    small_neighborhood = dynamics.neighborhood.plan.with_capacity(_SMALL).prepare(
        dynamics.system.particles
    )
    small, small_table, accepted = dynamics.rebind_neighborhood(
        small_neighborhood, state, table, states
    )
    invalid = eqx.tree_at(
        lambda value: value.thermodynamic_state_index,
        accepted,
        jnp.asarray(9, dtype=jnp.int32),
    )
    rejected = small.step_detailed(invalid, small_table)
    reasons = int(rejected.rejection_reasons)
    assert reasons & int(phx.atomistic.AtomisticStepRejectionReason.PAIR_CAPACITY)
    assert reasons & int(phx.atomistic.AtomisticStepRejectionReason.THERMODYNAMIC_STATE)
    returned, returned_table, evaluation = (
        phx.atomistic.retry_atomistic_step_with_capacity(
            small,
            invalid,
            small_table,
            states,
            phx.discretization.ParticleImageCapacityLadder((_SMALL,)),
        )
    )
    assert returned.prepared_id == small.prepared_id
    assert returned_table.table_id == small_table.table_id
    assert not bool(evaluation.successful)
    assert int(evaluation.rejection_reasons) == reasons
    np.testing.assert_array_equal(
        evaluation.accepted_state.kinematics.positions,
        rejected.accepted_state.kinematics.positions,
    )
    np.testing.assert_array_equal(
        evaluation.accepted_state.random_key, invalid.random_key
    )
