from itertools import product
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._identity import IdentityWords
from phydrax.atomistic._cell_dynamics import bind_atomistic_bundle_geometry
from phydrax.atomistic._observables import (
    RadialDistributionPlan,
    RadialDistributionState,
)
from phydrax.units import COULOMB, KELVIN, KILOGRAM, SECOND
from tests._support.atomistic_bundles import (
    bundle_state,
    neighborhood_bundle_plan,
    prepare_program,
    role_capacity,
    search_capacity,
)


Role = phx.atomistic.AtomisticNeighborhoodRole
Reason = phx.atomistic.AtomisticStepRejectionReason


def _cell() -> Any:
    return phx.discretization.PeriodicCell(
        # ty: ignore[invalid-argument-type]
        [[3.0, 0.0, 0.0], [0.4, 2.8, 0.0], [0.2, 0.1, 3.1]]
    )


def _pair_resources(routes: int, degree: int) -> dict[str, Any]:
    """Explicit two-atom resources: one cell-list source and one role entry."""
    return {
        "search": phx.atomistic.AtomisticCellListImageSearch(
            maximum_candidate_slots=4096
        ),
        "skin": 0.3,
        "search_capacities": (search_capacity(16, particles_per_cell=2, images=27),),
        "role_capacities": (role_capacity(routes, degree),),
    }


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
    ).prepare(system.particle_discretization())
    dense = phx.discretization.DenseParticleNeighborhoodPlan(6, box=cell).prepare(
        system.particle_discretization()
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
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([1.0], [1.0], 1.2)]
        ),
        system,
        **_pair_resources(1, 1),
    )
    fractional = jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]])
    positions = cell.cartesian(fractional)
    result = phx.atomistic.atomistic_cell_energy_and_stress(
        potential, fractional, bundle_state(potential, positions)
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
    model = phx.nn.atomistic.PaiNNPotential(
        model_units.scale,
        cutoff=2.0,
        feature_count=4,
        interaction_count=1,
        radial_basis_count=3,
        key=jr.key(77),
    )
    # Two directed routes of the single minimum-image pair, one per receiver.
    program = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [phx.atomistic.LearnedGraphPotentialTerm(model, allow_periodic=True)]
        ),
        system,
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(1, backend="particle"),
        **_pair_resources(2, 1),
    )
    positions = jnp.asarray([[0.2, 0.2, 0.2], [4.4, 0.2, 0.2]])
    result = program.evaluate(
        positions,
        bundle_state(program, positions),
        species=system.plan.atomic_numbers,
    )
    assert bool(result.successful)
    assert bool(jnp.isfinite(result.energy))
    # A directed image graph is defined only under a finite cutoff; only pair
    # roles declare the complete cutoff-free relation.
    with pytest.raises(ValueError, match="complete finite all-pairs"):
        phx.atomistic.AtomisticNeighborhoodRequirement(
            model.architecture_id, Role.DIRECTED_IMAGE_GRAPH, None
        )


class _EdgeExponentialTerm(phx.atomistic.AbstractAtomisticEnergyTerm):
    """Pure directed-graph energy ``sum_e exp(-|d_e|)`` over image routes."""

    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    requirement: phx.atomistic.AtomisticNeighborhoodRequirement
    capabilities: phx.atomistic.AtomisticPotentialCapabilities
    requirements: phx.atomistic.AtomisticPotentialRequirements

    def __init__(self, cutoff: float) -> None:
        self.name = "edge-exponential"
        self.force_group = 0
        self.term_identity = IdentityWords.of(f"edge-exponential-{cutoff}")
        self.requirement = phx.atomistic.AtomisticNeighborhoodRequirement(
            self.name, Role.DIRECTED_IMAGE_GRAPH, cutoff
        )
        self.capabilities = phx.atomistic.AtomisticPotentialCapabilities(
            orthorhombic_periodic=True, triclinic_periodic=True, cell_derivative=True
        )
        self.requirements = phx.atomistic.AtomisticPotentialRequirements(
            neighborhoods=(self.requirement,)
        )

    def prepare(self, system: Any, /) -> Any:
        return _PreparedEdgeExponentialTerm(self)


class _PreparedEdgeExponentialTerm(phx.atomistic.AbstractPreparedAtomisticEnergyTerm):
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_identity: IdentityWords
    prepared_identity: IdentityWords
    requirement: phx.atomistic.AtomisticNeighborhoodRequirement
    capabilities: phx.atomistic.AtomisticPotentialCapabilities
    requirements: phx.atomistic.AtomisticPotentialRequirements

    def __init__(self, plan: _EdgeExponentialTerm) -> None:
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_identity = plan.term_identity
        self.prepared_identity = IdentityWords.of(f"prepared-{plan.term_id}")
        self.requirement = plan.requirement
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements

    def energy(self, context: Any, /) -> Any:
        view = context.neighborhood(self.requirement)
        positions = context.positions
        vectors = jnp.asarray(context.cell_vectors).astype(positions.dtype)
        displacement = view.displacement(positions, vectors)
        mask = view.cutoff_mask(positions, vectors)
        safe = jnp.where(mask[:, None], displacement, 1.0)
        edge = jnp.where(mask, jnp.exp(-jnp.sqrt(jnp.sum(safe * safe, axis=-1))), 0.0)
        atom_energy = (
            jnp.zeros((positions.shape[0],), dtype=edge.dtype)
            .at[view.relation.receiver_indices]
            .add(edge)
        )
        return phx.atomistic.AtomisticTermEvaluation(
            jnp.sum(edge), atom_energy, jnp.all(view.evidence.successful)
        )


# Explicit image resources of the beyond-unique-radius graph runtime.  The
# source search storage is the former image capacity; the directed role
# ladder starts at ``_SMALL_ROLE`` where a ladder is declared.
_IMAGE_SEARCH = phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=10**6)
_SEARCH_CAPACITY = search_capacity(512, particles_per_cell=4, images=512)
_SMALL_ROLE = role_capacity(27, 256)
_LARGE_ROLE = role_capacity(512, 256)
# Configuration ``A`` stores 26 directed routes within the 2.4 cutoff and the
# 2.7 search radius (12 self images, 14 pair images); one step with these
# opposite velocities moves the pair by 0.5 along ``_DIRECTION`` to a
# configuration needing 28, so ``_SMALL_ROLE`` overflows only on the step.
_START = np.asarray([[1.5, 0.3, 1.0], [0.137, 0.991, 0.42]])
_DIRECTION = np.asarray([0.226, 0.4386, -0.8698])
_VELOCITY = 250.0 * np.stack((-_DIRECTION, _DIRECTION)) / np.linalg.norm(_DIRECTION)


def _image_runtime(
    role_capacities: tuple[Any, ...] = (_LARGE_ROLE,),
    cutoff: float = 2.4,
    skin: float = 0.3,
    *,
    graph_execution: Any = None,
) -> Any:
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
    program = prepare_program(
        phx.atomistic.AtomisticPotentialProgram([_EdgeExponentialTerm(cutoff)]),
        system,
        graph_execution=(
            phx.atomistic.AtomisticGraphExecutionPlan(256, backend="particle")
            if graph_execution is None
            else graph_execution
        ),
        search=_IMAGE_SEARCH,
        skin=skin,
        search_capacities=(_SEARCH_CAPACITY,),
        role_capacities=role_capacities,
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, program, phx.atomistic.BAOABLangevinPlan(1.0e-3, 1.0)
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


def _geometry(dynamics: Any, positions: Any, vectors: Any = None) -> Any:
    return bind_atomistic_bundle_geometry(
        dynamics.bundle, dynamics.system, positions, cell_vectors=vectors
    )


def test_image_dynamics_beyond_unique_radius_matches_independent_lattice_energy() -> None:
    cell, dynamics, thermodynamic, _ = _image_runtime()
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
    assert bool(diagnostics.successful)
    # A directed image graph keeps every image beyond the unique-image radius,
    # including nonzero self images; it carries no minimum-image refusal.
    view = state.neighborhoods.views[0]
    assert view.role is Role.DIRECTED_IMAGE_GRAPH
    assert bool(jnp.any(view.mask & view.self_image))
    assert not bool(jnp.any(view.evidence.repeated_minimum_image))
    step = eqx.filter_jit(dynamics.step_detailed)(state, thermodynamic)
    assert bool(step.successful)


def test_capacity_failure_retries_same_attempt_without_advancing_state() -> None:
    _, large, large_table, _ = _image_runtime()
    reference_state = large.initialize_state(
        _START, large_table, velocity=_VELOCITY, key=jr.key(4)
    )
    reference = large.step_detailed(reference_state, large_table)
    assert bool(reference.successful)

    _, small, small_table, states = _image_runtime((_SMALL_ROLE, _LARGE_ROLE))
    assert small.bundle.capacity(0).capacity_id == _SMALL_ROLE.capacity_id
    state = small.initialize_state(_START, small_table, velocity=_VELOCITY, key=jr.key(4))
    rejected = small.step_detailed(state, small_table)
    assert not bool(rejected.successful)
    assert int(rejected.rejection_reasons) & int(Reason.ROUTE_CAPACITY)
    assert int(rejected.accepted_state.step_index) == int(state.step_index)
    np.testing.assert_array_equal(
        rejected.accepted_state.kinematics.positions, state.kinematics.positions
    )
    np.testing.assert_array_equal(rejected.accepted_state.random_key, state.random_key)

    # Host lifecycle rebinding to the grown declared level retains the accepted
    # physical, stochastic and counter state; priming rebinds the force cache.
    grown_bundle = small.bundle.grow_capacities(rejected.candidate_state.neighborhoods)
    rebound_dynamics, rebound_table, rebound = small.rebind_hamiltonian(
        small.potential.with_neighborhood_bundle(grown_bundle),
        state,
        small_table,
        states,
    )
    np.testing.assert_array_equal(rebound.random_key, state.random_key)
    assert int(rebound.step_index) == int(state.step_index)
    assert int(rebound.configuration_revision) == int(state.configuration_revision)
    assert not bool(rebound_dynamics.force_cache_valid(rebound, rebound.force.controls))
    primed = rebound_dynamics.prime_state(rebound, rebound_table)
    assert bool(primed.successful)
    np.testing.assert_allclose(
        primed.accepted_state.force.forces, state.force.forces, rtol=1e-12
    )
    assert int(primed.accepted_state.step_index) == int(state.step_index)

    grown, grown_table, evaluation = phx.atomistic.retry_atomistic_step_with_capacity(
        small, state, small_table, states
    )
    assert bool(evaluation.successful)
    assert grown.bundle.role_levels == (1,)
    assert grown.bundle.capacity(0).capacity_id == _LARGE_ROLE.capacity_id
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

    _, exhausted, exhausted_table, exhausted_states = _image_runtime((_SMALL_ROLE,))
    exhausted_state = exhausted.initialize_state(
        _START, exhausted_table, velocity=_VELOCITY, key=jr.key(4)
    )
    with pytest.raises(RuntimeError, match="exhausted"):
        phx.atomistic.retry_atomistic_step_with_capacity(
            exhausted, exhausted_state, exhausted_table, exhausted_states
        )


def test_classical_cutoff_beyond_unique_radius_is_refused_at_preparation() -> None:
    cell, dynamics, _, _ = _image_runtime()
    # The cutoff exceeds even the shortest lattice vector: self images and
    # repeated pair images lie inside it, which minimum-image pair-once counting
    # cannot represent.  The bundle binding of the cell owns the refusal.
    assert 2.1 > float(jnp.min(jnp.linalg.norm(cell.vectors, axis=-1)))
    assert 2.1 > cell.unique_image_radius
    refused = phx.atomistic.AtomisticPotentialProgram(
        # ty: ignore[invalid-argument-type]
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.1)]
    )
    resources = neighborhood_bundle_plan(
        refused.requirements.neighborhoods,
        search=_IMAGE_SEARCH,
        skin=0.3,
        search_capacities=(_SEARCH_CAPACITY,),
        role_capacities=(_LARGE_ROLE,),
    )
    with pytest.raises(ValueError):
        resources.prepare(dynamics.system)
    with pytest.raises(ValueError):
        refused.prepare(dynamics.system, neighborhood_bundle=resources)

    # The same counting below the unique-image radius is admitted.
    assert 0.9 < cell.unique_image_radius
    admitted = _classical_program(dynamics.system, 0.9)
    (requirement,) = admitted.neighborhood_bundle.plan.requirements
    assert requirement.role is Role.CLASSICAL_MINIMUM_IMAGE_PAIRS
    assert requirement.cutoff == 0.9


def _classical_program(system: Any, cutoff: float) -> Any:
    return prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([1.0], [0.3], cutoff)]
        ),
        system,
        search=_IMAGE_SEARCH,
        skin=0.3,
        search_capacities=(_SEARCH_CAPACITY,),
        role_capacities=(_LARGE_ROLE,),
    )


def test_classical_pairs_refuse_repeated_minimum_images_after_cell_compression() -> None:
    cell, dynamics, _, _ = _image_runtime()
    classical = _classical_program(dynamics.system, 0.9)
    bundle = classical.neighborhood_bundle
    reference = np.asarray(cell.vectors)
    # Half a lattice vector apart: 1.0 in the reference cell (outside the 0.9
    # cutoff through both images) and 0.8 through two images of the compressed
    # cell, whose unique-image radius falls below the admitted cutoff.
    fractions = np.asarray([[0.1, 0.1, 0.1], [0.6, 0.1, 0.1]])
    accepted = bundle.initialize(
        bind_atomistic_bundle_geometry(
            bundle, classical.system, jnp.asarray(fractions @ reference)
        )
    )
    assert bool(accepted.successful[0])
    compressed = 0.8 * reference
    positions = jnp.asarray(fractions @ compressed)
    candidate = bundle.refresh(
        accepted,
        bind_atomistic_bundle_geometry(
            bundle, classical.system, positions, cell_vectors=jnp.asarray(compressed)
        ),
    )
    view = candidate.views[0]
    assert view.role is Role.CLASSICAL_MINIMUM_IMAGE_PAIRS
    assert bool(view.evidence.repeated_minimum_image[0])
    assert bool(candidate.scientific_failure[0])
    assert not bool(candidate.capacity_only[0])
    refused = classical.evaluate(positions, candidate)
    assert not bool(refused.successful)
    assert bool(jnp.isnan(refused.energy))


def test_role_graph_revision_tracks_routes_not_only_owner_and_epoch() -> None:
    _, dynamics, _, _ = _image_runtime()
    bundle = dynamics.bundle
    near_positions = jnp.asarray([[0.2, 0.3, 0.1], [0.9, 0.6, 0.5]])
    far_positions = jnp.asarray([[0.2, 0.3, 0.1], [1.2, 1.1, 1.3]])
    near = bundle.initialize(_geometry(dynamics, near_positions))
    far = bundle.initialize(_geometry(dynamics, far_positions))
    near_view, far_view = near.views[0], far.views[0]
    assert near.plan_id == far.plan_id
    np.testing.assert_array_equal(near.graph_epochs, far.graph_epochs)
    assert not np.array_equal(
        np.asarray(near_view.relation.source_indices)[np.asarray(near_view.mask)],
        np.asarray(far_view.relation.source_indices)[np.asarray(far_view.mask)],
    ) or not np.array_equal(
        np.asarray(near_view.relation.image_shifts),
        np.asarray(far_view.relation.image_shifts),
    )
    reused = bundle.refresh(near, _geometry(dynamics, near_positions))
    np.testing.assert_array_equal(reused.graph_epochs, near.graph_epochs)
    np.testing.assert_array_equal(reused.epochs, near.epochs)
    moved = bundle.refresh(near, _geometry(dynamics, far_positions))
    np.testing.assert_array_equal(moved.graph_epochs, near.graph_epochs + 1)
    assert bool(jnp.all(moved.epochs > near.epochs))


def test_program_binds_only_certified_routes_of_its_own_bundle() -> None:
    _, dynamics, _, _ = _image_runtime()
    program = dynamics.potential
    bundle = dynamics.bundle
    near_positions = jnp.asarray([[0.2, 0.3, 0.1], [0.9, 0.6, 0.5]])
    far_positions = jnp.asarray([[0.2, 0.3, 0.1], [1.2, 1.1, 1.3]])
    near = bundle.initialize(_geometry(dynamics, near_positions))
    far = bundle.initialize(_geometry(dynamics, far_positions))
    fresh = program.evaluate(far_positions, far)
    assert bool(fresh.successful)
    reused = program.evaluate(
        far_positions, bundle.refresh(far, _geometry(dynamics, far_positions))
    )
    np.testing.assert_array_equal(reused.energy, fresh.energy)
    np.testing.assert_array_equal(reused.forces, fresh.forces)
    # Routes of another configuration, retained without a valid certificate.
    stale = bundle.refresh(near, _geometry(dynamics, far_positions), allow_rebuild=False)
    assert bool(stale.stale[0])
    rejected = program.evaluate(far_positions, stale)
    assert not bool(rejected.successful)
    assert bool(jnp.isnan(rejected.energy))
    # A state of another prepared bundle (another streamed plan) is refused.
    _, other, _, _ = _image_runtime(
        graph_execution=phx.atomistic.AtomisticGraphExecutionPlan(
            256,
            backend="particle",
            streamed=phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=8),
        )
    )
    assert other.bundle.prepared_id != bundle.prepared_id
    with pytest.raises(ValueError):
        program.evaluate(
            far_positions, other.bundle.initialize(_geometry(other, far_positions))
        )


def test_image_state_lattice_is_default_and_overrides_need_its_certificate() -> None:
    cell, dynamics, _, _ = _image_runtime()
    program = dynamics.potential
    bundle = dynamics.bundle
    positions = jnp.asarray([[0.2, 0.3, 0.1], [1.2, 1.1, 1.3]])
    deformed_vectors = 0.99 * cell.vectors
    deformed = bundle.initialize(_geometry(dynamics, positions, deformed_vectors))
    default = program.evaluate(positions, deformed, compute_stress=True)
    explicit = program.evaluate(
        positions, deformed, compute_stress=True, cell_vectors=deformed_vectors
    )
    assert bool(default.successful)
    np.testing.assert_array_equal(default.energy, explicit.energy)
    np.testing.assert_array_equal(default.forces, explicit.forces)
    np.testing.assert_array_equal(default.stress, explicit.stress)
    state = bundle.initialize(_geometry(dynamics, positions))
    # Inside the search skin the cached routes are certified: exact parity.
    nearby_vectors = 1.001 * cell.vectors
    nearby = _geometry(dynamics, positions, nearby_vectors)
    assert bool(bundle.certify(state, nearby)[0].certified[0])
    certified = program.evaluate(
        positions, bundle.refresh(state, nearby), cell_vectors=nearby_vectors
    )
    rebuilt = program.evaluate(
        positions, bundle.initialize(nearby), cell_vectors=nearby_vectors
    )
    assert bool(certified.successful)
    np.testing.assert_allclose(certified.energy, rebuilt.energy, rtol=1e-12)
    np.testing.assert_allclose(certified.forces, rebuilt.forces, atol=1e-12)
    # A compressed lattice or moved atom admits images the state never searched.
    compressed_vectors = 0.8 * cell.vectors
    moved_positions = positions.at[1].add(0.6)
    for refused_positions, refused_vectors in (
        (positions, compressed_vectors),
        (moved_positions, cell.vectors),
    ):
        geometry = _geometry(dynamics, refused_positions, refused_vectors)
        assert not bool(bundle.certify(state, geometry)[0].certified[0])
        stale = bundle.refresh(state, geometry, allow_rebuild=False)
        refused = program.evaluate(refused_positions, stale, cell_vectors=refused_vectors)
        assert not bool(refused.successful)
        assert bool(jnp.isnan(refused.energy))


def test_capacity_retry_preserves_independent_thermodynamic_rejection() -> None:
    _, small, small_table, states = _image_runtime((_SMALL_ROLE,))
    accepted = small.initialize_state(
        _START, small_table, velocity=_VELOCITY, key=jr.key(4)
    )
    invalid = eqx.tree_at(
        lambda value: value.thermodynamic_state_index,
        accepted,
        jnp.asarray(9, dtype=jnp.int32),
    )
    rejected = small.step_detailed(invalid, small_table)
    reasons = int(rejected.rejection_reasons)
    assert reasons & int(Reason.ROUTE_CAPACITY)
    assert reasons & int(Reason.THERMODYNAMIC_STATE)
    # The one-entry ladder would raise if a capacity replay were attempted.
    returned, returned_table, evaluation = (
        phx.atomistic.retry_atomistic_step_with_capacity(
            small, invalid, small_table, states
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


def test_radial_distribution_counts_only_pair_once_classical_views() -> None:
    cell = _cell()
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
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([1.0], [1.0], 1.2)]
        ),
        system,
        **_pair_resources(1, 1),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system, potential, phx.atomistic.VelocityVerletPlan(1.0e-3)
    ).prepare()
    table = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    positions = cell.cartesian(jnp.asarray([[0.1, 0.1, 0.1], [0.4, 0.1, 0.1]]))
    state = dynamics.initialize_state(
        positions, table, velocity=jnp.zeros_like(positions), key=jr.key(2)
    )
    # Pair distance 0.9 lies inside bin 3 of width 0.24 (away from bin edges).
    plan = RadialDistributionPlan(5, 1.2)
    counts = RadialDistributionState.empty(plan, positions.dtype).update(
        plan, dynamics, state
    )
    distance = float(jnp.linalg.norm(positions[1] - positions[0]))
    np.testing.assert_allclose(distance, 0.9, rtol=1e-12)
    expected = np.zeros((5,))
    expected[3] = 1.0
    np.testing.assert_array_equal(counts.counts, expected)
    assert int(counts.samples) == 1
    # A bin beyond the classical cutoff would silently omit pairs.
    wide = RadialDistributionPlan(4, 1.5)
    with pytest.raises(ValueError, match="maximum_radius"):
        RadialDistributionState.empty(wide, positions.dtype).update(wide, dynamics, state)
    # Directed image routes are not a pair relation.
    _, graph, graph_table, _ = _image_runtime()
    unwrapped = jnp.asarray([[0.2, 0.3, 0.1], [2.1, 1.9, 2.4]])
    graph_state = graph.initialize_state(
        unwrapped, graph_table, velocity=jnp.zeros_like(unwrapped), key=jr.key(0)
    )
    with pytest.raises(ValueError, match="pair-once"):
        RadialDistributionState.empty(plan, unwrapped.dtype).update(
            plan, graph, graph_state
        )
