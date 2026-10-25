"""Domain terms: classical, PME, polarization and model compositions.

Every reference is the same canonical term evaluated by the single-device
``AtomisticPotentialProgram`` over its own neighborhood bundle (or, for the
layered model, the brute-force global MACE oracle of the same model); the
domain executes each term owner-locally on its own role topology and the
term's reciprocal work through the distributed reciprocal owner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax.atomistic._classical import LennardJonesPotential
from phydrax.atomistic._domain_checkpoint import (
    AtomisticDomainCheckpointError,
    migrate_domain_checkpoint,
    publish_domain_checkpoint,
)
from phydrax.atomistic._domain_plan import AtomisticDomainPlan, AtomisticDomainRolePlan
from phydrax.atomistic._domain_runtime import owner_predictions, prepare_atomistic_domain
from phydrax.atomistic._domain_state import AtomisticDomainStrategy
from phydrax.atomistic._domain_terms import (
    AtomisticDomainEwaldTerm,
    AtomisticDomainExceptionTable,
    AtomisticDomainLennardJonesTerm,
    AtomisticDomainPolarizationTerm,
    domain_exception_topology,
)
from phydrax.atomistic._dynamics import AtomisticStepRejectionReason
from phydrax.atomistic._electrostatics import ParticleMeshEwaldPotential
from phydrax.atomistic._graph_partition import AtomisticGraphPartitionPlan
from phydrax.atomistic._multipole_electrostatics import ParticleMeshMultipoleRoute
from phydrax.atomistic._neighborhood_bundle import (
    AtomisticCellListImageSearch,
    AtomisticNeighborhoodRole,
)
from phydrax.atomistic._polarization import (
    InducedDipolePolarizationPotential,
    PermanentMultipoleSiteData,
    PolarizationPlan,
    PolarizationScaleData,
    PolarizationStabilityPolicy,
)
from phydrax.atomistic._reciprocal_distribution import (
    AtomisticReciprocalDistributionPlan,
    OwnerPolarizationPlan,
)
from phydrax.atomistic._reciprocal_mesh import ReciprocalMeshPlan
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.particle._distributed import FractionalOwnerPartition
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._restart_topology import TopologyRestartPolicy
from phydrax.sparse._streamed import StreamedRelationPlan
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_domain import (
    global_reference,
    mace_model,
    ownership,
    role_plan,
    SPECIES,
)


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
CELL = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])
SKIN = 0.4
LOCAL = 24
REASON = AtomisticStepRejectionReason


def _configuration(seed: int, count: int, cell: np.ndarray, minimum: float) -> np.ndarray:
    """Rejection-sampled positions with minimum-image separations above ``minimum``."""
    rng = np.random.default_rng(seed)
    inverse = np.linalg.inv(cell)
    points: list[np.ndarray] = []
    while len(points) < count:
        candidate = rng.uniform(0.0, 1.0, 3) @ cell
        separated = True
        for point in points:
            fractional = (candidate - point) @ inverse
            fractional -= np.round(fractional)
            if np.linalg.norm(fractional @ cell) < minimum:
                separated = False
                break
        if separated:
            points.append(candidate)
    return np.asarray(points)


def _minimum_image(positions: np.ndarray, cell: np.ndarray, i: int, j: int) -> float:
    fractional = (positions[i] - positions[j]) @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    return float(np.linalg.norm(fractional @ cell))


def _system(
    charges: np.ndarray,
    types: np.ndarray,
    cell: np.ndarray,
    *,
    numbers: np.ndarray | None = None,
    exceptions: np.ndarray | None = None,
) -> Any:
    count = len(charges)
    topology = None
    if exceptions is not None:
        topology = MolecularTopologyPlan(
            pair_exceptions=exceptions,
            lennard_jones_scales=np.full((len(exceptions),), 0.5),
            electrostatic_scales=np.full((len(exceptions),), 0.8333),
        )
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(10, 10 + count, dtype=np.int64),
        np.ones((count,), dtype=np.int64) if numbers is None else np.asarray(numbers),
        np.ones((count,), dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.asarray(types),
        charges=np.asarray(charges),
        cell=PeriodicCell(cell),
        topology=topology,
    ).prepare()


def _role(term: Any, cell: np.ndarray, *, skin: float = SKIN) -> AtomisticDomainRolePlan:
    (requirement,) = term.requirements.neighborhoods
    return AtomisticDomainRolePlan(
        requirement,
        PeriodicCell(cell).image_stencil(
            requirement.bounded_cutoff() + skin, maximum_image_count=4096
        ),
        streaming=StreamedRelationPlan(),
        skin=skin,
        edge_capacity=2048,
        alias_capacity=2048,
        halo_capacity=LOCAL,
        term_key=term.name,
    )


def _plan(
    system: Any,
    roles: tuple[AtomisticDomainRolePlan, ...],
    terms: tuple[Any, ...],
    owners: int,
    *,
    devices: bool = False,
) -> AtomisticDomainPlan:
    return AtomisticDomainPlan(
        ownership(owners, devices=devices, local_capacity=LOCAL),
        FractionalOwnerPartition(system.cell, (owners, 1, 1)),
        roles,
        migration_capacity=LOCAL,
        message_capacity_bytes=1 << 24,
        terms=terms,
    )


def _reference(terms: list[Any], system: Any, positions: np.ndarray) -> Any:
    program = phx.atomistic.AtomisticPotentialProgram(terms)
    prepared = prepare_program(
        program,
        system,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 16),
        skin=SKIN,
        search_capacities=(search_capacity(4096, particles_per_cell=32, images=343),),
        role_capacities=(role_capacity(4096, 256),),
    )
    state = bundle_state(prepared, positions)
    evaluation = prepared.evaluate(jnp.asarray(positions), state, compute_stress=True)
    assert bool(evaluation.successful)
    return evaluation


def _classical_case(seed: int = 3, *, grid: int = 16) -> tuple[Any, np.ndarray, Any, Any]:
    count = 12
    positions = _configuration(seed, count, CELL, 1.0)
    rng = np.random.default_rng(seed + 1)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    types = rng.integers(0, 2, count)
    near = sorted(
        (
            (_minimum_image(positions, CELL, i, j), i, j)
            for i in range(count)
            for j in range(i + 1, count)
        )
    )
    exceptions = np.asarray([[10 + i, 10 + j] for _, i, j in near[:3]])
    system = _system(charges, types, CELL, exceptions=exceptions)
    lennard_jones = LennardJonesPotential(
        np.asarray([0.2, 0.1]), np.asarray([1.0, 1.2]), 2.5
    )
    pme = ParticleMeshEwaldPotential(1.1, 2.8, (grid, grid, grid), spline_degree=4)
    return system, positions, lennard_jones, pme


def _domain_terms(
    system: Any, lennard_jones: Any, pme: Any, *, exception_capacity: int = 8
) -> tuple[Any, ...]:
    prepared_lj = lennard_jones.prepare(system)
    prepared_pme = pme.prepare(system)
    return (
        AtomisticDomainLennardJonesTerm(prepared_lj),
        AtomisticDomainEwaldTerm(
            prepared_pme,
            AtomisticReciprocalDistributionPlan(prepared_pme, reuse_reach=SKIN),
            exception_capacity=exception_capacity,
        ),
    )


def _assert_matches(evaluation: Any, reference: Any, state: Any) -> None:
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        float(evaluation.energy), float(reference.energy), rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)),
        np.asarray(reference.forces),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(evaluation.stress), np.asarray(reference.stress), rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize(
    ("owners", "devices"),
    [(2, False), (3, False), (3, True)],
    ids=["lanes-2", "lanes-3", "devices-3"],
)
def test_lennard_jones_and_pme_match_single_device_program(
    owners: int, devices: bool
) -> None:
    system, positions, lennard_jones, pme = _classical_case(
        grid=24 if owners == 3 else 16
    )
    terms = _domain_terms(system, lennard_jones, pme)
    plan = _plan(
        system,
        (_role(lennard_jones, CELL), _role(pme, CELL)),
        terms,
        owners,
        devices=devices,
    )
    runtime = prepare_atomistic_domain(plan)
    state = runtime.ingress_global(
        positions,
        np.asarray(system.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    evaluation = runtime.evaluate(state)
    reference = _reference([lennard_jones, pme], system, positions)
    _assert_matches(evaluation, reference, state)
    np.testing.assert_allclose(
        np.asarray(evaluation.term_energies),
        np.asarray(reference.term_energies),
        rtol=1e-9,
        atol=1e-10,
    )
    assert bool(jnp.all(evaluation.status.term_capacity_successful))
    assert not bool(evaluation.atom_energy_complete)

    # Translate across owners: term payload (types, charges) migrates with atoms.
    shift = np.broadcast_to(CELL[0] / owners, positions.shape)
    moved = runtime.with_dynamics(
        state,
        state.positions + state.layout.distribute(shift),
        state.momenta,
        step_index=1,
    )
    transition = runtime.rebuild(moved)
    assert bool(transition.committed)
    rebuilt = transition.state
    _assert_matches(
        runtime.evaluate(rebuilt),
        _reference([lennard_jones, pme], system, positions + shift),
        rebuilt,
    )


def test_minimum_image_ambiguity_fails_closed() -> None:
    """A lattice translation inside the LJ cutoff gives a self image: no unique pair."""
    small = 0.4 * CELL
    count = 4
    positions = _configuration(5, count, small, 0.9)
    system = _system(np.zeros(count), np.zeros(count, dtype=np.int32), small)
    lennard_jones = LennardJonesPotential(np.asarray([0.2]), np.asarray([1.0]), 2.5)
    term = AtomisticDomainLennardJonesTerm(lennard_jones.prepare(system))
    runtime = prepare_atomistic_domain(
        _plan(system, (_role(lennard_jones, small),), (term,), 2)
    )
    state = runtime.ingress_global(
        positions, np.zeros(count, dtype=np.int32), rng_key=jax.random.key(0)
    )
    evaluation = runtime.evaluate(state)
    assert not bool(evaluation.successful)
    assert bool(evaluation.status.term_image_ambiguity[0])
    assert int(evaluation.reasons) & int(REASON.IMAGE_AMBIGUITY)
    assert np.isnan(float(evaluation.energy))


def _distant_exception_case(*, grid: int = 16) -> tuple[Any, np.ndarray, Any, Any]:
    """An exception pair whose minimum image lies beyond the PME real cutoff."""
    system, positions, lennard_jones, pme = _classical_case(seed=7, grid=grid)
    count = positions.shape[0]
    far = max(
        (
            (_minimum_image(positions, CELL, i, j), i, j)
            for i in range(count)
            for j in range(i + 1, count)
        )
    )
    assert far[0] > 2.8
    near = min(
        (
            (_minimum_image(positions, CELL, i, j), i, j)
            for i in range(count)
            for j in range(i + 1, count)
        )
    )
    distant = _system(
        np.asarray(system.plan.charges),
        np.asarray(system.plan.atom_type_ids),
        CELL,
        exceptions=np.asarray([[10 + far[1], 10 + far[2]], [10 + near[1], 10 + near[2]]]),
    )
    return distant, positions, lennard_jones, pme


@pytest.mark.parametrize("owners", [2, 3])
def test_exception_outside_the_real_cutoff_matches_single_device_program(
    owners: int,
) -> None:
    """Listed pairs act on their current minimum image independently of the cutoff."""
    distant, positions, lennard_jones, pme = _distant_exception_case(
        grid=24 if owners == 3 else 16
    )
    terms = _domain_terms(distant, lennard_jones, pme)
    runtime = prepare_atomistic_domain(
        _plan(distant, (_role(lennard_jones, CELL), _role(pme, CELL)), terms, owners)
    )
    state = runtime.ingress_global(
        positions,
        np.asarray(distant.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    evaluation = runtime.evaluate(state)
    reference = _reference([lennard_jones, pme], distant, positions)
    _assert_matches(evaluation, reference, state)
    assert bool(jnp.all(evaluation.status.term_capacity_successful))
    assert int(evaluation.reasons) == 0


def test_exception_route_overflow_is_capacity_evidence() -> None:
    """Two pairs give four directed routes; one route per owner cannot hold them."""
    distant, positions, lennard_jones, pme = _distant_exception_case()
    terms = _domain_terms(distant, lennard_jones, pme, exception_capacity=1)
    runtime = prepare_atomistic_domain(
        _plan(distant, (_role(lennard_jones, CELL), _role(pme, CELL)), terms, 2)
    )
    state = runtime.ingress_global(
        positions,
        np.asarray(distant.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    evaluation = runtime.evaluate(state)
    assert not bool(evaluation.successful)
    assert not bool(evaluation.status.term_capacity_successful[1])
    assert bool(evaluation.status.term_scientific_successful[1])
    assert int(evaluation.reasons) & int(REASON.ROUTE_CAPACITY)
    assert not int(evaluation.reasons) & int(REASON.POTENTIAL)
    assert np.isnan(float(evaluation.energy))


def test_exception_absence_is_scientific_only_after_complete_registration() -> None:
    """A partner absent from a fully registered directory is a missing pair;
    under a registration overflow it is unresolved capacity, never missing."""
    distant, positions, lennard_jones, pme = _distant_exception_case()
    terms = _domain_terms(distant, lennard_jones, pme)
    runtime = prepare_atomistic_domain(
        _plan(distant, (_role(lennard_jones, CELL), _role(pme, CELL)), terms, 2)
    )
    state = runtime.ingress_global(
        positions,
        np.asarray(distant.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    table = terms[1].exception_table
    absent = 10 + positions.shape[0] + 7
    for capacity, fits, missing in ((8, True, 1), (1, False, 0)):
        topology = domain_exception_topology(
            state.layout,
            AtomisticDomainExceptionTable(
                left_ids=table.left_ids,
                right_ids=table.right_ids.at[0].set(absent),
                weights=table.weights,
                capacity=capacity,
            ),
        )
        assert bool(topology.capacity_successful) is fits
        assert int(topology.missing) == missing


def test_ewald_term_requires_an_exception_capacity_for_listed_pairs() -> None:
    system, _, _, pme = _classical_case()
    prepared = pme.prepare(system)
    with pytest.raises(ValueError, match="exception_capacity"):
        AtomisticDomainEwaldTerm(
            prepared, AtomisticReciprocalDistributionPlan(prepared, reuse_reach=SKIN)
        )


def _mace_pme_case(owners: int = 2) -> tuple[Any, ...]:
    model = mace_model(2)
    count = 8
    positions = _configuration(11, count, CELL, 1.0)
    rng = np.random.default_rng(12)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    numbers = np.asarray(SPECIES)[rng.integers(0, 2, count)]
    system = _system(charges, np.zeros(count, dtype=np.int32), CELL, numbers=numbers)
    grid = 24 if owners == 3 else 16
    pme = ParticleMeshEwaldPotential(1.1, 2.8, (grid, grid, grid), spline_degree=4)
    prepared_pme = pme.prepare(system)
    term = AtomisticDomainEwaldTerm(
        prepared_pme, AtomisticReciprocalDistributionPlan(prepared_pme, reuse_reach=SKIN)
    )
    model_role = role_plan(model, CELL, edge_capacity=2048, halo_capacity=LOCAL)
    plan = _plan(system, (model_role, _role(pme, CELL)), (term,), owners)
    runtime = prepare_atomistic_domain(plan, model)
    mask = jnp.ones((count,), dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(numbers), mask))
    state = runtime.ingress_global(positions, species, rng_key=jax.random.key(0))
    return model, system, positions, numbers, pme, runtime, state


def test_pme_and_mace_composition_matches_sum_of_single_device_references() -> None:
    model, system, positions, numbers, pme, runtime, state = _mace_pme_case()
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    assert not bool(evaluation.atom_energy_complete)
    energy, forces, strain = global_reference(model, positions, numbers, CELL)
    electrostatic = _reference([pme], system, positions)
    np.testing.assert_allclose(
        float(evaluation.energy),
        float(energy) + float(electrostatic.energy),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(evaluation.forces)),
        np.asarray(forces) + np.asarray(electrostatic.forces),
        rtol=1e-7,
        atol=1e-9,
    )
    volume = abs(np.linalg.det(CELL))
    np.testing.assert_allclose(
        np.asarray(evaluation.stress),
        0.5 * (np.asarray(strain) + np.asarray(strain).T) / volume
        + np.asarray(electrostatic.stress),
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        float(evaluation.term_energies[0]), float(electrostatic.energy), rtol=1e-9
    )
    np.testing.assert_allclose(float(evaluation.local_energy), float(energy), rtol=1e-11)


@pytest.mark.parametrize("owners", [2, 3])
def test_owner_predictions_include_domain_terms(owners: int) -> None:
    """Region-level predictions (the training route) equal the full evaluation."""
    _, _, _, _, _, runtime, state = _mace_pme_case(owners)
    evaluation = runtime.evaluate(state)
    assert bool(evaluation.successful)
    view = runtime.owner_view(state)
    assert view.terms is not None
    axis = runtime.plan.ownership.axis_name

    def body(local_view: Any, lane: Any) -> tuple[jax.Array, ...]:
        result = owner_predictions(runtime, lane, local_view, forces=True, stress=True)
        forces, stress = result.forces, result.stress
        assert forces is not None and stress is not None  # both were requested
        return (
            result.energy[None],
            forces,
            stress[None],
            result.successful[None],
            result.capacity_failure[None],
        )

    energy, forces, stress, successful, capacity_failure = runtime.plan.ownership.map(
        body, (view.partition_specs(), PartitionSpec()), PartitionSpec(axis)
    )(view, runtime.parameter_lane)
    assert bool(jnp.all(successful)) and not bool(jnp.any(capacity_failure))
    for owner in range(owners):
        np.testing.assert_allclose(
            float(energy[owner]), float(evaluation.energy), rtol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(stress[owner]), np.asarray(evaluation.stress), atol=1e-12
        )
    np.testing.assert_allclose(
        np.asarray(state.layout.collect(forces)),
        np.asarray(state.layout.collect(evaluation.forces)),
        atol=1e-12,
    )


def test_atom_energy_completeness_is_structural() -> None:
    """A local model alone decomposes its energy; a domain term never does."""
    model, system, positions, numbers, _, runtime, state = _mace_pme_case()
    local = prepare_atomistic_domain(
        _plan(system, (runtime.plan.roles[0],), (), 2), model
    )
    mask = jnp.ones(numbers.shape, dtype=jnp.bool_)
    species = np.asarray(model.species_indices(jnp.asarray(numbers), mask))
    local_state = local.ingress_global(positions, species, rng_key=jax.random.key(0))
    evaluation = local.evaluate(local_state)
    assert bool(evaluation.successful)
    assert bool(evaluation.atom_energy_complete)
    assert not bool(runtime.evaluate(state).atom_energy_complete)


def test_reciprocal_epilogue_must_be_the_models_own_definition() -> None:
    """A local model or a term-only runtime refuses a reciprocal epilogue."""
    model, system, _, _, pme, runtime, _ = _mace_pme_case()
    prepared_pme = pme.prepare(system)
    reciprocal = AtomisticReciprocalDistributionPlan(prepared_pme, reuse_reach=SKIN)
    plan = runtime.plan
    with_epilogue = AtomisticDomainPlan(
        plan.ownership,
        plan.partition,
        plan.roles,
        migration_capacity=LOCAL,
        message_capacity_bytes=1 << 24,
        reciprocal=reciprocal,
        terms=plan.terms,
    )
    with pytest.raises(ValueError, match="declares none"):
        prepare_atomistic_domain(with_epilogue, model)
    terms_only = AtomisticDomainPlan(
        plan.ownership,
        plan.partition,
        (plan.roles[1],),
        migration_capacity=LOCAL,
        message_capacity_bytes=1 << 24,
        reciprocal=reciprocal,
        terms=plan.terms,
    )
    with pytest.raises(ValueError, match="declares none"):
        prepare_atomistic_domain(terms_only)


def _graph_partition_plan(
    system: Any, roles: tuple[AtomisticDomainRolePlan, ...], terms: tuple[Any, ...]
) -> AtomisticDomainPlan:
    owners = 2
    placement = ownership(owners, local_capacity=LOCAL)
    partition = FractionalOwnerPartition(system.cell, (owners, 1, 1))
    graph = AtomisticGraphPartitionPlan(
        placement,
        partition,
        roles,
        replicated_memory_bytes=1 << 24,
        receiver_edge_capacity=512,
        maximum_particles_per_cell=32,
        search=AtomisticCellListImageSearch(maximum_candidate_slots=1 << 22),
    )
    return AtomisticDomainPlan(
        placement,
        partition,
        roles,
        migration_capacity=LOCAL,
        message_capacity_bytes=1 << 24,
        strategy=AtomisticDomainStrategy.GRAPH_PARTITION,
        graph_partition=graph,
        terms=terms,
    )


def test_lennard_jones_and_pme_execute_on_graph_partition_topologies() -> None:
    system, positions, lennard_jones, pme = _classical_case()
    terms = _domain_terms(system, lennard_jones, pme)
    runtime = prepare_atomistic_domain(
        _graph_partition_plan(
            system, (_role(lennard_jones, CELL), _role(pme, CELL)), terms
        )
    )
    state = runtime.ingress_global(
        positions,
        np.asarray(system.plan.atom_type_ids, dtype=np.int32),
        rng_key=jax.random.key(0),
    )
    evaluation = runtime.evaluate(state)
    reference = _reference([lennard_jones, pme], system, positions)
    _assert_matches(evaluation, reference, state)
    np.testing.assert_allclose(
        np.asarray(evaluation.term_energies),
        np.asarray(reference.term_energies),
        rtol=1e-9,
        atol=1e-10,
    )


def test_polarization_refuses_the_graph_partition_strategy() -> None:
    cell = np.asarray([[7.0, 0.0, 0.0], [0.8, 7.2, 0.0], [0.4, 0.9, 7.5]])
    count = 4
    system = _system(np.zeros(count), np.zeros(count, dtype=np.int32), cell)
    rng = np.random.default_rng(3)
    term = InducedDipolePolarizationPotential(
        PolarizationPlan(
            coulomb_constant=system.plan.units.coulomb_constant,
            route=ParticleMeshMultipoleRoute(
                1.0,
                3.0,
                ReciprocalMeshPlan((12, 12, 12), spline_degree=4),
                real_space="neighborhood",
            ),
            tolerance=1e-12,
            maximum_iterations=60,
            stability=PolarizationStabilityPolicy("assumed-positive-definite"),
        ),
        PermanentMultipoleSiteData(
            rng.normal(size=count),
            np.zeros((count, 3)),
            np.zeros((count, 3, 3)),
            rng.uniform(0.1, 0.3, size=count),
            np.full((count,), 4.0),
        ),
    )
    prepared = term.prepare(system)
    domain_term = AtomisticDomainPolarizationTerm(
        prepared, OwnerPolarizationPlan(prepared.solver, reuse_reach=SKIN)
    )
    with pytest.raises(ValueError, match="HALO"):
        prepare_atomistic_domain(
            _graph_partition_plan(system, (_role(term, cell),), (domain_term,))
        )


def _polarization_case(
    scaled: bool, *, maximum_iterations: int = 60
) -> tuple[Any, np.ndarray, Any, Any, Any]:
    """Two-owner PME polarization domain and its ingress state."""
    cell = np.asarray([[7.0, 0.0, 0.0], [0.8, 7.2, 0.0], [0.4, 0.9, 7.5]])
    count = 6
    positions = _configuration(43, count, cell, 1.2)
    rng = np.random.default_rng(37)
    charges = rng.normal(size=count)
    charges -= charges.mean()
    multipoles = PermanentMultipoleSiteData(
        charges,
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        rng.uniform(0.1, 0.3, size=count),
        np.full((count,), 4.0),
    )
    scaling = None
    if scaled:
        near = sorted(
            (_minimum_image(positions, cell, i, j), i, j)
            for i in range(count)
            for j in range(i + 1, count)
        )
        pairs = np.asarray([[near[0][1], near[0][2]], [near[1][1], near[1][2]]])
        scaling = PolarizationScaleData(
            pairs,
            np.asarray([0.0, 0.5]),
            np.asarray([0.3, 0.5]),
            np.asarray([0.6, 0.8]),
            site_capacity=count,
        )
    system = _system(np.zeros(count), np.zeros(count, dtype=np.int32), cell)
    term = InducedDipolePolarizationPotential(
        PolarizationPlan(
            coulomb_constant=system.plan.units.coulomb_constant,
            route=ParticleMeshMultipoleRoute(
                1.0,
                3.0,
                ReciprocalMeshPlan((12, 12, 12), spline_degree=4),
                real_space="neighborhood",
            ),
            tolerance=1e-12,
            maximum_iterations=maximum_iterations,
            stability=PolarizationStabilityPolicy("assumed-positive-definite"),
        ),
        multipoles,
        scaling=scaling,
    )
    prepared = term.prepare(system)
    domain_term = AtomisticDomainPolarizationTerm(
        prepared,
        OwnerPolarizationPlan(prepared.solver, reuse_reach=SKIN),
        exception_capacity=8 if scaled else None,
    )
    runtime = prepare_atomistic_domain(
        _plan(system, (_role(term, cell),), (domain_term,), 2)
    )
    state = runtime.ingress_global(
        positions, np.zeros(count, dtype=np.int32), rng_key=jax.random.key(0)
    )
    return system, positions, term, runtime, state


@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled-pairs"])
def test_polarization_term_matches_single_device_program(scaled: bool) -> None:
    system, positions, term, runtime, state = _polarization_case(scaled)
    evaluation = runtime.evaluate(state)
    reference = _reference([term], system, positions)
    _assert_matches(evaluation, reference, state)
    assert (
        runtime.plan.role(AtomisticNeighborhoodRole.IMAGE_PAIR_SUM, term.name).term_key
        == term.name
    )


def test_unconverged_owner_polarization_is_a_scientific_refusal() -> None:
    """A PCG stopped by its step limit (no happy stop) refuses the term."""
    _, _, _, runtime, state = _polarization_case(False, maximum_iterations=1)
    evaluation = runtime.evaluate(state)
    assert not bool(evaluation.successful)
    assert bool(evaluation.status.term_capacity_successful[0])
    assert not bool(evaluation.status.term_scientific_successful[0])
    assert int(evaluation.reasons) & int(REASON.POTENTIAL)
    assert not int(evaluation.reasons) & int(REASON.ROUTE_CAPACITY)
    assert np.isnan(float(evaluation.energy))


def test_term_roles_must_carry_the_term_requirement() -> None:
    system, _, lennard_jones, pme = _classical_case()
    terms = _domain_terms(system, lennard_jones, pme)
    with pytest.raises(ValueError, match="role plan"):
        _plan(system, (_role(lennard_jones, CELL),), terms, 2)
    with pytest.raises(ValueError, match="layered model or domain terms"):
        prepare_atomistic_domain(
            AtomisticDomainPlan(
                ownership(2, local_capacity=LOCAL),
                FractionalOwnerPartition(system.cell, (2, 1, 1)),
                (_role(lennard_jones, CELL),),
                migration_capacity=LOCAL,
                message_capacity_bytes=1 << 24,
            )
        )


def _repository(tmp_path: Path) -> POSIXArtifactRepository:
    profile = HPCFilesystemProfile(
        "posix.domain-terms-test",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        tmp_path / "repository",
        POSIXRepositoryPolicy(
            profile, maximum_chunk_bytes=4096, maximum_metadata_bytes=1 << 20
        ),
    )


def test_term_only_identity_separates_physics_from_layout_and_migrates(
    tmp_path: Path,
) -> None:
    """A term-only revision digests placement-bearing term identities; the
    domain physics identity does not, so a changed owner count and reciprocal
    placement migrate while numerically different physics is refused and a
    numeric-only change of one layout keeps one executable structure."""
    system, positions, lennard_jones, pme = _classical_case()
    roles = (_role(lennard_jones, CELL), _role(pme, CELL))
    types = np.asarray(system.plan.atom_type_ids, dtype=np.int32)

    def runtime_for(
        owners: int, potential: Any = lennard_jones, **reciprocal: Any
    ) -> Any:
        prepared_pme = pme.prepare(system)
        terms = (
            AtomisticDomainLennardJonesTerm(potential.prepare(system)),
            AtomisticDomainEwaldTerm(
                prepared_pme,
                AtomisticReciprocalDistributionPlan(
                    prepared_pme, reuse_reach=SKIN, **reciprocal
                ),
                exception_capacity=8,
            ),
        )
        return prepare_atomistic_domain(_plan(system, roles, terms, owners))

    source = runtime_for(2)
    # Both owner counts divide the fixed 16-point reciprocal mesh axes.
    target = runtime_for(4, maximum_packet_bytes=1 << 26)
    assert source.model_revision_id != target.model_revision_id
    assert source.runtime_id != target.runtime_id
    assert source.scientific_id == target.scientific_id
    assert source.hamiltonian_revision_id == target.hamiltonian_revision_id
    stronger = LennardJonesPotential(np.asarray([0.3, 0.1]), np.asarray([1.0, 1.2]), 2.5)
    reweighted = runtime_for(2, stronger)
    assert reweighted.scientific_id != source.scientific_id
    assert reweighted.hamiltonian_revision_id != source.hamiltonian_revision_id
    # Numeric definitions travel as dynamic identity words: one treedef, one
    # executable for both parameter sets of the same layout.
    assert jax.tree.structure(reweighted) == jax.tree.structure(source)

    state = source.ingress_global(positions, types, rng_key=jax.random.key(0))
    repository = _repository(tmp_path)
    limits = ArrayArchiveLimits(max_members=512)
    publish_domain_checkpoint(
        repository,
        source,
        state,
        checkpoint_id="terms-migrate",
        writer_id="w",
        limits=limits,
    )
    policy = TopologyRestartPolicy(
        allow_topology_change=True,
        allow_tolerance_restart=True,
        maximum_relative_tolerance=1e-10,
    )
    restore = migrate_domain_checkpoint(
        repository,
        target,
        target.ingress_global(positions, types, rng_key=jax.random.key(0)),
        "terms-migrate",
        policy=policy,
        relative_tolerance=1e-10,
        limits=limits,
    )
    assert restore.record.scientific_id == target.scientific_id
    assert restore.record.hamiltonian_revision_id == target.hamiltonian_revision_id
    migrated = target.evaluate(restore.state)
    reference = source.evaluate(state)
    assert bool(migrated.successful)
    np.testing.assert_allclose(
        float(migrated.energy), float(reference.energy), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(restore.state.layout.collect(migrated.forces)),
        np.asarray(state.layout.collect(reference.forces)),
        rtol=1e-9,
        atol=1e-11,
    )
    other = runtime_for(4, stronger)
    with pytest.raises(AtomisticDomainCheckpointError):
        migrate_domain_checkpoint(
            repository,
            other,
            other.ingress_global(positions, types, rng_key=jax.random.key(0)),
            "terms-migrate",
            policy=policy,
            relative_tolerance=1e-10,
            limits=limits,
        )
