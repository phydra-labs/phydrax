from itertools import product
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.atomistic._graph as graph_module
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticScaleContract,
    bind_atomistic_graph,
    prepare_atomistic_graph_topology,
    realize_atomistic_graph,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.units import ANGSTROM, ELECTRONVOLT, JOULE, METER


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def test_structure_preserves_particles_masks_ids_masses_and_scale() -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [8, 1, 1, 0],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0], [0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [15.999, 1.008, 1.008, 0.0],
        SCALE,
        # ty: ignore[invalid-argument-type]
        particle_ids=[40, 11, 23, 99],
        # ty: ignore[invalid-argument-type]
        active_mask=[True, True, True, False],
    )
    assert structure.scale.scale_id == SCALE.scale_id
    np.testing.assert_array_equal(structure.particle_ids, [40, 11, 23, 99])
    np.testing.assert_array_equal(structure.active_mask, [True, True, True, False])
    np.testing.assert_allclose(structure.masses[:3], [15.999, 1.008, 1.008])
    assert structure.particles.entities.entity_set_id
    assert structure.axis_names == ("atom", "cartesian")


@pytest.mark.parametrize(
    ("numbers", "mask", "message"),
    [
        ([0, 1], [True, True], "positive"),
        ([1, 6], [True, False], "padded"),
    ],
)
def test_atomic_number_zero_is_padding_only(
    numbers: Any, mask: Any, message: Any
) -> None:
    with pytest.raises(ValueError, match=message):
        AtomicStructure(
            numbers,
            np.zeros((2, 3)),
            np.ones((2,)),
            SCALE,
            active_mask=mask,
        )


def test_batch_padding_does_not_change_structure_identity_or_graph_isolation() -> None:
    # ty: ignore[invalid-argument-type]
    hydrogen = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.008], SCALE, name="h")
    oxygen = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [8, 8],
        # ty: ignore[invalid-argument-type]
        [[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [15.999, 15.999],
        SCALE,
        # ty: ignore[invalid-argument-type]
        particle_ids=[7, 3],
        name="o2",
    )
    batch = AtomisticBatch.from_structures((hydrogen, oxygen), atom_capacity=3)
    graph = realize_atomistic_graph(
        batch, AtomisticGraphExecutionPlan(2, maximum_dense_atoms=3), cutoff=2.0
    )
    assert graph.graph.num_graphs == 2
    assert graph.graph.nodes["atomic_numbers"].shape == (6,)
    assert not bool(jnp.any(graph.overflow))
    # ty: ignore[unsupported-operator]
    send_case = graph.graph.senders // batch.atom_capacity
    # ty: ignore[unsupported-operator]
    receive_case = graph.graph.receivers // batch.atom_capacity
    np.testing.assert_array_equal(send_case, receive_case)
    np.testing.assert_array_equal(batch.atomic_numbers[0], [1, 0, 0])
    np.testing.assert_array_equal(batch.particle_ids[1, :2], [7, 3])


def test_graph_displacement_distance_direction_and_coincident_atom_semantics() -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        SCALE,
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(1, maximum_dense_atoms=2),
        cutoff=1.0,
    )
    np.testing.assert_allclose(graph.graph.edges["distance"], 0.0)
    np.testing.assert_allclose(graph.graph.edges["direction"], 0.0)
    assert bool(jnp.all(jnp.isfinite(graph.graph.edges["direction"])))
    # ty: ignore[invalid-argument-type]
    assert bool(jnp.all(graph.graph.edge_mask))


def test_neighborhood_overflow_is_reported_without_truncation() -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0],
        SCALE,
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(1, maximum_dense_atoms=3),
        cutoff=1.0,
    )
    assert bool(graph.overflow[0])
    assert int(graph.maximum_neighbor_count[0]) == 2
    # ty: ignore[invalid-argument-type]
    assert int(jnp.sum(graph.graph.edge_mask)) == 6


def test_dense_graph_guards_before_candidate_allocation(monkeypatch: Any) -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        SCALE,
    )
    batch = AtomisticBatch.from_structure(structure)

    def forbidden_allocation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("candidate allocation happened before the guard")

    monkeypatch.setattr(graph_module.np, "repeat", forbidden_allocation)
    with pytest.raises(ValueError, match="resource guard"):
        realize_atomistic_graph(
            batch, AtomisticGraphExecutionPlan(1, maximum_dense_atoms=1), cutoff=2.0
        )


def test_scale_mismatch_prevents_batch_construction() -> None:
    second_scale = AtomisticScaleContract(METER, JOULE)
    # ty: ignore[invalid-argument-type]
    first = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], SCALE)
    # ty: ignore[invalid-argument-type]
    second = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], second_scale)
    with pytest.raises(ValueError, match="scale"):
        AtomisticBatch.from_structures((first, second))


def test_with_positions_preserves_topology_and_refreshes_content_identity() -> None:
    batch = AtomisticBatch.from_structure(
        AtomicStructure(
            # ty: ignore[invalid-argument-type]
            [1, 1],
            # ty: ignore[invalid-argument-type]
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            # ty: ignore[invalid-argument-type]
            [1.0, 1.0],
            SCALE,
        )
    )
    moved = batch.with_positions(batch.positions + 0.25)
    assert moved.atom_topology_id == batch.atom_topology_id
    assert moved.batch_id != batch.batch_id


IMAGE_CAPACITY = ParticleImageCapacity(
    maximum_particles_per_cell=8,
    maximum_edges=8192,
    maximum_degree=1024,
    maximum_images=4096,
)


def _periodic_structure(
    numbers: list[int],
    fractional: np.ndarray,
    cell: np.ndarray,
    periodic: tuple[bool, bool, bool],
    ids: list[int],
) -> AtomicStructure:
    return AtomicStructure(
        # ty: ignore[invalid-argument-type]
        numbers,
        fractional @ cell,
        # ty: ignore[invalid-argument-type]
        [1.0] * len(numbers),
        SCALE,
        # ty: ignore[invalid-argument-type]
        particle_ids=ids,
        cell=cell,
        # ty: ignore[invalid-argument-type]
        periodic_axes=list(periodic),
    )


def _lattice_routes(
    positions: np.ndarray, cell: np.ndarray, periodic: Any, radius: float
) -> set[tuple[int, int, tuple[int, ...]]]:
    choices = [range(-6, 7) if axis else (0,) for axis in periodic]
    routes = set()
    for shift in product(*choices):
        translation = np.asarray(shift, dtype=np.float64) @ cell
        for source, receiver in product(range(positions.shape[0]), repeat=2):
            if source == receiver and not any(shift):
                continue
            if (
                np.linalg.norm(positions[receiver] - positions[source] + translation)
                < radius
            ):
                routes.add((source, receiver, shift))
    return routes


def _graph_routes(graph: Any, case: int, atom_capacity: int) -> set:
    topology = graph.topology
    mask = np.asarray(graph.graph.edge_mask) & (np.asarray(topology.edge_cases) == case)
    return {
        (
            int(sender) - case * atom_capacity,
            int(receiver) - case * atom_capacity,
            tuple(int(value) for value in shift),
        )
        for sender, receiver, shift in zip(
            np.asarray(topology.senders)[mask],
            np.asarray(topology.receivers)[mask],
            np.asarray(topology.image_shifts)[mask],
            strict=True,
        )
    }


@pytest.mark.parametrize("backend", ["particle", "dense"])
def test_periodic_batch_topology_matches_lattice_oracle_with_case_isolation(
    backend: str,
) -> None:
    triclinic = np.asarray([[2.2, 0.0, 0.0], [0.6, 2.0, 0.0], [0.2, -0.3, 2.4]])
    slab = np.asarray([[2.5, 0.0, 0.0], [0.0, 2.3, 0.0], [0.0, 0.0, 9.0]])
    first = _periodic_structure(
        [8, 1, 1],
        np.asarray([[0.1, 0.2, 0.3], [0.9, 0.95, 0.1], [0.5, 0.5, 0.7]]),
        triclinic,
        (True, True, True),
        [10, 11, 12],
    )
    second = _periodic_structure(
        [6, 6],
        np.asarray([[0.2, 0.3, 0.4], [0.7, 0.6, 0.45]]),
        slab,
        (True, True, False),
        [3, 4],
    )
    batch = AtomisticBatch.from_structures((first, second))
    execution = (
        AtomisticGraphExecutionPlan(
            512, backend="particle", image_capacity=IMAGE_CAPACITY
        )
        if backend == "particle"
        else AtomisticGraphExecutionPlan(
            512, maximum_dense_atoms=3, image_capacity=IMAGE_CAPACITY
        )
    )
    cutoff = 2.9
    topology = prepare_atomistic_graph_topology(batch, execution, cutoff=cutoff)
    graph = realize_atomistic_graph(batch, execution, cutoff=cutoff, topology=topology)
    assert not bool(jnp.any(graph.overflow))
    positions = np.asarray(batch.positions)
    assert _graph_routes(graph, 0, 3) == _lattice_routes(
        positions[0], triclinic, (True, True, True), cutoff
    )
    assert _graph_routes(graph, 1, 3) == _lattice_routes(
        positions[1, :2], slab, (True, True, False), cutoff
    )
    candidate = np.asarray(topology.candidate_mask)
    np.testing.assert_array_equal(
        np.asarray(topology.senders)[candidate] // 3,
        np.asarray(topology.receivers)[candidate] // 3,
    )
    shifts = np.asarray(topology.image_shifts)
    cases = np.asarray(topology.edge_cases)
    cells = np.stack((triclinic, slab))
    flat = positions.reshape((-1, 3))
    active = np.asarray(graph.graph.edge_mask)
    expected = (
        flat[np.asarray(topology.receivers)]
        - flat[np.asarray(topology.senders)]
        + np.einsum("ei,eij->ej", shifts, cells[cases])
    )
    np.testing.assert_allclose(
        np.asarray(graph.graph.edges["displacement"])[active],
        expected[active],
        atol=1e-12,
    )


def test_periodic_batch_requires_explicit_topology_with_matching_atom_identity() -> None:
    cell = 3.0 * np.eye(3)
    fractional = np.asarray([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    batch = AtomisticBatch.from_structure(
        _periodic_structure([1, 1], fractional, cell, (True, True, True), [1, 2])
    )
    stale = AtomisticBatch.from_structure(
        _periodic_structure([1, 1], fractional, cell, (True, True, True), [7, 9])
    )
    execution = AtomisticGraphExecutionPlan(
        64, backend="particle", image_capacity=IMAGE_CAPACITY
    )
    with pytest.raises(ValueError, match="require a topology"):
        realize_atomistic_graph(
            batch, AtomisticGraphExecutionPlan(64, maximum_dense_atoms=2), cutoff=2.0
        )
    topology = prepare_atomistic_graph_topology(stale, execution, cutoff=2.0)
    assert topology.atom_topology_id == stale.atom_topology_id
    assert topology.atom_topology_id != batch.atom_topology_id
    with pytest.raises(ValueError, match="another atom identity"):
        realize_atomistic_graph(batch, execution, cutoff=2.0, topology=topology)


def _edge_energy(graph: Any) -> jax.Array:
    distance = graph.graph.edges["distance"][:, 0]
    return jnp.sum(jnp.where(graph.graph.edge_mask, jnp.exp(-distance), 0.0))


def test_primitive_and_supercell_agree_and_one_atom_cell_has_only_strain_response() -> (
    None
):
    primitive = np.diag([1.7, 1.9, 2.1])
    supercell = np.diag([3.4, 1.9, 2.1])
    one = AtomisticBatch.from_structure(
        _periodic_structure(
            [1], np.asarray([[0.1, 0.2, 0.3]]), primitive, (True, True, True), [1]
        )
    )
    two = AtomisticBatch.from_structure(
        _periodic_structure(
            [1, 1],
            np.asarray([[0.05, 0.2, 0.3], [0.55, 0.2, 0.3]]),
            supercell,
            (True, True, True),
            [1, 2],
        )
    )
    execution = AtomisticGraphExecutionPlan(
        512, backend="particle", image_capacity=IMAGE_CAPACITY
    )
    cutoff = 3.3
    one_topology = prepare_atomistic_graph_topology(
        one, execution, cutoff=cutoff, skin=0.1
    )
    two_topology = prepare_atomistic_graph_topology(
        two, execution, cutoff=cutoff, skin=0.1
    )

    def energy(topology: Any, positions: jax.Array, cells: jax.Array) -> jax.Array:
        graph = bind_atomistic_graph(
            topology, execution, positions, cutoff=cutoff, cell_vectors=cells
        )
        return graph.require_success(_edge_energy(graph))

    one_cells = one.cells
    two_cells = two.cells
    assert one_cells is not None and two_cells is not None
    one_energy = energy(one_topology, one.positions, one_cells)
    two_energy = energy(two_topology, two.positions, two_cells)
    np.testing.assert_allclose(2.0 * one_energy, two_energy, rtol=1e-12)
    position_gradient = jax.grad(energy, argnums=1)(
        one_topology, one.positions, one_cells
    )
    np.testing.assert_allclose(position_gradient, 0.0, atol=1e-12)
    two_gradient = jax.grad(energy, argnums=1)(two_topology, two.positions, two_cells)
    np.testing.assert_allclose(two_gradient, 0.0, atol=1e-12)
    cell_gradient = jax.grad(energy, argnums=2)(one_topology, one.positions, one_cells)
    assert float(jnp.max(jnp.abs(cell_gradient))) > 1e-3
    strain = 1.0e-6
    deformed = one_cells.at[0, 0, 0].multiply(1.0 + strain)
    finite_difference = (energy(one_topology, one.positions, deformed) - one_energy) / (
        strain * one_cells[0, 0, 0]
    )
    np.testing.assert_allclose(cell_gradient[0, 0, 0], finite_difference, rtol=1e-4)


def test_bound_topology_certificate_fails_when_positions_leave_the_skin() -> None:
    cell = 2.6 * np.eye(3)
    batch = AtomisticBatch.from_structure(
        _periodic_structure(
            [1, 1],
            np.asarray([[0.1, 0.1, 0.1], [0.6, 0.5, 0.4]]),
            cell,
            (True, True, True),
            [1, 2],
        )
    )
    execution = AtomisticGraphExecutionPlan(
        512, backend="particle", image_capacity=IMAGE_CAPACITY
    )
    topology = prepare_atomistic_graph_topology(batch, execution, cutoff=2.0, skin=0.4)
    near = batch.positions.at[0, 1, 0].add(0.1).reshape((-1, 3))
    assert not bool(
        jnp.any(bind_atomistic_graph(topology, execution, near, cutoff=2.0).overflow)
    )
    far = batch.positions.at[0, 1, 0].add(0.3).reshape((-1, 3))
    assert bool(
        jnp.all(bind_atomistic_graph(topology, execution, far, cutoff=2.0).overflow)
    )
    with pytest.raises(ValueError, match="candidate radius"):
        bind_atomistic_graph(topology, execution, near, cutoff=2.5)


def test_edge_slots_rank_active_edges_per_receiver_by_sender() -> None:
    structure = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 1, 1, 1],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.6, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0, 1.0, 1.0],
        SCALE,
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(4, maximum_dense_atoms=4),
        cutoff=1.0,
    )
    senders = np.asarray(graph.graph.senders)
    receivers = np.asarray(graph.graph.receivers)
    active = np.asarray(graph.graph.edge_mask)
    slots = np.asarray(graph.edge_slots)
    for receiver in range(4):
        incoming = np.flatnonzero(active & (receivers == receiver))
        ordered = incoming[np.argsort(senders[incoming])]
        np.testing.assert_array_equal(slots[ordered], np.arange(ordered.size))
    np.testing.assert_array_equal(slots[~active], 0)


def test_nonfinite_active_geometry_invalidates_edgeless_case_but_masked_padding_does_not() -> (
    None
):
    first = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 0],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 0.0],
        SCALE,
        # ty: ignore[invalid-argument-type]
        active_mask=[True, False],
    )
    second = AtomicStructure(
        # ty: ignore[invalid-argument-type]
        [1, 0],
        # ty: ignore[invalid-argument-type]
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        # ty: ignore[invalid-argument-type]
        [1.0, 0.0],
        SCALE,
        # ty: ignore[invalid-argument-type]
        active_mask=[True, False],
    )
    batch = AtomisticBatch.from_structures((first, second))
    execution = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=2)
    finite = realize_atomistic_graph(batch, execution, cutoff=1.0)
    np.testing.assert_array_equal(finite.valid, [True, True])
    masked = batch.positions.at[0, 1].set(jnp.nan).at[1, 1].set(jnp.inf)
    padded = realize_atomistic_graph(batch, execution, cutoff=1.0, positions=masked)
    np.testing.assert_array_equal(padded.nonfinite, [False, False])
    np.testing.assert_array_equal(padded.valid, [True, True])
    active = batch.positions.at[1, 0, 2].set(jnp.nan)
    graph = realize_atomistic_graph(batch, execution, cutoff=1.0, positions=active)
    edge_mask = graph.graph.edge_mask
    assert edge_mask is not None
    assert int(jnp.sum(edge_mask)) == 0
    np.testing.assert_array_equal(graph.overflow, [False, False])
    np.testing.assert_array_equal(graph.nonfinite, [False, True])
    np.testing.assert_array_equal(graph.valid, [True, False])
    with pytest.raises(Exception, match="nonfinite geometry"):
        jax.block_until_ready(graph.require_success(jnp.zeros(())))


def test_nonfinite_runtime_cell_invalidates_periodic_case() -> None:
    cell = 3.0 * np.eye(3)
    batch = AtomisticBatch.from_structure(
        _periodic_structure(
            [1], np.asarray([[0.1, 0.2, 0.3]]), cell, (True, True, True), [1]
        )
    )
    execution = AtomisticGraphExecutionPlan(
        64, backend="particle", image_capacity=IMAGE_CAPACITY
    )
    topology = prepare_atomistic_graph_topology(batch, execution, cutoff=2.0)
    assert batch.cells is not None
    cells = batch.cells.at[0, 2, 2].set(jnp.nan)
    graph = bind_atomistic_graph(
        topology, execution, batch.positions, cutoff=2.0, cell_vectors=cells
    )
    np.testing.assert_array_equal(graph.nonfinite, [True])
    np.testing.assert_array_equal(graph.valid, [False])


def _direct_batch(
    numbers: list[int],
    positions: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    periodic: tuple[bool, bool, bool] | None = None,
) -> AtomisticBatch:
    """Batch with an explicit case identity independent of species and periodicity."""
    return AtomisticBatch(
        # ty: ignore[invalid-argument-type]
        [numbers],
        positions[None],
        # ty: ignore[invalid-argument-type]
        [[1.0] * len(numbers)],
        SCALE,
        # ty: ignore[invalid-argument-type]
        particle_ids=[list(range(3, 3 + len(numbers)))],
        cells=None if cell is None else cell[None],
        # ty: ignore[invalid-argument-type]
        periodic_axes=None if periodic is None else [list(periodic)],
        structure_ids=("case",),
    )


def test_cached_topology_refuses_changed_periodicity_but_fresh_slab_is_exact() -> None:
    cell = 2.0 * np.eye(3)
    position = np.asarray([[0.6, 0.6, 0.6]])
    full = _direct_batch([1], position, cell=cell, periodic=(True, True, True))
    slab = _direct_batch([1], position, cell=cell, periodic=(True, True, False))
    assert full.atom_topology_id == slab.atom_topology_id
    execution = AtomisticGraphExecutionPlan(
        64, backend="particle", image_capacity=IMAGE_CAPACITY
    )
    cached = prepare_atomistic_graph_topology(full, execution, cutoff=2.1)
    with pytest.raises(ValueError, match="periodic axes differ"):
        realize_atomistic_graph(slab, execution, cutoff=2.1, topology=cached)
    traced = jax.jit(
        lambda axes: (
            bind_atomistic_graph(
                cached,
                execution,
                slab.positions,
                cutoff=2.1,
                cell_vectors=slab.cells,
                periodic_axes=axes,
            ).overflow
        )
    )
    np.testing.assert_array_equal(traced(slab.periodic_axes), [True])
    np.testing.assert_array_equal(traced(full.periodic_axes), [False])
    fresh = prepare_atomistic_graph_topology(slab, execution, cutoff=2.1)
    graph = realize_atomistic_graph(slab, execution, cutoff=2.1, topology=fresh)
    assert bool(graph.valid[0])
    shifts = np.asarray(fresh.image_shifts)[np.asarray(graph.graph.edge_mask)]
    assert {tuple(int(value) for value in row) for row in shifts} == {
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    }


def test_cached_topology_publishes_current_batch_species_and_masses() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    hydrogen = _direct_batch([1, 1], positions)
    oxygen = _direct_batch([8, 1], positions)
    assert hydrogen.atom_topology_id == oxygen.atom_topology_id
    execution = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=2)
    topology = prepare_atomistic_graph_topology(hydrogen, execution, cutoff=1.0)
    np.testing.assert_array_equal(topology.atomic_numbers, [1, 1])
    graph = realize_atomistic_graph(oxygen, execution, cutoff=1.0, topology=topology)
    np.testing.assert_array_equal(graph.graph.nodes["atomic_numbers"], [8, 1])
    np.testing.assert_array_equal(
        graph.graph.nodes["atom_type_ids"], oxygen.atom_type_ids.reshape((-1,))
    )
    np.testing.assert_allclose(graph.graph.nodes["masses"], [1.0, 1.0])
