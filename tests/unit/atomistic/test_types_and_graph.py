from copy import copy, deepcopy
from itertools import product
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.atomistic._graph as graph_module
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
    AtomisticScaleContract,
    bind_atomistic_graph,
)
from phydrax.discretization import ParticleImageCapacity
from phydrax.discretization.particle import ImageNeighborSelectionPolicy
from phydrax.units import ANGSTROM, ELECTRONVOLT, JOULE, METER


SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)


def _payload(
    batch: AtomisticBatch,
    execution: AtomisticGraphExecutionPlan,
    *,
    cutoff: float,
    skin: float = 0.0,
) -> AtomisticGraphPayload:
    return AtomisticGraphLayout.for_batch(
        batch, execution, cutoff=cutoff, skin=skin
    ).prepare(batch)


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


def test_structure_mass_precision_survives_copies_batching_and_position_updates() -> None:
    arguments = (
        np.asarray([1, 1], dtype=np.int32),
        np.zeros((2, 3), dtype=np.float32),
        np.asarray([2**24, 1], dtype=np.float32),
        SCALE,
    )
    default = AtomicStructure(*arguments, coordinate_dtype=jnp.float32)
    selected = AtomicStructure(
        *arguments,
        coordinate_dtype=jnp.float32,
        mass_accumulation_dtype=jnp.float32,
    )

    assert default.particles.mass_measure.total_mass.dtype == jnp.float64
    assert float(default.particles.mass_measure.total_mass) == 2**24 + 1
    assert selected.particles.mass_measure.total_mass.dtype == jnp.float32
    assert float(selected.particles.mass_measure.total_mass) == 2**24
    assert selected.structure_id != default.structure_id
    assert selected.particles.mass_measure.measure_id == (
        default.particles.mass_measure.measure_id
    )
    for duplicate in (copy(selected), deepcopy(selected)):
        assert duplicate.structure_id == selected.structure_id
        assert duplicate.particles.mass_measure.total_mass.dtype == jnp.float32

    batch = AtomisticBatch.from_structure(selected)
    padded = AtomisticBatch.from_structures((selected,), atom_capacity=4)
    moved = padded.with_positions(padded.positions + np.float32(0.25))
    for value in (batch, padded, moved):
        assert value.particles[0].mass_measure.total_mass.dtype == jnp.float32
        floating = [
            leaf
            for leaf in jax.tree.leaves(value)
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
        ]
        assert floating
        assert all(leaf.dtype == jnp.float32 for leaf in floating)
    assert moved.particles[0].prepared_id == padded.particles[0].prepared_id
    assert moved.batch_id != padded.batch_id
    mixed = AtomisticBatch.from_structures((selected, default), atom_capacity=3)
    assert tuple(value.mass_measure.total_mass.dtype for value in mixed.particles) == (
        jnp.float32,
        jnp.float64,
    )
    assert mixed.with_positions(mixed.positions).batch_id == mixed.batch_id


def test_batch_mass_precision_is_explicit_and_part_of_container_identity() -> None:
    arguments = (
        np.asarray([[1, 1]], dtype=np.int32),
        np.zeros((1, 2, 3), dtype=np.float32),
        np.asarray([[2**24, 1]], dtype=np.float32),
        SCALE,
    )
    default = AtomisticBatch(*arguments)
    selected = AtomisticBatch(*arguments, mass_accumulation_dtype=jnp.float32)
    assert default.particles[0].mass_measure.total_mass.dtype == jnp.float64
    assert selected.particles[0].mass_measure.total_mass.dtype == jnp.float32
    assert selected.batch_id != default.batch_id
    assert selected.with_positions(selected.positions).batch_id == selected.batch_id
    with pytest.raises(ValueError, match="one precision per case"):
        AtomisticBatch(*arguments, mass_accumulation_dtype=(jnp.float32, jnp.float64))


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_, jnp.complex64])
def test_atomic_structure_refuses_nonreal_mass_precision(dtype: Any) -> None:
    with pytest.raises(ValueError, match="precision dtype|not real"):
        AtomicStructure(
            np.asarray([1], dtype=np.int32),
            np.zeros((1, 3), dtype=np.float32),
            np.ones(1, dtype=np.float32),
            SCALE,
            mass_accumulation_dtype=dtype,
        )
    with pytest.raises(ValueError, match="precision dtype|not real"):
        AtomisticBatch(
            np.asarray([[1]], dtype=np.int32),
            np.zeros((1, 1, 3), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            SCALE,
            mass_accumulation_dtype=dtype,
        )


def test_atomic_structure_refuses_overflow_in_selected_mass_precision() -> None:
    arguments = (
        np.asarray([1, 1], dtype=np.int32),
        np.zeros((2, 3), dtype=np.float32),
        np.full(2, np.finfo(np.float32).max, dtype=np.float32),
        SCALE,
    )
    default = AtomicStructure(*arguments)
    assert bool(default.particles.mass_measure.valid)
    with pytest.raises(ValueError, match="total mass must be finite"):
        AtomicStructure(*arguments, mass_accumulation_dtype=jnp.float32)


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
    graph = _payload(
        batch, AtomisticGraphExecutionPlan(2, maximum_dense_atoms=3), cutoff=2.0
    ).graph(cutoff=2.0)
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
    graph = _payload(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(1, maximum_dense_atoms=2),
        cutoff=1.0,
    ).graph(cutoff=1.0)
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
    graph = _payload(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(1, maximum_dense_atoms=3),
        cutoff=1.0,
    ).graph(cutoff=1.0)
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
        AtomisticGraphLayout.for_batch(
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
    payload = _payload(batch, execution, cutoff=cutoff)
    topology = payload.topology
    graph = payload.graph(cutoff=cutoff)
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


def test_payload_admission_refuses_foreign_material_as_scientific_failure() -> None:
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
    with pytest.raises(ValueError, match="image_capacity"):
        AtomisticGraphLayout.for_batch(
            batch, AtomisticGraphExecutionPlan(64, maximum_dense_atoms=2), cutoff=2.0
        )
    payload = _payload(stale, execution, cutoff=2.0)
    assert (
        payload.topology.content_id()
        != _payload(batch, execution, cutoff=2.0).topology.content_id()
    )
    graph = payload.admit(batch).graph(cutoff=2.0)
    np.testing.assert_array_equal(graph.overflow, [False])
    np.testing.assert_array_equal(graph.topology.search_failure, [True])
    np.testing.assert_array_equal(graph.valid, [False])
    assert bool(payload.admit(stale).graph(cutoff=2.0).valid[0])


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
    one_topology = _payload(one, execution, cutoff=cutoff, skin=0.1).topology
    two_topology = _payload(two, execution, cutoff=cutoff, skin=0.1).topology

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
    topology = _payload(batch, execution, cutoff=2.0, skin=0.4).topology
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
    graph = _payload(
        AtomisticBatch.from_structure(structure),
        AtomisticGraphExecutionPlan(4, maximum_dense_atoms=4),
        cutoff=1.0,
    ).graph(cutoff=1.0)
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
    payload = _payload(batch, execution, cutoff=1.0)
    finite = payload.graph(cutoff=1.0)
    np.testing.assert_array_equal(finite.valid, [True, True])
    masked = batch.positions.at[0, 1].set(jnp.nan).at[1, 1].set(jnp.inf)
    padded = payload.graph(cutoff=1.0, positions=masked)
    np.testing.assert_array_equal(padded.nonfinite, [False, False])
    np.testing.assert_array_equal(padded.valid, [True, True])
    active = batch.positions.at[1, 0, 2].set(jnp.nan)
    graph = payload.graph(cutoff=1.0, positions=active)
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
    topology = _payload(batch, execution, cutoff=2.0).topology
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


def test_cached_payload_refuses_changed_periodicity_but_fresh_slab_is_exact() -> None:
    cell = 2.0 * np.eye(3)
    position = np.asarray([[0.6, 0.6, 0.6]])
    full = _direct_batch([1], position, cell=cell, periodic=(True, True, True))
    slab = _direct_batch([1], position, cell=cell, periodic=(True, True, False))
    execution = AtomisticGraphExecutionPlan(
        64, backend="particle", image_capacity=IMAGE_CAPACITY
    )
    cached_payload = _payload(full, execution, cutoff=2.1)
    cached = cached_payload.topology
    with pytest.raises(ValueError, match="periodic axes differ"):
        cached_payload.admit(slab).graph(cutoff=2.1)
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
    fresh_payload = _payload(slab, execution, cutoff=2.1)
    fresh = fresh_payload.topology
    graph = fresh_payload.graph(cutoff=2.1)
    assert bool(graph.valid[0])
    shifts = np.asarray(fresh.image_shifts)[np.asarray(graph.graph.edge_mask)]
    assert {tuple(int(value) for value in row) for row in shifts} == {
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
    }


def test_cached_payload_refuses_other_species_and_fresh_payload_publishes_them() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    hydrogen = _direct_batch([1, 1], positions)
    oxygen = _direct_batch([8, 1], positions)
    execution = AtomisticGraphExecutionPlan(4, maximum_dense_atoms=2)
    cached = _payload(hydrogen, execution, cutoff=1.0)
    np.testing.assert_array_equal(cached.topology.atomic_numbers, [1, 1])
    refused = cached.admit(oxygen).graph(cutoff=1.0)
    np.testing.assert_array_equal(refused.overflow, [False])
    np.testing.assert_array_equal(refused.valid, [False])
    assert cached.layout_id == _payload(oxygen, execution, cutoff=1.0).layout_id
    graph = _payload(oxygen, execution, cutoff=1.0).graph(cutoff=1.0)
    np.testing.assert_array_equal(graph.graph.nodes["atomic_numbers"], [8, 1])
    np.testing.assert_array_equal(
        graph.graph.nodes["atom_type_ids"], oxygen.atom_type_ids.reshape((-1,))
    )
    np.testing.assert_allclose(graph.graph.nodes["masses"], [1.0, 1.0])


ENVELOPE = 3.0 * np.eye(3)


def _envelope_execution(
    maximum_particles_per_cell: int = 8,
) -> AtomisticGraphExecutionPlan:
    return AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=ParticleImageCapacity(
            maximum_particles_per_cell=maximum_particles_per_cell,
            maximum_edges=4096,
            maximum_degree=512,
            maximum_images=4096,
        ),
        search_cell=ENVELOPE,
    )


def _envelope_layout(execution: AtomisticGraphExecutionPlan) -> AtomisticGraphLayout:
    return AtomisticGraphLayout(
        SCALE,
        execution,
        case_capacity=1,
        atom_capacity=4,
        coordinate_dtype=jnp.float64,
        cutoff=1.0,
        periodic_axes=(True, True, True),
    )


def _envelope_batch(
    numbers: list[int],
    fractional: np.ndarray,
    cell: np.ndarray,
    *,
    ids: list[int],
    name: str,
) -> AtomisticBatch:
    count = len(numbers)
    padded = np.zeros((4, 3), dtype=np.float64)
    padded[:count] = fractional @ cell
    return AtomisticBatch(
        # ty: ignore[invalid-argument-type]
        [numbers + [0] * (4 - count)],
        padded[None],
        # ty: ignore[invalid-argument-type]
        [[1.0] * 4],
        SCALE,
        # ty: ignore[invalid-argument-type]
        particle_ids=[ids + list(range(1000, 1000 + 4 - count))],
        # ty: ignore[invalid-argument-type]
        atom_mask=[[True] * count + [False] * (4 - count)],
        cells=cell[None],
        # ty: ignore[invalid-argument-type]
        periodic_axes=[[True, True, True]],
        structure_ids=(name,),
        coordinate_dtype=jnp.float64,
    )


def test_layout_payloads_share_one_static_structure_and_compile_once() -> None:
    """Compile-count experiment: payloads of one layout never recompile."""
    execution = _envelope_execution()
    layout = _envelope_layout(execution)
    rng = np.random.default_rng(7)
    batches = [
        _envelope_batch(
            [int(value) for value in rng.integers(1, 9, size=count)],
            rng.uniform(0.0, 1.0, size=(count, 3)),
            float(rng.uniform(3.0, 3.6)) * np.eye(3),
            ids=[int(value) for value in rng.choice(10_000, size=count, replace=False)],
            name=f"record-{index}",
        )
        for index, count in enumerate((2, 3, 4, 3, 2))
    ]
    assert len({batch.batch_id for batch in batches}) == len(batches)

    @eqx.filter_jit
    def objective(payload: AtomisticGraphPayload) -> tuple[jax.Array, jax.Array]:
        def energy(positions: jax.Array) -> jax.Array:
            graph = payload.graph(cutoff=1.0, positions=positions)
            return _edge_energy(graph)

        return jax.value_and_grad(energy)(payload.positions)

    compiles: list[str] = []
    recording = [False]

    def listener(event: str, duration_secs: float, **kwargs: str | int) -> None:
        del duration_secs, kwargs
        if recording[0] and event == "/jax/core/compile/backend_compile_duration":
            compiles.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        first = layout.prepare(batches[0])
        jax.block_until_ready(objective(first))
        recording[0] = True
        payloads = []
        for batch in batches[1:]:
            payload = _envelope_layout(execution).prepare(batch)
            jax.block_until_ready(objective(payload))
            payloads.append(payload)
    finally:
        recording[0] = False
    assert compiles == []
    structure = jax.tree_util.tree_structure(first)
    for payload in payloads:
        assert payload.layout_id == first.layout_id
        assert payload.topology.topology_id == first.topology.topology_id
        assert jax.tree_util.tree_structure(payload) == structure
        assert payload.topology.content_id() != first.topology.content_id()
        assert not bool(jnp.any(payload.topology.overflow))
        assert not bool(jnp.any(payload.topology.search_failure))
    words = {tuple(np.asarray(p.topology.case_identity_words)[0]) for p in payloads}
    assert len(words) == len(payloads)


def test_layout_envelope_capacity_failure_is_distinct_from_scientific_failure() -> None:
    execution = _envelope_execution(maximum_particles_per_cell=2)
    layout = _envelope_layout(execution)
    crowded = _envelope_batch(
        [1, 1, 1, 1],
        np.asarray(
            [
                [0.10, 0.10, 0.10],
                [0.12, 0.10, 0.10],
                [0.14, 0.10, 0.10],
                [0.16, 0.10, 0.10],
            ]
        ),
        ENVELOPE,
        ids=[1, 2, 3, 4],
        name="crowded",
    )
    narrow = _envelope_batch(
        [1, 1],
        np.asarray([[0.1, 0.1, 0.1], [0.6, 0.5, 0.4]]),
        2.0 * np.eye(3),
        ids=[5, 6],
        name="narrow",
    )
    admitted = _envelope_batch(
        [1, 1],
        np.asarray([[0.1, 0.1, 0.1], [0.6, 0.5, 0.4]]),
        3.3 * np.eye(3),
        ids=[7, 8],
        name="admitted",
    )
    capacity = layout.prepare(crowded)
    scientific = layout.prepare(narrow)
    valid = layout.prepare(admitted)
    assert capacity.layout_id == scientific.layout_id == valid.layout_id
    assert jax.tree_util.tree_structure(capacity) == jax.tree_util.tree_structure(
        scientific
    )
    evidence = capacity.topology.evidence
    assert evidence is not None
    np.testing.assert_array_equal(evidence.cell_overflow, [True])
    np.testing.assert_array_equal(capacity.topology.overflow, [True])
    np.testing.assert_array_equal(capacity.topology.search_failure, [False])
    narrow_evidence = scientific.topology.evidence
    assert narrow_evidence is not None
    np.testing.assert_array_equal(narrow_evidence.stencil_overflow, [True])
    np.testing.assert_array_equal(scientific.topology.overflow, [False])
    np.testing.assert_array_equal(scientific.topology.search_failure, [True])
    capacity_graph = capacity.graph(cutoff=1.0)
    scientific_graph = scientific.graph(cutoff=1.0)
    np.testing.assert_array_equal(capacity_graph.overflow, [True])
    np.testing.assert_array_equal(scientific_graph.overflow, [False])
    np.testing.assert_array_equal(scientific_graph.valid, [False])
    np.testing.assert_array_equal(valid.graph(cutoff=1.0).valid, [True])


def test_layout_refuses_batches_outside_its_static_class() -> None:
    execution = _envelope_execution()
    finite = AtomisticGraphLayout(
        SCALE,
        AtomisticGraphExecutionPlan(4, maximum_dense_atoms=4),
        case_capacity=1,
        atom_capacity=4,
        coordinate_dtype=jnp.float64,
        cutoff=1.0,
    )
    assert finite.lattice_rank == 0
    periodic = _envelope_batch(
        [1, 1],
        np.asarray([[0.1, 0.1, 0.1], [0.6, 0.5, 0.4]]),
        ENVELOPE,
        ids=[1, 2],
        name="periodic",
    )
    with pytest.raises(ValueError, match="cell class"):
        finite.prepare(periodic)
    with pytest.raises(ValueError, match="search envelope"):
        AtomisticGraphLayout(
            SCALE,
            AtomisticGraphExecutionPlan(
                64, backend="particle", image_capacity=IMAGE_CAPACITY
            ),
            case_capacity=1,
            atom_capacity=4,
            coordinate_dtype=jnp.float64,
            cutoff=1.0,
        )
    slab_layout = AtomisticGraphLayout(
        SCALE,
        execution,
        case_capacity=1,
        atom_capacity=4,
        coordinate_dtype=jnp.float64,
        cutoff=1.0,
        periodic_axes=(True, True, False),
    )
    with pytest.raises(ValueError, match="periodicity class"):
        slab_layout.prepare(periodic)
    other = _envelope_batch(
        [1], np.asarray([[0.1, 0.1, 0.1]]), ENVELOPE, ids=[1], name="other"
    )
    assert _envelope_layout(execution).prepare(other).layout_id == (
        _envelope_layout(execution).layout_id
    )


@pytest.mark.parametrize("backend", ["dense", "particle"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_image_layout_covers_closed_selection_boundary(
    backend: str, dtype: type[np.float32] | type[np.float64]
) -> None:
    cutoff = 2.5
    structure = AtomicStructure(
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0], [cutoff, 0.0, 0.0]], dtype=dtype),
        np.ones(2, dtype=dtype),
        SCALE,
        cell=10.0 * np.eye(3, dtype=dtype),
        periodic_axes=np.ones(3, dtype=np.bool_),
        coordinate_dtype=dtype,
    )
    execution = (
        AtomisticGraphExecutionPlan(
            4, maximum_dense_atoms=2, image_capacity=IMAGE_CAPACITY
        )
        if backend == "dense"
        else AtomisticGraphExecutionPlan(
            4, backend="particle", image_capacity=IMAGE_CAPACITY
        )
    )
    batch = AtomisticBatch.from_structure(structure)
    payload = _payload(batch, execution, cutoff=cutoff)
    selection = ImageNeighborSelectionPolicy(
        cutoff,
        1,
        degeneracy_tolerance=0.0,
        measure="squared-distance",
        strict=True,
        radius_boundary="closed",
        coincidence_threshold=0.0,
    )
    closed = payload.graph(cutoff=cutoff, selection=selection)
    np.testing.assert_array_equal(closed.valid, [True])
    assert _graph_routes(closed, 0, 2) == {
        (0, 1, (0, 0, 0)),
        (1, 0, (0, 0, 0)),
    }
    # Padding belongs only to candidate storage, not the physical open cutoff.
    opened = payload.graph(cutoff=cutoff)
    np.testing.assert_array_equal(opened.valid, [True])
    assert _graph_routes(opened, 0, 2) == set()
