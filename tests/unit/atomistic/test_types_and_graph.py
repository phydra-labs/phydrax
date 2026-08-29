import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.atomistic import (
    AtomisticBatch,
    AtomisticScaleContract,
    AtomicStructure,
    realize_atomistic_graph,
)


SCALE = AtomisticScaleContract("angstrom", "electronvolt")


def test_structure_preserves_particles_masks_ids_masses_and_scale():
    structure = AtomicStructure(
        [8, 1, 1, 0],
        [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.2, 0.7, 0.0], [0.0, 0.0, 0.0]],
        [15.999, 1.008, 1.008, 0.0],
        SCALE,
        particle_ids=[40, 11, 23, 99],
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
        ([1, 6], [True, False], "padding"),
    ],
)
def test_atomic_number_zero_is_padding_only(numbers, mask, message):
    with pytest.raises(ValueError, match=message):
        AtomicStructure(
            numbers,
            np.zeros((2, 3)),
            np.ones((2,)),
            SCALE,
            active_mask=mask,
        )


def test_batch_padding_does_not_change_structure_identity_or_graph_isolation():
    hydrogen = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.008], SCALE, name="h")
    oxygen = AtomicStructure(
        [8, 8],
        [[100.0, 0.0, 0.0], [101.0, 0.0, 0.0]],
        [15.999, 15.999],
        SCALE,
        particle_ids=[7, 3],
        name="o2",
    )
    batch = AtomisticBatch.from_structures((hydrogen, oxygen), atom_capacity=3)
    graph = realize_atomistic_graph(
        batch, cutoff=2.0, maximum_neighbors=2, maximum_dense_atoms=3
    )
    assert graph.graph.num_graphs == 2
    assert graph.graph.nodes["atomic_numbers"].shape == (6,)
    assert not bool(jnp.any(graph.overflow))
    send_case = graph.graph.senders // batch.atom_capacity
    receive_case = graph.graph.receivers // batch.atom_capacity
    np.testing.assert_array_equal(send_case, receive_case)
    np.testing.assert_array_equal(batch.atomic_numbers[0], [1, 0, 0])
    np.testing.assert_array_equal(batch.particle_ids[1, :2], [7, 3])


def test_graph_displacement_distance_direction_and_coincident_atom_semantics():
    structure = AtomicStructure(
        [1, 1],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [1.0, 1.0],
        SCALE,
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        cutoff=1.0,
        maximum_neighbors=1,
        maximum_dense_atoms=2,
    )
    np.testing.assert_allclose(graph.graph.edges["distance"], 0.0)
    np.testing.assert_allclose(graph.graph.edges["direction"], 0.0)
    assert bool(jnp.all(jnp.isfinite(graph.graph.edges["direction"])))
    assert bool(jnp.all(graph.graph.edge_mask))


def test_neighborhood_overflow_is_reported_without_truncation():
    structure = AtomicStructure(
        [1, 1, 1],
        [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]],
        [1.0, 1.0, 1.0],
        SCALE,
    )
    graph = realize_atomistic_graph(
        AtomisticBatch.from_structure(structure),
        cutoff=1.0,
        maximum_neighbors=1,
        maximum_dense_atoms=3,
    )
    assert bool(graph.overflow[0])
    assert int(graph.maximum_neighbor_count[0]) == 2
    assert int(jnp.sum(graph.graph.edge_mask)) == 6


def test_dense_graph_requires_explicit_atom_resource_guard():
    structure = AtomicStructure(
        [1, 1], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 1.0], SCALE
    )
    with pytest.raises(ValueError, match="resource guard"):
        realize_atomistic_graph(
            AtomisticBatch.from_structure(structure),
            cutoff=2.0,
            maximum_neighbors=1,
            maximum_dense_atoms=1,
        )


def test_scale_mismatch_prevents_batch_construction():
    second_scale = AtomisticScaleContract("bohr", "hartree")
    first = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], SCALE)
    second = AtomicStructure([1], [[0.0, 0.0, 0.0]], [1.0], second_scale)
    with pytest.raises(ValueError, match="scale"):
        AtomisticBatch.from_structures((first, second))
