#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _graph(*, masked: bool = False):
    return phx.graph.GraphIR(
        nodes={"type": jnp.asarray([0, 1, 1], dtype=jnp.int32)},
        edges={"weight": jnp.ones((3,))},
        senders=jnp.asarray([0, 1, 2]),
        receivers=jnp.asarray([1, 2, 0]),
        n_node=jnp.asarray([3]),
        n_edge=jnp.asarray([3]),
        node_mask=jnp.asarray([True, True, not masked]),
        edge_mask=jnp.asarray([True, not masked, True]),
        graph_mask=jnp.asarray([True]),
    )


def _batch(*, source_mask=None):
    graph = _graph()
    topology = phx.nn.OperatorTopology.from_graph(graph)
    coordinates = jnp.asarray([[0.0], [0.5], [1.0]])
    source = phx.nn.FunctionSamples(
        values=jnp.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        coordinates=coordinates,
        quadrature_weights=jnp.asarray([0.2, 0.3, 0.5]),
        mask=source_mask,
        topology=topology,
    )
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=coordinates,
        topology=topology,
    )
    return phx.nn.OperatorBatch(inputs={"u": source}, queries={"query": query}, case_axes=("case",),)


def test_graph_batch_roundtrip_preserves_all_masks():
    first = _graph(masked=False)
    second = _graph(masked=True)
    batched = phx.graph.batch_graphs((first, second))
    restored = phx.graph.unbatch_graph(batched)

    assert jnp.array_equal(batched.node_mask, jnp.concatenate((first.node_mask, second.node_mask)))
    assert jnp.array_equal(batched.edge_mask, jnp.concatenate((first.edge_mask, second.edge_mask)))
    assert jnp.array_equal(batched.graph_mask, jnp.asarray([True, True]))
    for expected, actual in zip((first, second), restored, strict=True):
        assert jnp.array_equal(actual.node_mask, expected.node_mask)
        assert jnp.array_equal(actual.edge_mask, expected.edge_mask)
        assert jnp.array_equal(actual.graph_mask, expected.graph_mask)


def test_operator_topology_materializes_and_gathers_case_local_graph_fields():
    batch = _batch(source_mask=jnp.asarray([[True, True, False], [True, True, True]]))
    graph = phx.nn.operator_graph_from_samples(
        batch.input("u"), case_shape=batch.case_shape
    )
    gathered = phx.nn.gather_operator_graph_entities(
        batch.require_single_query(),
        graph.nodes["features"],
        case_shape=batch.case_shape,
    )

    assert graph.num_graphs == 2
    assert graph.nodes["features"].shape == (6,)
    assert jnp.array_equal(graph.nodes["type"], jnp.asarray([0, 1, 1, 0, 1, 1]))
    assert jnp.array_equal(graph.nodes["sample_mask"], jnp.asarray([True, True, False, True, True, True]))
    assert jnp.array_equal(graph.nodes["features"], jnp.asarray([0.0, 1.0, 0.0, 3.0, 4.0, 5.0]))
    assert jnp.array_equal(gathered, jnp.asarray([[0.0, 1.0, 0.0], [3.0, 4.0, 5.0]]))


def test_operator_topology_materializes_edge_and_global_entity_fields():
    graph = _graph()
    edge_topology = phx.nn.OperatorTopology.from_graph(graph, site="edge")
    edge_samples = phx.nn.FunctionSamples(
        values=jnp.asarray([2.0, 3.0, 5.0]),
        coordinates=jnp.asarray([[0.25], [0.5], [0.75]]),
        quadrature_weights=jnp.asarray([0.2, 0.3, 0.5]),
        topology=edge_topology,
    )
    edge_graph = phx.nn.operator_graph_from_samples(edge_samples)
    edge_values = phx.nn.gather_operator_graph_entities(
        edge_samples,
        edge_graph.edges["features"],
    )

    global_topology = phx.nn.OperatorTopology.from_graph(graph, site="global")
    global_samples = phx.nn.FunctionSamples(
        values=jnp.asarray([7.0]),
        coordinates=jnp.asarray([[0.5]]),
        quadrature_weights=jnp.asarray([1.0]),
        topology=global_topology,
    )
    global_graph = phx.nn.operator_graph_from_samples(global_samples)
    global_values = phx.nn.gather_operator_graph_entities(
        global_samples,
        global_graph.globals["features"],
    )

    assert edge_topology.entity == "edge"
    assert edge_topology.entity_count == 3
    assert jnp.array_equal(edge_graph.edges["features"], jnp.asarray([2.0, 3.0, 5.0]))
    assert jnp.array_equal(edge_values, edge_samples.values)
    assert global_topology.entity == "global"
    assert global_topology.entity_count == 1
    assert jnp.array_equal(global_graph.globals["features"], jnp.asarray([7.0]))
    assert jnp.array_equal(global_values, global_samples.values)


def test_native_graph_operator_executes_graphir_and_is_jittable_and_differentiable():
    batch = _batch()
    processor = phx.graph.GraphNeuralOperator(
        input_key="features",
        output_key="result",
        edge_weight_key=None,
        normalize=False,
    )
    model = phx.nn.NativeGraphOperator(
        processor,
        in_size="scalar",
        out_size="scalar",
        source_name="u",
        output_key="result",
    )

    eager = model(batch)
    compiled = eqx.filter_jit(lambda item, value: item(value))(model, batch)
    source_values = batch.input("u").values
    input_gradient = jax.grad(
        lambda values: jnp.sum(
            model(
                eqx.tree_at(
                    lambda item: item.inputs["u"].values,
                    batch,
                    values,
                )
            )
            ** 2
        )
    )(source_values)

    assert jnp.array_equal(eager, jnp.asarray([[2.0, 0.0, 1.0], [5.0, 3.0, 4.0]]))
    assert jnp.array_equal(compiled, eager)
    assert jnp.all(jnp.isfinite(input_gradient))


def test_topology_survives_padding_stacking_slicing_and_sampling():
    topology = phx.nn.OperatorTopology.from_graph(_graph())
    samples = phx.nn.FunctionSamples(
        values=jnp.asarray([1.0, 2.0, 3.0]),
        coordinates=jnp.arange(3.0)[:, None],
        topology=topology,
    )
    padded = phx.nn.pad_function_samples(samples, 5)
    selected = phx.nn.take_operator_topology(topology, jnp.asarray([2, 0]))
    stacked = phx.nn.stack_operator_topologies((topology, topology))
    batch = _batch()
    sliced = phx.nn.slice_operator_batch(batch, 1)

    assert padded.topology is not None
    assert padded.topology.sample_shape == (5,)
    assert jnp.array_equal(padded.topology.sample_entities, jnp.asarray([0, 1, 2, -1, -1]))
    assert jnp.array_equal(selected.sample_entities, jnp.asarray([2, 0]))
    assert stacked.case_shape == (2,)
    assert stacked.graph.num_graphs == 2
    assert sliced.case_shape == ()
    assert sliced.input("u").topology is not None
    assert sliced.input("u").topology.case_shape == ()


def test_simplicial_complex_maps_vertices_edges_and_faces_to_native_sites():
    complex_graph = phx.graph.triangle_mesh_to_simplicial_graph(
        jnp.asarray([[0, 1, 2], [0, 2, 3]]),
        num_vertices=4,
    )
    vertices = phx.nn.OperatorTopology.from_simplicial(complex_graph, site="vertex")
    edges = phx.nn.OperatorTopology.from_simplicial(complex_graph, site="edge")
    faces = phx.nn.OperatorTopology.from_simplicial(complex_graph, site="face")

    assert vertices.kind == edges.kind == faces.kind == "simplicial"
    assert vertices.sample_shape == (4,)
    assert edges.sample_shape == (5,)
    assert faces.sample_shape == (2,)
    assert vertices.graph_fingerprint == edges.graph_fingerprint == faces.graph_fingerprint


def test_topology_fingerprint_changes_with_connectivity_not_only_sample_shape():
    first = phx.nn.OperatorTopology.from_graph(_graph())
    changed_graph = _graph().replace(
        senders=jnp.asarray([0, 0, 2]),
        receivers=jnp.asarray([1, 2, 0]),
    )
    second = phx.nn.OperatorTopology.from_graph(changed_graph)

    assert phx.nn.operator_graph_fingerprint(first.graph) != phx.nn.operator_graph_fingerprint(second.graph)
    assert phx.nn.operator_topology_fingerprint(first) != phx.nn.operator_topology_fingerprint(second)
