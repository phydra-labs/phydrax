#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_graph_query_scenario_1() -> None:
    source = jnp.array([[0.0], [1.0], [3.0]])
    target = jnp.array([[0.2], [2.6]])

    bundle = phx.graph.radius_query_graph(
        source,
        target,
        radius=0.5,
        source_features=jnp.array([[1.0], [2.0], [3.0]]),
        weight_kind="hat",
    )
    graph = bundle.graph
    assert graph.senders is not None
    assert graph.receivers is not None

    assert graph.num_nodes == 5
    assert graph.num_edges == 2
    assert jnp.allclose(graph.senders, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.allclose(graph.receivers, jnp.array([3, 4], dtype=jnp.int32))
    assert jnp.allclose(graph.edges["relative"], jnp.array([[0.2], [-0.4]]), atol=1e-7)
    assert jnp.allclose(graph.edges["distance"], jnp.array([[0.2], [0.4]]), atol=1e-7)
    assert jnp.allclose(
        graph.edges["kernel_weight"], jnp.array([[0.6], [0.2]]), atol=1e-7
    )
    assert jnp.allclose(
        graph.nodes["features"][:, 0], jnp.array([1.0, 2.0, 3.0, 0.0, 0.0])
    )
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.9]]),
        jnp.array([[0.1]]),
        radius=0.25,
        periodic_box=1.0,
        weight_kind=None,
    )

    assert bundle.graph.num_edges == 1
    assert jnp.allclose(bundle.graph.edges["relative"], jnp.array([[0.2]]), atol=1e-7)
    assert jnp.allclose(bundle.graph.edges["distance"], jnp.array([[0.2]]), atol=1e-7)
    source = jnp.array([[0.0], [2.0], [5.0]])
    target = jnp.array([[1.0]])
    bundle = phx.graph.knn_query_graph(source, target, k=2, weight_kind=None)
    rebuilt = phx.graph.query_graph_from_edges(
        source,
        target + 1.0,
        bundle.graph.edges["source_index"],
        bundle.graph.edges["target_index"],
        weight_kind=None,
    )
    assert bundle.graph.senders is not None
    assert bundle.graph.receivers is not None

    assert jnp.allclose(bundle.graph.senders, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(bundle.graph.receivers, jnp.array([3, 3], dtype=jnp.int32))
    assert jnp.allclose(rebuilt.graph.edges["relative"], jnp.array([[2.0], [0.0]]))
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    domain = phx.domain.GraphDomain(bundle.graph, measure="count")
    structure = phx.domain.SampleLayout((("graph",),))
    sources = domain.component({"graph": bundle.source_nodes_component()})
    targets = domain.component({"graph": bundle.target_nodes_component()})
    query_edges = domain.component({"graph": bundle.query_edges_component()})

    source_batch = sources.sample(phx.domain.PointSampling(2, layout=structure))
    target_batch = targets.sample(phx.domain.PointSampling(1, layout=structure))
    edge_batch = query_edges.sample(phx.domain.PointSampling(2, layout=structure))

    assert jnp.allclose(
        source_batch["graph"]["features"].data[:, 0], jnp.array([1.0, 3.0])
    )
    assert jnp.allclose(target_batch["graph"]["features"].data[:, 0], jnp.array([0.0]))
    assert jnp.allclose(edge_batch["graph"]["distance"].data[:, 0], jnp.array([0.5, 0.5]))
    # ty: ignore[unresolved-attribute]
    assert targets.mass.value == 1.0
    neighborhood = phx.graph.query_neighbors(
        jnp.array([[0.25], [0.75]]),
        jnp.array([[0.0]]),
        max_neighbors=2,
        periodic_lengths=(1.0,),
    )

    assert jnp.array_equal(neighborhood.indices[0, 0], jnp.array([0, 1]))
    assert jnp.allclose(neighborhood.distance[0, 0], jnp.array([0.25, 0.25]))
    assert jnp.allclose(
        neighborhood.relative[0, 0, :, 0],
        jnp.array([-0.25, 0.25]),
    )


def test_graph_query_scenario_2() -> None:
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    out = phx.graph.GraphNeuralOperator(
        input_key="features",
        output_key="gno",
        edge_weight_key=None,
        normalize=False,
        target_node_type=bundle.target_type,
    )(bundle.graph)

    assert jnp.allclose(out.nodes["gno"][:, 0], jnp.array([0.0, 0.0, 4.0]))
    source = jnp.array(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
        ]
    )
    target = jnp.array([[[0.2], [1.8]], [[10.2], [11.8]]])
    source_mask = jnp.array([[True, True, True], [True, True, False]])
    target_mask = jnp.array([[True, True], [True, False]])
    query = phx.graph.batched_knn_query_graph(
        source,
        target,
        k=2,
        source_mask=source_mask,
        target_mask=target_mask,
        source_features=jnp.arange(6.0).reshape((2, 3, 1)),
        source_measure=jnp.ones((2, 3)),
    )
    graph = query.graph
    assert graph.senders is not None
    assert graph.receivers is not None
    assert graph.edge_mask is not None
    assert graph.node_mask is not None

    assert jnp.array_equal(graph.n_node, jnp.array([5, 5], dtype=jnp.int32))
    assert jnp.array_equal(graph.n_edge, jnp.array([4, 4], dtype=jnp.int32))
    assert jnp.all(graph.senders[:4] < 5)
    assert jnp.all(graph.receivers[:4] < 5)
    assert jnp.all(graph.senders[4:] >= 5)
    assert jnp.all(graph.receivers[4:] >= 5)
    assert jnp.array_equal(
        graph.edge_mask,
        jnp.array([True, True, True, True, True, True, False, False]),
    )
    assert jnp.array_equal(
        graph.node_mask,
        jnp.array([True, True, True, True, True, True, True, False, True, False]),
    )
    graph = phx.graph.batched_knn_graph(
        jnp.array([[[0.0], [1.0], [3.0]], [[10.0], [11.0], [13.0]]]),
        k=1,
    )
    assert graph.senders is not None
    assert graph.receivers is not None

    assert graph.senders.shape == (6,)
    assert jnp.all(graph.senders != graph.receivers)
    assert jnp.all(graph.senders[:3] < 3)
    assert jnp.all(graph.senders[3:] >= 3)
    points = jnp.array([[0.0], [1.0], [3.0], [6.0]])
    neighborhood = phx.graph.query_neighbors(
        points,
        points,
        max_neighbors=1,
        exclude_self=True,
        target_chunk_size=2,
    )
    assert jnp.array_equal(neighborhood.indices[0, :, 0], jnp.array([1, 0, 1, 2]))


def test_graph_neural_operator_wraps_as_graph_model_on_query_targets() -> None:
    bundle = phx.graph.radius_query_graph(
        jnp.array([[0.0], [1.0]]),
        jnp.array([[0.5]]),
        radius=1.0,
        source_features=jnp.array([[1.0], [3.0]]),
        weight_kind=None,
    )
    domain = phx.domain.GraphDomain(bundle.graph)
    targets = domain.component({"graph": bundle.target_nodes_component()})
    batch = targets.sample(
        phx.domain.PointSampling(1, layout=phx.domain.SampleLayout((("graph",),)))
    )

    @domain.Function("graph")
    def u(node: Any) -> Any:
        return node.get("features")[0]

    model = domain.GraphModel(
        phx.graph.GraphNeuralOperator(
            input_key="u",
            output_key="gno",
            edge_weight_key=None,
            normalize=False,
            target_node_type=bundle.target_type,
        ),
        input_fn=u,
        input_key="u",
        output_key="gno",
    )

    assert jnp.allclose(jnp.asarray(model(batch).data), jnp.array([4.0]))


def test_batched_query_graph_is_jittable_and_differentiable() -> None:
    source = jnp.array([[[0.0], [0.5], [1.0]]])
    target = jnp.array([[[0.2], [0.8]]])

    def distances(source_points: Any, target_points: Any) -> Any:
        graph = phx.graph.batched_knn_query_graph(
            source_points,
            target_points,
            k=2,
            source_measure=jnp.full((1, 3), 1.0 / 3.0),
            validate=False,
        ).graph
        return graph.edges["distance"]

    result = jax.jit(distances)(source, target)
    gradient = jax.grad(lambda points: jnp.sum(distances(source, points)))(target)

    assert result.shape == (4, 1)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.all(jnp.isfinite(gradient))


def test_query_neighbors_support_morton_execution_and_geometry_gradients() -> None:
    plan = phx.discretization.spatial.MortonNeighborQueryPlan(
        phx.discretization.spatial.MortonSourcePlan(
            phx.discretization.spatial.MortonAddressPlan(
                (0.0,), (1.0,), 16, periodic_axes=(True,)
            ),
            2,
        ),
        1,
        2,
    )

    def distances(target: Any) -> Any:
        return phx.graph.query_neighbors(
            jnp.array([[0.25], [0.75]]),
            target,
            periodic_lengths=(1.0,),
            plan=plan,
        )

    neighborhood = distances(jnp.array([[0.0]]))
    gradient = jax.grad(lambda target: jnp.sum(distances(target).distance))(
        jnp.array([[0.1]])
    )

    assert neighborhood.evidence.backend == "morton"
    assert bool(neighborhood.evidence.successful)
    assert jnp.array_equal(neighborhood.indices[0, 0], jnp.array([0, 1]))
    assert jnp.allclose(neighborhood.distance[0, 0], jnp.array([0.25, 0.25]))
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize("radius", [None, 0.5], ids=["knn", "inclusive-radius"])
@pytest.mark.parametrize("source_tile", [1, 3, 8], ids=["singleton", "partial", "whole"])
def test_blocked_neighbors_match_independent_masked_periodic_reference(
    radius: float | None,
    source_tile: int,
) -> None:
    source = np.asarray([[0.0], [0.5], [0.25], [0.75], [0.0]], dtype=np.float64)
    target = source.copy()
    source_mask = np.asarray([True, True, False, True, True], dtype=np.bool_)
    target_mask = np.asarray([True, False, True, True, True], dtype=np.bool_)
    relative = target[:, None, :] - source[None, :, :]
    relative -= np.round(relative)
    squared = np.sum(relative * relative, axis=-1)
    valid = source_mask[None, :] & target_mask[:, None] & ~np.eye(5, dtype=np.bool_)
    if radius is not None:
        valid &= squared <= radius * radius
    indices = np.argsort(np.where(valid, squared, np.inf), axis=-1, kind="stable")[:, :3]
    expected_valid = np.take_along_axis(valid, indices, axis=1)
    expected_squared = np.where(
        expected_valid,
        np.take_along_axis(squared, indices, axis=1),
        0,
    )
    result = phx.graph.query_neighbors(
        source,
        target,
        source_mask=source_mask,
        target_mask=target_mask,
        max_neighbors=3,
        radius=radius,
        periodic_lengths=(1.0,),
        exclude_self=True,
        source_chunk_size=source_tile,
        target_chunk_size=2,
    )
    np.testing.assert_array_equal(result.indices[0], indices)
    np.testing.assert_array_equal(result.mask[0], expected_valid)
    np.testing.assert_array_equal(result.distance_squared[0], expected_squared)
    np.testing.assert_array_equal(result.count[0], expected_valid.sum(axis=-1))
    np.testing.assert_allclose(
        result.relative[0],
        np.where(
            expected_valid[..., None],
            np.take_along_axis(
                relative,
                indices[..., None],
                axis=1,
            ),
            0,
        ),
    )
    assert result.evidence.backend == "blocked"
    assert bool(result.evidence.successful & result.evidence.complete)


@pytest.mark.parametrize(
    "kind", ["radius", "knn"], ids=["inclusive-radius", "stable-knn"]
)
def test_host_query_edges_match_brute_force_across_source_tiles(kind: str) -> None:
    source = np.linspace(-1.0, 1.0, 261, dtype=np.float64)[:, None]
    source[0] = source[260] = 0.5
    target = np.asarray([[0.0], [0.5], [-0.5]], dtype=np.float64)
    distance = np.linalg.norm(target[:, None, :] - source[None, :, :], axis=-1)
    if kind == "radius":
        target_ids, source_ids = np.nonzero(distance <= 0.5)
        bundle = phx.graph.radius_query_graph(
            source, target, radius=0.5, weight_kind=None
        )
    else:
        source_ids = np.argsort(distance, axis=-1, kind="stable")[:, :4].reshape(-1)
        target_ids = np.repeat(np.arange(3, dtype=np.int32), 4)
        bundle = phx.graph.knn_query_graph(source, target, k=4)
    np.testing.assert_array_equal(bundle.graph.edges["source_index"], source_ids)
    np.testing.assert_array_equal(bundle.graph.edges["target_index"], target_ids)


def test_graph_query_morton_overflow_refuses_neighbors() -> None:
    source = jnp.full((8, 1), 0.25, dtype=jnp.float64)
    plan = phx.discretization.spatial.MortonNeighborQueryPlan(
        phx.discretization.spatial.MortonSourcePlan(
            phx.discretization.spatial.MortonAddressPlan((0.0,), (1.0,), 8), 8
        ),
        1,
        1,
        maximum_candidates=2,
    )
    result = phx.graph.query_neighbors(source, jnp.asarray([[0.25]]), plan=plan)
    assert not bool(result.evidence.complete)
    assert not bool(result.evidence.successful)
    assert not bool(jnp.any(result.mask))
    np.testing.assert_array_equal(result.count, 0)


@pytest.mark.parametrize("axis", ["source", "target"])
def test_graph_query_refuses_nonpositive_chunk_size(axis: str) -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        if axis == "source":
            phx.graph.query_neighbors([[0.0]], [[0.0]], source_chunk_size=0)
        else:
            phx.graph.query_neighbors([[0.0]], [[0.0]], target_chunk_size=0)


@pytest.mark.parametrize("axis", ["source", "target"])
@pytest.mark.parametrize(
    "tile", [True, 1.5, "2"], ids=["boolean", "fractional", "string"]
)
def test_graph_query_refuses_noninteger_chunk_sizes(axis: str, tile: Any) -> None:
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        if axis == "source":
            phx.graph.query_neighbors([[0.0]], [[0.0]], source_chunk_size=tile)
        else:
            phx.graph.query_neighbors([[0.0]], [[0.0]], target_chunk_size=tile)


def test_blocked_neighbors_preserve_nonfinite_failure_routes() -> None:
    source = np.asarray([[np.nan], [1.0], [0.0], [2.0], [np.inf]], dtype=np.float64)
    expected = np.argsort(np.square(source[:, 0]), kind="stable")
    result = phx.graph.query_neighbors(
        source,
        [[0.0]],
        max_neighbors=5,
        source_chunk_size=3,
    )
    np.testing.assert_array_equal(result.indices[0, 0], expected)
    np.testing.assert_array_equal(result.mask, True)
    np.testing.assert_array_equal(result.count, 5)
    assert not bool(result.evidence.finite)
    assert not bool(result.evidence.successful)


def test_query_budget_boundary_preserves_exact_neighborhoods() -> None:
    source = np.asarray([[0.0], [1.0], [-1.0], [3.0]], dtype=np.float64)
    target = np.asarray([[0.5], [2.0], [-0.5]], dtype=np.float64)
    # Four 29-byte pairs plus two 24-byte merge entries per target.
    full_budget = 3 * (4 * 29 + 2 * 24)
    full = phx.graph.query_neighbors(
        source, target, max_neighbors=2, query_workspace_bytes=full_budget
    )
    tiled = phx.graph.query_neighbors(
        source, target, max_neighbors=2, query_workspace_bytes=full_budget - 1
    )
    explicit = phx.graph.query_neighbors(
        source,
        target,
        max_neighbors=2,
        source_chunk_size=4,
        target_chunk_size=3,
        query_workspace_bytes=1,
    )
    assert full.evidence.backend == explicit.evidence.backend == "dense"
    assert tiled.evidence.backend == "blocked"
    assert full.evidence.source_chunk_size == 4
    assert full.evidence.target_chunk_size == 3
    assert tiled.evidence.target_chunk_size == 2
    assert tiled.evidence.query_workspace_bytes == full_budget - 1
    expected = np.argsort((target[:, None, 0] - source[None, :, 0]) ** 2, kind="stable")[
        :, :2
    ]
    np.testing.assert_array_equal(full.indices[0], expected)
    for result in (tiled, explicit):
        np.testing.assert_array_equal(result.indices, full.indices)
        np.testing.assert_array_equal(result.relative, full.relative)
        np.testing.assert_array_equal(result.distance_squared, full.distance_squared)
        np.testing.assert_array_equal(result.mask, full.mask)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, None])
def test_query_refuses_invalid_workspace_budget(budget: Any) -> None:
    with pytest.raises((TypeError, ValueError), match="query_workspace_bytes"):
        phx.graph.query_neighbors([[0.0]], [[0.0]], query_workspace_bytes=budget)


def test_batched_graph_budget_routes_preserve_edges() -> None:
    points = np.asarray([[0.0], [1.0], [3.0], [6.0]], dtype=np.float64)
    for builder in (
        lambda budget: phx.graph.batched_knn_graph(
            points, k=2, query_workspace_bytes=budget
        ),
        lambda budget: (
            phx.graph.batched_knn_query_graph(
                points, points, k=2, query_workspace_bytes=budget
            ).graph
        ),
    ):
        full = builder(4096)
        tiled = builder(64)
        np.testing.assert_array_equal(full.senders, tiled.senders)
        np.testing.assert_array_equal(full.receivers, tiled.receivers)
        np.testing.assert_array_equal(full.edge_mask, tiled.edge_mask)


def test_query_budget_preserves_empty_case_refusal() -> None:
    with pytest.raises(ValueError, match="case dimensions must be positive"):
        phx.graph.query_neighbors(
            jnp.zeros((0, 3, 2), dtype=jnp.float64),
            jnp.zeros((0, 2, 2), dtype=jnp.float64),
            max_neighbors=1,
        )


def _spatial_plan(family: str, periodic: bool = False) -> Any:
    spatial = phx.discretization.spatial
    if family == "bvh":
        source = spatial.BVHSourcePlan(
            1, 4, leaf_size=2, periodic_lengths=(1.0,) if periodic else None
        )
        return spatial.BVHNeighborQueryPlan(source, 4, 2)
    address = spatial.MortonAddressPlan((0.0,), (1.0,), 3, periodic_axes=(periodic,))
    return spatial.MortonNeighborQueryPlan(
        spatial.MortonSourcePlan(address, 4), 4, 2, maximum_candidates=4
    )


@pytest.mark.parametrize("family", ["morton", "bvh"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("exclude_self", [False, True])
def test_planned_prepared_graph_parity(
    family: str, periodic: bool, exclude_self: bool
) -> None:
    points = jnp.asarray([[0.125], [0.375], [0.375], [0.875]], dtype=jnp.float64)
    targets = jnp.stack((points, points + 0.015625))
    plan = _spatial_plan(family, periodic)
    prepared = plan.source_plan.prepare(points)
    periodic_lengths = (1.0,) if periodic else None

    def query(source: Any, query_plan: Any = None) -> phx.graph.QueryNeighborhood:
        return phx.graph.query_neighbors(
            source,
            targets,
            max_neighbors=2,
            radius=0.25,
            exclude_self=exclude_self,
            periodic_lengths=periodic_lengths,
            plan=query_plan,
        )

    dense = query(points)
    raw = query(points, plan)
    reused = query(prepared, plan)
    stacked = jax.vmap(plan.source_plan.prepare)(jnp.stack((points, points)))
    stacked_result = query(stacked, plan)
    for result in (raw, reused, stacked_result):
        assert result.evidence.backend == family
        assert bool(result.evidence.successful)
        np.testing.assert_array_equal(
            result.indices[result.mask], dense.indices[dense.mask]
        )
        np.testing.assert_array_equal(result.mask, dense.mask)
        np.testing.assert_allclose(result.relative, dense.relative)
        np.testing.assert_allclose(result.distance_squared, dense.distance_squared)
    for builder in (
        lambda source: (
            phx.graph.batched_knn_query_graph(source, targets, k=2, plan=plan).graph
        ),
        lambda source: phx.graph.batched_knn_graph(source, k=2, plan=plan),
    ):
        expected = builder(points)
        actual = builder(prepared)
        np.testing.assert_array_equal(actual.senders, expected.senders)
        np.testing.assert_array_equal(actual.edge_mask, expected.edge_mask)


@pytest.mark.parametrize("family", ["morton", "bvh"])
def test_prepared_graph_refusals(family: str) -> None:
    plan = _spatial_plan(family)
    points = jnp.full((4, 1), 0.3, dtype=jnp.float64)
    prepared = plan.source_plan.prepare(points)
    with pytest.raises(ValueError, match="explicit"):
        phx.graph.query_neighbors(prepared, points)
    with pytest.raises(ValueError, match="activity"):
        phx.graph.query_neighbors(prepared, points, plan=plan, source_mask=[True] * 4)
    with pytest.raises(TypeError, match="float64"):
        phx.graph.query_neighbors(
            plan.source_plan.prepare(points.astype(jnp.float32)), points, plan=plan
        )
    with pytest.raises(ValueError, match="logical-index"):
        phx.graph.query_neighbors(
            plan.source_plan.prepare(points, stable_ids=jnp.arange(4)),
            points,
            plan=plan,
        )
    with pytest.raises(TypeError, match="requires Prepared"):
        phx.graph.query_neighbors(
            prepared,
            points,
            plan=_spatial_plan("bvh" if family == "morton" else "morton"),
        )
    with pytest.raises(ValueError, match="source plan"):
        phx.graph.query_neighbors(prepared, points, plan=_spatial_plan(family, True))
    with pytest.raises(ValueError, match="periodic_lengths"):
        phx.graph.query_neighbors(prepared, points, plan=plan, periodic_lengths=(1.0,))


@pytest.mark.parametrize("family", ["morton", "bvh"])
def test_prepared_graph_live_geometry_gradients(family: str) -> None:
    plan = _spatial_plan(family)
    points = jnp.asarray([[0.1], [0.3], [0.6], [0.9]], dtype=jnp.float64)
    targets = points + 0.01

    def loss(source: Any, target: Any, planned: bool) -> Any:
        result = phx.graph.query_neighbors(
            plan.source_plan.prepare(source) if planned else source,
            target,
            max_neighbors=2,
            plan=plan if planned else None,
        )
        return jnp.sum(result.relative**2) + jnp.sum(result.distance)

    dense = jax.grad(loss, argnums=(0, 1))(points, targets, False)
    prepared = jax.grad(loss, argnums=(0, 1))(points, targets, True)
    for actual, expected in zip(prepared, dense, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_stacked_bvh_graph_shares_capacity_fixed_topology() -> None:
    import equinox as eqx

    plan = _spatial_plan("bvh")
    points = jnp.asarray(
        [[[0.125], [0.375], [0.625], [0.875]], [[0.25], [0.375], [0.625], [0.875]]],
        dtype=jnp.float64,
    )
    stacked = jax.vmap(plan.source_plan.prepare)(points)
    shared = eqx.tree_at(
        lambda sources: (
            sources.tree.left,
            sources.tree.right,
            sources.tree.leaf_id,
            sources.tree.leaf_node,
        ),
        stacked,
        tuple(
            value[0]
            for value in (
                stacked.tree.left,
                stacked.tree.right,
                stacked.tree.leaf_id,
                stacked.tree.leaf_node,
            )
        ),
    )
    result = phx.graph.query_neighbors(shared, points, plan=plan)
    raw = phx.graph.query_neighbors(points, points, plan=plan)
    np.testing.assert_array_equal(result.indices, raw.indices)
    np.testing.assert_array_equal(result.mask, raw.mask)


@pytest.mark.parametrize("family", ["morton", "bvh"])
def test_prepared_graph_squared_distance_gradient_at_coincidence(family: str) -> None:
    plan = _spatial_plan(family)
    points = jnp.full((4, 1), 0.5, dtype=jnp.float64)

    def loss(source: Any, planned: bool) -> Any:
        result = phx.graph.query_neighbors(
            plan.source_plan.prepare(source) if planned else source,
            points,
            max_neighbors=2,
            plan=plan if planned else None,
        )
        return jnp.sum(result.distance_squared)

    actual = jax.grad(loss)(points, True)
    expected = jax.grad(loss)(points, False)
    np.testing.assert_array_equal(actual, expected)
    assert bool(jnp.all(jnp.isfinite(actual)))
