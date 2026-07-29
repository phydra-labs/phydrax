#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _points_batch():
    structure = phx.domain.ProductStructure((("case",), ("x",))).canonicalize(
        ("case", "x")
    )
    case_axis = structure.axis_for("case")
    sample_axis = structure.axis_for("x")
    coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 4)[None, :, None],
        (2, 4, 1),
    )
    values = jnp.arange(8.0).reshape((2, 4))
    batch = phx.domain.PointsBatch(
        {
            "u": cx.Field(values, dims=(case_axis, sample_axis)),
            "x": cx.Field(coordinates, dims=(case_axis, sample_axis, None)),
        },
        structure,
    )
    return batch, case_axis, sample_axis


def test_points_domain_view_round_trips_named_prediction_fields():
    batch, case_axis, sample_axis = _points_batch()
    view = phx.nn.operator_domain_view_from_points(
        batch,
        inputs={"source": "u"},
        queries={"state": "x"},
        input_coordinates={"source": "x"},
        quadrature={"state": jnp.full((2, 4), 0.25)},
        case_axes=(case_axis,),
    )
    values = 2.0 * view.batch.input("source").values
    prediction = phx.nn.OperatorPrediction.from_field(
        "solution",
        values,
        "state",
        view.batch.query("state"),
        spec=phx.nn.OperatorOutputSpec("scalar"),
        case_axes=view.batch.case_axes,
        case_shape=view.batch.case_shape,
    )

    restored = view.restore(prediction)

    assert view.kind == "points"
    assert view.batch.case_shape == (2,)
    assert view.batch.query("state").sample_shape == (4,)
    assert restored["solution"].dims == (case_axis, sample_axis)
    assert jnp.array_equal(restored["solution"].data, values)


def test_points_domain_model_dispatches_shared_query_geometry_end_to_end():
    data = phx.domain.DatasetDomain(jnp.ones((2, 4)), label="data")
    domain = data @ phx.domain.Interval1d(0.0, 1.0)
    sampled = domain.component().sample(
        (2, 4),
        structure=phx.domain.ProductStructure((("data",), ("x",))),
        key=jr.key(2),
    )
    latent = 4
    model = phx.nn.DeepONet(
        branch=phx.nn.MLP(
            in_size=4,
            out_size=latent,
            width_size=6,
            depth=1,
            key=jr.key(3),
        ),
        trunk=phx.nn.MLP(
            in_size="scalar",
            out_size=latent,
            width_size=6,
            depth=1,
            key=jr.key(4),
        ),
        coord_dim=1,
        latent_size=latent,
        out_size="scalar",
        in_size=4,
    )
    output = domain.Model("data", "x")(model)(sampled)

    assert output.dims == tuple(sampled.structure.axis_names)
    assert output.data.shape == (2, 4)
    assert jnp.all(jnp.isfinite(output.data))


def test_coord_separable_domain_view_preserves_axes_and_restores_output():
    nx = 8
    data = phx.domain.DatasetDomain(jnp.ones((3, nx)), label="data")
    geometry = phx.domain.Square(center=(0.0, 0.0), side=1.0)
    domain = data @ geometry
    sampled = domain.component().sample_coord_separable(
        {"x": (phx.domain.FourierAxisSpec(nx), phx.domain.FourierAxisSpec(nx))},
        num_points=2,
        dense_structure=phx.domain.ProductStructure((("data",),)),
        key=jr.key(0),
    )
    view = phx.nn.operator_domain_view_from_coord_separable(
        sampled,
        inputs={"source": "data"},
        queries={"query": ("x",)},
    )
    output = jnp.ones(view.batch.case_shape + view.batch.query("query").sample_shape)
    prediction = phx.nn.OperatorPrediction.from_field(
        "solution",
        output,
        "query",
        view.batch.query("query"),
        spec=phx.nn.OperatorOutputSpec("scalar"),
        case_axes=view.batch.case_axes,
        case_shape=view.batch.case_shape,
    )

    restored = view.restore_field(prediction, "solution")

    assert view.kind == "coord_separable"
    assert view.batch.query("query").axis_names == sampled.coord_axes_by_label["x"]
    assert restored.dims == view.batch.case_axes + sampled.coord_axes_by_label["x"]
    assert restored.data.shape == output.shape


def _graph(node_count):
    positions = jnp.arange(float(node_count))[:, None]
    return phx.graph.GraphIR(
        nodes={"positions": positions},
        edges=None,
        senders=None,
        receivers=None,
        n_node=jnp.asarray([node_count]),
        n_edge=jnp.asarray([0]),
    )


def test_graph_domain_view_pads_graph_cases_and_restores_ragged_entity_axis():
    domain = phx.domain.GraphDatasetDomain((_graph(2), _graph(3)), label="graph")
    sampled = domain.points_from_indices(
        jnp.asarray([0, 1]),
        component=phx.domain.Nodes(),
    )
    view = phx.nn.operator_domain_view_from_graph(
        sampled,
        inputs={"source": "graph"},
    )
    output = jnp.arange(6.0).reshape((2, 3))
    restored = view.layouts["query"].restore(output)

    assert view.kind == "graph"
    assert view.batch.case_shape == (2,)
    assert view.batch.query("query").sample_shape == (3,)
    assert jnp.array_equal(
        view.batch.query("query").mask,
        jnp.asarray([[True, True, False], [True, True, True]]),
    )
    assert restored.dims == (sampled.structure.axis_for("graph"),)
    assert jnp.array_equal(restored.data, jnp.asarray([0.0, 1.0, 3.0, 4.0, 5.0]))

    model = phx.nn.NativeGraphOperator(
        lambda graph: graph,
        in_size="scalar",
        out_size="scalar",
        source_name="graph",
        output_key="features",
    )
    function = domain.Model("graph")(model)
    evaluated = function(sampled)

    assert evaluated.dims == restored.dims
    assert jnp.allclose(
        jnp.sum(view.batch.query("query").weights(), axis=-1),
        jnp.ones((2,)),
    )
    assert jnp.array_equal(evaluated.data, jnp.asarray([0.0, 1.0, 0.0, 1.0, 2.0]))


def test_simplicial_domain_view_retains_cell_site_and_graph_node_entity():
    complex_graph = phx.graph.triangle_mesh_to_simplicial_graph(
        jnp.asarray([[0, 1, 2], [0, 2, 3]]),
        num_vertices=4,
    )
    domain = phx.domain.GraphDatasetDomain((complex_graph.graph,), label="cell")
    sampled = domain.points_from_indices(
        jnp.asarray([0]),
        component=phx.domain.NodeSet(complex_graph.face_cells),
    )
    view = phx.nn.operator_domain_view_from_simplicial(
        sampled,
        inputs={"source": "cell"},
        site="face",
    )
    topology = view.batch.query("query").topology

    assert view.kind == "simplicial"
    assert topology is not None
    assert topology.kind == "simplicial"
    assert topology.site == "face"
    assert topology.entity == "node"
    assert jnp.array_equal(topology.sample_entities, complex_graph.face_cells)


def test_ragged_series_domain_view_preserves_masks_weights_and_model_dispatch():
    domain = phx.domain.RaggedSeriesDatasetDomain(
        jnp.arange(12.0).reshape((3, 4, 1)),
        jnp.asarray([2, 4, 3]),
        static=jnp.asarray([[1.0], [2.0], [3.0]]),
        label="series",
    )
    sampled = domain.points_from_indices(jnp.asarray([0, 1, 2]))
    view = phx.nn.operator_domain_view_from_ragged_series(sampled, "series")
    latent = 4
    branch = phx.nn.IntegralBranchEncoder(
        feature_model=phx.nn.MLP(
            in_size=3,
            out_size=latent,
            width_size=6,
            depth=1,
            key=jr.key(5),
        ),
        latent_size=latent,
        value_channels=2,
        coord_dim=1,
    )
    model = phx.nn.DeepONet(
        branch=branch,
        trunk=phx.nn.MLP(
            in_size="scalar",
            out_size=latent,
            width_size=6,
            depth=1,
            key=jr.key(6),
        ),
        coord_dim=1,
        latent_size=latent,
        source_key="series",
    )
    evaluated = domain.Model("series")(model)(sampled)

    assert view.kind == "ragged_series"
    assert view.batch.input("series").values.shape == (3, 4, 2)
    assert jnp.array_equal(
        view.batch.query("query").mask,
        jnp.asarray(
            [
                [True, True, False, False],
                [True, True, True, True],
                [True, True, True, False],
            ]
        ),
    )
    assert evaluated.dims == (sampled.structure.axis_for("series"), None)
    assert evaluated.data.shape == (3, 4)
    assert jnp.all(evaluated.data[~view.batch.query("query").mask] == 0.0)


def test_trajectory_domain_views_group_cases_and_restore_observation_order():
    regular = phx.domain.TrajectoryDatasetDomain(
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.asarray([3, 3]),
        dt=0.5,
    )
    irregular = phx.domain.IrregularTrajectoryDatasetDomain(
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.asarray([[0.0, 0.4, 1.0], [0.0, 0.7, 1.5]]),
        jnp.asarray([3, 3]),
    )
    latent = 3
    model = phx.nn.DeepONet(
        branch=phx.nn.MLP(
            in_size=2,
            out_size=latent,
            width_size=5,
            depth=1,
            key=jr.key(7),
        ),
        trunk=phx.nn.MLP(
            in_size="scalar",
            out_size=latent,
            width_size=5,
            depth=1,
            key=jr.key(8),
        ),
        coord_dim=1,
        latent_size=latent,
        in_size=2,
    )
    case_indices = jnp.asarray([1, 0, 1])
    time_indices = jnp.asarray([0, 1, 2])
    for domain in (regular, irregular):
        times = domain.observation_times(case_indices, time_indices)
        sampled = domain.points_from_case_time(
            case_indices,
            times,
            time_indices=time_indices,
        )
        view = phx.nn.operator_domain_view_from_trajectory(
            sampled,
            inputs={"data": "data"},
            query_label="t",
        )
        restored = view.layouts["query"].restore(
            jnp.asarray([[10.0, 11.0], [20.0, 21.0]])
        )
        evaluated = domain.Model("data", "t")(model)(sampled)

        assert view.kind == "trajectory"
        assert view.batch.case_shape == (2,)
        assert jnp.array_equal(
            view.batch.query("query").mask,
            jnp.asarray([[True, False], [True, True]]),
        )
        assert jnp.array_equal(restored.data, jnp.asarray([20.0, 10.0, 21.0]))
        assert evaluated.dims == restored.dims
        assert evaluated.data.shape == (3,)
        assert jnp.all(jnp.isfinite(evaluated.data))


def test_graph_trajectory_domain_view_includes_time_in_query_geometry():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        (_graph(2), _graph(3)),
        jnp.asarray([3, 4]),
        dt=0.5,
    )
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    sampled = domain.points_from_case_time(
        [0, 1],
        [0.5, 1.0],
        component=component,
        structure=phx.domain.ProductStructure((("graph", "t"),)),
    )
    view = phx.nn.operator_domain_view_from_graph(
        sampled,
        inputs={"graph": "graph"},
        query_labels=("t",),
    )
    model = phx.nn.NativeGraphOperator(
        lambda graph: graph,
        in_size="scalar",
        out_size="scalar",
        source_name="graph",
        output_key="features",
    )
    evaluated = domain.Model("graph", "t")(model)(sampled)

    coordinates = view.batch.query("query").coordinates
    assert coordinates.shape == (2, 3, 2)
    assert jnp.allclose(coordinates[0, :2, 1], 0.5)
    assert jnp.allclose(coordinates[1, :, 1], 1.0)
    assert evaluated.data.shape == (5,)
    assert jnp.array_equal(evaluated.data, jnp.asarray([0.0, 1.0, 0.0, 1.0, 2.0]))


def test_operator_domain_preflight_rejects_unsupported_geometry_before_execution():
    batch, case_axis, _ = _points_batch()
    view = phx.nn.operator_domain_view_from_points(
        batch,
        inputs={"u": "u"},
        queries={"query": "x"},
        case_axes=(case_axis,),
    )
    model = phx.nn.FNO(
        in_channels="scalar",
        out_channels="scalar",
        width=4,
        depth=1,
        n_modes=(2,),
        key=jr.key(9),
    )

    with pytest.raises(ValueError, match="UNSUPPORTED_GEOMETRY"):
        model.predict(view.batch)
