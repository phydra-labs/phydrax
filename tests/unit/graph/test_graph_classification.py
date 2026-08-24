#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _graphs() -> tuple[phx.graph.GraphIR, phx.graph.GraphIR]:
    graph0 = phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0]]),
        edges=jnp.array([[1.0]]),
        globals=jnp.array([[0.0]]),
        senders=jnp.array([0], dtype=jnp.int32),
        receivers=jnp.array([1], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([1], dtype=jnp.int32),
    )
    graph1 = phx.graph.GraphIR(
        nodes=jnp.array([[2.0], [4.0], [8.0]]),
        edges=jnp.array([[1.0], [1.0]]),
        globals=jnp.array([[1.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )
    return graph0, graph1


@pytest.mark.parametrize(
    ("selection", "component_kind", "values", "expected"),
    [
        (
            phx.domain.Nodes(),
            "nodes",
            ([1, 2], [3, 4, 5]),
            [3, 4, 5, 1, 2],
        ),
        (phx.domain.Edges(), "edges", ([7], [8, 9]), [8, 9, 7]),
        (phx.domain.Globals(), "globals", (10, 11), [11, 10]),
    ],
)
def test_graph_classification_target_preserves_hard_dtype_for_entity_kinds(
    selection, component_kind, values, expected
):
    domain = phx.domain.GraphDatasetDomain(_graphs())
    batch = domain.points_from_indices(
        [1, 0],
        component=selection,
        structure=phx.domain.SampleLayout((("graph",),)),
    )
    target = phx.terms.GraphClassificationTarget(
        domain,
        values,
        component_kind=component_kind,
    )

    observed = jnp.asarray(target(batch).data)
    assert jnp.issubdtype(observed.dtype, jnp.integer)
    assert jnp.array_equal(observed, jnp.asarray(expected))


def test_graph_classification_target_uses_cochain_cell_node_selection():
    graph = phx.graph.GraphIR(
        nodes={
            "x": jnp.array([[0.0], [1.0], [2.0], [3.0]]),
            "cell_dim": jnp.array([0, 1, 1, 2], dtype=jnp.int32),
            "boundary": jnp.array([False, True, False, True]),
        },
        edges=jnp.zeros((0, 1)),
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.array([4], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    domain = phx.domain.GraphDatasetDomain((graph,))
    batch = domain.points_from_indices(
        [0],
        component=phx.domain.CochainCells(1, region="boundary"),
    )
    target = phx.terms.GraphClassificationTarget(domain, ([3, 4, 5, 6],))

    assert jnp.array_equal(jnp.asarray(target(batch).data), jnp.array([4]))


def test_graph_trajectory_classification_is_ragged_nearest_and_dtype_safe():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(),
        jnp.array([2, 3], dtype=jnp.int32),
        dt=1.0,
    )
    values = (
        jnp.array([[0, 1], [2, 3]], dtype=jnp.int16),
        jnp.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=jnp.int16),
    )
    signal = phx.terms.GraphTrajectoryClassificationSignal(domain, values)
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0, 1],
        [0.6, 1.6],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )

    observed = jnp.asarray(signal(batch).data)
    assert observed.dtype == jnp.int16
    assert jnp.array_equal(observed, jnp.array([2, 3, 10, 11, 12]))


def test_graph_trajectory_soft_target_interpolation_must_be_explicit():
    graph = _graphs()[0].replace(
        nodes=jnp.array([[0.0]]),
        edges=jnp.zeros((0, 1)),
        senders=jnp.zeros((0,), dtype=jnp.int32),
        receivers=jnp.zeros((0,), dtype=jnp.int32),
        n_node=jnp.array([1], dtype=jnp.int32),
        n_edge=jnp.array([0], dtype=jnp.int32),
    )
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        (graph,), jnp.array([2], dtype=jnp.int32), dt=1.0
    )
    values = (jnp.array([[[1.0, 0.0]], [[0.0, 1.0]]]),)
    with pytest.raises(ValueError, match="Hard.*nearest"):
        phx.terms.GraphTrajectoryClassificationSignal(
            domain,
            values,
            interpolation="linear",
        )

    signal = phx.terms.GraphTrajectoryClassificationSignal(
        domain,
        values,
        interpolation="linear",
        target_encoding="soft",
    )
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.Interior()}
    )
    batch = domain.points_from_case_time(
        [0],
        [0.25],
        component=component,
        structure=phx.domain.SampleLayout((("graph", "t"),)),
    )
    assert jnp.allclose(jnp.asarray(signal(batch).data), jnp.array([[0.75, 0.25]]))


def test_graph_classification_mean_and_integral_use_graph_measure():
    domain = phx.domain.GraphDatasetDomain(_graphs(), measure="count")
    component = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def logits(node):
        del node
        return 0.0

    schema = phx.ml.TargetSchema("binary", class_labels=("no", "yes"))
    targets = (jnp.array([0, 1]), jnp.array([1, 0, 1]))
    sampling = phx.domain.PointSampling(8)
    mean = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        targets,
        schema,
        sampling=sampling,
        reduction="mean",
    )
    integral = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        targets,
        schema,
        sampling=sampling,
        reduction="integral",
    )

    assert jnp.allclose(mean.loss({"logits": logits}, key=jr.key(0)), jnp.log(2.0))
    assert jnp.allclose(
        integral.loss({"logits": logits}, key=jr.key(0)),
        5.0 * jnp.log(2.0),
    )


def test_graph_component_selection_excludes_invalid_unselected_label():
    domain = phx.domain.GraphDatasetDomain((_graphs()[0],))
    component = domain.component({"graph": phx.domain.BoundaryNodes([1])})

    @domain.Function("graph")
    def logits(node):
        del node
        return jnp.zeros((3,))

    term = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        (jnp.array([99, 0]),),
        phx.ml.TargetSchema("multiclass", class_labels=("a", "b", "c")),
        sampling=phx.domain.PointSampling(2),
    )
    assert jnp.isfinite(term.loss({"logits": logits}, key=jr.key(1)))


def test_graph_hard_multiclass_gathers_without_one_hot(monkeypatch):
    domain = phx.domain.GraphDatasetDomain((_graphs()[0],))
    component = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def logits(node):
        return jnp.array([node[0], 0.0, -node[0]])

    def fail_one_hot(*args, **kwargs):
        del args, kwargs
        raise AssertionError("hard categorical scoring must not allocate one-hot targets")

    monkeypatch.setattr(jax.nn, "one_hot", fail_one_hot)
    term = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        ([0, 2],),
        phx.ml.TargetSchema("multiclass", class_labels=(0, 1, 2)),
        sampling=phx.domain.PointSampling(2),
    )
    assert jnp.isfinite(term.loss({"logits": logits}, key=jr.key(2)))


def test_graph_invalid_active_label_is_infinite_and_masked_label_is_zero():
    domain = phx.domain.GraphDatasetDomain((_graphs()[0],))
    component = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def logits(node):
        del node
        return jnp.zeros((3,))

    schema = phx.ml.TargetSchema("multiclass", class_labels=(0, 1, 2))
    active = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        ([7, 7],),
        schema,
        sampling=phx.domain.PointSampling(2),
        target_mask=([True, True],),
    )
    masked = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        ([7, 7],),
        schema,
        sampling=phx.domain.PointSampling(2),
        target_mask=([False, False],),
    )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        active.loss({"logits": logits}, key=jr.key(3))
    assert jnp.array_equal(
        masked.loss({"logits": logits}, key=jr.key(3)), jnp.asarray(0.0)
    )


def test_graph_soft_focal_multilabel_and_ordinal_objectives():
    graph = _graphs()[0]
    domain = phx.domain.GraphDatasetDomain((graph,))
    component = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def multiclass_logits(node):
        del node
        return jnp.zeros((3,))

    soft = phx.terms.GraphClassificationTerm(
        "soft",
        component,
        (jnp.array([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]]),),
        phx.ml.TargetSchema("multiclass", class_labels=(0, 1, 2)),
        sampling=phx.domain.PointSampling(2),
        objective=phx.ml.ClassificationObjective.soft_cross_entropy(),
    )

    @domain.Function("graph")
    def multilabel_logits(node):
        del node
        return jnp.zeros((2,))

    focal = phx.terms.GraphClassificationTerm(
        "focal",
        component,
        (jnp.array([[True, False], [False, True]]),),
        phx.ml.TargetSchema("multilabel", names=("hot", "cold")),
        sampling=phx.domain.PointSampling(2),
        objective=phx.ml.ClassificationObjective.focal(gamma=1.5, alpha=0.25),
        target_mask=(jnp.array([[True, False], [True, True]]),),
    )

    @domain.Function("graph")
    def ordinal_location(node):
        return node[0] - 0.5

    ordinal = phx.terms.GraphClassificationTerm(
        "ordinal",
        component,
        ([0, 2],),
        phx.ml.TargetSchema("ordinal", class_labels=("low", "middle", "high")),
        sampling=phx.domain.PointSampling(2),
        objective=phx.ml.ClassificationObjective.nll(thresholds=(-1.0, 1.0)),
    )

    assert jnp.isfinite(soft.loss({"soft": multiclass_logits}, key=jr.key(4)))
    assert jnp.isfinite(focal.loss({"focal": multilabel_logits}, key=jr.key(5)))
    assert jnp.isfinite(ordinal.loss({"ordinal": ordinal_location}, key=jr.key(6)))


def test_graph_trajectory_classification_composes_with_graph_physics_residual():
    domain = phx.domain.GraphTrajectoryDatasetDomain(
        _graphs(), jnp.array([2, 3], dtype=jnp.int32), dt=1.0
    )
    component = domain.component(
        {"graph": phx.domain.Nodes(), "t": phx.domain.FixedStart()}
    )
    sampling = phx.domain.PointSampling(
        2, layout=phx.domain.SampleLayout((("graph", "t"),))
    )

    @domain.Function("graph", "t")
    def logits(node, time):
        del node, time
        return 0.0

    @domain.Function("graph", "t")
    def state(node, time):
        del node, time
        return 2.0

    targets = (
        jnp.zeros((2, 2), dtype=jnp.int32),
        jnp.zeros((3, 3), dtype=jnp.int32),
    )
    classification = phx.terms.GraphTrajectoryClassificationTerm(
        "logits",
        component,
        targets,
        phx.ml.TargetSchema("binary", class_labels=(0, 1)),
        sampling=sampling,
    )

    def diffusion(field):
        return domain.GraphModel(phx.graph.GraphDiffusion(), input_fn=field)

    condition = phx.conditions.Residual("state", component, diffusion)
    physics = phx.terms.ResidualPenalty(
        condition,
        phx.integration.per_step(phx.integration.mean_over(component), sampling),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"logits": logits, "state": state},
        terms=(classification, physics),
    )

    assert jnp.allclose(solver.loss(key=jr.key(7)), jnp.log(2.0))


def test_graph_zero_weight_skips_nonfinite_integrand():
    domain = phx.domain.GraphDatasetDomain((_graphs()[0],))
    component = domain.component({"graph": phx.domain.Nodes()})

    @domain.Function("graph")
    def poisoned(node):
        del node
        return jnp.nan

    term = phx.terms.GraphClassificationTerm(
        "logits",
        component,
        (jnp.zeros((2,), dtype=jnp.int32),),
        phx.ml.TargetSchema("binary", class_labels=(0, 1)),
        sampling=phx.domain.PointSampling(2),
        weight=0.0,
    )

    assert term.loss({"logits": poisoned}, key=jr.key(0)) == 0.0
