#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.exterior import FormTwist
from tests._support.cochain import reoriented_lowering, triangle_cochain_lowering


def _square_complex(*, shift: Any = 0.0) -> Any:
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vertices = vertices + float(shift)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return triangle_cochain_lowering(vertices, faces)


def _square_sparse_complex() -> tuple[Any, tuple[np.ndarray, ...]]:
    base = _square_complex().discretization
    matrices = tuple(
        np.diag(np.asarray(hodge.weights))
        + 0.07 * np.outer(np.arange(1, count + 1), np.arange(1, count + 1))
        for count, hodge in zip(base.cell_counts, base.hodges, strict=True)
    )
    hodges = []
    for matrix in matrices:
        rows, columns = np.triu_indices(matrix.shape[0])
        hodges.append(
            phx.discretization.SparseHodge(
                rows, columns, matrix[rows, columns], matrix.shape[0]
            )
        )
    owner = phx.discretization.CochainDiscretization(
        base.topology,
        tuple(hodges),
        coordinates=base.coordinates,
        boundary_masks=base.boundary_masks,
        primal_measures=base.primal_measures,
        dual_measures=base.dual_measures,
    )
    return phx.graph.CochainComplexIR(owner), matrices


def _annulus_complex(*, harmonics: Any = False) -> Any:
    outer = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [(index, (index + 1) % 4, 4 + (index + 1) % 4) for index in range(4)]
        + [(index, 4 + (index + 1) % 4, 4 + index) for index in range(4)],
        dtype=np.int32,
    )
    complex_ir = triangle_cochain_lowering(vertices, faces)
    if harmonics:
        realization = complex_ir.discretization
        complex_ir = phx.graph.CochainComplexIR(
            realization,
            harmonic=tuple(
                phx.exterior.validate_harmonic_cohomology(realization, degree)[0]
                for degree in range(realization.dimension + 1)
            ),
        )
    return complex_ir


def _fields() -> Any:
    return (
        phx.nn.operator.OperatorFieldSpec(
            "vertex",
            role="both",
            source_name="vertex_source",
            query_name="vertex_query",
            form_type=phx.exterior.FormType(2, 0),
        ),
        phx.nn.operator.OperatorFieldSpec(
            "edge",
            role="both",
            source_name="edge_source",
            query_name="edge_query",
            form_type=phx.exterior.FormType(2, 1),
        ),
    )


def _field_binding() -> Any:
    # The cochain operator is built from `_fields()`: each output is the field port.
    ports = {field.name: field.value_port() for field in _fields() if field.is_target}
    return {
        "output_ports": ports,
        "port_mapping": phx.PortMapping(
            outputs=tuple((port.port_id, port.port_id) for port in ports.values())
        ),
    }


def _task(fields: Any = None) -> Any:
    resolved_fields = _fields() if fields is None else tuple(fields)
    query_names = tuple(field.query_name for field in resolved_fields if field.is_target)
    return phx.nn.operator.OperatorTask(
        "cochain-map",
        fields=resolved_fields,
        queries=tuple(
            phx.nn.operator.OperatorQuerySpec(
                name,
                geometry_kind="cell_complex",
                coordinate_components=("x", "y"),
                topology_site="cell",
                quadrature="native_pairing_required",
            )
            for name in query_names
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="shared_topology",
            query_is_fixed=False,
            requires_resolution_transfer=True,
        ),
    )


def _batch(complex_ir: Any = None, *, cases: Any = 3, edge_values: Any = None) -> Any:
    complex_ir = _square_complex() if complex_ir is None else complex_ir
    vertex_count, edge_count = complex_ir.cell_counts[:2]
    vertex_values = jnp.arange(cases * vertex_count, dtype="float64").reshape(
        cases, vertex_count
    )
    if edge_values is None:
        edge_values = jnp.linspace(
            -1.0,
            2.0,
            cases * edge_count,
        ).reshape(cases, edge_count)
    vertex_source = phx.nn.operator.function_samples_from_cochain(
        complex_ir,
        0,
        values=vertex_values,
    )
    edge_source = phx.nn.operator.function_samples_from_cochain(
        complex_ir,
        1,
        values=edge_values,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={
            "vertex_source": vertex_source,
            "edge_source": edge_source,
        },
        queries={
            "vertex_query": phx.nn.operator.function_samples_from_cochain(
                complex_ir,
                0,
                values=None,
            ),
            "edge_query": phx.nn.operator.function_samples_from_cochain(
                complex_ir,
                1,
                values=None,
            ),
        },
        case_axes=("case",),
        case_shape=(cases,),
    )


def _dataset(batch: Any) -> Any:
    fields = _fields()
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {
            "vertex": 0.5 * batch.input("vertex_source").values + 0.1,
            "edge": -0.25 * batch.input("edge_source").values,
        },
        batch,
        query_names={
            "vertex": "vertex_query",
            "edge": "edge_query",
        },
        specs={field.name: field.output_spec for field in fields if field.is_target},
    )
    return phx.nn.operator.training.OperatorDataset(batch, targets)


def _model(*, key: Any = jr.key(0), routes: Any = None) -> Any:
    return phx.nn.operator.architectures.CochainNeuralOperator(
        _fields(),
        width=5,
        depth=2,
        routes=routes,
        key=key,
    )


def _trainable_arrays(model: Any) -> Any:
    return jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))


def test_old_field_payload_is_refused_without_form_type() -> None:
    payload = _fields()[0].to_dict()
    del payload["form_type"]
    payload["cochain"] = {
        "degree": 0,
        "complex_side": "primal",
        "cell_orientation": "invariant",
        "sampling": "point_value",
    }
    with pytest.raises(ValueError, match="form_type"):
        phx.nn.operator.OperatorFieldSpec.from_dict(payload)


def test_old_task_field_payload_is_refused_without_form_type() -> None:
    payload = _task().to_dict()
    del payload["fields"][0]["form_type"]
    with pytest.raises(ValueError, match="form_type"):
        phx.nn.operator.OperatorTask.from_dict(payload)


def test_twisted_zero_form_offsets_are_refused() -> None:
    with pytest.raises(ValueError, match="zero dimensional offsets"):
        phx.nn.operator.OperatorFieldSpec(
            "density",
            role="source",
            form_type=phx.exterior.FormType(2, 0, twist="twisted"),
            representation="cochain",
            offset=1.0,
        )


@pytest.mark.parametrize(
    ("dimension", "twist"),
    [(3, "untwisted"), (2, "twisted")],
    ids=["dimension", "placement"],
)
def test_cochain_form_type_must_match_graph_realization(
    dimension: int, twist: FormTwist
) -> None:
    fields = tuple(
        phx.nn.operator.OperatorFieldSpec(
            field.name,
            role=field.role,
            source_name=field.source_name,
            query_name=field.query_name,
            form_type=phx.exterior.FormType(dimension, degree, twist=twist),
            representation="cochain",
        )
        for degree, field in enumerate(_fields())
    )
    rejected = phx.nn.operator.validate_operator_architecture(
        "CochainNeuralOperator",
        _batch(cases=2),
        problem=_task().problem,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        fields=fields,
    )
    assert not rejected.accepted
    assert "COCHAIN_FORM_TYPE_MISMATCH" in rejected.codes


def test_cochain_contracts() -> None:
    task = _task()
    restored = phx.nn.operator.OperatorTask.from_dict(task.to_dict())
    assert restored.fields[0].form_type is not None
    assert restored.fields[1].form_type is not None

    assert restored.fingerprint == task.fingerprint
    assert (
        restored.fields[0].form_type.form_type_id
        == phx.exterior.FormType(2, 0).form_type_id
    )
    assert (
        restored.fields[1].form_type.form_type_id
        == phx.exterior.FormType(2, 1).form_type_id
    )
    assert restored.fields[0].representation == "cochain"
    assert restored.fields[1].representation == "cochain"
    with pytest.raises(ValueError, match="zero dimensional offsets"):
        phx.nn.operator.OperatorFieldSpec(
            "invalid",
            role="source",
            offset=1.0,
            form_type=phx.exterior.FormType(2, 1),
        )
    batch = _batch(cases=2)
    first = phx.nn.operator.slice_operator_batch(batch, 0)
    second = phx.nn.operator.slice_operator_batch(batch, 1)
    restacked = phx.nn.operator.stack_operator_batches(
        (first, second),
        case_axis="case",
    )
    padded = phx.nn.operator.pad_function_samples(first.input("edge_source"), 7)
    graph = phx.nn.operator.materialize_operator_fields(restacked, _fields())

    topology = restacked.input("edge_source").topology
    query_topology = restacked.query("vertex_query").topology
    padded_topology = padded.topology
    restacked_values = restacked.input("edge_source").values
    batch_values = batch.input("edge_source").values
    assert topology is not None
    assert query_topology is not None
    assert padded_topology is not None
    assert restacked_values is not None
    assert batch_values is not None
    assert topology.kind == "cell_complex"
    assert topology.site == "cell"
    assert topology.graph_fingerprint == query_topology.graph_fingerprint
    assert jnp.array_equal(
        padded_topology.sample_entities,
        jnp.asarray([4, 5, 6, 7, 8, -1, -1]),
    )
    assert graph.num_graphs == 2
    assert graph.nodes["field:vertex"].shape == (22,)
    assert graph.nodes["field:edge"].shape == (22,)
    assert jnp.allclose(
        restacked_values,
        batch_values,
    )
    batch = _batch(cases=2)
    accepted = phx.nn.operator.validate_operator_architecture(
        "CochainNeuralOperator",
        batch,
        problem=_task().problem,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        fields=_fields(),
    )

    mismatched = phx.nn.operator.OperatorBatch(
        inputs=batch.inputs,
        queries={
            "vertex_query": batch.query("vertex_query"),
            "edge_query": _batch(_square_complex(shift=0.2), cases=2).query("edge_query"),
        },
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    rejected = phx.nn.operator.validate_operator_architecture(
        "CochainNeuralOperator",
        mismatched,
        problem=_task().problem,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        fields=_fields(),
    )

    assert accepted.accepted
    assert not rejected.accepted
    assert "COCHAIN_TOPOLOGY_MISMATCH" in rejected.codes
    complex_ir = _square_complex()
    batch = _batch(complex_ir, cases=2)
    signs = (
        jnp.ones((complex_ir.cell_counts[0],)),
        jnp.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        jnp.asarray([1.0, -1.0]),
    )
    reoriented = reoriented_lowering(complex_ir, signs)
    transformed_edges = phx.discretization.reorient_cochain(
        batch.input("edge_source").values, signs[1], cell_axis=1
    )
    transformed_batch = _batch(
        reoriented,
        cases=2,
        edge_values=transformed_edges,
    )
    model = _model(key=jr.key(4))

    original = model.evaluate(batch)
    transformed = model.evaluate(transformed_batch)

    assert jnp.allclose(
        transformed.field("vertex").values,
        original.field("vertex").values,
        atol=1e-10,
    )
    assert jnp.allclose(
        transformed.field("edge").values,
        phx.discretization.reorient_cochain(
            original.field("edge").values, signs[1], cell_axis=1
        ),
        atol=1e-10,
    )
    batch = _batch(cases=3)
    dataset = _dataset(batch)
    policy = phx.nn.operator.training.fit_operator_normalization(
        batch,
        dataset.targets,
        fields=_fields(),
        weighting="quadrature",
    )
    signs = jnp.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
    reoriented_edges = phx.discretization.reorient_cochain(
        batch.input("edge_source").values, signs, cell_axis=1
    )
    reoriented_batch = eqx.tree_at(
        lambda item: item.inputs["edge_source"].values,
        batch,
        reoriented_edges,
    )
    reoriented_targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {
            "vertex": dataset.targets.field("vertex").values,
            "edge": phx.discretization.reorient_cochain(
                dataset.targets.field("edge").values, signs, cell_axis=1
            ),
        },
        reoriented_batch,
        query_names={"vertex": "vertex_query", "edge": "edge_query"},
    )
    transformed_policy = phx.nn.operator.training.fit_operator_normalization(
        reoriented_batch,
        reoriented_targets,
        fields=_fields(),
        weighting="quadrature",
    )

    assert jnp.allclose(policy.input_values["edge_source"].mean, 0.0)
    assert jnp.allclose(policy.targets["edge"].mean, 0.0)
    assert not jnp.allclose(policy.input_values["vertex_source"].mean, 0.0)
    assert jnp.allclose(
        policy.input_values["edge_source"].scale,
        transformed_policy.input_values["edge_source"].scale,
    )
    assert jnp.allclose(
        policy.targets["edge"].scale,
        transformed_policy.targets["edge"].scale,
    )
    dataset = _targetless_dataset(cases=2)
    model = _small_cochain_model(key=jr.key(30))
    term = _source_matching_loss()
    value = _physics_loss_value(term, model, dataset)
    topology = dataset.batch.input("vertex_source").topology

    assert topology is not None
    assert jnp.isfinite(value)
    assert value > 0.0
    assert term.fingerprint == _source_matching_loss().fingerprint
    assert (
        term.fingerprint
        != _source_matching_loss(
            identity="tests.cochain.changed_source_matching"
        ).fingerprint
    )

    locked = _source_matching_loss(topology_fingerprint="not-this-topology")
    with pytest.raises(ValueError, match="does not match its declared fingerprint"):
        _physics_loss_value(locked, model, dataset)


@pytest.mark.parametrize("sparse_metric", [False, True])
def test_cochain_operator_is_multi_output_batched_jittable_and_differentiable(
    sparse_metric: bool,
) -> None:
    lowering = _square_sparse_complex()[0] if sparse_metric else _square_complex()
    batch = _batch(lowering, cases=2)
    model = _model()

    prediction = model.evaluate(batch)
    compiled = eqx.filter_jit(lambda current, value: current.predict_prevalidated(value))(
        model, batch
    )
    captured = eqx.filter_jit(lambda current: current.evaluate(batch))(model)

    def objective(edge_values: Any) -> Any:
        changed = eqx.tree_at(
            lambda item: item.inputs["edge_source"].values,
            batch,
            edge_values,
        )
        fields = model.predict_fields(changed)
        return jnp.sum(fields["vertex"] ** 2) + jnp.sum(fields["edge"] ** 2)

    gradient = jax.grad(objective)(batch.input("edge_source").values)

    assert tuple(prediction.fields) == ("edge", "vertex")
    assert prediction.field("vertex").values.shape == (2, 4)
    assert prediction.field("edge").values.shape == (2, 5)
    assert jnp.allclose(
        compiled.field("vertex").values,
        prediction.field("vertex").values,
    )
    assert jnp.allclose(
        compiled.field("edge").values,
        prediction.field("edge").values,
    )
    for name in ("vertex", "edge"):
        np.testing.assert_allclose(
            captured.field(name).values,
            prediction.field(name).values,
            rtol=1e-11,
            atol=1e-12,
        )
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0


def test_harmonic_route_requires_and_uses_precomputed_topological_basis() -> None:
    fields = (
        phx.nn.operator.OperatorFieldSpec(
            "edge",
            role="both",
            source_name="edge_source",
            query_name="edge_query",
            form_type=phx.exterior.FormType(2, 1),
        ),
    )
    routes = phx.nn.operator.architectures.TopologicalRouteConfig(
        self_route=False,
        exterior_derivative=False,
        codifferential=False,
        lower_laplacian=False,
        upper_laplacian=False,
        harmonic=True,
    )
    model = phx.nn.operator.architectures.CochainNeuralOperator(
        fields,
        width=3,
        depth=1,
        routes=routes,
        key=jr.key(5),
    )

    def edge_batch(complex_ir: Any) -> Any:
        count = complex_ir.cell_counts[1]
        return phx.nn.operator.OperatorBatch(
            inputs={
                "edge_source": phx.nn.operator.function_samples_from_cochain(
                    complex_ir,
                    1,
                    values=jnp.linspace(-1.0, 1.0, count),
                )
            },
            queries={
                "edge_query": phx.nn.operator.function_samples_from_cochain(
                    complex_ir,
                    1,
                    values=None,
                )
            },
        )

    output = model(edge_batch(_annulus_complex(harmonics=True)))

    assert output.shape == (16,)
    assert jnp.all(jnp.isfinite(output))
    with pytest.raises(ValueError, match="precomputed HarmonicSubspace"):
        model(edge_batch(_annulus_complex(harmonics=False)))


@pytest.mark.parametrize("boundary", ["absolute", "relative"])
@pytest.mark.parametrize("sparse_metric", [False, True])
def test_degree_boundary_routes_match_incidence_and_gram(
    boundary: Any, sparse_metric: bool
) -> None:
    if sparse_metric:
        lowering, full_gram = _square_sparse_complex()
    else:
        lowering = _square_complex()
        full_gram = tuple(
            np.diag(np.asarray(hodge.weights)) for hodge in lowering.discretization.hodges
        )
    realization = lowering.discretization
    block = phx.nn.operator.architectures.TopologicalCochainBlock(
        2, (0, 1, 2), dimension=2, boundary=boundary, key=jr.key(72)
    )
    hidden = np.asarray(jr.normal(jr.key(73), (lowering.num_cells, 2)))
    indices = tuple(
        np.asarray(realization.active_indices(k, boundary=boundary))
        + lowering.cell_offsets[k]
        for k in range(3)
    )
    gram = tuple(
        full_gram[k][
            np.ix_(
                np.asarray(realization.active_indices(k, boundary=boundary)),
                np.asarray(realization.active_indices(k, boundary=boundary)),
            )
        ]
        for k in range(3)
    )
    differential = tuple(
        np.asarray(incidence.scipy_boundary().toarray()).T[
            np.ix_(
                np.asarray(realization.active_indices(k + 1, boundary=boundary)),
                np.asarray(realization.active_indices(k, boundary=boundary)),
            )
        ]
        for k, incidence in enumerate(realization.topology.incidences)
    )
    codifferential = tuple(
        np.linalg.solve(gram[k], differential[k].T @ gram[k + 1]) for k in range(2)
    )
    expected = np.zeros_like(hidden)
    for degree in range(3):
        values = hidden[indices[degree]]
        routes = {"self": values}
        if degree > 0:
            routes["exterior_derivative"] = (
                differential[degree - 1] @ hidden[indices[degree - 1]]
            )
            routes["lower_laplacian"] = (
                differential[degree - 1] @ codifferential[degree - 1] @ values
            )
        if degree < 2:
            routes["codifferential"] = (
                codifferential[degree] @ hidden[indices[degree + 1]]
            )
            routes["upper_laplacian"] = (
                codifferential[degree] @ differential[degree] @ values
            )
        mixed = np.zeros_like(values)
        for route_index, name in enumerate(block.route_names):
            if name not in routes:
                with pytest.raises(ValueError, match="is absent at degree"):
                    block._route(name, lowering.graph, jnp.asarray(hidden), degree)
                continue
            actual = block._route(name, lowering.graph, jnp.asarray(hidden), degree)
            np.testing.assert_allclose(
                np.asarray(actual)[indices[degree]], routes[name], rtol=1e-11, atol=1e-12
            )
            mixed += routes[name] @ np.asarray(block.route_weights[route_index][degree])
        rms = np.sqrt(np.mean(mixed**2, axis=-1, keepdims=True) + block.norm_epsilon)
        gate = 1.0 + 0.1 * np.tanh(np.asarray(block.degree_embeddings[degree]))
        expected[indices[degree]] = values + np.asarray(
            block.residual_scales[degree]
        ) * gate * np.tanh(mixed / rms)
    actual = eqx.filter_jit(lambda current, value: current(lowering.graph, value))(
        block, jnp.asarray(hidden)
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-11, atol=1e-12)


def test_full_gram_signed_covariance_and_metric_graph_identity() -> None:
    lowering, grams = _square_sparse_complex()
    owner = lowering.discretization
    signs = (
        np.ones(4),
        np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        np.asarray([1.0, -1.0]),
    )
    reoriented_hodges = []
    for degree, gram in enumerate(grams):
        transformed = signs[degree][:, None] * gram * signs[degree][None, :]
        rows, columns = np.triu_indices(gram.shape[0])
        reoriented_hodges.append(
            phx.discretization.SparseHodge(
                rows, columns, transformed[rows, columns], gram.shape[0]
            )
        )
    reoriented = phx.graph.CochainComplexIR(
        phx.discretization.CochainDiscretization(
            phx.discretization.reorient_cell_complex(owner.topology, signs),
            tuple(reoriented_hodges),
            coordinates=owner.coordinates,
            boundary_masks=owner.boundary_masks,
            primal_measures=owner.primal_measures,
            dual_measures=owner.dual_measures,
        )
    )
    batch = _batch(lowering, cases=2)
    transformed_batch = _batch(
        reoriented,
        cases=2,
        edge_values=batch.input("edge_source").values * signs[1],
    )
    model = _model(key=jr.key(75))
    original = model.predict_fields(batch)
    transformed = model.predict_fields(transformed_batch)
    np.testing.assert_allclose(
        transformed["vertex"], original["vertex"], rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        transformed["edge"], original["edge"] * signs[1], rtol=1e-11, atol=1e-12
    )
    changed_hodges = tuple(
        hodge.refresh(
            hodge.upper_values
            + 0.02 * (jnp.asarray(hodge.rows) != jnp.asarray(hodge.columns))
        )
        for hodge in owner.hodges
    )
    changed = phx.graph.CochainComplexIR(
        owner.with_metric(changed_hodges, numeric_revision="changed-off-diagonal")
    )
    assert phx.graph.operator_graph_fingerprint(
        changed.graph
    ) != phx.graph.operator_graph_fingerprint(lowering.graph)


def test_cochain_admission_refuses_graph_metadata_without_native_pairing() -> None:
    batch = _batch(cases=2)

    def without_owner(samples: Any) -> Any:
        topology = samples.topology
        assert topology is not None
        graph = topology.graph.replace(cochain_bindings=())
        topology = phx.graph.OperatorTopology(
            graph,
            topology.sample_entities,
            case_shape=topology.case_shape,
            kind=topology.kind,
            site=topology.site,
            entity=topology.entity,
        )
        return phx.nn.operator.FunctionSamples(
            values=samples.values,
            coordinates=samples.coordinates,
            quadrature_weights=samples.quadrature_weights,
            mask=samples.mask,
            topology=topology,
        )

    metadata_only = phx.nn.operator.OperatorBatch(
        inputs={name: without_owner(samples) for name, samples in batch.inputs.items()},
        queries={name: without_owner(samples) for name, samples in batch.queries.items()},
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    with pytest.raises(ValueError, match="COCHAIN_METRIC_REQUIRED"):
        _model().evaluate(metadata_only)
    with pytest.raises(ValueError, match="native cochain pairing"):
        _task().validate_batch(metadata_only)


def test_compiled_self_routes_guard_current_native_metric_evidence() -> None:
    lowering, grams = _square_sparse_complex()
    owner = lowering.discretization
    batch = _batch(lowering, cases=2)
    routes = phx.nn.operator.architectures.TopologicalRouteConfig(
        exterior_derivative=False,
        codifferential=False,
        lower_laplacian=False,
        upper_laplacian=False,
    )
    model = _model(routes=routes, key=jr.key(91))

    @eqx.filter_jit
    def predict(values: Any) -> Any:
        metric = owner.hodges[0].refresh(values)
        refreshed = owner.with_metric(
            (metric,) + owner.hodges[1:], numeric_revision=owner.numeric_revision
        )

        def replace_owner(samples: Any) -> Any:
            topology = samples.topology
            binding = topology.graph.cochain_bindings[0]
            binding = eqx.tree_at(lambda item: item.discretization, binding, refreshed)
            graph = topology.graph.replace(cochain_bindings=(binding,))
            topology = eqx.tree_at(lambda item: item.graph, topology, graph)
            return eqx.tree_at(lambda item: item.topology, samples, topology)

        current = phx.nn.operator.OperatorBatch(
            inputs={name: replace_owner(value) for name, value in batch.inputs.items()},
            queries={name: replace_owner(value) for name, value in batch.queries.items()},
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )
        return model.evaluate(current).field("vertex").values

    expected = model.evaluate(batch).field("vertex").values
    np.testing.assert_allclose(
        predict(owner.hodges[0].upper_values), expected, rtol=1e-11, atol=1e-12
    )
    invalid = grams[0].copy()
    invalid[0, 1] = invalid[1, 0] = 10.0
    metric = owner.hodges[0]
    values = invalid[np.asarray(metric.rows), np.asarray(metric.columns)]
    with pytest.raises(eqx.EquinoxRuntimeError):
        predict(jnp.asarray(values)).block_until_ready()


def test_sparse_native_pairing_refuses_non_spd_numerical_update() -> None:
    lowering, grams = _square_sparse_complex()
    owner = lowering.discretization
    invalid = grams[0].copy()
    invalid[0, 1] = invalid[1, 0] = 10.0
    assert np.all(np.diag(invalid) > 0.0)
    assert np.min(np.linalg.eigvalsh(invalid)) < 0.0
    metric = owner.hodges[0]
    changed = metric.refresh(invalid[np.asarray(metric.rows), np.asarray(metric.columns)])
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError)):
        candidate = owner.with_metric(
            (changed,) + owner.hodges[1:], numeric_revision="non-spd-cochain-input"
        )
        _model().evaluate(_batch(phx.graph.CochainComplexIR(candidate), cases=2))


def test_cochain_routes_refuse_unknown_selectors_and_invalid_degrees() -> None:
    lowering = _square_complex()
    block = phx.nn.operator.architectures.TopologicalCochainBlock(
        1, (0, 1, 2), dimension=2
    )
    hidden = jnp.ones((lowering.num_cells, 1))
    invalid_route: Any = "typo_laplacian"
    with pytest.raises(ValueError):
        block._route(invalid_route, lowering.graph, hidden, 1)
    with pytest.raises(ValueError):
        phx.nn.operator.architectures.TopologicalCochainBlock(1, (0, 3), dimension=2)
    with pytest.raises(ValueError):
        phx.nn.operator.architectures.CochainNeuralOperator(
            _fields(), active_degrees=(0, 1, 3)
        )


def test_operator_cochain_scenario_1() -> None:
    complex_ir = _square_complex()
    block = phx.nn.operator.architectures.TopologicalCochainBlock(
        2,
        (0, 1, 2),
        dimension=2,
        routes=phx.nn.operator.architectures.TopologicalRouteConfig(
            self_route=True,
            exterior_derivative=False,
            codifferential=False,
            lower_laplacian=False,
            upper_laplacian=False,
            harmonic=False,
        ),
        key=jr.key(8),
    )
    block = eqx.tree_at(
        lambda item: item.route_weights,
        block,
        tuple(jnp.zeros_like(weight) for weight in block.route_weights),
    )
    hidden = jr.normal(jr.key(9), (complex_ir.num_cells, 2))

    one_step = block(complex_ir.graph, hidden)
    three_steps = block(
        complex_ir.graph,
        block(complex_ir.graph, block(complex_ir.graph, hidden)),
    )

    assert jnp.array_equal(one_step, hidden)
    assert jnp.array_equal(three_steps, hidden)
    dataset = _targetless_dataset(cases=2)
    model = _small_cochain_model(key=jr.key(32))
    common: dict[str, Any] = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        **_field_binding(),
        "batch_size": 2,
        "steps": 1,
        "shuffle": False,
        "seed": 20,
    }

    with pytest.raises(ValueError, match="explicit physics loss_terms"):
        phx.nn.operator.training.fit_operator(model, dataset, **common)
    with pytest.raises(ValueError, match="supervised targets"):
        phx.nn.operator.training.fit_operator(
            model,
            dataset,
            loss_terms=(_source_matching_loss(),),
            normalization="fit",
            **common,
        )


@pytest.mark.parametrize("sparse_metric", [False, True])
def test_multi_field_training_and_checkpoint_resume_are_exact(
    tmp_path: Any,
    sparse_metric: bool,
) -> None:
    lowering = _square_sparse_complex()[0] if sparse_metric else _square_complex()
    dataset = _dataset(_batch(lowering, cases=3))
    model = _model(key=jr.key(12))
    common: dict[str, Any] = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        **_field_binding(),
        "learning_rate": 1e-3,
        "batch_size": 3,
        "epochs": 2,
        "shuffle": False,
        "seed": 17,
        "normalization": "fit",
        "checkpoint_every": 1,
    }

    uninterrupted = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        **common,
    )
    checkpoint = tmp_path / "cochain-checkpoint"
    first = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=1,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    uninterrupted_prediction = uninterrupted.execution_model.evaluate(dataset.batch)
    resumed_prediction = resumed.execution_model.evaluate(dataset.batch)
    assert first.progress.update_step == 1
    assert resumed.resumed_from_step == 1
    assert resumed.progress.update_step == 2
    assert tuple(resumed_prediction.fields) == ("edge", "vertex")
    for name in ("vertex", "edge"):
        assert jnp.array_equal(
            resumed_prediction.field(name).values,
            uninterrupted_prediction.field(name).values,
        )
    assert len(_trainable_arrays(resumed.execution_model)) == len(
        _trainable_arrays(uninterrupted.execution_model)
    )
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            _trainable_arrays(resumed.execution_model),
            _trainable_arrays(uninterrupted.execution_model),
            strict=True,
        )
    )


def _plain_source_residual(graph: Any, fields: Any, *, key: Any) -> Any:
    del graph, key
    return {"residual": fields["u"] - 0.1 * fields["forcing"]}


def _plain_scaled_residual(graph: Any, fields: Any, *, key: Any) -> Any:
    del graph, key
    return {"residual": fields["u"] - 0.2 * fields["forcing"]}


def test_cochain_residual_program_identity_uses_canonical_callable_payload() -> None:
    zero_spec = phx.exterior.FormType(2, 0)

    def program(residual_fn: Any, **ids: Any) -> Any:
        return phx.graph.CochainResidualProgram(
            inputs={"u": zero_spec, "forcing": zero_spec},
            outputs={"residual": zero_spec},
            residual_fn=residual_fn,
            **ids,
        )

    plain = program(_plain_source_residual)
    assert plain.fingerprint == program(_plain_source_residual).fingerprint
    assert plain.fingerprint != program(_plain_scaled_residual).fingerprint

    first = lambda graph, fields, *, key: {"residual": fields["u"]}
    second = lambda graph, fields, *, key: {"residual": -fields["u"]}
    for opaque in (first, second):
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            program(opaque)
    declared_first = program(
        first, residual_semantic_id="identity", residual_numeric_id="identity"
    )
    declared_second = program(
        second, residual_semantic_id="negation", residual_numeric_id="negation"
    )
    assert declared_first.fingerprint != declared_second.fingerprint


def _source_matching_program(*, identity: Any = "tests.cochain.source_matching") -> Any:
    zero_spec = phx.exterior.FormType(2, 0)

    def residual(graph: Any, fields: Any, *, key: Any) -> Any:
        del graph, key
        return {"residual": fields["u"] - 0.1 * fields["forcing"]}

    return phx.graph.CochainResidualProgram(
        inputs={"u": zero_spec, "forcing": zero_spec},
        outputs={"residual": zero_spec},
        residual_fn=residual,
        residual_semantic_id=identity,
        residual_numeric_id=identity,
    )


def _source_matching_loss(
    *,
    identity: Any = "tests.cochain.source_matching",
    topology_fingerprint: Any = None,
) -> Any:
    return phx.nn.operator.training.CochainResidualLoss(
        name="zero_form_physics",
        program=_source_matching_program(identity=identity),
        inputs={
            "u": phx.nn.operator.training.CochainResidualInput("prediction", "vertex"),
            "forcing": phx.nn.operator.training.CochainResidualInput("source", "vertex"),
        },
        output="residual",
        reduction="metric_mean",
        topology_fingerprint=topology_fingerprint,
    )


def _targetless_dataset(*, cases: Any = 2) -> Any:
    batch = _batch(cases=cases)
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch)
    return phx.nn.operator.training.OperatorDataset(batch, targets)


def _small_cochain_model(*, key: Any) -> Any:
    return phx.nn.operator.architectures.CochainNeuralOperator(
        _fields(),
        width=3,
        depth=1,
        key=key,
    )


def _physics_loss_value(term: Any, model: Any, dataset: Any) -> Any:
    prediction = model.evaluate(dataset.batch)
    context = phx.nn.operator.training.OperatorLossContext(
        prediction,
        dataset.batch,
        dataset.targets,
        prediction,
        dataset.batch,
        dataset.targets,
        task=_task(),
    )
    return term(
        model,
        prediction,
        dataset.batch,
        dataset.targets,
        key=jr.key(91),
        step=jnp.asarray(0),
        training=False,
        context=context,
    )


def test_full_gram_physics_loss_matches_independent_quadratic_form() -> None:
    lowering, grams = _square_sparse_complex()
    batch = _batch(lowering, cases=2)
    dataset = phx.nn.operator.training.OperatorDataset(
        batch, phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch)
    )
    model = _small_cochain_model(key=jr.key(74))
    prediction = model.evaluate(batch)
    residual = np.asarray(prediction.field("vertex").values) - 0.1 * np.asarray(
        batch.input("vertex_source").values
    )
    expected = np.mean(
        np.einsum("ci,ij,cj->c", residual, grams[0], residual) / np.trace(grams[0])
    )
    actual = _physics_loss_value(_source_matching_loss(), model, dataset)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("sparse_metric", [False, True])
def test_targetless_cochain_pino_update_and_checkpoint_resume_are_exact(
    tmp_path: Any,
    sparse_metric: bool,
) -> None:
    lowering = _square_sparse_complex()[0] if sparse_metric else _square_complex()
    batch = _batch(lowering, cases=2)
    dataset = phx.nn.operator.training.OperatorDataset(
        batch, phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch)
    )
    model = _small_cochain_model(key=jr.key(31))
    term = _source_matching_loss()
    common: dict[str, Any] = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        **_field_binding(),
        "loss_terms": (term,),
        "learning_rate": 1e-3,
        "batch_size": 2,
        "epochs": 2,
        "shuffle": False,
        "seed": 19,
        "normalization": None,
        "checkpoint_every": 1,
    }
    initial_loss = _physics_loss_value(term, model, dataset)
    uninterrupted = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        **common,
    )
    trained_loss = _physics_loss_value(term, uninterrupted.execution_model, dataset)

    checkpoint = tmp_path / "targetless-cochain-checkpoint"
    first = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=1,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed = phx.nn.operator.training.fit_operator(
        model,
        dataset,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    assert trained_loss < initial_loss
    assert first.progress.update_step == 1
    assert resumed.resumed_from_step == 1
    uninterrupted_prediction = uninterrupted.execution_model.evaluate(dataset.batch)
    resumed_prediction = resumed.execution_model.evaluate(dataset.batch)
    for name in ("vertex", "edge"):
        assert jnp.array_equal(
            resumed_prediction.field(name).values,
            uninterrupted_prediction.field(name).values,
        )

    changed_common: dict[str, Any] = dict(common)
    changed_common["loss_terms"] = (
        _source_matching_loss(identity="tests.cochain.incompatible_physics"),
    )
    with pytest.raises(ValueError, match="checkpoint contract mismatch"):
        phx.nn.operator.training.fit_operator(
            model,
            dataset,
            steps=2,
            checkpoint_path=checkpoint,
            resume=True,
            **changed_common,
        )
