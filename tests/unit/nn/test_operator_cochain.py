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


def _square_complex(*, shift=0.0):
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vertices = vertices + float(shift)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)


def _annulus_complex(*, harmonics=False):
    outer = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    vertices = np.concatenate((outer, 0.4 * outer), axis=0)
    faces = np.asarray(
        [(index, (index + 1) % 4, 4 + (index + 1) % 4) for index in range(4)]
        + [(index, 4 + (index + 1) % 4, 4 + index) for index in range(4)],
        dtype=np.int32,
    )
    complex_ir = phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)
    if harmonics:
        complex_ir = complex_ir.with_harmonic_subspace(
            phx.graph.compute_harmonic_subspace(complex_ir, max_modes=3)
        )
    return complex_ir


def _fields():
    return (
        phx.nn.operator.OperatorFieldSpec(
            "vertex",
            role="both",
            source_name="vertex_source",
            query_name="vertex_query",
            cochain=phx.graph.CochainFieldSpec(
                0,
                cell_orientation="invariant",
                sampling="point_value",
            ),
        ),
        phx.nn.operator.OperatorFieldSpec(
            "edge",
            role="both",
            source_name="edge_source",
            query_name="edge_query",
            cochain=phx.graph.CochainFieldSpec(
                1,
                cell_orientation="signed",
                sampling="cell_integral",
            ),
        ),
    )


def _task(fields=None):
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
                quadrature="physical_required",
            )
            for name in query_names
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="shared_topology",
            query_is_fixed=False,
            requires_resolution_transfer=True,
        ),
    )


def _batch(complex_ir=None, *, cases=3, edge_values=None):
    complex_ir = _square_complex() if complex_ir is None else complex_ir
    vertex_count, edge_count = complex_ir.cell_counts[:2]
    vertex_values = jnp.arange(cases * vertex_count, dtype=float).reshape(
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


def _dataset(batch):
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


def _model(*, key=jr.key(0), routes=None):
    return phx.nn.operator.architectures.CochainNeuralOperator(
        _fields(),
        width=5,
        depth=2,
        routes=routes,
        key=key,
    )


def _trainable_arrays(model):
    return jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))


def test_cochain_field_semantics_roundtrip_through_operator_task():
    task = _task()
    restored = phx.nn.operator.OperatorTask.from_dict(task.to_dict())
    assert restored.fields[0].cochain is not None
    assert restored.fields[1].cochain is not None

    assert restored.fingerprint == task.fingerprint
    assert restored.fields[0].cochain.to_dict() == {
        "degree": 0,
        "complex_side": "primal",
        "cell_orientation": "invariant",
        "sampling": "point_value",
    }
    assert restored.fields[1].cochain.cell_orientation == "signed"
    with pytest.raises(ValueError, match="zero dimensional offsets"):
        phx.nn.operator.OperatorFieldSpec(
            "invalid",
            role="source",
            offset=1.0,
            cochain=phx.graph.CochainFieldSpec(
                1,
                cell_orientation="signed",
                sampling="cell_integral",
            ),
        )


def test_cochain_topology_survives_materialization_padding_stacking_and_slicing():
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
    assert topology.case_shape == (2,)
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


def test_cochain_capability_contract_accepts_typed_fields_and_rejects_mismatch():
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


def test_cochain_operator_is_multi_output_batched_jittable_and_differentiable():
    batch = _batch(cases=2)
    model = _model()

    prediction = model.predict(batch)
    compiled = eqx.filter_jit(lambda current, value: current.predict_prevalidated(value))(
        model, batch
    )

    def objective(edge_values):
        changed = eqx.tree_at(
            lambda item: item.inputs["edge_source"].values,
            batch,
            edge_values,
        )
        fields = model.predict_fields(changed)
        return jnp.sum(fields["vertex"] ** 2) + jnp.sum(fields["edge"] ** 2)

    gradient = jax.grad(objective)(batch.input("edge_source").values)

    assert tuple(prediction.fields) == ("vertex", "edge")
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
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0


def test_cochain_operator_is_equivariant_to_independent_cell_reorientation():
    complex_ir = _square_complex()
    batch = _batch(complex_ir, cases=2)
    signs = (
        jnp.ones((complex_ir.cell_counts[0],)),
        jnp.asarray([-1.0, 1.0, -1.0, 1.0, -1.0]),
        jnp.asarray([1.0, -1.0]),
    )
    reoriented = phx.graph.reorient_cochain_complex(complex_ir, signs)
    transformed_edges = phx.graph.reorient_cochain(
        batch.input("edge_source").values,
        signs[1],
    )
    transformed_batch = _batch(
        reoriented,
        cases=2,
        edge_values=transformed_edges,
    )
    model = _model(key=jr.key(4))

    original = model.predict(batch)
    transformed = model.predict(transformed_batch)

    assert jnp.allclose(
        transformed.field("vertex").values,
        original.field("vertex").values,
        atol=1e-10,
    )
    assert jnp.allclose(
        transformed.field("edge").values,
        phx.graph.reorient_cochain(original.field("edge").values, signs[1]),
        atol=1e-10,
    )


def test_harmonic_route_requires_and_uses_precomputed_topological_basis():
    fields = (
        phx.nn.operator.OperatorFieldSpec(
            "edge",
            role="both",
            source_name="edge_source",
            query_name="edge_query",
            cochain=phx.graph.CochainFieldSpec(
                1,
                cell_orientation="signed",
                sampling="cell_integral",
            ),
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

    def edge_batch(complex_ir):
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


def test_zero_update_topological_block_has_exact_semigroup_identity():
    complex_ir = _square_complex()
    block = phx.nn.operator.architectures.TopologicalCochainBlock(
        2,
        (0, 1, 2),
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


def test_cochain_normalization_centers_invariant_fields_but_not_signed_fields():
    batch = _batch(cases=3)
    dataset = _dataset(batch)
    policy = phx.nn.operator.training.fit_operator_normalization(
        batch,
        dataset.targets,
        fields=_fields(),
        weighting="quadrature",
    )
    signs = jnp.asarray([-1.0, 1.0, -1.0, 1.0, -1.0])
    reoriented_edges = phx.graph.reorient_cochain(
        batch.input("edge_source").values,
        signs,
    )
    reoriented_batch = eqx.tree_at(
        lambda item: item.inputs["edge_source"].values,
        batch,
        reoriented_edges,
    )
    reoriented_targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {
            "vertex": dataset.targets.field("vertex").values,
            "edge": phx.graph.reorient_cochain(
                dataset.targets.field("edge").values,
                signs,
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


def test_multi_field_training_and_checkpoint_resume_are_exact(tmp_path):
    dataset = _dataset(_batch(cases=3))
    model = _model(key=jr.key(12))
    common: dict[str, Any] = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
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

    uninterrupted_prediction = uninterrupted.execution_model.predict(dataset.batch)
    resumed_prediction = resumed.execution_model.predict(dataset.batch)
    assert first.progress.update_step == 1
    assert resumed.resumed_from_step == 1
    assert resumed.progress.update_step == 2
    assert tuple(resumed_prediction.fields) == ("vertex", "edge")
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


def _source_matching_program(*, identity="tests.cochain.source_matching"):
    zero_spec = phx.graph.CochainFieldSpec(
        0,
        cell_orientation="invariant",
        sampling="point_value",
    )

    def residual(graph, fields, *, key):
        del graph, key
        return {"residual": fields["u"] - 0.1 * fields["forcing"]}

    return phx.graph.CochainResidualProgram(
        inputs={"u": zero_spec, "forcing": zero_spec},
        outputs={"residual": zero_spec},
        residual_fn=residual,
        identity=identity,
    )


def _source_matching_loss(
    *,
    identity="tests.cochain.source_matching",
    topology_fingerprint=None,
):
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


def _targetless_dataset(*, cases=2):
    batch = _batch(cases=cases)
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays({}, batch)
    return phx.nn.operator.training.OperatorDataset(batch, targets)


def _small_cochain_model(*, key):
    return phx.nn.operator.architectures.CochainNeuralOperator(
        _fields(),
        width=3,
        depth=1,
        key=key,
    )


def _physics_loss_value(term, model, dataset):
    prediction = model.predict(dataset.batch)
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


def test_cochain_residual_loss_scatters_sparse_fields_and_locks_topology():
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


def test_targetless_cochain_pino_update_and_checkpoint_resume_are_exact(tmp_path):
    dataset = _targetless_dataset(cases=2)
    model = _small_cochain_model(key=jr.key(31))
    term = _source_matching_loss()
    common: dict[str, Any] = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
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
    uninterrupted_prediction = uninterrupted.execution_model.predict(dataset.batch)
    resumed_prediction = resumed.execution_model.predict(dataset.batch)
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


def test_targetless_operator_fit_requires_explicit_physics_and_scaling():
    dataset = _targetless_dataset(cases=2)
    model = _small_cochain_model(key=jr.key(32))
    common: dict[str, Any] = {
        "task": _task(),
        "training_evidence": phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
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
