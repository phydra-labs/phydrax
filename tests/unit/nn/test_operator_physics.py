#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.operator import AbstractOperatorModel


def _axis(size=5):
    return phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, size),
        quadrature_weights=jnp.asarray([0.1, 0.2, 0.4, 0.2, 0.1]),
    )


def _prediction(values, query, *, name="output", channels="scalar"):
    return phx.nn.operator.OperatorPrediction.from_field(
        name,
        values,
        "query",
        query,
        spec=phx.nn.operator.OperatorOutputSpec(channels),
        case_axes=("case",),
        case_shape=(2,),
    )


def test_hilbert_metrics_are_complex_measure_and_mask_aware():
    axis = _axis()
    query = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(axis,),
        mask=jnp.asarray([True, True, True, True, False]),
    )
    values = jnp.asarray(
        [
            [1.0 + 1.0j, 2.0j, -1.0, 0.5, 20.0],
            [2.0, 1.0 - 1.0j, 0.5j, -2.0, 30.0],
        ]
    )
    expected_energy = jnp.sum(
        jnp.abs(values[:, :4]) ** 2 * jnp.asarray([0.1, 0.2, 0.4, 0.2]),
        axis=1,
    )
    energy = phx.nn.operator.training.operator_hilbert_norm(
        values,
        query,
        case_shape=(2,),
        squared=True,
    )
    assert jnp.allclose(energy, expected_energy)
    assert jnp.allclose(
        phx.nn.operator.training.operator_hilbert_relative_error(
            2.0 * values,
            values,
            query,
            case_shape=(2,),
        ),
        jnp.ones((2,)),
    )


def test_physical_quadrature_predicate_accepts_tensor_and_case_shaped_measures():
    tensor = phx.nn.operator.FunctionSamples(values=None, axes=(_axis(),))
    coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 5)[None, :, None],
        (2, 5, 1),
    )
    point_cloud = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=jnp.asarray(
            [[0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.1, 0.3, 0.1, 0.3]]
        ),
    )

    assert tensor.has_physical_quadrature
    assert point_cloud.has_physical_quadrature
    explicit_grid = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(
            phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 2)),
            phx.nn.operator.OperatorAxis("y", jnp.linspace(-1.0, 1.0, 3)),
        ),
        quadrature_weights=jnp.full((2, 3), 1.0 / 6.0),
    )
    assert explicit_grid.has_physical_quadrature


def test_conservation_projection_is_exact_and_differentiable():
    axis = _axis()
    query = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(axis,),
        mask=jnp.asarray([True, True, True, True, False]),
    )
    values = jnp.arange(20.0).reshape((2, 5, 2))
    target = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])

    projected = phx.nn.operator.training.project_operator_conservation(
        values,
        query,
        target,
        case_shape=(2,),
    )
    assert jnp.allclose(
        phx.nn.operator.training.operator_integral(projected, query, case_shape=(2,)),
        target,
    )
    assert jnp.all(projected[:, -1] == 0.0)

    def objective(raw):
        constrained = phx.nn.operator.training.project_operator_conservation(
            raw,
            query,
            target,
            case_shape=(2,),
        )
        return jnp.sum(constrained**2)

    gradient = jax.grad(objective)(values)
    assert gradient.shape == values.shape
    assert jnp.all(jnp.isfinite(gradient))


def test_output_pipeline_enforces_lift_and_boundary_envelope():
    axis = _axis()
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 5)), axes=(axis,)
            )
        },
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )
    raw = _prediction(jnp.full((2, 5), 7.0), query)
    pipeline = phx.nn.operator.training.OperatorOutputPipeline(
        phx.nn.operator.training.HardConstraintTransform(
            "output",
            envelope_fn=lambda coordinates, batch, *, key: (
                coordinates[..., 0] * (1.0 - coordinates[..., 0])
            ),
            identity="unit-interval-dirichlet",
            lift_fn=lambda coordinates, batch, *, key: coordinates[..., 0],
        ),
        phx.nn.operator.training.ConservationProjection(
            "output",
            source_name="source",
            correction_fn=lambda coordinates, batch, *, key: (
                coordinates[..., 0] * (1.0 - coordinates[..., 0])
            ),
            identity="dirichlet-compatible-mass",
        ),
    )
    transformed = pipeline(raw, batch, key=jr.key(0))
    values = transformed.field("output").values
    assert jnp.allclose(values[:, 0], 0.0)
    assert jnp.allclose(values[:, -1], 1.0)
    assert jnp.allclose(
        phx.nn.operator.training.operator_integral(values, query, case_shape=(2,)),
        jnp.ones((2,)),
    )
    assert transformed.case_axes == raw.case_axes
    assert transformed.field("output").query_name == "query"


def test_weak_form_loss_detects_and_normalizes_test_moments():
    axis = _axis()
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    x = axis.nodes
    residual = jnp.broadcast_to(x - 0.5, (2, 5))
    constant_test = jnp.ones((5, 1))
    scaled_test = 9.0 * constant_test
    assert (
        phx.nn.operator.training.operator_weak_form_loss(
            residual,
            constant_test,
            query,
            case_shape=(2,),
        )
        < 1e-28
    )
    assert jnp.allclose(
        phx.nn.operator.training.operator_weak_form_loss(
            residual + 1.0,
            constant_test,
            query,
            case_shape=(2,),
        ),
        phx.nn.operator.training.operator_weak_form_loss(
            residual + 1.0,
            scaled_test,
            query,
            case_shape=(2,),
        ),
    )


def test_dynamic_weak_loss_selects_physical_integration_measure():
    execution_axis = _axis()
    physical_axis = phx.nn.operator.OperatorAxis(
        "x",
        execution_axis.nodes,
        quadrature_weights=2.0 * execution_axis.quadrature_weights,
    )
    execution_batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 5)),
                axes=(execution_axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(execution_axis,))
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    physical_batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 5)),
                axes=(physical_axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(physical_axis,))
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {"output": jnp.zeros((2, 5))},
        execution_batch,
    )
    physical_targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {"output": jnp.zeros((2, 5))},
        physical_batch,
    )
    prediction = _prediction(jnp.zeros((2, 5)), execution_batch.query("query"))
    term = phx.nn.operator.training.WeakOperatorLoss(
        "weak_constant",
        residual_fn=lambda prediction, batch, targets, **kwargs: jnp.ones((2, 5)),
        test_fn=lambda batch, **kwargs: jnp.ones((5, 1)),
        identity="constant-residual",
        space="physical",
    )
    value = term(
        None,
        prediction,
        execution_batch,
        targets,
        key=jr.key(0),
        step=jnp.asarray(0),
        training=True,
        context=phx.nn.operator.training.OperatorLossContext(
            execution_prediction=prediction,
            execution_batch=execution_batch,
            execution_targets=targets,
            physical_prediction=_prediction(
                jnp.zeros((2, 5)),
                physical_batch.query("query"),
            ),
            physical_batch=physical_batch,
            physical_targets=physical_targets,
        ),
    )
    assert jnp.allclose(value, 2.0)
    assert (
        term.fingerprint
        != phx.nn.operator.training.WeakOperatorLoss(
            "weak_constant",
            residual_fn=lambda prediction, batch, targets, **kwargs: jnp.ones((2, 5)),
            test_fn=lambda batch, **kwargs: jnp.ones((5, 1)),
            identity="constant-residual",
            space="execution",
        ).fingerprint
    )


class _NonlinearPointwiseOperator(AbstractOperatorModel):
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("source").values
        assert values is not None
        return values**2 + 2.0 * values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")


class _ComplexPointwiseOperator(AbstractOperatorModel):
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = "scalar"
        self.out_size = "scalar"

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("source").values
        assert values is not None
        return (1.0 + 2.0j) * values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def test_trained_operator_applies_conservation_inside_physical_prediction():
    axis = _axis()
    source = jnp.linspace(0.2, 1.1, 10).reshape((2, 5))
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": phx.nn.operator.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    task = phx.nn.operator.OperatorTask(
        "conservative-pointwise",
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "source", role="source", source_name="source"
            ),
            phx.nn.operator.OperatorFieldSpec(
                "output", role="target", query_name="query"
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=(phx.units.LENGTH,),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )
    output_port = task.field_by_name["output"].value_port()
    model = phx.nn.operator.training.TrainedOperator(
        _NonlinearPointwiseOperator(),
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_ports={"output": output_port},
        port_mapping=phx.PortMapping(
            outputs=((output_port.port_id, output_port.port_id),)
        ),
        output_pipeline=phx.nn.operator.training.OperatorOutputPipeline(
            phx.nn.operator.training.ConservationProjection(
                "output", source_name="source"
            )
        ),
    )
    prediction = model.predict(batch, key=jr.key(2))
    assert jnp.allclose(
        phx.nn.operator.training.operator_integral(
            prediction.field("output").values,
            batch.query("query"),
            case_shape=batch.case_shape,
        ),
        phx.nn.operator.training.operator_integral(
            source,
            batch.input("source"),
            case_shape=batch.case_shape,
        ),
    )


def test_matrix_free_linearization_satisfies_weighted_adjoint_identity():
    axis = _axis()
    source = jnp.linspace(-0.4, 0.8, 10).reshape((2, 5))
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": phx.nn.operator.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    linearization = phx.nn.operator.training.linearize_operator(
        _NonlinearPointwiseOperator(),
        batch,
        "source",
    )
    tangent = jnp.sin(jnp.arange(10.0)).reshape((2, 5))
    cotangent = jnp.cos(jnp.arange(10.0)).reshape((2, 5))
    expected = (2.0 * source + 2.0) * tangent
    assert jnp.allclose(linearization.pushforward(tangent), expected)
    assert jnp.max(linearization.adjoint_identity_error(tangent, cotangent)) < 1e-12


def test_complex_operator_adjoint_is_hermitian():
    axis = _axis()
    source = (jnp.linspace(0.1, 1.0, 10) + 1.0j * jnp.linspace(-0.5, 0.4, 10)).reshape(
        (2, 5)
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": phx.nn.operator.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    linearization = phx.nn.operator.training.linearize_operator(
        _ComplexPointwiseOperator(),
        batch,
        "source",
    )
    cotangent = (jnp.cos(jnp.arange(10.0)) + 1.0j * jnp.sin(jnp.arange(10.0))).reshape(
        (2, 5)
    )
    assert jnp.allclose(
        linearization.adjoint(cotangent),
        (1.0 - 2.0j) * cotangent,
    )


def test_trained_operator_linearization_uses_physical_units():
    axis = _axis()
    source = jnp.linspace(1.0, 3.0, 10).reshape((2, 5))
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": phx.nn.operator.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    task = phx.nn.operator.OperatorTask(
        "scaled-pointwise-map",
        revision="1",
        dimension_basis=("value",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="source",
                dimension=phx.units.DimensionSignature({"value": 1}),
                scale=2.0,
                offset=1.0,
            ),
            phx.nn.operator.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="query",
                dimension=phx.units.DimensionSignature({"value": 1}),
                scale=3.0,
                offset=4.0,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=(phx.units.DimensionSignature({"value": 1}),),
            ),
        ),
    )
    solution_port = task.field_by_name["solution"].value_port()
    trained = phx.nn.operator.training.TrainedOperator(
        _NonlinearPointwiseOperator(),
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        output_ports={"output": solution_port},
        port_mapping=phx.PortMapping(
            outputs=((solution_port.port_id, solution_port.port_id),)
        ),
    )
    linearization = phx.nn.operator.training.linearize_operator(
        trained,
        batch,
        "source",
        field_name="solution",
    )
    tangent = jnp.cos(jnp.arange(10.0)).reshape((2, 5))
    expected = 1.5 * (source + 1.0) * tangent
    assert jnp.allclose(linearization.pushforward(tangent), expected)
    cotangent = jnp.sin(jnp.arange(10.0)).reshape((2, 5))
    physical_derivative = 1.5 * (source + 1.0)
    assert jnp.allclose(
        linearization.adjoint(cotangent),
        physical_derivative * cotangent,
    )
    tolerance = 8.0 * jnp.finfo(linearization.base_output.dtype).eps
    assert jnp.max(linearization.adjoint_identity_error(tangent, cotangent)) < tolerance


def _predict_two_fields(model, batch, key):
    del model, key
    query = batch.query("points")
    coordinates = query.coordinates_array(case_shape=batch.case_shape)
    radius_squared = jnp.sum(coordinates**2, axis=-1)
    product = coordinates[..., 0] * coordinates[..., 1]
    return phx.nn.operator.OperatorPrediction(
        {
            "radius_squared": phx.nn.operator.OperatorFieldBatch(
                radius_squared,
                query_name="points",
                spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            ),
            "product": phx.nn.operator.OperatorFieldBatch(
                product,
                query_name="points",
                spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            ),
        },
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


class _TwoFieldPointOperator(AbstractOperatorModel):
    _operator_prediction_builder: ClassVar = staticmethod(_predict_two_fields)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 1
        self.out_size = 1

    @property
    def operator_output_specs(self):
        spec = phx.nn.operator.OperatorOutputSpec("scalar")
        return {"radius_squared": spec, "product": spec}

    def __call_operator_batch__(self, batch, *, key=None):
        return _predict_two_fields(self, batch, key).field("radius_squared").values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def test_operator_context_supports_multiple_coordinates_queries_and_outputs():
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.asarray([[0.0, 0.0]]),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 1)),
                coordinates=jnp.asarray([[0.0, 0.0]]),
            )
        },
        queries={
            "points": query,
            "unused": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.asarray([[1.0, 1.0]]),
            ),
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    context = phx.nn.operator.adapters.bind_operator_context(
        _TwoFieldPointOperator(),
        batch,
        query_name="points",
        field_name="radius_squared",
    )
    domain = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )
    function = context.domain_function(domain, "x")
    laplacian = phx.operators.laplacian(function, var="x")
    point = jnp.asarray([0.2, -0.3])
    assert jnp.allclose(function.func(point), jnp.full((2,), 0.13))
    assert jnp.allclose(laplacian.func(point), jnp.full((2,), 4.0))


def _predict_split_fields(model, batch, key):
    del model, key
    coordinates = batch.query("query").coordinates_array(case_shape=batch.case_shape)
    x = coordinates[..., 0]
    spec = phx.nn.operator.OperatorOutputSpec("scalar")
    return phx.nn.operator.OperatorPrediction(
        {
            "first": phx.nn.operator.OperatorFieldBatch(
                x**2, query_name="query", spec=spec
            ),
            "second": phx.nn.operator.OperatorFieldBatch(
                2.0 * x, query_name="query", spec=spec
            ),
        },
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


class _SplitQueryOperator(AbstractOperatorModel):
    _operator_prediction_builder: ClassVar = staticmethod(_predict_split_fields)
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_output_specs(self):
        spec = phx.nn.operator.OperatorOutputSpec("scalar")
        return {"first": spec, "second": spec}

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        return _predict_split_fields(self, batch, key).field("first").values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _split_task():
    value = phx.units.DimensionSignature({"value": 1})
    return phx.nn.operator.OperatorTask(
        "split-query-map",
        dimension_basis=("length", "value"),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "source", role="source", source_name="source"
            ),
            phx.nn.operator.OperatorFieldSpec(
                "left", role="target", query_name="query", dimension=value
            ),
            phx.nn.operator.OperatorFieldSpec(
                "right",
                role="target",
                query_name="query",
                dimension=value,
                scale=10.0,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=(phx.units.LENGTH,),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )


def _split_batch():
    axis = _axis()
    return phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, 5)), axes=(axis,)
            )
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )


def _split_trained(first_port, second_port, mapping):
    return phx.nn.operator.training.TrainedOperator(
        _SplitQueryOperator(),
        _split_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_ports={"first": first_port, "second": second_port},
        port_mapping=phx.PortMapping(outputs=mapping),
    )


def _split_ports():
    task = _split_task()
    return task.field_by_name["left"].value_port(), task.field_by_name[
        "right"
    ].value_port()


def test_trained_operator_routes_model_outputs_by_bound_port_ids():
    left, right = _split_ports()
    # Model output names never match task fields: only the port binding routes.
    trained = _split_trained(
        right,
        left,
        ((right.port_id, right.port_id), (left.port_id, left.port_id)),
    )
    x = jnp.linspace(0.0, 1.0, 5)
    prediction = trained.predict(_split_batch())

    assert jnp.allclose(prediction.field("right").values, 10.0 * x**2)
    assert jnp.allclose(prediction.field("left").values, 2.0 * x)
    assert trained.port_binding.outputs == (
        (right.port_id, right.port_id),
        (left.port_id, left.port_id),
    )
    assert trained.port_binding.dimensions_verified
    assert trained.port_binding.normalizations_verified
    assert tuple(port.port_id for port in trained.model_ports().outputs) == (
        left.port_id,
        right.port_id,
    )


def test_trained_operator_binding_requires_explicit_compatible_ports():
    left, right = _split_ports()
    task = _split_task()
    batch = _split_batch()
    evidence = phx.nn.operator.OperatorTrainingEvidence("task_specific")
    with pytest.raises(ValueError, match="explicit output_ports and a port_mapping"):
        phx.nn.operator.training.TrainedOperator(
            _SplitQueryOperator(),
            task,
            training_evidence=evidence,
            output_ports={"first": right, "second": left},
        )
    dataset = phx.nn.operator.training.OperatorDataset(
        batch,
        phx.nn.operator.OperatorTargetBatch.from_arrays(
            {"left": jnp.zeros((2, 5)), "right": jnp.zeros((2, 5))}, batch
        ),
    )
    with pytest.raises(ValueError, match="explicit output_ports and a port_mapping"):
        phx.nn.operator.training.fit_operator(
            _SplitQueryOperator(),
            dataset,
            task=task,
            training_evidence=evidence,
            output_ports={"first": right, "second": left},
        )

    length = phx.ValuePort(
        "right",
        event_shape=(),
        component_ids=("right",),
        representation="scalar",
        dimensions=(phx.units.LENGTH,),
        normalization_id=right.normalization_id,
    )
    with pytest.raises(
        ValueError,
        match=rf"output port pair 'right' \({length.port_id} -> {right.port_id}\) "
        "dimensions mismatch",
    ):
        _split_trained(
            length,
            left,
            ((length.port_id, right.port_id), (left.port_id, left.port_id)),
        )
    channels = phx.ValuePort(
        "right",
        event_shape=(2,),
        component_ids=("right[0]", "right[1]"),
        representation="scalar",
    )
    with pytest.raises(ValueError, match="output port pair 'right' .*component_ids"):
        _split_trained(
            channels,
            left,
            ((channels.port_id, right.port_id), (left.port_id, left.port_id)),
        )
    source = task.field_by_name["source"].value_port()
    with pytest.raises(ValueError, match="unknown owner output ports"):
        _split_trained(
            right,
            left,
            ((right.port_id, source.port_id), (left.port_id, left.port_id)),
        )


def test_port_binding_records_aspects_a_side_leaves_undeclared():
    left, right = _split_ports()
    bare = phx.ValuePort(
        "right", event_shape=(), component_ids=("right",), representation="scalar"
    )
    relaxed = _split_trained(
        bare, left, ((bare.port_id, right.port_id), (left.port_id, left.port_id))
    )
    exact = _split_trained(
        right, left, ((right.port_id, right.port_id), (left.port_id, left.port_id))
    )

    unverified = relaxed.port_binding.unverified
    assert ("output", bare.port_id, "dimensions") in unverified
    assert ("output", bare.port_id, "normalization") in unverified
    assert ("output", left.port_id, "dimensions") not in unverified
    assert not relaxed.port_binding.dimensions_verified
    assert exact.port_binding.dimensions_verified
    assert relaxed.contract_fingerprint != exact.contract_fingerprint


def test_trained_operator_context_selects_query_and_field_through_ports():
    left, right = _split_ports()
    trained = _split_trained(
        right, left, ((right.port_id, right.port_id), (left.port_id, left.port_id))
    )
    query = _split_task().query_by_name["query"].value_port()
    mapping = phx.PortMapping(
        inputs=((query.port_id, query.port_id),),
        outputs=((right.port_id, right.port_id),),
    )
    owner = phx.ModelPorts(inputs=(query,), outputs=(right,))
    context = phx.nn.operator.adapters.bind_operator_context(
        trained, _split_batch(), port_mapping=mapping, owner_ports=owner
    )
    points = jnp.asarray([[0.3], [0.6]])

    assert jnp.allclose(context(points), 10.0 * points[:, 0] ** 2)
    assert context.port_binding.outputs == ((right.port_id, right.port_id),)
    assert context.port_binding.inputs == ((query.port_id, query.port_id),)
    with pytest.raises(ValueError, match="explicit port_mapping"):
        phx.nn.operator.adapters.bind_operator_context(trained, _split_batch())
    with pytest.raises(ValueError, match="through port_mapping"):
        phx.nn.operator.adapters.bind_operator_context(
            trained,
            _split_batch(),
            field_name="right",
            port_mapping=mapping,
            owner_ports=owner,
        )
    with pytest.raises(ValueError, match="not a target field port"):
        phx.nn.operator.adapters.bind_operator_context(
            trained,
            _split_batch(),
            port_mapping=phx.PortMapping(
                inputs=((query.port_id, query.port_id),),
                outputs=((query.port_id, right.port_id),),
            ),
            owner_ports=owner,
        )
    with pytest.raises(ValueError, match="declares no model ports"):
        phx.nn.operator.adapters.bind_operator_context(
            _SplitQueryOperator(),
            _split_batch(),
            field_name="first",
            port_mapping=mapping,
            owner_ports=owner,
        )
