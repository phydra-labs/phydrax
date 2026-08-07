#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx
from phydrax.nn.models.core._base import _AbstractOperatorModel


def _axis(size=5):
    return phx.nn.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, size),
        quadrature_weights=jnp.asarray([0.1, 0.2, 0.4, 0.2, 0.1]),
    )


def _prediction(values, query, *, name="output", channels="scalar"):
    return phx.nn.OperatorPrediction.from_field(
        name,
        values,
        "query",
        query,
        spec=phx.nn.OperatorOutputSpec(channels),
        case_axes=("case",),
        case_shape=(2,),
    )


def test_hilbert_metrics_are_complex_measure_and_mask_aware():
    axis = _axis()
    query = phx.nn.FunctionSamples(
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
    energy = phx.nn.operator_hilbert_norm(
        values,
        query,
        case_shape=(2,),
        squared=True,
    )
    assert jnp.allclose(energy, expected_energy)
    assert jnp.allclose(
        phx.nn.operator_hilbert_relative_error(
            2.0 * values,
            values,
            query,
            case_shape=(2,),
        ),
        jnp.ones((2,)),
    )


def test_physical_quadrature_predicate_accepts_tensor_and_case_shaped_measures():
    tensor = phx.nn.FunctionSamples(values=None, axes=(_axis(),))
    coordinates = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0, 5)[None, :, None],
        (2, 5, 1),
    )
    point_cloud = phx.nn.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=jnp.asarray(
            [[0.1, 0.2, 0.4, 0.2, 0.1], [0.2, 0.1, 0.3, 0.1, 0.3]]
        ),
    )

    assert tensor.has_physical_quadrature
    assert point_cloud.has_physical_quadrature


def test_conservation_projection_is_exact_and_differentiable():
    axis = _axis()
    query = phx.nn.FunctionSamples(
        values=None,
        axes=(axis,),
        mask=jnp.asarray([True, True, True, True, False]),
    )
    values = jnp.arange(20.0).reshape((2, 5, 2))
    target = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])

    projected = phx.nn.project_operator_conservation(
        values,
        query,
        target,
        case_shape=(2,),
    )
    assert jnp.allclose(
        phx.nn.operator_integral(projected, query, case_shape=(2,)),
        target,
    )
    assert jnp.all(projected[:, -1] == 0.0)

    def objective(raw):
        constrained = phx.nn.project_operator_conservation(
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
    query = phx.nn.FunctionSamples(values=None, axes=(axis,))
    batch = phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=jnp.ones((2, 5)), axes=(axis,))},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )
    raw = _prediction(jnp.full((2, 5), 7.0), query)
    pipeline = phx.nn.OperatorOutputPipeline(
        phx.nn.HardConstraintTransform(
            "output",
            envelope_fn=lambda coordinates, batch, *, key: (
                coordinates[..., 0] * (1.0 - coordinates[..., 0])
            ),
            identity="unit-interval-dirichlet-v1",
            lift_fn=lambda coordinates, batch, *, key: coordinates[..., 0],
        ),
        phx.nn.ConservationProjection(
            "output",
            source_name="source",
            correction_fn=lambda coordinates, batch, *, key: (
                coordinates[..., 0] * (1.0 - coordinates[..., 0])
            ),
            identity="dirichlet-compatible-mass-v1",
        ),
    )
    transformed = pipeline(raw, batch, key=jr.key(0))
    values = transformed.field("output").values
    assert jnp.allclose(values[:, 0], 0.0)
    assert jnp.allclose(values[:, -1], 1.0)
    assert jnp.allclose(
        phx.nn.operator_integral(values, query, case_shape=(2,)),
        jnp.ones((2,)),
    )
    assert transformed.case_axes == raw.case_axes
    assert transformed.field("output").query_name == "query"


def test_weak_form_loss_detects_and_normalizes_test_moments():
    axis = _axis()
    query = phx.nn.FunctionSamples(values=None, axes=(axis,))
    x = axis.nodes
    residual = jnp.broadcast_to(x - 0.5, (2, 5))
    constant_test = jnp.ones((5, 1))
    scaled_test = 9.0 * constant_test
    assert phx.nn.operator_weak_form_loss(
        residual,
        constant_test,
        query,
        case_shape=(2,),
    ) < 1e-28
    assert jnp.allclose(
        phx.nn.operator_weak_form_loss(
            residual + 1.0,
            constant_test,
            query,
            case_shape=(2,),
        ),
        phx.nn.operator_weak_form_loss(
            residual + 1.0,
            scaled_test,
            query,
            case_shape=(2,),
        ),
    )

def test_dynamic_weak_loss_selects_physical_integration_measure():
    execution_axis = _axis()
    physical_axis = phx.nn.OperatorAxis(
        "x",
        execution_axis.nodes,
        quadrature_weights=2.0 * execution_axis.quadrature_weights,
    )
    execution_batch = phx.nn.OperatorBatch(
        inputs={
            "source": phx.nn.FunctionSamples(
                values=jnp.ones((2, 5)),
                axes=(execution_axis,),
            )
        },
        queries={
            "query": phx.nn.FunctionSamples(values=None, axes=(execution_axis,))
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    physical_batch = phx.nn.OperatorBatch(
        inputs={
            "source": phx.nn.FunctionSamples(
                values=jnp.ones((2, 5)),
                axes=(physical_axis,),
            )
        },
        queries={
            "query": phx.nn.FunctionSamples(values=None, axes=(physical_axis,))
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    targets = phx.nn.OperatorTargetBatch.from_arrays(
        {"output": jnp.zeros((2, 5))},
        execution_batch,
    )
    physical_targets = phx.nn.OperatorTargetBatch.from_arrays(
        {"output": jnp.zeros((2, 5))},
        physical_batch,
    )
    prediction = _prediction(jnp.zeros((2, 5)), execution_batch.query("query"))
    term = phx.nn.WeakOperatorLoss(
        "weak_constant",
        residual_fn=lambda prediction, batch, targets, **kwargs: jnp.ones((2, 5)),
        test_fn=lambda batch, **kwargs: jnp.ones((5, 1)),
        identity="constant-residual-v1",
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
        context=phx.nn.OperatorLossContext(
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
    assert term.fingerprint != phx.nn.WeakOperatorLoss(
        "weak_constant",
        residual_fn=lambda prediction, batch, targets, **kwargs: jnp.ones((2, 5)),
        test_fn=lambda batch, **kwargs: jnp.ones((5, 1)),
        identity="constant-residual-v1",
        space="execution",
    ).fingerprint



class _NonlinearPointwiseOperator(_AbstractOperatorModel):
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
        return phx.nn.operator_architecture_contract("DeepONet")



class _ComplexPointwiseOperator(_AbstractOperatorModel):
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
    batch = phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    task = phx.nn.OperatorTask(
        "conservative-pointwise",
        dimension_basis=("length",),
        fields=(
            phx.nn.OperatorFieldSpec("source", role="source", source_name="source"),
            phx.nn.OperatorFieldSpec("output", role="target", query_name="query"),
        ),
        queries=(
            phx.nn.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            ),
        ),
        problem=phx.nn.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )
    model = phx.nn.TrainedOperator(
        _NonlinearPointwiseOperator(),
        task,
        training_evidence=phx.nn.OperatorTrainingEvidence("task_specific"),
        output_pipeline=phx.nn.OperatorOutputPipeline(
            phx.nn.ConservationProjection("output", source_name="source")
        ),
    )
    prediction = model.predict(batch, key=jr.key(2))
    assert jnp.allclose(
        phx.nn.operator_integral(
            prediction.field("output").values,
            batch.query("query"),
            case_shape=batch.case_shape,
        ),
        phx.nn.operator_integral(
            source,
            batch.input("source"),
            case_shape=batch.case_shape,
        ),
    )


def test_matrix_free_linearization_satisfies_weighted_adjoint_identity():
    axis = _axis()
    source = jnp.linspace(-0.4, 0.8, 10).reshape((2, 5))
    batch = phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    linearization = phx.nn.linearize_operator(
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
    source = (
        jnp.linspace(0.1, 1.0, 10) + 1.0j * jnp.linspace(-0.5, 0.4, 10)
    ).reshape((2, 5))
    batch = phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    linearization = phx.nn.linearize_operator(
        _ComplexPointwiseOperator(),
        batch,
        "source",
    )
    cotangent = (
        jnp.cos(jnp.arange(10.0)) + 1.0j * jnp.sin(jnp.arange(10.0))
    ).reshape((2, 5))
    assert jnp.allclose(
        linearization.adjoint(cotangent),
        (1.0 - 2.0j) * cotangent,
    )


def test_trained_operator_linearization_uses_physical_units():
    axis = _axis()
    source = jnp.linspace(1.0, 3.0, 10).reshape((2, 5))
    batch = phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=source, axes=(axis,))},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=(axis,))},
        case_axes=("case",),
        case_shape=(2,),
    )
    task = phx.nn.OperatorTask(
        "scaled-pointwise-map",
        revision="1",
        dimension_basis=("value",),
        fields=(
            phx.nn.OperatorFieldSpec(
                "input",
                role="source",
                source_name="source",
                physical_dimension=(1.0,),
                scale=2.0,
                offset=1.0,
            ),
            phx.nn.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="query",
                physical_dimension=(1.0,),
                scale=3.0,
                offset=4.0,
            ),
        ),
        queries=(
            phx.nn.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=((1.0,),),
            ),
        ),
    )
    trained = phx.nn.TrainedOperator(
        _NonlinearPointwiseOperator(),
        task,
        training_evidence=phx.nn.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        output_field_map={"output": "solution"},
    )
    linearization = phx.nn.linearize_operator(
        trained,
        batch,
        "source",
        field_name="solution",
    )
    tangent = jnp.cos(jnp.arange(10.0)).reshape((2, 5))
    expected = 1.5 * (source + 1.0) * tangent
    assert jnp.allclose(linearization.pushforward(tangent), expected)


def _predict_two_fields(model, batch, key):
    del model, key
    query = batch.query("points")
    coordinates = query.coordinates_array(case_shape=batch.case_shape)
    radius_squared = jnp.sum(coordinates**2, axis=-1)
    product = coordinates[..., 0] * coordinates[..., 1]
    return phx.nn.OperatorPrediction(
        {
            "radius_squared": phx.nn.OperatorFieldBatch(
                radius_squared,
                query_name="points",
                spec=phx.nn.OperatorOutputSpec("scalar"),
            ),
            "product": phx.nn.OperatorFieldBatch(
                product,
                query_name="points",
                spec=phx.nn.OperatorOutputSpec("scalar"),
            ),
        },
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


class _TwoFieldPointOperator(_AbstractOperatorModel):
    _operator_prediction_builder: ClassVar = staticmethod(_predict_two_fields)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 1
        self.out_size = 1

    @property
    def operator_output_specs(self):
        spec = phx.nn.OperatorOutputSpec("scalar")
        return {"radius_squared": spec, "product": spec}

    def __call_operator_batch__(self, batch, *, key=None):
        return _predict_two_fields(self, batch, key).field("radius_squared").values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def test_operator_context_supports_multiple_coordinates_queries_and_outputs():
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=jnp.asarray([[0.0, 0.0]]),
    )
    batch = phx.nn.OperatorBatch(
        inputs={"source": phx.nn.FunctionSamples(values=jnp.ones((2, 1)), coordinates=jnp.asarray([[0.0, 0.0]]))},
        queries={
            "points": query,
            "unused": phx.nn.FunctionSamples(
                values=None,
                coordinates=jnp.asarray([[1.0, 1.0]]),
            ),
        },
        case_axes=("case",),
        case_shape=(2,),
    )
    context = phx.nn.bind_operator_context(
        _TwoFieldPointOperator(),
        batch,
        query_name="points",
        field_name="radius_squared",
    )
    domain = phx.domain.GeometryDomain(phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile())
    function = context.domain_function(domain, "x")
    laplacian = phx.operators.laplacian(function, var="x")
    point = jnp.asarray([0.2, -0.3])
    assert jnp.allclose(function.func(point), jnp.full((2,), 0.13))
    assert jnp.allclose(laplacian.func(point), jnp.full((2,), 4.0))
