#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _tensor_batch(*, cases: int = 2) -> phx.nn.operator.OperatorBatch:
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 3),
        quadrature_weights=jnp.asarray([0.25, 0.5, 0.25]),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.arange(cases * 3, dtype=float).reshape(cases, 3),
        axes=(axis,),
    )
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    return phx.nn.operator.OperatorBatch(inputs={"source": source}, queries={"query": query}, case_axes=("case",),)


def _point_batch(*, shifted: float = 0.0) -> phx.nn.operator.OperatorBatch:
    coordinates = (
        jnp.asarray(
            [
                [[0.0], [0.5], [1.0]],
                [[0.0], [0.4], [1.0]],
            ]
        )
        + shifted
    )
    mask = jnp.asarray([[True, True, False], [True, True, True]])
    weights = jnp.asarray([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25]])
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, 3)),
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    return phx.nn.operator.OperatorBatch(inputs={"source": source}, queries={"query": query}, case_axes=("case",),)


def _multi_output_prediction():
    spatial = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 3)),),
    )
    sensors = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.asarray([[0.2], [0.8]]),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, 3)),
        axes=spatial.axes,
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": source},
        queries={"spatial": spatial, "sensors": sensors},
        case_axes=("draw",),
        case_shape=(2,),
    )
    prediction = phx.nn.operator.OperatorPrediction(
        {
            "state": phx.nn.operator.OperatorFieldBatch(
                jnp.arange(6.0).reshape(2, 3),
                query_name="spatial",
                spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            ),
            "flux": phx.nn.operator.OperatorFieldBatch(
                jnp.arange(8.0).reshape(2, 2, 2),
                query_name="sensors",
                spec=phx.nn.operator.OperatorOutputSpec(
                    2,
                    component_names=("x", "y"),
                ),
            ),
        },
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    return batch, prediction


def test_multi_output_prediction_selection_propagates_through_uq():
    batch, target = _multi_output_prediction()
    with pytest.raises(KeyError, match="Unknown operator output field"):
        phx.uq.operator_prediction_field(target, field_name="missing")
    flux = phx.uq.operator_prediction_field(target, field_name="flux")
    assert flux.dims == (
        "draw",
        "__phydra_operator_point",
        "__phydra_operator_channel",
    )

    values = target.field("flux").values
    predictive = phx.uq.operator_predictive_from_samples(
        jnp.stack((values - 1.0, values + 1.0)),
        batch,
        target.field("flux").spec,
        sample_axes=(phx.uq.SampleAxis("member", "epistemic"),),
        field_name="flux",
        query_name="sensors",
    )
    mean = predictive.mean()
    assert tuple(mean.fields) == ("flux",)
    assert tuple(mean.queries) == ("sensors",)
    assert jnp.allclose(mean.field("flux").values, values)
    assert jnp.isfinite(phx.uq.operator_ensemble_crps(predictive, target))

    input_predictive = phx.uq.operator_input_predictive(
        target,
        input_sample_axes=("draw",),
        field_name="flux",
    )
    assert input_predictive.field_name == "flux"
    assert input_predictive.query_name == "sensors"
    assert tuple(input_predictive.mean().fields) == ("flux",)


def test_operator_predictive_tensor_geometry_and_statistics():
    batch = _tensor_batch()
    values = jnp.arange(24.0).reshape(4, 2, 3)
    prediction = phx.uq.operator_predictive_from_samples(
        values,
        batch,
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis("draw", "epistemic"),),
        field_name="output",
        query_name="query",
    )

    assert prediction.predictive.samples.dims == ("draw", "case", "x")
    assert prediction.case_axes == ("case",)
    assert prediction.case_shape == (2,)
    assert prediction.output_mask().shape == (2, 3)
    assert jnp.allclose(
        prediction.output_weights(normalized=True).sum(axis=-1),
        1.0,
    )
    assert jnp.allclose(
        prediction.mean().field("output").values,
        values.mean(axis=0),
    )
    assert jnp.allclose(
        prediction.variance().field("output").values,
        values.var(axis=0),
    )
    assert jnp.allclose(
        prediction.std().field("output").values,
        values.std(axis=0),
    )
    assert jnp.allclose(
        prediction.quantile(0.5).field("output").values,
        jnp.median(values, axis=0),
    )

    interval = prediction.interval(0.25, 0.75)
    assert interval.lower.case_axes == ("case",)
    assert interval.nominal_coverage == pytest.approx(0.5)
    assert not interval.simultaneous
    assert not interval.calibrated


def test_operator_predictive_masks_padding_and_records_valid_draws():
    batch = _point_batch()
    spec = phx.nn.operator.OperatorOutputSpec(2, component_names=("u", "v"))
    values = jnp.ones((3, 2, 3, 2))
    values = values.at[:, 0, 2, :].set(jnp.nan)
    values = values.at[1, 1, 1, 0].set(jnp.nan)

    prediction = phx.uq.operator_predictive_from_samples(
        values,
        batch,
        spec,
        sample_axes=(phx.uq.SampleAxis("member", "epistemic"),),
        field_name="output",
        query_name="query",
    )

    assert prediction.predictive.samples.dims == (
        "member",
        "case",
        "__phydra_operator_point",
        "__phydra_operator_channel",
    )
    assert jnp.array_equal(
        prediction.predictive.valid.data,
        jnp.asarray([True, False, True]),
    )
    assert jnp.all(prediction.predictive.samples.data[:, 0, 2, :] == 0.0)
    mean = prediction.mean().field("output").values
    assert jnp.allclose(mean[prediction.output_mask()], 1.0)
    assert jnp.all(mean[~prediction.output_mask()] == 0.0)
    assert prediction.output_mask().shape == (2, 3, 2)

    with pytest.raises(FloatingPointError, match="invalid realizations"):
        phx.uq.operator_predictive_from_samples(
            values,
            batch,
            spec,
            sample_axes=(phx.uq.SampleAxis("member", "epistemic"),),
            field_name="output",
            query_name="query",
            valid_policy="raise",
        )


def test_operator_input_predictive_collapses_common_query_geometry():
    first = _point_batch()
    second = _point_batch()
    stacked = phx.nn.operator.stack_operator_batches(
        (first, second),
        case_axis="input_draw",
    )
    values = jnp.stack((jnp.ones((2, 3)), 3.0 * jnp.ones((2, 3))))
    deterministic = phx.nn.operator.OperatorPrediction.from_field(
        "output",
        values,
        "query",
        stacked.require_single_query(),
        spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=stacked.case_axes,
        case_shape=stacked.case_shape,
    )

    prediction = phx.uq.operator_input_predictive(
        deterministic,
        input_sample_axes=("input_draw",),
        field_name="output",
    )

    assert prediction.predictive.samples.dims == (
        "input_draw",
        "case",
        "__phydra_operator_point",
    )
    assert prediction.predictive.sample_axes[0].source == "input"
    assert prediction.case_axes == ("case",)
    assert prediction.case_shape == (2,)
    assert prediction.query.geometry_case_shape == (2,)
    mean = prediction.mean().field("output").values
    variance = prediction.variance().field("output").values
    assert jnp.allclose(mean[prediction.output_mask()], 2.0)
    assert jnp.allclose(variance[prediction.output_mask()], 1.0)
    assert jnp.all(mean[~prediction.output_mask()] == 0.0)
    assert jnp.all(variance[~prediction.output_mask()] == 0.0)


def test_operator_input_predictive_rejects_varying_output_queries():
    stacked = phx.nn.operator.stack_operator_batches(
        (_point_batch(), _point_batch(shifted=0.1)),
        case_axis="input_draw",
    )
    deterministic = phx.nn.operator.OperatorPrediction.from_field(
        "output",
        jnp.ones((2, 2, 3)),
        "query",
        stacked.require_single_query(),
        spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=stacked.case_axes,
        case_shape=stacked.case_shape,
    )

    with pytest.raises(ValueError, match="common query"):
        phx.uq.operator_input_predictive(
            deterministic,
            input_sample_axes=("input_draw",),
            field_name="output",
        )


def test_operator_prediction_field_rejects_dimension_collisions():
    axis = phx.nn.operator.OperatorAxis("case", jnp.linspace(0.0, 1.0, 3))
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    prediction = phx.nn.operator.OperatorPrediction.from_field(
        "output",
        jnp.ones((2, 3)),
        "query",
        query,
        spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=("case",),
        case_shape=(2,),
    )

    with pytest.raises(ValueError, match="must be unique"):
        phx.uq.operator_prediction_field(prediction, field_name="output")


def test_operator_predictive_rejects_complex_physical_outputs():
    batch = _tensor_batch()
    with pytest.raises(TypeError, match="real physical outputs"):
        phx.uq.operator_predictive_from_samples(
            jnp.ones((2, 2, 3), dtype=complex),
            batch,
            phx.nn.operator.OperatorOutputSpec("scalar"),
            sample_axes=(phx.uq.SampleAxis("draw", "epistemic"),),
            field_name="output",
            query_name="query",
        )
