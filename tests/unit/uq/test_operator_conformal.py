#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _batch(
    *,
    cases: int,
    coordinates,
    weights,
    mask=None,
) -> phx.nn.operator.OperatorBatch:
    nodes = jnp.asarray(coordinates, dtype=float)
    query_coordinates = jnp.broadcast_to(nodes[None, :, None], (cases, nodes.size, 1))
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        quadrature_weights=jnp.broadcast_to(
            jnp.asarray(weights, dtype=float),
            (cases, nodes.size),
        ),
        mask=None if mask is None else jnp.asarray(mask, dtype=bool),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.zeros((cases, nodes.size)),
        coordinates=query_coordinates,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(cases,),
    )


def _prediction(values, batch):
    return phx.nn.operator.OperatorPrediction.from_field(
        "output",
        jnp.asarray(values, dtype=float),
        "query",
        batch.require_single_query(),
        spec=phx.nn.operator.OperatorOutputSpec("scalar"),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def test_operator_functional_conformal_calibrates_complete_physical_cases():
    calibration_batch = _batch(
        cases=5,
        coordinates=[0.0, 0.5, 1.0],
        weights=[0.25, 0.5, 0.25],
        mask=jnp.asarray(
            [
                [True, True, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        ),
    )
    center = _prediction(jnp.zeros((5, 3)), calibration_batch)
    residuals = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])[:, None]
    target = jnp.broadcast_to(residuals, (5, 3)).at[0, 2].set(jnp.nan)

    calibrator = phx.uq.OperatorFunctionalConformal.calibrate(
        center,
        target,
        alpha=0.2,
        field_name="output",
    )

    assert calibrator.calibrator.radius == pytest.approx(0.5)
    evaluation_batch = _batch(
        cases=2,
        coordinates=[0.0, 0.25, 0.5, 1.0],
        weights=[0.1, 0.2, 0.3, 0.4],
    )
    evaluation_center = _prediction(
        jnp.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]),
        evaluation_batch,
    )
    interval = calibrator.interval(evaluation_center)

    assert interval.nominal_coverage == pytest.approx(0.8)
    assert interval.simultaneous
    assert interval.calibrated
    assert jnp.allclose(
        interval.lower.field("output").values,
        evaluation_center.field("output").values - 0.5,
    )
    assert jnp.allclose(
        interval.upper.field("output").values,
        evaluation_center.field("output").values + 0.5,
    )


def test_normalized_operator_conformal_uses_scale_field():
    batch = _batch(
        cases=5,
        coordinates=[0.0, 1.0],
        weights=[0.5, 0.5],
    )
    center = _prediction(jnp.zeros((5, 2)), batch)
    scale = _prediction(2.0 * jnp.ones((5, 2)), batch)
    target = jnp.asarray([0.2, 0.4, 0.6, 0.8, 1.0])[:, None]
    target = jnp.broadcast_to(target, (5, 2))
    calibrator = phx.uq.OperatorFunctionalConformal.calibrate(
        center,
        target,
        alpha=0.2,
        field_name="output",
        scale=scale,
    )
    interval = calibrator.interval(center, scale)

    assert calibrator.calibrator.radius == pytest.approx(0.5)
    assert jnp.allclose(
        interval.upper.field("output").values - center.field("output").values,
        1.0,
    )
    with pytest.raises(ValueError, match="requires a scale field"):
        calibrator.interval(center)


def test_l2_operator_conformal_is_quadrature_split_invariant_and_not_pointwise():
    base_batch = _batch(
        cases=5,
        coordinates=[0.0, 1.0],
        weights=[0.4, 0.6],
    )
    split_batch = _batch(
        cases=5,
        coordinates=[0.0, 1.0, 1.0],
        weights=[0.4, 0.3, 0.3],
    )
    base_center = _prediction(jnp.zeros((5, 2)), base_batch)
    split_center = _prediction(jnp.zeros((5, 3)), split_batch)
    case_values = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5])[:, None]
    base_target = jnp.broadcast_to(case_values, (5, 2))
    split_target = base_target[..., [0, 1, 1]]

    base = phx.uq.OperatorFunctionalConformal.calibrate(
        base_center,
        base_target,
        alpha=0.2,
        field_name="output",
        score="l2",
    )
    split = phx.uq.OperatorFunctionalConformal.calibrate(
        split_center,
        split_target,
        alpha=0.2,
        field_name="output",
        score="l2",
    )

    assert jnp.allclose(base.calibrator.radius, split.calibrator.radius)
    with pytest.raises(ValueError, match="norm ball"):
        base.interval(base_center)
