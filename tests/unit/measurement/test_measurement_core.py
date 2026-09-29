from typing import Any

import numpy as np
import pytest

import phydrax as phx


def _quantity(
    name: Any = "signal", key: Any = "test.signal", unit: Any = phx.units.ONE
) -> Any:
    return phx.measurement.QuantitySpec("test", name, name, unit, key)


def test_measurement_core_scenario_1() -> None:
    support = phx.measurement.IndexSampleSupport((2,), ("sample",))
    layout = phx.measurement.ValueLayout(
        phx.measurement.ValueKind.VECTOR,
        (2,),
        ("x", "y"),
        "world",
    )
    field = phx.measurement.QuantityField(
        "velocity-field",
        _quantity("velocity", "physical.velocity", phx.units.METER),
        layout,
        support,
        phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT),
        np.asarray(((1.0, 2.0), (np.nan, np.nan))),
        np.asarray((True, False)),
        phx.measurement.IndependentStandardUncertainty(
            np.asarray((0.1, 0.2)), phx.units.METER
        ),
    )
    prepared = field.prepare()
    assert prepared.values.shape == (2, 2)
    assert prepared.valid_mask.shape == (2,)
    assert bool(prepared.successful)
    with pytest.raises(ValueError, match="component_frame_id"):
        phx.measurement.ValueLayout(phx.measurement.ValueKind.VECTOR, (2,))
    support = phx.measurement.IndexSampleSupport((2,), ("sample",))
    sampling = phx.measurement.SamplingSemantics(
        phx.measurement.SpatialSamplingKind.POINT
    )
    observed = phx.measurement.QuantityField(
        "observed",
        _quantity(),
        phx.measurement.ValueLayout.scalar(),
        support,
        sampling,
        np.asarray((1.0, 2.0)),
        uncertainty=phx.measurement.IndependentStandardUncertainty(
            np.asarray((0.5, 0.5)), phx.units.ONE
        ),
    ).prepare()
    predicted = phx.measurement.PreparedQuantityField(
        np.asarray((1.5, 1.0)),
        np.asarray((True, True)),
        standard_uncertainty=None,
        quantity_id=observed.quantity_id,
        compatibility_id=observed.compatibility_id,
        layout_id=observed.layout_id,
        support_id=observed.support_id,
        sampling_id=observed.sampling_id,
        unit_id=observed.unit_id,
        field_id="predicted",
    )
    result = phx.observation.MeasurementComparisonPlan(observed).evaluate(predicted)
    assert result.noise_model == "independent_uncertainty"
    assert result.whitened_residual is not None and result.quadratic is not None
    np.testing.assert_allclose(result.whitened_residual, (1.0, -2.0))
    np.testing.assert_allclose(result.quadratic, 5.0)
    assert bool(result.successful)
    covariance = phx.observation.DiagonalCovarianceAction(
        np.asarray((0.25, 0.25)),
        phx.observation.CoordinateLayout(("first", "second")),
    )
    correlated = phx.observation.MeasurementComparisonPlan(
        observed, covariance=covariance
    ).evaluate(predicted)
    assert correlated.whitening == "diagonal" and correlated.quadratic is not None
    np.testing.assert_allclose(correlated.quadratic, 5.0)
    assert bool(correlated.successful)
    incompatible = phx.measurement.QuantityField(
        "other",
        _quantity("other", "test.other"),
        phx.measurement.ValueLayout.scalar(),
        support,
        sampling,
        np.asarray((1.0, 2.0)),
    ).prepare()
    with pytest.raises(ValueError, match="quantity identities differ"):
        phx.observation.MeasurementComparisonPlan(observed).evaluate(incompatible)
    support = phx.measurement.IndexSampleSupport((1,), ("sample",))
    quantity = _quantity("complex-signal", "test.complex-signal")
    layout = phx.measurement.ValueLayout(phx.measurement.ValueKind.COMPLEX_SCALAR)
    sampling = phx.measurement.SamplingSemantics(
        phx.measurement.SpatialSamplingKind.POINT
    )
    observed = phx.measurement.QuantityField(
        "complex-observed",
        quantity,
        layout,
        support,
        sampling,
        np.asarray((0.0 + 0.0j,)),
    ).prepare()
    predicted = phx.measurement.QuantityField(
        "complex-predicted",
        quantity,
        layout,
        support,
        sampling,
        np.asarray((1.0 + 1.0j,)),
    ).prepare()
    result = phx.observation.MeasurementComparisonPlan(observed).evaluate(predicted)
    assert result.noise_model == "unquantified"
    assert result.quadratic is None and result.log_likelihood is None
    np.testing.assert_allclose(result.residual, (1.0 + 1.0j,))
    assert bool(result.successful)
    weighted = phx.observation.MeasurementComparisonPlan(
        observed, reference_scale=1.0
    ).evaluate(predicted)
    assert weighted.noise_model == "reference_weighting"
    assert weighted.quadratic is not None and weighted.log_likelihood is None
    np.testing.assert_allclose(weighted.quadratic, 2.0)
    assert bool(weighted.successful)


def test_measurement_core_scenario_2() -> None:
    axis = phx.measurement.SampleTimeAxis.uniform(
        "camera-clock", 3, 0.5, phx.units.SECOND
    )
    assert axis.axis_label == "camera-clock"
    assert axis.time_axis_id != axis.axis_label
    with pytest.raises(ValueError, match="interval_bounds"):
        phx.measurement.TemporalSampling(
            phx.measurement.TemporalSamplingKind.INTERVAL_MEAN
        )
    with pytest.raises(ValueError, match="generator"):
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RAW,
        )
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian",
        reference_frame="sensor",
    )
    with pytest.raises(ValueError, match="sample_count, 3"):
        phx.measurement.PointSampleSupport(np.zeros((1, 2)), ("point",), contract)
    with pytest.raises(ValueError, match="time UnitDefinition"):
        phx.measurement.PointSampleSupport(
            np.zeros((2, 3)),
            ("first", "second"),
            contract,
            sample_times=np.asarray((0.0, 1.0)),
            time_unit=phx.units.ONE,
        )
    with pytest.raises(ValueError, match="monotonically increasing"):
        phx.measurement.RaySampleSupport(
            np.zeros((2, 3)),
            np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            ("first", "second"),
            contract,
            sample_times=np.asarray((1.0, 0.0)),
            time_unit=phx.units.SECOND,
        )
