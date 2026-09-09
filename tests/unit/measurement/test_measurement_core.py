import numpy as np
import pytest

import phydrax as phx


def _quantity(name="signal", key="test.signal", unit=phx.units.ONE):
    return phx.measurement.QuantitySpec("test", name, name, unit, key)


def test_quantity_fields_keep_sample_validity_separate_from_components_and_units():
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


def test_measurement_comparison_requires_semantic_and_support_compatibility():
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
    np.testing.assert_allclose(result.standardized_residual, (1.0, -2.0))
    np.testing.assert_allclose(result.quadratic, 5.0)
    assert bool(result.successful)
    covariance = phx.observation.DiagonalCovarianceAction(
        np.asarray((0.25, 0.25)),
        phx.observation.CoordinateLayout(("first", "second")),
    )
    correlated = phx.observation.MeasurementComparisonPlan(
        observed, covariance=covariance
    ).evaluate(predicted)
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


def test_complex_measurement_residual_uses_hermitian_magnitude():
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
    np.testing.assert_allclose(result.quadratic, 2.0)
    assert bool(result.successful)


def test_time_and_lineage_do_not_collapse_acquisition_semantics():
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
