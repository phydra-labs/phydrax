import hashlib
import math

import numpy as np
import pytest

import phydrax as phx
from phydrax.nuclear import dosimetry


def _data(name="internal-dosimetry"):
    payload = name.encode()
    reference = phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    return phx.nuclear.NuclearDataProvenance(
        reference,
        "https://example.invalid/internal-dosimetry",
        "synthetic-s-values",
        "release",
        name,
    )


def _transition(data):
    iodine = phx.nuclear.NuclideKey(53, 131)
    xenon = phx.nuclear.NuclideKey(54, 131)
    return phx.nuclear.InventoryTransition(
        "beta-minus",
        iodine,
        ((xenon, 1.0),),
        math.log(2.0) / 10.0,
        np.asarray((0.0,)),
        1.0e-13,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        data,
    )


def _derivation():
    return phx.measurement.DerivationRecord(
        phx.measurement.DataOrigin.SYNTHETIC,
        phx.measurement.DataStage.RECONSTRUCTED,
        transformation_id="synthetic-time-activity",
    )


def _quantity(kind, unit, *, support):
    return phx.measurement.resolve_radiation_quantity(
        "iodine-131-activity",
        kind,
        unit,
        support_association=support,
        reference_configuration="iodine-131-calibrated-activity",
    )


def _regional_series(values=(4.0, 2.0, 1.0)):
    data = _data()
    transition = _transition(data)
    axis = phx.measurement.SampleTimeAxis(
        "activity-time", np.asarray((0.0, 1.0, 3.0)), phx.units.SECOND
    )
    support = phx.measurement.IndexSampleSupport(
        (3,), ("time",), axis, 0, frame_id="source-regions"
    )
    field = phx.measurement.QuantityField(
        "activity-field",
        _quantity(
            phx.measurement.RadiationQuantityKind.ACTIVITY,
            phx.units.BECQUEREL,
            support="source-region-instantaneous",
        ),
        phx.measurement.ValueLayout.scalar(),
        support,
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.CELL_AVERAGE
        ),
        np.asarray(values),
    )
    asset = phx.measurement.MeasurementAsset(
        "activity-asset",
        field,
        None,
        (data.reference,),
        _derivation(),
        "research",
        {"radionuclide": transition.parent.nuclide_id},
    )
    return dosimetry.TimeActivitySeries(asset, transition), data


def _affine():
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-lps",
        reference_frame="synthetic-patient",
    )
    return phx.imaging.ImageIndexAffine(
        np.eye(4), "voxel-index", contract, phx.imaging.ImageAxisConvention.LPS
    )


def _spatial_series():
    data = _data("spatial-s-values")
    transition = _transition(data)
    axis = phx.measurement.SampleTimeAxis(
        "activity-time", np.asarray((0.0, 1.0, 3.0)), phx.units.SECOND
    )
    values = np.zeros((3, 3, 3, 3), dtype="float64")
    values[1, 1, 1, :] = np.asarray((4.0, 2.0, 1.0))
    spec = phx.imaging.ImageFieldSpec(
        _quantity(
            phx.measurement.RadiationQuantityKind.ACTIVITY_CONCENTRATION,
            phx.units.BECQUEREL_PER_CUBIC_METER,
            support="voxel-cell-average",
        ),
        phx.measurement.ValueLayout.scalar(),
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.CELL_AVERAGE
        ),
    )
    image = phx.imaging.MedicalImageAsset(
        "activity-image",
        "synthetic-nuclear-medicine",
        values,
        _affine(),
        spec,
        phx.imaging.DeidentificationEvidence(
            "synthetic-deid", "subject-0", "synthetic", True, True, True
        ),
        (data.reference,),
        _derivation(),
        time_axis=axis,
    )
    return dosimetry.TimeActivitySeries(image.measurement, transition), data


def test_time_activity_trapezoid_is_six_becquerel_seconds_and_refuses_extrapolation():
    series, _ = _regional_series()
    result = dosimetry.TimeActivityIntegrationPlan(
        series.time_axis, 0.0, 3.0, phx.units.SECOND
    ).integrate(series)

    np.testing.assert_allclose(result.values, 6.0)
    assert (
        result.asset.field.quantity.quantity_kind
        == phx.measurement.RadiationQuantityKind.TIME_INTEGRATED_ACTIVITY.value
    )
    assert result.asset.field.support.sample_shape == ()
    assert result.asset.field.uncertainty is None

    with pytest.raises(ValueError, match="inside the sampled time support"):
        dosimetry.TimeActivityIntegrationPlan(
            series.time_axis, -1.0, 3.0, phx.units.SECOND
        )


def test_time_activity_result_binds_evidence_to_asset_and_radionuclide():
    series, _ = _regional_series()
    result = dosimetry.TimeActivityIntegrationPlan(
        series.time_axis, 0.0, 3.0, phx.units.SECOND
    ).integrate(series)
    evidence = result.evidence
    wrong_target = dosimetry.TimeActivityIntegrationEvidence(
        evidence.source_id,
        "different-target",
        evidence.radionuclide_id,
        evidence.transition_id,
        evidence.time_axis_id,
        evidence.start_s,
        evidence.end_s,
        evidence.method,
    )
    with pytest.raises(ValueError, match="target does not match"):
        dosimetry.TimeActivityIntegrationResult(
            result.asset, result.transition, wrong_target
        )

    wrong_radionuclide = dosimetry.TimeActivityIntegrationEvidence(
        evidence.source_id,
        result.asset.content_id,
        "different-radionuclide",
        evidence.transition_id,
        evidence.time_axis_id,
        evidence.start_s,
        evidence.end_s,
        evidence.method,
    )
    with pytest.raises(ValueError, match="radionuclide and transition disagree"):
        dosimetry.TimeActivityIntegrationResult(
            result.asset, result.transition, wrong_radionuclide
        )


def test_regional_s_values_apply_target_by_source_arithmetic_and_preserve_unknown_uncertainty():
    series, data = _regional_series()
    integrated = dosimetry.TimeActivityIntegrationPlan(
        series.time_axis, 0.0, 3.0, phx.units.SECOND
    ).integrate(series)
    table = dosimetry.RegionalSValueTable(
        ("source",),
        ("target-a", "target-b"),
        np.asarray(((2.0,), (0.5,))),
        dosimetry.S_VALUE_UNIT,
        "source-regions",
        "target-regions",
        series.transition,
        data,
    )
    source_support = phx.measurement.IndexSampleSupport(
        (1,), ("source_region",), frame_id="source-regions"
    )
    source_field = phx.measurement.QuantityField(
        "regional-integral",
        integrated.asset.field.quantity,
        integrated.asset.field.layout,
        source_support,
        integrated.asset.field.sampling,
        np.asarray((6.0,)),
    )
    source_asset = phx.measurement.MeasurementAsset(
        "regional-integral-asset",
        source_field,
        None,
        integrated.asset.references,
        integrated.asset.derivation,
        "research",
        integrated.asset.metadata,
    )
    base_evidence = integrated.evidence
    regional_evidence = dosimetry.TimeActivityIntegrationEvidence(
        base_evidence.source_id,
        source_asset.content_id,
        base_evidence.radionuclide_id,
        base_evidence.transition_id,
        base_evidence.time_axis_id,
        base_evidence.start_s,
        base_evidence.end_s,
        base_evidence.method,
    )
    regional_source = dosimetry.TimeActivityIntegrationResult(
        source_asset, integrated.transition, regional_evidence
    )

    result = dosimetry.RegionalSValuePlan(table, source_support).apply(regional_source)
    np.testing.assert_allclose(result.dose_gy, (12.0, 3.0))
    assert result.asset.field.uncertainty is None
    assert result.target_region_ids == ("target-a", "target-b")


def test_spatial_delta_kernel_converts_activity_concentration_through_voxel_volume():
    series, data = _spatial_series()
    integrated = dosimetry.TimeActivityIntegrationPlan(
        series.time_axis, 0.0, 3.0, phx.units.SECOND
    ).integrate(series)
    support = integrated.asset.field.support
    kernel = dosimetry.SpatialSValueKernel(
        np.ones((1, 1, 1)),
        dosimetry.S_VALUE_UNIT,
        support,
        support,
        "activity-grid",
        "dose-grid",
        series.transition,
        data,
    )
    result = dosimetry.SpatialSValueConvolutionPlan(kernel).apply(integrated)

    expected = np.zeros((3, 3, 3), dtype="float64")
    expected[1, 1, 1] = 6.0
    np.testing.assert_allclose(result.dose_gy, expected)
    assert np.all(result.valid_mask)
    assert result.asset.field.uncertainty is None


def test_spatial_dosimetry_refuses_wrong_quantity_and_grid():
    series, data = _spatial_series()
    integrated = dosimetry.TimeActivityIntegrationPlan(
        series.time_axis, 0.0, 3.0, phx.units.SECOND
    ).integrate(series)
    support = integrated.asset.field.support
    shifted = phx.imaging.MedicalImageSupport(
        support.spatial_shape,
        phx.imaging.ImageIndexAffine(
            np.asarray(
                (
                    (1.0, 0.0, 0.0, 0.1),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                )
            ),
            "voxel-index",
            support.spatial_affine.coordinate_contract,
            phx.imaging.ImageAxisConvention.LPS,
        ),
    )
    with pytest.raises(ValueError, match="exactly compatible"):
        dosimetry.SpatialSValueKernel(
            np.ones((1, 1, 1)),
            dosimetry.S_VALUE_UNIT,
            support,
            shifted,
            "activity-grid",
            "dose-grid",
            series.transition,
            data,
        )
