import numpy as np
import pytest

import phydrax as phx
from phydrax.imaging._ct import (
    apply_hu_calibration,
    HUCalibrationAnchor,
    HUToMaterialCalibration,
)


def _manifest(name="synthetic-ct-source", checksum_digit="0"):
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=checksum_digit * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"ct_number": 1.0},
        uncertainty={"hu": 10.0},
        lineage_ids=(name,),
    )


def _asset(values):
    coordinate_contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="synthetic-ct",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4),
        "voxel-index",
        coordinate_contract,
        phx.imaging.ImageAxisConvention.LPS,
    )
    spec = phx.imaging.ImageFieldSpec.named(
        "ct-number",
        phx.units.ONE,
        phx.measurement.ValueKind.REAL_SCALAR,
        quantity_kind="ct_number",
        compatibility_key="imaging.ct_number",
    )
    array = np.asarray(values, dtype=float).reshape((-1, 1, 1))
    return phx.imaging.MedicalImageAsset(
        "synthetic-ct",
        "ct",
        array,
        affine,
        spec,
        phx.imaging.DeidentificationEvidence(
            "synthetic-deid", "subject-0", "synthetic", True, True, True
        ),
        (_manifest(),),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="synthetic-ct-generator",
        ),
        uncertainty=phx.measurement.IndependentStandardUncertainty(
            np.full(array.shape, 10.0), phx.units.ONE
        ),
    )


def _calibration():
    return HUToMaterialCalibration(
        "synthetic-three-anchor",
        ("air-like", "water-like"),
        (
            HUCalibrationAnchor(-1000.0, 1.0, np.asarray((1.0, 0.0))),
            HUCalibrationAnchor(0.0, 1000.0, np.asarray((0.0, 1.0))),
            HUCalibrationAnchor(1000.0, 1800.0, np.asarray((0.0, 1.0))),
        ),
        _manifest("synthetic-ct-calibration", "1"),
    )


def test_hu_calibration_preserves_support_and_material_basis_order():
    source = _asset((-500.0, 0.0, 500.0))
    calibrated = apply_hu_calibration(source, _calibration())

    assert calibrated.support.support_id == source.support.support_id
    assert calibrated.density.support.support_id == source.support.support_id
    assert calibrated.material_fractions.support.support_id == source.support.support_id
    assert calibrated.material_fractions.layout.component_labels == (
        "air-like",
        "water-like",
    )
    np.testing.assert_allclose(
        calibrated.density.values[:, 0, 0], (500.5, 1000.0, 1400.0)
    )
    np.testing.assert_allclose(
        calibrated.material_fractions.values[:, 0, 0],
        ((0.5, 0.5), (0.0, 1.0), (0.0, 1.0)),
    )
    np.testing.assert_allclose(
        calibrated.density.uncertainty.values[:, 0, 0], (9.99, 8.0, 8.0)
    )
    assert calibrated.material_fractions.uncertainty is None
    assert (
        calibrated.material_fractions.metadata["material_fraction_uncertainty"]
        == "not-represented-shared-hu-covariance"
    )
    assert len(calibrated.density.references) == 2


def test_hu_calibration_refuses_values_outside_closed_support():
    with pytest.raises(ValueError, match="clamping and extrapolation are not permitted"):
        apply_hu_calibration(_asset((-1000.0, 1000.1)), _calibration())
