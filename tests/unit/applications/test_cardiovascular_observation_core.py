#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.observations._cine import CineTimingPlan
from phydrax.applications.cardiovascular.observations._metadata import (
    DataRightsIdentity,
    DeidentificationIdentity,
    MedicalImageAsset,
    ObservationRecord,
    SpatialAffine,
    SpatialConvention,
    SpatialFrame,
    TimeBase,
)
from phydrax.applications.cardiovascular.observations._registration import (
    RegistrationDirection,
    RegistrationEvaluationPlan,
)
from phydrax.applications.cardiovascular.observations._sampling import (
    P1ObservationPlan,
    SurfaceObservationPlan,
    TimeObservationPlan,
    VoxelObservationPlan,
)
from phydrax.applications.cardiovascular.observations._strain import (
    eulerian_strain,
    green_lagrange_strain,
    StrainEvaluationPlan,
    StrainMeasure,
)


def _lps_affine() -> SpatialAffine:
    frame = SpatialFrame("scanner-lps", SpatialConvention.LPS)
    matrix = np.asarray(
        [
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, -4.0],
            [0.0, 0.0, 5.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return SpatialAffine(matrix, "voxel-ijk", frame)


def _safe_identites() -> tuple[DeidentificationIdentity, DataRightsIdentity]:
    deidentification = DeidentificationIdentity(
        "deid-1",
        "subject-pseudonym-7",
        "protocol-1",
        True,
        True,
        True,
    )
    rights = DataRightsIdentity(
        "rights-1",
        "institutional-research-grant",
        ("research",),
        "data-controller",
    )
    return deidentification, rights


def test_spatial_affine_roundtrip_and_lps_ras_conversion() -> None:
    affine = _lps_affine()
    indices = np.asarray([[0.25, 1.5, 2.0], [4.0, 3.0, 2.0]])
    world_lps = affine.index_to_world(indices)
    np.testing.assert_allclose(affine.world_to_index(world_lps), indices)

    ras = affine.to_convention(SpatialFrame("scanner-ras", SpatialConvention.RAS))
    world_ras = ras.index_to_world(indices)
    np.testing.assert_allclose(world_ras, world_lps * np.asarray([-1.0, -1.0, 1.0]))
    np.testing.assert_allclose(ras.world_to_index(world_ras), indices)


def test_qform_sform_agreement_and_conflict_are_explicit() -> None:
    frame = SpatialFrame("scanner-lps", SpatialConvention.LPS)
    matrix = np.eye(4)
    resolved = SpatialAffine.from_qform_sform(
        qform_mm=matrix,
        sform_mm=matrix + np.diag([1.0e-7, 0.0, 0.0, 0.0]),
        source_frame_id="voxel-ijk",
        target_frame=frame,
    )
    assert resolved.provenance == "qform+sform-agree"

    conflict = matrix.copy()
    conflict[0, 3] = 2.0
    with pytest.raises(ValueError, match="qform/sform conflict"):
        SpatialAffine.from_qform_sform(
            qform_mm=matrix,
            sform_mm=conflict,
            source_frame_id="voxel-ijk",
            target_frame=frame,
        )


def test_medical_asset_refuses_phi_and_incomplete_deidentification() -> None:
    affine = _lps_affine()
    deidentification, rights = _safe_identites()
    values = np.arange(8.0).reshape((2, 2, 2))
    asset = MedicalImageAsset(
        "cine-1",
        "cine-mri",
        values,
        affine,
        None,
        deidentification,
        rights,
        "signal-intensity",
        "arbitrary-unit",
        metadata={"series_description": "short-axis cine"},
    )
    assert not asset.values.flags.writeable
    assert len(asset.content_id) == 64

    with pytest.raises(PermissionError, match="PHI refusal"):
        MedicalImageAsset(
            "cine-phi",
            "cine-mri",
            values,
            affine,
            None,
            deidentification,
            rights,
            "signal-intensity",
            "arbitrary-unit",
            metadata={"patient_name": "identifying value"},
        )

    unsafe = DeidentificationIdentity(
        "deid-unsafe",
        "subject-pseudonym-8",
        "protocol-1",
        False,
        True,
        True,
    )
    with pytest.raises(PermissionError, match="PHI refusal"):
        MedicalImageAsset(
            "cine-unsafe",
            "cine-mri",
            values,
            affine,
            None,
            unsafe,
            rights,
            "signal-intensity",
            "arbitrary-unit",
        )


def test_observation_record_is_an_immutable_host_channel() -> None:
    record = ObservationRecord(
        "pressure-1",
        "catheter-pressure",
        np.asarray([10.0, np.nan]),
        np.asarray([True, False]),
        "pressure",
        "kPa",
        timebase_id="pressure-clock",
    )
    assert not record.values.flags.writeable
    assert not record.valid_mask.flags.writeable
    with pytest.raises(ValueError):
        record.values[0] = 11.0


def test_voxel_coverage_transpose_and_jvp() -> None:
    frame = SpatialFrame("world-lps", SpatialConvention.LPS)
    affine = SpatialAffine(np.eye(4), "voxel-ijk", frame)
    points = np.asarray([[0.25, 0.5, 0.75], [2.0, 0.0, 0.0]])
    operator = VoxelObservationPlan((2, 2, 2), affine, points).prepare()
    values = jnp.asarray(
        np.fromfunction(lambda i, j, k: i + 2.0 * j + 4.0 * k, (2, 2, 2))
    )
    candidate = operator.apply(values)
    np.testing.assert_allclose(np.asarray(candidate.values), np.asarray([4.25, 0.0]))
    assert int(candidate.evidence.covered_count) == 1
    assert float(candidate.evidence.coverage_fraction) == pytest.approx(0.5)
    assert not bool(candidate.evidence.complete_coverage)
    assert bool(candidate.evidence.successful)

    cotangent = jnp.asarray([1.5, -3.0])
    transpose = operator.transpose(cotangent)
    lhs = jnp.vdot(candidate.values, cotangent)
    rhs = jnp.vdot(values, transpose)
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1.0e-6, atol=1.0e-6)

    tangent = jnp.arange(8.0).reshape((2, 2, 2))
    jvp = operator.jvp(values, tangent)
    expected_tangent = operator.apply(tangent).values
    np.testing.assert_allclose(np.asarray(jvp.tangent), np.asarray(expected_tangent))


def test_tetrahedral_surface_and_time_p1_sampling() -> None:
    nodes = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    tetrahedral = P1ObservationPlan(
        nodes,
        np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        np.asarray([[0.1, 0.2, 0.3]]),
        "tetra-mesh",
        require_complete_coverage=True,
    ).prepare()
    values = jnp.asarray([0.0, 1.0, 2.0, 3.0])
    result = tetrahedral.apply(values)
    np.testing.assert_allclose(np.asarray(result.values), np.asarray([1.4]))
    assert bool(result.evidence.successful)

    surface = SurfaceObservationPlan(
        nodes[:3],
        np.asarray([[0, 1, 2]], dtype=np.int32),
        np.asarray([[0.25, 0.5, 0.0]]),
        "endo-surface",
        require_complete_coverage=True,
    ).prepare()
    np.testing.assert_allclose(
        np.asarray(surface.apply(jnp.asarray([2.0, 4.0, 8.0])).values),
        np.asarray([5.5]),
    )

    timebase = TimeBase("cine-clock", np.asarray([0.0, 10.0, 20.0]))
    temporal = TimeObservationPlan(timebase, np.asarray([5.0, 20.0, 25.0])).prepare()
    temporal_result = temporal.apply(jnp.asarray([0.0, 10.0, 40.0]))
    np.testing.assert_allclose(
        np.asarray(temporal_result.values), np.asarray([5.0, 40.0, 0.0])
    )
    assert int(temporal_result.evidence.covered_count) == 2


def test_cine_timing_has_periodic_phase_and_conservative_frame_widths() -> None:
    timebase = TimeBase.uniform("cine-clock", 4, 200.0)
    timing = CineTimingPlan(timebase, 800.0, 0.0).prepare().evaluate()
    np.testing.assert_allclose(
        np.asarray(timing.phase), np.asarray([0.0, 0.25, 0.5, 0.75])
    )
    np.testing.assert_allclose(np.asarray(timing.frame_duration_ms), 200.0)
    assert float(jnp.sum(timing.frame_duration_ms)) == pytest.approx(800.0)
    assert float(timing.evidence.phase_coverage) == pytest.approx(0.75)
    assert bool(timing.evidence.successful)


def test_registration_translation_inverse_consistency_and_folding() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    prepared = RegistrationEvaluationPlan(
        points,
        "end-diastole",
        "end-systole",
        require_inverse_consistency=True,
        require_uncertainty=True,
    ).prepare()
    displacement = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gradient = jnp.zeros((2, 3, 3))
    with pytest.raises(TypeError, match="reference_frame_id"):
        prepared.evaluate(displacement, gradient)
    with pytest.raises(ValueError, match="REFERENCE_TO_TARGET"):
        RegistrationEvaluationPlan(
            points,
            "end-diastole",
            "end-systole",
            direction=RegistrationDirection.TARGET_TO_REFERENCE,
        )
    result = prepared.evaluate(
        displacement,
        gradient,
        inverse_displacement_at_deformed_mm=-displacement,
        displacement_standard_deviation_mm=jnp.full_like(displacement, 0.1),
        reference_frame_id="end-diastole",
        target_frame_id="end-systole",
    )
    np.testing.assert_allclose(
        np.asarray(result.deformed_points_mm), points + np.asarray(displacement)
    )
    np.testing.assert_allclose(np.asarray(result.evidence.jacobian_determinant), 1.0)
    assert bool(result.evidence.inverse_consistent)
    assert bool(result.evidence.uncertainty_valid)
    assert bool(result.evidence.successful)
    assert prepared.commit(result).plan_id == prepared.plan_id

    folding_gradient = gradient.at[:, 0, 0].set(-2.0)
    folded = prepared.evaluate(
        displacement,
        folding_gradient,
        inverse_displacement_at_deformed_mm=-displacement,
        displacement_standard_deviation_mm=jnp.full_like(displacement, 0.1),
        reference_frame_id="end-diastole",
        target_frame_id="end-systole",
    )
    assert int(folded.evidence.folding_count) == 2
    assert not bool(folded.evidence.successful)
    with pytest.raises(ValueError, match="unsuccessful"):
        prepared.commit(folded)


def test_green_lagrange_and_eulerian_synthetic_stretch() -> None:
    deformation_gradient = jnp.diag(jnp.asarray([2.0, 1.0, 1.0]))[None, ...]
    green = green_lagrange_strain(deformation_gradient)
    eulerian = eulerian_strain(deformation_gradient)
    np.testing.assert_allclose(np.asarray(green[0, 0, 0]), 1.5)
    np.testing.assert_allclose(np.asarray(eulerian[0, 0, 0]), 0.375)
    np.testing.assert_allclose(np.asarray(green[0, 1:, 1:]), np.zeros((2, 2)))

    prepared = StrainEvaluationPlan(
        (1,),
        "end-diastole",
        StrainMeasure.GREEN_LAGRANGE,
        require_uncertainty=True,
    ).prepare()
    result = prepared.evaluate(
        deformation_gradient,
        deformation_gradient_standard_deviation=jnp.full((1, 3, 3), 0.01),
        reference_frame_id="end-diastole",
    )
    np.testing.assert_allclose(np.asarray(result.strain), np.asarray(green))
    assert np.all(np.asarray(result.strain_standard_deviation) >= 0.0)
    assert bool(result.evidence.uncertainty_valid)
    assert bool(result.evidence.successful)
    frame_mismatch = prepared.evaluate(
        deformation_gradient,
        deformation_gradient_standard_deviation=jnp.full((1, 3, 3), 0.01),
        reference_frame_id="end-systole",
    )
    assert not bool(frame_mismatch.evidence.reference_frame_matched)
    assert not bool(frame_mismatch.evidence.successful)
    with pytest.raises(TypeError, match="reference_frame_id"):
        prepared.evaluate(
            deformation_gradient,
            deformation_gradient_standard_deviation=jnp.full((1, 3, 3), 0.01),
        )
