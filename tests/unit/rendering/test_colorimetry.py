#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.artifacts import (
    admit_external_artifact,
    AdmittedExternalArtifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.optics.wave import ThinFilmInterferencePlan, ThinFilmInterferenceStatus
from phydrax.rendering import (
    AnalyticColorMatchingFunctions,
    COLOR_MATCHING_TABLE_MODEL,
    encode_srgb,
    read_spectral_illuminant,
    SpectralColorimetryPlan,
    SpectralColorimetryStatus,
    SpectralIlluminant,
    TabulatedColorMatchingFunctions,
    ThinFilmAppearancePlan,
    ThinFilmAppearanceStatus,
    xyz_to_linear_srgb,
)


VISIBLE = np.arange(380.0, 781.0, 5.0) / 1.0e9
SRGB_WHITE_XY = (0.3127, 0.3290)


def _planck(wavelengths: np.ndarray, temperature: float) -> np.ndarray:
    second_radiation_constant = 1.438776877e-2
    return wavelengths**-5 / np.expm1(
        second_radiation_constant / (wavelengths * temperature)
    )


def _daylight_like() -> SpectralIlluminant:
    grid = np.arange(360.0, 831.0, 1.0) / 1.0e9
    return SpectralIlluminant(
        grid, _planck(grid, 6504.0), illuminant_id="planckian-6504-kelvin"
    )


def _admitted(
    root: Path, name: str, rows: np.ndarray, model: str
) -> tuple[AdmittedExternalArtifact, ArtifactManifest, ExternalArtifactPolicy]:
    payload = "\r\n".join(
        ",".join(f"{value:.12g}" for value in row) for row in rows
    ).encode("ascii")
    (root / name).write_bytes(payload)
    manifest = ArtifactManifest(
        artifact_id=name,
        producer="phydrax-tests",
        version="1",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        source_uri=f"memory://{name}",
        license_id="internal-test",
        model=model,
        coverage="synthetic spectral table",
    )
    policy = ExternalArtifactPolicy(
        root,
        maximum_bytes=1 << 20,
        allowed_license_ids=["internal-test"],
        allowed_suffixes=[".csv"],
    )
    return admit_external_artifact(name, manifest, policy=policy), manifest, policy


def test_srgb_transfer_fixes_endpoints_and_joins_its_branches() -> None:
    threshold = 0.0031308
    values = jnp.asarray([0.0, threshold, 0.5, 1.0])
    encoded = encode_srgb(values)

    assert float(encoded[0]) == 0.0
    assert float(encoded[-1]) == pytest.approx(1.0, abs=4.0 * np.finfo(np.float64).eps)
    power_branch = 1.055 * threshold ** (1.0 / 2.4) - 0.055
    # The standard's rounded constants leave a join mismatch of about 3e-8.
    assert float(encoded[1]) == pytest.approx(power_branch, abs=5.0e-8)
    assert bool(jnp.all(jnp.diff(encode_srgb(jnp.linspace(0.0, 1.0, 1001))) > 0.0))
    np.testing.assert_allclose(encode_srgb(-values), -encoded)


def test_d65_white_point_is_neutral_in_linear_and_encoded_srgb() -> None:
    x, y = SRGB_WHITE_XY
    white = jnp.asarray([x / y, 1.0, (1.0 - x - y) / y])
    linear = xyz_to_linear_srgb(white)

    np.testing.assert_allclose(linear, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(encode_srgb(linear), 1.0, atol=1.0e-12)


def test_unit_reflector_maps_to_the_illuminant_white_with_unit_luminance() -> None:
    plan = SpectralColorimetryPlan(VISIBLE, _daylight_like())
    result = plan.evaluate(jnp.ones((2, VISIBLE.size)))

    assert float(plan.white_xyz[1]) == pytest.approx(1.0, abs=1.0e-14)
    np.testing.assert_allclose(result.xyz, np.broadcast_to(plan.white_xyz, (2, 3)))
    np.testing.assert_allclose(
        result.linear_srgb, np.broadcast_to(plan.white_linear_srgb, (2, 3)), rtol=1e-14
    )
    assert plan.white_neutral_error == pytest.approx(
        float(jnp.max(jnp.abs(plan.white_linear_srgb - 1.0)))
    )
    assert bool(jnp.all(result.accepted))


def test_equal_energy_white_sits_at_the_equal_energy_chromaticity() -> None:
    grid = np.arange(360.0, 831.0, 1.0) / 1.0e9
    equal_energy = SpectralIlluminant(grid, np.ones_like(grid), illuminant_id="cie-e")
    plan = SpectralColorimetryPlan(grid, equal_energy)
    chromaticity = plan.white_xyz[:2] / jnp.sum(plan.white_xyz)

    # CIE 1931 functions have equal integrals; the analytic fit keeps them within
    # its published error.
    np.testing.assert_allclose(chromaticity, 1.0 / 3.0, atol=1.0e-3)
    assert plan.fit_error is not None
    assert plan.color_matching_coverage == pytest.approx((1.0, 1.0, 1.0), abs=1.0e-6)


def test_narrowband_color_is_reported_out_of_gamut_and_mapped_by_policy() -> None:
    spike = np.zeros(VISIBLE.size)
    spike[np.argmin(np.abs(VISIBLE - 520.0e-9))] = 1.0
    clipped = SpectralColorimetryPlan(VISIBLE, _daylight_like(), exposure=20.0).evaluate(
        spike
    )
    extended = SpectralColorimetryPlan(
        VISIBLE, _daylight_like(), exposure=20.0, gamut_mapping="extended"
    ).evaluate(spike)

    assert bool(clipped.evidence.out_of_gamut)
    assert float(clipped.evidence.gamut_excess) > 0.0
    assert float(clipped.linear_srgb[0]) < 0.0
    display = clipped.display_linear_srgb
    assert bool(jnp.all((display >= 0.0) & (display <= 1.0)))
    np.testing.assert_allclose(extended.display_linear_srgb, extended.linear_srgb)
    assert bool(clipped.accepted) and bool(extended.accepted)


def test_invalid_spectra_are_rejected_with_status_bits() -> None:
    plan = SpectralColorimetryPlan(VISIBLE, _daylight_like())
    spectra = np.full((3, VISIBLE.size), 0.5)
    spectra[0, 3] = np.nan
    spectra[1, 7] = -0.1
    result = plan.evaluate(spectra)

    np.testing.assert_array_equal(
        result.status,
        [
            SpectralColorimetryStatus.NONFINITE_SPECTRUM,
            SpectralColorimetryStatus.NEGATIVE_SPECTRUM,
            SpectralColorimetryStatus.SUCCESS,
        ],
    )
    assert bool(jnp.all(jnp.isnan(result.encoded_srgb[:2])))
    assert bool(jnp.all(jnp.isfinite(result.encoded_srgb[2])))


def test_plan_refuses_wavelengths_outside_observer_or_illuminant_support() -> None:
    with pytest.raises(ValueError, match="color-matching support"):
        SpectralColorimetryPlan(np.arange(350.0, 701.0, 5.0) / 1.0e9, _daylight_like())
    narrow = SpectralIlluminant(
        VISIBLE[:20], np.ones(20), illuminant_id="narrow-lamp"
    )
    with pytest.raises(ValueError, match="illuminant table"):
        SpectralColorimetryPlan(VISIBLE, narrow)


def test_hash_verified_tables_decode_and_reproduce_the_array_route(
    tmp_path: Path,
) -> None:
    table_grid = np.arange(360.0, 831.0, 1.0)
    observer_rows = np.column_stack(
        (table_grid, AnalyticColorMatchingFunctions().sample(table_grid / 1.0e9))
    )
    power = _planck(table_grid / 1.0e9, 6504.0)
    illuminant_rows = np.column_stack((table_grid, power / power.max()))
    observer_artifact, observer_manifest, observer_policy = _admitted(
        tmp_path, "observer.csv", observer_rows, COLOR_MATCHING_TABLE_MODEL
    )
    observer = TabulatedColorMatchingFunctions(
        observer_artifact, observer_manifest, policy=observer_policy
    )
    artifact, manifest, policy = _admitted(
        tmp_path, "lamp.csv", illuminant_rows, "relative-spectral-power"
    )
    lamp = read_spectral_illuminant(
        artifact, manifest, policy=policy, illuminant_id="planckian-6504-kelvin"
    )

    tabulated = SpectralColorimetryPlan(VISIBLE, lamp, color_matching=observer)
    analytic = SpectralColorimetryPlan(VISIBLE, _daylight_like())
    assert lamp.source_sha256 == manifest.sha256
    assert tabulated.fit_error is None
    np.testing.assert_allclose(tabulated.white_xyz, analytic.white_xyz, atol=1.0e-9)

    (tmp_path / "lamp.csv").write_bytes(b"380,1\r\n780,1")
    with pytest.raises(ValueError, match="size|SHA-256"):
        read_spectral_illuminant(
            artifact, manifest, policy=policy, illuminant_id="planckian-6504-kelvin"
        )
    wrong_model = _admitted(tmp_path, "other.csv", observer_rows, "lamp-spectrum")
    with pytest.raises(ValueError, match="model"):
        TabulatedColorMatchingFunctions(
            wrong_model[0], wrong_model[1], policy=wrong_model[2]
        )


def _soap_appearance(*, two_sided: bool) -> ThinFilmAppearancePlan:
    interference = ThinFilmInterferencePlan(VISIBLE, 1.0, 1.33, 1.0)
    colorimetry = SpectralColorimetryPlan(VISIBLE, _daylight_like(), exposure=4.0)
    return ThinFilmAppearancePlan(interference, colorimetry, two_sided=two_sided)


def test_appearance_composes_interference_and_colorimetry_at_the_view_angle() -> None:
    plan = _soap_appearance(two_sided=False)
    thickness = jnp.asarray([0.0, 150.0e-9, 450.0e-9, 900.0e-9])
    normal = jnp.asarray([0.0, 0.0, 1.0])
    view = jnp.asarray([0.0, np.sin(0.6), np.cos(0.6)])
    result = plan.evaluate(thickness, normal, view)

    direct = plan.colorimetry.evaluate(
        plan.interference.evaluate(thickness, np.cos(0.6)).unpolarized_reflectance
    )
    np.testing.assert_allclose(result.colors.encoded_srgb, direct.encoded_srgb)
    np.testing.assert_allclose(result.evidence.incidence_cosine, np.cos(0.6))
    np.testing.assert_allclose(
        result.evidence.illumination_directions[0], [0.0, -np.sin(0.6), np.cos(0.6)]
    )
    np.testing.assert_allclose(result.colors.encoded_srgb[0], 0.0, atol=1.0e-12)
    assert bool(jnp.all(result.accepted))


def test_appearance_masks_undersampled_interference_before_colorimetry() -> None:
    wavelengths = np.arange(400.0, 701.0, 20.0) / 1.0e9
    interference = ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0)
    plan = ThinFilmAppearancePlan(
        interference,
        SpectralColorimetryPlan(wavelengths, _daylight_like()),
        two_sided=True,
    )

    result = plan.evaluate(
        2.0e-6,
        jnp.asarray([0.0, 0.0, 1.0]),
        jnp.asarray([0.0, 0.0, 1.0]),
    )

    assert int(result.status) == ThinFilmAppearanceStatus.INTERFERENCE_REJECTED
    assert (
        int(result.evidence.interference_status)
        == ThinFilmInterferenceStatus.SPECTRAL_UNDERSAMPLED
    )
    assert int(result.colors.status) == SpectralColorimetryStatus.NONFINITE_SPECTRUM
    assert not bool(result.colors.accepted)
    assert bool(jnp.isfinite(result.evidence.minimum_samples_per_fringe))
    assert bool(jnp.all(jnp.isnan(result.colors.xyz)))
    assert bool(jnp.all(jnp.isnan(result.colors.linear_srgb)))
    assert bool(jnp.all(jnp.isnan(result.colors.display_linear_srgb)))
    assert bool(jnp.all(jnp.isnan(result.colors.encoded_srgb)))


def test_appearance_masks_energy_residual_rejection_before_colorimetry() -> None:
    interference = ThinFilmInterferencePlan(
        VISIBLE,
        1.0,
        1.33,
        1.52,
        required_samples_per_fringe=None,
        energy_tolerance=1.0e-30,
    )
    plan = ThinFilmAppearancePlan(
        interference,
        SpectralColorimetryPlan(VISIBLE, _daylight_like()),
    )

    result = plan.evaluate(
        300.0e-9,
        jnp.asarray([0.0, 0.0, 1.0]),
        jnp.asarray([0.0, 0.0, 1.0]),
    )

    assert int(result.status) == ThinFilmAppearanceStatus.INTERFERENCE_REJECTED
    assert (
        int(result.evidence.interference_status)
        == ThinFilmInterferenceStatus.ENERGY_RESIDUAL_EXCEEDED
    )
    assert int(result.colors.status) == SpectralColorimetryStatus.NONFINITE_SPECTRUM
    assert not bool(result.colors.accepted)
    assert bool(jnp.isfinite(result.evidence.maximum_energy_residual))
    assert bool(jnp.all(jnp.isnan(result.colors.xyz)))
    assert bool(jnp.all(jnp.isnan(result.colors.linear_srgb)))
    assert bool(jnp.all(jnp.isnan(result.colors.display_linear_srgb)))
    assert bool(jnp.all(jnp.isnan(result.colors.encoded_srgb)))


def test_appearance_mixed_batch_preserves_valid_color_and_masks_rejection() -> None:
    wavelengths = np.arange(400.0, 701.0, 20.0) / 1.0e9
    interference = ThinFilmInterferencePlan(wavelengths, 1.0, 1.33, 1.0)
    plan = ThinFilmAppearancePlan(
        interference,
        SpectralColorimetryPlan(wavelengths, _daylight_like()),
        two_sided=True,
    )
    normal = jnp.asarray([0.0, 0.0, 1.0])
    view = jnp.asarray([0.0, 0.0, 1.0])

    result = plan.evaluate(jnp.asarray([100.0e-9, 2.0e-6]), normal, view)
    valid_spectrum = interference.evaluate(100.0e-9, 1.0)
    expected = plan.colorimetry.evaluate(valid_spectrum.unpolarized_reflectance)

    np.testing.assert_array_equal(
        result.status,
        [
            ThinFilmAppearanceStatus.SUCCESS,
            ThinFilmAppearanceStatus.INTERFERENCE_REJECTED,
        ],
    )
    np.testing.assert_array_equal(
        result.evidence.interference_status,
        [
            ThinFilmInterferenceStatus.SUCCESS,
            ThinFilmInterferenceStatus.SPECTRAL_UNDERSAMPLED,
        ],
    )
    np.testing.assert_array_equal(
        result.colors.status,
        [
            SpectralColorimetryStatus.SUCCESS,
            SpectralColorimetryStatus.NONFINITE_SPECTRUM,
        ],
    )
    np.testing.assert_allclose(result.colors.xyz[0], expected.xyz)
    np.testing.assert_allclose(result.colors.linear_srgb[0], expected.linear_srgb)
    np.testing.assert_allclose(result.colors.encoded_srgb[0], expected.encoded_srgb)
    assert bool(jnp.all(jnp.isnan(result.colors.xyz[1])))
    assert bool(jnp.all(jnp.isnan(result.colors.linear_srgb[1])))
    assert bool(jnp.all(jnp.isnan(result.colors.encoded_srgb[1])))


def test_appearance_geometry_statuses_and_directional_light() -> None:
    one_sided = _soap_appearance(two_sided=False)
    two_sided = _soap_appearance(two_sided=True)
    thickness = jnp.full((3,), 300.0e-9)
    normals = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    view = jnp.asarray([0.0, 0.0, 1.0])

    front = one_sided.evaluate(thickness, normals, view)
    np.testing.assert_array_equal(
        front.status,
        [
            ThinFilmAppearanceStatus.SUCCESS,
            ThinFilmAppearanceStatus.BACK_FACING,
            ThinFilmAppearanceStatus.INVALID_GEOMETRY,
        ],
    )
    assert bool(jnp.all(jnp.isnan(front.colors.encoded_srgb[1:])))
    flipped = two_sided.evaluate(thickness, normals, view)
    np.testing.assert_allclose(
        flipped.colors.encoded_srgb[1], front.colors.encoded_srgb[0], rtol=1e-12
    )

    angle = 0.4
    mirror_view = jnp.asarray([0.0, np.sin(angle), np.cos(angle)])
    mirror_light = jnp.asarray([0.0, -np.sin(angle), np.cos(angle)])
    off_light = jnp.asarray([0.0, 0.0, 1.0])
    lit = one_sided.evaluate(
        thickness[:2],
        jnp.asarray([0.0, 0.0, 1.0]),
        mirror_view,
        jnp.stack((mirror_light, off_light)),
    )
    np.testing.assert_allclose(lit.evidence.facet_alignment[0], 1.0, atol=1.0e-12)
    np.testing.assert_allclose(lit.evidence.incidence_cosine[0], np.cos(angle))
    assert float(lit.evidence.facet_alignment[1]) < 1.0
    np.testing.assert_allclose(
        lit.evidence.incidence_cosine[1], np.cos(angle / 2.0), rtol=1.0e-12
    )

    with pytest.raises(ValueError, match="two_sided"):
        ThinFilmAppearancePlan(
            ThinFilmInterferencePlan(VISIBLE, 1.0, 1.33, 1.52),
            one_sided.colorimetry,
            two_sided=True,
        )
    with pytest.raises(ValueError, match="identical wavelengths"):
        ThinFilmAppearancePlan(
            ThinFilmInterferencePlan(VISIBLE[1:], 1.0, 1.33, 1.0),
            one_sided.colorimetry,
        )
