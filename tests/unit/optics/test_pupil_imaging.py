#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave._fields import PlaneFieldSpace, ScalarPlaneField
from phydrax.optics.wave._imaging import (
    fraunhofer_psf,
    FraunhoferImagingPlan,
    normalized_otf_mtf,
    strehl_ratio,
)
from phydrax.optics.wave._pupil import (
    apply_pupil_opd,
    evaluate_noll_zernike_opd,
    noll_to_radial_azimuthal,
    noll_zernike,
    NollZernikeOPD,
)


def _finite_space(count, lower, upper, *, z=0.0):
    grid = TensorGridPlan(
        (UniformAxisSpec(count), UniformAxisSpec(count)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((lower, lower), (upper, upper))))
    return PlaneFieldSpace(
        grid,
        RigidFrame(jnp.eye(3), jnp.asarray((0.0, 0.0, z))),
        "finite-window",
    )


def test_noll_mapping_and_modes_are_continuous_unit_rms():
    space = _finite_space(129, -1.05, 1.05)
    result = evaluate_noll_zernike_opd(
        space,
        NollZernikeOPD(tuple(range(1, 7)), jnp.zeros((6,)), 1.0),
    )

    assert tuple(noll_to_radial_azimuthal(index) for index in range(1, 7)) == (
        (0, 0),
        (1, 1),
        (1, -1),
        (2, 0),
        (2, -2),
        (2, 2),
    )
    np.testing.assert_allclose(result.evidence.discrete_mode_rms, 1.0, atol=2.5e-2)
    np.testing.assert_allclose(
        result.evidence.discrete_mode_means[1:],
        0.0,
        atol=2.5e-2,
    )
    assert bool(result.evidence.adequate)


def test_piston_and_tilt_have_exact_noll_normalization():
    coordinates = jnp.asarray(((0.0, 0.0), (0.25, 0.0), (0.0, -0.25)))

    np.testing.assert_allclose(noll_zernike(1, coordinates), (1.0, 1.0, 1.0))
    np.testing.assert_allclose(noll_zernike(2, coordinates), (0.0, 0.5, 0.0))
    np.testing.assert_allclose(noll_zernike(3, coordinates), (0.0, 0.0, -0.5))


def _fraunhofer_case():
    pupil_space = _finite_space(65, -0.5, 0.5)
    image_space = _finite_space(101, -4.0, 4.0, z=1.0)
    radius = jnp.sqrt(jnp.sum(pupil_space.transverse_coordinates**2, axis=-1))
    aperture = radius <= 0.5
    field = ScalarPlaneField(pupil_space, aperture.astype(float), 1.0, 0.0)
    prepared = FraunhoferImagingPlan(
        pupil_space,
        image_space,
        1.0,
        2.0 * jnp.pi,
        1.0,
    ).prepare()
    return pupil_space, image_space, aperture, field, prepared


def test_fraunhofer_circular_pupil_has_airy_null_and_unit_power():
    _, image_space, _, field, prepared = _fraunhofer_case()
    result = fraunhofer_psf(prepared, field)
    axis = np.asarray(image_space.coordinate_axes[0])
    center = image_space.shape[1] // 2
    null_index = int(np.argmin(np.abs(axis - 1.22)))
    peak = float(result.plane.values[image_space.shape[0] // 2, center])
    first_null = float(result.plane.values[null_index, center])
    integrated_power = jnp.sum(result.plane.values * image_space.area_weights)

    np.testing.assert_allclose(integrated_power, 1.0, rtol=2e-6)
    assert first_null / peak < 1.5e-2
    assert float(result.sampling.samples_per_airy_radius) > 2.0
    assert bool(result.valid)


def test_fraunhofer_prepared_dft_matches_direct_quadrature():
    pupil = _finite_space(5, -0.5, 0.5)
    image = _finite_space(7, -0.4, 0.4, z=0.8)
    coordinates = pupil.transverse_coordinates
    values = (1.0 + coordinates[..., 0]) * jnp.exp(0.3j * coordinates[..., 1])
    field = ScalarPlaneField(pupil, values, 2.0, 0.0)
    plan = FraunhoferImagingPlan(pupil, image, 0.8, 8.0, 0.8).prepare()
    result = fraunhofer_psf(plan, field)
    target = image.transverse_coordinates[1, 5]
    phase = jnp.exp(
        -1j * 8.0 / 0.8 * jnp.sum(coordinates * target[None, None, :], axis=-1)
    )
    amplitude = jnp.sum(values * phase * pupil.area_weights)
    expected = jnp.abs(8.0 / (2.0 * jnp.pi * 0.8) * amplitude) ** 2

    np.testing.assert_allclose(result.raw_intensity[1, 5], expected, rtol=2e-6)


def test_circular_pupil_mtf_matches_diffraction_limited_formula():
    _, _, _, field, prepared = _fraunhofer_case()
    transfer = normalized_otf_mtf(fraunhofer_psf(prepared, field).plane)
    frequencies = np.asarray(transfer.frequency_axes[0])
    center = transfer.modulation_transfer_function.shape[1] // 2
    index = int(np.argmin(np.abs(frequencies - 0.5)))
    normalized_frequency = abs(frequencies[index])
    expected = (
        2.0
        / np.pi
        * (
            np.arccos(normalized_frequency)
            - normalized_frequency * np.sqrt(1.0 - normalized_frequency**2)
        )
    )

    np.testing.assert_allclose(
        transfer.modulation_transfer_function[index, center],
        expected,
        atol=8e-2,
    )
    assert float(transfer.evidence.hermitian_error) < 1e-5
    assert bool(transfer.evidence.valid)


def test_strehl_is_piston_invariant_and_detects_defocus():
    pupil_space, _, aperture, field, prepared = _fraunhofer_case()
    reference = fraunhofer_psf(prepared, field).plane
    piston = evaluate_noll_zernike_opd(
        pupil_space,
        NollZernikeOPD((1,), (0.3,), 0.5),
    )
    defocus = evaluate_noll_zernike_opd(
        pupil_space,
        NollZernikeOPD((4,), (0.15,), 0.5),
    )
    piston_field = apply_pupil_opd(field, piston, 2.0 * jnp.pi)
    defocus_field = apply_pupil_opd(field, defocus, 2.0 * jnp.pi)
    piston_psf = fraunhofer_psf(prepared, piston_field).plane
    defocus_psf = fraunhofer_psf(prepared, defocus_field).plane
    piston_strehl = strehl_ratio(piston_psf, reference)
    defocus_strehl = strehl_ratio(defocus_psf, reference)

    np.testing.assert_allclose(jnp.abs(piston_field.values), aperture, atol=1e-6)
    np.testing.assert_allclose(piston_strehl.ratio, 1.0, rtol=2e-5)
    assert bool(piston_strehl.sampling.adequate)
    assert 0.0 < float(defocus_strehl.ratio) < 1.0
    assert bool(piston_strehl.valid)
    assert bool(defocus_strehl.valid)
