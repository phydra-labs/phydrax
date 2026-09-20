from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.astrophysics._gr_interchange import (
    NeutralArrayPayload,
    stokes_image_from_fits_payload,
    stokes_image_to_fits_payload,
    visibility_data_from_uvfits_payload,
    visibility_data_to_uvfits_payload,
)
from phydrax.applications.astrophysics._gr_products import GRImageScreen, StokesImage
from phydrax.applications.astrophysics._interferometry import (
    apply_station_gains,
    closure_products,
    ClosureTopology,
    direct_stokes_visibilities,
    InterferometryStatus,
    polarization_visibility_products,
    StokesVisibilityData,
    VisibilitySampling,
)
from phydrax.applications.astrophysics._photometry import ObservationDataProvenance
from phydrax.units import (
    derived_unit,
    DIMENSIONLESS,
    HERTZ,
    JANSKY,
    MASS,
    RADIAN,
    SI_REFERENCE_SYSTEM_ID,
    TIME,
    UnitDefinition,
)


STERADIAN = derived_unit("sr", ((RADIAN, 2),))
FLUX_DENSITY = JANSKY
INTENSITY = derived_unit("Jy/sr", ((JANSKY, 1), (STERADIAN, -1)))
FREQUENCY = 230.0e9


def test_jansky_has_exact_physical_si_contract():
    assert JANSKY.dimension == MASS / TIME**2
    assert JANSKY.scale_to_reference == Fraction(1, 10**26)


def _image(
    stokes,
    coordinates,
    solid_angle,
    *,
    source_id="synthetic-gr-image",
    intensity_unit=INTENSITY,
    flux_density_unit=FLUX_DENSITY,
):
    shape = np.asarray(solid_angle).shape
    screen = GRImageScreen(
        coordinates,
        solid_angle,
        np.ones(shape, dtype="bool"),
        angular_unit=RADIAN,
        solid_angle_unit=STERADIAN,
    )
    return StokesImage(
        stokes,
        screen,
        ObservationDataProvenance.native(source_id),
        frequency=FREQUENCY,
        redshift=np.ones(shape),
        redshift_valid=np.ones(shape, dtype="bool"),
        lensing_masks=np.ones((1, *shape), dtype="bool"),
        lensing_labels=("direct",),
        intensity_unit=intensity_unit,
        flux_density_unit=flux_density_unit,
        frequency_unit=HERTZ,
    )


def _sampling(uv, pairs, station_ids):
    return VisibilitySampling(uv, pairs, FREQUENCY, station_ids)


def test_direct_point_source_has_exact_flux_and_fourier_phase():
    position = np.asarray([0.125, -0.25])
    solid_angle = np.asarray([[0.2]])
    flux = 3.5
    stokes = np.zeros((4, 1, 1))
    stokes[0, 0, 0] = flux / solid_angle[0, 0]
    stokes[1, 0, 0] = 0.5 / solid_angle[0, 0]
    image = _image(stokes, position.reshape((1, 1, 2)), solid_angle)
    uv = np.asarray([[0.0, 0.0], [1.5, -0.5]])
    sampling = _sampling(uv, [[0, 1], [0, 1]], ("A", "B"))

    result = direct_stokes_visibilities(image, sampling)

    phase = np.exp(-2j * np.pi * (uv @ position))
    np.testing.assert_allclose(result.visibilities[0], flux * phase, rtol=2.0e-6)
    np.testing.assert_allclose(result.visibilities[1], 0.5 * phase, rtol=2.0e-6)
    np.testing.assert_array_equal(result.visibilities[2:], 0.0)
    assert result.parent_product_ids == (image.content_id,)


def test_direct_centered_gaussian_matches_analytic_fourier_transform():
    sigma = 0.02
    flux = 2.75
    axis = np.linspace(-0.2, 0.2, 201)
    spacing = axis[1] - axis[0]
    x_coordinate, y_coordinate = np.meshgrid(axis, axis)
    radius_squared = x_coordinate**2 + y_coordinate**2
    intensity = flux / (2.0 * np.pi * sigma**2) * np.exp(-0.5 * radius_squared / sigma**2)
    stokes = np.zeros((4, axis.size, axis.size))
    stokes[0] = intensity
    coordinates = np.stack((x_coordinate, y_coordinate), axis=-1)
    solid_angle = np.full(intensity.shape, spacing**2)
    image = _image(stokes, coordinates, solid_angle)
    uv = np.asarray([[0.0, 0.0], [2.0, -1.5], [-3.0, 4.0]])
    sampling = _sampling(uv, [[0, 1], [0, 1], [0, 1]], ("A", "B"))

    result = direct_stokes_visibilities(image, sampling)

    expected = flux * np.exp(-2.0 * np.pi**2 * sigma**2 * np.sum(uv**2, axis=1))
    np.testing.assert_allclose(result.total_intensity.real, expected, rtol=2.0e-5)
    np.testing.assert_allclose(result.total_intensity.imag, 0.0, atol=1.0e-6)


def _closure_fixture():
    station_ids = ("A", "B", "C", "D")
    pairs = np.asarray(
        [
            [0, 1],  # AB
            [1, 2],  # BC
            [0, 2],  # AC; conjugated for CA
            [2, 3],  # CD
            [1, 3],  # BD
        ]
    )
    sampling = _sampling(np.zeros((5, 2)), pairs, station_ids)
    topology = ClosureTopology.from_station_cycles(
        sampling,
        phase_cycles=(("A", "B", "C"),),
        amplitude_cycles=(("A", "B", "C", "D"),),
    )
    intensity = np.asarray(
        [
            2.0 * np.exp(0.2j),
            3.0 * np.exp(-0.4j),
            5.0 * np.exp(0.1j),
            7.0 * np.exp(0.3j),
            11.0 * np.exp(-0.2j),
        ]
    )
    values = np.zeros((4, 5), dtype="complex128")
    values[0] = intensity
    data = StokesVisibilityData(
        values,
        sampling,
        FLUX_DENSITY,
        ObservationDataProvenance.native("closure-fixture"),
    )
    return data, topology


def test_bispectrum_and_closures_are_exact_and_station_gain_invariant():
    data, topology = _closure_fixture()
    result = closure_products(data, topology)
    intensity = np.asarray(data.total_intensity)
    expected_bispectrum = intensity[0] * intensity[1] * np.conj(intensity[2])
    expected_amplitude = abs(intensity[0] * intensity[3]) / abs(
        intensity[2] * intensity[4]
    )
    np.testing.assert_allclose(result.bispectrum, [expected_bispectrum], rtol=2.0e-6)
    np.testing.assert_allclose(result.closure_phase, [np.angle(expected_bispectrum)])
    np.testing.assert_allclose(result.closure_amplitude, [expected_amplitude])
    np.testing.assert_allclose(result.log_closure_amplitude, [np.log(expected_amplitude)])

    gains = np.asarray(
        [
            1.2 * np.exp(0.7j),
            0.8 * np.exp(-0.5j),
            1.7 * np.exp(0.1j),
            0.6 * np.exp(-0.8j),
        ]
    )
    transformed = closure_products(apply_station_gains(data, gains), topology)
    np.testing.assert_allclose(
        transformed.closure_phase, result.closure_phase, atol=2.0e-6
    )
    np.testing.assert_allclose(
        transformed.closure_amplitude, result.closure_amplitude, rtol=2.0e-6
    )
    np.testing.assert_allclose(
        transformed.log_closure_amplitude,
        result.log_closure_amplitude,
        atol=2.0e-6,
    )
    assert bool(result.phase_physically_valid[0])
    assert bool(result.amplitude_physically_valid[0])


def test_zero_visibility_has_explicit_safe_failure_status():
    data, topology = _closure_fixture()
    values = np.asarray(data.visibilities).copy()
    values[0, 0] = 0.0
    zero = StokesVisibilityData(
        values,
        data.sampling,
        data.visibility_unit,
        data.provenance,
    )

    result = closure_products(zero, topology)

    assert int(result.phase_status[0]) == int(InterferometryStatus.ZERO_AMPLITUDE)
    assert int(result.amplitude_status[0]) == int(InterferometryStatus.ZERO_AMPLITUDE)
    assert not bool(result.phase_physically_valid[0])
    assert not bool(result.amplitude_physically_valid[0])
    assert not bool(result.phase_derivative_valid[0])
    assert not bool(result.amplitude_derivative_valid[0])
    np.testing.assert_array_equal(result.closure_phase, 0.0)
    np.testing.assert_array_equal(result.closure_amplitude, 0.0)
    np.testing.assert_array_equal(result.log_closure_amplitude, 0.0)
    assert bool(jnp.all(jnp.isfinite(result.bispectrum)))


def test_polarization_products_use_physical_stokes_correlations_and_safe_ratios():
    sampling = _sampling([[0.0, 0.0]], [[0, 1]], ("A", "B"))
    values = np.asarray([[2.0], [0.5], [0.25], [0.1]], dtype="complex128")
    data = StokesVisibilityData(
        values,
        sampling,
        FLUX_DENSITY,
        ObservationDataProvenance.native("polarization-fixture"),
    )

    products = polarization_visibility_products(data)

    np.testing.assert_allclose(products.complex_linear_polarization, [0.5 + 0.25j])
    np.testing.assert_allclose(
        products.fractional_linear_polarization, [(0.5 + 0.25j) / 2.0]
    )
    np.testing.assert_allclose(products.fractional_circular_polarization, [0.05])
    np.testing.assert_allclose(
        products.circular_correlations[:, 0],
        [2.1, 1.9, 0.5 + 0.25j, 0.5 - 0.25j],
    )
    np.testing.assert_allclose(
        products.linear_correlations[:, 0],
        [2.5, 1.5, 0.25 + 0.1j, 0.25 - 0.1j],
    )
    zero = StokesVisibilityData(
        np.zeros((4, 1), dtype="complex128"),
        sampling,
        FLUX_DENSITY,
        data.provenance,
    )
    undefined = polarization_visibility_products(zero)
    assert int(undefined.status[0]) == int(InterferometryStatus.ZERO_AMPLITUDE)
    np.testing.assert_array_equal(undefined.fractional_linear_polarization, 0.0)
    np.testing.assert_array_equal(undefined.fractional_circular_polarization, 0.0)


def test_content_identities_bind_units_provenance_and_fixed_topology():
    coordinates = np.zeros((1, 1, 2))
    stokes = np.asarray([[[1.0]], [[0.0]], [[0.0]], [[0.0]]])
    image = _image(stokes, coordinates, np.ones((1, 1)))
    other_provenance = _image(
        stokes, coordinates, np.ones((1, 1)), source_id="other-source"
    )
    other_flux_unit = UnitDefinition("arb", DIMENSIONLESS, SI_REFERENCE_SYSTEM_ID)
    other_intensity_unit = derived_unit("arb/sr", ((other_flux_unit, 1), (STERADIAN, -1)))
    other_units = _image(
        stokes,
        coordinates,
        np.ones((1, 1)),
        intensity_unit=other_intensity_unit,
        flux_density_unit=other_flux_unit,
    )
    assert image.content_id != other_provenance.content_id
    assert image.content_id != other_units.content_id

    first = _sampling([[0.0, 0.0]], [[0, 1]], ("A", "B"))
    moved = _sampling([[1.0, 0.0]], [[0, 1]], ("A", "B"))
    renamed = _sampling([[0.0, 0.0]], [[0, 1]], ("left", "right"))
    assert first.topology_id != moved.topology_id
    assert first.topology_id != renamed.topology_id
    visibility_values = np.ones((4, 1), dtype="complex128")
    provenance = ObservationDataProvenance.native("visibility-source")
    visibility = StokesVisibilityData(visibility_values, first, FLUX_DENSITY, provenance)
    changed_unit = StokesVisibilityData(
        visibility_values, first, other_flux_unit, provenance
    )
    changed_provenance = StokesVisibilityData(
        visibility_values,
        first,
        FLUX_DENSITY,
        ObservationDataProvenance.native("other-visibility-source"),
    )
    changed_topology = StokesVisibilityData(
        visibility_values, moved, FLUX_DENSITY, provenance
    )
    assert visibility.content_id != changed_unit.content_id
    assert visibility.content_id != changed_provenance.content_id
    assert visibility.content_id != changed_topology.content_id


def test_fits_and_uvfits_neutral_payloads_round_trip_without_rendering_state():
    position = np.asarray([[[0.1, -0.2]]])
    stokes = np.asarray([[[2.0]], [[0.2]], [[0.1]], [[0.0]]])
    image = _image(stokes, position, np.asarray([[0.5]]))
    image_payload = stokes_image_to_fits_payload(image)

    assert isinstance(image_payload, NeutralArrayPayload)
    assert "rendering" not in image_payload.metadata
    assert "rgb" not in image_payload.metadata
    restored_image = stokes_image_from_fits_payload(image_payload)
    assert restored_image.content_id == image.content_id
    np.testing.assert_array_equal(restored_image.stokes, image.stokes)

    sampling = _sampling([[0.0, 0.0], [1.0, -2.0]], [[0, 1], [0, 1]], ("A", "B"))
    visibility = direct_stokes_visibilities(image, sampling)
    uv_payload = visibility_data_to_uvfits_payload(visibility)
    restored_visibility = visibility_data_from_uvfits_payload(uv_payload)
    assert restored_visibility.content_id == visibility.content_id
    assert restored_visibility.sampling.topology_id == sampling.topology_id
    np.testing.assert_array_equal(
        restored_visibility.visibilities, visibility.visibilities
    )

    tampered_arrays = list(image_payload.arrays)
    tampered_arrays[0] = tampered_arrays[0].at[0, 0, 0].add(1.0)
    tampered = NeutralArrayPayload(
        tampered_arrays, image_payload.array_names, image_payload.metadata
    )
    with pytest.raises(ValueError, match="identity does not match"):
        stokes_image_from_fits_payload(tampered)
