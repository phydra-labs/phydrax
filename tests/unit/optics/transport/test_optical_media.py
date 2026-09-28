#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from scipy import stats

from phydrax._physical import RelativityScaleContract
from phydrax.optics.geometric._nonsequential import NonSequentialSurfaceTable
from phydrax.optics.transport._optical_media import (
    HenyeyGreensteinScattering,
    lorenz_mie,
    MieParticles,
    RayleighScattering,
    SpectralOpticalMedium,
    WavelengthShifter,
)
from phydrax.optics.transport._optical_monte_carlo import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalTransportStatus,
    prepare_optical_monte_carlo,
    rotate_jones_to_scattering_frame,
    simulate_optical_photons,
)


# Wiscombe, "Mie scattering calculations: advances in technique and fast,
# vector-speed computer codes", NCAR/TN-140+STR (1979, revised 1996), MIEV0
# test cases as printed (six decimals), transcribed in S. Prahl's miepython
# test suite. MIEV0 uses m = n - i kappa and conjugate amplitudes; the same
# spheres are m = n + i kappa here. Columns: x, m, Q_ext, Q_sca, g, and the
# MIEV0 backscattering amplitude S1(180 deg).
_MIEV0_CASES = (
    ("case05", 0.099, 0.75 + 0.0j, None, 0.000007, 0.001448, 1.81756e-8 - 1.64810e-4j),
    ("case06", 0.101, 0.75 + 0.0j, None, 0.000008, 0.001507, 2.04875e-8 - 1.74965e-4j),
    ("case07", 10.0, 0.75 + 0.0j, None, 2.232265, 0.896473, -1.07857 - 3.60881e-2j),
    ("case08", 1000.0, 0.75 + 0.0j, None, 1.997908, 0.844944, 1.70578e1 + 4.84251e2j),
    ("case09", 1.0, 1.33 + 1e-5j, None, 0.093923, 0.184517, None),
    ("case10", 100.0, 1.33 + 1e-5j, None, 2.096594, 0.868959, None),
    ("case11", 10000.0, 1.33 + 1e-5j, None, 1.723857, 0.907840, None),
    ("case12", 0.055, 1.5 + 1.0j, 0.101491, 0.000011, 0.000491, None),
    ("case13", 0.056, 1.5 + 1.0j, 0.1033467, 0.000012, 0.000509, None),
    ("case14", 1.0, 1.5 + 1.0j, 2.336321, 0.6634538, 0.192136, None),
    ("case15", 100.0, 1.5 + 1.0j, 2.097502, 1.283697, 0.850252, None),
    ("case16", 10000.0, 1.5 + 1.0j, 2.004368, 1.236575, 0.846309, None),
    ("case17", 1.0, 10.0 + 10.0j, None, 2.049405, -0.110664, None),
    ("case18", 100.0, 10.0 + 10.0j, None, 1.836785, 0.556215, None),
    ("case19", 10000.0, 10.0 + 10.0j, None, 1.795393, 0.548194, None),
)

# Wiscombe MIEV0 angular case (x = 1, m = 1.5 - 1i), theta = 0, 30, ..., 180.
_MIEV0_S1 = np.asarray(
    [
        0.584080 + 0.190515j,
        0.565702 + 0.187200j,
        0.517525 + 0.178443j,
        0.456340 + 0.167167j,
        0.400212 + 0.156643j,
        0.362157 + 0.149391j,
        0.348844 + 0.146829j,
    ]
)
_MIEV0_S2 = np.asarray(
    [
        0.584080 + 0.190515j,
        0.500161 + 0.145611j,
        0.287964 + 0.041054j,
        0.0362285 - 0.0618265j,
        -0.174875 - 0.122959j,
        -0.305682 - 0.143846j,
        -0.348844 - 0.146829j,
    ]
)

_COUNT = 100_000
_LINEAR = np.asarray([1.0, 0.0], dtype=np.complex128)
_CIRCULAR = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)


def _keys(count: int, seed: int) -> Array:
    root = jr.key(seed)
    return jax.vmap(lambda index: jr.fold_in(root, index))(jnp.arange(count))


def _lanes(count: int) -> Array:
    return jnp.zeros((count,), dtype=jnp.int32)


def _scatter(
    medium: SpectralOpticalMedium,
    wavelength: float,
    jones: np.ndarray,
    count: int = _COUNT,
    seed: int = 0,
) -> Any:
    return medium.scatter(
        _lanes(count),
        jnp.full((count,), wavelength),
        jnp.asarray(np.broadcast_to(jones, (count, 2))),
        _keys(count, seed),
    )


def _scattered_geometry(
    cosines: np.ndarray, azimuths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scattered direction and new frame for incidence along z with e1 = x."""

    sine = np.sqrt(1.0 - cosines * cosines)
    azimuthal = np.stack((np.cos(azimuths), np.sin(azimuths), 0.0 * azimuths), axis=-1)
    direction = cosines[:, None] * np.asarray([0.0, 0.0, 1.0]) + sine[:, None] * azimuthal
    s_axis = np.cross(np.asarray([0.0, 0.0, 1.0]), azimuthal)
    return direction, s_axis, np.cross(direction, s_axis)


def _grid_medium(**processes: Any) -> SpectralOpticalMedium:
    grid = np.asarray([300e-9, 500e-9, 700e-9])
    return SpectralOpticalMedium(
        grid,
        np.full((1, 3), 1.33),
        np.full((1, 3), np.inf),
        **processes,
    )


@pytest.mark.parametrize(
    ("x", "m", "extinction", "scattering", "asymmetry", "backscatter"),
    [case[1:] for case in _MIEV0_CASES],
    ids=[case[0] for case in _MIEV0_CASES],
)
def test_lorenz_mie_reproduces_wiscombe_miev0_cases(
    x: float,
    m: complex,
    extinction: float | None,
    scattering: float,
    asymmetry: float,
    backscatter: complex | None,
) -> None:
    result = lorenz_mie(x, m, np.asarray([-1.0]))
    tolerance = 1.5e-6
    if extinction is not None:
        assert abs(float(result.extinction_efficiency) - extinction) < tolerance
    assert abs(float(result.scattering_efficiency) - scattering) < tolerance
    assert abs(float(result.asymmetry) - asymmetry) < tolerance
    if backscatter is not None:
        np.testing.assert_allclose(complex(result.s1[0]), np.conj(backscatter), rtol=2e-5)
        np.testing.assert_allclose(
            float(result.backscattering_efficiency),
            abs(2.0 * backscatter / x) ** 2,
            rtol=4e-5,
        )
    assert float(result.absorption_efficiency) >= -1e-12


def test_lorenz_mie_amplitudes_reproduce_the_wiscombe_angular_case() -> None:
    cosines = np.cos(np.deg2rad(np.arange(0.0, 181.0, 30.0)))
    result = lorenz_mie(1.0, 1.5 + 1.0j, cosines)
    # Bohren--Huffman amplitudes are the conjugates of MIEV0's (m = n - i kappa).
    np.testing.assert_allclose(np.asarray(result.s1), np.conj(_MIEV0_S1), atol=1.5e-6)
    np.testing.assert_allclose(np.asarray(result.s2), np.conj(_MIEV0_S2), atol=1.5e-6)


def test_lorenz_mie_small_sphere_limit_is_the_rayleigh_dipole() -> None:
    x = 1e-3
    m = 1.5 + 0.1j
    cosines = np.linspace(-1.0, 1.0, 9)
    result = lorenz_mie(x, m, cosines)
    polarizability = (m * m - 1.0) / (m * m + 2.0)
    np.testing.assert_allclose(
        float(result.scattering_efficiency),
        8.0 / 3.0 * x**4 * abs(polarizability) ** 2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        float(result.absorption_efficiency), 4.0 * x * polarizability.imag, rtol=1e-5
    )
    ratio = np.asarray(result.s2) / np.asarray(result.s1)
    np.testing.assert_allclose(ratio.real, cosines, atol=1e-5)
    np.testing.assert_allclose(ratio.imag, 0.0, atol=1e-5)
    assert abs(float(result.asymmetry)) < 1e-5


@pytest.mark.parametrize(
    ("x", "m"),
    [(0.5, 1.33 + 0.0j), (3.0, 1.5 + 0.01j), (25.0, 1.59 + 0.5j)],
    ids=["small-dielectric", "weak-absorber", "strong-absorber"],
)
def test_lorenz_mie_satisfies_optical_theorem_and_angular_normalization(
    x: float, m: complex
) -> None:
    nodes, weights = np.polynomial.legendre.leggauss(1200)
    result = lorenz_mie(x, m, np.concatenate(([1.0], nodes)))
    s1 = np.asarray(result.s1)
    s2 = np.asarray(result.s2)
    # Optical theorem: forward amplitude sum against the coefficient series.
    np.testing.assert_allclose(
        float(result.extinction_efficiency), 4.0 / x**2 * s1[0].real, rtol=1e-12
    )
    np.testing.assert_allclose(s1[0], s2[0], rtol=1e-12)
    intensity = np.abs(s1[1:]) ** 2 + np.abs(s2[1:]) ** 2
    np.testing.assert_allclose(
        np.sum(weights * intensity) / x**2,
        float(result.scattering_efficiency),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.sum(weights * nodes * intensity) / np.sum(weights * intensity),
        float(result.asymmetry),
        rtol=1e-10,
    )
    if m.imag == 0.0:
        assert abs(float(result.absorption_efficiency)) < 1e-12
    else:
        assert float(result.absorption_efficiency) > 0.0


@pytest.mark.parametrize(
    "x", [0.5, 5.0, 100.0, 5000.0], ids=["x0.5", "x5", "x100", "x5000"]
)
def test_lorenz_mie_reports_wiscombe_truncation_evidence(x: float) -> None:
    result = lorenz_mie(x, 1.33 + 0.0j, np.asarray([1.0]))
    # Wiscombe (1980) eq. (1): three size-parameter regimes of N_stop.
    if x <= 8.0:
        expected = math.floor(x + 4.0 * x ** (1.0 / 3.0) + 1.0)
    elif x < 4200.0:
        expected = math.floor(x + 4.05 * x ** (1.0 / 3.0) + 2.0)
    else:
        expected = math.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    assert result.term_count == expected
    assert result.a_coefficients.shape == (expected,)
    assert result.lentz_iterations >= 2
    assert result.truncation_contribution < 1e-10


def test_lorenz_mie_refuses_gain_media_and_empty_spheres() -> None:
    with pytest.raises(ValueError, match="relative indices"):
        lorenz_mie(1.0, 1.5 - 0.1j, np.asarray([1.0]))
    with pytest.raises(ValueError, match="size parameters"):
        lorenz_mie(0.0, 1.5 + 0.0j, np.asarray([1.0]))
    with pytest.raises(ValueError, match="cosines"):
        lorenz_mie(1.0, 1.5 + 0.0j, np.asarray([1.5]))


def test_rayleigh_samples_the_polarized_dipole_law() -> None:
    medium = _grid_medium(rayleigh=RayleighScattering(np.full((1, 3), 2.0)))
    sample = _scatter(medium, 450e-9, _LINEAR)
    cosines = np.asarray(sample.cosines)
    azimuths = np.asarray(sample.azimuths)
    polar = stats.kstest(cosines, lambda mu: (mu**3 / 3.0 + mu + 4.0 / 3.0) / (8.0 / 3.0))
    assert polar.pvalue > 1e-3
    # A dipole driven along e1 radiates as sin^2 of the angle to e1:
    # the cosine c to e1 has density 3 (1 - c^2) / 4 and E[c^2] = 1/5.
    direction, _, _ = _scattered_geometry(cosines, azimuths)
    along_field = direction[:, 0]
    dipole = stats.kstest(along_field, lambda c: 0.75 * (c - c**3 / 3.0) + 0.5)
    assert dipole.pvalue > 1e-3
    second = along_field**2
    assert abs(second.mean() - 0.2) < 4.0 * second.std() / np.sqrt(_COUNT)


def test_rayleigh_azimuth_is_uniform_for_circular_polarization() -> None:
    medium = _grid_medium(rayleigh=RayleighScattering(np.full((1, 3), 2.0)))
    sample = _scatter(medium, 450e-9, _CIRCULAR, seed=3)
    azimuths = np.mod(np.asarray(sample.azimuths), 2.0 * np.pi)
    assert stats.kstest(azimuths, "uniform", args=(0.0, 2.0 * np.pi)).pvalue > 1e-3


@pytest.mark.parametrize(
    "jones", [_LINEAR, _CIRCULAR, np.asarray([0.6, 0.8j])], ids=["x", "circular", "ell"]
)
def test_rayleigh_jones_vector_is_the_transverse_dipole_field(jones: np.ndarray) -> None:
    medium = _grid_medium(rayleigh=RayleighScattering(np.full((1, 3), 2.0)))
    count = 4096
    sample = _scatter(medium, 450e-9, jones, count=count, seed=5)
    direction, s_axis, p_axis = _scattered_geometry(
        np.asarray(sample.cosines), np.asarray(sample.azimuths)
    )
    scattered = np.asarray(sample.jones_vectors)
    field = scattered[:, :1] * s_axis + scattered[:, 1:] * p_axis
    incident = jones[0] * np.asarray([1.0, 0.0, 0.0]) + jones[1] * np.asarray(
        [0.0, 1.0, 0.0]
    )
    radiated = incident - np.sum(direction * incident, axis=-1)[:, None] * direction
    radiated /= np.linalg.norm(radiated, axis=-1)[:, None]
    np.testing.assert_allclose(field, radiated, atol=1e-12)
    np.testing.assert_allclose(np.asarray(sample.wavelengths), 450e-9)
    np.testing.assert_allclose(np.asarray(sample.delays), 0.0)


def _mie_medium(angle_count: int | None = None) -> SpectralOpticalMedium:
    # 0.5 um polystyrene spheres in water; wavelengths in m, lengths in cm.
    return _grid_medium(
        mie=MieParticles(
            np.asarray([0.5e-4]),
            np.full((1, 3), 1.59 + 0.001j),
            np.asarray([1e8]),
            length_per_wavelength_unit=100.0,
            angle_count=angle_count,
        )
    )


def _reference_phase(x: float, m: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent fine-grid CDF and polarization of the unpolarized phase law."""

    cosines = np.cos(np.linspace(np.pi, 0.0, 200_001))
    result = lorenz_mie(x, m, cosines)
    perpendicular = np.abs(np.asarray(result.s1)) ** 2
    parallel = np.abs(np.asarray(result.s2)) ** 2
    intensity = 0.5 * (perpendicular + parallel)
    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * np.diff(cosines) * (intensity[1:] + intensity[:-1])))
    )
    polarization = (perpendicular - parallel) / (perpendicular + parallel)
    return cosines, cumulative / cumulative[-1], polarization


def test_mie_medium_couples_series_coefficients_and_interpolates_spectrally() -> None:
    medium = _mie_medium()
    evidence = medium.mie_evidence
    assert evidence is not None
    x = 2.0 * np.pi * 0.5e-4 * 1.33 / (np.asarray([300e-9, 500e-9, 700e-9]) * 100.0)
    np.testing.assert_allclose(np.asarray(evidence.size_parameters[0]), x, rtol=1e-14)
    series = [lorenz_mie(value, (1.59 + 0.001j) / 1.33, np.asarray([1.0])) for value in x]
    area = 1e8 * np.pi * (0.5e-4) ** 2
    extinction = area * np.asarray([float(r.extinction_efficiency) for r in series])
    albedo = np.asarray(
        [float(r.scattering_efficiency / r.extinction_efficiency) for r in series]
    )
    lanes = _lanes(4)
    queries = jnp.asarray([300e-9, 500e-9, 700e-9, 400e-9])
    np.testing.assert_allclose(
        np.asarray(medium.extinction(lanes, queries)),
        np.append(extinction, 0.5 * (extinction[0] + extinction[1])),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(medium.albedo(lanes, queries))[:3], albedo, rtol=1e-12
    )
    assert float(np.max(np.asarray(evidence.normalization_residuals))) < 1e-3
    assert float(np.max(np.asarray(evidence.truncation_contributions))) < 1e-10
    assert evidence.angle_count >= 1025


def test_mie_cosines_follow_the_phase_function_at_and_between_nodes() -> None:
    medium = _mie_medium()
    evidence = medium.mie_evidence
    assert evidence is not None
    references = [
        _reference_phase(
            float(evidence.size_parameters[0, node]),
            complex(evidence.relative_indices[0, node]),
        )[:2]
        for node in (0, 1)
    ]

    def node_cdf(node: int) -> Any:
        cosines, cumulative = references[node]
        return lambda mu: np.interp(mu, cosines, cumulative)

    at_node = np.asarray(_scatter(medium, 300e-9, _LINEAR, seed=7).cosines)
    assert stats.kstest(at_node, node_cdf(0)).pvalue > 1e-3
    # A quarter of the way to the next node the law is the 3:1 node mixture.
    between = np.asarray(_scatter(medium, 350e-9, _LINEAR, seed=8).cosines)
    mixture = stats.kstest(
        between, lambda mu: 0.75 * node_cdf(0)(mu) + 0.25 * node_cdf(1)(mu)
    )
    assert mixture.pvalue > 1e-3


def test_mie_azimuth_and_jones_follow_the_amplitude_functions() -> None:
    medium = _mie_medium()
    evidence = medium.mie_evidence
    assert evidence is not None
    x = float(evidence.size_parameters[0, 1])
    m = complex(evidence.relative_indices[0, 1])
    sample = _scatter(medium, 500e-9, _LINEAR, seed=11)
    cosines = np.asarray(sample.cosines)
    azimuths = np.asarray(sample.azimuths)
    # For incidence polarized along e1, E[cos 2 phi | theta] = -P(theta) / 2
    # with P = (|S1|^2 - |S2|^2) / (|S1|^2 + |S2|^2).
    grid, _, polarization = _reference_phase(x, m)
    expected = -0.5 * np.interp(cosines, grid, polarization)
    residual = np.cos(2.0 * azimuths) - expected
    assert abs(residual.mean()) < 4.0 * residual.std() / np.sqrt(_COUNT)
    subset = slice(0, 2048)
    amplitudes = lorenz_mie(x, m, cosines[subset])
    frame = np.asarray(
        rotate_jones_to_scattering_frame(
            jnp.asarray(np.broadcast_to(_LINEAR, (2048, 2))),
            jnp.asarray(azimuths[subset]),
        )
    )
    expected_jones = np.stack(
        (
            np.asarray(amplitudes.s1) * frame[:, 0],
            np.asarray(amplitudes.s2) * frame[:, 1],
        ),
        axis=-1,
    )
    expected_jones /= np.linalg.norm(expected_jones, axis=-1)[:, None]
    np.testing.assert_allclose(
        np.asarray(sample.jones_vectors)[subset], expected_jones, atol=1e-10
    )


def test_mie_tables_refuse_unresolved_angular_grids() -> None:
    with pytest.raises(ValueError, match="increase angle_count"):
        _mie_medium(angle_count=9)


def test_spectral_henyey_greenstein_keeps_scalar_jones_components() -> None:
    anisotropy = 0.85
    medium = _grid_medium(
        henyey_greenstein=HenyeyGreensteinScattering(
            np.full((1, 3), 0.5), np.full((1, 3), anisotropy)
        )
    )
    sample = _scatter(medium, 600e-9, _CIRCULAR, seed=13)
    g2 = anisotropy * anisotropy

    def hg_cdf(mu: np.ndarray) -> np.ndarray:
        return (
            (1.0 - g2)
            / (2.0 * anisotropy)
            * (1.0 / np.sqrt(1.0 + g2 - 2.0 * anisotropy * mu) - 1.0 / (1.0 + anisotropy))
        )

    assert stats.kstest(np.asarray(sample.cosines), hg_cdf).pvalue > 1e-3
    np.testing.assert_allclose(
        np.asarray(sample.jones_vectors),
        np.asarray(
            rotate_jones_to_scattering_frame(
                jnp.asarray(np.broadcast_to(_CIRCULAR, (_COUNT, 2))), sample.azimuths
            )
        ),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(medium.albedo(_lanes(1), jnp.asarray([600e-9]))), 1.0
    )


def test_spectral_tables_interpolate_linearly_and_report_support() -> None:
    grid = np.asarray([400e-9, 500e-9, 600e-9])
    medium = SpectralOpticalMedium(
        grid,
        np.asarray([[1.50, 1.48, 1.47], [1.0, 1.0, 1.0]]),
        np.asarray([[2.0, 4.0, np.inf], [np.inf, np.inf, np.inf]]),
        rayleigh=RayleighScattering(np.asarray([[1.0, 2.0, 4.0], [np.inf] * 3])),
    )
    media = jnp.asarray([0, 0, 0, 1, 0, 0], dtype=jnp.int32)
    queries = jnp.asarray([400e-9, 450e-9, 575e-9, 450e-9, 399e-9, 601e-9])
    index = np.asarray(medium.refractive_index(media, queries))
    np.testing.assert_allclose(index[:4], [1.50, 1.49, 1.4725, 1.0], rtol=1e-14)
    # Coefficients (inverse lengths) interpolate linearly; +inf lengths are zero.
    absorption = np.asarray([0.5, 0.375, 0.0625, 0.0])
    rayleigh = np.asarray([1.0, 0.75, 0.3125, 0.0])
    np.testing.assert_allclose(
        np.asarray(medium.extinction(media, queries))[:4], absorption + rayleigh
    )
    np.testing.assert_allclose(
        np.asarray(medium.albedo(media, queries))[:3],
        rayleigh[:3] / (absorption[:3] + rayleigh[:3]),
    )
    np.testing.assert_array_equal(
        np.asarray(medium.spectral_support(media, queries)),
        [True, True, True, True, False, False],
    )
    assert np.all(np.isnan(index[4:]))


def test_spectral_media_refuse_invalid_tables() -> None:
    grid = np.asarray([400e-9, 500e-9])
    index = np.full((1, 2), 1.5)
    clear = np.full((1, 2), np.inf)
    with pytest.raises(ValueError, match="strictly increasing"):
        SpectralOpticalMedium(np.asarray([500e-9, 400e-9]), index, clear)
    with pytest.raises(ValueError, match="positive"):
        SpectralOpticalMedium(grid, index, np.asarray([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="refractive_indices"):
        SpectralOpticalMedium(grid, np.asarray([[1.5, -1.0]]), clear)
    with pytest.raises(ValueError):
        SpectralOpticalMedium(grid, index, np.full((2, 2), np.inf))
    with pytest.raises(ValueError, match="between -1 and 1"):
        SpectralOpticalMedium(
            grid,
            index,
            clear,
            henyey_greenstein=HenyeyGreensteinScattering(clear, np.full((1, 2), 1.0)),
        )
    with pytest.raises(ValueError, match="inside the medium wavelength grid"):
        SpectralOpticalMedium(
            grid,
            index,
            clear,
            wavelength_shifter=WavelengthShifter(
                np.full((1, 2), 1.0),
                np.asarray([450e-9, 550e-9]),
                np.asarray([[1.0, 1.0]]),
                np.asarray([1.0]),
                np.asarray([0.0]),
            ),
        )
    with pytest.raises(ValueError, match="positive integral"):
        SpectralOpticalMedium(
            grid,
            index,
            clear,
            wavelength_shifter=WavelengthShifter(
                np.full((1, 2), 1.0),
                grid,
                np.zeros((1, 2)),
                np.asarray([1.0]),
                np.asarray([0.0]),
            ),
        )


_SHIFT_GRID = np.asarray([300e-9, 400e-9, 450e-9, 500e-9, 550e-9, 700e-9])
_EXCITATION = 350e-9
_YIELD = 0.8
_DELAY = 2e-9


def _triangle_cdf(wavelengths: np.ndarray) -> np.ndarray:
    """Triangular emission density on [450, 550] nm peaking at 500 nm."""

    t = (wavelengths - 450e-9) / 50e-9
    return np.where(
        t < 1.0,
        0.5 * np.clip(t, 0.0, 1.0) ** 2,
        1.0 - 0.5 * np.clip(2.0 - t, 0.0, 1.0) ** 2,
    )


def _shifter_medium(delay: Any = "exponential") -> SpectralOpticalMedium:
    media = 2
    shifting = np.full((media, 6), np.inf)
    shifting[0, :2] = 0.01
    return SpectralOpticalMedium(
        _SHIFT_GRID,
        np.full((media, 6), 1.5),
        np.full((media, 6), np.inf),
        wavelength_shifter=WavelengthShifter(
            shifting,
            np.asarray([450e-9, 500e-9, 550e-9]),
            np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            np.asarray([_YIELD, _YIELD]),
            np.asarray([_DELAY, _DELAY]),
            delay=delay,
        ),
    )


def test_wavelength_shifter_samples_emission_spectrum_delay_and_isotropy() -> None:
    medium = _shifter_medium()
    np.testing.assert_allclose(
        np.asarray(medium.albedo(_lanes(1), jnp.asarray([_EXCITATION]))), _YIELD
    )
    sample = _scatter(medium, _EXCITATION, _LINEAR, seed=17)
    emitted = np.asarray(sample.wavelengths)
    assert stats.kstest(emitted, _triangle_cdf).pvalue > 1e-3
    delays = np.asarray(sample.delays)
    standard_error = _DELAY / np.sqrt(_COUNT)
    assert abs(delays.mean() - _DELAY) < 4.0 * standard_error
    # Var of an exponential is tau^2; its sample variance has relative sd ~ sqrt(8/N).
    assert abs(delays.var() / _DELAY**2 - 1.0) < 4.0 * np.sqrt(8.0 / _COUNT)
    uniform = stats.kstest(np.asarray(sample.cosines), "uniform", args=(-1.0, 2.0))
    assert uniform.pvalue > 1e-3
    jones = np.asarray(sample.jones_vectors)
    stokes_q = np.abs(jones[:, 0]) ** 2 - np.abs(jones[:, 1]) ** 2
    stokes_v = 2.0 * np.imag(np.conj(jones[:, 0]) * jones[:, 1])
    for stokes in (stokes_q, stokes_v):
        assert abs(stokes.mean()) < 4.0 * stokes.std() / np.sqrt(_COUNT)
    fixed = _scatter(_shifter_medium("delta"), _EXCITATION, _LINEAR, count=64)
    np.testing.assert_allclose(np.asarray(fixed.delays), _DELAY)


def _half_space() -> NonSequentialSurfaceTable:
    vertices = jnp.asarray(
        [[-100.0, -100.0, -1.0], [100.0, -100.0, -1.0], [100.0, 100.0, -1.0]]
        + [[-100.0, 100.0, -1.0]]
    )
    return NonSequentialSurfaceTable(
        vertices,
        jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
        jnp.asarray([1, 1]),
        jnp.asarray([0, 0]),
        jnp.asarray([1.5, 1.5]),
        surface_ids=jnp.asarray([0, 0]),
    )


def _transport(
    medium: SpectralOpticalMedium, maximum_interactions: int, wavelength: float
) -> Any:
    relativity = RelativityScaleContract.si()
    prepared = prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            _half_space(),
            medium,
            relativity=relativity,
            maximum_interactions=maximum_interactions,
        )
    )
    count = 20_000
    photons = launch_optical_photons(
        np.zeros((count, 3)),
        np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), (count, 3)),
        0,
        wavelengths=wavelength,
    )
    return simulate_optical_photons(prepared, ExplicitPhotonSource(photons), jr.key(21))


def test_wavelength_shifting_in_transport_conserves_quanta_energy_and_timing() -> None:
    medium = _shifter_medium()
    first = _transport(medium, 1, _EXCITATION)
    terminal = first.terminal_state
    assert bool(jnp.all(first.terminal_live[:, 0]))
    # Implicit capture: every re-emitted packet carries exactly the quantum yield.
    np.testing.assert_allclose(np.asarray(terminal.weights[:, 0]), _YIELD, rtol=1e-14)
    np.testing.assert_allclose(
        np.asarray(first.per_photon_tallies.absorption[:, 0]), 1.0 - _YIELD, rtol=1e-14
    )
    emitted = np.asarray(terminal.wavelengths[:, 0])
    assert stats.kstest(emitted, _triangle_cdf).pvalue > 1e-3
    speed = float(RelativityScaleContract.si().speed_of_light)
    flight = 1.5 * np.asarray(terminal.positions[:, 0, 2]) / speed
    delays = np.asarray(terminal.times[:, 0]) - flight
    count = delays.shape[0]
    assert abs(delays.mean() - _DELAY) < 4.0 * _DELAY / np.sqrt(count)
    # Energy per absorbed quantum: QY * E[lambda_exc / lambda_em] < 1 (Stokes loss),
    # against an independent quadrature of the triangular emission density.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    left = 475e-9 + 25e-9 * nodes
    right = 525e-9 + 25e-9 * nodes
    density_left = (left - 450e-9) / 50e-9**2
    density_right = (550e-9 - right) / 50e-9**2
    mean_inverse = 25e-9 * np.sum(weights * (density_left / left + density_right / right))
    ratio = np.asarray(terminal.weights[:, 0]) * _EXCITATION / emitted
    expected = _YIELD * _EXCITATION * mean_inverse
    assert abs(ratio.mean() - expected) < 4.0 * ratio.std() / np.sqrt(count)
    assert expected < _YIELD
    # Re-emitted light escapes through transparent media: escape equals the yield.
    full = _transport(medium, 8, _EXCITATION)
    assert bool(full.all_successful)
    np.testing.assert_allclose(float(full.tallies.escape), _YIELD, rtol=1e-12)
    assert float(full.maximum_absolute_ledger_residual) < 1e-12


def test_transport_refuses_wavelengths_outside_the_medium_tables() -> None:
    result = _transport(_shifter_medium(), 4, 800e-9)
    assert bool(
        jnp.all(result.status == int(OpticalTransportStatus.SPECTRAL_SUPPORT_EXCEEDED))
    )
    np.testing.assert_allclose(np.asarray(result.per_photon_tallies.truncated), 1.0)
    assert not bool(jnp.any(result.terminal_live))
    assert float(result.maximum_absolute_ledger_residual) < 1e-12
