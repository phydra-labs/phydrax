#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax._physical import RelativityScaleContract
from phydrax.optics.geometric._nonsequential import NonSequentialSurfaceTable
from phydrax.optics.transport._optical_monte_carlo import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalTransportResult,
    OpticalVarianceReduction,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
    TissueOpticalMedium,
)


# Wang, Jacques, Zheng, Comput. Methods Programs Biomed. 47 (1995) 131-146,
# Tables 1 and 2: mu_a = 10 /cm, mu_s = 90 /cm, lengths in centimeters.
_MU_A = 10.0
_MU_S = 90.0


def _planes(depths: tuple[float, ...], index: float) -> Any:
    """Planes at ``depths``: ambient medium 1 outside, tissue medium 0 between."""

    extent = 1e3
    vertices = []
    for depth in depths:
        vertices.extend(
            [
                [-extent, -extent, depth],
                [extent, -extent, depth],
                [extent, extent, depth],
                [-extent, extent, depth],
            ]
        )
    triangles = []
    for plane in range(len(depths)):
        first = 4 * plane
        triangles.extend([[first, first + 1, first + 2], [first, first + 2, first + 3]])
    negative = [1, 0][: len(depths)]
    positive = [0, 1][: len(depths)]
    return NonSequentialSurfaceTable(
        jnp.asarray(vertices),
        jnp.asarray(triangles, dtype=jnp.int32),
        jnp.asarray([side for side in negative for _ in range(2)]),
        jnp.asarray([side for side in positive for _ in range(2)]),
        jnp.asarray([index, 1.0]),
        surface_ids=jnp.asarray(
            [plane for plane in range(len(depths)) for _ in range(2)]
        ),
    )


def _normally_incident_beam(
    surfaces: Any, anisotropy: float, index: float, count: int, seed: int
) -> OpticalTransportResult:
    medium = TissueOpticalMedium(
        jnp.asarray([_MU_A, 0.0]),
        jnp.asarray([_MU_S, 0.0]),
        jnp.asarray([anisotropy, 0.0]),
        jnp.asarray([index, 1.0]),
    )
    prepared = prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=RelativityScaleContract.si(),
            maximum_interactions=400,
            photon_batch_size=4096,
            variance_reduction=OpticalVarianceReduction(
                roulette_threshold=1e-4, roulette_survival_probability=0.1
            ),
        )
    )
    # The collimated beam starts in the ambient medium, so the owner applies
    # the entrance Fresnel reflection itself.
    state = launch_optical_photons(
        np.broadcast_to(np.asarray([0.0, 0.0, -1e-3]), (count, 3)),
        np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), (count, 3)),
        1,
        wavelengths=500e-9,
    )
    result = simulate_optical_photons(prepared, ExplicitPhotonSource(state), jr.key(seed))
    # Success means no live, truncated, or failed weight remains.
    assert bool(result.all_successful)
    assert float(result.maximum_absolute_ledger_residual) < 1e-10
    return result


def _mean_and_standard_error(values: Any) -> tuple[float, float]:
    samples = np.asarray(values)
    return float(np.mean(samples)), float(np.std(samples, ddof=1) / np.sqrt(samples.size))


def test_matched_slab_reflectance_and_transmittance_match_van_de_hulst() -> None:
    # MCML Table 1: n = 1, g = 0.75, d = 0.02 cm; van de Hulst (1980) gives
    # total diffuse reflectance 0.09739 and total transmittance 0.66096
    # (unscattered exp(-2) included).
    result = _normally_incident_beam(_planes((0.0, 0.02), 1.0), 0.75, 1.0, 40_000, 1980)
    transmitted = result.per_photon_tallies.surface_flux[:, 1]
    reflectance, reflectance_error = _mean_and_standard_error(
        result.per_photon_tallies.escape - transmitted
    )
    transmittance, transmittance_error = _mean_and_standard_error(transmitted)
    assert abs(reflectance - 0.09739) < 4.0 * reflectance_error
    assert abs(transmittance - 0.66096) < 4.0 * transmittance_error


def test_semi_infinite_mismatched_total_reflectance_matches_giovanelli() -> None:
    # MCML Table 2: semi-infinite medium, n = 1.5, isotropic scattering;
    # Giovanelli (1955) total reflectance 0.2600 of the incident beam, which
    # includes the 4% normal-incidence entrance reflection.
    result = _normally_incident_beam(_planes((0.0,), 1.5), 0.0, 1.5, 40_000, 1955)
    reflectance, error = _mean_and_standard_error(result.per_photon_tallies.escape)
    assert abs(reflectance - 0.2600) < 4.0 * error


def test_layered_nonscattering_slab_matches_piecewise_beer_lambert_reference() -> None:
    count = 32_768
    vertices = jnp.asarray(
        [
            [-20.0, -20.0, 0.4],
            [20.0, -20.0, 0.4],
            [20.0, 20.0, 0.4],
            [-20.0, 20.0, 0.4],
            [-20.0, -20.0, 1.0],
            [20.0, -20.0, 1.0],
            [20.0, 20.0, 1.0],
            [-20.0, 20.0, 1.0],
        ]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=jnp.int32)
    surfaces = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0, 1, 1]),
        jnp.asarray([1, 1, 2, 2]),
        jnp.asarray([1.0, 1.0, 1.0]),
        surface_ids=jnp.asarray([0, 0, 1, 1]),
    )
    medium = TissueOpticalMedium(
        jnp.asarray([0.2, 0.7, 0.0]), jnp.zeros((3,)), jnp.zeros((3,)), jnp.ones((3,))
    )
    prepared = prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=RelativityScaleContract.si(),
            maximum_interactions=3,
        )
    )
    state = launch_optical_photons(
        np.broadcast_to(np.asarray([0.0, 0.0, 0.0]), (count, 3)),
        np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), (count, 3)),
        0,
        wavelengths=500e-9,
    )
    result = simulate_optical_photons(prepared, ExplicitPhotonSource(state), jr.key(4401))

    reference = np.exp(-(0.2 * 0.4 + 0.7 * 0.6))
    uncertainty = result.standard_errors.escape
    assert abs(float(result.tallies.escape - reference)) < max(
        4.0 * float(uncertainty), 0.01
    )
    np.testing.assert_allclose(
        result.tallies.absorption.sum() + result.tallies.escape,
        1.0,
        atol=2e-6,
    )
    assert float(result.maximum_absolute_ledger_residual) < 2e-6
