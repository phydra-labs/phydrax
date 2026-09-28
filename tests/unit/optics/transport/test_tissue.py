#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax._physical import RelativityScaleContract
from phydrax.optics.geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)
from phydrax.optics.transport._optical_monte_carlo import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalTransportStatus,
    OpticalVarianceReduction,
    prepare_optical_monte_carlo,
    PreparedOpticalMonteCarlo,
    simulate_optical_photons,
    TissueOpticalMedium,
)


_WAVELENGTH = 500e-9


def _plane(z: Any, negative_medium: Any, positive_medium: Any, indices: Any) -> Any:
    vertices = jnp.asarray(
        [[-100.0, -100.0, z], [100.0, -100.0, z], [100.0, 100.0, z], [-100.0, 100.0, z]]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    return NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([negative_medium, negative_medium]),
        jnp.asarray([positive_medium, positive_medium]),
        jnp.asarray(indices),
        surface_ids=jnp.asarray([0, 0]),
    )


def _origins(count: Any, z: Any) -> Any:
    return np.broadcast_to(np.asarray([0.0, 0.0, z]), (count, 3))


def _directions(count: Any, direction: Any = (0.0, 0.0, 1.0)) -> Any:
    return np.broadcast_to(np.asarray(direction), (count, 3))


def _prepared(
    surfaces: Any, medium: Any, maximum_interactions: int, **kwargs: Any
) -> PreparedOpticalMonteCarlo:
    return prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=RelativityScaleContract.si(),
            maximum_interactions=maximum_interactions,
            **kwargs,
        )
    )


def _simulate(
    prepared: PreparedOpticalMonteCarlo,
    origins: Any,
    directions: Any,
    medium: int,
    key: Any,
    *,
    first_identity: tuple[int, int] = (0, 0),
) -> Any:
    state = launch_optical_photons(
        origins,
        directions,
        medium,
        wavelengths=_WAVELENGTH,
        first_identity=first_identity,
    )
    return simulate_optical_photons(prepared, ExplicitPhotonSource(state), key)


def test_tissue_scenario_1() -> None:
    mu_a = float(np.log(2.0))
    medium = TissueOpticalMedium(
        jnp.asarray([mu_a, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
    )
    prepared = _prepared(_plane(1.0, 0, 1, (1.0, 1.0)), medium, 2)

    small_count = 2048
    large_count = 8192
    small = _simulate(
        prepared, _origins(small_count, 0.0), _directions(small_count), 0, jr.key(81)
    )
    large = _simulate(
        prepared, _origins(large_count, 0.0), _directions(large_count), 0, jr.key(81)
    )

    assert abs(float(large.tallies.escape) - 0.5) < 4.0 * float(
        large.standard_errors.escape
    )
    ratio = float(large.standard_errors.escape / small.standard_errors.escape)
    assert 0.4 < ratio < 0.6
    assert float(large.maximum_absolute_ledger_residual) < 2e-6
    count = 16_384
    g = 0.72
    surfaces = _plane(-10.0, 0, 0, (1.0,))
    medium = TissueOpticalMedium(
        jnp.asarray([0.0]), jnp.asarray([1.0]), jnp.asarray([g]), jnp.asarray([1.0])
    )
    prepared = _prepared(surfaces, medium, 1, branch_capacity=1)
    result = _simulate(prepared, _origins(count, 0.0), _directions(count), 0, jr.key(912))
    cosine = result.terminal_state.directions[:, 0, 2]
    assert abs(float(jnp.mean(cosine)) - g) < 0.01
    assert bool(jnp.all(result.terminal_live))
    assert bool(
        jnp.all(
            result.status == int(OpticalTransportStatus.INTERACTION_CAPACITY_EXHAUSTED)
        )
    )
    count = 20_000
    indices = (1.0, 1.5)
    medium = TissueOpticalMedium(
        jnp.zeros((2,)), jnp.zeros((2,)), jnp.zeros((2,)), jnp.asarray(indices)
    )
    surfaces = _plane(0.0, 0, 1, indices)
    stochastic = _prepared(
        surfaces,
        medium,
        1,
        variance_reduction=OpticalVarianceReduction(interface_branching="stochastic"),
    )
    result = _simulate(
        stochastic, _origins(count, -1.0), _directions(count), 0, jr.key(710)
    )
    reflected = jnp.mean(
        (result.terminal_state.medium_indices[:, 0] == 0).astype(jnp.float64)
    )
    expected_reflectance = 0.04
    binomial_se = jnp.sqrt(expected_reflectance * (1.0 - expected_reflectance) / count)
    assert abs(float(reflected) - expected_reflectance) < 4.0 * float(binomial_se)

    expected = _prepared(
        surfaces,
        medium,
        1,
        branch_capacity=2,
        variance_reduction=OpticalVarianceReduction(interface_branching="expected-split"),
    )
    split = _simulate(expected, _origins(1, -1.0), _directions(1), 0, jr.key(710))
    live_weights = np.asarray(split.terminal_state.weights[0])[
        np.asarray(split.terminal_live[0])
    ]
    np.testing.assert_allclose(live_weights, [0.04, 0.96], atol=2e-6)
    np.testing.assert_allclose(split.per_photon_tallies.surface_flux, [[0.96]], atol=2e-6)

    sine = float(np.sin(np.deg2rad(60.0)))
    cosine = float(np.cos(np.deg2rad(60.0)))
    tir = _simulate(
        stochastic,
        _origins(4096, 1.0),
        _directions(4096, (sine, 0.0, -cosine)),
        1,
        jr.key(17),
    )
    assert bool(jnp.all(tir.terminal_state.medium_indices[:, 0] == 1))
    assert bool(jnp.all(tir.terminal_state.directions[:, 0, 2] > 0.0))


def test_tissue_scenario_2() -> None:
    key = jr.key(62)
    thin = _plane(1e-6, 0, 1, (1.0, 1.0))
    attenuating = TissueOpticalMedium(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.1, 3.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
    )
    transparent = TissueOpticalMedium(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 3.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
    )
    result = _simulate(
        _prepared(thin, attenuating, 1),
        _origins(3, 0.0),
        _directions(3),
        0,
        key,
        first_identity=(0, 3),
    )
    reference = _simulate(
        _prepared(thin, transparent, 1),
        _origins(3, 0.0),
        _directions(3),
        0,
        key,
        first_identity=(0, 3),
    )
    # Same identities and key draw the same initial optical depth; crossing the
    # thin layer removes exactly its optical thickness and keeps the remainder.
    initial_depth = reference.terminal_optical_depths[:, 0]
    assert bool(jnp.all(initial_depth > 0.1e-6))
    np.testing.assert_allclose(
        result.terminal_optical_depths[:, 0], initial_depth - 0.1e-6, atol=2e-6
    )
    np.testing.assert_array_equal(result.terminal_state.medium_indices[:, 0], 1)
    count = 16_384
    medium = TissueOpticalMedium(
        jnp.asarray([0.9]), jnp.asarray([0.1]), jnp.asarray([0.0]), jnp.asarray([1.0])
    )
    prepared = _prepared(
        _plane(-10.0, 0, 0, (1.0,)),
        medium,
        1,
        variance_reduction=OpticalVarianceReduction(
            roulette_threshold=0.2, roulette_survival_probability=0.5
        ),
    )
    result = _simulate(prepared, _origins(count, 0.0), _directions(count), 0, jr.key(991))
    assert abs(float(result.tallies.roulette)) < 4.0 * float(
        result.standard_errors.roulette
    )
    np.testing.assert_allclose(result.per_photon_tallies.ledger_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.tallies.absorption, [0.9], atol=2e-6)
    medium = TissueOpticalMedium(
        jnp.asarray([0.3]), jnp.asarray([0.7]), jnp.asarray([0.2]), jnp.asarray([1.0])
    )
    prepared = _prepared(_plane(-10.0, 0, 0, (1.0,)), medium, 3)
    key = jr.key(1234)
    whole = _simulate(prepared, _origins(32, 0.0), _directions(32), 0, key)
    repeat = _simulate(prepared, _origins(32, 0.0), _directions(32), 0, key)
    first = _simulate(prepared, _origins(13, 0.0), _directions(13), 0, key)
    second = _simulate(
        prepared,
        _origins(19, 0.0),
        _directions(19),
        0,
        key,
        first_identity=(0, 13),
    )
    np.testing.assert_array_equal(
        whole.terminal_state.positions, repeat.terminal_state.positions
    )
    np.testing.assert_array_equal(
        whole.per_photon_tallies.absorption,
        jnp.concatenate(
            [first.per_photon_tallies.absorption, second.per_photon_tallies.absorption]
        ),
    )
    # Split batches replay the same identity-keyed histories; directions may
    # differ only by the rounding of differently vectorized arithmetic.
    np.testing.assert_allclose(
        whole.terminal_state.directions,
        jnp.concatenate(
            [first.terminal_state.directions, second.terminal_state.directions]
        ),
        rtol=1e-12,
        atol=1e-15,
    )


def test_tissue_scenario_3() -> None:
    indices = (1.0, 1.5)
    medium = TissueOpticalMedium(
        jnp.zeros((2,)), jnp.zeros((2,)), jnp.zeros((2,)), jnp.asarray(indices)
    )
    prepared = _prepared(
        _plane(0.0, 0, 1, indices),
        medium,
        1,
        branch_capacity=1,
        variance_reduction=OpticalVarianceReduction(interface_branching="expected-split"),
    )
    result = _simulate(prepared, _origins(1, -1.0), _directions(1), 0, jr.key(7))
    assert int(result.status[0]) == int(OpticalTransportStatus.BRANCH_CAPACITY_EXHAUSTED)
    np.testing.assert_allclose(result.per_photon_tallies.truncated, [0.96], atol=2e-6)
    np.testing.assert_allclose(result.per_photon_tallies.live, [0.04], atol=2e-6)
    np.testing.assert_allclose(result.per_photon_tallies.ledger_residual, 0.0, atol=2e-6)
    vertices = jnp.asarray(
        [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    detector_surface = NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([0, 0]),
        jnp.asarray([0, 0]),
        jnp.asarray([1.0]),
        surface_ids=jnp.asarray([0, 0]),
        surface_kinds=jnp.asarray([int(NonSequentialSurfaceKind.DETECTOR)] * 2),
        detector_indices=jnp.asarray([0, 0]),
        detector_acceptance_cosines=jnp.asarray([0.25, 0.25]),
    )
    medium = TissueOpticalMedium(
        jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)), jnp.ones((1,))
    )
    detected = _simulate(
        _prepared(detector_surface, medium, 1),
        _origins(1, -1.0),
        _directions(1),
        0,
        jr.key(5),
    )
    np.testing.assert_allclose(detected.per_photon_tallies.detector, [[1.0]])
    np.testing.assert_allclose(detected.per_photon_tallies.surface_flux, [[0.0]])
    np.testing.assert_allclose(
        detected.per_photon_tallies.ledger_residual, 0.0, atol=2e-6
    )
