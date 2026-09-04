#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.optics.geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)
from phydrax.optics.transport._tissue import (
    prepare_tissue_transport,
    simulate_tissue_transport,
    TissueTransportCoefficients,
    TissueTransportPlan,
    TissueTransportStatus,
)


def _plane(z, negative_medium, positive_medium, indices):
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


def _origins(count, z):
    return jnp.broadcast_to(jnp.asarray([0.0, 0.0, z]), (count, 3))


def _directions(count, direction=(0.0, 0.0, 1.0)):
    return jnp.broadcast_to(jnp.asarray(direction), (count, 3))


def test_beer_lambert_escape_probability_and_standard_error_scaling():
    mu_a = jnp.log(2.0)
    coefficients = TissueTransportCoefficients(
        jnp.asarray([mu_a, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            _plane(1.0, 0, 1, (1.0, 1.0)),
            coefficients,
            maximum_interactions=2,
        )
    )

    small_count = 2048
    large_count = 8192
    small = simulate_tissue_transport(
        prepared,
        _origins(small_count, 0.0),
        _directions(small_count),
        jnp.zeros((small_count,), dtype=jnp.int32),
        jr.PRNGKey(81),
        photon_ids=jnp.arange(small_count),
    )
    large = simulate_tissue_transport(
        prepared,
        _origins(large_count, 0.0),
        _directions(large_count),
        jnp.zeros((large_count,), dtype=jnp.int32),
        jr.PRNGKey(81),
        photon_ids=jnp.arange(large_count),
    )

    assert abs(float(large.tallies.escape) - 0.5) < 4.0 * float(
        large.standard_errors.escape
    )
    ratio = float(large.standard_errors.escape / small.standard_errors.escape)
    assert 0.4 < ratio < 0.6
    assert float(large.maximum_absolute_ledger_residual) < 2e-6


def test_henyey_greenstein_first_moment_matches_g():
    count = 16_384
    g = 0.72
    surfaces = _plane(-10.0, 0, 0, (1.0,))
    coefficients = TissueTransportCoefficients(
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.asarray([g]),
        jnp.asarray([1.0]),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            surfaces,
            coefficients,
            maximum_interactions=1,
            branch_capacity=1,
        )
    )
    result = simulate_tissue_transport(
        prepared,
        _origins(count, 0.0),
        _directions(count),
        jnp.zeros((count,), dtype=jnp.int32),
        jr.PRNGKey(912),
        photon_ids=jnp.arange(count),
    )
    cosine = result.terminal_directions[:, 0, 2]
    assert abs(float(jnp.mean(cosine)) - g) < 0.01
    assert bool(jnp.all(result.terminal_live))
    assert bool(
        jnp.all(
            result.status == int(TissueTransportStatus.INTERACTION_CAPACITY_EXHAUSTED)
        )
    )


def test_stochastic_fresnel_frequency_expected_split_and_tir():
    count = 20_000
    indices = (1.0, 1.5)
    coefficients = TissueTransportCoefficients(
        jnp.zeros((2,)), jnp.zeros((2,)), jnp.zeros((2,)), jnp.asarray(indices)
    )
    surfaces = _plane(0.0, 0, 1, indices)
    stochastic = prepare_tissue_transport(
        TissueTransportPlan(
            surfaces,
            coefficients,
            maximum_interactions=1,
            fresnel_branching="stochastic",
        )
    )
    result = simulate_tissue_transport(
        stochastic,
        _origins(count, -1.0),
        _directions(count),
        jnp.zeros((count,), dtype=jnp.int32),
        jr.PRNGKey(710),
        photon_ids=jnp.arange(count),
    )
    reflected = jnp.mean((result.terminal_medium_indices[:, 0] == 0).astype(float))
    expected_reflectance = 0.04
    binomial_se = jnp.sqrt(expected_reflectance * (1.0 - expected_reflectance) / count)
    assert abs(float(reflected) - expected_reflectance) < 4.0 * float(binomial_se)

    expected = prepare_tissue_transport(
        TissueTransportPlan(
            surfaces,
            coefficients,
            maximum_interactions=1,
            branch_capacity=2,
            fresnel_branching="expected-split",
        )
    )
    split = simulate_tissue_transport(
        expected,
        _origins(1, -1.0),
        _directions(1),
        jnp.asarray([0]),
        jr.PRNGKey(710),
        photon_ids=jnp.asarray([0]),
    )
    live_weights = np.asarray(split.terminal_weights[0])[
        np.asarray(split.terminal_live[0])
    ]
    np.testing.assert_allclose(live_weights, [0.04, 0.96], atol=2e-6)
    np.testing.assert_allclose(split.per_photon_tallies.surface_flux, [[0.96]], atol=2e-6)

    sine = jnp.sin(jnp.deg2rad(60.0))
    cosine = jnp.cos(jnp.deg2rad(60.0))
    tir = simulate_tissue_transport(
        stochastic,
        _origins(4096, 1.0),
        _directions(4096, (sine, 0.0, -cosine)),
        jnp.ones((4096,), dtype=jnp.int32),
        jr.PRNGKey(17),
        photon_ids=jnp.arange(4096),
    )
    assert bool(jnp.all(tir.terminal_medium_indices[:, 0] == 1))
    assert bool(jnp.all(tir.terminal_directions[:, 0, 2] > 0.0))


def test_remaining_optical_depth_crosses_media_without_distance_rescaling():
    key = jr.PRNGKey(62)
    photon_ids = jnp.asarray([3, 11, 29], dtype=jnp.uint32)
    coefficients = TissueTransportCoefficients(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.1, 3.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([1.0, 1.0]),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            _plane(1e-6, 0, 1, (1.0, 1.0)),
            coefficients,
            maximum_interactions=1,
        )
    )
    result = simulate_tissue_transport(
        prepared,
        _origins(3, 0.0),
        _directions(3),
        jnp.zeros((3,), dtype=jnp.int32),
        key,
        photon_ids=photon_ids,
    )
    initial_depth = jnp.asarray(
        [
            -jnp.log(jr.uniform(jr.fold_in(jr.fold_in(key, photon_id), 0)))
            for photon_id in photon_ids
        ]
    )
    expected_remaining = initial_depth - 0.1e-6
    assert bool(jnp.all(initial_depth > 0.1e-6))
    np.testing.assert_allclose(
        result.terminal_optical_depths[:, 0], expected_remaining, atol=2e-6
    )
    np.testing.assert_array_equal(result.terminal_medium_indices[:, 0], 1)


def test_roulette_is_unbiased_and_weight_ledger_is_pathwise_complete():
    count = 16_384
    coefficients = TissueTransportCoefficients(
        jnp.asarray([0.9]),
        jnp.asarray([0.1]),
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            _plane(-10.0, 0, 0, (1.0,)),
            coefficients,
            maximum_interactions=1,
            roulette_threshold=0.2,
            roulette_survival_probability=0.5,
        )
    )
    result = simulate_tissue_transport(
        prepared,
        _origins(count, 0.0),
        _directions(count),
        jnp.zeros((count,), dtype=jnp.int32),
        jr.PRNGKey(991),
        photon_ids=jnp.arange(count),
    )
    assert abs(float(result.tallies.roulette)) < 4.0 * float(
        result.standard_errors.roulette
    )
    np.testing.assert_allclose(result.per_photon_tallies.ledger_residual, 0.0, atol=2e-6)
    np.testing.assert_allclose(result.tallies.absorption, [0.9], atol=2e-6)


def test_same_key_is_reproducible_and_photon_ids_are_batching_invariant():
    coefficients = TissueTransportCoefficients(
        jnp.asarray([0.3]),
        jnp.asarray([0.7]),
        jnp.asarray([0.2]),
        jnp.asarray([1.0]),
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            _plane(-10.0, 0, 0, (1.0,)), coefficients, maximum_interactions=3
        )
    )
    key = jr.PRNGKey(1234)
    ids = jnp.arange(32, dtype=jnp.uint32)
    whole = simulate_tissue_transport(
        prepared,
        _origins(32, 0.0),
        _directions(32),
        jnp.zeros((32,), dtype=jnp.int32),
        key,
        photon_ids=ids,
    )
    repeat = simulate_tissue_transport(
        prepared,
        _origins(32, 0.0),
        _directions(32),
        jnp.zeros((32,), dtype=jnp.int32),
        key,
        photon_ids=ids,
    )
    first = simulate_tissue_transport(
        prepared,
        _origins(13, 0.0),
        _directions(13),
        jnp.zeros((13,), dtype=jnp.int32),
        key,
        photon_ids=ids[:13],
    )
    second = simulate_tissue_transport(
        prepared,
        _origins(19, 0.0),
        _directions(19),
        jnp.zeros((19,), dtype=jnp.int32),
        key,
        photon_ids=ids[13:],
    )
    np.testing.assert_array_equal(whole.terminal_positions, repeat.terminal_positions)
    np.testing.assert_array_equal(
        whole.per_photon_tallies.absorption,
        jnp.concatenate(
            [first.per_photon_tallies.absorption, second.per_photon_tallies.absorption]
        ),
    )
    np.testing.assert_array_equal(
        whole.terminal_directions,
        jnp.concatenate([first.terminal_directions, second.terminal_directions]),
    )


def test_expected_split_capacity_exhaustion_reports_truncated_weight():
    indices = (1.0, 1.5)
    coefficients = TissueTransportCoefficients(
        jnp.zeros((2,)), jnp.zeros((2,)), jnp.zeros((2,)), jnp.asarray(indices)
    )
    prepared = prepare_tissue_transport(
        TissueTransportPlan(
            _plane(0.0, 0, 1, indices),
            coefficients,
            maximum_interactions=1,
            branch_capacity=1,
            fresnel_branching="expected-split",
        )
    )
    result = simulate_tissue_transport(
        prepared,
        _origins(1, -1.0),
        _directions(1),
        jnp.asarray([0]),
        jr.PRNGKey(7),
    )
    assert int(result.status[0]) == int(TissueTransportStatus.BRANCH_CAPACITY_EXHAUSTED)
    np.testing.assert_allclose(result.per_photon_tallies.truncated, [0.96], atol=2e-6)
    np.testing.assert_allclose(result.per_photon_tallies.live, [0.04], atol=2e-6)
    np.testing.assert_allclose(result.per_photon_tallies.ledger_residual, 0.0, atol=2e-6)


def test_detector_and_signed_surface_flux_use_fixed_tallies():
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
    coefficients = TissueTransportCoefficients(
        jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)), jnp.ones((1,))
    )
    detected = simulate_tissue_transport(
        prepare_tissue_transport(
            TissueTransportPlan(detector_surface, coefficients, maximum_interactions=1)
        ),
        _origins(1, -1.0),
        _directions(1),
        jnp.asarray([0]),
        jr.PRNGKey(5),
    )
    np.testing.assert_allclose(detected.per_photon_tallies.detector, [[1.0]])
    np.testing.assert_allclose(detected.per_photon_tallies.surface_flux, [[0.0]])
    np.testing.assert_allclose(
        detected.per_photon_tallies.ledger_residual, 0.0, atol=2e-6
    )
