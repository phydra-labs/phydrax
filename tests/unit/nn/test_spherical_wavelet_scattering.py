#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _problem(scattering_order):
    space = phx.discretization.SphericalSpectralPlan(4).prepare()
    theta, phi = jnp.meshgrid(
        space.transform.theta,
        space.transform.phi,
        indexing="ij",
    )
    orientations = jnp.stack((phi, theta, jnp.zeros_like(theta)), axis=-1).reshape(
        (-1, 3)
    )
    orientation_weights = (
        space.quadrature_weights / (4.0 * jnp.pi * space.radius**2)
    ).reshape((-1,))
    plan = phx.nn.operator.architectures.DirectionalSphericalWaveletPlan(
        space,
        orientations,
        scales=(0, 1, 2),
        azimuthal_bandlimit=2,
        scattering_order=scattering_order,
        orientation_weights=orientation_weights,
    )
    degree = space.layout.degrees
    order = space.layout.orders
    wavelets = jnp.stack(
        tuple(
            jnp.exp(-0.8 * (degree - center) ** 2) * (1.0 + 0.15j * (center + 1) * order)
            for center in (1, 2, 3)
        )
    )
    layer = phx.nn.operator.architectures.DirectionalSphericalWaveletLayer(plan, wavelets)
    values = (
        0.4
        + 0.7 * jnp.sin(theta) * jnp.cos(phi)
        + 0.2 * jnp.sin(theta) ** 2 * jnp.sin(2.0 * phi)
    )
    return space, plan, layer, space.project(values)


def test_order_one_scattering_preserves_orientation_average_and_layout():
    _, _, layer, coefficients = _problem(1)
    scattering = phx.nn.operator.architectures.SphericalWaveletScattering(layer)

    directional = jnp.abs(layer(coefficients))
    expected = jnp.sum(directional * layer.plan.orientation_weights[None, :], axis=-1)
    actual = scattering(coefficients)

    assert actual.shape == (1, len(layer.plan.scales))
    np.testing.assert_allclose(actual[0], expected, rtol=1e-12, atol=1e-12)

    channels = jnp.stack((coefficients, 0.5 * coefficients), axis=-1)
    channel_directional = jnp.abs(layer(channels))
    channel_expected = jnp.sum(
        channel_directional * layer.plan.orientation_weights[None, :, None], axis=1
    )
    channel_actual = scattering(channels)
    assert channel_actual.shape == (len(layer.plan.scales), 1, 2)
    np.testing.assert_allclose(
        channel_actual[:, 0, :], channel_expected, rtol=1e-12, atol=1e-12
    )


def test_second_order_is_recursive_wavelet_modulus_and_masks_inadmissible_paths():
    space, plan, layer, coefficients = _problem(2)
    scattering = phx.nn.operator.architectures.SphericalWaveletScattering(layer)

    first = jnp.abs(layer(coefficients))
    projected_modulus = layer.project_orientation_samples(first)
    reapplied = jnp.abs(layer(projected_modulus))
    explicit_second = jnp.sum(
        reapplied * plan.orientation_weights[None, :, None], axis=1
    ).T.reshape((-1,))
    second_mask = plan.path_mask[1]
    expected_second = jnp.where(second_mask, explicit_second, 0.0)

    actual = scattering(coefficients)
    candidate_paths = tuple(product(plan.scales, repeat=2))
    expected_mask = jnp.asarray(tuple(left < right for left, right in candidate_paths))

    assert actual.shape == (2, len(plan.scales) ** plan.scattering_order)
    assert plan.path_admissibility == "strictly-increasing-scale"
    assert jnp.array_equal(plan.path_scales[1], jnp.asarray(candidate_paths))
    assert jnp.array_equal(second_mask, expected_mask)
    assert plan.recursive_projection == "weighted-Wigner-n0-scalar-S2"
    assert plan.analysis_frame_lower_bound > 0.0
    np.testing.assert_allclose(actual[1], expected_second, rtol=1e-12, atol=1e-12)
    assert jnp.array_equal(actual[1, ~second_mask], jnp.zeros((6,)))

    first_layer_power = jnp.sum(first**2 * plan.orientation_weights[None, :], axis=1)
    active_first_scales = plan.path_indices[1, second_mask, 0]
    assert not jnp.allclose(
        actual[1, second_mask], first_layer_power[active_first_scales]
    )

    constant = layer.project_orientation_samples(jnp.ones(plan.orientations.shape[0]))
    expected_constant = (
        jnp.zeros_like(constant)
        .at[0, space.layout.bandlimit - 1]
        .set(jnp.sqrt(4.0 * jnp.pi))
    )
    np.testing.assert_allclose(constant, expected_constant, rtol=1e-12, atol=1e-12)

    repeated_orientation = jnp.zeros((space.layout.logical_mode_count, 3))
    with pytest.raises(ValueError, match="full-rank weighted Wigner"):
        phx.nn.operator.architectures.DirectionalSphericalWaveletPlan(
            space,
            repeated_orientation,
            scales=(0,),
            azimuthal_bandlimit=1,
            scattering_order=2,
        )


def test_recursive_scattering_retains_z_rotation_invariance_and_stability():
    space, plan, layer, coefficients = _problem(2)
    scattering = phx.nn.operator.architectures.SphericalWaveletScattering(layer)
    rotation = phx.discretization.spectral.SphericalRotationPlan(space).prepare()
    azimuth_step = 2.0 * jnp.pi / space.transform.phi.shape[0]

    reference = scattering(coefficients)
    rotated = scattering(
        rotation.apply(
            coefficients,
            jnp.asarray([azimuth_step, 0.0, 0.0]),
        )
    )
    perturbation = space.project(
        1e-7
        * jnp.sin(space.transform.theta)[:, None]
        * jnp.sin(space.transform.phi)[None, :]
    )
    perturbed = scattering(coefficients + perturbation)
    active = plan.path_mask

    assert jnp.all(jnp.isfinite(reference))
    np.testing.assert_allclose(
        rotated[active],
        reference[active],
        rtol=2e-9,
        atol=2e-9,
    )
    assert jnp.max(jnp.abs(perturbed[active] - reference[active])) < 1e-5
