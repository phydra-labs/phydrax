#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.optics.geometric import (
    evaluate_refractive_interface,
    PlanarRefractiveStack,
    RefractiveInterfaceStatus,
    SequentialOpticsStatus,
    trace_planar_refractive_stack,
)


def _direction(angle):
    return jnp.asarray((jnp.sin(angle), 0.0, jnp.cos(angle)))


def test_matched_and_oblique_snell_fresnel_energy_conservation():
    matched = evaluate_refractive_interface(
        jnp.asarray((0.0, 0.0, 4.0)),
        jnp.asarray((0.0, 0.0, 2.0)),
        1.3,
        1.3,
    )
    np.testing.assert_allclose(matched.transmitted_directions, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(matched.reflection_amplitudes, (0.0, 0.0), atol=1e-7)
    np.testing.assert_allclose(matched.transmission_amplitudes, (1.0, 1.0))
    np.testing.assert_allclose(matched.transmittance, (1.0, 1.0))

    angle = jnp.deg2rad(37.0)
    oblique = evaluate_refractive_interface(
        _direction(angle),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        1.5,
    )
    expected_sine = jnp.sin(angle) / 1.5
    np.testing.assert_allclose(oblique.transmitted_directions[0], expected_sine)
    np.testing.assert_allclose(
        oblique.transmitted_directions[2], jnp.sqrt(1.0 - expected_sine**2)
    )
    np.testing.assert_allclose(
        oblique.reflectance + oblique.transmittance,
        jnp.ones((2,)),
        rtol=2e-6,
        atol=2e-6,
    )
    assert bool(oblique.transmission_valid)
    assert float(oblique.energy_balance_error) < 2e-6


def test_brewster_reflection_and_total_internal_reflection_phase():
    brewster = evaluate_refractive_interface(
        _direction(jnp.arctan(1.5)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        1.5,
    )
    np.testing.assert_allclose(brewster.reflection_amplitudes[1], 0.0, atol=2e-7)

    tir = evaluate_refractive_interface(
        _direction(jnp.deg2rad(60.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.5,
        1.0,
    )
    assert int(tir.status) == int(RefractiveInterfaceStatus.TOTAL_INTERNAL_REFLECTION)
    assert bool(tir.reflection_valid)
    assert not bool(tir.transmission_valid)
    np.testing.assert_allclose(tir.reflectance, (1.0, 1.0), rtol=2e-6)
    np.testing.assert_allclose(tir.transmittance, (0.0, 0.0))
    assert bool(jnp.all(jnp.imag(tir.reflection_amplitudes) < 0.0))
    np.testing.assert_allclose(tir.transmitted_directions, (0.0, 0.0, 0.0))


def test_wrong_side_and_grazing_incidence_are_distinct():
    wrong_side = evaluate_refractive_interface(
        jnp.asarray((0.0, 0.0, -1.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        1.5,
    )
    grazing = evaluate_refractive_interface(
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        1.5,
    )

    assert int(wrong_side.status) == int(RefractiveInterfaceStatus.WRONG_SIDE_INCIDENCE)
    assert int(grazing.status) == int(RefractiveInterfaceStatus.GRAZING_INCIDENCE)
    assert not bool(wrong_side.reflection_valid)
    assert not bool(grazing.reflection_valid)


def test_zero_capacity_and_inactive_suffix_have_fixed_identity_semantics():
    empty = PlanarRefractiveStack(np.empty((0, 3)), np.empty((0, 3)), np.asarray((1.2,)))
    origin = jnp.asarray(((0.0, 0.0, 0.0), (1.0, -2.0, 3.0)))
    direction = jnp.asarray(((0.0, 0.0, 2.0), (0.0, 3.0, 4.0)))
    identity = trace_planar_refractive_stack(empty, origin, direction)

    np.testing.assert_allclose(identity.rays.origins, origin)
    np.testing.assert_allclose(
        identity.rays.directions,
        direction / jnp.sqrt(jnp.sum(direction**2, axis=-1, keepdims=True)),
    )
    np.testing.assert_allclose(identity.rays.refractive_indices, 1.2)
    np.testing.assert_allclose(identity.rays.geometric_path_lengths, 0.0)
    assert bool(jnp.all(jnp.isinf(identity.minimum_snell_discriminant)))
    assert bool(jnp.all(identity.successful))

    padded = PlanarRefractiveStack(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -10.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        [1.0, 1.5, 9.0],
        interface_active=[True, False],
    )
    result = trace_planar_refractive_stack(
        padded, jnp.zeros((3,)), jnp.asarray((0.0, 0.0, 1.0))
    )
    assert bool(result.successful)
    assert int(result.traversed_surfaces) == 1
    np.testing.assert_allclose(result.rays.origins, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(result.rays.refractive_indices, 1.5)


def test_two_interface_parallel_slab_recovers_angle_and_accumulates_paths():
    first_angle = jnp.deg2rad(30.0)
    stack = PlanarRefractiveStack(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [1.0, 1.5, 1.0],
    )
    result = trace_planar_refractive_stack(
        stack, jnp.zeros((3,)), _direction(first_angle)
    )
    transmitted_cosine = jnp.sqrt(1.0 - (jnp.sin(first_angle) / 1.5) ** 2)
    expected_first = 1.0 / jnp.cos(first_angle)
    expected_second = 1.0 / transmitted_cosine

    assert bool(result.successful)
    assert int(result.traversed_surfaces) == 2
    np.testing.assert_allclose(result.rays.directions, _direction(first_angle), atol=2e-6)
    np.testing.assert_allclose(
        result.rays.geometric_path_lengths, expected_first + expected_second
    )
    np.testing.assert_allclose(
        result.rays.optical_path_lengths,
        expected_first + 1.5 * expected_second,
    )
    np.testing.assert_allclose(result.rays.origins[2], 2.0)


def test_first_failure_retains_last_successful_ray_and_path():
    stack = PlanarRefractiveStack(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [1.0, 1.5, 1.0],
    )
    result = trace_planar_refractive_stack(
        stack, jnp.zeros((3,)), jnp.asarray((0.0, 0.0, 1.0))
    )

    assert not bool(result.valid)
    assert int(result.status) == int(SequentialOpticsStatus.BEHIND_RAY)
    assert int(result.traversed_surfaces) == 1
    np.testing.assert_allclose(result.rays.origins, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(result.rays.directions, (0.0, 0.0, 1.0))
    np.testing.assert_allclose(result.rays.geometric_path_lengths, 1.0)
    np.testing.assert_allclose(result.rays.optical_path_lengths, 1.0)


def test_planar_trace_is_jittable_vmappable_and_differentiable_away_from_boundaries():
    stack = PlanarRefractiveStack([[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [1.0, 1.5])

    def transmitted_x(angle):
        return trace_planar_refractive_stack(
            stack, jnp.zeros((3,)), _direction(angle)
        ).rays.directions[0]

    angles = jnp.asarray((0.1, 0.2, 0.3))
    eager_batch = jax.vmap(transmitted_x)(angles)
    compiled_batch = jax.jit(jax.vmap(transmitted_x))(angles)
    derivative = jax.grad(transmitted_x)(jnp.asarray(0.2))

    np.testing.assert_allclose(compiled_batch, eager_batch)
    np.testing.assert_allclose(eager_batch, jnp.sin(angles) / 1.5)
    np.testing.assert_allclose(derivative, jnp.cos(0.2) / 1.5, rtol=2e-5)

    camera_shaped = trace_planar_refractive_stack(
        stack,
        jnp.zeros((2, 4, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (2, 4, 3)),
    )
    assert camera_shaped.rays.origins.shape == (2, 4, 3)
    assert camera_shaped.status.shape == (2, 4)
