#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import (
    boost_event,
    boost_fields,
    boost_matrix,
    boost_proper_velocity,
    boost_wavevector,
    FourMomentum,
    LorentzFrame,
    minkowski_dot,
    transform_spectral_energy,
)


def _gamma(speed: float) -> float:
    return 1.0 / np.sqrt(1.0 - speed * speed)


def test_lorentz_frame_boost_preserves_mass_and_inverts() -> None:
    mass = 0.511
    spatial = np.asarray([0.3, -0.2, 0.7])
    momentum = FourMomentum(
        jnp.asarray(np.concatenate(([np.sqrt(mass**2 + spatial @ spatial)], spatial)))
    )
    boost = LorentzFrame.boost(jnp.asarray([0.2, -0.1, 0.05]))
    transformed = boost.apply(momentum)

    np.testing.assert_allclose(
        minkowski_dot(transformed.value, transformed.value), mass**2, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        boost.inverse().apply(transformed).value, momentum.value, atol=1.0e-12
    )


def test_active_frame_boost_carries_rest_momentum_to_velocity() -> None:
    velocity = np.asarray([0.3, -0.4, 0.1])
    gamma = _gamma(float(np.linalg.norm(velocity)))
    moved = LorentzFrame.boost(velocity).apply(jnp.asarray([2.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(moved.energy, 2.0 * gamma, rtol=1.0e-13)
    np.testing.assert_allclose(moved.spatial, 2.0 * gamma * velocity, rtol=1.0e-13)


def test_collinear_composition_is_relativistic_velocity_addition() -> None:
    first, second = 0.6, 0.7
    axis = np.asarray([2.0, -1.0, 2.0]) / 3.0
    composed = boost_matrix(first * axis) @ boost_matrix(second * axis)
    added = (first + second) / (1.0 + first * second)

    np.testing.assert_allclose(composed, boost_matrix(added * axis), atol=1.0e-12)


def test_perpendicular_composition_has_thomas_wigner_rotation() -> None:
    first, second = 0.8, 0.6
    composed = np.asarray(
        boost_matrix(jnp.asarray([0.0, second, 0.0]))
        @ boost_matrix(jnp.asarray([first, 0.0, 0.0]))
    )
    # The pure-boost factor is fixed by the image of the time axis.
    composite_velocity = -composed[1:, 0] / composed[0, 0]
    rotation = np.asarray(boost_matrix(-composite_velocity)) @ composed
    gamma_one, gamma_two = _gamma(first), _gamma(second)
    # Wigner angle for orthogonal boosts: cos θ = (γ₁ + γ₂)/(1 + γ₁γ₂).
    expected_cosine = (gamma_one + gamma_two) / (1.0 + gamma_one * gamma_two)

    np.testing.assert_allclose(rotation[0], [1.0, 0.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(
        rotation[1:, 1:].T @ rotation[1:, 1:], np.eye(3), atol=1.0e-12
    )
    np.testing.assert_allclose(
        (np.trace(rotation[1:, 1:]) - 1.0) / 2.0, expected_cosine, rtol=1.0e-12
    )
    np.testing.assert_allclose(rotation[3, 3], 1.0, atol=1.0e-12)
    assert abs(rotation[1, 2] - rotation[2, 1]) > 1.0e-2


def test_boost_event_preserves_interval_and_simultaneity_shift() -> None:
    speed = 0.6
    event = jnp.asarray([[0.0, 1.0, 0.0, 0.0], [2.0, 0.5, -1.0, 3.0]])
    boosted = boost_event(jnp.asarray([speed, 0.0, 0.0]), event)
    gamma = _gamma(speed)

    np.testing.assert_allclose(
        boosted[0], [-gamma * speed, gamma, 0.0, 0.0], atol=1.0e-14
    )
    np.testing.assert_allclose(
        minkowski_dot(boosted, boosted), minkowski_dot(event, event), atol=1.0e-12
    )


def test_proper_velocity_boost_matches_velocity_addition() -> None:
    lab_speed, frame_speed = 0.9, 0.5
    lab_proper = _gamma(lab_speed) * lab_speed
    boosted = boost_proper_velocity(
        jnp.asarray([frame_speed, 0.0, 0.0]),
        jnp.asarray([[lab_proper, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    relative = (lab_speed - frame_speed) / (1.0 - lab_speed * frame_speed)

    np.testing.assert_allclose(
        boosted[0], [_gamma(relative) * relative, 0.0, 0.0], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        boosted[1], [-_gamma(frame_speed) * frame_speed, 0.0, 0.0], rtol=1.0e-12
    )


def test_field_boost_matches_textbook_transform_and_preserves_invariants() -> None:
    c = 3.0
    rng = np.random.default_rng(7)
    electric = rng.normal(size=(5, 3))
    magnetic = rng.normal(size=(5, 3))
    beta = np.asarray([0.3, -0.5, 0.4])
    boosted_electric, boosted_magnetic = boost_fields(
        jnp.asarray(beta), jnp.asarray(electric), jnp.asarray(magnetic), speed_of_light=c
    )
    gamma = _gamma(float(np.linalg.norm(beta)))
    parallel = gamma * gamma / (gamma + 1.0)
    # Jackson (11.149) with v = cβ.
    expected_electric = gamma * (
        electric + c * np.cross(beta, magnetic)
    ) - parallel * np.outer(electric @ beta, beta)
    expected_magnetic = gamma * (
        magnetic - np.cross(beta, electric) / c
    ) - parallel * np.outer(magnetic @ beta, beta)

    np.testing.assert_allclose(boosted_electric, expected_electric, atol=1.0e-12)
    np.testing.assert_allclose(boosted_magnetic, expected_magnetic, atol=1.0e-12)
    np.testing.assert_allclose(
        np.sum(boosted_electric * boosted_magnetic, axis=-1),
        np.sum(electric * magnetic, axis=-1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(boosted_electric**2, axis=-1)
        - c**2 * np.sum(boosted_magnetic**2, axis=-1),
        np.sum(electric**2, axis=-1) - c**2 * np.sum(magnetic**2, axis=-1),
        atol=1.0e-10,
    )


def test_wavevector_boost_has_doppler_factors_and_aberration() -> None:
    speed = 0.6
    theta = np.linspace(0.1, 3.0, 7)
    directions = np.stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)), axis=-1)
    frequency, direction = boost_wavevector(
        jnp.asarray([speed, 0.0, 0.0]),
        jnp.full(theta.shape, 2.0),
        jnp.asarray(directions),
    )
    gamma = _gamma(speed)

    np.testing.assert_allclose(
        frequency, 2.0 * gamma * (1.0 - speed * np.cos(theta)), rtol=1.0e-13
    )
    np.testing.assert_allclose(
        direction[:, 0],
        (np.cos(theta) - speed) / (1.0 - speed * np.cos(theta)),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=1.0e-13)
    receding, _ = boost_wavevector(
        jnp.asarray([speed, 0.0, 0.0]), 1.0, jnp.asarray([1.0, 0.0, 0.0])
    )
    transverse, _ = boost_wavevector(
        jnp.asarray([speed, 0.0, 0.0]), 1.0, jnp.asarray([0.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(
        receding, np.sqrt((1.0 - speed) / (1.0 + speed)), rtol=1.0e-13
    )
    np.testing.assert_allclose(transverse, gamma, rtol=1.0e-13)


def test_boosted_dipole_spectral_energy_integrates_to_boosted_energy() -> None:
    speed = 0.5
    beta = jnp.asarray([speed, 0.0, 0.0])
    center, width = 10.0, 1.0
    total_energy = np.sqrt(2.0 * np.pi) * width

    # Quadrature over the boosted frame; sample the rest-frame spectrum at the
    # inverse-boosted points and push those samples forward.
    frequency = np.linspace(1.0e-3, 40.0, 4001)
    cosine, cosine_weights = np.polynomial.legendre.leggauss(64)
    azimuth = 2.0 * np.pi * np.arange(64) / 64.0
    sine = np.sqrt(1.0 - cosine**2)
    boosted_directions = np.stack(
        (
            sine[:, None] * np.cos(azimuth)[None, :],
            sine[:, None] * np.sin(azimuth)[None, :],
            np.broadcast_to(cosine[:, None], (64, 64)),
        ),
        axis=-1,
    )
    boosted_frequency = jnp.asarray(frequency)[:, None, None]
    rest_frequency, rest_direction = boost_wavevector(
        -beta, boosted_frequency, jnp.asarray(boosted_directions)[None]
    )
    rest_spectrum = (
        jnp.exp(-0.5 * ((rest_frequency - center) / width) ** 2)
        * 3.0
        / (8.0 * jnp.pi)
        * (1.0 - rest_direction[..., 2] ** 2)
    )
    result = transform_spectral_energy(
        beta, rest_frequency, rest_direction, rest_spectrum, emission="complete"
    )
    frequency_weights = np.full(frequency.shape, frequency[1] - frequency[0])
    frequency_weights[[0, -1]] *= 0.5
    angular_weights = cosine_weights[:, None] * np.full(64, 2.0 * np.pi / 64.0)[None, :]
    boosted_energy = np.einsum(
        "f,da,fda->",
        frequency_weights,
        angular_weights,
        np.asarray(result.spectral_energy),
    )

    assert bool(jnp.all(result.valid))
    np.testing.assert_allclose(
        result.angular_frequencies,
        np.broadcast_to(boosted_frequency, rest_frequency.shape),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(boosted_energy, _gamma(speed) * total_energy, rtol=1.0e-6)


def test_spectral_energy_transform_refuses_media_and_truncated_emission() -> None:
    arguments = (
        jnp.asarray([0.2, 0.0, 0.0]),
        jnp.asarray([1.0]),
        jnp.asarray([[0.0, 0.0, 1.0]]),
        jnp.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="vacuum"):
        transform_spectral_energy(*arguments, emission="complete", refractive_index=1.33)
    with pytest.raises(ValueError, match="complete emission"):
        transform_spectral_energy(*arguments, emission="truncated")
    with pytest.raises(ValueError, match="emission"):
        transform_spectral_energy(*arguments, emission="partial")  # ty: ignore[invalid-argument-type]


def test_spectral_energy_transform_marks_superluminal_samples_invalid() -> None:
    result = transform_spectral_energy(
        jnp.asarray([[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        jnp.asarray([1.0, 1.0]),
        emission="complete",
    )

    np.testing.assert_array_equal(result.valid, [True, False])
