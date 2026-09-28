#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic uniform-motion field against independent electrodynamics.

References: Frank–Tamm, d²W/(dω dl) = (q² μ ω / 4π)(1 − 1/(β² n²)) (Jackson
13.48 in SI, one-sided ω > 0); the 2-D line-charge Cherenkov slab flux follows
from the plane-wave energy of the two cone waves (derived in the test); the
field itself is checked against the source-free Maxwell curl equations by
automatic differentiation and against Gauss's law at the path.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


em = phx.electromagnetics
# Code units with ε₀ = μ₀ = c = 1: every formula below is unit-agnostic.
OMEGA = np.asarray([0.7, 1.3, 2.9])


def _plan(
    geometry: str,
    permittivity: complex,
    permeability: complex,
    speed: float,
    *,
    charge: float = 1.3,
) -> em.UniformMotionFieldPlan:
    medium = em.UniformMotionMedium(OMEGA, permittivity, permeability)
    dimension = 3 if geometry == "point" else 2
    direction = np.zeros(dimension)
    direction[0] = 1.0
    return em.UniformMotionFieldPlan(
        geometry,  # ty: ignore[invalid-argument-type]
        medium,
        charge=charge,
        speed=speed,
        origin=np.full(dimension, 0.25),
        direction=direction,
    )


def test_point_charge_cherenkov_flux_equals_frank_tamm() -> None:
    epsilon, mu, beta, charge = 2.25, 1.1, 0.9, 1.3
    plan = _plan("point", epsilon, mu, beta, charge=charge)
    frank_tamm = (
        charge**2 * mu * OMEGA / (4.0 * np.pi) * (1.0 - 1.0 / (beta**2 * epsilon * mu))
    )
    for radius in (0.8, 7.5, 41.0):
        np.testing.assert_allclose(
            plan.radial_energy_flux(radius), frank_tamm, rtol=1e-11
        )
    evidence = plan.evidence()
    assert bool(jnp.all(evidence.radiating))
    np.testing.assert_allclose(
        evidence.cherenkov_cosine, 1.0 / (beta * np.sqrt(epsilon * mu)), rtol=1e-12
    )


def test_line_charge_cherenkov_flux_equals_cone_wave_energy() -> None:
    # Each cone wave carries |H_z|² η cosθ_⊥ per unit area of the slab faces,
    # with |H_z| = λ/2 (half the surface-current jump λ v e^{iωx/v}/v); the
    # transverse direction cosine is sinθ_c = √(1 − 1/(β²n²)). Two faces and the
    # one-sided factor 2/π·½ give (λ²/2π) η sinθ_c.
    epsilon, mu, beta, density = 4.0, 1.0, 0.8, 0.6
    plan = _plan("line", epsilon, mu, beta, charge=density)
    sine = np.sqrt(1.0 - 1.0 / (beta**2 * epsilon * mu))
    expected = density**2 / (2.0 * np.pi) * np.sqrt(mu / epsilon) * sine
    for distance in (0.3, 12.0):
        np.testing.assert_allclose(
            plan.radial_energy_flux(distance), np.full(OMEGA.shape, expected), rtol=1e-12
        )


@pytest.mark.parametrize("geometry", ["point", "line"])
def test_below_threshold_bound_field_radiates_nothing(geometry: str) -> None:
    plan = _plan(geometry, 1.44, 1.0, 0.5)
    flux = np.asarray(plan.radial_energy_flux(0.9))
    scale = 1.3**2 * OMEGA / (4.0 * np.pi)
    assert np.all(np.abs(flux) <= 1e-13 * scale)
    evidence = plan.evidence()
    assert not bool(jnp.any(evidence.radiating))
    # Vacuum-like bound reach: 2π/Im k_ρ = 2π γ_n v / ω with β_n² = εμv².
    gamma = 1.0 / np.sqrt(1.0 - 1.44 * 0.25)
    np.testing.assert_allclose(
        evidence.bound_extent, 2.0 * np.pi * gamma * 0.5 / OMEGA, rtol=1e-12
    )


@pytest.mark.parametrize(
    ("permittivity", "permeability", "speed"),
    [(1.44, 1.0, 0.5), (2.25, 1.1, 0.9), (2.0 + 0.3j, 1.0 + 0.05j, 0.9)],
    ids=["bound", "radiating", "lossy"],
)
def test_point_field_solves_source_free_maxwell(
    permittivity: complex, permeability: complex, speed: float
) -> None:
    plan = _plan("point", permittivity, permeability, speed)
    point = jnp.asarray([0.7, 1.1, -0.4])

    def electric(value: Array) -> Array:
        field = plan.evaluate(value[None, :]).electric[:, 0]
        return jnp.concatenate((jnp.real(field), jnp.imag(field)), axis=-1)

    def magnetic(value: Array) -> Array:
        field = plan.evaluate(value[None, :]).magnetic[:, 0]
        return jnp.concatenate((jnp.real(field), jnp.imag(field)), axis=-1)

    def complex_curl(field: Callable[[Array], Array]) -> Array:
        jacobian = jax.jacfwd(field)(point)
        derivative = jacobian[:, :3] + 1j * jacobian[:, 3:]
        return jnp.stack(
            (
                derivative[:, 2, 1] - derivative[:, 1, 2],
                derivative[:, 0, 2] - derivative[:, 2, 0],
                derivative[:, 1, 0] - derivative[:, 0, 1],
            ),
            axis=-1,
        )

    field = plan.evaluate(point[None, :])
    omega = OMEGA[:, None]
    np.testing.assert_allclose(
        complex_curl(electric),
        1j * omega * permeability * field.magnetic[:, 0],
        rtol=1e-9,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        complex_curl(magnetic),
        -1j * omega * permittivity * field.electric[:, 0],
        rtol=1e-9,
        atol=1e-12,
    )


def test_line_field_solves_source_free_maxwell() -> None:
    epsilon, mu = 3.0 + 0.2j, 1.2
    plan = _plan("line", epsilon, mu, 0.85)
    point = jnp.asarray([0.9, -0.6])

    def split(value: Array) -> Array:
        field = plan.evaluate(value[None, :])
        packed = jnp.concatenate((field.electric[:, 0], field.magnetic[:, 0]), axis=-1)
        return jnp.concatenate((jnp.real(packed), jnp.imag(packed)), axis=-1)

    jacobian = jax.jacfwd(split)(point)
    derivative = jacobian[:, :3] + 1j * jacobian[:, 3:]
    field = plan.evaluate(point[None, :])
    omega = OMEGA
    # z-invariant: (∇×E)_z = ∂x E_y − ∂y E_x and ∇×(H_z ẑ) = (∂y H_z, −∂x H_z).
    curl_electric = derivative[:, 1, 0] - derivative[:, 0, 1]
    np.testing.assert_allclose(
        curl_electric, 1j * omega * mu * field.magnetic[:, 0, 0], rtol=1e-10
    )
    np.testing.assert_allclose(
        jnp.stack((derivative[:, 2, 1], -derivative[:, 2, 0]), axis=-1),
        -1j * omega[:, None] * epsilon * field.electric[:, 0],
        rtol=1e-10,
    )


def test_gauss_law_fixes_the_charge_normalization() -> None:
    epsilon, speed, charge = 2.0 + 0.1j, 0.6, 1.3
    point_plan = _plan("point", epsilon, 1.0, speed, charge=charge)
    radius = 1e-7
    along = 0.4
    probe = jnp.asarray([[0.25 + along, 0.25 + radius, 0.25]])
    field = point_plan.evaluate(probe)
    phase = np.exp(1j * OMEGA * along / speed)
    # Flux of D through a thin cylinder per unit length → ρ̃ line density q/v.
    np.testing.assert_allclose(
        2.0 * np.pi * radius * epsilon * field.electric[:, 0, 1] / phase,
        np.full(OMEGA.shape, charge / speed),
        rtol=1e-6,
    )
    line_plan = _plan("line", epsilon, 1.0, speed, charge=charge)
    sides = jnp.asarray([[0.25 + along, 0.25 + 1e-12], [0.25 + along, 0.25 - 1e-12]])
    field = line_plan.evaluate(sides)
    jump = epsilon * (field.electric[:, 0, 1] - field.electric[:, 1, 1]) / phase
    np.testing.assert_allclose(jump, np.full(OMEGA.shape, charge / speed), rtol=1e-9)


def test_radiating_branch_is_the_passive_limit() -> None:
    lossless = _plan("point", 2.25, 1.0, 0.9)
    lossy = _plan("point", 2.25 + 1e-9j, 1.0, 0.9)
    wavenumber = lossless.medium.transverse_wavenumber(0.9)
    assert bool(jnp.all(jnp.real(wavenumber) > 0.0))
    points = jnp.asarray([[3.0, 2.0, 0.25], [1.0, 0.25, 9.0]])
    np.testing.assert_allclose(
        lossless.evaluate(points).electric, lossy.evaluate(points).electric, rtol=1e-6
    )
    # Negative index: phase runs inward (reversed Cherenkov) in the passive limit.
    negative = _plan("point", -2.25, -1.0, 0.9)
    negative_lossy = _plan("point", -2.25 + 1e-9j, -1.0 + 1e-9j, 0.9)
    reversed_wavenumber = negative.medium.transverse_wavenumber(0.9)
    assert bool(jnp.all(jnp.real(reversed_wavenumber) < 0.0))
    np.testing.assert_allclose(
        negative.evaluate(points).electric,
        negative_lossy.evaluate(points).electric,
        rtol=1e-6,
    )
    frank_tamm = 1.3**2 * -1.0 * OMEGA / (4.0 * np.pi) * (1.0 - 1.0 / (0.81 * 2.25))
    # μ < 0 flips the sign of the textbook expression; energy still leaves.
    np.testing.assert_allclose(negative.radial_energy_flux(5.0), -frank_tamm, rtol=1e-10)


def test_path_points_are_unsupported_not_zero() -> None:
    plan = _plan("point", 1.0, 1.0, 0.5)
    field = plan.evaluate(jnp.asarray([[3.0, 0.25, 0.25], [3.0, 1.0, 0.25]]))
    np.testing.assert_array_equal(field.supported, [False, True])
    assert bool(jnp.all(jnp.isnan(field.electric[:, 0])))
    assert bool(jnp.all(jnp.isfinite(field.electric[:, 1])))


def test_invalid_inputs_are_refused() -> None:
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="passive"):
        em.UniformMotionMedium(OMEGA, 2.0 - 0.1j, 1.0)
    medium = em.UniformMotionMedium(OMEGA, 2.0, 1.0)
    with pytest.raises(ValueError, match="3 origin and direction"):
        em.UniformMotionFieldPlan(
            "point",
            medium,
            charge=1.0,
            speed=0.5,
            origin=[0.0, 0.0],
            direction=[1.0, 0.0],
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="speed"):
        em.UniformMotionFieldPlan(
            "line",
            medium,
            charge=1.0,
            speed=-0.5,
            origin=[0.0, 0.0],
            direction=[1.0, 0.0],
        )
    with pytest.raises(ValueError, match="geometry"):
        em.UniformMotionFieldPlan(
            "sheet",  # ty: ignore[invalid-argument-type]
            medium,
            charge=1.0,
            speed=0.5,
            origin=[0.0, 0.0],
            direction=[1.0, 0.0],
        )
