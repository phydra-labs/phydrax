#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import phydrax.bubble_dynamics as bd


DIFFUSIVITY = 2.0e-9
SATURATION = 0.6
AMBIENT = 101325.0
TEMPERATURE = 293.15
RADIUS = 1.0e-5


def _properties(ratio: float, tension: float) -> bd.GasSolutionProperties:
    return bd.GasSolutionProperties(DIFFUSIVITY, SATURATION, ratio, tension, AMBIENT, TEMPERATURE)


class _RadiusFloor:
    """Terminal `solve_ivp` event at the dissolution radius `10⁻³ R₀`."""

    terminal = True

    def __call__(self, root: float, radius: np.ndarray) -> float:
        del root
        return float(radius[0] - 1.0e-3 * RADIUS)


def test_quasi_static_lifetime_matches_the_closed_form_in_its_assumptions() -> None:
    properties = _properties(0.2, 0.0)
    lifetime = float(bd.quasi_static_dissolution_time(RADIUS, properties))
    times = np.linspace(0.0, 1.2 * lifetime, 13)
    plan = bd.EpsteinPlessetPlan(properties, times, route="quasi_static", initial_radius=RADIUS)
    result = bd.solve_epstein_plesset(plan.prepare(RADIUS))
    assert int(result.status) == bd.BubbleDynamicsStatus.DISSOLVED
    assert bool(result.successful)
    # The event stops at 10⁻³ R₀, i.e. at (1 − 10⁻⁶) of the full lifetime.
    assert float(result.evidence.lifetime) == pytest.approx(lifetime * (1.0 - 1.0e-6), rel=1.0e-8)
    valid = np.asarray(result.valid)
    expected = np.asarray(bd.quasi_static_dissolution_radius(times, RADIUS, lifetime))
    np.testing.assert_allclose(np.asarray(result.radius)[valid], expected[valid], rtol=1.0e-7, atol=1.0e-13)
    assert abs(float(result.evidence.amount_residual)) < 1.0e-8
    assert result.evidence.route == "quasi_static"


def test_full_history_route_matches_independent_epstein_plesset_integration() -> None:
    properties = _properties(0.0, 0.0)
    lifetime = float(bd.quasi_static_dissolution_time(RADIUS, properties))
    plan = bd.EpsteinPlessetPlan(
        properties, np.linspace(0.0, 1.2 * lifetime, 13), route="full_history", initial_radius=RADIUS
    )
    result = bd.solve_epstein_plesset(plan.prepare(RADIUS))
    rate = DIFFUSIVITY * SATURATION / float(properties.gas_molar_density())

    def field(root: float, radius: np.ndarray) -> list[float]:
        return [-2.0 * root * rate / radius[0] - 2.0 * rate / np.sqrt(np.pi * DIFFUSIVITY)]

    reference = solve_ivp(
        field,
        (0.0, np.sqrt(1.2 * lifetime)),
        [RADIUS],
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-20,
        events=_RadiusFloor(),
    )
    assert int(result.status) == bd.BubbleDynamicsStatus.DISSOLVED
    assert float(result.evidence.lifetime) == pytest.approx(reference.t_events[0][0] ** 2, rel=1.0e-8)
    assert float(result.evidence.lifetime) < lifetime
    assert plan.history_kernel_error < 1.0e-5


def test_laplace_pressure_drives_dissolution_consistently_in_a_saturated_liquid() -> None:
    properties = _properties(1.0, 0.072)
    times = np.linspace(0.0, 2.0, 9)
    results = [
        bd.solve_epstein_plesset(
            bd.EpsteinPlessetPlan(properties, times, route=route, initial_radius=RADIUS).prepare(RADIUS)
        )
        for route in ("quasi_static", "full_history")
    ]
    for result in results:
        radius = np.asarray(result.radius)
        assert np.all(np.diff(radius) < 0.0)
        margin = 2.0 * 0.072 / (RADIUS * AMBIENT)
        assert float(result.evidence.initial_saturation_margin) == pytest.approx(margin, rel=1.0e-12)
        amount = np.asarray(result.amount)
        expected = (AMBIENT + 2.0 * 0.072 / radius) * 4.0 * np.pi * radius**3 / (
            3.0 * bd.MOLAR_GAS_CONSTANT * TEMPERATURE
        )
        np.testing.assert_allclose(amount, expected, rtol=1.0e-12)
        assert abs(float(result.evidence.amount_residual)) < 1.0e-8
    quasi, history = (np.asarray(result.radius)[-1] for result in results)
    assert history < quasi


def test_lohse_zhang_pinned_equilibrium_and_stability_sign() -> None:
    footprint, oversaturation, tension = 1.0e-6, 1.0, 0.072
    plan = bd.PinnedSurfaceBubblePlan(
        _properties(2.0, tension), oversaturation, np.linspace(0.0, 1.0e-3, 11), contact="pinned"
    )
    equilibrium = plan.equilibrium(footprint)
    critical = 4.0 * tension / AMBIENT
    expected = np.arcsin(oversaturation * footprint / critical)
    assert bool(equilibrium.exists)
    assert float(equilibrium.gas_contact_angle) == pytest.approx(expected, rel=1.0e-12)
    assert float(np.degrees(equilibrium.gas_contact_angle)) == pytest.approx(20.6, abs=0.05)
    assert float(equilibrium.liquid_contact_angle) == pytest.approx(np.pi - expected, rel=1.0e-12)
    assert float(equilibrium.stability_derivative) < 0.0
    unstable = plan.angle_rate
    steep = np.pi - expected
    assert float(jax.grad(unstable, argnums=1)(footprint, steep)) > 0.0

    for start in (35.0, 12.0):
        prepared = plan.prepare(footprint, np.pi - np.radians(start))
        result = bd.solve_surface_bubble(prepared)
        assert bool(result.successful)
        angles = np.asarray(result.gas_contact_angle)
        distance = np.abs(angles - expected)
        assert np.all(np.diff(distance) < 0.0)
        np.testing.assert_allclose(np.asarray(result.footprint_diameter), footprint, rtol=1.0e-14)
        assert abs(float(result.evidence.amount_residual)) < 1.0e-8


def test_unpinned_surface_bubble_dissolves_under_undersaturation() -> None:
    plan = bd.PinnedSurfaceBubblePlan(
        _properties(0.9, 0.072), -0.1, np.linspace(0.0, 1.0e-2, 11), contact="unpinned"
    )
    result = bd.solve_surface_bubble(plan.prepare(1.0e-6, np.pi - np.radians(20.0)))
    assert int(result.status) == bd.BubbleDynamicsStatus.DISSOLVED
    assert bool(result.evidence.dissolved)
    assert 0.0 < float(result.evidence.lifetime) < 1.0e-2
    valid = np.asarray(result.valid)
    np.testing.assert_allclose(np.asarray(result.gas_contact_angle)[valid], np.radians(20.0), rtol=1.0e-14)
    assert np.all(np.diff(np.asarray(result.footprint_diameter)[valid]) < 0.0)


def test_popov_flux_factor_limits_and_continuum_evidence() -> None:
    assert float(bd.popov_flux_factor(np.pi / 2.0)) == pytest.approx(2.0, rel=1.0e-12)
    assert float(bd.popov_flux_factor(1.0e-8)) == pytest.approx(4.0 / np.pi, rel=1.0e-7)
    policy = bd.BubbleValidityPolicy(
        knudsen_limit=0.01, molecular_diameter=3.64e-10, tolman_length=2.0e-10
    )
    plan = bd.PinnedSurfaceBubblePlan(
        _properties(2.0, 0.072), 1.0, np.linspace(0.0, 1.0e-6, 3), contact="pinned", validity=policy
    )
    result = bd.solve_surface_bubble(plan.prepare(1.0e-7, np.pi - np.radians(10.0)))
    assert result.evidence.max_knudsen is not None
    assert result.evidence.max_tolman_ratio is not None
    assert float(result.evidence.max_knudsen) > 0.01
    assert not bool(result.evidence.continuum_support)
    assert float(result.evidence.max_laplace_ratio) > 1.0
