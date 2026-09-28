#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.optimize import brentq

import phydrax as phx
from phydrax.electromagnetics import PlasmaWaveMode
from phydrax.optics.geometric import (
    AnalyticRefractiveIndexField,
    ColdPlasmaHamiltonian,
    ColdPlasmaProfile,
    DispersionRayPlan,
    DispersionRayStatus,
    GradedIndexRayPlan,
    RefractiveIndexHamiltonian,
)


pytestmark = pytest.mark.strict_jax

SCALE = phx.ElectromagneticScaleContract.si()
E = float(SCALE.elementary_charge)
M_E = float(SCALE.electron_mass)
EPS0 = float(SCALE.vacuum_permittivity)
OMEGA = 2.0 * np.pi * 1.0e9
CRITICAL_DENSITY = EPS0 * M_E * OMEGA**2 / E**2
COORDINATES = phx.SpatialCoordinateContract(
    phx.units.METER, coordinate_system="cartesian", reference_frame="world"
)
RAMP_LENGTH = 1.0
BASE_DENSITY = 0.1
FIELD = 0.02  # tesla: Y = |Ω_e|/ω ≈ 0.56


def _ramp_profile(field: Callable[[Array], Array] | None = None) -> ColdPlasmaProfile:
    """Electron density ``n_c (X₀ + x/L)``; ``B₀`` along ``ẑ`` unless given."""
    return ColdPlasmaProfile(
        SCALE,
        COORDINATES,
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        density=lambda x: jnp.stack(
            [CRITICAL_DENSITY * (BASE_DENSITY + x[0] / RAMP_LENGTH)]
        ),
        magnetic_field=field
        if field is not None
        else (lambda x: jnp.stack([0.0 * x[0], 0.0 * x[0], FIELD + 0.0 * x[0]])),
        profile_id="linear-ramp",
    )


def _extraordinary_index_squared(x: float) -> float:
    """Independent Appleton–Hartree X mode across ``B₀``: ``1 − X(1−X)/(1−X−Y²)``."""
    plasma = BASE_DENSITY + x / RAMP_LENGTH
    gyro = (E * FIELD / M_E / OMEGA) ** 2
    return 1.0 - plasma * (1.0 - plasma) / (1.0 - plasma - gyro)


def test_ordinary_ray_reflects_on_linear_ramp_along_the_analytic_parabola() -> None:
    angle = 0.5
    hamiltonian = ColdPlasmaHamiltonian(
        _ramp_profile(), angular_frequency=OMEGA, mode=PlasmaWaveMode.ORDINARY
    )
    rays = (
        DispersionRayPlan(hamiltonian, 0.01, 360, hamiltonian_tolerance=1.0e-6)
        .prepare()
        .integrate(
            np.asarray(((0.0, 0.0, 0.0),)),
            np.asarray(((np.cos(angle), np.sin(angle), 0.0),)),
        )
    )
    assert bool(rays.evidence.successful)
    assert int(rays.evidence.status[0]) == 0
    # Across B₀ the O mode has n² = 1 − X: H ∝ ½(p² − n²) gives the parabola
    # x = y cot θ₀ − y²/(4 L n₀² sin²θ₀), turning at x_t = L n₀² cos²θ₀.
    index0_squared = 1.0 - BASE_DENSITY
    history = np.asarray(rays.position_history[:, 0])
    x, y = history[:, 0], history[:, 1]
    parabola = y / np.tan(angle) - y**2 / (
        4.0 * RAMP_LENGTH * index0_squared * np.sin(angle) ** 2
    )
    np.testing.assert_allclose(x, parabola, atol=1.0e-7)
    np.testing.assert_allclose(history[:, 2], 0.0, atol=1.0e-12)
    turning = RAMP_LENGTH * index0_squared * np.cos(angle) ** 2
    assert turning - 1.0e-4 <= x.max() <= turning
    # |dx/dτ| = v_g/c = n for a cold unmagnetized-like O mode: τ = c t.
    exit_index = int(np.argmax(x < 0.0))
    fraction = x[exit_index - 1] / (x[exit_index - 1] - x[exit_index])
    exit_y = y[exit_index - 1] + fraction * (y[exit_index] - y[exit_index - 1])
    np.testing.assert_allclose(
        exit_y,
        4.0 * RAMP_LENGTH * index0_squared * np.sin(angle) * np.cos(angle),
        rtol=1.0e-6,
    )


def test_extraordinary_ray_turns_at_the_appleton_hartree_turning_point() -> None:
    angle = 0.4
    hamiltonian = ColdPlasmaHamiltonian(
        _ramp_profile(), angular_frequency=OMEGA, mode=PlasmaWaveMode.EXTRAORDINARY
    )
    rays = (
        DispersionRayPlan(hamiltonian, 0.005, 300, hamiltonian_tolerance=1.0e-6)
        .prepare()
        .integrate(
            np.asarray(((0.0, 0.0, 0.0),)),
            np.asarray(((np.cos(angle), np.sin(angle), 0.0),)),
        )
    )
    assert bool(rays.evidence.successful)
    transverse = _extraordinary_index_squared(0.0) * np.sin(angle) ** 2
    gyro = E * FIELD / (M_E * OMEGA)
    right_cutoff = (1.0 - gyro - BASE_DENSITY) * RAMP_LENGTH
    turning = brentq(
        lambda x: _extraordinary_index_squared(x) - transverse,
        0.0,
        right_cutoff * (1.0 - 1.0e-9),
    )
    history = np.asarray(rays.position_history[:, 0])
    assert turning - 1.0e-4 <= history[:, 0].max() <= turning + 1.0e-7
    momenta = np.asarray(rays.momentum_history[:, 0])
    # Translation invariance in y conserves p_y = n₀ sin θ₀.
    np.testing.assert_allclose(momenta[:, 1], np.sqrt(transverse), rtol=1.0e-10)


def test_normal_incidence_reaches_the_cutoff_and_returns() -> None:
    hamiltonian = ColdPlasmaHamiltonian(
        _ramp_profile(), angular_frequency=OMEGA, mode=PlasmaWaveMode.ORDINARY
    )
    index0 = np.sqrt(1.0 - BASE_DENSITY)
    # τ = 4 L n₀ returns the ray to the launch plane.
    steps = 360
    rays = (
        DispersionRayPlan(
            hamiltonian,
            4.0 * RAMP_LENGTH * index0 / steps,
            steps,
            hamiltonian_tolerance=1.0e-6,
        )
        .prepare()
        .integrate(np.asarray(((0.0, 0.0, 0.0),)), np.asarray(((1.0, 0.0, 0.0),)))
    )
    status = DispersionRayStatus(int(rays.evidence.status[0]))
    assert status == DispersionRayStatus.CUTOFF
    assert bool(rays.evidence.successful)
    assert float(rays.evidence.minimum_refractive_index[0]) < 1.0e-2
    # The turning point x = L n₀² is where n² = 1 − X vanishes; the reflected ray
    # returns along the incident line with reversed momentum, up to O(h²) timing.
    np.testing.assert_allclose(
        np.max(rays.position_history[:, 0, 0]), RAMP_LENGTH * index0**2, atol=1.0e-3
    )
    np.testing.assert_allclose(rays.state.positions[0], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(rays.state.momenta[0], (-index0, 0.0, 0.0), atol=1.0e-3)
    np.testing.assert_allclose(rays.position_history[:, 0, 1:], 0.0, atol=1.0e-12)


def test_implicit_midpoint_energy_error_is_second_order_and_symplectic() -> None:
    hamiltonian = ColdPlasmaHamiltonian(
        _ramp_profile(), angular_frequency=OMEGA, mode=PlasmaWaveMode.EXTRAORDINARY
    )
    launch = (np.asarray(((0.0, 0.0, 0.0),)), np.asarray(((0.8, 0.6, 0.0),)))
    drift = []
    for step, count in ((0.02, 40), (0.01, 80)):
        rays = (
            DispersionRayPlan(hamiltonian, step, count, hamiltonian_tolerance=1.0e-4)
            .prepare()
            .integrate(*launch)
        )
        assert bool(rays.evidence.root_converged)
        assert float(rays.evidence.maximum_symplectic_residual) < 1.0e-10
        drift.append(float(rays.evidence.maximum_hamiltonian_drift))
    # Implicit midpoint conserves a nonquadratic H to O(h²).
    assert 3.0 < drift[0] / drift[1] < 5.0


def test_implicit_midpoint_agrees_with_kick_drift_kick_on_graded_index() -> None:
    field = AnalyticRefractiveIndexField(
        lambda point: 1.2 - 0.05 * (point[0] - 1.0) ** 2 + 0.02 * point[1],
        COORDINATES,
        field_id="bent-lens",
    )
    positions = np.asarray(((1.3, 0.0, 0.0), (0.8, 0.2, 0.0)))
    directions = np.asarray(((0.1, 0.0, 1.0), (0.0, 0.1, 1.0)))
    separable = (
        GradedIndexRayPlan(field, 0.005, 400).prepare().integrate(positions, directions)
    )
    implicit = (
        DispersionRayPlan(
            RefractiveIndexHamiltonian(field), 0.005, 400, hamiltonian_tolerance=1.0e-8
        )
        .prepare()
        .integrate(positions, directions)
    )
    assert bool(implicit.evidence.successful)
    np.testing.assert_allclose(
        implicit.state.positions, separable.state.positions, atol=2.0e-5
    )
    np.testing.assert_allclose(
        implicit.state.optical_lengths, separable.state.optical_lengths, rtol=1.0e-6
    )


def test_plasma_rays_refuse_invalid_launches_and_media() -> None:
    profile = _ramp_profile()
    hamiltonian = ColdPlasmaHamiltonian(
        profile, angular_frequency=OMEGA, mode=PlasmaWaveMode.ORDINARY
    )
    with pytest.raises(ValueError, match="Separable"):
        DispersionRayPlan(hamiltonian, 0.01, 4, method="kick-drift-kick")
    # Beyond the O-mode cutoff (X > 1) the launch root is evanescent.
    rays = (
        DispersionRayPlan(hamiltonian, 0.01, 2)
        .prepare()
        .integrate(np.asarray(((1.2, 0.0, 0.0),)), np.asarray(((1.0, 0.0, 0.0),)))
    )
    assert int(rays.evidence.status[0]) & DispersionRayStatus.LAUNCH_INVALID
    assert not bool(rays.evidence.successful)
    # In vacuum both modes coincide and no mode can be selected.
    vacuum = ColdPlasmaProfile(
        SCALE,
        COORDINATES,
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        density=lambda x: jnp.stack([0.0 * x[0]]),
        magnetic_field=lambda x: jnp.stack([0.0 * x[0], 0.0 * x[0], FIELD + 0.0 * x[0]]),
        profile_id="vacuum",
    )
    rays = (
        DispersionRayPlan(
            ColdPlasmaHamiltonian(
                vacuum, angular_frequency=OMEGA, mode=PlasmaWaveMode.ORDINARY
            ),
            0.01,
            2,
        )
        .prepare()
        .integrate(np.asarray(((0.0, 0.0, 0.0),)), np.asarray(((1.0, 0.0, 0.0),)))
    )
    assert int(rays.evidence.status[0]) & DispersionRayStatus.LAUNCH_INVALID
    other = ColdPlasmaHamiltonian(
        profile, angular_frequency=OMEGA, mode=PlasmaWaveMode.EXTRAORDINARY
    )
    with pytest.raises(ValueError, match="ColdPlasmaHamiltonian"):
        other.sample_path(rays, (0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="length unit"):
        ColdPlasmaProfile(
            SCALE,
            phx.SpatialCoordinateContract(phx.units.MILLIMETER),
            charge_numbers=[-1.0],
            mass_ratios=[1.0],
            density=lambda x: x[:1],
            magnetic_field=lambda x: x,
            profile_id="millimeter",
        )
