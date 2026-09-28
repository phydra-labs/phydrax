#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rest-frame integrated-Green-function space charge of relativistic bunches.

The reference is the closed-form field of a spherical Gaussian charge in its
rest frame, carried to the lab by the textbook field transformation of a
uniformly moving source: ``E⊥ = γE'⊥``, ``E_z = E'_z``, ``B = v × E/c²``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import erf

import phydrax as phx
from phydrax.applications.accelerator import AcceleratorBunch
from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec


def _spherical_gaussian_field(
    points: np.ndarray, sigma: float, charge: float, permittivity: float
) -> np.ndarray:
    radius = np.sqrt(np.sum(points**2, axis=-1))
    radial = (
        charge
        / (4.0 * np.pi * permittivity * radius**2)
        * (
            erf(radius / (np.sqrt(2.0) * sigma))
            - np.sqrt(2.0 / np.pi)
            * (radius / sigma)
            * np.exp(-0.5 * (radius / sigma) ** 2)
        )
    )
    return radial[..., None] * points / radius[..., None]


def test_relativistic_gaussian_bunch_fields_and_kick_match_boosted_coulomb_field() -> (
    None
):
    accelerator = phx.applications.accelerator
    scale = phx.ElectromagneticScaleContract.si()
    light = float(scale.speed_of_light)
    elementary = float(scale.elementary_charge)
    permittivity = float(scale.vacuum_permittivity)
    rest_energy = float(scale.electron_mass) * light**2
    momentum = 5.0e6 * elementary
    gamma = np.sqrt(1.0 + (momentum / rest_energy) ** 2)
    beta = momentum / rest_energy / gamma
    sigma = 1.0e-3
    count = 34
    half = 4.25 * sigma
    grid = TensorGridPlan((UniformCellAxisSpec(count),) * 3).prepare(
        np.asarray([[-half] * 3, [half] * 3])
    )
    centers = np.asarray(grid.points).reshape(-1, 3)
    spacing = 2.0 * half / count
    # Macro-particles at the interior rest-frame cell centers (|x_i| < 4σ) of a
    # spherical Gaussian of rest-frame width σ, i.e. a lab bunch of length σ/γ.
    # Multilinear assignment of a particle on a cell center is exact; the
    # remaining solve error is second order, ≈ 0.18 (h/σ)² ≈ 1.1 % at h = σ/4.
    interior = np.all(np.abs(centers) < 4.0 * sigma, axis=-1)
    positions = centers[interior]
    total_charge = 1.0e-9
    profile = np.exp(-0.5 * np.sum((positions / sigma) ** 2, axis=-1)) / (
        (2.0 * np.pi) ** 1.5 * sigma**3
    )
    weights = total_charge * profile * spacing**3 / elementary
    coordinates = np.zeros((positions.shape[0], 6))
    coordinates[:, 0] = positions[:, 0]
    coordinates[:, 2] = positions[:, 1]
    coordinates[:, 4] = -positions[:, 2] / gamma  # positive-late: zeta = −z
    bunch = accelerator.AcceleratorBunch(
        jnp.asarray(coordinates),
        jnp.asarray(weights),
        jnp.arange(positions.shape[0]),
        reference_rest_energy=rest_energy,
        reference_momentum=momentum,
        reference_charge=-1.0,
        bunch_id="gaussian",
    )
    plan = accelerator.SpaceChargeIGFPlan(scale, grid, capacity=positions.shape[0])
    step = 0.1
    result = plan.kick(bunch, step)

    assert bool(result.accepted)
    assert bool(jnp.all(result.support))
    np.testing.assert_allclose(float(result.reference_gamma), gamma, rtol=1e-12)
    deposited = -elementary * float(np.sum(weights))
    np.testing.assert_allclose(float(result.deposited_charge), deposited, rtol=1e-12)

    rest_field = _spherical_gaussian_field(centers, sigma, deposited, permittivity)
    field_scale = np.max(np.abs(rest_field))
    np.testing.assert_allclose(
        np.asarray(result.rest_frame_electric_field).reshape(-1, 3),
        rest_field,
        atol=2e-2 * field_scale,
    )

    rest_electric = np.asarray(result.rest_frame_electric_field)
    electric = np.asarray(result.electric_field)
    magnetic = np.asarray(result.magnetic_field)
    np.testing.assert_allclose(
        electric[..., :2], gamma * rest_electric[..., :2], rtol=1e-12, atol=0.0
    )
    np.testing.assert_allclose(
        electric[..., 2], rest_electric[..., 2], rtol=0.0, atol=1e-12 * field_scale
    )
    velocity = np.asarray([0.0, 0.0, beta * light])
    np.testing.assert_allclose(
        magnetic,
        np.cross(velocity, electric) / light**2,
        rtol=0.0,
        atol=1e-12 * gamma * field_scale / light,
    )

    # Reference particles: Δp⊥c = qE⊥(1 − β²)Δs/β = qγE'⊥Δs/(γ²β); Δp_z c = qE'_zΔs/β.
    charge = -elementary
    expected_kick = charge * rest_field[interior] * step / beta / momentum
    expected_kick[:, :2] /= gamma
    kick = np.asarray(result.momentum_kick)
    for axis in range(3):
        scale_axis = np.max(np.abs(expected_kick[:, axis]))
        np.testing.assert_allclose(
            kick[:, axis], expected_kick[:, axis], atol=2e-2 * scale_axis
        )
    updated = np.asarray(result.bunch.coordinates)
    np.testing.assert_allclose(updated[:, 1], kick[:, 0], atol=1e-15)
    np.testing.assert_allclose(updated[:, 3], kick[:, 1], atol=1e-15)


def test_space_charge_kick_is_refused_outside_its_admissible_domain() -> None:
    accelerator = phx.applications.accelerator
    scale = phx.ElectromagneticScaleContract.si()
    light = float(scale.speed_of_light)
    rest_energy = float(scale.electron_mass) * light**2
    momentum = 50.0e6 * float(scale.elementary_charge)
    grid = TensorGridPlan((UniformCellAxisSpec(6),) * 3).prepare(
        np.asarray([[-1.0e-3] * 3, [1.0e-3] * 3])
    )
    plan = accelerator.SpaceChargeIGFPlan(
        scale, grid, capacity=2, maximum_rest_frame_speed=0.1
    )

    def bunch(coordinates: np.ndarray) -> AcceleratorBunch:
        return accelerator.AcceleratorBunch(
            jnp.asarray(coordinates),
            jnp.full((2,), 1.0e6),
            jnp.arange(2),
            reference_rest_energy=rest_energy,
            reference_momentum=momentum,
            reference_charge=-1.0,
            bunch_id="probe",
        )

    inside = np.asarray(
        [[1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    accepted = plan.kick(bunch(inside), 0.1)
    assert bool(accepted.accepted)

    escaped = inside.copy()
    escaped[0, 0] = 5.0e-3
    refused = plan.kick(bunch(escaped), 0.1)
    assert not bool(refused.accepted)
    assert not bool(refused.support[0])
    np.testing.assert_array_equal(np.asarray(refused.bunch.coordinates), escaped)

    # A transverse angle of 0.1 rad at γ ≈ 98 is rest-frame |β'| ≈ 1.
    hot = inside.copy()
    hot[0, 1] = 0.1
    too_fast = plan.kick(bunch(hot), 0.1)
    assert float(too_fast.maximum_rest_frame_speed) > 0.1
    assert not bool(too_fast.accepted)
    np.testing.assert_array_equal(np.asarray(too_fast.bunch.coordinates), hot)
