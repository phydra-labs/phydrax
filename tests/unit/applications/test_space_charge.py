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
from phydrax.applications.accelerator import AcceleratorBunch, SpaceChargeIGFPlan
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
    # Two probe charges have no rest-frame extent across the line joining them;
    # this test isolates support and speed refusal from resolution refusal.
    plan = accelerator.SpaceChargeIGFPlan(
        scale, grid, capacity=2, maximum_rest_frame_speed=0.1, minimum_cells_per_sigma=0.0
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


def _gaussian_bunch(
    sigma: float, gamma: float, spacing: float, momentum: float
) -> tuple[AcceleratorBunch, np.ndarray]:
    """Spherical rest-frame Gaussian sampled at the centers of a ``spacing`` lattice."""
    scale = phx.ElectromagneticScaleContract.si()
    rest_energy = float(scale.electron_mass) * float(scale.speed_of_light) ** 2
    axis = (np.arange(-16, 16) + 0.5) * spacing
    positions = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    positions = positions.reshape(-1, 3)
    positions = positions[np.all(np.abs(positions) < 4.0 * sigma, axis=-1)]
    profile = np.exp(-0.5 * np.sum((positions / sigma) ** 2, axis=-1))
    weights = 1.0e-9 / float(scale.elementary_charge) * profile / np.sum(profile)
    coordinates = np.zeros((positions.shape[0], 6))
    coordinates[:, 0] = positions[:, 0]
    coordinates[:, 2] = positions[:, 1]
    coordinates[:, 4] = -positions[:, 2] / gamma
    bunch = AcceleratorBunch(
        jnp.asarray(coordinates),
        jnp.asarray(weights),
        jnp.arange(positions.shape[0]),
        reference_rest_energy=rest_energy,
        reference_momentum=momentum,
        reference_charge=-1.0,
        bunch_id="gaussian",
    )
    return bunch, positions


def test_under_resolved_bunch_is_refused_with_cells_per_sigma_evidence() -> None:
    accelerator = phx.applications.accelerator
    scale = phx.ElectromagneticScaleContract.si()
    rest_energy = float(scale.electron_mass) * float(scale.speed_of_light) ** 2
    momentum = 5.0e6 * float(scale.elementary_charge)
    gamma = np.sqrt(1.0 + (momentum / rest_energy) ** 2)
    sigma = 1.0e-3
    bunch, positions = _gaussian_bunch(sigma, gamma, sigma / 4.0, momentum)
    rms = np.sqrt(np.average(positions**2, axis=0, weights=np.asarray(bunch.weights)))

    def plan(cells_per_sigma: float) -> SpaceChargeIGFPlan:
        count = int(round(9.0 * cells_per_sigma))
        grid = TensorGridPlan((UniformCellAxisSpec(count),) * 3).prepare(
            np.asarray([[-4.5 * sigma] * 3, [4.5 * sigma] * 3])
        )
        return accelerator.SpaceChargeIGFPlan(scale, grid, capacity=positions.shape[0])

    resolved = plan(4.0).kick(bunch, 0.1)
    assert bool(resolved.resolved)
    assert bool(resolved.accepted)
    np.testing.assert_allclose(
        np.asarray(resolved.cells_per_sigma), rms / (sigma / 4.0), rtol=1e-9
    )
    # Deposit and gather each smooth over one cell: at one cell per σ the field
    # would be ≈ 45 % low, so the kick is refused and the bunch left unchanged.
    coarse = plan(1.0).kick(bunch, 0.1)
    np.testing.assert_allclose(np.asarray(coarse.cells_per_sigma), rms / sigma, rtol=1e-9)
    assert not bool(coarse.resolved)
    assert not bool(coarse.accepted)
    np.testing.assert_array_equal(
        np.asarray(coarse.bunch.coordinates), np.asarray(bunch.coordinates)
    )


def test_kick_depends_on_the_declared_frame_not_the_reference_momentum() -> None:
    # The same particles referenced to a 1.5× larger momentum, kicked in the
    # frame of their own motion, receive the same absolute momentum increments.
    accelerator = phx.applications.accelerator
    scale = phx.ElectromagneticScaleContract.si()
    rest_energy = float(scale.electron_mass) * float(scale.speed_of_light) ** 2
    momentum = 5.0e6 * float(scale.elementary_charge)
    gamma = float(np.sqrt(1.0 + (momentum / rest_energy) ** 2))
    sigma = 1.0e-3
    bunch, positions = _gaussian_bunch(sigma, gamma, sigma / 4.0, momentum)
    rereferenced = accelerator.AcceleratorBunch(
        bunch.coordinates.at[:, 5].set(1.0 / 1.5 - 1.0),
        bunch.weights,
        bunch.particle_ids,
        reference_rest_energy=rest_energy,
        reference_momentum=1.5 * momentum,
        reference_charge=-1.0,
        bunch_id="rereferenced",
    )
    grid = TensorGridPlan((UniformCellAxisSpec(36),) * 3).prepare(
        np.asarray([[-4.5 * sigma] * 3, [4.5 * sigma] * 3])
    )
    plan = accelerator.SpaceChargeIGFPlan(scale, grid, capacity=positions.shape[0])
    own = plan.kick(bunch, 0.1)
    framed = plan.kick(rereferenced, 0.1, frame_lorentz_factor=gamma)
    assert bool(own.accepted) and bool(framed.accepted)
    np.testing.assert_allclose(float(framed.reference_gamma), gamma, rtol=1e-15)
    increment = np.asarray(own.momentum_kick) * momentum
    np.testing.assert_allclose(
        np.asarray(framed.momentum_kick) * 1.5 * momentum,
        increment,
        rtol=1e-9,
        atol=1e-9 * np.max(np.abs(increment)),
    )
    # A frame at γ ≤ 1 is refused.
    assert not bool(plan.kick(bunch, 0.1, frame_lorentz_factor=1.0).accepted)
