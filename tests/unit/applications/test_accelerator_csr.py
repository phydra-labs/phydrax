#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coherent synchrotron radiation contracts.

Independent references: the Saldin–Derbenev steady wake and the Saldin,
Schneidmiller, Yurkov (NIM A 398, 373, 1997) entrance-transient formula
evaluated with SciPy quadrature; the closed-form mean loss of a Gaussian bunch
``2^{5/3}Γ(5/6)/(4·3^{1/3}√π) · Q q/(4πε₀) R^{-2/3} σ^{-4/3}``; the
parallel-plate shielded steady impedance (Murphy, Krinsky, Gluckstern 1997, in
the Airy form of Agoh and Yokoya) evaluated with SciPy Airy functions; the
residual centripetal coefficient ``Λ = 2`` of the steady effective transverse
force (Derbenev and Shiltsev 1996; Stupakov, PRAB 25, 014401, 2022), checked
on both 3-D routes; and the first-order chicane ``R₅₆``. External Ocelot and PyCSR3D
oracles skip only when their pinned interpreter is not configured.
"""

from __future__ import annotations

import math
import os
import shutil
from typing import TypedDict

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import airye

import phydrax as phx
from phydrax._external_runtime import pin_executable, PinnedExecutable
from phydrax.applications import accelerator
from phydrax.discretization import PreparedTensorGrid, TensorGridPlan, UniformCellAxisSpec


SCALE = phx.ElectromagneticScaleContract.si()
ELEMENTARY_CHARGE = float(SCALE.elementary_charge)
REST_ENERGY = float(SCALE.electron_mass) * float(SCALE.speed_of_light) ** 2
COULOMB = -ELEMENTARY_CHARGE / (4.0 * math.pi * float(SCALE.vacuum_permittivity))
CHARGE = 1.0e-9


class _PlanOptions(TypedDict, total=False):
    reference_rest_energy: float
    reference_momentum: float
    capacity: int
    smoothing: float
    far_nodes: int


# Mean steady CSR loss of a Gaussian bunch in units of Q q/(4πε₀) R^{-2/3} σ^{-4/3}.
GAUSSIAN_LOSS = (
    -(2.0 ** (5.0 / 3.0))
    * math.gamma(5.0 / 6.0)
    / (4.0 * 3.0 ** (1.0 / 3.0) * math.sqrt(math.pi))
)


def _momentum(gamma: float) -> float:
    return REST_ENERGY * math.sqrt(gamma * gamma - 1.0)


def _grid(
    lower: tuple[float, ...], upper: tuple[float, ...], counts: tuple[int, ...]
) -> PreparedTensorGrid:
    return TensorGridPlan(tuple(UniformCellAxisSpec(count) for count in counts)).prepare(
        np.asarray([lower, upper])
    )


def _gaussian(z: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (z / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def _gaussian_value(z: float, sigma: float) -> float:
    return math.exp(-0.5 * (z / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def _saldin_steady(z: float, sigma: float, radius: float) -> float:
    def integrand(lag: float) -> float:
        source = z - lag
        return lag ** (-1.0 / 3.0) * (-source / sigma**2) * _gaussian_value(source, sigma)

    near = min(1.0e-3 * sigma, 1.0e-9)
    value = quad(integrand, 0.0, near, limit=200)[0]
    value += quad(integrand, near, 20.0 * sigma, limit=400)[0]
    return -2.0 / (3.0 ** (1.0 / 3.0) * radius ** (2.0 / 3.0)) * value


def _saldin_entrance(z: float, angle: float, sigma: float, radius: float) -> float:
    reach = radius * angle**3 / 24.0

    def integrand(lag: float) -> float:
        source = z - lag
        return lag ** (-1.0 / 3.0) * (-source / sigma**2) * _gaussian_value(source, sigma)

    near = min(1.0e-7, reach)
    value = quad(integrand, 0.0, near, limit=200)[0]
    if reach > near:
        value += quad(integrand, near, reach, limit=400)[0]
    boundary = (
        _gaussian_value(z - reach, sigma) - _gaussian_value(z - 4.0 * reach, sigma)
    ) / (reach ** (1.0 / 3.0))
    return -2.0 / (3.0 ** (1.0 / 3.0) * radius ** (2.0 / 3.0)) * (boundary + value)


def _line_setup(
    sigma: float, count: int, half: float = 6.0
) -> tuple[PreparedTensorGrid, np.ndarray, np.ndarray]:
    grid = _grid((-half * sigma,), (half * sigma,), (count,))
    z = np.asarray(grid.points).reshape(-1)
    return grid, z, CHARGE * _gaussian(z, sigma)


def test_steady_wake_and_gaussian_energy_loss_match_saldin_derbenev() -> None:
    radius, sigma = 10.0, 3.0e-4
    grid, z, density = _line_setup(sigma, 128)
    lattice = accelerator.CSRLattice([5.0], [1.0 / radius], element_ids=["bend"])
    plan = accelerator.CSRPlan(
        "1d-steady",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(1.0e4),
        capacity=8,
    )
    wake = plan.wake(density, position=4.0, reference_charge=-1.0)
    reference = (
        COULOMB * CHARGE * np.asarray([_saldin_steady(v, sigma, radius) for v in z])
    )
    np.testing.assert_allclose(
        wake.wake, reference, atol=6.0e-3 * np.max(np.abs(reference))
    )
    spacing = z[1] - z[0]
    mean_loss = float(np.sum(np.asarray(wake.wake) * density) * spacing / CHARGE)
    closed_form = (
        GAUSSIAN_LOSS * COULOMB * CHARGE / (radius ** (2.0 / 3.0) * sigma ** (4.0 / 3.0))
    )
    assert mean_loss == pytest.approx(closed_form, rel=5.0e-3)
    assert int(wake.evidence.status) == 0
    assert float(wake.evidence.overtaking_length) == pytest.approx(
        (24.0 * sigma * radius**2) ** (1.0 / 3.0), rel=2.0e-2
    )


def test_steady_wake_flags_bends_shorter_than_overtaking_length() -> None:
    radius, sigma = 10.0, 3.0e-4
    grid, _, density = _line_setup(sigma, 64)
    lattice = accelerator.CSRLattice(
        [0.5, 1.0], [0.0, 1.0 / radius], element_ids=["d", "b"]
    )
    plan = accelerator.CSRPlan(
        "1d-steady",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(1.0e4),
        capacity=8,
    )
    early = plan.wake(density, position=0.6, reference_charge=-1.0)
    drift = plan.wake(density, position=0.2, reference_charge=-1.0)
    assert int(early.evidence.status) & int(accelerator.CSRStatus.NOT_STEADY)
    assert bool(early.evidence.accepted)
    assert float(early.evidence.bend_entry_distance) == pytest.approx(0.1)
    np.testing.assert_array_equal(np.asarray(drift.wake), 0.0)


@pytest.mark.parametrize("angle_fraction", [0.5, 1.0, 2.0], ids=["half", "one", "two"])
def test_transient_entrance_matches_saldin_1997(angle_fraction: float) -> None:
    radius, sigma = 10.0, 3.0e-4
    grid, z, density = _line_setup(sigma, 128)
    lattice = accelerator.CSRLattice([5.0], [1.0 / radius], element_ids=["bend"])
    plan = accelerator.CSRPlan(
        "1d-transient-shielded",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(1.0e4),
        capacity=8,
    )
    angle = angle_fraction * (24.0 * sigma / radius) ** (1.0 / 3.0)
    position = radius * angle
    wake = plan.wake(
        density,
        position=position,
        reference_charge=-1.0,
        derivative_step=1.0e-3 * position,
    )
    reference = (
        COULOMB
        * CHARGE
        * np.asarray([_saldin_entrance(v, angle, sigma, radius) for v in z])
    )
    np.testing.assert_allclose(
        wake.wake, reference, atol=4.0e-2 * np.max(np.abs(reference))
    )
    assert int(wake.evidence.status) == 0
    assert int(wake.evidence.retarded_failures) == 0


def _airy_shielding_ratio(radius: float, gap: float, sigma: float) -> float:
    def form(x: np.ndarray) -> np.ndarray:
        ai, aip, bi, bip = airye(x)
        decay = np.exp(-(4.0 / 3.0) * x**1.5)
        return decay * (aip**2 + x * ai**2) - 1j * (aip * bip + x * ai * bi)

    continuum = complex(
        quad(lambda b: float(form(np.asarray(b * b)).real), 0.0, 40.0, limit=400)[0],
        quad(lambda b: float(form(np.asarray(b * b)).imag), 0.0, 40.0, limit=400)[0],
    )
    wavenumbers = np.geomspace(1.0, 30.0 / sigma, 800)
    order = np.arange(20000)
    shielded = []
    for k in wavenumbers:
        arguments = (
            (2 * order + 1) * (math.pi / gap) * (radius / (2.0 * k * k)) ** (1.0 / 3.0)
        ) ** 2
        arguments = arguments[arguments < 1.0e4]
        total = (
            (2.0 * math.pi / gap)
            * (2.0 / (k * radius)) ** (1.0 / 3.0)
            * np.sum(form(arguments))
        )
        shielded.append(total / ((4.0 * k / radius**2) ** (1.0 / 3.0) * continuum))
    free = wavenumbers ** (1.0 / 3.0) * (math.sqrt(3.0) + 1j)
    weight = np.exp(-((wavenumbers * sigma) ** 2))
    return float(
        np.trapezoid((np.asarray(shielded) * free).real * weight, wavenumbers)
        / np.trapezoid(free.real * weight, wavenumbers)
    )


def test_parallel_plate_shielding_matches_murphy_krinsky_gluckstern() -> None:
    radius, sigma, gap = 10.0, 3.0e-4, 0.02
    grid, z, density = _line_setup(sigma, 64)
    lattice = accelerator.CSRLattice([10.0], [1.0 / radius], element_ids=["bend"])
    common = _PlanOptions(
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(1957.0),
        capacity=8,
    )
    shielded = accelerator.CSRPlan(
        "1d-transient-shielded",
        lattice,
        SCALE,
        grid,
        plate_gap=gap,
        image_count=24,
        **common,
    )
    free = accelerator.CSRPlan("1d-transient-shielded", lattice, SCALE, grid, **common)
    shielded_wake = shielded.wake(
        density, position=9.0, reference_charge=-1.0, derivative_step=1.0e-3
    )
    free_wake = free.wake(
        density, position=9.0, reference_charge=-1.0, derivative_step=1.0e-3
    )
    ratio = float(np.sum(np.asarray(shielded_wake.wake) * density)) / float(
        np.sum(np.asarray(free_wake.wake) * density)
    )
    assert ratio == pytest.approx(_airy_shielding_ratio(radius, gap, sigma), rel=2.0e-2)
    truncation = float(shielded_wake.evidence.shielding_truncation)
    assert 0.0 < truncation <= 5.0e-3
    few = accelerator.CSRPlan(
        "1d-transient-shielded",
        lattice,
        SCALE,
        grid,
        plate_gap=gap,
        image_count=2,
        shielding_tolerance=1.0e-3,
        **common,
    )
    truncated = few.wake(
        density, position=9.0, reference_charge=-1.0, derivative_step=1.0e-3
    )
    assert int(truncated.evidence.status) & int(accelerator.CSRStatus.SHIELDING_TRUNCATED)
    assert not bool(truncated.evidence.accepted)


def _bunch_density(
    grid: PreparedTensorGrid,
    counts: tuple[int, int, int],
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
) -> np.ndarray:
    points = np.asarray(grid.points).reshape(counts + (3,))
    return (
        CHARGE
        * np.exp(
            -0.5 * (points[..., 0] / sigma_x) ** 2
            - 0.5 * (points[..., 1] / sigma_y) ** 2
            - 0.5 * (points[..., 2] / sigma_z) ** 2
        )
        / ((2.0 * math.pi) ** 1.5 * sigma_x * sigma_y * sigma_z)
    )


def _centripetal_coefficient(
    force: np.ndarray, z: np.ndarray, sigma_z: float, radius: float
) -> np.ndarray:
    """``Λ`` of ``F_x = −Λ qλ(z)/(4πε₀ρ)`` on the central cells, ``|z| ≤ σ_z/2``."""
    line = COULOMB * CHARGE * _gaussian(z, sigma_z) / radius
    core = np.abs(z) <= 0.5 * sigma_z
    return -force[core] / line[core]


def test_steady_igf_reduces_to_1d_on_a_thin_round_beam() -> None:
    radius, sigma_z, sigma_x = 1.0, 1.0e-5, 2.0e-6
    counts = (8, 8, 48)
    grid = _grid(
        (-4 * sigma_x, -4 * sigma_x, -5 * sigma_z),
        (4 * sigma_x, 4 * sigma_x, 5 * sigma_z),
        counts,
    )
    lattice = accelerator.CSRLattice([1.0], [1.0 / radius], element_ids=["bend"])
    plan = accelerator.CSRPlan(
        "3d-steady-igf",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(500.0),
        capacity=8,
        kernel_quadrature=2,
    )
    density = _bunch_density(grid, counts, sigma_x, sigma_x, sigma_z)
    wake = plan.wake(density, position=0.9, reference_charge=-1.0)
    z = np.asarray(grid.points).reshape(counts + (3,))[0, 0, :, 2]
    longitudinal = np.asarray(wake.wake)[3:5, 3:5].mean(axis=(0, 1))
    reference = (
        COULOMB * CHARGE * np.asarray([_saldin_steady(v, sigma_z, radius) for v in z])
    )
    np.testing.assert_allclose(
        longitudinal, reference, atol=5.0e-2 * np.max(np.abs(reference))
    )
    assert int(wake.evidence.status) == 0
    assert plan.kernel_defect < 5.0e-2


@pytest.mark.parametrize(
    ("sigma_x", "sigma_y"),
    [(2.0e-6, 2.0e-6), (4.0e-6, 1.0e-6), (1.0e-6, 4.0e-6)],
    ids=["round", "flat-horizontal", "flat-vertical"],
)
def test_steady_igf_residual_centripetal_force_is_two_for_any_aspect_ratio(
    sigma_x: float, sigma_y: float
) -> None:
    # Derbenev and Shiltsev (1996), Stupakov (PRAB 25, 014401, 2022): the
    # effective steady force is −2qλ(z)/(4πε₀ρ). Cai and Ding's paraxial
    # potentials give 2 + 2⟨x²/r²⟩ instead (3 for a round beam).
    radius, sigma_z = 1.0, 1.0e-5
    counts = (8, 8, 48)
    grid = _grid(
        (-4 * sigma_x, -4 * sigma_y, -5 * sigma_z),
        (4 * sigma_x, 4 * sigma_y, 5 * sigma_z),
        counts,
    )
    lattice = accelerator.CSRLattice([1.0], [1.0 / radius], element_ids=["bend"])
    plan = accelerator.CSRPlan(
        "3d-steady-igf",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(500.0),
        capacity=8,
        kernel_quadrature=2,
    )
    density = _bunch_density(grid, counts, sigma_x, sigma_y, sigma_z)
    wake = plan.wake(density, position=0.9, reference_charge=-1.0)
    z = np.asarray(grid.points).reshape(counts + (3,))[0, 0, :, 2]
    horizontal = np.asarray(wake.horizontal_force)[3:5, 3:5].mean(axis=(0, 1))
    np.testing.assert_allclose(
        _centripetal_coefficient(horizontal, z, sigma_z, radius), 2.0, rtol=1.0e-2
    )
    assert int(wake.evidence.status) == 0


def test_retarded_mesh_and_steady_igf_agree_on_longitudinal_and_horizontal_forces() -> (
    None
):
    radius, sigma_z, sigma_x = 1.0, 1.0e-5, 2.0e-6
    lattice = accelerator.CSRLattice([1.0], [1.0 / radius], element_ids=["bend"])
    common = _PlanOptions(
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(500.0),
        capacity=8,
        far_nodes=48,
    )
    longitudinal_errors = []
    horizontal_errors = []
    for transverse in (4, 6):
        counts = (transverse, transverse, 32)
        grid = _grid(
            (-4 * sigma_x, -4 * sigma_x, -5 * sigma_z),
            (4 * sigma_x, 4 * sigma_x, 5 * sigma_z),
            counts,
        )
        density = _bunch_density(grid, counts, sigma_x, sigma_x, sigma_z)
        mesh = accelerator.CSRPlan("3d-retarded-mesh", lattice, SCALE, grid, **common)
        igf = accelerator.CSRPlan(
            "3d-steady-igf", lattice, SCALE, grid, kernel_quadrature=2, **common
        )
        mesh_wake = mesh.wake(
            density, position=0.9, reference_charge=-1.0, derivative_step=1.0e-3
        )
        igf_wake = igf.wake(density, position=0.9, reference_charge=-1.0)
        middle = slice(transverse // 2 - 1, transverse // 2 + 1)
        for errors, mesh_field, igf_field in (
            (longitudinal_errors, mesh_wake.wake, igf_wake.wake),
            (horizontal_errors, mesh_wake.horizontal_force, igf_wake.horizontal_force),
        ):
            a = np.asarray(mesh_field)[middle, middle].mean(axis=(0, 1))
            b = np.asarray(igf_field)[middle, middle].mean(axis=(0, 1))
            errors.append(np.max(np.abs(a - b)) / np.max(np.abs(b)))
        assert int(mesh_wake.evidence.status) == 0
    # Both routes converge to the same residual centripetal force, Λ = 2.
    z = np.asarray(grid.points).reshape(counts + (3,))[0, 0, :, 2]
    horizontal = np.asarray(mesh_wake.horizontal_force)[2:4, 2:4].mean(axis=(0, 1))
    np.testing.assert_allclose(
        _centripetal_coefficient(horizontal, z, sigma_z, radius), 2.0, rtol=3.0e-2
    )
    assert max(longitudinal_errors) < 6.0e-2
    assert max(horizontal_errors) < 3.0e-2


def _bunch(
    coordinates: np.ndarray, charge: float, gamma: float, identifier: str
) -> accelerator.AcceleratorBunch:
    count = coordinates.shape[0]
    return accelerator.AcceleratorBunch(
        coordinates,
        np.full((count,), charge / ELEMENTARY_CHARGE / count),
        np.arange(count),
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(gamma),
        reference_charge=-1.0,
        bunch_id=identifier,
    )


def _gaussian_coordinates(
    count: int, sigma_z: float, seed: int, sigma_x: float = 1.0e-4
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coordinates = np.zeros((count, 6))
    coordinates[:, 0] = sigma_x * rng.standard_normal(count)
    coordinates[:, 1] = 1.0e-5 * rng.standard_normal(count)
    coordinates[:, 2] = sigma_x * rng.standard_normal(count)
    coordinates[:, 3] = 1.0e-5 * rng.standard_normal(count)
    coordinates[:, 4] = sigma_z * rng.standard_normal(count)
    return coordinates


def test_retarded_mesh_kick_converges_in_particle_count_and_kernel_width() -> None:
    radius, sigma_z, sigma_x = 1.0, 1.0e-5, 2.0e-6
    counts = (4, 4, 32)
    grid = _grid(
        (-4 * sigma_x, -4 * sigma_x, -5 * sigma_z),
        (4 * sigma_x, 4 * sigma_x, 5 * sigma_z),
        counts,
    )
    lattice = accelerator.CSRLattice([1.0], [1.0 / radius], element_ids=["bend"])
    reference_plan = accelerator.CSRPlan(
        "3d-retarded-mesh",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(500.0),
        capacity=8,
        far_nodes=48,
    )
    reference = np.asarray(
        reference_plan.wake(
            -_bunch_density(grid, counts, sigma_x, sigma_x, sigma_z),
            position=0.9,
            reference_charge=-1.0,
            derivative_step=1.0e-3,
        ).radiative_wake
    )
    # The reference density carries the bunch's (electron) charge sign.
    errors = []
    for count, width in ((2000, 0.5), (32000, 0.25)):
        plan = accelerator.CSRPlan(
            "3d-retarded-mesh",
            lattice,
            SCALE,
            grid,
            reference_rest_energy=REST_ENERGY,
            reference_momentum=_momentum(500.0),
            capacity=count,
            far_nodes=48,
            smoothing=(0.0, 0.0, width * sigma_z),
        )
        coordinates = _gaussian_coordinates(count, sigma_z, 7, sigma_x)
        # Keep every macroparticle within the outermost cell centers.
        coordinates[:, (0, 2)] = np.clip(
            coordinates[:, (0, 2)], -2.9 * sigma_x, 2.9 * sigma_x
        )
        coordinates[:, 4] = np.clip(coordinates[:, 4], -4.5 * sigma_z, 4.5 * sigma_z)
        bunch = _bunch(coordinates, CHARGE, 500.0, "mesh")
        state = plan.initial_state(bunch, position=0.9)
        kick = plan.kick(bunch, state, position=0.9, length=1.0e-3)
        assert bool(kick.evidence.accepted)
        sampled = np.asarray(kick.wake)
        errors.append(np.max(np.abs(sampled - reference)) / np.max(np.abs(reference)))
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 0.15


def _chicane(theta: float, bend: float, drift: float) -> accelerator.CSRLattice:
    curvature = theta / bend
    return accelerator.CSRLattice(
        [bend, drift, bend, 0.5, bend, drift, bend, 0.5],
        [curvature, 0.0, -curvature, 0.0, -curvature, 0.0, curvature, 0.0],
        element_ids=["b1", "d1", "b2", "d2", "b3", "d3", "b4", "d4"],
        entrance_edges=[0.0, 0.0, -theta, 0.0, 0.0, 0.0, theta, 0.0],
        exit_edges=[theta, 0.0, 0.0, 0.0, -theta, 0.0, 0.0, 0.0],
    )


def test_zero_charge_chicane_reproduces_first_order_optics() -> None:
    theta, bend, drift, gamma = 0.02, 0.5, 2.0, 1000.0
    lattice = _chicane(theta, bend, drift)
    grid = _grid((-8.0e-4,), (8.0e-4,), (32,))
    plan = accelerator.CSRPlan(
        "1d-steady",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(gamma),
        capacity=64,
    )
    rng = np.random.default_rng(3)
    coordinates = 1.0e-5 * rng.standard_normal((64, 6))
    bunch = _bunch(coordinates, 0.0, gamma, "probe")
    result = accelerator.track_csr(accelerator.CSRTrackingPlan(plan, substeps=2), bunch)
    matrix = np.linalg.lstsq(
        coordinates, np.asarray(result.bunch.coordinates), rcond=None
    )[0].T
    total = 4 * bend + 2 * drift + 1.0
    r56 = -2.0 * theta**2 * (2.0 * bend / 3.0 + drift) - total / gamma**2
    assert matrix[4, 5] == pytest.approx(r56, rel=1.0e-3)
    np.testing.assert_allclose(matrix[0:2, 5], 0.0, atol=1.0e-12)
    assert bool(result.accepted)
    np.testing.assert_allclose(np.asarray(result.energy_change), 0.0, atol=0.0)


def test_tracked_steady_energy_loss_integrates_the_wake_along_a_bend() -> None:
    radius, sigma, gamma, length = 10.0, 1.0e-4, 2000.0, 2.0
    lattice = accelerator.CSRLattice([length], [1.0 / radius], element_ids=["bend"])
    grid = _grid((-6 * sigma,), (6 * sigma,), (64,))
    count = 20000
    plan = accelerator.CSRPlan(
        "1d-steady",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(gamma),
        capacity=count,
        smoothing=sigma / 8,
    )
    coordinates = _gaussian_coordinates(count, sigma, 11, sigma_x=1.0e-6)
    coordinates[:, 1] = 0.0
    bunch = _bunch(coordinates, CHARGE, gamma, "loss")
    result = accelerator.track_csr(accelerator.CSRTrackingPlan(plan, substeps=10), bunch)
    per_electron = float(np.sum(result.energy_change)) / (CHARGE / ELEMENTARY_CHARGE)
    closed_form = (
        GAUSSIAN_LOSS * COULOMB * CHARGE / (radius ** (2.0 / 3.0) * sigma ** (4.0 / 3.0))
    )
    assert per_electron == pytest.approx(closed_form * length, rel=5.0e-2)
    assert bool(result.accepted)


def test_transient_tracking_records_history_and_refuses_dropped_history() -> None:
    theta, bend, drift, gamma, sigma = 0.05, 0.5, 2.0, 1000.0, 5.0e-5
    lattice = _chicane(theta, bend, drift)
    grid = _grid((-8 * sigma,), (8 * sigma,), (32,))
    count = 2000
    bunch = _bunch(
        _gaussian_coordinates(count, sigma, 5, 1.0e-5), CHARGE, gamma, "chicane"
    )
    common = _PlanOptions(
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(gamma),
        capacity=count,
        smoothing=sigma / 6,
        far_nodes=32,
    )
    substeps = [4, 2, 4, 1, 4, 2, 4, 1]
    complete = accelerator.CSRPlan(
        "1d-transient-shielded", lattice, SCALE, grid, history_capacity=32, **common
    )
    result = accelerator.track_csr(
        accelerator.CSRTrackingPlan(complete, substeps=substeps), bunch
    )
    assert bool(result.accepted)
    assert int(result.state.count) == sum(substeps)
    assert float(np.sum(result.energy_change)) < 0.0
    short = accelerator.CSRPlan(
        "1d-transient-shielded", lattice, SCALE, grid, history_capacity=2, **common
    )
    dropped = accelerator.track_csr(
        accelerator.CSRTrackingPlan(short, substeps=substeps), bunch
    )
    status = np.asarray(dropped.status)
    assert np.any(status & int(accelerator.CSRStatus.HISTORY_INCOMPLETE))
    assert not bool(dropped.accepted)


def test_kick_refuses_particles_outside_the_grid_and_mismatched_bunches() -> None:
    sigma, gamma = 1.0e-4, 1000.0
    lattice = accelerator.CSRLattice([1.0], [0.1], element_ids=["bend"])
    grid = _grid((-3 * sigma,), (3 * sigma,), (16,))
    plan = accelerator.CSRPlan(
        "1d-steady",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(gamma),
        capacity=100,
    )
    coordinates = _gaussian_coordinates(100, sigma, 2)
    coordinates[0, 4] = 10 * sigma
    bunch = _bunch(coordinates, CHARGE, gamma, "outside")
    state = plan.initial_state(bunch, position=0.5)
    kick = plan.kick(bunch, state, position=0.5, length=0.1)
    assert int(kick.evidence.status) & int(accelerator.CSRStatus.UNSUPPORTED)
    assert not bool(kick.evidence.accepted)
    np.testing.assert_array_equal(np.asarray(kick.bunch.coordinates), coordinates)
    with pytest.raises(ValueError, match="reference energy"):
        plan.kick(
            _bunch(coordinates, CHARGE, 2 * gamma, "other"),
            state,
            position=0.5,
            length=0.1,
        )
    with pytest.raises(ValueError, match="capacity"):
        plan.kick(
            _bunch(coordinates[:50], CHARGE, gamma, "small"),
            state,
            position=0.5,
            length=0.1,
        )


def test_plan_refuses_invalid_selectors_shielding_and_resources() -> None:
    lattice = accelerator.CSRLattice([1.0], [0.1], element_ids=["bend"])
    line = _grid((-1.0e-3,), (1.0e-3,), (16,))
    common = _PlanOptions(
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(1000.0),
        capacity=8,
    )
    with pytest.raises(ValueError):
        accelerator.CSRPlan("2d-steady", lattice, SCALE, line, **common)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Parallel-plate"):
        accelerator.CSRPlan(
            "1d-steady", lattice, SCALE, line, plate_gap=0.01, image_count=4, **common
        )
    with pytest.raises(ValueError, match="3-dimensional|three"):
        accelerator.CSRPlan("3d-steady-igf", lattice, SCALE, line, **common)
    with pytest.raises(accelerator.CSRResourceError):
        accelerator.CSRPlan(
            "1d-transient-shielded",
            lattice,
            SCALE,
            line,
            resources=accelerator.CSRResources(maximum_retarded_pairs=100),
            **common,
        )
    with pytest.raises(ValueError, match="kernel_quadrature"):
        accelerator.CSRPlan(
            "1d-steady", lattice, SCALE, line, kernel_quadrature=3, **common
        )
    with pytest.raises(ValueError):
        accelerator.CSRLattice([1.0, -1.0], [0.0, 0.1], element_ids=["a", "b"])
    with pytest.raises(ValueError, match="pole-face"):
        accelerator.CSRLattice([1.0], [0.0], element_ids=["d"], entrance_edges=[0.1])
    steady = accelerator.CSRPlan("1d-steady", lattice, SCALE, line, **common)
    with pytest.raises(ValueError, match="derivative_step"):
        steady.wake(
            np.zeros(16), position=0.5, reference_charge=-1.0, derivative_step=1.0e-3
        )


def _pinned_python(variable: str) -> PinnedExecutable | None:
    path = os.environ.get(variable)
    version = os.environ.get(variable + "_VERSION")
    if path is None or version is None or shutil.which(path) is None:
        return None
    return pin_executable(
        shutil.which(path) or path, version=version, license_id="provider"
    )


def test_chicane_emittance_matches_pinned_ocelot() -> None:
    executable = _pinned_python("PHYDRAX_OCELOT_PYTHON")
    if executable is None:
        pytest.skip(
            "set PHYDRAX_OCELOT_PYTHON and PHYDRAX_OCELOT_PYTHON_VERSION to a pinned Ocelot interpreter"
        )
    theta, bend, drift, gamma, sigma = 0.05, 0.5, 2.0, 1000.0, 5.0e-5
    lattice = _chicane(theta, bend, drift)
    count = 4000
    rng = np.random.default_rng(1)
    coordinates = _gaussian_coordinates(count, sigma, 1)
    coordinates[:, 0] = math.sqrt(1.0e-9 * 10.0) * rng.standard_normal(count)
    coordinates[:, 1] = math.sqrt(1.0e-9 / 10.0) * rng.standard_normal(count)
    bunch = _bunch(coordinates, CHARGE, gamma, "ocelot")
    grid = _grid((-8 * sigma,), (8 * sigma,), (64,))
    plan = accelerator.CSRPlan(
        "1d-transient-shielded",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(gamma),
        capacity=count,
        smoothing=sigma / 8,
        history_capacity=128,
        far_nodes=48,
    )
    tracked = accelerator.track_csr(
        accelerator.CSRTrackingPlan(plan, substeps=[10, 4, 10, 2, 10, 4, 10, 2]), bunch
    )
    external = accelerator.ocelot_csr_tracking(
        accelerator.OcelotCSRProvider(executable),
        lattice,
        bunch,
        electron_charge=ELEMENTARY_CHARGE,
        trajectory_step=5.0e-4,
        apply_step=5.0e-3,
        bins=300,
        sigma_min=sigma / 20,
    )

    def emittance(values: np.ndarray) -> float:
        return float(math.sqrt(max(np.linalg.det(np.cov(values[:, 0:2].T)), 0.0)))

    initial = emittance(coordinates)
    ours = emittance(np.asarray(tracked.bunch.coordinates)) - initial
    theirs = emittance(external) - initial
    assert ours == pytest.approx(theirs, rel=0.2)


def test_steady_igf_matches_pinned_pycsr3d() -> None:
    executable = _pinned_python("PHYDRAX_PYCSR3D_PYTHON")
    if executable is None:
        pytest.skip(
            "set PHYDRAX_PYCSR3D_PYTHON and PHYDRAX_PYCSR3D_PYTHON_VERSION to a pinned PyCSR3D interpreter"
        )
    radius, sigma_z, sigma_x = 1.0, 1.0e-5, 2.0e-6
    counts = (8, 8, 48)
    grid = _grid(
        (-4 * sigma_x, -4 * sigma_x, -5 * sigma_z),
        (4 * sigma_x, 4 * sigma_x, 5 * sigma_z),
        counts,
    )
    lattice = accelerator.CSRLattice([1.0], [1.0 / radius], element_ids=["bend"])
    plan = accelerator.CSRPlan(
        "3d-steady-igf",
        lattice,
        SCALE,
        grid,
        reference_rest_energy=REST_ENERGY,
        reference_momentum=_momentum(500.0),
        capacity=8,
        kernel_quadrature=2,
    )
    density = _bunch_density(grid, counts, sigma_x, sigma_x, sigma_z)
    ours = (
        np.asarray(plan.wake(density, position=0.9, reference_charge=-1.0).wake) / COULOMB
    )
    theirs = accelerator.pycsr3d_longitudinal_wake(
        accelerator.PyCSR3DProvider(executable), plan, density, curvature=1.0 / radius
    )
    center = (slice(3, 5), slice(3, 5))
    np.testing.assert_allclose(
        ours[center], theirs[center], atol=5.0e-2 * np.max(np.abs(theirs[center]))
    )
