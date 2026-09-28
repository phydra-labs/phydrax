#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.applications.radiation_transport import (
    magnetobremsstrahlung_path_coefficients,
    PlasmaRayTransferPlan,
    PlasmaRayTransferStatus,
)
from phydrax.electromagnetics import (
    ColdPlasmaDielectric,
    MagnetobremsstrahlungPlan,
    PlasmaWaveMode,
    ThermalJuttnerDistribution,
)
from phydrax.optics.geometric import (
    ColdPlasmaHamiltonian,
    ColdPlasmaProfile,
    ColdPlasmaRayPath,
    DispersionRayPlan,
)


SCALE = phx.ElectromagneticScaleContract.si()
E = float(SCALE.elementary_charge)
M_E = float(SCALE.electron_mass)
EPS0 = float(SCALE.vacuum_permittivity)
C = float(SCALE.speed_of_light)
COORDINATES = phx.SpatialCoordinateContract(
    phx.units.METER, coordinate_system="cartesian", reference_frame="world"
)
OMEGA = 2.0 * np.pi * 1.0e10
K0 = OMEGA / C


def _density(plasma_parameter: float) -> float:
    return plasma_parameter * EPS0 * M_E * OMEGA**2 / E**2


def _field(gyro_parameter: float) -> float:
    return gyro_parameter * M_E * OMEGA / E


def _appleton_hartree(x: float, y: float, theta: float) -> np.ndarray:
    """Independent electron Appleton–Hartree ``n²`` of both roots."""
    sine2 = np.sin(theta) ** 2
    root = np.sqrt(y**4 * sine2**2 + 4.0 * (1.0 - x) ** 2 * y**2 * np.cos(theta) ** 2)
    base = 2.0 * (1.0 - x) - y**2 * sine2
    return 1.0 - 2.0 * x * (1.0 - x) / np.asarray((base + root, base - root))


def _ray_index_squared(x: float, y: float, theta: float, index_squared: float) -> float:
    """Bekefi ``n_r²`` by central differences of the Appleton–Hartree branch."""
    slot = int(np.argmin(np.abs(_appleton_hartree(x, y, theta) - index_squared)))

    def index(angle: float) -> float:
        return float(np.sqrt(_appleton_hartree(x, y, angle)[slot]))

    def group_angle(angle: float) -> float:
        h = 1.0e-5
        derivative = (index(angle + h) - index(angle - h)) / (2.0 * h)
        return angle + np.arctan(-derivative / index(angle))

    h = 1.0e-4
    turn = (group_angle(theta + h) - group_angle(theta - h)) / (2.0 * h)
    alpha = group_angle(theta) - theta
    return (
        index(theta) ** 2
        * abs(np.sin(theta) / (np.sin(group_angle(theta)) * turn))
        / np.cos(alpha)
    )


def _profile(
    density: Callable[[Array], Array], field: Callable[[Array], Array], name: str
) -> ColdPlasmaProfile:
    return ColdPlasmaProfile(
        SCALE,
        COORDINATES,
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        density=lambda point: jnp.stack([density(point)]),
        magnetic_field=field,
        profile_id=name,
    )


def _trace(
    hamiltonian: ColdPlasmaHamiltonian,
    direction: tuple[float, float, float],
    step: float,
    count: int,
    reference: tuple[float, float, float],
) -> ColdPlasmaRayPath:
    rays = (
        DispersionRayPlan(hamiltonian, step, count, hamiltonian_tolerance=1.0e-6)
        .prepare()
        .integrate(np.zeros((1, 3)), np.asarray((direction,)))
    )
    assert bool(rays.evidence.successful)
    return hamiltonian.sample_path(rays, reference)


@pytest.mark.strict_jax
def test_faraday_rotation_along_the_ray_equals_the_rotation_measure_integral() -> None:
    field = _field(0.1)
    ramp = 0.5
    profile = _profile(
        lambda point: _density(0.05) * (1.0 + point[2] / ramp),
        lambda point: jnp.stack([0.0 * point[2], 0.0 * point[2], field + 0.0 * point[2]]),
        "faraday-ramp",
    )
    hamiltonian = ColdPlasmaHamiltonian(
        profile, angular_frequency=OMEGA, mode=PlasmaWaveMode.RIGHT
    )
    path = _trace(hamiltonian, (0.0, 0.0, 1.0), 0.004, 250, (1.0, 0.0, 0.0))
    transfer = PlasmaRayTransferPlan(
        path, coupling="strong", anisotropy_tolerance=0.1
    ).evaluate(
        np.zeros((1, 250, 2)), np.zeros((1, 250, 2)), np.asarray(((1.0, 1.0, 0.0, 0.0),))
    )
    assert bool(transfer.successful)
    end = float(np.sum(path.segment_lengths))
    z = np.linspace(0.0, end, 20001)
    plasma = 0.05 * (1.0 + z / ramp)
    right = np.sqrt(1.0 - plasma / (1.0 - 0.1))
    left = np.sqrt(1.0 - plasma / (1.0 + 0.1))
    integrand = 0.5 * K0 * (left - right)
    rotation = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z)))
    np.testing.assert_allclose(transfer.faraday_rotation[0], rotation, rtol=1.0e-5)
    stokes = np.asarray(transfer.stokes[0])
    np.testing.assert_allclose(
        stokes[1:] / stokes[0],
        (np.cos(2.0 * rotation), np.sin(2.0 * rotation), 0.0),
        atol=1.0e-5,
    )
    # High-frequency limit: ψ = e³/(2 ε₀ m² c ω²) ∫ n_e B dz within O(X/2, Y²).
    measure = E**3 / (2.0 * EPS0 * M_E**2 * C * OMEGA**2)
    electrons = _density(0.05) * (end + end**2 / (2.0 * ramp))
    np.testing.assert_allclose(rotation, measure * electrons * field, rtol=0.1)


@pytest.mark.strict_jax
def test_transfer_conserves_intensity_over_ray_index_squared() -> None:
    x0, y, tilt = 0.2, 0.4, 0.7
    profile = _profile(
        lambda point: _density(x0) * (1.0 + point[0]),
        lambda point: (
            _field(y)
            * jnp.stack(
                [
                    np.sin(tilt) + 0.0 * point[0],
                    0.0 * point[0],
                    np.cos(tilt) + 0.0 * point[0],
                ]
            )
        ),
        "oblique-ramp",
    )
    hamiltonian = ColdPlasmaHamiltonian(
        profile, angular_frequency=OMEGA, mode=PlasmaWaveMode.EXTRAORDINARY
    )
    path = _trace(hamiltonian, (0.6, 0.0, 0.8), 0.01, 60, (0.0, 1.0, 0.0))
    segments = path.segment_lengths.shape[1]
    for segment in (0, segments - 1):
        plasma = x0 * (1.0 + float(path.midpoints[0, segment, 0]))
        np.testing.assert_allclose(
            path.ray_index_squared[0, segment],
            _ray_index_squared(
                plasma,
                y,
                float(path.angle[0, segment]),
                float(path.refractive_index[0, segment, 0]) ** 2,
            ),
            rtol=1.0e-6,
        )
    zeros = np.zeros((1, segments, 2))
    result = PlasmaRayTransferPlan(path, coupling="weak").evaluate(
        zeros, zeros, np.asarray(((1.0, 0.0, 0.0, 0.0),))
    )
    assert bool(result.successful)
    # I/n_r² is conserved; the polarization follows the local X mode.
    np.testing.assert_allclose(
        result.invariant[:, 0], result.incident_invariant[:, 0], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        result.invariant[0, 1:] / result.invariant[0, 0],
        path.mode_stokes[0, -1, 0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.stokes[0, 0],
        0.5 * path.ray_index_squared[0, -1] / path.ray_index_squared[0, 0],
        rtol=1.0e-12,
    )
    assert not np.isclose(
        float(path.ray_index_squared[0, -1]), float(path.ray_index_squared[0, 0])
    )


def _twisted(
    coupling: float, length: float, count: int
) -> tuple[ColdPlasmaRayPath, float, float]:
    """O-mode ray along ẑ through a field rotating as ``(cos κz, sin κz, 0)``."""
    x, y = 0.3, 0.5
    ordinary = np.sqrt(1.0 - x)
    extraordinary = np.sqrt(1.0 - x * (1.0 - x) / (1.0 - x - y * y))
    beat = K0 * (extraordinary - ordinary)
    twist = 0.5 * coupling * abs(beat)
    field = _field(y)
    profile = _profile(
        lambda point: _density(x) + 0.0 * point[2],
        lambda point: (
            field
            * jnp.stack(
                [jnp.cos(twist * point[2]), jnp.sin(twist * point[2]), 0.0 * point[2]]
            )
        ),
        f"twisted-{coupling}",
    )
    hamiltonian = ColdPlasmaHamiltonian(
        profile, angular_frequency=OMEGA, mode=PlasmaWaveMode.ORDINARY
    )
    # Across B₀ the O mode has v_g/c = n, so τ = z / n_O.
    path = _trace(
        hamiltonian, (0.0, 0.0, 1.0), length / (ordinary * count), count, (1.0, 0.0, 0.0)
    )
    return path, beat, twist


def _rotating_frame_reference(
    incident: np.ndarray, beat: float, twist: float, length: float
) -> np.ndarray:
    """Exact Stokes solution of ``dS/dz = Δk ŝ_O(z) × S`` with ``ŝ_O = (cos 2κz, sin 2κz, 0)``.

    In the frame co-rotating with the field the rotation vector is constant,
    ``W = (Δk, 0, −2κ)``.
    """
    axis = np.asarray((beat, 0.0, -2.0 * twist))
    rate = np.linalg.norm(axis)
    unit = axis / rate
    angle = rate * length
    vector = incident[1:]
    rotated = (
        vector * np.cos(angle)
        + np.cross(unit, vector) * np.sin(angle)
        + unit * np.dot(unit, vector) * (1.0 - np.cos(angle))
    )
    frame = 2.0 * twist * length
    q, u, v = rotated
    return np.asarray(
        (
            incident[0],
            np.cos(frame) * q - np.sin(frame) * u,
            np.sin(frame) * q + np.cos(frame) * u,
            v,
        )
    )


@pytest.mark.parametrize(
    ("coupling", "length", "count"),
    ((0.02, 1.0, 120), (1.0, 0.3, 160), (5.0, 0.1, 220)),
    ids=("weak-coupling", "transition", "strong-coupling"),
)
@pytest.mark.strict_jax
def test_coupled_stokes_transfer_matches_the_twisted_field_solution(
    coupling: float, length: float, count: int
) -> None:
    path, beat, twist = _twisted(coupling, length, count)
    np.testing.assert_allclose(path.coupling_parameter, coupling, rtol=1.0e-3)
    assert bool(np.all(path.quasi_transverse))
    incident = np.asarray((1.0, 1.0, 0.0, 0.0))
    zeros = np.zeros((1, count, 2))
    result = PlasmaRayTransferPlan(
        path, coupling="strong", anisotropy_tolerance=0.2
    ).evaluate(zeros, zeros, incident[None])
    assert bool(result.successful)
    assert int(result.status[0]) & PlasmaRayTransferStatus.QUASI_TRANSVERSE
    np.testing.assert_allclose(
        result.stokes[0],
        _rotating_frame_reference(incident, beat, twist, length),
        atol=5.0e-3,
    )


@pytest.mark.strict_jax
def test_mode_coupling_limits_bracket_the_twisted_field_solution() -> None:
    incident = np.asarray((1.0, 1.0, 0.0, 0.0))
    # Weak coupling: the polarization follows the O mode, as independent modes predict.
    path, beat, twist = _twisted(0.02, 1.0, 120)
    zeros = np.zeros((1, 120, 2))
    weak = PlasmaRayTransferPlan(path, coupling="weak").evaluate(
        zeros, zeros, incident[None]
    )
    assert bool(weak.successful)
    reference = _rotating_frame_reference(incident, beat, twist, 1.0)
    np.testing.assert_allclose(weak.stokes[0], reference, atol=0.05)
    # Strong coupling: the polarization stays nearly fixed while the mode axis
    # turns; the independent-mode limit is refused.
    path, beat, twist = _twisted(5.0, 0.1, 220)
    zeros = np.zeros((1, 220, 2))
    weak = PlasmaRayTransferPlan(path, coupling="weak").evaluate(
        zeros, zeros, incident[None]
    )
    assert int(weak.status[0]) & PlasmaRayTransferStatus.WEAK_COUPLING_VIOLATED
    assert not bool(weak.successful)
    reference = _rotating_frame_reference(incident, beat, twist, 0.1)
    assert float(np.linalg.norm(np.asarray(weak.stokes[0]) - reference)) > 0.5
    strong = PlasmaRayTransferPlan(path, coupling="strong").evaluate(
        zeros, zeros, incident[None]
    )
    assert int(strong.status[0]) & PlasmaRayTransferStatus.ANISOTROPY_VIOLATED


def test_thermal_magnetobremsstrahlung_saturates_at_kirchhoff_along_the_ray() -> None:
    field, density, tilt = 1.0, 1.0e18, 1.2
    temperature = 0.02
    profile = ColdPlasmaProfile(
        SCALE,
        COORDINATES,
        charge_numbers=[-1.0],
        mass_ratios=[1.0],
        density=lambda point: jnp.stack([density + 0.0 * point[0]]),
        magnetic_field=lambda point: (
            field
            * jnp.stack(
                [
                    np.cos(tilt) + 0.0 * point[0],
                    0.0 * point[0],
                    np.sin(tilt) + 0.0 * point[0],
                ]
            )
        ),
        profile_id="thermal-slab",
    )

    def factory(
        dielectric: ColdPlasmaDielectric, _point: np.ndarray
    ) -> MagnetobremsstrahlungPlan:
        return MagnetobremsstrahlungPlan(
            dielectric,
            ThermalJuttnerDistribution(temperature),
            emitter_density=1.0e15,
            maximum_harmonics=32,
        )

    omega = 2.2 * E * field / M_E
    probe = factory(profile.dielectric_at(np.zeros(3)), np.zeros(3)).evaluate(omega, tilt)
    extraordinary = float(probe.select(PlasmaWaveMode.EXTRAORDINARY, probe.absorption))
    hamiltonian = ColdPlasmaHamiltonian(
        profile, angular_frequency=omega, mode=PlasmaWaveMode.EXTRAORDINARY
    )
    # Four segments of optical depth ≈ 3 keep each exact exponential well resolved.
    count = 4
    rays = (
        DispersionRayPlan(
            hamiltonian,
            3.0 / extraordinary,
            count,
            hamiltonian_tolerance=1.0e-8,
        )
        .prepare()
        .integrate(np.zeros((1, 3)), np.asarray(((1.0, 0.0, 0.0),)))
    )
    path = hamiltonian.sample_path(rays, (0.0, 1.0, 0.0))
    coefficients = magnetobremsstrahlung_path_coefficients(hamiltonian, path, factory)
    rayleigh_jeans = temperature * M_E * C**2 * omega**2 / (8.0 * np.pi**3 * C**2)
    incident = np.zeros((1, 4))
    weak = PlasmaRayTransferPlan(path, coupling="weak").evaluate(
        coefficients.emission, coefficients.absorption, incident
    )
    assert bool(weak.successful)
    # Uniform slab from zero intensity: I/n_r² = (kTω²/8π³c²)(1 − e^{−τ}) (Kirchhoff).
    depth = float(np.sum(coefficients.absorption[0, :, 0] * path.normal_lengths[0]))
    assert depth > 10.0
    np.testing.assert_allclose(
        weak.invariant[0, 0], rayleigh_jeans * -np.expm1(-depth), rtol=1.0e-4
    )
    np.testing.assert_allclose(
        weak.stokes[0, 1:] / weak.stokes[0, 0], path.mode_stokes[0, -1, 0], atol=1.0e-12
    )
    # The ray carries the X mode: its coefficients lead the (ray, companion) order.
    np.testing.assert_allclose(
        coefficients.absorption[..., 0], extraordinary, rtol=1.0e-12
    )
    np.testing.assert_array_equal(coefficients.status, 0)


@pytest.mark.strict_jax
def test_plasma_transfer_refuses_mismatched_inputs() -> None:
    path, _, _ = _twisted(0.02, 0.2, 8)
    with pytest.raises(ValueError, match="coupling"):
        PlasmaRayTransferPlan(path, coupling="adiabatic")  # ty: ignore[invalid-argument-type]
    plan = PlasmaRayTransferPlan(path, coupling="weak")
    with pytest.raises(ValueError, match="segment"):
        plan.evaluate(np.zeros((1, 7, 2)), np.zeros((1, 7, 2)), np.zeros((1, 4)))
    nan = np.full((1, 8, 2), np.nan)
    result = plan.evaluate(nan, nan, np.asarray(((1.0, 1.0, 0.0, 0.0),)))
    assert int(result.status[0]) & PlasmaRayTransferStatus.COEFFICIENT_UNSUPPORTED
    assert not bool(result.successful)
