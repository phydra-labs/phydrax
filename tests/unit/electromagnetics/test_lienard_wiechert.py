#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Near-zone Liénard–Wiechert fields against independent closed-form references.

References: Heaviside's field of a uniformly moving charge written in its
present position (Jackson, *Classical Electrodynamics*, 3rd ed., eq. 11.154;
Feynman Lectures II §26-2) and the uniform-motion retarded distance
``c (t − t_r) = γ² (β·R_p + √((β·R_p)² + R_p²/γ²))``; Born's field of a charge
in hyperbolic motion (Born 1909; Fulton & Rohrlich, Ann. Phys. 9, 499, 1960),
checked here against its own divergence-free form; Gauss's law on a
Gauss–Legendre sphere; and the A1 far-field observer-time waveform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import ElectromagneticScaleContract
from phydrax.electromagnetics import (
    ChargedTrajectory,
    LienardWiechertFieldPlan,
    LienardWiechertFieldResult,
    LienardWiechertHistory,
    LienardWiechertInterpolation,
    LienardWiechertResourceError,
    LienardWiechertResources,
    LienardWiechertStatus,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
)


SCALE = ElectromagneticScaleContract.si()
C = float(SCALE.speed_of_light)
EPS0 = float(SCALE.vacuum_permittivity)
Q = float(SCALE.elementary_charge)
K = Q / (4.0 * np.pi * EPS0)
INTERPOLATIONS = ("hermite-cubic", "hermite-quintic")


def _trajectory(
    times: np.ndarray,
    positions: np.ndarray | jax.Array,
    proper: np.ndarray,
    rates: np.ndarray | None = None,
    *,
    charges: np.ndarray | None = None,
    active: np.ndarray | None = None,
) -> ChargedTrajectory:
    """Lanes ``[T, P, 3]`` with unit multiplicity and charge ``Q`` by default."""
    count = positions.shape[1]
    return ChargedTrajectory(
        times,
        positions,
        proper,
        np.full(count, Q) if charges is None else charges,
        np.ones(count),
        np.ones(positions.shape[:2], dtype=bool) if active is None else active,
        (np.arange(count, dtype=np.uint32), np.zeros(count, dtype=np.uint32)),
        proper_accelerations=rates,
    )


def _evaluate(
    trajectory: ChargedTrajectory,
    events: np.ndarray,
    *,
    interpolation: LienardWiechertInterpolation = "hermite-quintic",
    history: LienardWiechertHistory = "refuse",
    exclusion_radius: float = 1.0e-9,
    resources: LienardWiechertResources | None = None,
) -> LienardWiechertFieldResult:
    plan = LienardWiechertFieldPlan(
        SCALE,
        history=history,
        exclusion_radius=exclusion_radius,
        interpolation=interpolation,
        resources=resources,
    )
    return plan.prepare().evaluate(trajectory, events)


def _uniform_lane(
    velocity_gamma: float, direction: np.ndarray, times: np.ndarray, start: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions, proper velocities, and zero rates of uniform motion."""
    unit = direction / np.linalg.norm(direction)
    proper = C * np.sqrt(velocity_gamma**2 - 1.0) * unit
    velocity = proper / velocity_gamma
    positions = start[None, :] + times[:, None] * velocity[None, :]
    return (
        positions,
        np.broadcast_to(proper, positions.shape).copy(),
        np.zeros_like(positions),
    )


def _heaviside(
    events: np.ndarray, start: np.ndarray, proper: np.ndarray, charge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Present-position field ``E``, ``B = β × E / c``, and retarded times."""
    gamma = np.sqrt(1.0 + proper @ proper / C**2)
    beta = proper / (gamma * C)
    present = events[:, 1:] - (start[None, :] + events[:, :1] * (beta * C)[None, :])
    distance = np.linalg.norm(present, axis=1)
    along = present @ beta
    # 1 − β² sin²ψ = 1/γ² + (β·R̂)², free of cancellation at large γ.
    shape = 1.0 / gamma**2 + (along / distance) ** 2
    field = (
        charge
        / (4.0 * np.pi * EPS0)
        * present
        / (gamma**2 * distance[:, None] ** 3 * shape[:, None] ** 1.5)
    )
    magnetic = np.cross(beta[None, :], field) / C
    # c (t − t_r) = γ² (β·R_p + √((β·R_p)² + R_p²/γ²)), rationalized for β·R_p < 0.
    delay = distance**2 / (np.sqrt(along**2 + distance**2 / gamma**2) - along) / C
    return field, magnetic, events[:, 0] - delay


def _hyperbolic(
    alpha: float, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant proper acceleration ``c²/α`` along ``z``: ``z² − c² t² = α²``."""
    zeros = np.zeros_like(times)
    positions = np.stack((zeros, zeros, np.sqrt(alpha**2 + (C * times) ** 2)), axis=-1)
    proper = np.stack((zeros, zeros, C**2 * times / alpha), axis=-1)
    rates = np.stack((zeros, zeros, np.full_like(times, C**2 / alpha)), axis=-1)
    return positions[:, None], proper[:, None], rates[:, None]


def _born(alpha: float, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Born's hyperbolic-motion field in Cartesian components (``z + ct > 0``)."""
    ct = C * events[:, 0]
    x, y, z = events[:, 1], events[:, 2], events[:, 3]
    rho = np.hypot(x, y)
    xi = np.sqrt((alpha**2 + ct**2 - rho**2 - z**2) ** 2 + 4.0 * alpha**2 * rho**2)
    radial = K * 8.0 * alpha**2 * rho * z / xi**3
    axial = -K * 4.0 * alpha**2 * (alpha**2 + ct**2 + rho**2 - z**2) / xi**3
    azimuthal = K * 8.0 * alpha**2 * rho * ct / (C * xi**3)
    electric = np.stack((radial * x / rho, radial * y / rho, axial), axis=-1)
    magnetic = np.stack((-azimuthal * y / rho, azimuthal * x / rho, 0.0 * rho), axis=-1)
    return electric, magnetic


def _hyperbolic_events() -> np.ndarray:
    alpha = 1.0
    return np.array(
        [
            [0.0, 0.5 * alpha, 0.0, 1.2 * alpha],
            [0.4 * alpha / C, 0.3 * alpha, 0.4 * alpha, 0.8 * alpha],
            [1.1 * alpha / C, -0.6 * alpha, 0.2 * alpha, 2.0 * alpha],
            [-0.3 * alpha / C, 0.2 * alpha, -0.9 * alpha, 1.6 * alpha],
        ]
    )


def _circular_lane(
    radius: float, beta: float, times: np.ndarray, phase: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = beta * C / radius
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    angle = omega * times + phase
    zeros = np.zeros_like(times)
    cosine, sine = np.cos(angle), np.sin(angle)
    positions = radius * np.stack((cosine, sine, zeros), axis=-1)
    proper = gamma * beta * C * np.stack((-sine, cosine, zeros), axis=-1)
    rates = -gamma * beta * C * omega * np.stack((cosine, sine, zeros), axis=-1)
    return positions, proper, rates


# The cubic route differentiates rounded positions twice, so its acceleration
# noise grows as ε N² γ²; at γ = 10⁴ only the quintic route, whose β̇ comes
# from the proper velocity, is exact.
@pytest.mark.parametrize(
    ("interpolation", "gamma"),
    (
        ("hermite-cubic", 1.25),
        ("hermite-quintic", 1.25),
        ("hermite-quintic", 1.0e4),
    ),
    ids=("cubic-gamma-1.25", "quintic-gamma-1.25", "quintic-gamma-1e4"),
)
def test_uniform_motion_matches_heaviside_field(
    interpolation: LienardWiechertInterpolation, gamma: float
) -> None:
    # A long uniform window keeps the retarded points of transverse observers
    # of the γ = 10⁴ charge (c (t − t_r) ≈ γ ρ) inside the samples.
    times = np.linspace(-1.0e-5, 4.0e-9, 65)
    start = np.array([0.01, -0.02, 0.0])
    direction = np.array([0.6, 0.0, 0.8])
    positions, proper, rates = _uniform_lane(gamma, direction, times, start)
    first = np.array([0.8, 0.0, -0.6])
    second = np.cross(direction, first)
    # Present-position separations: transverse, trailing, and nearly transverse.
    separations = np.stack(
        (
            0.1 * first,
            -0.2 * direction + 0.05 * second,
            -0.03 * direction + 0.3 * first,
            -0.5 * direction + 0.2 * first - 0.1 * second,
        )
    )
    event_times = np.array([0.0, 1.0e-9, 2.5e-10, 1.5e-9])
    present = start[None, :] + event_times[:, None] * (proper[0] / gamma)[None, :]
    events = np.concatenate((event_times[:, None], present + separations), axis=1)
    result = _evaluate(
        _trajectory(times, positions[:, None], proper[:, None], rates[:, None]),
        events,
        interpolation=interpolation,
    )
    field, magnetic, retarded = _heaviside(events, start, proper[0], Q)
    scale = np.max(np.linalg.norm(field, axis=1))
    np.testing.assert_allclose(
        result.electric_field, field, rtol=1e-9, atol=1e-10 * scale
    )
    np.testing.assert_allclose(
        result.magnetic_field, magnetic, rtol=1e-9, atol=1e-10 * scale / C
    )
    np.testing.assert_allclose(result.acceleration_field, 0.0, atol=1e-12 * scale)
    # The root is solved in observer time; its retarded-time error is the
    # reported observer-time residual amplified by 1/κ.
    delay = events[:, 0] - retarded
    evidence = result.evidence
    rounding = 1e-15 * (delay + np.max(np.abs(positions)) / C)
    bound = (np.asarray(evidence.maximum_root_residual) + rounding) / np.asarray(
        evidence.minimum_retardation_factor
    )
    assert np.all(np.abs(np.asarray(result.retarded_times)[:, 0] - retarded) <= bound)
    assert np.all(np.asarray(result.evidence.status) == 0)
    assert np.all(np.asarray(result.evidence.supported))
    assert np.all(np.asarray(result.evidence.resolved))


def test_hyperbolic_motion_matches_born_field() -> None:
    alpha = 1.0
    times = np.linspace(-3.0 * alpha / C, 3.0 * alpha / C, 481)
    positions, proper, rates = _hyperbolic(alpha, times)
    events = _hyperbolic_events()
    result = _evaluate(_trajectory(times, positions, proper, rates), events)
    electric, magnetic = _born(alpha, events)
    scale = np.max(np.linalg.norm(electric, axis=1))
    np.testing.assert_allclose(
        result.electric_field, electric, rtol=0.0, atol=1e-8 * scale
    )
    np.testing.assert_allclose(
        result.magnetic_field, magnetic, rtol=0.0, atol=1e-8 * scale / C
    )
    # The acceleration part carries the radiation; it is far from negligible.
    assert np.max(np.abs(np.asarray(result.acceleration_field))) > 1e-2 * scale
    assert np.all(np.asarray(result.evidence.status) == 0)
    assert np.all(np.asarray(result.evidence.derivative_valid))


def test_hermite_reconstructions_converge_at_their_orders() -> None:
    alpha = 1.0
    events = _hyperbolic_events()
    electric, _ = _born(alpha, events)
    errors: dict[str, list[float]] = {name: [] for name in INTERPOLATIONS}
    for count in (61, 121):
        times = np.linspace(-3.0 * alpha / C, 3.0 * alpha / C, count)
        positions, proper, rates = _hyperbolic(alpha, times)
        trajectory = _trajectory(times, positions, proper, rates)
        for name in INTERPOLATIONS:
            result = _evaluate(trajectory, events, interpolation=name)
            errors[name].append(
                float(np.max(np.abs(np.asarray(result.electric_field) - electric)))
            )
    # Cubic: second-order acceleration and proper-velocity chord; quintic:
    # fourth-order acceleration and cubic proper velocity.
    assert errors["hermite-cubic"][0] / errors["hermite-cubic"][1] > 3.0
    assert errors["hermite-quintic"][0] / errors["hermite-quintic"][1] > 12.0
    assert errors["hermite-quintic"][1] < 1e-2 * errors["hermite-cubic"][1]


def test_far_zone_acceleration_field_matches_trajectory_waveform() -> None:
    sigma = 1.0e-9
    count = 1601
    times = np.linspace(-8.0 * sigma, 8.0 * sigma, count)
    peak = 0.05 / np.sqrt(1.0 - 0.05**2) * C
    u = peak * np.exp(-(times**2) / (2.0 * sigma**2))
    nodes, weights = np.polynomial.legendre.leggauss(12)
    lower, upper = times[:-1, None], times[1:, None]
    points = 0.5 * (upper - lower) * nodes[None, :] + 0.5 * (upper + lower)
    u_points = peak * np.exp(-(points**2) / (2.0 * sigma**2))
    speed = u_points / np.sqrt(1.0 + (u_points / C) ** 2)
    x = np.concatenate(
        ([0.0], np.cumsum(0.5 * (upper[:, 0] - lower[:, 0]) * (speed @ weights)))
    )
    zeros = np.zeros_like(times)
    positions = np.stack((x, zeros, zeros), axis=-1)[:, None]
    proper = np.stack((u, zeros, zeros), axis=-1)[:, None]
    rates = np.stack((-times / sigma**2 * u, zeros, zeros), axis=-1)[:, None]
    trajectory = _trajectory(times, positions, proper, rates)
    directions = np.array([[0.0, 1.0, 0.0], [np.sin(0.7), 0.0, np.cos(0.7)]])
    plan = TrajectoryRadiationPlan(
        SCALE,
        RadiationObserverPlan(directions, np.array([0.0, 0.0, 1.0])),
        np.linspace(0.05, 12.0, 240) / sigma,
        coherence="coherent",
        route="segment-hermite",
    )
    prepared = plan.prepare()
    spectrum = prepared.evaluate(trajectory)
    distance = 1.0e3
    tau = np.linspace(-2.0 * sigma, 2.0 * sigma, 13)
    for index, direction in enumerate(directions):
        waveform = np.asarray(prepared.waveform(spectrum, tau))[:, index]
        events = np.concatenate(
            (
                (tau + distance / C)[:, None],
                np.broadcast_to(distance * direction, (tau.shape[0], 3)),
            ),
            axis=1,
        )
        near = _evaluate(trajectory, events)
        field = distance * np.asarray(near.acceleration_field)
        projected = np.stack(
            (
                field @ np.asarray(plan.observers.basis_first)[index],
                field @ np.asarray(plan.observers.basis_second)[index],
            ),
            axis=-1,
        )
        np.testing.assert_allclose(
            projected, waveform, rtol=0.0, atol=1e-3 * np.max(np.abs(waveform))
        )
        assert np.all(np.asarray(near.evidence.supported))


def test_electric_flux_through_a_sphere_equals_enclosed_charge() -> None:
    times = np.linspace(-20.0e-9, 10.0e-9, 4001)
    inside, inside_u, inside_rates = _circular_lane(0.3, 0.6, times)
    outside, outside_u, outside_rates = _uniform_lane(
        1.0 / np.sqrt(1.0 - 0.4**2),
        np.array([0.0, 1.0, 0.0]),
        times,
        np.array([2.5, 0.0, 0.0]),
    )
    trajectory = _trajectory(
        times,
        np.stack((inside, outside), axis=1),
        np.stack((inside_u, outside_u), axis=1),
        np.stack((inside_rates, outside_rates), axis=1),
        charges=np.array([Q, -2.0 * Q]),
    )
    cosines, polar_weights = np.polynomial.legendre.leggauss(48)
    azimuth = 2.0 * np.pi * np.arange(96) / 96
    sines = np.sqrt(1.0 - cosines**2)
    normals = np.stack(
        (
            sines[:, None] * np.cos(azimuth)[None, :],
            sines[:, None] * np.sin(azimuth)[None, :],
            np.broadcast_to(cosines[:, None], (48, 96)),
        ),
        axis=-1,
    ).reshape(-1, 3)
    weights = np.repeat(polar_weights, 96) * (2.0 * np.pi / 96)
    events = np.concatenate((np.full((normals.shape[0], 1), 5.0e-9), normals), axis=1)
    result = _evaluate(trajectory, events)
    flux = np.sum(weights * np.sum(np.asarray(result.electric_field) * normals, axis=1))
    np.testing.assert_allclose(flux * EPS0, Q, rtol=1e-6)
    assert np.all(np.asarray(result.evidence.supported))


def test_history_refusal_marks_retarded_times_outside_the_window() -> None:
    times = np.linspace(0.0, 2.0e-9, 41)
    positions, proper, rates = _uniform_lane(
        1.5, np.array([1.0, 0.0, 0.0]), times, np.zeros(3)
    )
    trajectory = _trajectory(times, positions[:, None], proper[:, None], rates[:, None])
    events = np.array(
        [
            [1.0e-9, 0.0, 0.5, 0.0],  # light from t = 0 has not arrived
            [2.0e-9, 0.2, 0.1, 0.0],  # interior
            [9.0e-9, 0.2, 0.1, 0.0],  # after the last sample's light passed
        ]
    )
    result = _evaluate(trajectory, events)
    status = np.asarray(result.evidence.status)
    assert status[0] == LienardWiechertStatus.RETARDED_BEFORE_WINDOW
    assert status[1] == 0
    assert status[2] == LienardWiechertStatus.RETARDED_AFTER_WINDOW
    np.testing.assert_array_equal(result.evidence.supported, [False, True, False])
    np.testing.assert_array_equal(result.evidence.derivative_valid, [False, True, False])
    field = np.asarray(result.electric_field)
    assert np.all(np.isnan(field[[0, 2]])) and np.all(np.isfinite(field[1]))
    assert np.isnan(np.asarray(result.retarded_times)[[0, 2], 0]).all()


def test_inertial_extrapolation_continues_uniform_history_exactly() -> None:
    times = np.linspace(0.0, 2.0e-9, 41)
    start = np.array([0.0, 0.0, 0.0])
    positions, proper, rates = _uniform_lane(3.0, np.array([1.0, 0.2, 0.0]), times, start)
    trajectory = _trajectory(times, positions[:, None], proper[:, None], rates[:, None])
    events = np.array([[1.0e-9, 0.0, 0.5, 0.0], [-2.0e-9, 0.4, -0.3, 0.2]])
    result = _evaluate(trajectory, events, history="inertial-extrapolation")
    field, magnetic, retarded = _heaviside(events, start, proper[0], Q)
    np.testing.assert_allclose(result.electric_field, field, rtol=1e-11)
    np.testing.assert_allclose(result.magnetic_field, magnetic, rtol=1e-11, atol=1e-30)
    np.testing.assert_allclose(
        result.retarded_times[:, 0], retarded, rtol=0.0, atol=1e-20
    )
    np.testing.assert_array_equal(
        result.evidence.status, [int(LienardWiechertStatus.RETARDED_BEFORE_WINDOW)] * 2
    )
    np.testing.assert_array_equal(result.evidence.extrapolated_count, [1, 1])
    assert np.all(np.asarray(result.evidence.supported))
    assert np.all(np.asarray(result.evidence.derivative_valid))


@pytest.mark.parametrize(
    ("defect", "flag"),
    (
        ("superluminal", LienardWiechertStatus.SUPERLUMINAL_SAMPLES),
        ("nonmonotone", LienardWiechertStatus.NONMONOTONE_TIME),
    ),
)
def test_causality_violating_samples_are_unsupported(
    defect: str, flag: LienardWiechertStatus
) -> None:
    times = np.linspace(0.0, 2.0e-9, 41)
    positions, proper, rates = _uniform_lane(
        1.5, np.array([1.0, 0.0, 0.0]), times, np.zeros(3)
    )
    second, second_u, second_rates = _uniform_lane(
        1.2, np.array([0.0, 1.0, 0.0]), times, np.array([0.0, 0.0, 0.3])
    )
    lane_times = np.stack((times, times), axis=1)
    if defect == "superluminal":
        positions[20:, 0] += 2.0 * C * (times[1] - times[0])
    else:
        lane_times[20, 0] = lane_times[19, 0]
    trajectory = _trajectory(
        lane_times,
        np.stack((positions, second), axis=1),
        np.stack((proper, second_u), axis=1),
        np.stack((rates, second_rates), axis=1),
    )
    result = _evaluate(trajectory, np.array([[1.5e-9, 0.2, 0.3, 0.1]]))
    pair_status = np.asarray(result.evidence.pair_status)[0]
    assert pair_status[0] & flag
    assert pair_status[1] == 0
    assert not bool(result.evidence.supported[0])
    assert np.isnan(np.asarray(result.retarded_times)[0, 0])
    assert np.isfinite(np.asarray(result.retarded_times)[0, 1])


def test_exclusion_radius_removes_and_accounts_the_near_charge() -> None:
    times = np.linspace(0.0, 2.0e-9, 41)
    near, near_u, near_rates = _uniform_lane(
        1.1, np.array([1.0, 0.0, 0.0]), times, np.zeros(3)
    )
    far, far_u, far_rates = _uniform_lane(
        2.0, np.array([0.0, 0.0, 1.0]), times, np.array([0.3, 0.2, 0.0])
    )
    charges = np.array([-3.0 * Q, Q])
    both = _trajectory(
        times,
        np.stack((near, far), axis=1),
        np.stack((near_u, far_u), axis=1),
        np.stack((near_rates, far_rates), axis=1),
        charges=charges,
    )
    # The observer sits 1 mm from the near charge's retarded position.
    retarded_time = 1.0e-9
    point = near[20] + np.array([0.0, 1.0e-3, 0.0])
    events = np.concatenate(([retarded_time + 1.0e-3 / C], point))[None, :]
    result = _evaluate(both, events, exclusion_radius=1.0e-2)
    field, _, _ = _heaviside(events, np.array([0.3, 0.2, 0.0]), far_u[0], Q)
    np.testing.assert_allclose(result.electric_field, field, rtol=1e-10)
    np.testing.assert_allclose(result.evidence.excluded_charge, [-3.0 * Q], rtol=1e-15)
    np.testing.assert_array_equal(result.evidence.excluded_count, [1])
    assert result.evidence.status[0] == LienardWiechertStatus.EXCLUDED_CHARGE
    assert bool(result.evidence.supported[0])
    assert not bool(result.evidence.derivative_valid[0])


def test_inactive_retarded_samples_report_absent_charge_of_a_created_pair() -> None:
    times = np.linspace(-5.0e-9, 5.0e-9, 101)
    after = times >= 0.0
    lanes = []
    for sign in (1.0, -1.0):
        positions, proper, rates = _uniform_lane(
            1.0 / np.sqrt(0.75), np.array([sign, 0.0, 0.0]), times, np.zeros(3)
        )
        # Before creation the lane holds the creation point: causal placeholders.
        positions[~after] = 0.0
        lanes.append((positions, proper, rates))
    trajectory = _trajectory(
        times,
        np.stack([lane[0] for lane in lanes], axis=1),
        np.stack([lane[1] for lane in lanes], axis=1),
        np.stack([lane[2] for lane in lanes], axis=1),
        charges=np.array([-Q, Q]),
        active=np.stack((after, after), axis=1),
    )
    events = np.array([[1.0e-9, 0.0, 1.0, 0.0], [4.5e-9, 0.0, 1.0, 0.0]])
    result = _evaluate(trajectory, events)
    field = np.asarray(result.electric_field)
    np.testing.assert_array_equal(field[0], 0.0)
    np.testing.assert_allclose(result.evidence.absent_charge, [0.0, 0.0], atol=1e-30)
    assert result.evidence.status[0] == LienardWiechertStatus.INACTIVE_RETARDED
    np.testing.assert_array_equal(result.evidence.supported, [True, True])
    np.testing.assert_array_equal(result.evidence.derivative_valid, [False, True])
    expected = sum(
        _heaviside(events[1:], np.zeros(3), lane[1][0], charge)[0]
        for lane, charge in zip(lanes, (-Q, Q), strict=True)
    )
    np.testing.assert_allclose(field[1:], expected, rtol=1e-10)


def test_chunked_execution_matches_one_block() -> None:
    times = np.linspace(-20.0e-9, 10.0e-9, 1201)
    lanes = [_circular_lane(0.3, 0.5, times, phase) for phase in (0.0, 2.0, 4.0)]
    trajectory = _trajectory(
        times,
        np.stack([lane[0] for lane in lanes], axis=1),
        np.stack([lane[1] for lane in lanes], axis=1),
        np.stack([lane[2] for lane in lanes], axis=1),
    )
    rng = np.random.default_rng(3)
    events = np.concatenate(
        (np.full((7, 1), 2.0e-9), rng.uniform(-1.0, 1.0, (7, 3))), axis=1
    )
    whole = _evaluate(trajectory, events)
    chunked = _evaluate(
        trajectory,
        events,
        resources=LienardWiechertResources(observer_chunk=3, particle_chunk=2),
    )
    np.testing.assert_allclose(chunked.electric_field, whole.electric_field, rtol=1e-13)
    np.testing.assert_allclose(chunked.magnetic_field, whole.magnetic_field, rtol=1e-13)
    np.testing.assert_array_equal(chunked.retarded_times, whole.retarded_times)
    np.testing.assert_array_equal(
        chunked.evidence.pair_status, whole.evidence.pair_status
    )


def test_field_derivatives_match_finite_differences_and_transpose() -> None:
    alpha = 1.0
    times = np.linspace(-3.0 * alpha / C, 3.0 * alpha / C, 481)
    positions, proper, rates = _hyperbolic(alpha, times)
    events = jnp.asarray(_hyperbolic_events())
    prepared = LienardWiechertFieldPlan(
        SCALE, history="refuse", exclusion_radius=1.0e-9, interpolation="hermite-quintic"
    ).prepare()
    positions_ = jnp.asarray(positions)

    def field(observer: jax.Array, shift: jax.Array) -> jax.Array:
        trajectory = _trajectory(times, positions_ + shift, proper, rates)
        return prepared.evaluate(trajectory, observer).electric_field / K

    shift = jnp.zeros((3,), dtype=jnp.float64)
    observer_tangent = jnp.asarray(
        np.array([[0.3 / C, 0.1, -0.2, 0.05]] * 4), dtype=jnp.float64
    )
    shift_tangent = jnp.asarray([0.02, -0.01, 0.03])
    primal, tangent = jax.jvp(field, (events, shift), (observer_tangent, shift_tangent))
    step = 1.0e-4
    difference = (
        field(events + step * observer_tangent, shift + step * shift_tangent)
        - field(events - step * observer_tangent, shift - step * shift_tangent)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        tangent, difference, rtol=1e-6, atol=1e-8 * np.max(np.abs(tangent))
    )
    cotangent = jnp.asarray(np.random.default_rng(5).normal(size=primal.shape))
    _, pullback = jax.vjp(field, events, shift)
    observer_cotangent, shift_cotangent = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, tangent),
        jnp.vdot(observer_cotangent, observer_tangent)
        + jnp.vdot(shift_cotangent, shift_tangent),
        rtol=1e-10,
    )


def test_resource_limits_refuse_before_execution() -> None:
    times = np.linspace(0.0, 1.0e-9, 11)
    positions, proper, rates = _uniform_lane(
        1.5, np.array([1.0, 0.0, 0.0]), times, np.zeros(3)
    )
    trajectory = _trajectory(times, positions[:, None], proper[:, None], rates[:, None])
    events = np.array([[2.0e-9, 0.1, 0.1, 0.1]])
    with pytest.raises(LienardWiechertResourceError, match="working bytes"):
        _evaluate(
            trajectory,
            events,
            resources=LienardWiechertResources(maximum_working_bytes=64),
        )
    with pytest.raises(LienardWiechertResourceError, match="outputs"):
        _evaluate(
            trajectory,
            events,
            resources=LienardWiechertResources(maximum_output_bytes=64),
        )


def test_invalid_plans_and_inputs_are_refused() -> None:
    with pytest.raises(TypeError, match="ElectromagneticScaleContract"):
        LienardWiechertFieldPlan(
            object(),  # ty: ignore[invalid-argument-type]
            history="refuse",
            exclusion_radius=1.0,
            interpolation="hermite-cubic",
        )
    with pytest.raises(ValueError):
        LienardWiechertFieldPlan(
            SCALE,
            history="forward",  # ty: ignore[invalid-argument-type]
            exclusion_radius=1.0,
            interpolation="hermite-cubic",
        )
    with pytest.raises(ValueError):
        LienardWiechertFieldPlan(
            SCALE,
            history="refuse",
            exclusion_radius=1.0,
            interpolation="linear",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="exclusion_radius"):
        LienardWiechertFieldPlan(
            SCALE, history="refuse", exclusion_radius=0.0, interpolation="hermite-cubic"
        )
    times = np.linspace(0.0, 1.0e-9, 11)
    positions, proper, _ = _uniform_lane(
        1.5, np.array([1.0, 0.0, 0.0]), times, np.zeros(3)
    )
    without_rates = _trajectory(times, positions[:, None], proper[:, None])
    events = np.array([[2.0e-9, 0.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="proper_accelerations"):
        _evaluate(without_rates, events, interpolation="hermite-quintic")
    with pytest.raises(TypeError, match="float64"):
        _evaluate(without_rates, events.astype(np.float32), interpolation="hermite-cubic")
    with pytest.raises(ValueError, match="observers, 4"):
        _evaluate(without_rates, events[:, :3], interpolation="hermite-cubic")
    with pytest.raises(ValueError, match="two samples"):
        _evaluate(
            _trajectory(times[:1], positions[:1, None], proper[:1, None]),
            events,
            interpolation="hermite-cubic",
        )
    with pytest.raises(TypeError, match="ChargedTrajectory"):
        LienardWiechertFieldPlan(
            SCALE, history="refuse", exclusion_radius=1.0, interpolation="hermite-cubic"
        ).prepare().evaluate(
            object(),  # ty: ignore[invalid-argument-type]
            events,
        )
