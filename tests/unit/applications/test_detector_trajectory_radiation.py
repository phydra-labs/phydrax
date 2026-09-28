#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Detector constant-field tracks as far-field radiation lanes.

References: the relativistic cyclotron frequency ``ω_c = |q| B / (γ m)`` and
its on-axis Doppler image ``ω_c / (1 − β∥)`` for a helix observed along the
field (Jackson, *Classical Electrodynamics*, 3rd ed., §12.1 and §14.6), and
``u = p c² / E₀`` for the proper velocity of a track of momentum ``p`` and rest
energy ``E₀``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import DTypeLike

from phydrax import ElectromagneticScaleContract, units
from phydrax.applications import detector
from phydrax.discretization.pic import PIC_CODE_RELATIVITY, RelativisticPushPlan
from phydrax.electromagnetics import (
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
    TrajectoryRadiationStatus,
)
from phydrax.units import CHARGE, UnitDefinition


def _code_scale(speed_of_light: int) -> ElectromagneticScaleContract:
    return ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=speed_of_light,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="detector-trajectory-test",
    )


def _conditions(magnetic: float) -> detector.DetectorConditions:
    return detector.DetectorConditions(
        magnetic_field=jnp.asarray([0.0, 0.0, magnetic]),
        electric_field=jnp.zeros(3),
        momentum_unit=units.JOULE,
        length_unit=units.METER,
        time_unit=units.SECOND,
        geometry_id="vacuum-solenoid",
        material_id="vacuum",
        field_id="uniform-axial",
        alignment_id="nominal",
        calibration_id="nominal",
        validity_interval=(0, 1),
    )


def _mixed_bank(
    conditions: detector.DetectorConditions, *, dtype: DTypeLike = jnp.float64
) -> detector.TransportTrackBank:
    """Two events: a slow track, two fast tracks, and one empty slot."""
    return detector.TransportTrackBank(
        event_ids=jnp.asarray([4, 9]),
        track_ids=jnp.asarray([[0, 1], [5, -1]]),
        parent_track_ids=jnp.full((2, 2), -1),
        pdg_ids=jnp.asarray([[11, -11], [13, 0]]),
        positions=jnp.asarray(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [np.nan, 0.0, 0.0]]],
            dtype=dtype,
        ),
        momenta=jnp.asarray(
            [[[0.1, 0.0, 0.0], [0.0, 2.0, 0.0]], [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]],
            dtype=dtype,
        ),
        rest_energies=jnp.asarray([[1.0, 1.0], [2.0, 0.0]]),
        charges=jnp.asarray([[-1.0, 1.0], [-2.0, 0.0]]),
        active=jnp.asarray([[True, True], [True, False]]),
        conditions_id=conditions.conditions_id,
    )


def test_lanes_prepend_initial_sample_with_identities_and_physical_kinematics() -> None:
    scale = _code_scale(2)
    conditions = _conditions(0.5)
    plan = detector.ChargedPropagationPlan(
        conditions,
        step_size=0.1,
        step_count=5,
        pusher=RelativisticPushPlan(scale.relativity, method="vay"),
    )
    tracks = _mixed_bank(conditions)
    result = detector.propagate_charged_tracks(plan, tracks)
    trajectory = detector.charged_trajectory(plan, tracks, result, scale)

    assert (trajectory.sample_count, trajectory.particle_count) == (6, 4)
    np.testing.assert_array_equal(trajectory.id_hi, np.array([4, 4, 9, 9], np.uint32))
    np.testing.assert_array_equal(
        trajectory.id_lo, np.array([0, 1, 5, 2**32 - 1], np.uint32)
    )
    np.testing.assert_allclose(
        trajectory.times, np.broadcast_to(0.1 * np.arange(6)[:, None], (6, 4))
    )
    np.testing.assert_array_equal(trajectory.charges, [-1.0, 1.0, -2.0, 0.0])
    np.testing.assert_array_equal(trajectory.multiplicities, np.ones(4))
    live = slice(0, 3)
    initial_momenta = np.asarray(tracks.momenta).reshape(4, 3)[live]
    rest = np.array([1.0, 1.0, 2.0])
    np.testing.assert_allclose(
        trajectory.positions[0, live], np.asarray(tracks.positions).reshape(4, 3)[live]
    )
    np.testing.assert_allclose(
        trajectory.proper_velocities[0, live],
        initial_momenta * 4.0 / rest[:, None],
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        trajectory.positions[1:, live],
        np.asarray(result.position_history).reshape(5, 4, 3)[:, live],
    )
    final_momenta = np.asarray(result.tracks.momenta).reshape(4, 3)[live]
    np.testing.assert_allclose(
        trajectory.proper_velocities[-1, live],
        final_momenta * 4.0 / rest[:, None],
        rtol=1e-14,
    )


def test_activity_marks_committed_states_and_holds_inactive_lanes_causal() -> None:
    scale = _code_scale(1)
    conditions = _conditions(0.5)
    loss = 2.0
    plan = detector.ChargedPropagationPlan(
        conditions, step_size=0.1, step_count=6, mean_energy_loss_per_length=loss
    )
    tracks = _mixed_bank(conditions)
    result = detector.propagate_charged_tracks(plan, tracks)
    trajectory = detector.charged_trajectory(plan, tracks, result, scale)
    active = np.asarray(trajectory.active)
    positions = np.asarray(trajectory.positions)

    # Kinetic energy √1.01 − 1 is below the first step's loss 2·|v|·Δt, so the
    # slow track comes to rest at sample 1 and stays there.
    np.testing.assert_array_equal(active[:, 0], [True, True] + [False] * 5)
    np.testing.assert_array_equal(trajectory.proper_velocities[1, 0], np.zeros(3))
    np.testing.assert_array_equal(
        positions[2:, 0], np.broadcast_to(positions[1, 0], (5, 3))
    )
    # |v| < c bounds the six-step loss by 1.2, below both fast kinetic energies.
    assert active[:, 1:3].all()
    assert not active[:, 3].any()
    assert np.isfinite(positions[:, 3]).all()
    np.testing.assert_array_equal(
        positions[:, 3], np.broadcast_to(positions[0, 3], (7, 3))
    )

    radiation = TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 0.0, 1.0])),
        np.array([1.0, 2.0]),
        coherence="incoherent",
        route="segment-exact",
        emission="truncated",
    )
    evidence = radiation.prepare().evaluate(trajectory).evidence
    assert bool(evidence.finite)
    assert int(evidence.status) & TrajectoryRadiationStatus.ACTIVITY_TRANSITION


@pytest.mark.parametrize(
    ("scale", "message"),
    [
        pytest.param(_code_scale(2), "speed of light", id="speed-of-light"),
        pytest.param(ElectromagneticScaleContract.si(), "units", id="dimensional-scale"),
    ],
)
def test_scale_mismatch_with_pusher_is_refused(
    scale: ElectromagneticScaleContract, message: str
) -> None:
    conditions = _conditions(0.5)
    plan = detector.ChargedPropagationPlan(conditions, step_size=0.1, step_count=2)
    tracks = _mixed_bank(conditions)
    result = detector.propagate_charged_tracks(plan, tracks)
    with pytest.raises(ValueError, match=message):
        detector.charged_trajectory(plan, tracks, result, scale)


def test_float32_detector_kinematics_are_refused() -> None:
    conditions = _conditions(0.5)
    plan = detector.ChargedPropagationPlan(conditions, step_size=0.1, step_count=2)
    tracks = _mixed_bank(conditions, dtype=jnp.float32)
    result = detector.propagate_charged_tracks(plan, tracks)
    with pytest.raises(TypeError, match="float64"):
        detector.charged_trajectory(plan, tracks, result, _code_scale(1))


def _log_parabola_peak(frequencies: np.ndarray, energy: np.ndarray) -> float:
    index = int(np.argmax(energy))
    low, mid, high = np.log(energy[index - 1 : index + 2])
    offset = 0.5 * (low - high) / (low - 2.0 * mid + high)
    return float(frequencies[index] + offset * (frequencies[1] - frequencies[0]))


def test_constant_field_helix_radiates_the_relativistic_cyclotron_line() -> None:
    scale = ElectromagneticScaleContract.si()
    light = float(scale.speed_of_light)
    charge = float(scale.elementary_charge)
    mass = float(scale.electron_mass)
    field, gamma, pitch = 1.0, 1.5, np.deg2rad(60.0)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    beta_parallel = beta * np.cos(pitch)
    cyclotron = charge * field / (gamma * mass)
    steps_per_turn, turns = 256, 32
    conditions = _conditions(field)
    plan = detector.ChargedPropagationPlan(
        conditions,
        step_size=2.0 * np.pi / cyclotron / steps_per_turn,
        step_count=steps_per_turn * turns,
        pusher=RelativisticPushPlan(scale.relativity, method="boris"),
    )
    momentum = gamma * mass * light * np.array([0.0, beta * np.sin(pitch), beta_parallel])
    tracks = detector.TransportTrackBank(
        event_ids=jnp.asarray([1]),
        track_ids=jnp.asarray([[0]]),
        parent_track_ids=jnp.asarray([[-1]]),
        pdg_ids=jnp.asarray([[11]]),
        positions=jnp.zeros((1, 1, 3)),
        momenta=jnp.asarray(momentum)[None, None],
        rest_energies=jnp.asarray([[mass * light**2]]),
        charges=jnp.asarray([[-charge]]),
        active=jnp.asarray([[True]]),
        conditions_id=conditions.conditions_id,
    )
    result = detector.propagate_charged_tracks(plan, tracks)
    trajectory = detector.charged_trajectory(plan, tracks, result, scale)

    doppler = cyclotron / (1.0 - beta_parallel)
    transverse = cyclotron * np.linspace(0.95, 1.05, 201)
    axial = doppler * np.linspace(0.95, 1.05, 201)
    radiation = TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.0, 1.0, 0.0])
        ),
        np.concatenate((transverse, axial)),
        coherence="coherent",
        route="segment-exact",
        emission="truncated",
    )
    spectrum = radiation.prepare().evaluate(trajectory)
    energy = np.asarray(spectrum.spectral_energy)

    assert bool(spectrum.evidence.finite)
    assert not int(spectrum.evidence.status) & TrajectoryRadiationStatus.UNRESOLVED_PHASE
    np.testing.assert_allclose(
        _log_parabola_peak(transverse, energy[:201, 0]), cyclotron, rtol=5e-4
    )
    np.testing.assert_allclose(
        _log_parabola_peak(axial, energy[201:, 1]), doppler, rtol=2e-4
    )
