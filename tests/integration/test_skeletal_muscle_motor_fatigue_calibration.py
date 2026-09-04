#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications.skeletal_muscle.fatigue import (
    commit_liu_brown_yue_2002,
    LiuBrownYue2002Parameters,
    LiuBrownYue2002Plan,
)
from phydrax.applications.skeletal_muscle.motor_units import (
    commit_fuglevand_winter_patla_1993,
    FuglevandWinterPatla1993Plan,
    FuglevandWinterPatla1993RandomInput,
)
from phydrax.applications.skeletal_muscle.personalization import (
    commit_physical_relative_force_calibration,
    PhysicalRelativeForceCalibrationPlan,
)


def test_motor_force_calibration_and_macroscopic_fatigue_remain_separate_routes():
    motor = FuglevandWinterPatla1993Plan(
        24,
        event_capacity_per_unit=4,
        random_stream_id="integration-motor-events",
    ).prepare()
    motor_state = motor.initialize()
    random_input = FuglevandWinterPatla1993RandomInput(
        jr.key(9),
        motor_state.random_step,
        stream_id="integration-motor-events",
    )
    motor_candidate = motor.evaluate(
        motor_state,
        motor.maximum_excitation,
        10.0,
        random_input,
    )
    motor_state = commit_fuglevand_winter_patla_1993(
        motor_candidate, motor_state
    )
    arbitrary_force = motor.force(motor_state).total_force_arbitrary
    assert float(arbitrary_force) > 0.0

    relative_protocol = jnp.asarray([0.0, 0.2, 0.5, 0.75, 1.0])
    calibration = PhysicalRelativeForceCalibrationPlan(
        jnp.ones((5, 1)),
        ("load-cell-offset",),
        protocol_id="independent-motor-route-ramp",
        asset_id="load-cell-integration-fixture",
    ).prepare()
    calibration_state = calibration.initialize(100.0)
    fit = calibration.evaluate(
        calibration_state,
        relative_protocol,
        600.0 * relative_protocol + 2.0,
        jnp.ones((5,)),
    )
    calibration_state = commit_physical_relative_force_calibration(
        fit, calibration_state
    )
    normalized_motor_force = arbitrary_force / jnp.sum(
        motor.peak_twitch_force_arbitrary
    )
    observed = calibration.observe(calibration_state, normalized_motor_force)
    np.testing.assert_allclose(
        observed.force_newton,
        600.0 * normalized_motor_force,
        rtol=2e-5,
    )

    fatigue = LiuBrownYue2002Plan(
        LiuBrownYue2002Parameters(
            fatigue_rate_per_s=0.0206,
            recovery_rate_per_s=0.0084,
        ),
        muscle_id="separate-macroscopic-route",
        protocol_id="constant-brain-effort",
    ).prepare()
    fatigue_state = fatigue.initialize()
    fatigue_candidate = fatigue.evaluate(fatigue_state, 1.0, 30.0)
    fatigue_state = commit_liu_brown_yue_2002(fatigue_candidate, fatigue_state)
    capacity = fatigue.capacity(fatigue_state)

    assert 0.0 < float(capacity.active_relative_force) < 1.0
    assert motor_state.model_id != fatigue_state.model_id
    assert observed.plan_id == calibration_state.plan_id
