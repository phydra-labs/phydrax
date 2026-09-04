#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.continuum import (
    EngelhardtGasam2025Parameters,
    EngelhardtGasam2025Plan,
    HomogenizedShortenGasamCouplingPlan,
    ShortenGasamActivationCalibration,
    UniformFiberArchitecturePlan,
)
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
)


def _source_candidate():
    mask = jnp.ones((1, 1, 3), dtype=bool)
    stimulus = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.05]),
        jnp.asarray([150.0]),
        mask,
    )
    runtime = SkeletalFiberBundlePlan(
        ("fiber-0",),
        3,
        jnp.asarray([10.0]),
        jnp.asarray([0.0]),
        stimulus,
        maximum_step_ms=0.05,
    ).prepare()
    return runtime.candidate(runtime.initialize(), 0.05)


def _material():
    architecture = UniformFiberArchitecturePlan("uniform-x").prepare(
        jnp.asarray((1.0, 0.0, 0.0))
    )
    return EngelhardtGasam2025Plan("cell-driven-test").prepare(
        EngelhardtGasam2025Parameters.published_multiload_fit(),
        architecture,
        0.0,
    )


def test_homogenized_crossbridge_driver_commits_both_routes_atomically():
    source = _source_candidate()
    material = _material()
    coupling = HomogenizedShortenGasamCouplingPlan(
        jnp.full((1, 3), 1.0 / 3.0),
        calibration_asset_id="source-specific-a2-anchors",
    ).prepare(ShortenGasamActivationCalibration(0.23, 0.24))
    candidate = coupling.candidate(source, material)
    committed = candidate.commit()

    assert bool(candidate.evidence.successful)
    assert 0.0 < candidate.evidence.prescribed_activation < 1.0
    assert bool(committed.committed)
    assert committed.fiber_state.time_ms == source.candidate_state.time_ms
    np.testing.assert_allclose(
        committed.material.state.activation,
        candidate.evidence.prescribed_activation,
    )
    assert committed.material.state.state_id != material.state.state_id
    assert candidate.evidence.force_owner == "engelhardt-gasam-2025"


def test_invalid_calibration_rolls_back_fiber_and_material():
    source = _source_candidate()
    material = _material()
    coupling = HomogenizedShortenGasamCouplingPlan(
        jnp.full((1, 3), 1.0 / 3.0),
        calibration_asset_id="invalid-a2-anchors",
    ).prepare(ShortenGasamActivationCalibration(0.23, 0.24))
    coupling = eqx.tree_at(
        lambda value: value.calibration.saturated_crossbridge_uM,
        coupling,
        jnp.asarray(0.23),
    )
    committed = coupling.candidate(source, material).commit()

    assert not bool(committed.committed)
    assert bool(committed.rollback_applied)
    np.testing.assert_array_equal(
        committed.fiber_state.values, source.source_state.values
    )
    np.testing.assert_array_equal(
        committed.material.state.activation, material.state.activation
    )
    np.testing.assert_array_equal(
        committed.material.state.state_id, material.state.state_id
    )
