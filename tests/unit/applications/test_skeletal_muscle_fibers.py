#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
    SkeletalFiberBundleStatus,
)
from phydrax.applications.skeletal_muscle.fibers._bundle import _FiberBundleDrift


def _schedule():
    mask = jnp.zeros((1, 2, 5), dtype=bool).at[0, 0, 0].set(True)
    return PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.05]),
        jnp.asarray([150.0]),
        mask,
    )


def _runtime(diffusivity=(0.1, 0.1)):
    return SkeletalFiberBundlePlan(
        ("fiber-a", "fiber-b"),
        5,
        jnp.asarray((10.0, 12.0)),
        jnp.asarray(diffusivity),
        _schedule(),
        maximum_step_ms=0.1,
    ).prepare()


def test_prescribed_stimulus_support_is_left_closed_right_open():
    schedule = _schedule()
    at_start = schedule.current(0.0)
    before_end = schedule.current(0.049999)
    at_end = schedule.current(0.05)

    assert at_start[0, 0] == 150.0
    assert before_end[0, 0] == 150.0
    assert not bool(jnp.any(at_end))
    assert not bool(jnp.any(at_start[1]))
    np.testing.assert_allclose(schedule.event_boundaries_ms(), [0.0, 0.05])


def test_uniform_membrane_field_has_zero_no_flux_diffusion_increment():
    diffusive = _runtime((0.2, 0.3))
    nondiffusive = _runtime((0.0, 0.0))
    state = diffusive.initialize().values.at[..., 0].set(-70.0)
    diffusive_rate = _FiberBundleDrift(diffusive.model, diffusive.plan)(
        jnp.asarray(1.0), state, None
    )
    local_rate = _FiberBundleDrift(nondiffusive.model, nondiffusive.plan)(
        jnp.asarray(1.0), state, None
    )
    np.testing.assert_allclose(diffusive_rate, local_rate, rtol=2.0e-12, atol=2.0e-12)


def test_event_aligned_stimulated_step_advances_complete_bundle():
    runtime = _runtime()
    source = runtime.initialize()
    candidate = runtime.candidate(source, 0.05)

    assert bool(candidate.evidence.successful)
    assert bool(candidate.evidence.solver_successful)
    committed = candidate.commit()
    assert committed.time_ms == 0.05
    assert committed.values.shape == (2, 5, 56)
    assert committed.values[0, 0, 0] > source.values[0, 0, 0]
    assert candidate.output.cytosolic_calcium_uM.shape == (2, 5, 2)
    assert candidate.output.force_bearing_crossbridge_uM.shape == (2, 5)


def test_crossing_stimulus_event_rolls_back_whole_bundle():
    runtime = _runtime()
    source = runtime.initialize()
    candidate = runtime.candidate(source, 0.1)

    assert not bool(candidate.evidence.successful)
    assert int(candidate.evidence.status) & int(
        SkeletalFiberBundleStatus.STIMULUS_EVENT_CROSSED
    )
    committed = candidate.commit()
    assert committed.time_ms == source.time_ms
    np.testing.assert_array_equal(committed.values, source.values)
