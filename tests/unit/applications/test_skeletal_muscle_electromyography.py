#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.applications import skeletal_muscle
from phydrax.applications.skeletal_muscle import electromyography
from phydrax.applications.skeletal_muscle.electromyography import (
    MotorUnitActionPotentialTemplatePlan,
    PetersenRostalski2019PlanarConductorPlan,
    PlanarConductorParameters,
)


def test_skeletal_muscle_parent_exports_electromyography_namespace():
    assert skeletal_muscle.electromyography is electromyography
    assert "electromyography" in skeletal_muscle.__all__


def test_event_template_superposition_mask_and_fractional_delay():
    template = jnp.asarray([[[0.0, 1.0, 2.0, 1.0, 0.0]]])
    prepared = MotorUnitActionPotentialTemplatePlan(
        template,
        0.001,
        0,
        ("unit-0",),
        ("channel-0",),
        template_source_id="licensed-explicit-test-template",
    ).prepare()
    times = jnp.arange(8) * 0.001
    single = prepared.synthesize(
        jnp.asarray(((0.0,),)), jnp.asarray(((True,),)), times
    )
    double = prepared.synthesize(
        jnp.asarray(((0.0, 0.002),)), jnp.asarray(((True, True),)), times
    )
    masked = prepared.synthesize(
        jnp.asarray(((0.0, 0.002),)), jnp.asarray(((True, False),)), times
    )
    half = prepared.synthesize(
        jnp.asarray(((0.0005,),)), jnp.asarray(((True,),)), times
    )

    assert bool(single.evidence.successful)
    np.testing.assert_allclose(single.voltage_V[0, :5], (0.0, 1.0, 2.0, 1.0, 0.0))
    np.testing.assert_allclose(masked.voltage_V, single.voltage_V)
    np.testing.assert_allclose(double.voltage_V[0, 2], 2.0)
    np.testing.assert_allclose(half.voltage_V[0, 1], 0.5)


def test_event_template_retains_aligned_final_nonzero_sample():
    prepared = MotorUnitActionPotentialTemplatePlan(
        jnp.asarray([[[1.0, 2.0, 3.0]]]),
        1.0,
        0,
        ("unit-0",),
        ("channel-0",),
        template_source_id="aligned-final-sample-test-template",
    ).prepare()

    result = prepared.synthesize(
        jnp.asarray(((0.0,),)),
        jnp.asarray(((True,),)),
        jnp.arange(3.0),
    )

    np.testing.assert_allclose(result.voltage_V[0], (1.0, 2.0, 3.0))
    assert bool(result.evidence.template_support_complete)
    assert bool(result.evidence.successful)


def test_template_support_is_incomplete_when_any_active_event_is_clipped():
    prepared = MotorUnitActionPotentialTemplatePlan(
        jnp.asarray([[[1.0, 2.0, 3.0]]]),
        1.0,
        1,
        ("unit-0",),
        ("channel-0",),
        template_source_id="clipped-event-test-template",
    ).prepare()
    times = jnp.arange(3.0)
    events = jnp.asarray(((1.0, 0.0),))

    complete = prepared.synthesize(
        events,
        jnp.asarray(((True, False),)),
        times,
    )
    clipped = prepared.synthesize(
        events,
        jnp.asarray(((True, True),)),
        times,
    )

    assert bool(complete.evidence.template_support_complete)
    assert bool(complete.evidence.successful)
    assert not bool(clipped.evidence.template_support_complete)
    assert not bool(clipped.evidence.successful)


def _frequencies(count=8, spacing=0.005):
    return 2.0 * jnp.pi * jnp.fft.fftfreq(count, d=spacing)


def _conductor(depth=-0.01, *, muscle_longitudinal_conductivity=0.5):
    frequency = _frequencies()
    parameters = PlanarConductorParameters(
        muscle_longitudinal_conductivity,
        0.1,
        0.04,
        0.2,
        0.003,
        0.001,
        depth,
    )
    return PetersenRostalski2019PlanarConductorPlan(
        frequency,
        frequency,
        jnp.ones((frequency.size, frequency.size), dtype=jnp.complex128),
        jnp.asarray(((0.0, 0.0), (0.01, 0.0))),
        jnp.asarray((1.0, -1.0)),
        parameters,
    )


def test_planar_conductor_plan_identity_includes_physical_parameters():
    baseline = _conductor()
    different_depth = _conductor(-0.02)
    different_conductivity = _conductor(muscle_longitudinal_conductivity=0.6)

    assert len(
        {baseline.plan_id, different_depth.plan_id, different_conductivity.plan_id}
    ) == 3


def _neutral_source():
    source = jnp.zeros((8, 8), dtype=jnp.complex128)
    return source.at[1, 0].set(1.0).at[-1, 0].set(1.0)


def test_planar_surface_conductor_zero_mode_reality_and_depth_attenuation():
    source = _neutral_source()
    shallow = _conductor(-0.005)
    deep = _conductor(-0.02)
    shallow_result = shallow.evaluate(source)
    deep_result = deep.evaluate(source)

    assert bool(shallow_result.evidence.successful)
    assert shallow_result.evidence.real_signal_residual <= shallow.zero_tolerance
    assert shallow.transfer_function_V_per_A()[0, 0] == 0.0
    assert jnp.max(jnp.abs(deep_result.surface_voltage_V)) < jnp.max(
        jnp.abs(shallow_result.surface_voltage_V)
    )
    compiled = eqx.filter_jit(shallow.evaluate)(source)
    np.testing.assert_allclose(compiled.surface_voltage_V, shallow_result.surface_voltage_V)


def test_planar_conductor_rejects_non_neutral_source_and_montage():
    source = _neutral_source().at[0, 0].set(1.0)
    result = _conductor().evaluate(source)
    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.source_charge_neutral)

    frequency = _frequencies()
    nonneutral = PetersenRostalski2019PlanarConductorPlan(
        frequency,
        frequency,
        jnp.ones((8, 8), dtype=jnp.complex128),
        jnp.asarray(((0.0, 0.0),)),
        jnp.asarray((1.0,)),
        _conductor().parameters,
    ).evaluate(_neutral_source())
    assert not bool(nonneutral.evidence.successful)
    assert not bool(nonneutral.evidence.montage_charge_neutral)
