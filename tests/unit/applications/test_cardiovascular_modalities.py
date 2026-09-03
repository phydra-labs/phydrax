import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax.applications.cardiovascular.electrophysiology._activation import (
    ActivationObservationResult,
)
from phydrax.applications.cardiovascular.observations._electrograms import (
    ActionPotentialDurationPlan,
    ActivationTimePlan,
    ECGLeadFieldPlan,
    ElectricalGaugePlan,
    ElectrogramPlan,
    ExtracellularSourceDensity,
    FIRFilterPlan,
    TorsoObservationPlan,
)
from phydrax.applications.cardiovascular.observations._lge import (
    CategoricalLesionMap,
    LGEObservationPlan,
    LGETissueState,
)
from phydrax.applications.cardiovascular.observations._metadata import (
    ObservationRecord,
    SpatialAffine,
    SpatialConvention,
    SpatialFrame,
    TimeBase,
)
from phydrax.applications.cardiovascular.observations._pressure_volume import (
    FlowObservationPlan,
    PressureObservationPlan,
    PressureVolumeLoopPlan,
    VolumeObservationPlan,
)


def _spatial_affine():
    return SpatialAffine(
        np.eye(4),
        "voxel-index",
        SpatialFrame("patient", SpatialConvention.LPS),
    )


def _delta_psf():
    return np.pad(np.ones((1, 1, 1)), ((1, 1), (1, 1), (1, 1)))


def _lge_plan(*, motion=None, noise=0.0, acquisition_id="lge"):
    shape = (2, 2, 2)
    if motion is None:
        motion = np.eye(np.prod(shape))
    return LGEObservationPlan(
        shape,
        _spatial_affine(),
        _delta_psf(),
        np.asarray([0.25, 0.5, 0.25]),
        motion,
        inversion_time_ms=300.0,
        repetition_time_ms=1200.0,
        flip_angle_rad=0.3,
        inversion_efficiency=1.0,
        relaxivity_l_per_mmol_s=4.5,
        noise_standard_deviation=noise,
        acquisition_id=acquisition_id,
    )


def test_lat_apd_timing_censoring_and_record_boundary():
    timebase = TimeBase.uniform("ep-1ms", 8, 1.0)
    voltage = jnp.asarray(
        [
            [-80.0, -80.0],
            [-60.0, -80.0],
            [20.0, -80.0],
            [20.0, -80.0],
            [-10.0, -80.0],
            [-50.0, -80.0],
            [-80.0, -80.0],
            [-80.0, -80.0],
        ]
    )
    lat_plan = ActivationTimePlan(timebase, threshold_mv=-30.0)
    lat = lat_plan.evaluate(voltage)
    np.testing.assert_allclose(lat.activation_time_ms[0], 1.375)
    assert bool(lat.evidence.successful[0])
    assert bool(lat.evidence.censored[1])
    assert bool(jnp.isnan(lat.activation_time_ms[1]))

    record = ObservationRecord(
        "vm-record",
        "transmembrane-voltage",
        np.asarray(voltage),
        np.ones(voltage.shape, dtype=bool),
        "transmembrane_potential",
        "mV",
        timebase_id=timebase.timebase_id,
    )
    np.testing.assert_allclose(
        lat_plan.from_record(record).activation_time_ms[0],
        lat.activation_time_ms[0],
    )

    online = ActivationObservationResult(
        jnp.asarray([2, 5]),
        jnp.asarray([1.5, jnp.nan]),
        jnp.asarray([True, False]),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
        "online-observer",
    )
    adapted = lat_plan.consume_activation_observation(online)
    np.testing.assert_allclose(adapted.activation_time_ms[0], 1.5)
    assert bool(adapted.evidence.censored[1])
    assert type(adapted) is not type(online)

    apd = ActionPotentialDurationPlan(
        timebase,
        activation_threshold_mv=-30.0,
        resting_potential_mv=-80.0,
        repolarization_fraction=0.9,
    ).evaluate(voltage)
    np.testing.assert_allclose(apd.repolarization_level_mv[0], -70.0)
    np.testing.assert_allclose(apd.repolarization_time_ms[0], 5.6666665)
    np.testing.assert_allclose(apd.duration_ms[0], 4.2916665)
    assert bool(apd.evidence.successful[0])
    assert bool(apd.evidence.activation_censored[1])


def test_egm_gauge_electrode_filter_and_timebase_evidence():
    timebase = TimeBase.uniform("electrical-1ms", 5, 1.0)
    labels = ("e1", "e2", "e3")
    gauge = ElectricalGaugePlan(
        labels, jnp.asarray([1.0, 0.0, 0.0]), reference_id="e1-reference"
    )
    raw = jnp.asarray([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0]])
    referenced, evidence = gauge.apply(raw)
    shifted, shifted_evidence = gauge.apply(raw + 17.0)
    np.testing.assert_allclose(referenced, shifted, atol=1.0e-6)
    assert bool(evidence.successful & shifted_evidence.successful)

    filter_plan = FIRFilterPlan(
        jnp.asarray([0.25, 0.75]), timebase, filter_id="causal-smoother"
    )
    constant, filter_evidence = filter_plan.apply(jnp.full((5, 2), 3.0))
    np.testing.assert_allclose(constant, 3.0, atol=1.0e-6)
    np.testing.assert_allclose(filter_evidence.dc_gain, 1.0)

    transfer = jnp.asarray([[1.0, 0.2], [0.2, 1.0], [-0.5, 0.5]])
    source = ExtracellularSourceDensity(
        jnp.arange(10.0).reshape(5, 2),
        timebase,
        ("source-a", "source-b"),
        unit="uA/mm2",
        source_id="membrane-current",
    )
    plan = ElectrogramPlan(
        transfer,
        ("source-a", "source-b"),
        gauge,
        filter_plan,
        timebase,
        transfer_id="intracardiac-lead-field",
    )
    result = plan.observe(source)
    assert bool(result.evidence.successful)
    assert bool(result.evidence.electrode.every_electrode_responsive)
    assert bool(result.evidence.timebase.uniform)
    with pytest.raises(TypeError, match="sampled Vm"):
        plan.observe(jnp.zeros((5, 2)))


def test_ecg_lead_reciprocity_and_fail_closed_mismatch():
    timebase = TimeBase.uniform("ecg-2ms", 4, 2.0)
    electrodes = ("ra", "la", "ll")
    gauge = ElectricalGaugePlan(
        electrodes,
        jnp.full((3,), 1.0 / 3.0),
        reference_id="average-reference",
    )
    transfer = jnp.asarray([[1.0, 0.2], [0.1, 0.8], [-0.3, 0.4]])
    torso = TorsoObservationPlan(
        transfer,
        ("source-a", "source-b"),
        gauge,
        timebase,
        transfer_id="torso-transfer",
    )
    lead_matrix = jnp.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    direct = lead_matrix @ gauge.response.matrix @ transfer
    identity_filter = FIRFilterPlan(jnp.ones((1,)), timebase, filter_id="identity")
    reciprocal_plan = ECGLeadFieldPlan(
        torso,
        lead_matrix,
        ("I", "II"),
        direct.T,
        identity_filter,
        lead_field_id="limb-leads",
    )
    source = ExtracellularSourceDensity(
        jnp.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, -1.0], [1.0, 0.0]]),
        timebase,
        ("source-a", "source-b"),
        unit="uA/mm2",
        source_id="source",
    )
    torso_values = torso.observe(source).values_mv
    result = reciprocal_plan.observe(source)
    np.testing.assert_allclose(
        result.values_mv,
        torso_values @ lead_matrix.T,
        atol=1.0e-6,
    )
    assert bool(result.evidence.reciprocal & result.evidence.successful)

    mismatch = ECGLeadFieldPlan(
        torso,
        lead_matrix,
        ("I", "II"),
        direct.T + 0.1,
        identity_filter,
        lead_field_id="bad-reciprocity",
    ).observe(source)
    assert not bool(mismatch.evidence.reciprocal)
    assert not bool(mismatch.evidence.successful)
    np.testing.assert_array_equal(mismatch.values_mv, jnp.zeros_like(mismatch.values_mv))


def test_pressure_volume_flow_observations_and_pv_work_derivative():
    timebase = TimeBase.uniform("hemodynamics-1ms", 5, 1.0)
    pressure_plan = PressureObservationPlan(
        jnp.eye(2),
        ("lv", "ao"),
        ("lv-gauge", "ao-gauge"),
        jnp.asarray([5.0, 10.0]),
        timebase,
        reference_configuration="named catheter zero",
        observation_id="pressure",
    )
    pressure = jnp.asarray(
        [[6.0, 12.0], [7.0, 13.0], [8.0, 14.0], [7.0, 13.0], [6.0, 12.0]]
    )
    observed_pressure = pressure_plan.observe(pressure)
    np.testing.assert_allclose(observed_pressure.pressure_kpa[0], [1.0, 2.0])
    assert bool(observed_pressure.evidence.successful)

    singleton_timebase = TimeBase.uniform("singleton-pressure", 1, 1.0)
    singleton = PressureObservationPlan(
        jnp.ones((1, 1)),
        ("source",),
        ("channel",),
        0.0,
        singleton_timebase,
        reference_configuration="absolute pressure",
        observation_id="singleton-pressure",
    ).observe(jnp.asarray([[2.0]]))
    assert bool(singleton.evidence.successful)
    assert not bool(singleton.evidence.timebase.has_interval)

    record = ObservationRecord(
        "pressure-record",
        "pressure",
        np.asarray(pressure),
        np.ones(pressure.shape, dtype=bool),
        "pressure",
        "kPa",
        timebase_id=timebase.timebase_id,
    )
    np.testing.assert_allclose(
        pressure_plan.from_record(record).pressure_kpa,
        observed_pressure.pressure_kpa,
    )

    volume_plan = VolumeObservationPlan(
        jnp.asarray([[1.0, 1.0]]),
        ("left", "right"),
        ("total",),
        timebase,
        observation_id="volume",
    )
    volume_result = volume_plan.observe(jnp.ones((5, 2)))
    np.testing.assert_allclose(volume_result.volume_mm3, 2.0)

    flow_plan = FlowObservationPlan(
        jnp.eye(2),
        ("mitral-raw", "aortic-raw"),
        ("mitral", "aortic"),
        jnp.asarray([1.0, -1.0]),
        ("into LV", "out of LV"),
        timebase,
        observation_id="flow",
    )
    flow_result = flow_plan.observe(jnp.tile(jnp.asarray([2.0, -3.0]), (5, 1)))
    np.testing.assert_allclose(flow_result.flow_mm3_per_ms[0], [2.0, 3.0])
    assert bool(flow_result.evidence.successful)

    loop_plan = PressureVolumeLoopPlan(
        timebase,
        pressure_reference_kpa=0.0,
        reference_configuration="absolute chamber pressure",
        loop_id="rectangular-loop",
    )
    loop_pressure = jnp.asarray([1.0, 3.0, 3.0, 1.0, 1.0])
    loop_volume = jnp.asarray([3.0, 3.0, 1.0, 1.0, 3.0])
    loop = loop_plan.evaluate(loop_pressure, loop_volume)
    np.testing.assert_allclose(loop.line_integral_kpa_mm3, -4.0)
    np.testing.assert_allclose(loop.external_work_mg_mm2_per_ms2, 4.0)
    np.testing.assert_allclose(loop.external_work_mj, 0.004)
    assert bool(loop.evidence.closed & loop.evidence.counterclockwise)
    referenced_loop = PressureVolumeLoopPlan(
        timebase,
        pressure_reference_kpa=8.0,
        reference_configuration="gauge shifted by 8 kPa",
        loop_id="shifted-rectangular-loop",
    ).evaluate(loop_pressure + 8.0, loop_volume)
    np.testing.assert_allclose(
        referenced_loop.external_work_mg_mm2_per_ms2,
        loop.external_work_mg_mm2_per_ms2,
    )
    pressure_record = ObservationRecord(
        "pv-pressure",
        "pressure",
        np.asarray(loop_pressure),
        np.ones((5,), dtype=bool),
        "pressure",
        "kPa",
        timebase_id=timebase.timebase_id,
    )
    volume_record = ObservationRecord(
        "pv-volume",
        "volume",
        np.asarray(loop_volume),
        np.ones((5,), dtype=bool),
        "volume",
        "mm3",
        timebase_id=timebase.timebase_id,
    )
    np.testing.assert_allclose(
        loop_plan.from_records(
            pressure_record, volume_record
        ).external_work_mg_mm2_per_ms2,
        4.0,
    )
    open_loop = loop_plan.evaluate(loop_pressure, loop_volume.at[-1].set(2.0))
    assert not bool(open_loop.evidence.successful)
    np.testing.assert_allclose(open_loop.external_work_mg_mm2_per_ms2, 0.0)
    derivative = jax.grad(
        lambda scale: (
            loop_plan.evaluate(
                scale * loop_pressure, loop_volume
            ).external_work_mg_mm2_per_ms2
        )
    )(jnp.asarray(1.0))
    np.testing.assert_allclose(derivative, 4.0)


def test_lge_constant_noise_motion_limits_and_fixed_map_derivative():
    shape = (2, 2, 2)
    plan = _lge_plan()
    tissue = LGETissueState(
        jnp.full(shape, 900.0),
        jnp.full(shape, 0.15),
        jnp.full(shape, 1.2),
        jnp.full(shape, 0.2),
        plan.spatial_affine,
    )
    result = plan.evaluate(tissue, jr.key(4))
    np.testing.assert_allclose(result.after_psf, result.analytic_signal, atol=1.0e-6)
    np.testing.assert_allclose(
        result.after_slice_profile, result.analytic_signal, atol=1.0e-6
    )
    np.testing.assert_allclose(result.after_motion, result.analytic_signal, atol=1.0e-6)
    np.testing.assert_allclose(result.noisy_complex, result.after_motion, atol=1.0e-6)
    np.testing.assert_allclose(
        result.magnitude, jnp.abs(result.after_motion), atol=1.0e-6
    )
    assert bool(result.evidence.fixed_motion_map & result.evidence.successful)

    density = jnp.arange(1.0, 9.0).reshape(shape)
    varying = LGETissueState(
        jnp.full(shape, 900.0),
        jnp.full(shape, 0.15),
        density,
        jnp.zeros(shape),
        plan.spatial_affine,
    )
    reverse = np.eye(8)[::-1]
    moved = _lge_plan(motion=reverse, acquisition_id="reverse-motion").evaluate(
        varying, jr.key(5)
    )
    np.testing.assert_allclose(
        moved.after_motion.reshape(-1),
        moved.after_slice_profile.reshape(-1)[::-1],
        atol=1.0e-6,
    )

    noisy_plan = _lge_plan(noise=0.2, acquisition_id="complex-noise")
    noise_key = jr.key(6)
    noisy = noisy_plan.evaluate(tissue, noise_key)
    components = jr.normal(noise_key, (2,) + shape, dtype=tissue.native_t1_ms.dtype)
    expected_noise = 0.2 / np.sqrt(2.0) * (components[0] + 1j * components[1])
    np.testing.assert_allclose(
        noisy.noisy_complex - noisy.after_motion, expected_noise, atol=1.0e-6
    )
    np.testing.assert_allclose(noisy.magnitude, jnp.abs(noisy.noisy_complex))

    derivative = jax.grad(
        lambda concentration: jnp.sum(
            plan.evaluate(
                LGETissueState(
                    jnp.full(shape, 900.0),
                    jnp.full(shape, concentration),
                    jnp.full(shape, 1.2),
                    jnp.zeros(shape),
                    plan.spatial_affine,
                ),
                jr.key(7),
            ).magnitude
        )
    )(jnp.asarray(0.15))
    assert bool(jnp.isfinite(derivative))
    assert float(jnp.abs(derivative)) > 0.0

    mismatched_affine = SpatialAffine(
        np.eye(4),
        "different-voxel-index",
        SpatialFrame("patient", SpatialConvention.LPS),
    )
    mismatched_tissue = LGETissueState(
        jnp.full(shape, 900.0),
        jnp.full(shape, 0.15),
        jnp.full(shape, 1.2),
        jnp.zeros(shape),
        mismatched_affine,
    )
    with pytest.raises(ValueError, match="spatial affines must match exactly"):
        plan.evaluate(mismatched_tissue, jr.key(8))

    lesion = CategoricalLesionMap(
        jnp.asarray([[[0, 1], [0, 1]], [[0, 0], [1, 1]]]),
        plan.spatial_affine,
        ("background", "lesion"),
        annotation_id="independent-labels",
    )
    assert lesion.map_id != plan.plan_id
    np.testing.assert_array_equal(lesion.labels > 0, lesion.labels == 1)
