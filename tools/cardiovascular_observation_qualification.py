#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Analytic qualification cases for cardiovascular observation operators."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications.cardiovascular.observations._electrograms import (
    ActivationTimePlan,
    ECGLeadFieldPlan,
    ElectricalGaugePlan,
    ExtracellularSourceDensity,
    FIRFilterPlan,
    TorsoObservationPlan,
)
from phydrax.applications.cardiovascular.observations._lge import (
    LGEObservationPlan,
    LGETissueState,
)
from phydrax.applications.cardiovascular.observations._metadata import (
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


@dataclass(frozen=True)
class ElectricalQualificationEvidence:
    gauge_shift_error_mv: float
    activation_error_ms: float
    censoring_preserved: bool
    reciprocity_residual: float
    lead_observation_error_mv: float
    successful: bool


@dataclass(frozen=True)
class PressureVolumeQualificationEvidence:
    pressure_reference_error_kpa: float
    volume_response_error_mm3: float
    flow_orientation_error_mm3_per_ms: float
    singleton_timebase_supported: bool
    external_work_mg_mm2_per_ms2: float
    expected_work_mg_mm2_per_ms2: float
    work_error: float
    work_scale_derivative_error: float
    closed: bool
    successful: bool


@dataclass(frozen=True)
class LGEQualificationEvidence:
    constant_psf_error: float
    constant_slice_error: float
    identity_motion_error: float
    zero_noise_error: float
    permutation_motion_error: float
    source_affine_exact: bool
    concentration_derivative: float
    successful: bool


@dataclass(frozen=True)
class CardiovascularObservationQualificationReport:
    scope: str
    electrical: ElectricalQualificationEvidence
    pressure_volume: PressureVolumeQualificationEvidence
    lge: LGEQualificationEvidence

    @property
    def passed(self) -> bool:
        return (
            self.electrical.successful
            and self.pressure_volume.successful
            and self.lge.successful
        )


def _electrical_qualification() -> ElectricalQualificationEvidence:
    timebase = TimeBase.uniform("qualification-electrical", 5, 1.0)
    gauge = ElectricalGaugePlan(
        ("ra", "la", "ll"),
        np.full((3,), 1.0 / 3.0),
        reference_id="average-reference",
    )
    potentials = jnp.asarray([[1.0, 2.0, 4.0], [2.0, 1.0, 5.0]])
    referenced, gauge_evidence = gauge.apply(potentials)
    shifted, _ = gauge.apply(potentials + 23.0)
    gauge_error = jnp.max(jnp.abs(referenced - shifted))

    voltage = jnp.asarray(
        [
            [-80.0, -80.0],
            [-60.0, -80.0],
            [20.0, -80.0],
            [0.0, -80.0],
            [-80.0, -80.0],
        ]
    )
    activation = ActivationTimePlan(timebase, threshold_mv=-30.0).evaluate(voltage)
    activation_error = jnp.abs(activation.activation_time_ms[0] - 1.375)
    censoring = activation.evidence.censored[1] & jnp.isnan(
        activation.activation_time_ms[1]
    )

    transfer = jnp.asarray([[1.0, 0.2], [0.1, 0.8], [-0.3, 0.4]])
    torso = TorsoObservationPlan(
        transfer,
        ("source-a", "source-b"),
        gauge,
        timebase,
        transfer_id="qualified-torso",
    )
    lead_matrix = jnp.asarray([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])
    reciprocal = (lead_matrix @ gauge.response.matrix @ transfer).T
    lead_plan = ECGLeadFieldPlan(
        torso,
        lead_matrix,
        ("I", "II"),
        reciprocal,
        FIRFilterPlan(jnp.ones((1,)), timebase, filter_id="identity"),
        lead_field_id="qualified-leads",
    )
    source = ExtracellularSourceDensity(
        jnp.asarray([[1.0, 2.0], [2.0, 1.0], [3.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
        timebase,
        ("source-a", "source-b"),
        unit="uA/mm2",
        source_id="qualified-source-density",
    )
    torso_result = torso.observe(source)
    lead_result = lead_plan.observe(source)
    lead_error = jnp.max(
        jnp.abs(lead_result.values_mv - torso_result.values_mv @ lead_matrix.T)
    )
    successful = (
        gauge_evidence.successful
        & (gauge_error < 2.0e-5)
        & (activation_error < 1.0e-6)
        & censoring
        & lead_result.evidence.successful
        & (lead_error < 1.0e-6)
    )
    return ElectricalQualificationEvidence(
        float(gauge_error),
        float(activation_error),
        bool(censoring),
        float(lead_result.evidence.reciprocity_residual),
        float(lead_error),
        bool(successful),
    )


def _pressure_volume_qualification() -> PressureVolumeQualificationEvidence:
    timebase = TimeBase.uniform("qualification-pv", 5, 1.0)
    pressure_observation = PressureObservationPlan(
        jnp.ones((1, 1)),
        ("chamber-pressure",),
        ("gauge-pressure",),
        5.0,
        timebase,
        reference_configuration="qualified catheter zero",
        observation_id="qualified-pressure",
    ).observe(jnp.asarray([[6.0], [7.0], [8.0], [7.0], [6.0]]))
    singleton_timebase = TimeBase.uniform("qualification-singleton", 1, 1.0)
    singleton_observation = PressureObservationPlan(
        jnp.ones((1, 1)),
        ("source",),
        ("channel",),
        0.0,
        singleton_timebase,
        reference_configuration="qualified singleton pressure",
        observation_id="qualified-singleton-pressure",
    ).observe(jnp.asarray([[2.0]]))
    singleton_supported = (
        singleton_observation.evidence.successful
        & ~singleton_observation.evidence.timebase.has_interval
    )
    pressure_error = jnp.max(
        jnp.abs(
            pressure_observation.pressure_kpa[:, 0]
            - jnp.asarray([1.0, 2.0, 3.0, 2.0, 1.0])
        )
    )
    volume_observation = VolumeObservationPlan(
        jnp.asarray([[1.0, 1.0]]),
        ("left-volume", "right-volume"),
        ("total-volume",),
        timebase,
        observation_id="qualified-volume",
    ).observe(jnp.ones((5, 2)))
    volume_error = jnp.max(jnp.abs(volume_observation.volume_mm3 - 2.0))
    flow_observation = FlowObservationPlan(
        jnp.eye(2),
        ("inlet-raw", "outlet-raw"),
        ("inlet", "outlet"),
        jnp.asarray([1.0, -1.0]),
        ("into chamber", "out of chamber"),
        timebase,
        observation_id="qualified-flow",
    ).observe(jnp.tile(jnp.asarray([2.0, -3.0]), (5, 1)))
    flow_error = jnp.max(
        jnp.abs(flow_observation.flow_mm3_per_ms - jnp.asarray([2.0, 3.0]))
    )
    plan = PressureVolumeLoopPlan(
        timebase,
        pressure_reference_kpa=0.0,
        reference_configuration="absolute chamber pressure",
        loop_id="qualified-rectangle",
    )
    pressure = jnp.asarray([1.0, 3.0, 3.0, 1.0, 1.0])
    volume = jnp.asarray([3.0, 3.0, 1.0, 1.0, 3.0])
    result = plan.evaluate(pressure, volume)
    expected = 4.0
    error = jnp.abs(result.external_work_mg_mm2_per_ms2 - expected)
    derivative = jax.grad(
        lambda scale: plan.evaluate(pressure * scale, volume).external_work_mg_mm2_per_ms2
    )(jnp.asarray(1.0))
    derivative_error = jnp.abs(derivative - expected)
    successful = (
        pressure_observation.evidence.successful
        & volume_observation.evidence.successful
        & flow_observation.evidence.successful
        & singleton_supported
        & result.evidence.successful
        & (pressure_error < 1.0e-6)
        & (volume_error < 1.0e-6)
        & (flow_error < 1.0e-6)
        & (error < 1.0e-6)
        & (derivative_error < 1.0e-6)
    )
    return PressureVolumeQualificationEvidence(
        float(pressure_error),
        float(volume_error),
        float(flow_error),
        bool(singleton_supported),
        float(result.external_work_mg_mm2_per_ms2),
        expected,
        float(error),
        float(derivative_error),
        bool(result.evidence.closed),
        bool(successful),
    )


def _affine() -> SpatialAffine:
    return SpatialAffine(
        np.eye(4),
        "qualification-voxel-index",
        SpatialFrame("qualification-patient", SpatialConvention.LPS),
    )


def _lge_plan(
    motion: np.ndarray, *, noise: float, acquisition_id: str
) -> LGEObservationPlan:
    point_spread = np.zeros((3, 3, 3))
    point_spread[1, 1, 1] = 1.0
    return LGEObservationPlan(
        (2, 2, 2),
        _affine(),
        point_spread,
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


def _lge_qualification() -> LGEQualificationEvidence:
    shape = (2, 2, 2)
    identity_plan = _lge_plan(np.eye(8), noise=0.0, acquisition_id="identity")
    constant_tissue = LGETissueState(
        jnp.full(shape, 900.0),
        jnp.full(shape, 0.15),
        jnp.full(shape, 1.2),
        jnp.full(shape, 0.2),
        identity_plan.spatial_affine,
    )
    identity = identity_plan.evaluate(constant_tissue, jr.key(31))
    psf_error = jnp.max(jnp.abs(identity.after_psf - identity.analytic_signal))
    slice_error = jnp.max(
        jnp.abs(identity.after_slice_profile - identity.analytic_signal)
    )
    motion_error = jnp.max(jnp.abs(identity.after_motion - identity.analytic_signal))
    noise_error = jnp.max(jnp.abs(identity.noisy_complex - identity.after_motion))

    varying_tissue = LGETissueState(
        jnp.full(shape, 900.0),
        jnp.full(shape, 0.15),
        jnp.arange(1.0, 9.0).reshape(shape),
        jnp.zeros(shape),
        identity_plan.spatial_affine,
    )
    permutation_plan = _lge_plan(np.eye(8)[::-1], noise=0.0, acquisition_id="permutation")
    moved = permutation_plan.evaluate(varying_tissue, jr.key(32))
    permutation_error = jnp.max(
        jnp.abs(
            moved.after_motion.reshape((-1,))
            - moved.after_slice_profile.reshape((-1,))[::-1]
        )
    )
    derivative = jax.grad(
        lambda concentration: jnp.sum(
            identity_plan.evaluate(
                LGETissueState(
                    jnp.full(shape, 900.0),
                    jnp.full(shape, concentration),
                    jnp.full(shape, 1.2),
                    jnp.zeros(shape),
                    identity_plan.spatial_affine,
                ),
                jr.key(33),
            ).magnitude
        )
    )(jnp.asarray(0.15))
    source_affine_exact = (
        constant_tissue.spatial_affine.affine_id == identity_plan.spatial_affine.affine_id
    )
    successful = (
        identity.evidence.successful
        & moved.evidence.successful
        & (psf_error < 1.0e-6)
        & (slice_error < 1.0e-6)
        & (motion_error < 1.0e-6)
        & (noise_error < 1.0e-6)
        & (permutation_error < 1.0e-6)
        & jnp.isfinite(derivative)
        & source_affine_exact
        & (jnp.abs(derivative) > 0.0)
    )
    return LGEQualificationEvidence(
        float(psf_error),
        float(slice_error),
        float(motion_error),
        float(noise_error),
        float(permutation_error),
        bool(source_affine_exact),
        float(derivative),
        bool(successful),
    )


def run_cardiovascular_observation_qualification(
    *, smoke: bool = False
) -> CardiovascularObservationQualificationReport:
    """Run bounded analytic limits without scanner or clinical-performance claims."""

    return CardiovascularObservationQualificationReport(
        scope="smoke" if smoke else "analytic-research-operator",
        electrical=_electrical_qualification(),
        pressure_volume=_pressure_volume_qualification(),
        lge=_lge_qualification(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_cardiovascular_observation_qualification(smoke=arguments.smoke)
    payload = asdict(report) | {"passed": report.passed}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(rendered + "\n")
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
