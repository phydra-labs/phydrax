#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Coherent thin-screen, multislice, and Helmholtz-continuation Schlieren."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optics.wave import (
    AngularSpectrumPlan,
    ideal_square_law,
    PlaneFieldSpace,
    PreparedAngularSpectrum,
    ScalarPlaneField,
    ScalarThinTransmission,
)


class RefractivePhaseEvidence(StrictModule, NonTrainableState):
    maximum_absolute_phase: Array
    maximum_neighbor_phase_increment: Array
    adequately_sampled: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class RefractivePhaseScreenPlan:
    space: PlaneFieldSpace
    medium_wavenumber: float
    maximum_phase_increment: float = float(np.pi)
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.space, PlaneFieldSpace):
            raise TypeError("space must be PlaneFieldSpace.")
        wavenumber = float(self.medium_wavenumber)
        increment = float(self.maximum_phase_increment)
        if (
            not np.isfinite(wavenumber)
            or wavenumber <= 0.0
            or not np.isfinite(increment)
            or increment <= 0.0
        ):
            raise ValueError(
                "Wavenumber and phase-increment bound must be finite and positive."
            )
        object.__setattr__(self, "medium_wavenumber", wavenumber)
        object.__setattr__(self, "maximum_phase_increment", increment)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "refractive-phase-screen-plan",
                    "space": self.space.space_id,
                    "wavenumber": wavenumber,
                    "maximum_increment": increment,
                }
            ),
        )

    def evaluate(
        self, optical_path_difference: ArrayLike, /
    ) -> tuple[ScalarThinTransmission, RefractivePhaseEvidence]:
        path = jnp.asarray(optical_path_difference)
        if path.shape != self.space.shape:
            raise ValueError(
                f"optical_path_difference must have shape {self.space.shape}."
            )
        phase = self.medium_wavenumber * path
        increments = jnp.maximum(
            jnp.max(jnp.abs(jnp.diff(phase, axis=0))),
            jnp.max(jnp.abs(jnp.diff(phase, axis=1))),
        )
        finite = jnp.all(jnp.isfinite(phase))
        adequate = increments <= self.maximum_phase_increment
        transmission = ScalarThinTransmission(
            self.space,
            jnp.exp(1j * phase),
            operator_id=f"{self.plan_id}:transmission",
        )
        evidence = RefractivePhaseEvidence(
            jnp.max(jnp.abs(phase)),
            increments,
            adequate,
            finite,
            finite & adequate,
            self.plan_id,
        )
        return transmission, evidence


class WaveSchlierenEvidence(StrictModule, NonTrainableState):
    input_power: Array
    detector_power: Array
    rejected_power: Array
    maximum_leakage_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class WaveSchlierenResult(StrictModule):
    detector_field: ScalarPlaneField
    detector_intensity: object
    evidence: WaveSchlierenEvidence


class WaveSchlierenPlan(StrictModule, NonTrainableState):
    object_screen: RefractivePhaseScreenPlan = eqx.field(static=True)
    object_to_filter: PreparedAngularSpectrum
    filter_to_detector: PreparedAngularSpectrum
    filter_transmission: ScalarThinTransmission
    object_distance: float = eqx.field(static=True)
    detector_distance: float = eqx.field(static=True)
    medium_wavenumber: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        object_screen: RefractivePhaseScreenPlan,
        filter_transmission: ScalarThinTransmission,
        /,
        *,
        object_distance: float,
        detector_distance: float,
        padding: int,
        maximum_leakage_fraction: float = 1.0e-5,
    ):
        if object_screen.space.space_id != filter_transmission.space.space_id:
            raise ValueError(
                "Object and filter must use the same prepared plane support."
            )
        first = AngularSpectrumPlan(
            padding, maximum_leakage_fraction=maximum_leakage_fraction
        ).prepare(object_screen.space)
        second = AngularSpectrumPlan(
            padding, maximum_leakage_fraction=maximum_leakage_fraction
        ).prepare(object_screen.space)
        object_distance_ = float(object_distance)
        detector_distance_ = float(detector_distance)
        if object_distance_ < 0.0 or detector_distance_ < 0.0:
            raise ValueError("Propagation distances must be non-negative.")
        self.object_screen = object_screen
        self.object_to_filter = first
        self.filter_to_detector = second
        self.filter_transmission = filter_transmission
        self.object_distance = object_distance_
        self.detector_distance = detector_distance_
        self.medium_wavenumber = object_screen.medium_wavenumber
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-schlieren-plan",
                "screen": object_screen.plan_id,
                "filter": filter_transmission.operator_id,
                "object_distance": object_distance_,
                "detector_distance": detector_distance_,
                "padding": padding,
            }
        )

    def evaluate(
        self, incident: ScalarPlaneField, optical_path_difference: ArrayLike, /
    ) -> WaveSchlierenResult:
        if incident.space.space_id != self.object_screen.space.space_id:
            raise ValueError("Incident field belongs to another plane support.")
        screen, screen_evidence = self.object_screen.evaluate(optical_path_difference)
        object_field = screen.apply(incident)
        at_filter = self.object_to_filter.execute(
            object_field, self.object_distance, self.medium_wavenumber
        )
        filtered = self.filter_transmission.apply(at_filter.field)
        at_detector = self.filter_to_detector.execute(
            filtered, self.detector_distance, self.medium_wavenumber
        )
        intensity = ideal_square_law(at_detector.field)
        input_power = jnp.sum(jnp.abs(incident.values) ** 2 * incident.space.area_weights)
        detector_power = jnp.sum(intensity.values * intensity.space.area_weights)
        filtered_power = jnp.sum(
            jnp.abs(filtered.values) ** 2 * filtered.space.area_weights
        )
        rejected = jnp.maximum(at_filter.evidence.retained_energy - filtered_power, 0.0)
        leakage = jnp.maximum(
            at_filter.evidence.leakage_fraction, at_detector.evidence.leakage_fraction
        )
        finite = (
            screen_evidence.finite
            & at_filter.evidence.finite
            & at_detector.evidence.finite
            & jnp.isfinite(detector_power)
        )
        successful = (
            screen_evidence.successful
            & at_filter.successful
            & at_detector.successful
            & finite
        )
        evidence = WaveSchlierenEvidence(
            input_power,
            detector_power,
            rejected,
            leakage,
            finite,
            successful,
            self.plan_id,
        )
        return WaveSchlierenResult(at_detector.field, intensity, evidence)


class MultisliceEvidence(StrictModule, NonTrainableState):
    input_power: Array
    output_power: Array
    relative_power_drift: Array
    maximum_slice_phase: Array
    maximum_leakage_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MultisliceResult(StrictModule):
    field: ScalarPlaneField
    evidence: MultisliceEvidence


class MultisliceRefractivePlan(StrictModule, NonTrainableState):
    propagation: PreparedAngularSpectrum
    slice_thicknesses: Array
    medium_wavenumber: float = eqx.field(static=True)
    maximum_phase_per_slice: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        slice_thicknesses: ArrayLike,
        medium_wavenumber: float,
        /,
        *,
        padding: int,
        maximum_phase_per_slice: float = np.pi / 2.0,
        maximum_leakage_fraction: float = 1.0e-5,
    ):
        thicknesses = np.asarray(slice_thicknesses, dtype=float)
        if (
            thicknesses.ndim != 1
            or thicknesses.size < 1
            or not np.all(np.isfinite(thicknesses))
            or np.any(thicknesses <= 0.0)
        ):
            raise ValueError("slice_thicknesses must be a finite positive vector.")
        k = float(medium_wavenumber)
        phase_limit = float(maximum_phase_per_slice)
        if (
            not np.isfinite(k)
            or k <= 0.0
            or not np.isfinite(phase_limit)
            or phase_limit <= 0.0
        ):
            raise ValueError("medium_wavenumber and phase limit must be positive.")
        self.propagation = AngularSpectrumPlan(
            padding, maximum_leakage_fraction=maximum_leakage_fraction
        ).prepare(space)
        self.slice_thicknesses = jnp.asarray(thicknesses)
        self.medium_wavenumber = k
        self.maximum_phase_per_slice = phase_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multislice-refractive-plan",
                "space": space.space_id,
                "thicknesses": array_tree_fingerprint(thicknesses),
                "wavenumber": k,
                "padding": padding,
                "phase_limit": phase_limit,
            }
        )

    def evaluate(
        self, incident: ScalarPlaneField, refractive_index_perturbation: ArrayLike, /
    ) -> MultisliceResult:
        perturbation = jnp.asarray(refractive_index_perturbation)
        expected = (self.slice_thicknesses.size,) + incident.space.shape
        if perturbation.shape != expected:
            raise ValueError(f"refractive_index_perturbation must have shape {expected}.")
        input_power = jnp.sum(jnp.abs(incident.values) ** 2 * incident.space.area_weights)

        def step(field, inputs):
            delta_n, thickness = inputs
            half = self.propagation.execute(
                field, 0.5 * thickness, self.medium_wavenumber
            )
            phase = self.medium_wavenumber * delta_n * thickness
            transmission = jnp.exp(1j * phase)
            after_phase = ScalarPlaneField(
                field.space,
                half.field.values * transmission,
                field.angular_frequency,
                half.field.longitudinal_coordinate,
            )
            output = self.propagation.execute(
                after_phase, 0.5 * thickness, self.medium_wavenumber
            )
            leakage = jnp.maximum(
                half.evidence.leakage_fraction, output.evidence.leakage_fraction
            )
            successful = (
                half.successful & output.successful & jnp.all(jnp.isfinite(phase))
            )
            return output.field, (jnp.max(jnp.abs(phase)), leakage, successful)

        output, records = jax.lax.scan(
            step, incident, (perturbation, self.slice_thicknesses)
        )
        output_power = jnp.sum(jnp.abs(output.values) ** 2 * output.space.area_weights)
        drift = jnp.abs(output_power - input_power) / jnp.maximum(
            input_power, jnp.finfo(output_power.dtype).tiny
        )
        maximum_phase = jnp.max(records[0])
        maximum_leakage = jnp.max(records[1])
        finite = jnp.all(jnp.isfinite(output.values)) & jnp.isfinite(drift)
        successful = (
            jnp.all(records[2]) & finite & (maximum_phase <= self.maximum_phase_per_slice)
        )
        return MultisliceResult(
            output,
            MultisliceEvidence(
                input_power,
                output_power,
                drift,
                maximum_phase,
                maximum_leakage,
                finite,
                successful,
                self.plan_id,
            ),
        )


class HelmholtzContinuationEvidence(StrictModule, NonTrainableState):
    residual_norm: Array
    relative_residual: Array
    finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class HelmholtzContinuationResult(StrictModule):
    field: ScalarPlaneField
    evidence: HelmholtzContinuationEvidence


class ScalarHelmholtzContinuationPlan(StrictModule, NonTrainableState):
    space: PlaneFieldSpace
    green_symbol: Array
    background_wavenumber: float = eqx.field(static=True)
    iteration_count: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        background_wavenumber: float,
        /,
        *,
        damping: float,
        iteration_count: int,
        relative_tolerance: float = 1.0e-5,
    ):
        if space.topology != "periodic-cell":
            raise ValueError(
                "The initial Helmholtz continuation route requires periodic-cell support."
            )
        k = float(background_wavenumber)
        damping_ = float(damping)
        iterations = int(iteration_count)
        tolerance = float(relative_tolerance)
        if k <= 0.0 or damping_ <= 0.0 or iterations < 1 or tolerance <= 0.0:
            raise ValueError("Helmholtz parameters must be positive.")
        axes = space.coordinate_axes
        spacing = tuple(float(np.mean(np.diff(np.asarray(axis)))) for axis in axes)
        frequencies = tuple(
            2.0 * np.pi * np.fft.fftfreq(size, d=step)
            for size, step in zip(space.shape, spacing, strict=True)
        )
        laplacian = frequencies[0][:, None] ** 2 + frequencies[1][None, :] ** 2
        symbol = 1.0 / (k * k - laplacian + 1j * damping_)
        self.space = space
        self.green_symbol = jnp.asarray(symbol)
        self.background_wavenumber = k
        self.iteration_count = iterations
        self.relative_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-helmholtz-continuation",
                "space": space.space_id,
                "wavenumber": k,
                "damping": damping_,
                "iterations": iterations,
                "tolerance": tolerance,
            }
        )

    def solve(
        self,
        source: ArrayLike,
        susceptibility: ArrayLike,
        angular_frequency: ArrayLike,
        /,
    ) -> HelmholtzContinuationResult:
        source_ = jnp.asarray(source, dtype=jnp.complex64)
        susceptibility_ = jnp.asarray(susceptibility)
        if source_.shape != self.space.shape or susceptibility_.shape != self.space.shape:
            raise ValueError("source and susceptibility must match the Helmholtz space.")

        def apply_green(value):
            return jnp.fft.ifft2(jnp.fft.fft2(value) * self.green_symbol)

        incident = apply_green(source_)

        def step(field, _):
            return incident - apply_green(
                (self.background_wavenumber**2) * susceptibility_ * field
            ), None

        field, _ = jax.lax.scan(step, incident, xs=None, length=self.iteration_count)
        residual = (
            field
            - incident
            + apply_green((self.background_wavenumber**2) * susceptibility_ * field)
        )
        norm = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2))
        reference = jnp.maximum(
            jnp.sqrt(jnp.sum(jnp.abs(field) ** 2)), jnp.finfo(field.real.dtype).tiny
        )
        relative = norm / reference
        finite = jnp.all(jnp.isfinite(field)) & jnp.isfinite(relative)
        converged = relative <= self.relative_tolerance
        output = ScalarPlaneField(self.space, field, angular_frequency, 0.0)
        return HelmholtzContinuationResult(
            output,
            HelmholtzContinuationEvidence(
                norm, relative, finite, converged, finite & converged, self.plan_id
            ),
        )


__all__ = [
    "HelmholtzContinuationEvidence",
    "HelmholtzContinuationResult",
    "MultisliceEvidence",
    "MultisliceRefractivePlan",
    "MultisliceResult",
    "RefractivePhaseEvidence",
    "RefractivePhaseScreenPlan",
    "ScalarHelmholtzContinuationPlan",
    "WaveSchlierenEvidence",
    "WaveSchlierenPlan",
    "WaveSchlierenResult",
]
