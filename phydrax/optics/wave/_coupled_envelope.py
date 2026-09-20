#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Vector, multimode, spatiotemporal, quadratic, and plasma-audited envelopes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...lifecycle import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)


CoupledEnvelopeMethod: TypeAlias = Literal["fixed-symmetric", "adaptive-step-doubling"]


class CoupledEnvelopePlan(StrictModule):
    """Fixed-grid vector/multimode GNLSE with χ², Kerr, Raman, and diffraction."""

    time_points: Array
    x_points: Array
    y_points: Array
    carrier_frequencies: Array
    dispersion: Array
    loss: Array
    diffraction: Array
    nonlinear_overlap: Array
    quadratic_coupling: Array
    phase_mismatch: Array
    raman_frequency_response: Array
    raman_fraction: float = eqx.field(static=True)
    self_steepening: Array
    propagation_distance: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    method: CoupledEnvelopeMethod = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_points: ArrayLike,
        x_points: ArrayLike,
        y_points: ArrayLike,
        carrier_frequencies: ArrayLike,
        dispersion: ArrayLike,
        loss: ArrayLike,
        diffraction: ArrayLike,
        nonlinear_overlap: ArrayLike,
        quadratic_coupling: ArrayLike,
        phase_mismatch: ArrayLike,
        raman_frequency_response: ArrayLike,
        self_steepening: ArrayLike,
        /,
        *,
        raman_fraction: float,
        propagation_distance: float,
        step_size: float,
        method: CoupledEnvelopeMethod = "fixed-symmetric",
        relative_tolerance: float = 1e-6,
        minimum_step_size: float = 1e-8,
        maximum_steps: int = 100_000,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        time = np.asarray(time_points, dtype=np.float64)
        x = np.asarray(x_points, dtype=np.float64)
        y = np.asarray(y_points, dtype=np.float64)
        carrier = np.asarray(carrier_frequencies, dtype=np.float64)
        modes = carrier.size
        dispersion_ = np.asarray(dispersion, dtype=np.float64)
        loss_ = np.asarray(loss, dtype=np.float64)
        diffraction_ = np.asarray(diffraction, dtype=np.float64)
        overlap = np.asarray(nonlinear_overlap, dtype=np.complex128)
        quadratic = np.asarray(quadratic_coupling, dtype=np.complex128)
        mismatch = np.asarray(phase_mismatch, dtype=np.float64)
        raman = np.asarray(raman_frequency_response, dtype=np.complex128)
        steepening = np.asarray(self_steepening, dtype=np.float64)
        for name, values, minimum in (
            ("time_points", time, 16),
            ("x_points", x, 1),
            ("y_points", y, 1),
        ):
            if values.ndim != 1 or values.size < minimum:
                raise ValueError(f"{name} has insufficient one-dimensional support.")
            if values.size > 1 and (
                np.any(np.diff(values) <= 0.0)
                or not np.allclose(np.diff(values), np.diff(values)[0])
            ):
                raise ValueError(f"{name} must be increasing and uniform.")
        if modes < 1 or np.any(carrier <= 0.0):
            raise ValueError("Carrier frequencies must be positive and non-empty.")
        frequency_count = time.size
        if (
            dispersion_.shape != (modes, frequency_count)
            or loss_.shape != dispersion_.shape
        ):
            raise ValueError("Dispersion and loss must have shape (modes, time samples).")
        if diffraction_.shape != (modes,) or steepening.shape != (modes,):
            raise ValueError(
                "Diffraction and self-steepening require one value per mode."
            )
        if overlap.shape != (modes, modes, modes, modes):
            raise ValueError("nonlinear_overlap has the wrong rank-four mode shape.")
        if quadratic.shape != (modes, modes, modes) or mismatch.shape != quadratic.shape:
            raise ValueError("Quadratic coupling and mismatch have the wrong mode shape.")
        if raman.shape != (frequency_count,):
            raise ValueError("Raman frequency response must match the time grid.")
        fraction = float(raman_fraction)
        distance = float(propagation_distance)
        step = float(step_size)
        tolerance = float(relative_tolerance)
        minimum_step = float(minimum_step_size)
        maximum = int(maximum_steps)
        workspace = int(maximum_workspace_bytes)
        if method not in ("fixed-symmetric", "adaptive-step-doubling"):
            raise ValueError("Unknown coupled-envelope method.")
        if (
            not 0.0 <= fraction <= 1.0
            or distance <= 0.0
            or step <= 0.0
            or tolerance <= 0.0
            or minimum_step <= 0.0
            or maximum < 1
        ):
            raise ValueError("Coupled-envelope propagation controls are invalid.")
        fixed_steps = math.ceil(distance / step)
        if method == "fixed-symmetric" and fixed_steps > maximum:
            raise ValueError("Fixed coupled-envelope path exceeds maximum_steps.")
        elements = x.size * y.size * modes * time.size
        required = 32 * elements * np.dtype(np.complex128).itemsize
        if workspace < required:
            raise ValueError(
                "Coupled-envelope workspace exceeds maximum_workspace_bytes."
            )
        if not all(
            np.all(np.isfinite(value))
            for value in (
                carrier,
                dispersion_,
                loss_,
                diffraction_,
                overlap,
                quadratic,
                mismatch,
                raman,
                steepening,
            )
        ):
            raise ValueError("Coupled-envelope coefficients must be finite.")
        content = {
            "kind": "coupled-envelope-plan",
            "time_points": array_tree_fingerprint(time),
            "x_points": array_tree_fingerprint(x),
            "y_points": array_tree_fingerprint(y),
            "carrier_frequencies": array_tree_fingerprint(carrier),
            "dispersion": array_tree_fingerprint(dispersion_),
            "loss": array_tree_fingerprint(loss_),
            "diffraction": array_tree_fingerprint(diffraction_),
            "nonlinear_overlap": array_tree_fingerprint(overlap),
            "quadratic_coupling": array_tree_fingerprint(quadratic),
            "phase_mismatch": array_tree_fingerprint(mismatch),
            "raman_frequency_response": array_tree_fingerprint(raman),
            "raman_fraction": fraction,
            "self_steepening": array_tree_fingerprint(steepening),
            "propagation_distance": distance,
            "step_size": step,
            "method": method,
            "relative_tolerance": tolerance,
            "minimum_step_size": minimum_step,
            "maximum_steps": maximum,
            "maximum_workspace_bytes": workspace,
        }
        self.time_points = jnp.asarray(time)
        self.x_points = jnp.asarray(x)
        self.y_points = jnp.asarray(y)
        self.carrier_frequencies = jnp.asarray(carrier)
        self.dispersion = jnp.asarray(dispersion_)
        self.loss = jnp.asarray(loss_)
        self.diffraction = jnp.asarray(diffraction_)
        self.nonlinear_overlap = jnp.asarray(overlap)
        self.quadratic_coupling = jnp.asarray(quadratic)
        self.phase_mismatch = jnp.asarray(mismatch)
        self.raman_frequency_response = jnp.asarray(raman)
        self.raman_fraction = fraction
        self.self_steepening = jnp.asarray(steepening)
        self.propagation_distance = distance
        self.step_size = step
        self.method = method
        self.relative_tolerance = tolerance
        self.minimum_step_size = minimum_step
        self.maximum_steps = maximum
        self.maximum_workspace_bytes = workspace
        self.plan_id = canonical_fingerprint(content)

    @property
    def mode_count(self) -> int:
        return self.carrier_frequencies.shape[0]

    @property
    def field_shape(self) -> tuple[int, int, int, int]:
        return (
            self.x_points.shape[0],
            self.y_points.shape[0],
            self.mode_count,
            self.time_points.shape[0],
        )


class CoupledEnvelopeEvidence(StrictModule):
    initial_energy: Array
    final_energy: Array
    relative_energy_change: Array
    initial_photon_number: Array
    final_photon_number: Array
    relative_photon_change: Array
    accepted_steps: Array
    rejected_steps: Array
    maximum_step_error: Array
    finite: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class CoupledEnvelopeRun(StrictModule):
    final_field: Array
    distances: Array
    step_sizes: Array
    error_estimates: Array
    evidence: CoupledEnvelopeEvidence
    plan_id: str = eqx.field(static=True)


def _frequency_axes(plan: CoupledEnvelopePlan, /):
    dt = float(plan.time_points[1] - plan.time_points[0])
    frequency = 2.0 * jnp.pi * jnp.fft.fftfreq(plan.time_points.size, d=dt)
    if plan.x_points.size > 1:
        dx = float(plan.x_points[1] - plan.x_points[0])
        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(plan.x_points.size, d=dx)
    else:
        kx = jnp.zeros((1,), dtype=frequency.dtype)
    if plan.y_points.size > 1:
        dy = float(plan.y_points[1] - plan.y_points[0])
        ky = 2.0 * jnp.pi * jnp.fft.fftfreq(plan.y_points.size, d=dy)
    else:
        ky = jnp.zeros((1,), dtype=frequency.dtype)
    return frequency, kx, ky


def _linear_step(plan: CoupledEnvelopePlan, field: Array, distance: Array, /) -> Array:
    _frequency, kx, ky = _frequency_axes(plan)
    transverse = kx[:, None] ** 2 + ky[None, :] ** 2
    generator = (
        1j * plan.dispersion[None, None, :, :]
        - 0.5 * plan.loss[None, None, :, :]
        - 0.5j * transverse[:, :, None, None] * plan.diffraction[None, None, :, None]
    )
    spectrum = jnp.fft.fftn(field, axes=(0, 1, 3))
    return jnp.fft.ifftn(spectrum * jnp.exp(distance * generator), axes=(0, 1, 3))


def _nonlinear_rhs(
    plan: CoupledEnvelopePlan,
    field: Array,
    distance: Array,
    /,
) -> Array:
    intensity = jnp.sum(jnp.abs(field) ** 2, axis=2)
    delayed = jnp.real(
        jnp.fft.ifft(
            jnp.fft.fft(intensity, axis=-1)
            * plan.raman_frequency_response[None, None, :],
            axis=-1,
        )
    )
    kerr = 1j * ein.contract(
        "abcd,xybt,xyct,xydt->xyat",
        plan.nonlinear_overlap,
        field,
        field,
        jnp.conj(field),
    )
    diagonal_overlap = jnp.stack(
        tuple(
            plan.nonlinear_overlap[index, index, index, index]
            for index in range(plan.mode_count)
        )
    )
    kerr = kerr + (
        1j
        * plan.raman_fraction
        * diagonal_overlap[None, None, :, None]
        * field
        * (delayed - intensity)[:, :, None, :]
    )
    frequency, _, _ = _frequency_axes(plan)
    shock = jnp.fft.ifft(
        jnp.fft.fft(kerr, axis=-1)
        * (
            1.0
            + plan.self_steepening[None, None, :, None] * frequency[None, None, None, :]
        ),
        axis=-1,
    )
    phase = jnp.exp(1j * plan.phase_mismatch * distance)
    quadratic = 1j * ein.contract(
        "abc,abc,xybt,xyct->xyat",
        plan.quadratic_coupling,
        phase,
        field,
        field,
    )
    return shock + quadratic


def _nonlinear_rk4(
    plan: CoupledEnvelopePlan,
    field: Array,
    distance: Array,
    step: Array,
    /,
) -> Array:
    first = _nonlinear_rhs(plan, field, distance)
    second = _nonlinear_rhs(plan, field + 0.5 * step * first, distance + 0.5 * step)
    third = _nonlinear_rhs(plan, field + 0.5 * step * second, distance + 0.5 * step)
    fourth = _nonlinear_rhs(plan, field + step * third, distance + step)
    return field + step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0


def _symmetric_step(
    plan: CoupledEnvelopePlan,
    field: Array,
    distance: Array,
    step: Array,
    /,
) -> Array:
    first = _linear_step(plan, field, 0.5 * step)
    nonlinear = _nonlinear_rk4(plan, first, distance, step)
    return _linear_step(plan, nonlinear, 0.5 * step)


def _envelope_energy(plan: CoupledEnvelopePlan, field: Array, /) -> Array:
    dt = float(plan.time_points[1] - plan.time_points[0])
    dx = float(plan.x_points[1] - plan.x_points[0]) if plan.x_points.size > 1 else 1.0
    dy = float(plan.y_points[1] - plan.y_points[0]) if plan.y_points.size > 1 else 1.0
    return dt * dx * dy * jnp.sum(jnp.abs(field) ** 2)


def _photon_number(plan: CoupledEnvelopePlan, field: Array, /) -> Array:
    mode_energy = jnp.sum(jnp.abs(field) ** 2, axis=(0, 1, 3))
    return jnp.sum(mode_energy / plan.carrier_frequencies)


def propagate_coupled_envelope(
    plan: CoupledEnvelopePlan,
    initial_field: ArrayLike,
    /,
) -> CoupledEnvelopeRun:
    """Propagate the declared fixed path; adaptive refusal is explicit."""

    if not isinstance(plan, CoupledEnvelopePlan):
        raise TypeError("plan must be CoupledEnvelopePlan.")
    field = jnp.asarray(initial_field, dtype=jnp.complex128)
    if field.shape != plan.field_shape:
        raise ValueError(f"initial_field must have shape {plan.field_shape}.")
    initial_energy = _envelope_energy(plan, field)
    initial_photon = _photon_number(plan, field)
    distance = 0.0
    step = min(plan.step_size, plan.propagation_distance)
    distances = [0.0]
    step_sizes = []
    errors = []
    accepted = 0
    rejected = 0
    while distance < plan.propagation_distance:
        if accepted + rejected >= plan.maximum_steps:
            raise ValueError("Coupled-envelope propagation exceeds maximum_steps.")
        step = min(step, plan.propagation_distance - distance)
        if plan.method == "fixed-symmetric":
            field = _symmetric_step(plan, field, jnp.asarray(distance), jnp.asarray(step))
            error = 0.0
            accepted += 1
            distance += step
        else:
            full = _symmetric_step(plan, field, jnp.asarray(distance), jnp.asarray(step))
            half = _symmetric_step(
                plan, field, jnp.asarray(distance), jnp.asarray(0.5 * step)
            )
            half = _symmetric_step(
                plan, half, jnp.asarray(distance + 0.5 * step), jnp.asarray(0.5 * step)
            )
            error = float(
                jnp.linalg.norm(half - full) / jnp.maximum(1.0, jnp.linalg.norm(half))
            )
            if error <= plan.relative_tolerance:
                field = half
                distance += step
                accepted += 1
                factor = (
                    2.0
                    if error == 0.0
                    else min(
                        2.0, max(0.5, 0.9 * (plan.relative_tolerance / error) ** 0.2)
                    )
                )
                step *= factor
            else:
                rejected += 1
                step *= max(0.2, 0.9 * (plan.relative_tolerance / error) ** 0.2)
                if step < plan.minimum_step_size:
                    raise ValueError(
                        "Adaptive envelope propagation reached minimum_step_size."
                    )
                continue
        distances.append(distance)
        step_sizes.append(step)
        errors.append(error)
    final_energy = _envelope_energy(plan, field)
    final_photon = _photon_number(plan, field)
    relative_energy = jnp.abs(final_energy - initial_energy) / jnp.maximum(
        1.0, initial_energy
    )
    relative_photon = jnp.abs(final_photon - initial_photon) / jnp.maximum(
        1.0, initial_photon
    )
    finite = jnp.all(jnp.isfinite(field))
    evidence_id = canonical_fingerprint(
        {
            "kind": "coupled-envelope-evidence",
            "plan": plan.plan_id,
            "initial_field": array_tree_fingerprint(np.asarray(initial_field)),
            "accepted_steps": accepted,
            "rejected_steps": rejected,
        }
    )
    evidence = CoupledEnvelopeEvidence(
        initial_energy=initial_energy,
        final_energy=final_energy,
        relative_energy_change=relative_energy,
        initial_photon_number=initial_photon,
        final_photon_number=final_photon,
        relative_photon_change=relative_photon,
        accepted_steps=jnp.asarray(accepted, dtype=jnp.int32),
        rejected_steps=jnp.asarray(rejected, dtype=jnp.int32),
        maximum_step_error=jnp.asarray(max(errors, default=0.0)),
        finite=finite,
        claim="vector-multimode-spatiotemporal-envelope-only",
        evidence_id=evidence_id,
    )
    return CoupledEnvelopeRun(
        final_field=field,
        distances=jnp.asarray(distances),
        step_sizes=jnp.asarray(step_sizes),
        error_estimates=jnp.asarray(errors),
        evidence=evidence,
        plan_id=plan.plan_id,
    )


def coupled_envelope_adjoint(
    plan: CoupledEnvelopePlan,
    initial_field: ArrayLike,
    final_cotangent: ArrayLike,
    /,
) -> Array:
    """Return the fixed-path reverse derivative; adaptive paths are nondifferentiable."""

    if plan.method != "fixed-symmetric":
        raise ValueError(
            "Adaptive envelope accept/reject paths do not admit this adjoint."
        )
    initial = jnp.asarray(initial_field, dtype=jnp.complex128)
    cotangent = jnp.asarray(final_cotangent, dtype=initial.dtype)
    if initial.shape != plan.field_shape or cotangent.shape != initial.shape:
        raise ValueError("Envelope adjoint arrays have the wrong shape.")

    def evolve(value: Array) -> Array:
        field = value
        distance = 0.0
        while distance < plan.propagation_distance:
            step = min(plan.step_size, plan.propagation_distance - distance)
            field = _symmetric_step(
                plan,
                field,
                jnp.asarray(distance),
                jnp.asarray(step),
            )
            distance += step
        return field

    _, pullback = jax.vjp(evolve, initial)
    return pullback(cotangent)[0]


class EnvelopeFullFieldBridgeEvidence(StrictModule):
    reconstructed_field: Array
    relative_rms_error: Array
    fractional_bandwidth: Array
    slowly_varying: Array
    accepted: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def assess_envelope_full_field_bridge(
    plan: CoupledEnvelopePlan,
    envelope: ArrayLike,
    carrier_resolved_field: ArrayLike,
    /,
    *,
    maximum_relative_error: float,
    maximum_fractional_bandwidth: float,
) -> EnvelopeFullFieldBridgeEvidence:
    """Audit, never infer, the overlap between envelope and carrier-resolved fields."""

    values = jnp.asarray(envelope, dtype=jnp.complex128)
    carrier_field = jnp.asarray(carrier_resolved_field, dtype=jnp.float64)
    if values.shape != plan.field_shape or carrier_field.shape != plan.field_shape[:2] + (
        plan.field_shape[-1],
    ):
        raise ValueError("Envelope/full-field bridge arrays have incompatible shapes.")
    phase = jnp.exp(-1j * plan.carrier_frequencies[:, None] * plan.time_points[None, :])
    reconstructed = jnp.real(ein.contract("xyct,ct->xyt", values, phase))
    relative = jnp.linalg.norm(reconstructed - carrier_field) / jnp.maximum(
        1.0,
        jnp.linalg.norm(carrier_field),
    )
    spectrum = jnp.sum(jnp.abs(jnp.fft.fft(values, axis=-1)) ** 2, axis=(0, 1))
    frequency, _, _ = _frequency_axes(plan)
    bandwidths = []
    for mode in range(plan.mode_count):
        weight = spectrum[mode]
        mean = jnp.sum(weight * frequency) / jnp.maximum(jnp.sum(weight), 1e-300)
        variance = jnp.sum(weight * (frequency - mean) ** 2) / jnp.maximum(
            jnp.sum(weight), 1e-300
        )
        bandwidths.append(
            jnp.sqrt(jnp.maximum(variance, 0.0)) / plan.carrier_frequencies[mode]
        )
    fractional = jnp.stack(tuple(bandwidths))
    slowly_varying = jnp.all(fractional <= maximum_fractional_bandwidth)
    accepted = slowly_varying & (relative <= maximum_relative_error)
    evidence_id = canonical_fingerprint(
        {
            "kind": "envelope-full-field-bridge-evidence",
            "plan": plan.plan_id,
            "envelope": array_tree_fingerprint(np.asarray(envelope)),
            "carrier_field": array_tree_fingerprint(np.asarray(carrier_resolved_field)),
            "maximum_relative_error": float(maximum_relative_error),
            "maximum_fractional_bandwidth": float(maximum_fractional_bandwidth),
        }
    )
    return EnvelopeFullFieldBridgeEvidence(
        reconstructed_field=reconstructed,
        relative_rms_error=relative,
        fractional_bandwidth=fractional,
        slowly_varying=slowly_varying,
        accepted=accepted,
        claim="audited-overlap-regime-not-full-maxwell-equivalence",
        evidence_id=evidence_id,
    )


def write_coupled_envelope_archive(
    path: str | Path,
    run: CoupledEnvelopeRun,
    provenance: ArrayArtifactProvenance,
    /,
) -> ArrayArtifactReceipt:
    return write_typed_array_artifact(
        path,
        run,
        artifact_kind="coupled-envelope-run",
        provenance=provenance,
        structure_ids={"plan": run.plan_id, "evidence": run.evidence.evidence_id},
    )


def read_coupled_envelope_archive(
    path: str | Path,
    template: CoupledEnvelopeRun,
    provenance: ArrayArtifactProvenance,
    /,
) -> tuple[CoupledEnvelopeRun, ArrayArtifactReceipt]:
    restored, receipt = read_typed_array_artifact(
        path,
        template,
        artifact_kind="coupled-envelope-run",
        provenance=provenance,
        structure_ids={
            "plan": template.plan_id,
            "evidence": template.evidence.evidence_id,
        },
    )
    return restored, receipt


__all__ = [
    "CoupledEnvelopeEvidence",
    "CoupledEnvelopeMethod",
    "CoupledEnvelopePlan",
    "CoupledEnvelopeRun",
    "EnvelopeFullFieldBridgeEvidence",
    "assess_envelope_full_field_bridge",
    "coupled_envelope_adjoint",
    "propagate_coupled_envelope",
    "read_coupled_envelope_archive",
    "write_coupled_envelope_archive",
]
