#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar-envelope GNLSE propagation in a declared interaction picture."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from math import factorial

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._envelope import PulseEnvelopeField
from ._envelope_response import (
    envelope_spectrum,
    envelope_time_values,
    evaluate_envelope_nonlinear_response,
    PreparedEnvelopeNonlinearResponse,
)
from ._pulse_time import PulseTimeSpace


class EnvelopePropagationStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DISTANCE = 2
    SPECTRAL_EDGE_LIMIT = 3
    REFINEMENT_LIMIT = 4
    RESPONSE_FAILURE = 5
    ADAPTIVE_CAPACITY_EXHAUSTED = 6
    MINIMUM_STEP_REACHED = 7


class EnvelopePropagationPlan(StrictModule):
    time_space: PulseTimeSpace
    dispersion_coefficients: tuple[tuple[int, float], ...] = eqx.field(static=True)
    attenuation: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    dealias_fraction: float = eqx.field(static=True)
    edge_guard_fraction: float = eqx.field(static=True)
    maximum_spectral_edge_fraction: float = eqx.field(static=True)
    maximum_refinement_error: float = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_space: PulseTimeSpace,
        dispersion_coefficients: Mapping[int, float],
        /,
        *,
        attenuation: float = 0.0,
        step_count: int,
        dealias_fraction: float = 2.0 / 3.0,
        edge_guard_fraction: float = 0.1,
        maximum_spectral_edge_fraction: float = 1e-6,
        maximum_refinement_error: float = 1e-5,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be PulseTimeSpace.")
        if time_space.topology != "periodic-cell":
            raise ValueError("Envelope propagation requires periodic-cell pulse time.")
        coefficients = tuple(
            sorted(
                (int(order), float(value))
                for order, value in dispersion_coefficients.items()
            )
        )
        attenuation_value = float(attenuation)
        steps = int(step_count)
        dealias = float(dealias_fraction)
        edge = float(edge_guard_fraction)
        edge_limit = float(maximum_spectral_edge_fraction)
        refinement = float(maximum_refinement_error)
        workspace = int(maximum_workspace_bytes)
        if any(order < 2 or not np.isfinite(value) for order, value in coefficients):
            raise ValueError(
                "Envelope dispersion orders must be at least two and finite."
            )
        if len({order for order, _ in coefficients}) != len(coefficients):
            raise ValueError("Envelope dispersion orders must be unique.")
        if not np.isfinite(attenuation_value) or attenuation_value < 0.0:
            raise ValueError("attenuation must be finite and nonnegative.")
        if steps < 1 or not 0.0 < dealias <= 1.0 or not 0.0 < edge <= 0.5:
            raise ValueError("Envelope step/dealias/edge policies are invalid.")
        if (
            not 0.0 <= edge_limit <= 1.0
            or not np.isfinite(refinement)
            or refinement < 0.0
            or workspace < 1
        ):
            raise ValueError("Envelope evidence/resource limits are invalid.")
        required = 32 * time_space.size * np.dtype(np.complex128).itemsize
        if required > workspace:
            raise ValueError("Envelope propagation workspace exceeds admission.")
        self.time_space = time_space
        self.dispersion_coefficients = coefficients
        self.attenuation = attenuation_value
        self.step_count = steps
        self.dealias_fraction = dealias
        self.edge_guard_fraction = edge
        self.maximum_spectral_edge_fraction = edge_limit
        self.maximum_refinement_error = refinement
        self.maximum_workspace_bytes = workspace
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-envelope-gnlse-propagation-plan",
                "time_space": time_space.space_id,
                "dispersion_coefficients": coefficients,
                "attenuation": attenuation_value,
                "step_count": steps,
                "dealias_fraction": dealias,
                "edge_guard_fraction": edge,
                "maximum_spectral_edge_fraction": edge_limit,
                "maximum_refinement_error": refinement,
                "maximum_workspace_bytes": workspace,
                "transform": "ifft-time-to-frequency-exp-minus-i-omega-t",
            }
        )


class PreparedEnvelopePropagation(StrictModule):
    plan: EnvelopePropagationPlan
    response: PreparedEnvelopeNonlinearResponse
    angular_frequency_offsets: Array
    linear_generator: Array
    dealias_mask: Array
    edge_mask: Array
    workspace_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class EnvelopePropagationEvidence(StrictModule):
    initial_energy: Array
    final_energy: Array
    relative_energy_change: Array
    spectral_edge_fraction: Array
    rejected_spectral_fraction: Array
    fixed_step_refinement_error: Array
    response_finite: Array
    finite: Array
    status: Array
    accepted: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class EnvelopePropagationResult(StrictModule):
    field: PulseEnvelopeField
    evidence: EnvelopePropagationEvidence
    prepared_id: str = eqx.field(static=True)


class AdaptiveEnvelopePolicy(StrictModule):
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    maximum_attempts: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float,
        absolute_tolerance: float,
        initial_step: float,
        minimum_step: float,
        maximum_step: float,
        maximum_attempts: int,
    ):
        values = tuple(
            float(value)
            for value in (
                relative_tolerance,
                absolute_tolerance,
                initial_step,
                minimum_step,
                maximum_step,
            )
        )
        attempts = int(maximum_attempts)
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Adaptive envelope tolerances/steps must be positive and finite."
            )
        if not values[3] <= values[2] <= values[4] or attempts < 1:
            raise ValueError("Adaptive envelope step ordering/capacity is invalid.")
        self.relative_tolerance = values[0]
        self.absolute_tolerance = values[1]
        self.initial_step = values[2]
        self.minimum_step = values[3]
        self.maximum_step = values[4]
        self.maximum_attempts = attempts
        self.policy_id = canonical_fingerprint(
            {
                "kind": "adaptive-envelope-step-doubling-policy",
                "values": values,
                "maximum_attempts": attempts,
            }
        )


class AdaptiveEnvelopeEvidence(StrictModule):
    accepted_steps: Array
    rejected_steps: Array
    attempted_steps: Array
    minimum_realized_step: Array
    maximum_realized_step: Array
    maximum_local_error: Array
    final_distance: Array
    finite: Array
    status: Array
    successful: Array
    policy_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class AdaptiveEnvelopePropagationResult(StrictModule):
    field: PulseEnvelopeField
    propagation: EnvelopePropagationEvidence
    adaptive: AdaptiveEnvelopeEvidence


def prepare_envelope_propagation(
    plan: EnvelopePropagationPlan,
    response: PreparedEnvelopeNonlinearResponse,
    /,
) -> PreparedEnvelopePropagation:
    if not isinstance(plan, EnvelopePropagationPlan):
        raise TypeError("plan must be EnvelopePropagationPlan.")
    if not isinstance(response, PreparedEnvelopeNonlinearResponse):
        raise TypeError("response must be PreparedEnvelopeNonlinearResponse.")
    if response.time_space.space_id != plan.time_space.space_id:
        raise ValueError("Envelope response and propagation time spaces differ.")
    count = plan.time_space.size
    spacing = plan.time_space.sample_spacing
    offsets = jnp.asarray(2.0 * np.pi * np.fft.fftfreq(count, d=spacing))
    linear = -0.5 * plan.attenuation * jnp.ones_like(offsets, dtype=jnp.complex128)
    for order, coefficient in plan.dispersion_coefficients:
        linear = linear + 1.0j * coefficient * offsets**order / float(factorial(order))
    nyquist = float(np.pi / spacing)
    dealias = jnp.abs(offsets) <= plan.dealias_fraction * nyquist
    edge = jnp.abs(offsets) >= (1.0 - plan.edge_guard_fraction) * nyquist
    workspace = 32 * count * np.dtype(np.complex128).itemsize
    return PreparedEnvelopePropagation(
        plan=plan,
        response=response,
        angular_frequency_offsets=offsets,
        linear_generator=linear,
        dealias_mask=dealias,
        edge_mask=edge,
        workspace_bytes=workspace,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-scalar-envelope-gnlse",
                "plan": plan.plan_id,
                "response": response.prepared_id,
                "linear_generator": array_tree_fingerprint(linear),
                "dealias_mask": array_tree_fingerprint(dealias),
            }
        ),
    )


def _masked_spectrum(prepared: PreparedEnvelopePropagation, spectrum: Array, /) -> Array:
    return jnp.where(prepared.dealias_mask.reshape((1, 1, -1)), spectrum, 0.0j)


def _nonlinear_source(
    prepared: PreparedEnvelopePropagation,
    spectrum: Array,
    carrier: Array,
    /,
) -> tuple[Array, Array]:
    values = envelope_time_values(spectrum)
    evaluation = evaluate_envelope_nonlinear_response(
        prepared.response,
        values,
        prepared.angular_frequency_offsets,
        carrier,
    )
    return _masked_spectrum(
        prepared, evaluation.source_spectrum
    ), evaluation.evidence.finite


def _rk4ip_step(
    prepared: PreparedEnvelopePropagation,
    spectrum: Array,
    carrier: Array,
    step: float,
    /,
) -> tuple[Array, Array]:
    generator = prepared.linear_generator.reshape((1, 1, -1))

    def rhs(interaction_state, position):
        physical = jnp.exp(generator * position) * interaction_state
        source, finite = _nonlinear_source(prepared, physical, carrier)
        return jnp.exp(-generator * position) * source, finite

    k1, finite1 = rhs(spectrum, 0.0)
    k2, finite2 = rhs(spectrum + 0.5 * step * k1, 0.5 * step)
    k3, finite3 = rhs(spectrum + 0.5 * step * k2, 0.5 * step)
    k4, finite4 = rhs(spectrum + step * k3, step)
    interaction = spectrum + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result = jnp.exp(generator * step) * interaction
    return _masked_spectrum(prepared, result), finite1 & finite2 & finite3 & finite4


def _fixed_steps(
    prepared: PreparedEnvelopePropagation,
    initial_spectrum: Array,
    carrier: Array,
    distance: float,
    steps: int,
    /,
) -> tuple[Array, Array]:
    step = distance / steps
    value = initial_spectrum
    finite = jnp.asarray(True)
    for _ in range(steps):
        value, response_finite = _rk4ip_step(prepared, value, carrier, step)
        finite = finite & response_finite & jnp.all(jnp.isfinite(value))
    return value, finite


def _edge_fraction(prepared: PreparedEnvelopePropagation, spectrum: Array, /) -> Array:
    energy = jnp.abs(spectrum) ** 2
    total = jnp.sum(energy)
    edge = jnp.sum(jnp.where(prepared.edge_mask.reshape((1, 1, -1)), energy, 0.0))
    return jnp.where(total > 0.0, edge / total, 0.0)


def _propagation_evidence(
    prepared: PreparedEnvelopePropagation,
    initial_spectrum: Array,
    final_spectrum: Array,
    refined_spectrum: Array,
    response_finite: Array,
    /,
) -> EnvelopePropagationEvidence:
    initial_energy = jnp.sum(jnp.abs(initial_spectrum) ** 2)
    final_energy = jnp.sum(jnp.abs(final_spectrum) ** 2)
    relative_energy = jnp.abs(final_energy - initial_energy) / jnp.maximum(
        1.0, initial_energy
    )
    rejected = jnp.sum(
        jnp.where(
            prepared.dealias_mask.reshape((1, 1, -1)),
            0.0,
            jnp.abs(initial_spectrum) ** 2,
        )
    ) / jnp.maximum(1.0, initial_energy)
    refinement = jnp.linalg.norm(final_spectrum - refined_spectrum) / jnp.maximum(
        1.0, jnp.linalg.norm(refined_spectrum)
    )
    edge = _edge_fraction(prepared, final_spectrum)
    finite = (
        response_finite
        & jnp.all(jnp.isfinite(final_spectrum))
        & jnp.isfinite(relative_energy)
        & jnp.isfinite(refinement)
        & jnp.isfinite(edge)
    )
    status = jnp.where(
        ~finite,
        int(EnvelopePropagationStatus.RESPONSE_FAILURE),
        jnp.where(
            edge > prepared.plan.maximum_spectral_edge_fraction,
            int(EnvelopePropagationStatus.SPECTRAL_EDGE_LIMIT),
            jnp.where(
                refinement > prepared.plan.maximum_refinement_error,
                int(EnvelopePropagationStatus.REFINEMENT_LIMIT),
                int(EnvelopePropagationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    return EnvelopePropagationEvidence(
        initial_energy=initial_energy,
        final_energy=final_energy,
        relative_energy_change=relative_energy,
        spectral_edge_fraction=edge,
        rejected_spectral_fraction=rejected,
        fixed_step_refinement_error=refinement,
        response_finite=response_finite,
        finite=finite,
        status=status,
        accepted=status == int(EnvelopePropagationStatus.SUCCESS),
        prepared_id=prepared.prepared_id,
        claim="finite-scalar-envelope-gnlse-with-explicit-window-step-and-band-evidence",
    )


def _validate_envelope_field_resources(
    prepared: PreparedEnvelopePropagation,
    field: PulseEnvelopeField,
    /,
) -> None:
    workspace = 32 * field.values.size * field.values.dtype.itemsize
    if workspace > prepared.plan.maximum_workspace_bytes:
        raise ValueError("Envelope field propagation exceeds maximum_workspace_bytes.")


def propagate_envelope(
    prepared: PreparedEnvelopePropagation,
    field: PulseEnvelopeField,
    distance: float,
    /,
) -> EnvelopePropagationResult:
    if not isinstance(prepared, PreparedEnvelopePropagation):
        raise TypeError("prepared must be PreparedEnvelopePropagation.")
    if not isinstance(field, PulseEnvelopeField) or field.polarization != "scalar":
        raise TypeError(
            "Envelope GNLSE propagation requires a scalar PulseEnvelopeField."
        )
    if field.time_space.space_id != prepared.plan.time_space.space_id:
        raise ValueError("Envelope field and propagation time spaces differ.")
    _validate_envelope_field_resources(prepared, field)
    propagation_distance = float(distance)
    if not np.isfinite(propagation_distance) or propagation_distance <= 0.0:
        raise ValueError("Propagation distance must be positive and finite.")
    initial = _masked_spectrum(prepared, envelope_spectrum(field.values))
    carrier = field.carrier_angular_frequency
    final, finite = _fixed_steps(
        prepared, initial, carrier, propagation_distance, prepared.plan.step_count
    )
    refined, refined_finite = _fixed_steps(
        prepared,
        initial,
        carrier,
        propagation_distance,
        2 * prepared.plan.step_count,
    )
    evidence = _propagation_evidence(
        prepared, initial, final, refined, finite & refined_finite
    )
    result_field = PulseEnvelopeField(
        field.plane_space,
        field.time_space,
        envelope_time_values(final),
        field.carrier_angular_frequency,
        field.longitudinal_coordinate + propagation_distance,
        polarization="scalar",
    )
    return EnvelopePropagationResult(result_field, evidence, prepared.prepared_id)


def propagate_envelope_adaptive(
    prepared: PreparedEnvelopePropagation,
    field: PulseEnvelopeField,
    distance: float,
    policy: AdaptiveEnvelopePolicy,
    /,
) -> AdaptiveEnvelopePropagationResult:
    if not isinstance(prepared, PreparedEnvelopePropagation):
        raise TypeError("prepared must be PreparedEnvelopePropagation.")
    if not isinstance(policy, AdaptiveEnvelopePolicy):
        raise TypeError("policy must be AdaptiveEnvelopePolicy.")
    if not isinstance(field, PulseEnvelopeField) or field.polarization != "scalar":
        raise TypeError("Adaptive GNLSE propagation requires a scalar envelope field.")
    if field.time_space.space_id != prepared.plan.time_space.space_id:
        raise ValueError("Envelope field and propagation time spaces differ.")
    _validate_envelope_field_resources(prepared, field)
    target = float(distance)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("distance must be positive and finite.")
    initial = _masked_spectrum(prepared, envelope_spectrum(field.values))
    value = initial
    position = 0.0
    step = min(policy.initial_step, target)
    accepted = 0
    rejected = 0
    attempts = 0
    minimum_realized = np.inf
    maximum_realized = 0.0
    maximum_error = 0.0
    response_finite = True
    status = EnvelopePropagationStatus.SUCCESS
    while position < target and attempts < policy.maximum_attempts:
        actual_step = min(step, target - position)
        full, finite_full = _rk4ip_step(
            prepared, value, field.carrier_angular_frequency, actual_step
        )
        half, finite_half0 = _rk4ip_step(
            prepared, value, field.carrier_angular_frequency, 0.5 * actual_step
        )
        half, finite_half1 = _rk4ip_step(
            prepared, half, field.carrier_angular_frequency, 0.5 * actual_step
        )
        error = float(
            jnp.linalg.norm(half - full) / jnp.maximum(1.0, jnp.linalg.norm(half))
        )
        threshold = policy.absolute_tolerance + policy.relative_tolerance
        finite_attempt = bool(finite_full & finite_half0 & finite_half1) and np.isfinite(
            error
        )
        attempts += 1
        maximum_error = max(maximum_error, error if np.isfinite(error) else np.inf)
        if finite_attempt and error <= threshold:
            value = half
            position += actual_step
            accepted += 1
            minimum_realized = min(minimum_realized, actual_step)
            maximum_realized = max(maximum_realized, actual_step)
            factor = (
                2.0
                if error == 0.0
                else min(2.0, max(0.5, 0.9 * (threshold / error) ** 0.2))
            )
            step = min(
                policy.maximum_step, max(policy.minimum_step, actual_step * factor)
            )
        else:
            rejected += 1
            response_finite = response_finite and finite_attempt
            step = max(policy.minimum_step, 0.5 * actual_step)
            if actual_step <= policy.minimum_step:
                status = EnvelopePropagationStatus.MINIMUM_STEP_REACHED
                break
    if position < target and status == EnvelopePropagationStatus.SUCCESS:
        status = EnvelopePropagationStatus.ADAPTIVE_CAPACITY_EXHAUSTED
    propagation = _propagation_evidence(
        prepared, initial, value, value, jnp.asarray(response_finite)
    )
    output = PulseEnvelopeField(
        field.plane_space,
        field.time_space,
        envelope_time_values(value),
        field.carrier_angular_frequency,
        field.longitudinal_coordinate + position,
        polarization="scalar",
    )
    finite = bool(jnp.all(jnp.isfinite(value))) and np.isfinite(maximum_error)
    adaptive = AdaptiveEnvelopeEvidence(
        accepted_steps=jnp.asarray(accepted, dtype=jnp.int32),
        rejected_steps=jnp.asarray(rejected, dtype=jnp.int32),
        attempted_steps=jnp.asarray(attempts, dtype=jnp.int32),
        minimum_realized_step=jnp.asarray(0.0 if accepted == 0 else minimum_realized),
        maximum_realized_step=jnp.asarray(maximum_realized),
        maximum_local_error=jnp.asarray(maximum_error),
        final_distance=jnp.asarray(position),
        finite=jnp.asarray(finite),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        successful=jnp.asarray(
            finite and status == EnvelopePropagationStatus.SUCCESS and position >= target
        ),
        policy_id=policy.policy_id,
        prepared_id=prepared.prepared_id,
        claim="adaptive-step-doubling-envelope-gnlse-with-realized-work-evidence",
    )
    return AdaptiveEnvelopePropagationResult(output, propagation, adaptive)


__all__ = [
    "AdaptiveEnvelopeEvidence",
    "AdaptiveEnvelopePolicy",
    "AdaptiveEnvelopePropagationResult",
    "EnvelopePropagationEvidence",
    "EnvelopePropagationPlan",
    "EnvelopePropagationResult",
    "EnvelopePropagationStatus",
    "PreparedEnvelopePropagation",
    "prepare_envelope_propagation",
    "propagate_envelope",
    "propagate_envelope_adaptive",
]
