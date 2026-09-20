#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar envelope Kerr, delayed Raman, and self-steepening response."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._pulse_time import PulseTimeSpace


def envelope_spectrum(values: ArrayLike, /) -> Array:
    return jnp.fft.ifft(jnp.asarray(values), axis=2, norm="ortho")


def envelope_time_values(spectrum: ArrayLike, /) -> Array:
    return jnp.fft.fft(jnp.asarray(spectrum), axis=2, norm="ortho")


class EnvelopeNonlinearResponsePlan(StrictModule):
    raman_response: Array | None
    nonlinear_coefficient: float = eqx.field(static=True)
    raman_fraction: float = eqx.field(static=True)
    self_steepening: bool = eqx.field(static=True)
    maximum_response_elements: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        nonlinear_coefficient: float,
        /,
        *,
        raman_fraction: float = 0.0,
        raman_response: ArrayLike | None = None,
        self_steepening: bool = False,
        maximum_response_elements: int = 1_000_000,
        source_id: str,
    ):
        coefficient = float(nonlinear_coefficient)
        fraction = float(raman_fraction)
        maximum = int(maximum_response_elements)
        source = str(source_id)
        response = (
            None
            if raman_response is None
            else np.asarray(raman_response, dtype=np.float64)
        )
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("nonlinear_coefficient must be finite and nonnegative.")
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("raman_fraction must lie in [0, 1].")
        if (fraction > 0.0) != (response is not None):
            raise ValueError(
                "A positive Raman fraction requires exactly one response array."
            )
        if response is not None and (
            response.ndim != 1
            or response.size < 2
            or response.size > maximum
            or not np.all(np.isfinite(response))
        ):
            raise ValueError("Raman response is nonfinite or exceeds resource admission.")
        if maximum < 1 or not source:
            raise ValueError("Envelope response capacity and source_id are required.")
        self.raman_response = None if response is None else jnp.asarray(response)
        self.nonlinear_coefficient = coefficient
        self.raman_fraction = fraction
        self.self_steepening = bool(self_steepening)
        self.maximum_response_elements = maximum
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scalar-envelope-nonlinear-response-plan",
                "nonlinear_coefficient": coefficient,
                "raman_fraction": fraction,
                "raman_response": None
                if response is None
                else array_tree_fingerprint(response),
                "self_steepening": bool(self_steepening),
                "maximum_response_elements": maximum,
                "source_id": source,
            }
        )


class PreparedEnvelopeNonlinearResponse(StrictModule):
    plan: EnvelopeNonlinearResponsePlan
    time_space: PulseTimeSpace
    normalized_raman_response: Array | None
    raman_spectrum: Array | None
    prepared_id: str = eqx.field(static=True)


class EnvelopeResponseEvidence(StrictModule):
    instantaneous_energy: Array
    delayed_energy: Array
    nonlinear_source_energy: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class EnvelopeResponseEvaluation(StrictModule):
    source_spectrum: Array
    evidence: EnvelopeResponseEvidence


def prepare_envelope_nonlinear_response(
    plan: EnvelopeNonlinearResponsePlan,
    time_space: PulseTimeSpace,
    /,
) -> PreparedEnvelopeNonlinearResponse:
    if not isinstance(plan, EnvelopeNonlinearResponsePlan):
        raise TypeError("plan must be EnvelopeNonlinearResponsePlan.")
    if not isinstance(time_space, PulseTimeSpace):
        raise TypeError("time_space must be PulseTimeSpace.")
    if time_space.topology != "periodic-cell":
        raise ValueError("Envelope Raman convolution requires periodic-cell pulse time.")
    if plan.raman_response is None:
        normalized = None
        spectrum = None
    else:
        if plan.raman_response.shape != time_space.shape:
            raise ValueError("Raman response must match the pulse-time grid.")
        mass = time_space.sample_spacing * jnp.sum(plan.raman_response)
        if not bool(jnp.isfinite(mass) & (jnp.abs(mass) > 0.0)):
            raise ValueError("Raman response must have finite nonzero discrete mass.")
        normalized = plan.raman_response / mass
        spectrum = jnp.fft.fft(normalized)
    return PreparedEnvelopeNonlinearResponse(
        plan=plan,
        time_space=time_space,
        normalized_raman_response=normalized,
        raman_spectrum=spectrum,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-scalar-envelope-response",
                "plan": plan.plan_id,
                "time_space": time_space.space_id,
                "normalization": "dt-sum-hR=1",
            }
        ),
    )


def evaluate_envelope_nonlinear_response(
    prepared: PreparedEnvelopeNonlinearResponse,
    values: ArrayLike,
    angular_frequency_offsets: ArrayLike,
    carrier_angular_frequency: ArrayLike,
    /,
) -> EnvelopeResponseEvaluation:
    if not isinstance(prepared, PreparedEnvelopeNonlinearResponse):
        raise TypeError("prepared must be PreparedEnvelopeNonlinearResponse.")
    field = jnp.asarray(values)
    offsets = jnp.asarray(angular_frequency_offsets, dtype=jnp.real(field).dtype)
    carrier = jnp.asarray(carrier_angular_frequency, dtype=offsets.dtype)
    if field.ndim != 3 or field.shape[2:] != prepared.time_space.shape:
        raise ValueError("Scalar envelope values must have plane-plane-time shape.")
    if offsets.shape != prepared.time_space.shape or carrier.shape != ():
        raise ValueError("Envelope frequency offsets/carrier are incompatible.")
    carrier = eqx.error_if(
        carrier,
        ~jnp.isfinite(carrier) | (carrier <= 0.0),
        "Envelope carrier must be finite and positive.",
    )
    intensity = jnp.abs(field) ** 2
    if prepared.raman_spectrum is None:
        delayed = intensity
    else:
        delayed = (
            jnp.real(
                jnp.fft.ifft(
                    jnp.fft.fft(intensity, axis=2)
                    * prepared.raman_spectrum.reshape((1, 1, -1)),
                    axis=2,
                )
            )
            * prepared.time_space.sample_spacing
        )
    response = (
        1.0 - prepared.plan.raman_fraction
    ) * intensity + prepared.plan.raman_fraction * delayed
    nonlinear_time = field * response
    nonlinear_spectrum = envelope_spectrum(nonlinear_time)
    shock = (
        1.0 + offsets / carrier
        if prepared.plan.self_steepening
        else jnp.ones_like(offsets)
    )
    source = (
        1.0j
        * prepared.plan.nonlinear_coefficient
        * shock.reshape((1, 1, -1))
        * nonlinear_spectrum
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(source)))
        & jnp.all(jnp.isfinite(jnp.imag(source)))
        & jnp.all(jnp.isfinite(delayed))
    )
    evidence = EnvelopeResponseEvidence(
        instantaneous_energy=jnp.sum(intensity**2),
        delayed_energy=jnp.sum(delayed**2),
        nonlinear_source_energy=jnp.sum(jnp.abs(source) ** 2),
        finite=finite,
        prepared_id=prepared.prepared_id,
        claim="finite-scalar-envelope-kerr-raman-shock-response",
    )
    return EnvelopeResponseEvaluation(source_spectrum=source, evidence=evidence)


__all__ = [
    "EnvelopeNonlinearResponsePlan",
    "EnvelopeResponseEvaluation",
    "EnvelopeResponseEvidence",
    "PreparedEnvelopeNonlinearResponse",
    "envelope_spectrum",
    "envelope_time_values",
    "evaluate_envelope_nonlinear_response",
    "prepare_envelope_nonlinear_response",
]
