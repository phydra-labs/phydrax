#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._operators import FrequencyDomainSignal
from ._photometry import ObservationDataProvenance


class QnmModeTable(StrictModule, NonTrainableState):
    frequency: Array
    damping_time: Array
    mode_indices: Array
    provenance: ObservationDataProvenance
    table_id: str = eqx.field(static=True)

    def __init__(self, frequency, damping_time, mode_indices, provenance, /):
        frequency_ = jnp.asarray(frequency)
        damping = jnp.asarray(damping_time)
        indices = jnp.asarray(mode_indices, dtype=jnp.int32)
        if (
            frequency_.ndim != 1
            or damping.shape != frequency_.shape
            or indices.shape != (frequency_.size, 3)
        ):
            raise ValueError("QNM mode table arrays are inconsistent.")
        self.frequency = frequency_
        self.damping_time = damping
        self.mode_indices = indices
        self.provenance = provenance
        self.table_id = canonical_fingerprint(
            {
                "kind": "qnm-mode-table",
                "modes": int(frequency_.size),
                "provenance": provenance.provenance_id,
            }
        )


class RingdownPlan(StrictModule, NonTrainableState):
    modes: QnmModeTable
    plan_id: str = eqx.field(static=True)

    def __init__(self, modes: QnmModeTable, /):
        self.modes = modes
        self.plan_id = canonical_fingerprint(
            {"kind": "ringdown-plan", "modes": modes.table_id}
        )

    def time_domain(self, times: ArrayLike, complex_amplitudes: ArrayLike, /) -> Array:
        time = jnp.asarray(times)
        amplitudes = jnp.asarray(complex_amplitudes)
        if amplitudes.shape != self.modes.frequency.shape:
            raise ValueError("Ringdown amplitudes must match QNM modes.")
        basis = jnp.exp(
            time[:, None]
            * (
                -1.0 / self.modes.damping_time[None, :]
                + 2.0j * jnp.pi * self.modes.frequency[None, :]
            )
        )
        return contract("tm,m->t", basis, amplitudes)


class DetectorNetworkResult(StrictModule):
    detector_signal: Array
    log_likelihood: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


class DetectorNetworkPlan(StrictModule, NonTrainableState):
    response_plus: Array
    response_cross: Array
    observed: Array
    inverse_noise: Array
    frequency_spacing: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        response_plus,
        response_cross,
        observed,
        inverse_noise,
        frequency_spacing,
        /,
        *,
        network_id="detector-network",
    ):
        plus = jnp.asarray(response_plus)
        cross = jnp.asarray(response_cross)
        observed_ = jnp.asarray(observed)
        inverse = jnp.asarray(inverse_noise)
        if (
            plus.ndim != 2
            or cross.shape != plus.shape
            or observed_.shape != plus.shape
            or inverse.shape != plus.shape
        ):
            raise ValueError("Detector-network arrays are inconsistent.")
        self.response_plus = plus
        self.response_cross = cross
        self.observed = observed_
        self.inverse_noise = inverse
        self.frequency_spacing = jnp.asarray(frequency_spacing).reshape(())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "detector-network",
                "network_id": str(network_id),
                "detectors": int(plus.shape[0]),
                "frequencies": int(plus.shape[1]),
            }
        )

    def evaluate(self, signal: FrequencyDomainSignal, /) -> DetectorNetworkResult:
        model = (
            self.response_plus * signal.plus[None, :]
            + self.response_cross * signal.cross[None, :]
        )
        residual = self.observed - model
        log_likelihood = (
            -2.0
            * self.frequency_spacing
            * jnp.real(jnp.sum(residual * jnp.conj(residual) * self.inverse_noise))
        )
        valid = jnp.all(jnp.isfinite(model)) & jnp.all(self.inverse_noise > 0.0)
        return DetectorNetworkResult(
            model, jnp.where(valid, log_likelihood, -jnp.inf), valid, self.plan_id
        )


__all__ = ["DetectorNetworkPlan", "DetectorNetworkResult", "QnmModeTable", "RingdownPlan"]
