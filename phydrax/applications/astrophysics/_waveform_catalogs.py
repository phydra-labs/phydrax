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


__all__ = ["QnmModeTable", "RingdownPlan"]
