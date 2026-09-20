#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._photometry import ObservationDataProvenance


class QnmModeTable(StrictModule, NonTrainableState):
    """QNM frequencies and damping times with explicit units and provenance."""

    frequency: Array
    damping_time: Array
    mode_indices: Array
    provenance: ObservationDataProvenance
    time_unit: str = eqx.field(static=True)
    frequency_convention: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency,
        damping_time,
        mode_indices,
        provenance,
        /,
        *,
        time_unit="geometric-time",
        frequency_convention="cycles-per-time",
    ):
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        unit = str(time_unit).strip()
        convention = str(frequency_convention).strip().lower()
        if not unit:
            raise ValueError("time_unit must be non-empty.")
        if convention not in ("cycles-per-time", "angular-frequency"):
            raise ValueError(
                "frequency_convention must be cycles-per-time or angular-frequency."
            )
        frequency_raw = np.asarray(frequency)
        damping_raw = np.asarray(damping_time)
        indices_raw = np.asarray(mode_indices)
        if np.iscomplexobj(frequency_raw) or np.iscomplexobj(damping_raw):
            raise ValueError("QNM frequencies and damping times must be real.")
        frequency_host = np.asarray(frequency_raw, dtype=np.float64)
        damping_host = np.asarray(damping_raw, dtype=np.float64)
        if (
            frequency_host.ndim != 1
            or frequency_host.size == 0
            or damping_host.shape != frequency_host.shape
            or indices_raw.shape != (frequency_host.size, 3)
        ):
            raise ValueError("QNM mode table arrays are inconsistent.")
        if indices_raw.dtype.kind not in ("i", "u"):
            raise ValueError("QNM mode indices must be integers.")
        if np.any(indices_raw > np.iinfo(np.int32).max) or np.any(
            indices_raw < np.iinfo(np.int32).min
        ):
            raise ValueError("QNM mode indices exceed int32 storage.")
        indices_host = np.asarray(indices_raw, dtype=np.int64)
        ell = indices_host[:, 0]
        azimuthal = indices_host[:, 1]
        overtone = indices_host[:, 2]
        if (
            np.any(~np.isfinite(frequency_host))
            or np.any(~np.isfinite(damping_host))
            or np.any(damping_host <= 0.0)
            or np.any(ell < 0)
            or np.any(np.abs(azimuthal) > ell)
            or np.any(overtone < 0)
            or np.unique(indices_host, axis=0).shape[0] != indices_host.shape[0]
        ):
            raise ValueError(
                "QNM data must be finite with positive damping times and unique valid modes (ell, m, n)."
            )
        self.frequency = jnp.asarray(frequency_host)
        self.damping_time = jnp.asarray(damping_host)
        self.mode_indices = jnp.asarray(indices_host, dtype=jnp.int32)
        self.provenance = provenance
        self.time_unit = unit
        self.frequency_convention = convention
        self.table_id = canonical_fingerprint(
            {
                "kind": "qnm-mode-table",
                "time_unit": unit,
                "frequency_convention": convention,
                "content": array_tree_fingerprint(
                    {
                        "frequency": frequency_host,
                        "damping_time": damping_host,
                        "mode_indices": indices_host,
                    }
                ),
                "provenance": provenance.provenance_id,
            }
        )


class RingdownPlan(StrictModule, NonTrainableState):
    """Synthesize catalog modes on times expressed in the table's time unit."""

    modes: QnmModeTable
    plan_id: str = eqx.field(static=True)

    def __init__(self, modes: QnmModeTable, /):
        if not isinstance(modes, QnmModeTable):
            raise TypeError("modes must be a QnmModeTable.")
        self.modes = modes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ringdown-plan",
                "modes": modes.table_id,
                "basis": "exp(-time/damping_time+i*angular_frequency*time)",
            }
        )

    def time_domain(self, times: ArrayLike, complex_amplitudes: ArrayLike, /) -> Array:
        time = jnp.asarray(times)
        if time.ndim != 1:
            raise ValueError("Ringdown times must be a one-dimensional grid.")
        amplitudes = jnp.asarray(complex_amplitudes)
        if amplitudes.shape != self.modes.frequency.shape:
            raise ValueError("Ringdown amplitudes must match QNM modes.")
        angular_frequency = self.modes.frequency
        if self.modes.frequency_convention == "cycles-per-time":
            angular_frequency = 2.0 * jnp.pi * angular_frequency
        basis = jnp.exp(
            time[:, None]
            * (
                -1.0 / self.modes.damping_time[None, :]
                + 1.0j * angular_frequency[None, :]
            )
        )
        return contract("tm,m->t", basis, amplitudes)


__all__ = ["QnmModeTable", "RingdownPlan"]
