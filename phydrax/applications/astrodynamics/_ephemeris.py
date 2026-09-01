#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._bodies import CelestialBodyCatalog
from ._data import AstrodynamicsDataProvenance
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


EphemerisBoundsPolicy: TypeAlias = Literal["error", "clip"]


class EphemerisEvaluation(StrictModule):
    state: CartesianOrbitState
    valid: Array
    status: Array
    body_index: Array
    ephemeris_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)


class TabulatedEphemeris(StrictModule, NonTrainableState):
    """Fixed-grid Cartesian body ephemeris in one scale, epoch, and frame."""

    relative_times: Array
    states: Array
    catalog: CelestialBodyCatalog
    provenance: AstrodynamicsDataProvenance
    bounds_policy: EphemerisBoundsPolicy = eqx.field(static=True)
    ephemeris_id: str = eqx.field(static=True)

    def __init__(
        self,
        relative_times: ArrayLike,
        states: ArrayLike,
        catalog: CelestialBodyCatalog,
        provenance: AstrodynamicsDataProvenance,
        /,
        *,
        bounds_policy: EphemerisBoundsPolicy = "error",
    ):
        if not isinstance(catalog, CelestialBodyCatalog):
            raise TypeError("catalog must be a CelestialBodyCatalog.")
        if not isinstance(provenance, AstrodynamicsDataProvenance):
            raise TypeError("provenance must be AstrodynamicsDataProvenance.")
        if bounds_policy not in ("error", "clip"):
            raise ValueError("Unknown ephemeris bounds policy.")
        times_host = np.asarray(relative_times, dtype=float)
        states_host = np.asarray(states, dtype=float)
        expected = (times_host.size, catalog.capacity, 6)
        if times_host.ndim != 1 or times_host.size < 2 or states_host.shape != expected:
            raise ValueError(f"Ephemeris states must have shape {expected}.")
        if (
            np.any(~np.isfinite(times_host))
            or np.any(np.diff(times_host) <= 0.0)
            or np.any(~np.isfinite(states_host[:, np.asarray(catalog.active_mask), :]))
        ):
            raise ValueError(
                "Ephemeris nodes and active states must be finite and monotone."
            )
        if provenance.frame_id != catalog.context.frame.frame_id:
            raise ValueError("Ephemeris provenance frame does not match catalog context.")
        if provenance.epoch_id != catalog.context.epoch.epoch_id:
            raise ValueError("Ephemeris provenance epoch does not match catalog context.")
        if provenance.scale_id != catalog.context.scale.scale_id:
            raise ValueError("Ephemeris provenance scale does not match catalog context.")
        values = jnp.asarray(states_host)
        if provenance.differentiability in ("coordinate-only", "constant"):
            values = jax.lax.stop_gradient(values)
        self.relative_times = jnp.asarray(times_host)
        self.states = values
        self.catalog = catalog
        self.provenance = provenance
        self.bounds_policy = bounds_policy
        self.ephemeris_id = canonical_fingerprint(
            {
                "kind": "tabulated-ephemeris",
                "catalog": catalog.catalog_id,
                "provenance": provenance.provenance_id,
                "num_times": int(times_host.size),
                "bounds_policy": bounds_policy,
            }
        )

    def evaluate(
        self,
        relative_seconds: ArrayLike,
        body_index: ArrayLike,
        /,
    ) -> EphemerisEvaluation:
        time = jnp.asarray(relative_seconds).reshape(())
        index = jnp.asarray(body_index, dtype=jnp.int32).reshape(())
        time_finite = jnp.isfinite(time)
        body_valid = (index >= 0) & (index < self.catalog.capacity)
        safe_index = jnp.clip(index, 0, self.catalog.capacity - 1)
        body_valid = body_valid & self.catalog.active_mask[safe_index]
        support = (time >= self.relative_times[0]) & (time <= self.relative_times[-1])
        query = jnp.clip(time, self.relative_times[0], self.relative_times[-1])
        upper = jnp.searchsorted(self.relative_times, query, side="right")
        upper = jnp.clip(upper, 1, int(self.relative_times.size) - 1)
        lower = upper - 1
        start = self.relative_times[lower]
        end = self.relative_times[upper]
        weight = (query - start) / (end - start)
        start_state = self.states[lower, safe_index]
        end_state = self.states[upper, safe_index]
        state = (1.0 - weight) * start_state + weight * end_state
        bounds_valid = support | (self.bounds_policy == "clip")
        valid = time_finite & body_valid & bounds_valid & jnp.all(jnp.isfinite(state))
        status = jnp.where(
            ~time_finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                ~body_valid,
                int(AstrodynamicsStatus.INVALID_DOMAIN),
                jnp.where(
                    ~bounds_valid,
                    int(AstrodynamicsStatus.INVALID_DOMAIN),
                    jnp.where(
                        valid,
                        int(AstrodynamicsStatus.SUCCESS),
                        int(AstrodynamicsStatus.NONFINITE_INPUT),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        safe_state = jnp.where(valid, state, jnp.zeros((6,), dtype=state.dtype))
        return EphemerisEvaluation(
            CartesianOrbitState(safe_state[:3], safe_state[3:], self.catalog.context),
            valid,
            status,
            index,
            self.ephemeris_id,
            self.provenance.provenance_id,
        )


__all__ = [
    "EphemerisBoundsPolicy",
    "EphemerisEvaluation",
    "TabulatedEphemeris",
]
