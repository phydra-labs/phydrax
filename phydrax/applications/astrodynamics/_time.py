#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._data import AstrodynamicsDataProvenance
from ._status import AstrodynamicsStatus


TimeScaleName: TypeAlias = Literal["UTC", "TAI", "TT", "TDB"]
TimeInterpolation: TypeAlias = Literal["constant", "linear", "step"]


class TimeScaleTransformResult(StrictModule):
    relative_seconds: Array
    offset_seconds: Array
    valid: Array
    status: Array
    transform_id: str = eqx.field(static=True)


class TimeScaleTransform(StrictModule, NonTrainableState):
    """Prepared offset table mapping relative seconds between named time scales."""

    nodes: Array
    offsets: Array
    provenance: AstrodynamicsDataProvenance
    source_scale: TimeScaleName = eqx.field(static=True)
    target_scale: TimeScaleName = eqx.field(static=True)
    interpolation: TimeInterpolation = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_scale: TimeScaleName,
        target_scale: TimeScaleName,
        nodes: ArrayLike,
        offsets: ArrayLike,
        provenance: AstrodynamicsDataProvenance,
        /,
        *,
        interpolation: TimeInterpolation,
    ):
        source = str(source_scale).upper()
        target = str(target_scale).upper()
        if source not in ("UTC", "TAI", "TT", "TDB") or target not in (
            "UTC",
            "TAI",
            "TT",
            "TDB",
        ):
            raise ValueError("Unknown astrodynamics time scale.")
        if source == target:
            raise ValueError("Time-scale transform endpoints must differ.")
        if interpolation not in ("constant", "linear", "step"):
            raise ValueError("Unknown time offset interpolation policy.")
        if not isinstance(provenance, AstrodynamicsDataProvenance):
            raise TypeError("provenance must be AstrodynamicsDataProvenance.")
        nodes_host = np.asarray(nodes, dtype=float)
        offsets_host = np.asarray(offsets, dtype=float)
        if (
            nodes_host.ndim != 1
            or nodes_host.size == 0
            or offsets_host.shape != nodes_host.shape
            or np.any(~np.isfinite(nodes_host))
            or np.any(~np.isfinite(offsets_host))
            or np.any(np.diff(nodes_host) <= 0.0)
        ):
            raise ValueError(
                "Time transform nodes/offsets must be finite matching monotone vectors."
            )
        if interpolation == "constant" and nodes_host.size != 1:
            raise ValueError("A constant time transform requires one node and offset.")
        self.nodes = jnp.asarray(nodes_host)
        self.offsets = jnp.asarray(offsets_host)
        self.provenance = provenance
        self.source_scale = source  # type: ignore[assignment]
        self.target_scale = target  # type: ignore[assignment]
        self.interpolation = interpolation
        self.transform_id = canonical_fingerprint(
            {
                "kind": "time-scale-transform",
                "source": source,
                "target": target,
                "interpolation": interpolation,
                "nodes": nodes_host.tolist(),
                "offsets": offsets_host.tolist(),
                "provenance": provenance.provenance_id,
            }
        )

    def apply(self, relative_seconds: ArrayLike, /) -> TimeScaleTransformResult:
        query = jnp.asarray(relative_seconds)
        finite = jnp.isfinite(query)
        if self.interpolation == "constant":
            offset = jnp.broadcast_to(self.offsets[0], query.shape)
            support = jnp.ones_like(finite)
        else:
            support = (query >= self.nodes[0]) & (query <= self.nodes[-1])
            if self.interpolation == "linear":
                offset = jnp.interp(query, self.nodes, self.offsets)
            else:
                index = jnp.searchsorted(self.nodes, query, side="right") - 1
                offset = self.offsets[jnp.clip(index, 0, int(self.nodes.size) - 1)]
        valid = finite & support
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                support,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
        ).astype(jnp.int32)
        offset = jnp.where(valid, offset, 0.0)
        return TimeScaleTransformResult(
            query + offset,
            offset,
            valid,
            status,
            self.transform_id,
        )

    def inverse(self) -> TimeScaleTransform:
        return TimeScaleTransform(
            self.target_scale,
            self.source_scale,
            self.nodes + self.offsets,
            -self.offsets,
            self.provenance,
            interpolation=self.interpolation,
        )

    @classmethod
    def tai_to_tt(
        cls,
        provenance: AstrodynamicsDataProvenance,
        /,
    ) -> TimeScaleTransform:
        return cls(
            "TAI",
            "TT",
            jnp.asarray((0.0,)),
            jnp.asarray((32.184,)),
            provenance,
            interpolation="constant",
        )


__all__ = [
    "TimeInterpolation",
    "TimeScaleName",
    "TimeScaleTransform",
    "TimeScaleTransformResult",
]
