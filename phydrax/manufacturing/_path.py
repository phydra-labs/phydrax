#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint


ProcessEventKind: TypeAlias = Literal[
    "move", "deposit", "remove", "dwell", "heat", "fixture", "transfer"
]


@dataclass(frozen=True, slots=True)
class ToolpathEvent:
    event_id: str
    kind: ProcessEventKind
    start_time_s: float
    end_time_s: float
    frame_id: str
    start: tuple[float, ...]
    end: tuple[float, ...]
    power_w: float = 0.0
    mass_rate_kg_s: float = 0.0

    def __post_init__(self):
        if (
            not self.event_id
            or not self.frame_id
            or self.end_time_s < self.start_time_s
            or len(self.start) != len(self.end)
            or not self.start
        ):
            raise ValueError("Toolpath event is invalid.")
        if (
            any(
                not np.isfinite(v)
                for v in (*self.start, *self.end, self.power_w, self.mass_rate_kg_s)
            )
            or self.power_w < 0
            or self.mass_rate_kg_s < 0
        ):
            raise ValueError("Toolpath event data are invalid.")

    @property
    def event_fingerprint(self):
        return canonical_fingerprint(
            {
                "kind": "toolpath-event",
                "event_id": self.event_id,
                "event_kind": self.kind,
                "start_time_s": self.start_time_s,
                "end_time_s": self.end_time_s,
                "frame_id": self.frame_id,
                "start": self.start,
                "end": self.end,
                "power_w": self.power_w,
                "mass_rate_kg_s": self.mass_rate_kg_s,
            }
        )

    def position(self, time_s: ArrayLike, /) -> Array:
        duration = max(self.end_time_s - self.start_time_s, np.finfo(float).eps)
        fraction = jnp.clip((jnp.asarray(time_s) - self.start_time_s) / duration, 0, 1)
        return jnp.asarray(self.start) + fraction * (
            jnp.asarray(self.end) - jnp.asarray(self.start)
        )


__all__ = ["ProcessEventKind", "ToolpathEvent"]
