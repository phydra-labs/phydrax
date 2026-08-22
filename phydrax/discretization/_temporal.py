#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import resolved_identifier


TemporalMeshRole: TypeAlias = Literal[
    "internal",
    "collocation",
    "driver",
    "path",
]


class TemporalMesh(StrictModule, NonTrainableState):
    """Explicit finite partition of physical time, distinct from output sampling."""

    nodes: Array
    active_intervals: Array
    role: TemporalMeshRole = eqx.field(static=True)
    realized: bool = eqx.field(static=True)
    source_plan_id: str | None = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        /,
        *,
        role: TemporalMeshRole = "internal",
        active_intervals: ArrayLike | None = None,
        realized: bool = False,
        source_plan_id: str | None = None,
        mesh_id: str | None = None,
    ):
        values = np.asarray(nodes)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("Temporal meshes require at least two rank-1 nodes.")
        if not np.issubdtype(values.dtype, np.inexact):
            values = values.astype(float)
        if np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0):
            raise ValueError(
                "Temporal mesh nodes must be finite and strictly increasing."
            )
        if role not in ("internal", "collocation", "driver", "path"):
            raise ValueError("Unknown temporal mesh role.")
        if role == "path" and not np.allclose(
            np.diff(values),
            np.diff(values)[0],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Path temporal meshes must be uniform.")
        interval_count = int(values.size - 1)
        active = (
            np.ones((interval_count,), dtype=bool)
            if active_intervals is None
            else np.asarray(active_intervals, dtype=bool)
        )
        if active.shape != (interval_count,):
            raise ValueError(
                f"active_intervals must have shape {(interval_count,)}; got {active.shape}."
            )
        if not np.any(active):
            raise ValueError("Temporal meshes require at least one active interval.")
        source = None if source_plan_id is None else str(source_plan_id)
        if source is not None and not source:
            raise ValueError("source_plan_id must be non-empty when supplied.")
        if realized and source is None:
            raise ValueError(
                "Realized temporal meshes require source_plan_id provenance."
            )
        self.nodes = jnp.asarray(values)
        self.active_intervals = jnp.asarray(active)
        self.role = role
        self.realized = bool(realized)
        self.source_plan_id = source
        self.mesh_id = resolved_identifier(
            "mesh_id",
            mesh_id,
            {
                "kind": "temporal-mesh",
                "nodes": array_tree_fingerprint(values),
                "active_intervals": array_tree_fingerprint(active),
                "role": role,
                "realized": bool(realized),
                "source_plan": source,
            },
        )

    @classmethod
    def uniform(
        cls,
        start: float,
        stop: float,
        intervals: int,
        /,
        *,
        role: TemporalMeshRole = "internal",
        mesh_id: str | None = None,
    ) -> "TemporalMesh":
        if isinstance(intervals, bool) or not isinstance(intervals, (int, np.integer)):
            raise TypeError("intervals must be an integer.")
        count = int(intervals)
        if count <= 0:
            raise ValueError("intervals must be positive.")
        start_ = float(start)
        stop_ = float(stop)
        if not np.isfinite(start_) or not np.isfinite(stop_) or not stop_ > start_:
            raise ValueError(
                "Uniform temporal mesh bounds must be finite and increasing."
            )
        return cls(
            np.linspace(start_, stop_, count + 1),
            role=role,
            mesh_id=mesh_id,
        )

    @property
    def interval_count(self) -> int:
        return int(self.nodes.shape[0]) - 1

    @property
    def widths(self) -> Array:
        return jnp.diff(self.nodes)

    @property
    def midpoints(self) -> Array:
        return 0.5 * (self.nodes[:-1] + self.nodes[1:])

    @property
    def times(self) -> Array:
        return self.nodes

    @property
    def t0(self) -> Array:
        return self.nodes[0]

    @property
    def t1(self) -> Array:
        return self.nodes[-1]

    @property
    def duration(self) -> Array:
        return self.t1 - self.t0

    @property
    def num_steps(self) -> int:
        return self.interval_count

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def dt(self) -> Array:
        if self.role != "path":
            raise ValueError("dt is only canonical for uniform path temporal meshes.")
        return self.widths[0]

    @property
    def discretization_id(self) -> str:
        return self.mesh_id


class RealizedTemporalMesh(StrictModule, NonTrainableState):
    """Fixed-capacity accepted-step realization produced by an adaptive solver."""

    initial_time: Array
    accepted_times: Array
    valid: Array
    count: Array
    adaptive: bool = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)
    requested_time_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_time: ArrayLike,
        accepted_times: ArrayLike,
        valid: ArrayLike,
        count: ArrayLike,
        /,
        *,
        adaptive: bool,
        source_plan_id: str,
        requested_time_id: str,
    ):
        initial = jnp.asarray(initial_time)
        times = jnp.asarray(accepted_times)
        mask = jnp.asarray(valid, dtype=bool)
        count_ = jnp.asarray(count, dtype=jnp.int32)
        if initial.shape != () or times.ndim != 1 or mask.shape != times.shape:
            raise ValueError(
                "Realized temporal mesh requires scalar initial time and aligned "
                "rank-1 accepted times/mask."
            )
        if count_.shape != ():
            raise ValueError("Realized temporal mesh count must be scalar.")
        count_ = eqx.error_if(
            count_,
            (count_ < 0) | (count_ > times.size),
            "Accepted-step count exceeds temporal mesh capacity.",
        )
        expected_mask = jnp.arange(times.size, dtype=jnp.int32) < count_
        safe_times = jnp.where(mask, times, initial)
        times = eqx.error_if(
            times,
            jnp.any(mask != expected_mask)
            | jnp.any(mask & ~jnp.isfinite(times))
            | jnp.any(mask & (safe_times <= initial))
            | jnp.any(expected_mask[1:] & (safe_times[1:] <= safe_times[:-1])),
            "Accepted temporal nodes must form one finite increasing prefix.",
        )
        plan = str(source_plan_id)
        requested = str(requested_time_id)
        if not plan or not requested:
            raise ValueError("Temporal plan and requested-time IDs must be non-empty.")
        self.initial_time = initial
        self.accepted_times = times
        self.valid = mask
        self.count = count_
        self.adaptive = bool(adaptive)
        self.source_plan_id = plan
        self.requested_time_id = requested
        self.mesh_id = resolved_identifier(
            "mesh_id",
            None,
            {
                "kind": "realized-temporal-mesh",
                "source_plan": plan,
                "requested_time": requested,
                "adaptive": bool(adaptive),
                "capacity": int(times.size),
            },
        )

    @property
    def capacity(self) -> int:
        return int(self.accepted_times.size)

    @property
    def nodes(self) -> Array:
        return jnp.concatenate(
            (self.initial_time[None], self.accepted_times),
            axis=0,
        )

    @property
    def node_valid(self) -> Array:
        return jnp.concatenate(
            (jnp.ones((1,), dtype=bool), self.valid),
            axis=0,
        )


__all__ = ["RealizedTemporalMesh", "TemporalMesh", "TemporalMeshRole"]
