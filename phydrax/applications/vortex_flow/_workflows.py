#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.vortex import (
    VortexFilamentState,
    VortexFilamentTopology,
)
from ...discretization.vortex._source import (
    VortexSourceState,
    VortexTargetState,
)


class PassiveVortexProbes(StrictModule, NonTrainableState):
    position: Array
    probes_id: str = eqx.field(static=True)

    def __init__(self, position: ArrayLike, /):
        points = jnp.asarray(position, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] not in (2, 3)
            or not bool(jnp.all(jnp.isfinite(points)))
        ):
            raise ValueError("Passive probe positions require finite shape (N,2|3).")
        self.position = points
        self.probes_id = canonical_fingerprint(
            {
                "kind": "passive-vortex-probes",
                "count": int(points.shape[0]),
                "dimension": int(points.shape[1]),
            }
        )

    def sample(self, prepared_velocity, source: VortexSourceState, /):
        if not isinstance(source, VortexSourceState):
            raise TypeError("source must be VortexSourceState.")
        return prepared_velocity.evaluate(
            source,
            VortexTargetState(
                self.position,
                target_id=self.probes_id,
            ),
        )


def actuator_line_sources(
    position: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> VortexFilamentState:
    points = jnp.asarray(position)
    gamma = jnp.asarray(circulation, dtype=points.dtype)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError("Actuator line positions require shape (vertices >= 2, 3).")
    segment_count = int(points.shape[0] - 1)
    if gamma.shape == ():
        gamma = jnp.full((segment_count,), gamma, dtype=points.dtype)
    if gamma.shape != (segment_count,):
        raise ValueError("Actuator line circulation must be scalar or per segment.")
    core = jnp.asarray(core_radius, dtype=points.dtype)
    if core.shape == ():
        core = jnp.full((segment_count,), core, dtype=points.dtype)
    topology = VortexFilamentTopology.from_segments(
        tuple((index, index + 1) for index in range(segment_count)),
        vertex_capacity=int(points.shape[0]),
    )
    return VortexFilamentState(topology, points, gamma, core)


def actuator_surface_sources(
    panel_vertices: ArrayLike,
    circulation: ArrayLike,
    core_radius: ArrayLike,
    /,
) -> VortexFilamentState:
    panels = jnp.asarray(panel_vertices)
    gamma = jnp.asarray(circulation, dtype=panels.dtype)
    if panels.ndim != 3 or panels.shape[1:] != (4, 3):
        raise ValueError("Actuator surface panels require shape (panels,4,3).")
    if gamma.shape != (panels.shape[0],):
        raise ValueError("Actuator surface circulation requires one value per panel.")
    vertices = panels.reshape((-1, 3))
    segments = []
    strengths = []
    for panel in range(int(panels.shape[0])):
        base = 4 * panel
        for local in range(4):
            segments.append((base + local, base + (local + 1) % 4))
            strengths.append(gamma[panel])
    topology = VortexFilamentTopology.from_segments(
        tuple(segments),
        vertex_capacity=int(vertices.shape[0]),
    )
    core = jnp.asarray(core_radius, dtype=panels.dtype)
    if core.shape == ():
        core = jnp.full((len(segments),), core, dtype=panels.dtype)
    return VortexFilamentState(topology, vertices, jnp.stack(tuple(strengths)), core)


class VortexRigidMotionState(StrictModule):
    position: Array
    velocity: Array
    angle: Array
    angular_velocity: Array


class PrescribedVortexRigidMotion(StrictModule, NonTrainableState):
    law: Callable[[Array, Any], VortexRigidMotionState]
    law_id: str = eqx.field(static=True)

    def __init__(
        self, law: Callable[[Array, Any], VortexRigidMotionState], law_id: str, /
    ):
        if not callable(law) or not str(law_id):
            raise ValueError("Prescribed motion requires callable law and stable ID.")
        self.law = law
        self.law_id = str(law_id)

    def evaluate(self, time: ArrayLike, args: Any = None, /) -> VortexRigidMotionState:
        state = self.law(jnp.asarray(time), args)
        if not isinstance(state, VortexRigidMotionState):
            raise TypeError("Prescribed motion law must return VortexRigidMotionState.")
        return state


__all__ = [
    "PassiveVortexProbes",
    "PrescribedVortexRigidMotion",
    "VortexRigidMotionState",
    "actuator_line_sources",
    "actuator_surface_sources",
]
