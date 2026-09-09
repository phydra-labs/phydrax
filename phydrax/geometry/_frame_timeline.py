#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-dependent rigid-frame routes prepared outside compiled execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..measurement._quantity import canonical_quantity_text
from ..measurement._time import SampleTimeAxis
from .analytic import RigidFrame


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(1.0 + trace)
        value = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            value = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            value = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            value = np.asarray(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    value /= np.linalg.norm(value)
    return value if value[0] >= 0.0 else -value


def _quaternion_rotation(quaternion: Array) -> Array:
    q = quaternion / jnp.sqrt(jnp.sum(quaternion * quaternion, axis=-1, keepdims=True))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _slerp(left: Array, right: Array, weight: Array) -> Array:
    dot = jnp.sum(left * right, axis=-1, keepdims=True)
    right = jnp.where(dot < 0.0, -right, right)
    dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
    angle = jnp.arccos(dot)
    sine = jnp.sin(angle)
    linear = (1.0 - weight[..., None]) * left + weight[..., None] * right
    curved = (
        jnp.sin((1.0 - weight[..., None]) * angle) * left
        + jnp.sin(weight[..., None] * angle) * right
    ) / jnp.where(sine > 1.0e-8, sine, 1.0)
    value = jnp.where(sine > 1.0e-8, curved, linear)
    return value / jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))


class FrameQueryEvidence(StrictModule, NonTrainableState):
    in_domain: Array
    finite: Array
    extrapolated: Array
    successful: Array
    route_id: str = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class FrameTransformTimeline:
    source_frame: str
    target_frame: str
    time_axis: SampleTimeAxis
    frames: tuple[RigidFrame, ...]
    calibration_id: str
    timeline_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = canonical_quantity_text(self.source_frame, "source_frame")
        target = canonical_quantity_text(self.target_frame, "target_frame")
        if source == target:
            raise ValueError("Frame timelines require distinct source and target frames.")
        if not isinstance(self.time_axis, SampleTimeAxis):
            raise TypeError("time_axis must be SampleTimeAxis.")
        if self.time_axis.sample_count < 2:
            raise ValueError("Frame timelines require at least two samples.")
        frames = tuple(self.frames)
        if len(frames) != self.time_axis.sample_count or any(
            not isinstance(value, RigidFrame) or value.dimension != 3 for value in frames
        ):
            raise ValueError(
                "frames must contain one three-dimensional RigidFrame per sample."
            )
        calibration = canonical_quantity_text(self.calibration_id, "calibration_id")
        rotations = np.stack([np.asarray(value.rotation) for value in frames])
        translations = np.stack([np.asarray(value.translation) for value in frames])
        object.__setattr__(self, "source_frame", source)
        object.__setattr__(self, "target_frame", target)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "calibration_id", calibration)
        object.__setattr__(
            self,
            "timeline_id",
            canonical_fingerprint(
                {
                    "kind": "frame-transform-timeline",
                    "source": source,
                    "target": target,
                    "time": self.time_axis.time_axis_id,
                    "rotations": array_tree_fingerprint(rotations),
                    "translations": array_tree_fingerprint(translations),
                    "calibration": calibration,
                }
            ),
        )

    def prepare(self) -> PreparedFrameTransformTimeline:
        rotations = np.stack([np.asarray(value.rotation) for value in self.frames])
        quaternions = np.stack([_rotation_to_quaternion(value) for value in rotations])
        translations = np.stack([np.asarray(value.translation) for value in self.frames])
        return PreparedFrameTransformTimeline(
            jnp.asarray(self.time_axis.sample_times),
            jnp.asarray(quaternions),
            jnp.asarray(translations),
            self.timeline_id,
        )


class PreparedFrameTransformTimeline(StrictModule, NonTrainableState):
    times: Array
    quaternions: Array
    translations: Array
    timeline_id: str = eqx.field(static=True)

    def evaluate(
        self, time: ArrayLike, /, *, allow_extrapolation: bool = False
    ) -> tuple[Array, Array, FrameQueryEvidence]:
        value = jnp.asarray(time)
        in_domain = (value >= self.times[0]) & (value <= self.times[-1])
        index = jnp.clip(
            jnp.searchsorted(self.times, value, side="right") - 1, 0, self.times.size - 2
        )
        width = self.times[index + 1] - self.times[index]
        fraction = (value - self.times[index]) / width
        quaternion = _slerp(
            self.quaternions[index], self.quaternions[index + 1], fraction
        )
        translation = (1.0 - fraction[..., None]) * self.translations[index] + fraction[
            ..., None
        ] * self.translations[index + 1]
        rotation = _quaternion_rotation(quaternion)
        accepted = in_domain | jnp.asarray(bool(allow_extrapolation))
        rotation = jnp.where(accepted[..., None, None], rotation, jnp.nan)
        translation = jnp.where(accepted[..., None], translation, jnp.nan)
        finite = jnp.all(jnp.isfinite(rotation)) & jnp.all(jnp.isfinite(translation))
        evidence = FrameQueryEvidence(
            in_domain, finite, ~in_domain, finite & jnp.all(accepted), self.timeline_id
        )
        return rotation, translation, evidence


@dataclass(frozen=True, slots=True)
class FrameTransformGraph:
    timelines: tuple[FrameTransformTimeline, ...]
    graph_id: str = field(init=False)

    def __post_init__(self) -> None:
        timelines = tuple(self.timelines)
        if not timelines or any(
            not isinstance(value, FrameTransformTimeline) for value in timelines
        ):
            raise TypeError("timelines must contain FrameTransformTimeline values.")
        edges = {(value.source_frame, value.target_frame) for value in timelines}
        if len(edges) != len(timelines):
            raise ValueError("Frame graph edges must be unique.")
        object.__setattr__(self, "timelines", timelines)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "frame-transform-graph",
                    "timelines": [value.timeline_id for value in timelines],
                }
            ),
        )

    def prepare_route(
        self, source_frame: str, target_frame: str, /
    ) -> PreparedFrameRoute:
        source = canonical_quantity_text(source_frame, "source_frame")
        target = canonical_quantity_text(target_frame, "target_frame")
        if source == target:
            return PreparedFrameRoute(
                (),
                (),
                canonical_fingerprint({"kind": "identity-frame-route", "frame": source}),
            )
        adjacency: dict[str, list[tuple[str, int, bool]]] = {}
        for index, timeline in enumerate(self.timelines):
            adjacency.setdefault(timeline.source_frame, []).append(
                (timeline.target_frame, index, False)
            )
            adjacency.setdefault(timeline.target_frame, []).append(
                (timeline.source_frame, index, True)
            )
        queue: deque[tuple[str, tuple[int, ...], tuple[bool, ...]]] = deque(
            [(source, (), ())]
        )
        visited = {source}
        solutions: list[tuple[tuple[int, ...], tuple[bool, ...]]] = []
        shortest: int | None = None
        while queue:
            frame, indices, reverse = queue.popleft()
            if shortest is not None and len(indices) > shortest:
                continue
            if frame == target:
                shortest = len(indices)
                solutions.append((indices, reverse))
                continue
            for neighbor, index, inverted in adjacency.get(frame, ()):
                if neighbor not in visited or neighbor == target:
                    visited.add(neighbor)
                    queue.append((neighbor, indices + (index,), reverse + (inverted,)))
        if len(solutions) != 1:
            raise ValueError("Frame route must exist and be uniquely shortest.")
        indices, reverse = solutions[0]
        route_id = canonical_fingerprint(
            {
                "kind": "prepared-frame-route",
                "graph": self.graph_id,
                "source": source,
                "target": target,
                "edges": list(indices),
                "reverse": list(reverse),
            }
        )
        return PreparedFrameRoute(
            tuple(self.timelines[index].prepare() for index in indices), reverse, route_id
        )


class PreparedFrameRoute(StrictModule, NonTrainableState):
    timelines: tuple[PreparedFrameTransformTimeline, ...]
    reverse: tuple[bool, ...] = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def evaluate(self, time: ArrayLike, /) -> tuple[Array, Array, FrameQueryEvidence]:
        value = jnp.asarray(time)
        batch = value.shape
        rotation = jnp.broadcast_to(jnp.eye(3), batch + (3, 3))
        translation = jnp.zeros(batch + (3,))
        successful = jnp.asarray(True)
        in_domain = jnp.asarray(True)
        finite = jnp.asarray(True)
        extrapolated = jnp.asarray(False)
        for timeline, reverse in zip(self.timelines, self.reverse, strict=True):
            edge_rotation, edge_translation, evidence = timeline.evaluate(value)
            if reverse:
                edge_rotation = jnp.swapaxes(edge_rotation, -1, -2)
                edge_translation = -contract(
                    "...ij,...j->...i", edge_rotation, edge_translation
                )
            translation = (
                contract("...ij,...j->...i", edge_rotation, translation)
                + edge_translation
            )
            rotation = contract("...ij,...jk->...ik", edge_rotation, rotation)
            successful = successful & evidence.successful
            in_domain = in_domain & evidence.in_domain
            finite = finite & evidence.finite
            extrapolated = extrapolated | evidence.extrapolated
        return (
            rotation,
            translation,
            FrameQueryEvidence(
                in_domain, finite, extrapolated, successful, self.route_id
            ),
        )


__all__ = [
    "FrameQueryEvidence",
    "FrameTransformGraph",
    "FrameTransformTimeline",
    "PreparedFrameRoute",
    "PreparedFrameTransformTimeline",
]
