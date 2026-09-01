#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext, FrameDefinition
from ._frames import KinematicFrameTransform, KinematicTransformEvaluation
from ._state import CartesianOrbitState


class FrameTransformEdge(StrictModule, NonTrainableState):
    transform: KinematicFrameTransform
    precision_class: str = eqx.field(static=True)
    required_products: tuple[str, ...] = eqx.field(static=True)
    cost: int = eqx.field(static=True)
    edge_id: str = eqx.field(static=True)

    def __init__(
        self,
        transform: KinematicFrameTransform,
        /,
        *,
        precision_class: str,
        required_products: tuple[str, ...] = (),
        cost: int = 1,
    ):
        if not isinstance(transform, KinematicFrameTransform):
            raise TypeError("transform must be a KinematicFrameTransform.")
        precision = str(precision_class).strip()
        requirements = tuple(str(value).strip() for value in required_products)
        if not precision or any(not value for value in requirements):
            raise ValueError("Frame edge metadata must be non-empty.")
        if int(cost) <= 0:
            raise ValueError("Frame edge cost must be positive.")
        self.transform = transform
        self.precision_class = precision
        self.required_products = requirements
        self.cost = int(cost)
        self.edge_id = canonical_fingerprint(
            {
                "kind": "frame-transform-edge",
                "transform": transform.transform_id,
                "precision": precision,
                "requirements": list(requirements),
                "cost": int(cost),
            }
        )


class CompiledFramePath(StrictModule, NonTrainableState):
    edges: tuple[FrameTransformEdge, ...]
    path_id: str = eqx.field(static=True)

    def __init__(self, edges: tuple[FrameTransformEdge, ...], /):
        items = tuple(edges)
        if not items:
            raise ValueError("Compiled frame path requires at least one edge.")
        for left, right in zip(items[:-1], items[1:], strict=True):
            if (
                left.transform.target_frame.frame_id
                != right.transform.source_frame.frame_id
            ):
                raise ValueError("Compiled frame path is disconnected.")
        self.edges = items
        self.path_id = canonical_fingerprint(
            {"kind": "compiled-frame-path", "edges": [edge.edge_id for edge in items]}
        )

    def apply(
        self,
        state: CartesianOrbitState,
        relative_seconds,
        /,
        *,
        args=None,
    ) -> tuple[CartesianOrbitState, tuple[KinematicTransformEvaluation, ...]]:
        current = state
        evidence = []
        for edge in self.edges:
            source_context = current.context
            target_context = AstrodynamicsContext(
                source_context.scale,
                source_context.epoch,
                edge.transform.target_frame,
            )
            current, result = edge.transform.apply(
                current, relative_seconds, target_context, args
            )
            evidence.append(result)
        return current, tuple(evidence)


class FrameTransformGraph(StrictModule, NonTrainableState):
    """Immutable host graph compiled to one fixed transform edge sequence."""

    frames: tuple[FrameDefinition, ...]
    edges: tuple[FrameTransformEdge, ...]
    graph_id: str = eqx.field(static=True)

    def __init__(
        self,
        frames: tuple[FrameDefinition, ...],
        edges: tuple[FrameTransformEdge, ...],
        /,
    ):
        frame_items = tuple(frames)
        edge_items = tuple(edges)
        ids = tuple(frame.frame_id for frame in frame_items)
        if not frame_items or len(set(ids)) != len(ids):
            raise ValueError("Frame graph requires unique frame definitions.")
        known = set(ids)
        for edge in edge_items:
            if (
                edge.transform.source_frame.frame_id not in known
                or edge.transform.target_frame.frame_id not in known
            ):
                raise ValueError("Frame edge references an unknown frame.")
        self.frames = frame_items
        self.edges = edge_items
        self.graph_id = canonical_fingerprint(
            {
                "kind": "frame-transform-graph",
                "frames": list(ids),
                "edges": [edge.edge_id for edge in edge_items],
            }
        )

    def compile(
        self,
        source: FrameDefinition,
        target: FrameDefinition,
        /,
        *,
        available_products: tuple[str, ...] = (),
    ) -> CompiledFramePath:
        if source.frame_id == target.frame_id:
            raise ValueError("Identity frame paths do not require compilation.")
        available = set(available_products)
        frontier: list[tuple[int, str, tuple[FrameTransformEdge, ...]]] = [
            (0, source.frame_id, ())
        ]
        best: dict[str, int] = {source.frame_id: 0}
        solutions: list[tuple[int, tuple[FrameTransformEdge, ...]]] = []
        while frontier:
            frontier.sort(key=lambda item: (item[0], tuple(e.edge_id for e in item[2])))
            cost, frame_id, path = frontier.pop(0)
            if frame_id == target.frame_id:
                solutions.append((cost, path))
                continue
            for edge in self.edges:
                if edge.transform.source_frame.frame_id != frame_id:
                    continue
                if not set(edge.required_products).issubset(available):
                    continue
                next_id = edge.transform.target_frame.frame_id
                next_cost = cost + edge.cost
                if next_cost <= best.get(next_id, 2**31 - 1):
                    best[next_id] = next_cost
                    frontier.append((next_cost, next_id, (*path, edge)))
        if not solutions:
            raise ValueError("No qualified frame path exists.")
        minimum = min(cost for cost, _ in solutions)
        minimum_paths = [path for cost, path in solutions if cost == minimum]
        if len(minimum_paths) != 1:
            raise ValueError("Frame path is ambiguous at the selected cost.")
        return CompiledFramePath(minimum_paths[0])


__all__ = ["CompiledFramePath", "FrameTransformEdge", "FrameTransformGraph"]
