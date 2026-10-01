# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded exact Morton relations with explicit preparation boundaries."""

from __future__ import annotations

from numbers import Integral
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...sparse import EdgeRelation, RowRelation
from ..spatial import (
    MortonAddressPlan,
    MortonNeighborQueryEvidence,
    MortonNeighborQueryPlan,
    MortonRadiusRelationEvidence,
    MortonRadiusRelationPlan,
)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _points(value: ArrayLike, name: str, *, unique: bool) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must have real numeric coordinates.")
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] not in (1, 2, 3):
        raise ValueError(f"{name} must have shape (nonzero points, dimension 1/2/3).")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite.")
    if unique and np.unique(points, axis=0).shape[0] != points.shape[0]:
        raise ValueError(f"{name} must not contain exact duplicates.")
    return points


def _address(sources: np.ndarray, targets: np.ndarray) -> MortonAddressPlan:
    lower = np.minimum(sources.min(axis=0), targets.min(axis=0))
    upper = np.maximum(sources.max(axis=0), targets.max(axis=0))
    padding = np.maximum(upper - lower, 1.0) * 0.125
    return MortonAddressPlan(tuple(lower - padding), tuple(upper + padding), 16)


@final
class PreparedMeshfreeNeighborhood(StrictModule):
    relation: RowRelation
    distances: Array
    row_scale: Array
    trust_margin: Array
    evidence: MortonNeighborQueryEvidence
    neighborhood_id: str = eqx.field(static=True)


@final
class MeshfreeNeighborhoodPlan(StrictModule):
    sources: Array
    targets: Array
    source_active: Array
    neighbors: int = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sources: ArrayLike,
        neighbors: int,
        *,
        targets: ArrayLike | None = None,
        source_active: ArrayLike | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        source = _points(sources, "sources", unique=True)
        target = source if targets is None else _points(targets, "targets", unique=False)
        if target.shape[1] != source.shape[1]:
            raise ValueError("Source and target dimensions must match.")
        active = (
            np.ones(source.shape[0], dtype=np.bool_)
            if source_active is None
            else np.asarray(source_active)
        )
        if active.dtype != np.bool_ or active.shape != source.shape[:1]:
            raise ValueError("source_active must be Boolean with source shape.")
        k = _integer(neighbors, "neighbors")
        if k > np.count_nonzero(active):
            raise ValueError("neighbors exceeds active source count.")
        candidates = (
            None
            if maximum_candidates is None
            else _integer(maximum_candidates, "maximum_candidates")
        )
        requested = min(k + 1, int(np.count_nonzero(active)))
        if candidates is not None and not requested <= candidates <= source.shape[0]:
            raise ValueError(
                "maximum_candidates must include the neighbor-gap witness and fit sources."
            )
        chunk = (
            None
            if target_chunk_size is None
            else _integer(target_chunk_size, "target_chunk_size")
        )
        self.sources = jnp.asarray(source)
        self.targets = jnp.asarray(target)
        self.source_active = jnp.asarray(active)
        self.neighbors = k
        self.maximum_candidates = candidates
        self.target_chunk_size = chunk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-neighborhood",
                "sources": array_tree_fingerprint(source),
                "targets": array_tree_fingerprint(target),
                "active": array_tree_fingerprint(active),
                "neighbors": k,
                "candidates": candidates,
                "chunk": chunk,
            }
        )

    def prepare(self) -> PreparedMeshfreeNeighborhood:
        source, target = np.asarray(self.sources), np.asarray(self.targets)
        requested = min(
            self.neighbors + 1, int(np.count_nonzero(np.asarray(self.source_active)))
        )
        query = MortonNeighborQueryPlan(
            _address(source, target),
            source.shape[0],
            target.shape[0],
            requested,
            maximum_candidates=self.maximum_candidates,
            target_chunk_size=self.target_chunk_size,
        )
        result = query.query(self.sources, self.targets, source_mask=self.source_active)
        if not bool(np.asarray(result.evidence.successful)) or np.any(
            np.asarray(result.counts) < requested
        ):
            raise ValueError(
                "Meshfree neighborhood is incomplete: increase candidate capacity or repair inputs."
            )
        indices = result.source_indices[:, : self.neighbors]
        distances = jnp.linalg.norm(
            self.sources[result.source_indices] - self.targets[:, None, :], axis=-1
        )
        selected = distances[:, : self.neighbors]
        scale = jnp.max(selected, axis=1)
        # Each source-target distance changes by at most 2 delta when both sets
        # move by delta. The difference between selected and excluded distances
        # can therefore close by at most 4 delta. Equality is not certified.
        trust = (
            jnp.full((target.shape[0],), jnp.inf)
            if requested == self.neighbors
            else jnp.maximum(distances[:, self.neighbors] - selected[:, -1], 0.0) / 4.0
        )
        return PreparedMeshfreeNeighborhood(
            relation=RowRelation(
                indices,
                source_size=source.shape[0],
                valid=result.valid[:, : self.neighbors],
            ),
            distances=selected,
            row_scale=scale,
            trust_margin=trust,
            evidence=result.evidence,
            neighborhood_id=self.plan_id,
        )


@final
class PreparedMeshfreeEdgeRelation(StrictModule):
    relation: EdgeRelation
    evidence: MortonRadiusRelationEvidence
    neighborhood_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    active_id: str = eqx.field(static=True)
    radius: float = eqx.field(static=True)


@final
class MeshfreeEdgeRelationPlan(StrictModule):
    points: Array
    active: Array
    radius: float = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        radius: float,
        maximum_pairs: int,
        *,
        active: ArrayLike | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        cloud = _points(points, "points", unique=True)
        mask = (
            np.ones(cloud.shape[0], dtype=np.bool_)
            if active is None
            else np.asarray(active)
        )
        if mask.dtype != np.bool_ or mask.shape != cloud.shape[:1]:
            raise ValueError("active must be Boolean with point shape.")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive.")
        pairs = _integer(maximum_pairs, "maximum_pairs", 0)
        candidates = (
            None
            if maximum_candidates is None
            else _integer(maximum_candidates, "maximum_candidates")
        )
        chunk = (
            None
            if target_chunk_size is None
            else _integer(target_chunk_size, "target_chunk_size")
        )
        if candidates is not None and candidates > cloud.shape[0]:
            raise ValueError("maximum_candidates must not exceed source capacity.")
        self.points = jnp.asarray(cloud)
        self.active = jnp.asarray(mask)
        self.radius = float(radius)
        self.maximum_pairs = pairs
        self.maximum_candidates = candidates
        self.target_chunk_size = chunk
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-radius",
                "points": array_tree_fingerprint(cloud),
                "active": array_tree_fingerprint(mask),
                "radius": float(radius),
                "pairs": pairs,
                "candidates": candidates,
                "chunk": chunk,
            }
        )

    def prepare(self) -> PreparedMeshfreeEdgeRelation:
        cloud = np.asarray(self.points)
        plan = MortonRadiusRelationPlan(
            _address(cloud, cloud),
            cloud.shape[0],
            cloud.shape[0],
            self.maximum_pairs,
            maximum_candidates=self.maximum_candidates,
            target_chunk_size=self.target_chunk_size,
        )
        result = plan.query(
            self.points,
            self.points,
            self.radius,
            source_mask=self.active,
            target_mask=self.active,
            exclude_self=True,
            pair_once=True,
        )
        if not bool(np.asarray(result.evidence.successful)):
            raise ValueError(
                "Meshfree radius relation exceeds capacity or is uncertified."
            )
        return PreparedMeshfreeEdgeRelation(
            relation=result.relation,
            evidence=result.evidence,
            neighborhood_id=self.plan_id,
            source_id=canonical_fingerprint(array_tree_fingerprint(cloud)),
            active_id=canonical_fingerprint(
                array_tree_fingerprint(np.asarray(self.active))
            ),
            radius=self.radius,
        )
