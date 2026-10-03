# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded exact Morton relations with explicit preparation boundaries.

Neighborhood plans accept a canonical support declaration: an explicit Morton
address (physical periodic cell, per-axis periodicity and depth), stable point
identities, candidate/chunk/pair capacities and a
:class:`MeshfreePrecisionPolicy`. Coordinates are stored in its geometry role;
neighbor identity, distances and gap witnesses are decided in its (never
narrower) certification role. Without an address, a padded non-periodic box
snapped to a dyadic grid is inferred; a periodic cell is never inferred from
samples. Queries run at bucketed
power-of-two storage capacities (inactive padding, reported as
``storage_capacity``) through stable compiled entry points, so executables are
reused across clouds and hierarchy levels.
"""

from __future__ import annotations

from numbers import Integral
from typing import final

import equinox as eqx
import jax
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
    MortonNeighborQueryResult,
    MortonRadiusRelationEvidence,
    MortonRadiusRelationPlan,
)
from ..spatial._neighbor_query import _minimum_image
from ._capacity import bucketed_storage_capacity
from ._precision import MeshfreePrecisionPolicy


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


def _precision_policy(
    precision: MeshfreePrecisionPolicy | None,
) -> MeshfreePrecisionPolicy:
    """Declared stage precision of a neighborhood (float64 roles by default).

    Morton addressing needs 64-bit codes and identities, so meshfree spatial
    plans refuse construction when ``jax_enable_x64`` is disabled; reduced
    coordinate precision is declared through the policy's geometry role.
    """
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "Meshfree neighborhoods require jax_enable_x64 for 64-bit Morton "
            "addressing; declare a float32 geometry role for float32 coordinates."
        )
    if precision is None:
        return MeshfreePrecisionPolicy()
    if not isinstance(precision, MeshfreePrecisionPolicy):
        raise TypeError("precision must be a MeshfreePrecisionPolicy or None.")
    return precision


def _cast_points(
    points: np.ndarray, precision: np.dtype, name: str, *, unique: bool
) -> np.ndarray:
    cast = np.asarray(points, dtype=precision)
    if unique and np.unique(cast, axis=0).shape[0] != cast.shape[0]:
        raise ValueError(f"{name} must not contain duplicates at {precision.name}.")
    return cast


def _stable_ids(value: ArrayLike | None, count: int, name: str) -> np.ndarray:
    if value is None:
        return np.arange(count, dtype=np.int64)
    identifiers = np.asarray(value)
    if identifiers.shape != (count,) or not np.issubdtype(identifiers.dtype, np.integer):
        raise ValueError(f"{name} must be an integer vector with one id per point.")
    if np.unique(identifiers).size != count:
        raise ValueError(f"{name} must be unique stable identities.")
    return np.asarray(identifiers, dtype=np.int64)


def _resolved_address(
    address: MortonAddressPlan | None, *clouds: np.ndarray
) -> MortonAddressPlan:
    if address is None:
        lower = np.min([cloud.min(axis=0) for cloud in clouds], axis=0)
        upper = np.max([cloud.max(axis=0) for cloud in clouds], axis=0)
        padding = np.maximum(upper - lower, 1.0) * 0.125
        lower, upper = lower - padding, upper + padding
        # The inferred box is snapped outward to a dyadic grid of one
        # sixteenth of the largest padded extent: exact physical coordinates
        # are queried (ties keep their stable-id order), the box stays within
        # a few percent of the padded bounds (cell sizes and candidate counts
        # are unchanged in practice), and clouds or hierarchy levels over the
        # same region share one static address and therefore one executable.
        quantum = float(2.0 ** (np.floor(np.log2(np.max(upper - lower))) - 4))
        lower = np.floor(lower / quantum) * quantum
        upper = np.ceil(upper / quantum) * quantum
        return MortonAddressPlan(tuple(lower), tuple(upper), 16)
    if not isinstance(address, MortonAddressPlan):
        raise TypeError("address must be a MortonAddressPlan.")
    if address.dimension != clouds[0].shape[1]:
        raise ValueError("address dimension must match the point dimension.")
    lower = np.asarray(address.lower, dtype=np.float64)
    upper = np.asarray(address.upper, dtype=np.float64)
    periodic = np.asarray(address.periodic_axes, dtype=np.bool_)
    for cloud in clouds:
        inside = (cloud >= lower) & (cloud < upper)
        if not np.all(inside[:, ~periodic]):
            raise ValueError(
                "Points must lie inside the declared address on non-periodic axes."
            )
    if np.any(periodic):
        # Periodic axes are half-open: the upper-face copy of a lower-face node
        # of the closed fundamental box is the same seam orbit point.
        wrapped = np.where(
            periodic, lower + np.mod(clouds[0] - lower, upper - lower), clouds[0]
        )
        if np.unique(wrapped, axis=0).shape[0] != clouds[0].shape[0]:
            raise ValueError(
                "A periodic address holds one representative per seam orbit; points "
                "coincide after periodic wrapping (for example duplicate upper-face nodes)."
            )
    return address


def _periodic_half_cell(address: MortonAddressPlan) -> float:
    """Distances below half the shortest periodic length have one minimum image."""
    lengths = [
        upper - lower
        for lower, upper, periodic in zip(
            address.lower, address.upper, address.periodic_axes, strict=True
        )
        if periodic
    ]
    return 0.5 * min(lengths) if lengths else float("inf")


@final
class SmoothSupportEnvelope(StrictModule):
    """Fixed physical support radius and declared anchored displacement envelope.

    Smooth fixed-radius GMLS weights vanish, with their requested coordinate
    derivatives, at the physical ``radius``. While every source and target
    stays within ``displacement`` of its anchor, a source-target distance
    changes by at most ``2 * displacement``: the candidate relation of every
    source within ``candidate_radius = radius + 2 * displacement`` of each
    anchored target therefore contains the union of physical supports over the
    admitted trajectory. A neighbor entering or leaving the physical support is
    a zero-weight event of that fixed relation, not a selection change; motion
    at or beyond ``displacement`` is refused (a new envelope is an epoch event).
    """

    radius: float = eqx.field(static=True)
    displacement: float = eqx.field(static=True)
    envelope_id: str = eqx.field(static=True)

    def __init__(self, radius: float, displacement: float) -> None:
        radius_, displacement_ = float(radius), float(displacement)
        if not np.isfinite(radius_) or radius_ <= 0:
            raise ValueError("Smooth support radius must be finite and positive.")
        if not np.isfinite(displacement_) or displacement_ <= 0:
            raise ValueError(
                "Smooth support displacement envelope must be finite and positive."
            )
        self.radius = radius_
        self.displacement = displacement_
        self.envelope_id = canonical_fingerprint(
            {
                "kind": "meshfree-smooth-support-envelope",
                "radius": radius_,
                "displacement": displacement_,
            }
        )

    @property
    def candidate_radius(self) -> float:
        return self.radius + 2.0 * self.displacement


def _padded(values: np.ndarray, storage: int, fill: np.ndarray) -> np.ndarray:
    padding = np.broadcast_to(fill, (storage - values.shape[0],) + values.shape[1:])
    return np.concatenate((values, padding.astype(values.dtype)))


def _storage_neighborhood(
    query: MortonNeighborQueryPlan,
    sources: Array,
    targets: Array,
    source_mask: Array,
    target_mask: Array,
    source_ids: Array,
    radius: float | None,
) -> tuple[MortonNeighborQueryResult, Array]:
    result = query.query(
        sources,
        targets,
        source_mask=source_mask,
        target_mask=target_mask,
        source_stable_ids=source_ids,
        radius=radius,
    )
    relative = sources[result.source_indices] - targets[:, None, :]
    if any(query.address_plan.periodic_axes):
        relative = _minimum_image(relative, query.address_plan)
    return result, jnp.linalg.norm(relative, axis=-1)


# Stable compiled entry: the query plan (bucketed storage capacities and the
# dyadic or declared address) and the candidate radius are static;
# coordinates, masks and ids dynamic.
_compiled_storage_neighborhood = eqx.filter_jit(_storage_neighborhood)


@final
class PreparedMeshfreeNeighborhood(StrictModule):
    """Admitted row relation of one support epoch with its motion certificate.

    A nearest-neighbor relation certifies its selection while the anchored
    displacement stays strictly below ``trust_margin`` (a quarter of the
    selection gap). A smooth-support relation (``envelope`` set) holds every
    source within the envelope's candidate radius (invalid slots pad the fixed
    capacity, with infinite ``distances``); its ``trust_margin`` is the
    declared displacement envelope.
    """

    relation: RowRelation
    distances: Array
    row_scale: Array
    trust_margin: Array
    evidence: MortonNeighborQueryEvidence
    source_ids: Array
    address: MortonAddressPlan
    envelope: SmoothSupportEnvelope | None
    storage_capacity: tuple[int, int] = eqx.field(static=True)
    precision: MeshfreePrecisionPolicy
    neighborhood_id: str = eqx.field(static=True)

    def minimum_image(self, relative: ArrayLike, /) -> Array:
        """Periodic minimum-image displacement in the declared address cell."""
        value = jnp.asarray(relative)
        if not any(self.address.periodic_axes):
            return value
        return _minimum_image(value, self.address)

    def offsets(self, sources: ArrayLike, targets: ArrayLike, /) -> Array:
        """Traceable source-minus-target offsets on the frozen relation."""
        source, target = jnp.asarray(sources), jnp.asarray(targets)
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != self.relation.source_size
            or target.shape[0] != self.relation.targets_per_case
            or source.shape[1] != self.address.dimension
            or target.shape[1] != self.address.dimension
        ):
            raise ValueError(
                "Sources/targets must match neighborhood sizes and dimension."
            )
        return self.minimum_image(
            source[self.relation.source_indices] - target[:, None, :]
        )

    def host_offsets(self, sources: np.ndarray, targets: np.ndarray, /) -> np.ndarray:
        """Host-preparation counterpart of ``offsets`` (same minimum image)."""
        indices = np.asarray(jax.device_get(self.relation.source_indices))
        relative = sources[indices] - targets[:, None, :]
        if not any(self.address.periodic_axes):
            return relative
        lower = np.asarray(self.address.lower, dtype=relative.dtype)
        lengths = np.asarray(self.address.upper, dtype=relative.dtype) - lower
        wrapped = relative - np.round(relative / lengths) * lengths
        return np.where(np.asarray(self.address.periodic_axes), wrapped, relative)


@final
class MeshfreeNeighborhoodPlan(StrictModule):
    """Bounded neighborhood declaration of one support epoch.

    Without ``envelope`` each target row holds its ``neighbors`` nearest
    sources (a k-nearest selection, certified by its gap witness). With a
    :class:`SmoothSupportEnvelope`, ``neighbors`` is the fixed candidate
    capacity of a radius relation holding every source within the envelope's
    candidate radius; a row holding more candidates than that capacity is
    refused at preparation, never truncated. Without ``maximum_candidates`` the
    Morton query derives its quasi-uniform candidate capacity from the
    neighbor count and dimension. ``precision`` declares the geometry
    (stored coordinates) and certification (selection and distances) roles
    and travels with the prepared neighborhood to the stencil fit.
    """

    sources: Array
    targets: Array
    source_active: Array
    source_ids: Array
    address: MortonAddressPlan
    envelope: SmoothSupportEnvelope | None
    neighbors: int = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    precision: MeshfreePrecisionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sources: ArrayLike,
        neighbors: int,
        *,
        targets: ArrayLike | None = None,
        source_active: ArrayLike | None = None,
        source_ids: ArrayLike | None = None,
        address: MortonAddressPlan | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
        envelope: SmoothSupportEnvelope | None = None,
        precision: MeshfreePrecisionPolicy | None = None,
    ) -> None:
        policy = _precision_policy(precision)
        geometry = np.dtype(policy.geometry_dtype)
        source = _points(sources, "sources", unique=True)
        target = source if targets is None else _points(targets, "targets", unique=False)
        if target.shape[1] != source.shape[1]:
            raise ValueError("Source and target dimensions must match.")
        address_ = _resolved_address(address, source, target)
        source = _cast_points(source, geometry, "sources", unique=True)
        target = source if targets is None else np.asarray(target, dtype=geometry)
        active = (
            np.ones(source.shape[0], dtype=np.bool_)
            if source_active is None
            else np.asarray(source_active)
        )
        if active.dtype != np.bool_ or active.shape != source.shape[:1]:
            raise ValueError("source_active must be Boolean with source shape.")
        identifiers = _stable_ids(source_ids, source.shape[0], "source_ids")
        k = _integer(neighbors, "neighbors")
        if k > np.count_nonzero(active):
            raise ValueError("neighbors exceeds active source count.")
        candidates = (
            None
            if maximum_candidates is None
            else _integer(maximum_candidates, "maximum_candidates")
        )
        requested = min(k + 1, int(np.count_nonzero(active)))
        if candidates is not None and candidates < requested:
            raise ValueError("maximum_candidates must include the neighbor-gap witness.")
        chunk = (
            None
            if target_chunk_size is None
            else _integer(target_chunk_size, "target_chunk_size")
        )
        if envelope is not None:
            if not isinstance(envelope, SmoothSupportEnvelope):
                raise TypeError("envelope must be a SmoothSupportEnvelope or None.")
            if envelope.candidate_radius >= _periodic_half_cell(address_):
                raise ValueError(
                    "Smooth support candidate radius must be below half the periodic cell."
                )
        self.sources = jax.device_put(source)
        self.targets = jax.device_put(target)
        self.source_active = jax.device_put(active)
        self.source_ids = jax.device_put(identifiers)
        self.address = address_
        self.envelope = envelope
        self.neighbors = k
        self.maximum_candidates = candidates
        self.target_chunk_size = chunk
        self.precision = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-neighborhood",
                "sources": array_tree_fingerprint(source),
                "targets": array_tree_fingerprint(target),
                "active": array_tree_fingerprint(active),
                "ids": array_tree_fingerprint(identifiers),
                "address": address_.plan_id,
                "envelope": None if envelope is None else envelope.envelope_id,
                "neighbors": k,
                "candidates": candidates,
                "chunk": chunk,
                "precision": policy.policy_id,
            }
        )

    def prepare(self) -> PreparedMeshfreeNeighborhood:
        sources, targets, active, identifiers = jax.device_get(
            (self.sources, self.targets, self.source_active, self.source_ids)
        )
        # Selection is decided in the certification role; widening the stored
        # geometry coordinates to it is exact.
        certification = np.dtype(self.precision.certification_dtype)
        sources, targets = sources.astype(certification), targets.astype(certification)
        source_count, target_count = sources.shape[0], targets.shape[0]
        requested = min(self.neighbors + 1, int(np.count_nonzero(active)))
        source_storage = bucketed_storage_capacity(source_count)
        target_storage = bucketed_storage_capacity(target_count)
        # Padding slots repeat an in-domain coordinate, are masked inactive,
        # and carry identities above every physical identity.
        query = MortonNeighborQueryPlan(
            self.address,
            source_storage,
            target_storage,
            requested,
            # A declared bound above the storage capacity is never binding.
            maximum_candidates=(
                None
                if self.maximum_candidates is None
                else min(self.maximum_candidates, source_storage)
            ),
            target_chunk_size=self.target_chunk_size,
        )
        result, distances = _compiled_storage_neighborhood(
            query,
            jax.device_put(_padded(sources, source_storage, sources[0])),
            jax.device_put(_padded(targets, target_storage, sources[0])),
            jax.device_put(_padded(active, source_storage, np.asarray(False))),
            jax.device_put(np.arange(target_storage) < target_count),
            jax.device_put(
                np.concatenate(
                    (
                        identifiers,
                        identifiers.max()
                        + 1
                        + np.arange(source_storage - source_count, dtype=np.int64),
                    )
                )
            ),
            None if self.envelope is None else self.envelope.candidate_radius,
        )
        # The single admission synchronization; logical rows are cut on host.
        successful, counts, indices, valid, host_distances = jax.device_get(
            (
                result.evidence.successful,
                result.counts,
                result.source_indices,
                result.valid,
                distances,
            )
        )
        counts, host_distances = counts[:target_count], host_distances[:target_count]
        valid = valid[:target_count]
        if not bool(successful):
            raise ValueError(
                "Meshfree neighborhood is incomplete: increase candidate capacity or repair inputs."
            )
        if np.max(host_distances, where=valid, initial=0.0) >= _periodic_half_cell(
            self.address
        ):
            raise ValueError(
                "Periodic neighborhood reaches half the periodic cell; minimum-image offsets are ambiguous."
            )
        if self.envelope is None:
            selected, trust = self._nearest_selection(counts, host_distances, requested)
        else:
            selected, trust = self._envelope_selection(
                counts, host_distances, valid, requested
            )
        return PreparedMeshfreeNeighborhood(
            relation=RowRelation(
                jax.device_put(indices[:target_count, : self.neighbors]),
                source_size=source_count,
                valid=jax.device_put(valid[:, : self.neighbors]),
            ),
            distances=jax.device_put(selected),
            row_scale=jax.device_put(
                np.max(selected, axis=1, where=np.isfinite(selected), initial=0.0)
            ),
            trust_margin=jax.device_put(trust),
            evidence=result.evidence,
            source_ids=self.source_ids,
            address=self.address,
            envelope=self.envelope,
            storage_capacity=(source_storage, target_storage),
            precision=self.precision,
            neighborhood_id=self.plan_id,
        )

    def _nearest_selection(
        self, counts: np.ndarray, distances: np.ndarray, requested: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.any(counts < requested):
            raise ValueError(
                "Meshfree neighborhood is incomplete: increase candidate capacity or repair inputs."
            )
        selected = distances[:, : self.neighbors]
        # Each source-target distance changes by at most 2 delta when both sets
        # move by delta. The difference between selected and excluded distances
        # can therefore close by at most 4 delta. Equality is not certified.
        trust = (
            np.full((distances.shape[0],), np.inf, dtype=selected.dtype)
            if requested == self.neighbors
            else np.maximum(distances[:, self.neighbors] - selected[:, -1], 0.0) / 4.0
        )
        return selected, trust

    def _envelope_selection(
        self,
        counts: np.ndarray,
        distances: np.ndarray,
        valid: np.ndarray,
        requested: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        envelope = self.envelope
        if envelope is None:
            raise RuntimeError("Envelope selection requires a smooth support envelope.")
        # The query returns at most ``neighbors + 1`` sources within the
        # candidate radius; a full witness slot proves that the union of
        # physical supports over the envelope exceeds the fixed capacity.
        if requested > self.neighbors and np.any(counts > self.neighbors):
            row = int(np.argmax(counts > self.neighbors))
            raise ValueError(
                f"Smooth support envelope of target row {row} holds more than "
                f"{self.neighbors} candidates within radius {envelope.candidate_radius:.6g}: "
                "increase the candidate capacity or reduce the displacement envelope."
            )
        selected = np.where(
            valid[:, : self.neighbors], distances[:, : self.neighbors], np.inf
        )
        trust = np.full(
            (distances.shape[0],), envelope.displacement, dtype=selected.dtype
        )
        return selected, trust


@final
class PreparedMeshfreeEdgeRelation(StrictModule):
    relation: EdgeRelation
    evidence: MortonRadiusRelationEvidence
    point_ids: Array
    address: MortonAddressPlan
    neighborhood_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    active_id: str = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    precision: MeshfreePrecisionPolicy


@final
class MeshfreeEdgeRelationPlan(StrictModule):
    points: Array
    active: Array
    point_ids: Array
    address: MortonAddressPlan
    radius: float = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)
    precision: MeshfreePrecisionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        radius: float,
        maximum_pairs: int,
        *,
        active: ArrayLike | None = None,
        point_ids: ArrayLike | None = None,
        address: MortonAddressPlan | None = None,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
        precision: MeshfreePrecisionPolicy | None = None,
    ) -> None:
        policy = _precision_policy(precision)
        cloud = _points(points, "points", unique=True)
        address_ = _resolved_address(address, cloud)
        cloud = _cast_points(
            cloud, np.dtype(policy.geometry_dtype), "points", unique=True
        )
        mask = (
            np.ones(cloud.shape[0], dtype=np.bool_)
            if active is None
            else np.asarray(active)
        )
        if mask.dtype != np.bool_ or mask.shape != cloud.shape[:1]:
            raise ValueError("active must be Boolean with point shape.")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive.")
        if radius >= _periodic_half_cell(address_):
            raise ValueError(
                "Periodic radius must be below half the periodic cell for unique images."
            )
        identifiers = _stable_ids(point_ids, cloud.shape[0], "point_ids")
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
        self.points = jax.device_put(cloud)
        self.active = jax.device_put(mask)
        self.point_ids = jax.device_put(identifiers)
        self.address = address_
        self.radius = float(radius)
        self.maximum_pairs = pairs
        self.maximum_candidates = candidates
        self.target_chunk_size = chunk
        self.precision = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-radius",
                "points": array_tree_fingerprint(cloud),
                "active": array_tree_fingerprint(mask),
                "ids": array_tree_fingerprint(identifiers),
                "address": address_.plan_id,
                "radius": float(radius),
                "pairs": pairs,
                "candidates": candidates,
                "chunk": chunk,
                "precision": policy.policy_id,
            }
        )

    def prepare(self) -> PreparedMeshfreeEdgeRelation:
        count = self.points.shape[0]
        plan = MortonRadiusRelationPlan(
            self.address,
            count,
            count,
            self.maximum_pairs,
            maximum_candidates=self.maximum_candidates,
            target_chunk_size=self.target_chunk_size,
        )
        result = plan.query(
            self.points.astype(self.precision.certification_dtype),
            self.points.astype(self.precision.certification_dtype),
            self.radius,
            source_mask=self.active,
            target_mask=self.active,
            source_stable_ids=self.point_ids,
            target_stable_ids=self.point_ids,
            exclude_self=True,
            pair_once=True,
        )
        if not bool(jax.device_get(result.evidence.successful)):
            raise ValueError(
                "Meshfree radius relation exceeds capacity or is uncertified."
            )
        return PreparedMeshfreeEdgeRelation(
            relation=result.relation,
            evidence=result.evidence,
            point_ids=self.point_ids,
            address=self.address,
            neighborhood_id=self.plan_id,
            source_id=canonical_fingerprint(
                array_tree_fingerprint(np.asarray(self.points))
            ),
            active_id=canonical_fingerprint(
                array_tree_fingerprint(np.asarray(self.active))
            ),
            radius=self.radius,
            precision=self.precision,
        )
