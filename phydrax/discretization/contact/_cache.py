#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._compiled_search import (
    CompiledContactSearchPlan,
    ContactSearchPlan,
    LBVHContactSearchPlan,
)
from ._search import (
    ContactCandidateEpoch,
    ContactSearchScene,
    DenseContactSearchPlan,
    enclosure_admits,
    enclosure_binding,
    SweepAndPruneContactSearchPlan,
    vertex_path_boxes,
)
from ._trajectory import ContactPathEnclosure


class ContactSearchCacheState(StrictModule, NonTrainableState):
    epoch: ContactCandidateEpoch
    rebuild_count: Array
    reuse_count: Array
    state_version: Array
    scene_id: str = eqx.field(static=True)
    cache_id: str = eqx.field(static=True)


class ContactSearchCacheUpdate(StrictModule):
    """One cache refresh: candidate state, reuse decision and coverage evidence.

    ``maximum_displacement`` is the largest Euclidean distance of any vertex
    of the requested full-path boxes from the cached epoch's searched boxes;
    ``envelope_margin`` is the reuse threshold minus that distance.
    """

    candidate: ContactSearchCacheState
    reused: Array
    maximum_displacement: Array
    envelope_margin: Array
    successful: Array


def _box_excess(epoch: ContactCandidateEpoch, lower: Array, upper: Array, /) -> Array:
    excess = jnp.maximum(
        0.0,
        jnp.maximum(epoch.coverage_lower - lower, upper - epoch.coverage_upper),
    )
    distance = jnp.sqrt(jnp.sum(excess * excess, axis=-1))
    return jnp.max(distance, initial=0.0)


class CachedContactSearchPlan(StrictModule, NonTrainableState):
    """Accepted-state candidate cache with a fail-closed full-path skin.

    A cached epoch is reused only while every vertex box of the requested full
    path (from ``path_enclosure`` or the endpoint hull) lies within
    ``rebuild_fraction * skin`` of the boxes the epoch searched, with
    ``skin <= search.envelope_radius``; the search accounted for both moving
    primitives of a pair, so this keeps the reused epoch complete. Device
    searches decide reuse with a bounded ``lax.cond`` (one possible rebuild per
    update, no host synchronization); host authorities decide on the host.
    """

    search: ContactSearchPlan
    skin: float = eqx.field(static=True)
    rebuild_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        search: ContactSearchPlan,
        /,
        *,
        skin: float,
        rebuild_fraction: float = 0.5,
    ) -> None:
        if not isinstance(
            search,
            (
                DenseContactSearchPlan,
                SweepAndPruneContactSearchPlan,
                CompiledContactSearchPlan,
                LBVHContactSearchPlan,
            ),
        ):
            raise TypeError("search must be a concrete contact search plan.")
        skin_ = float(skin)
        fraction = float(rebuild_fraction)
        if not np.isfinite(skin_) or skin_ <= 0.0:
            raise ValueError("skin must be finite and positive.")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("rebuild_fraction must lie in (0, 1].")
        if search.envelope_radius < skin_:
            raise ValueError(
                "The underlying search envelope_radius must cover the cache skin."
            )
        self.search = search
        self.skin = skin_
        self.rebuild_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cached-contact-search-plan",
                "search": search.plan_id,
                "skin": skin_.hex(),
                "rebuild_fraction": fraction.hex(),
            }
        )

    @property
    def device(self) -> bool:
        return isinstance(
            self.search, (CompiledContactSearchPlan, LBVHContactSearchPlan)
        )

    def initialize(
        self,
        scene: ContactSearchScene,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
    ) -> ContactSearchCacheState:
        epoch = self.search.build(
            scene,
            positions,
            end_positions=end_positions,
            path_enclosure=path_enclosure,
        )
        return ContactSearchCacheState(
            epoch,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            scene.scene_id,
            self.plan_id,
        )

    def update(
        self,
        scene: ContactSearchScene,
        state: ContactSearchCacheState,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
        path_enclosure: ContactPathEnclosure | None = None,
    ) -> ContactSearchCacheUpdate:
        if (
            not isinstance(state, ContactSearchCacheState)
            or state.cache_id != self.plan_id
            or state.scene_id != scene.scene_id
        ):
            raise ValueError(
                "Contact search cache state belongs to another plan or scene."
            )
        trajectory_id = "" if path_enclosure is None else path_enclosure.trajectory_id
        if trajectory_id != state.epoch.trajectory_id:
            raise ValueError("Cached epochs are bound to one trajectory identity.")
        dtype = state.epoch.reference_positions.dtype
        start, lower, upper = vertex_path_boxes(
            scene, jnp.asarray(positions, dtype=dtype), end_positions, path_enclosure
        )
        end = start if end_positions is None else jnp.asarray(end_positions, dtype=dtype)
        admitted = enclosure_admits(path_enclosure, start, end, lower, upper)
        maximum = _box_excess(state.epoch, lower, upper)
        threshold = jnp.asarray(self.rebuild_fraction * self.skin, dtype=dtype)
        # Changed radii, clearances or planes invalidate coverage: rebuild.
        reusable = (
            state.epoch.successful
            & state.epoch.binds(scene)
            & admitted
            & (maximum <= threshold)
        )
        lower_time, upper_time, revision, _ = enclosure_binding(
            path_enclosure, None, dtype
        )
        rebound = eqx.tree_at(
            lambda epoch: (epoch.lower_time, epoch.upper_time, epoch.revision),
            state.epoch,
            (lower_time, upper_time, revision),
        )

        def rebuild(_: None, /) -> ContactCandidateEpoch:
            return self.search.build(
                scene,
                start,
                end_positions=end_positions,
                path_enclosure=path_enclosure,
            )

        def reuse(_: None, /) -> ContactCandidateEpoch:
            return rebound

        if self.device:
            epoch = jax.lax.cond(reusable, reuse, rebuild, None)
        else:
            epoch = reuse(None) if bool(reusable) else rebuild(None)
        rebuilt = (~reusable).astype(jnp.int32)
        candidate = ContactSearchCacheState(
            epoch,
            state.rebuild_count + rebuilt,
            state.reuse_count + 1 - rebuilt,
            state.state_version + 1,
            scene.scene_id,
            self.plan_id,
        )
        return ContactSearchCacheUpdate(
            candidate,
            reusable,
            maximum,
            threshold - maximum,
            epoch.successful,
        )


__all__ = [
    "CachedContactSearchPlan",
    "ContactSearchCacheState",
    "ContactSearchCacheUpdate",
]
