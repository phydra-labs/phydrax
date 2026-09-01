#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._search import (
    ContactCandidateEpoch,
    DenseContactSearchPlan,
    SweepAndPruneContactSearchPlan,
)
from ._surface import PreparedCollisionScene


ContactSearchPlan = DenseContactSearchPlan | SweepAndPruneContactSearchPlan


class ContactSearchCacheState(StrictModule, NonTrainableState):
    epoch: ContactCandidateEpoch
    reference_positions: Array
    rebuild_count: Array
    reuse_count: Array
    state_version: Array
    cache_id: str = eqx.field(static=True)


class ContactSearchCacheUpdate(StrictModule):
    candidate: ContactSearchCacheState
    reused: Array
    maximum_displacement: Array
    envelope_margin: Array
    successful: Array


class CachedContactSearchPlan(StrictModule, NonTrainableState):
    """Accepted-state candidate cache with a fail-closed Verlet-style skin."""

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
    ):
        if not isinstance(
            search, (DenseContactSearchPlan, SweepAndPruneContactSearchPlan)
        ):
            raise TypeError("search must be a concrete contact search plan.")
        skin_ = float(skin)
        fraction = float(rebuild_fraction)
        if not np.isfinite(skin_) or skin_ <= 0.0:
            raise ValueError("skin must be finite and positive.")
        if not 0.0 < fraction < 1.0:
            raise ValueError("rebuild_fraction must lie strictly between zero and one.")
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

    def initialize(
        self,
        scene: PreparedCollisionScene,
        positions: ArrayLike,
        /,
    ) -> ContactSearchCacheState:
        current = jnp.asarray(positions)
        epoch = self.search.build(scene, np.asarray(current))
        return ContactSearchCacheState(
            epoch,
            current,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def update(
        self,
        scene: PreparedCollisionScene,
        state: ContactSearchCacheState,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
    ) -> ContactSearchCacheUpdate:
        if (
            not isinstance(state, ContactSearchCacheState)
            or state.cache_id != self.plan_id
        ):
            raise ValueError("Contact search cache state belongs to another plan.")
        current = jnp.asarray(positions, dtype=state.reference_positions.dtype)
        if current.shape != state.reference_positions.shape:
            raise ValueError("Cached contact positions changed shape.")
        end = (
            current
            if end_positions is None
            else jnp.asarray(end_positions, dtype=current.dtype)
        )
        if end.shape != current.shape:
            raise ValueError("Cached contact end_positions changed shape.")
        displacement = jnp.sqrt(
            jnp.sum((current - state.reference_positions) ** 2, axis=-1)
        )
        swept_displacement = jnp.sqrt(
            jnp.sum((end - state.reference_positions) ** 2, axis=-1)
        )
        maximum = jnp.maximum(
            jnp.max(displacement, initial=0.0),
            jnp.max(swept_displacement, initial=0.0),
        )
        threshold = self.rebuild_fraction * self.skin
        reusable = state.epoch.successful & (maximum <= threshold)
        if bool(reusable):
            candidate = ContactSearchCacheState(
                state.epoch,
                state.reference_positions,
                state.rebuild_count,
                state.reuse_count + 1,
                state.state_version + 1,
                self.plan_id,
            )
        else:
            epoch = self.search.build(
                scene,
                np.asarray(current),
                end_positions=None if end_positions is None else np.asarray(end),
            )
            candidate = ContactSearchCacheState(
                epoch,
                current,
                state.rebuild_count + 1,
                state.reuse_count,
                state.state_version + 1,
                self.plan_id,
            )
        margin = jnp.asarray(threshold, dtype=current.dtype) - maximum
        return ContactSearchCacheUpdate(
            candidate,
            reusable,
            maximum,
            margin,
            candidate.epoch.successful,
        )


__all__ = [
    "CachedContactSearchPlan",
    "ContactSearchCacheState",
    "ContactSearchCacheUpdate",
]
