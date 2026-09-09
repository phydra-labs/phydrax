#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cover import SubdomainCover
from ._ownership import cover_integration_ownership, IntegrationOwnership


def _conflict_graph(cover: SubdomainCover, /) -> dict[str, set[str]]:
    conflicts = {patch_id: set() for patch_id in cover.patch_ids}
    for patch_id, neighbors in cover.adjacency:
        conflicts[patch_id].update(neighbors)
    bounds = []
    for patch in cover.patches:
        metadata = patch.support.metadata
        lower = metadata.get("support_lower")
        upper = metadata.get("support_upper")
        if lower is None or upper is None:
            return conflicts
        bounds.append((jnp.asarray(lower), jnp.asarray(upper)))
    for left_index, (left_lower, left_upper) in enumerate(bounds):
        for right_index in range(left_index + 1, len(bounds)):
            right_lower, right_upper = bounds[right_index]
            positive_overlap = bool(
                jnp.all(
                    jnp.minimum(left_upper, right_upper)
                    > jnp.maximum(left_lower, right_lower)
                )
            )
            if positive_overlap:
                left_id = cover.patch_ids[left_index]
                right_id = cover.patch_ids[right_index]
                conflicts[left_id].add(right_id)
                conflicts[right_id].add(left_id)
    return conflicts


def _greedy_colors(
    patch_ids: tuple[str, ...],
    conflicts: dict[str, set[str]],
    /,
) -> tuple[tuple[str, ...], ...]:
    assigned: dict[str, int] = {}
    for patch_id in patch_ids:
        unavailable = {
            assigned[neighbor] for neighbor in conflicts[patch_id] if neighbor in assigned
        }
        color = 0
        while color in unavailable:
            color += 1
        assigned[patch_id] = color
    return tuple(
        tuple(patch_id for patch_id in patch_ids if assigned[patch_id] == color)
        for color in range(max(assigned.values()) + 1)
    )


class PreparedFieldRouting(StrictModule, NonTrainableState):
    """Static active-patch capacity, ownership, and deterministic conflict colors."""

    cover: SubdomainCover
    ownership: IntegrationOwnership
    colors: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    maximum_overlap: int = eqx.field(static=True)

    def __init__(
        self,
        cover: SubdomainCover,
        /,
        *,
        ownership: IntegrationOwnership | None = None,
    ):
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be a SubdomainCover.")
        maximum = cover.maximum_overlap
        if maximum is None:
            raise ValueError("Prepared routing requires a declared maximum_overlap.")
        ownership_ = (
            cover_integration_ownership(cover, kind="window")
            if ownership is None
            else ownership
        )
        if ownership_.cover.cover_id != cover.cover_id:
            raise ValueError("Routing ownership and cover identities must match.")
        conflicts = _conflict_graph(cover)
        self.cover = cover
        self.ownership = ownership_
        self.colors = _greedy_colors(cover.patch_ids, conflicts)
        self.maximum_overlap = int(maximum)

    def active_mask(self, points: Any, /):
        return jnp.stack(
            tuple(
                jnp.asarray(patch.support(points).data) > 0
                for patch in self.cover.patches
            ),
            axis=-1,
        )

    def active_indices(self, points: Any, /):
        mask = self.active_mask(points)
        return jax.vmap(
            lambda row: jnp.nonzero(
                row,
                size=self.maximum_overlap,
                fill_value=-1,
            )[0]
        )(mask.reshape((-1, len(self.cover.patches))))

    def validate(self, points: Any, /) -> None:
        active = jnp.sum(self.active_mask(points), axis=-1)
        if bool(jnp.any(active == 0)):
            raise ValueError("Prepared route encountered an uncovered point.")
        if bool(jnp.any(active > self.maximum_overlap)):
            raise ValueError("Prepared route capacity was exceeded.")


def prepare_field_routing(
    cover: SubdomainCover,
    /,
    *,
    ownership: IntegrationOwnership | None = None,
) -> PreparedFieldRouting:
    return PreparedFieldRouting(cover, ownership=ownership)


__all__ = ["PreparedFieldRouting", "prepare_field_routing"]
