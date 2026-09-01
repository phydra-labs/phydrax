#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._identity import BaseSpanId, OverlayCellId
from ._topology import PatchAtlas


class IntegrationOverlay(StrictModule, NonTrainableState):
    """Explicit integration cells routed to positive base spans, never control sites."""

    atlas: PatchAtlas
    cell_ids: tuple[OverlayCellId, ...]
    source_spans: tuple[tuple[BaseSpanId, ...], ...]
    overlay_id: str = eqx.field(static=True)

    def __init__(
        self,
        atlas: PatchAtlas,
        source_spans: Sequence[Sequence[BaseSpanId]],
        /,
        *,
        name: str = "overlay",
    ):
        if not isinstance(atlas, PatchAtlas):
            raise TypeError("atlas must be a PatchAtlas.")
        name_ = str(name)
        if not name_:
            raise ValueError("Overlay name must be non-empty.")
        routes = tuple(tuple(route) for route in source_spans)
        if not routes or any(not route for route in routes):
            raise ValueError(
                "Every integration overlay cell requires one or more source spans."
            )
        if not all(isinstance(span, BaseSpanId) for route in routes for span in route):
            raise TypeError("Overlay source spans must be BaseSpanId values.")
        for route in routes:
            if len({span.value for span in route}) != len(route):
                raise ValueError("An overlay cell cannot repeat one source span.")
            for span in route:
                topology = atlas.topology(span.patch_id)
                if len(span.coordinates) != len(topology.span_shape) or any(
                    coordinate >= size
                    for coordinate, size in zip(
                        span.coordinates, topology.span_shape, strict=True
                    )
                ):
                    raise ValueError("Overlay source span is outside its patch topology.")
        for patch_id, topology in zip(atlas.patch_ids, atlas.topologies, strict=True):
            covered = {
                span.coordinates
                for route in routes
                for span in route
                if span.patch_id == patch_id
            }
            expected = {span.coordinates for span in topology.span_ids}
            if covered != expected:
                raise ValueError(
                    f"Integration overlay does not cover every span of patch {patch_id!r}."
                )
        route_values = tuple(tuple(span.value for span in route) for route in routes)
        overlay_id = canonical_fingerprint(
            {
                "kind": "iga-integration-overlay",
                "name": name_,
                "atlas": atlas.atlas_id,
                "source_spans": [list(route) for route in route_values],
            }
        )
        self.atlas = atlas
        self.source_spans = routes
        self.overlay_id = overlay_id
        self.cell_ids = tuple(
            OverlayCellId(overlay_id, (index,)) for index in range(len(routes))
        )

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)

    def _source_rows_host(self, patch_id: str, /) -> np.ndarray:
        patch = str(patch_id)
        topology = self.atlas.topology(patch)
        rows = np.full((self.cell_count,), -1, dtype=np.int32)
        for cell, route in enumerate(self.source_spans):
            selected = tuple(span for span in route if span.patch_id == patch)
            if len(selected) > 1:
                raise ValueError("One overlay cell cannot route twice to the same patch.")
            if selected:
                rows[cell] = np.ravel_multi_index(
                    selected[0].coordinates, topology.span_shape
                )
        return rows

    def source_rows(self, patch_id: str, /) -> Array:
        return jnp.asarray(self._source_rows_host(patch_id))

    def restrict(self, patch_id: str, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        topology = self.atlas.topology(patch_id)
        if values_.shape[0] != topology.cell_count:
            raise ValueError(
                "Base-span values must begin with the patch positive-span count."
            )
        rows_host = self._source_rows_host(patch_id)
        if np.any(rows_host < 0):
            raise ValueError(
                "Overlay does not cover every cell for this patch; fail closed."
            )
        rows = jnp.asarray(rows_host)
        return values_[rows]

    def restrict_transpose(self, patch_id: str, overlay_values: ArrayLike, /) -> Array:
        values = jnp.asarray(overlay_values)
        if values.shape[0] != self.cell_count:
            raise ValueError("Overlay values must begin with the overlay cell count.")
        topology = self.atlas.topology(patch_id)
        rows_host = self._source_rows_host(patch_id)
        if np.any(rows_host < 0):
            raise ValueError(
                "Overlay does not cover every cell for this patch; fail closed."
            )
        rows = jnp.asarray(rows_host)
        result = jnp.zeros((topology.cell_count,) + values.shape[1:], dtype=values.dtype)
        return result.at[rows].add(values)


__all__ = ["IntegrationOverlay"]
