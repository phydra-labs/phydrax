#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ThinWalledSection(StrictModule, NonTrainableState):
    """Midline plate-segment representation of an open or closed thin-walled section."""

    nodes: Array
    segments: Array
    thickness: Array
    material_indices: Array
    free_edge_nodes: Array
    closed_cells: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    section_id: str = eqx.field(static=True)

    def __init__(
        self,
        nodes: ArrayLike,
        segments: ArrayLike,
        thickness: ArrayLike,
        /,
        *,
        material_indices: ArrayLike | None = None,
        free_edge_nodes: ArrayLike | None = None,
        closed_cells: tuple[tuple[int, ...], ...] = (),
        section_id: str | None = None,
    ):
        nodes_ = jnp.asarray(nodes)
        segments_ = jnp.asarray(segments, dtype=jnp.int32)
        thickness_ = jnp.asarray(thickness, dtype=nodes_.dtype)
        if nodes_.ndim != 2 or nodes_.shape[1] != 2:
            raise ValueError("Thin-walled section nodes must have shape (nodes, 2).")
        if segments_.ndim != 2 or segments_.shape[1] != 2:
            raise ValueError("Thin-walled segments must have shape (plates, 2).")
        if thickness_.shape != (segments_.shape[0],):
            raise ValueError("Plate thickness must match segment count.")
        host_segments = np.asarray(segments_)
        if (
            np.any(host_segments < 0)
            or np.any(host_segments >= nodes_.shape[0])
            or np.any(host_segments[:, 0] == host_segments[:, 1])
        ):
            raise ValueError("Thin-walled section segments are invalid.")
        widths = jnp.sqrt(
            jnp.sum(
                (nodes_[segments_[:, 1]] - nodes_[segments_[:, 0]]) ** 2,
                axis=-1,
            )
        )
        if bool(
            jnp.any(~jnp.isfinite(nodes_))
            | jnp.any(~jnp.isfinite(thickness_))
            | jnp.any(thickness_ <= 0.0)
            | jnp.any(widths <= 0.0)
        ):
            raise ValueError("Thin-walled section geometry is degenerate.")
        materials = (
            jnp.zeros((segments_.shape[0],), dtype=jnp.int32)
            if material_indices is None
            else jnp.asarray(material_indices, dtype=jnp.int32)
        )
        free = (
            jnp.zeros((nodes_.shape[0],), dtype=bool)
            if free_edge_nodes is None
            else jnp.asarray(free_edge_nodes, dtype=bool)
        )
        if materials.shape != thickness_.shape or free.shape != (nodes_.shape[0],):
            raise ValueError("Section material/free-edge arrays have invalid shapes.")
        self.nodes = nodes_
        self.segments = segments_
        self.thickness = thickness_
        self.material_indices = materials
        self.free_edge_nodes = free
        self.closed_cells = tuple(
            tuple(int(index) for index in cell) for cell in closed_cells
        )
        self.section_id = str(
            section_id
            or canonical_fingerprint(
                {
                    "kind": "thin-walled-section",
                    "geometry": array_tree_fingerprint(
                        (nodes_, segments_, thickness_, materials, free)
                    ),
                    "cells": [list(cell) for cell in closed_cells],
                }
            )
        )

    @property
    def widths(self) -> Array:
        return jnp.sqrt(
            jnp.sum(
                (self.nodes[self.segments[:, 1]] - self.nodes[self.segments[:, 0]]) ** 2,
                axis=-1,
            )
        )


__all__ = ["ThinWalledSection"]
