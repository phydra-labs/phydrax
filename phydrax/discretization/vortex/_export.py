#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._population import VortexPopulationState
from ._ring_sheet import VortexRingSheetState


class VortexExportPayload(StrictModule):
    points: object
    connectivity: object
    point_data: Mapping[str, object] = eqx.field(static=True)
    cell_data: Mapping[str, object] = eqx.field(static=True)
    field_data: Mapping[str, object] = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)


def export_vortex_population(state: VortexPopulationState, /) -> VortexExportPayload:
    if not isinstance(state, VortexPopulationState):
        raise TypeError("state must be VortexPopulationState.")
    active = np.asarray(state.active_mask)
    indices = np.nonzero(active)[0]
    points = np.asarray(state.positions)[indices]
    if points.shape[1] == 2:
        points = np.pad(points, ((0, 0), (0, 1)))
    point_data = {
        "stable_id": np.asarray(state.stable_ids)[indices],
        "parent_id": np.asarray(state.parent_ids)[indices],
        "source_code": np.asarray(state.source_codes)[indices],
        "strength": np.asarray(state.strength)[indices],
        "core_radius": np.asarray(state.core_radius)[indices],
        "volume": np.asarray(state.volume)[indices],
        "age": np.asarray(state.age)[indices],
    }
    identifier = canonical_fingerprint(
        {
            "kind": "vortex-population-export",
            "active_count": int(indices.size),
            "dimension": int(state.positions.shape[1]),
        }
    )
    return VortexExportPayload(
        points,
        np.empty((0, 2), dtype=np.int32),
        point_data,
        {},
        {"kind": "vortex-particles"},
        identifier,
    )


def export_vortex_ring_sheet(state: VortexRingSheetState, /) -> VortexExportPayload:
    if not isinstance(state, VortexRingSheetState):
        raise TypeError("state must be VortexRingSheetState.")
    active_edges = np.asarray(state.topology.edge_active)
    edge_indices = np.nonzero(active_edges)[0]
    connectivity = np.stack(
        (
            np.asarray(state.topology.edge_start)[edge_indices],
            np.asarray(state.topology.edge_end)[edge_indices],
        ),
        axis=-1,
    )
    edge_circulation = np.asarray(
        state.topology.edge_circulation(state.ring_circulation)
    )[edge_indices]
    cell_data = {
        "circulation": edge_circulation,
        "core_radius": np.asarray(state.edge_core_radius)[edge_indices],
        "age": np.asarray(state.edge_age)[edge_indices],
    }
    identifier = canonical_fingerprint(
        {
            "kind": "vortex-ring-sheet-export",
            "topology": state.topology.topology_id,
            "active_edges": int(edge_indices.size),
        }
    )
    return VortexExportPayload(
        np.asarray(state.vertices),
        connectivity,
        {},
        cell_data,
        {"kind": "vortex-filaments", "topology_id": state.topology.topology_id},
        identifier,
    )


__all__ = [
    "VortexExportPayload",
    "export_vortex_population",
    "export_vortex_ring_sheet",
]
