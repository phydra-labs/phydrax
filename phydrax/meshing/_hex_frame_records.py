#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Store source-frame preparation and integer-grid connection state."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np

from .._strict import StrictModule
from ..discretization import CellMesh
from ..discretization._cell_geometry import CellGeometrySpec


class PreparedSourceHexFrame(StrictModule):
    """Prepared integrable frame with explicit source-root face connection."""

    reference_mesh: CellMesh
    geometry: CellGeometrySpec
    root_cells: np.ndarray
    corner_frames: np.ndarray
    reciprocal_frames: jax.Array
    face_connections: np.ndarray
    face_permutations: np.ndarray
    boundary_faces: np.ndarray
    varying_roots: np.ndarray
    source_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)


class HexFrameHolonomy(StrictModule):
    """Store octahedral cycle closure and source-root cut evidence."""

    spanning_edges: np.ndarray
    cycle_edges: np.ndarray
    cycle_holonomy: np.ndarray
    singular_cycles: np.ndarray
    root_gauges: np.ndarray
    source_id: str = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)


class SourceBlockConnection(StrictModule):
    """Source-authored tensor face transition, including integer offset."""

    transports: np.ndarray
    offsets: np.ndarray
    holonomy: HexFrameHolonomy
    frame_id: str = eqx.field(static=True)


class SourceBlockIntegerGrid(StrictModule):
    """Store a conforming integer mesh with source-chart restrictions."""

    mesh: CellMesh
    cell_regions: np.ndarray
    source_parent_rows: np.ndarray
    root_intervals: np.ndarray
    cell_integer_lower: np.ndarray
    chart_numerators: np.ndarray
    chart_denominators: np.ndarray
    source_frame: PreparedSourceHexFrame
    source_connection: SourceBlockConnection
    grid_id: str = eqx.field(static=True)


class SourceFrameCornerData(StrictModule):
    """Store admitted source-chart derivatives and reciprocal corner frames."""

    frames: np.ndarray
    reciprocal_frames: jax.Array
    varying_roots: np.ndarray


class SourceFrameFaceData(StrictModule):
    """Store source-face connections, reversed permutations, and boundary rows."""

    connections: np.ndarray
    permutations: np.ndarray
    boundary: np.ndarray


class SourceIntegerGridCounts(StrictModule):
    """Store compatible axis counts and admitted integer-grid capacities."""

    counts: np.ndarray
    material: np.ndarray
    source_ids: np.ndarray
    total_cells: int = eqx.field(static=True)
    upper_vertices: int = eqx.field(static=True)
    parity_work: int = eqx.field(static=True)


class SourceIntegerGridBanks(StrictModule):
    """Store support-indexed nodes and source-chart restriction banks."""

    points: np.ndarray
    cells: np.ndarray
    parent_rows: np.ndarray
    lower: np.ndarray
    chart_numerators: np.ndarray
    chart_denominators: np.ndarray
    node_count: int = eqx.field(static=True)


class SourceIntegerRootNodes(StrictModule):
    """Store one root's support-glued node rows and the next global node slot."""

    nodes: np.ndarray
    node_count: int = eqx.field(static=True)
