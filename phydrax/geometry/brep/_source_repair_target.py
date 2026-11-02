#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Assembly of a certified, occurrence-preserving repaired source."""

from __future__ import annotations

import numpy as np

from ._constructors import _check_solid_edge_balance, _face_trim_domain
from ._model import _certify_edge_face_gap_bounds, BRepGeometry, BRepModel
from ._source_repair_carriers import build_target_carriers
from ._source_repair_topology import target_topology


def build_repaired_target(model: BRepModel, /) -> BRepModel:
    """Build the exact target; refuse unsupported or incompletely proved sources."""
    model.require_current()
    source = model.geometry
    carriers = build_target_carriers(model)
    roots = target_topology(model, carriers.curves, carriers.patches, carriers.pcurves)
    bounds = np.asarray(roots.parameter_bounds, dtype=np.float64)
    geometry = BRepGeometry(
        vertex_points=roots.vertex_points,
        vertex_exact_points=roots.vertex_exact_points,
        vertex_roots=roots.vertex_roots,
        edge_endpoint_roots=roots.edge_endpoint_roots,
        coedge_endpoint_roots=roots.coedge_endpoint_roots,
        curves=carriers.curves,
        edge_curves=carriers.edge_curves,
        edge_ranges=roots.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=carriers.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=source.assembly_graph,
    )
    _check_solid_edge_balance(geometry, np.asarray(model.orientation, dtype=np.float64))
    if geometry.edge_vertex_gap_bounds is not None:
        raise ValueError("Repaired source has a nonzero edge-vertex gap.")
    face_gaps = _certify_edge_face_gap_bounds(
        geometry,
        carriers.patches,
        parameter_bounds=bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    if face_gaps is None or any(gap != 0 for gap in face_gaps):
        raise ValueError("Repaired source has an unproved or nonzero edge-face gap.")
    trims = tuple(
        _face_trim_domain(geometry, face, bounds[face])
        for face in range(len(carriers.patches))
    )
    return BRepModel(
        geometry=geometry,
        patches=carriers.patches,
        parameter_bounds=bounds,
        orientation=model.orientation,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=trims,
    )
