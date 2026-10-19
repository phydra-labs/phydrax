#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Publication of certified occurrence displacement from native carrier extents."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._rational import fraction_up
from ..geometry.brep._exact import (
    exact_determinant,
    gram_deviation_bound,
    square_root_interval,
)
from ..geometry.brep._model import (
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    BRepPlacementEvidence,
)
from ._cad import CadImportPolicy, refuse
from ._cad_carriers import RigidPlacement
from ._cad_placement_frame import pose_entry_error_bounds


def source_placement_evidence(
    placement: RigidPlacement, /
) -> BRepPlacementEvidence | None:
    """Carry certified source pose errors until native model extent is available."""
    rotation_bound = placement.rotation_deviation_bound
    translation_bound = placement.translation_deviation_bound
    if rotation_bound == 0.0 and translation_bound == 0.0:
        return None
    entry_bounds = None
    if placement.source_enclosure is not None:
        entries = pose_entry_error_bounds(placement.matrix(), placement.source_enclosure)
        entry_bounds = tuple(tuple(float(value) for value in row) for row in entries)
    return BRepPlacementEvidence(
        rotation_bound,
        0.0,
        translation_bound,
        translation_bound,
        entrywise_deviation_bounds=entry_bounds,
    )


def _solid_extent(
    model: BRepModel, solid: int, budget: CoordinateEnclosureBudget, /
) -> float:
    """Bound distance from the placement origin over every native solid chart."""
    budget.reserve(32768 * len(model.topology.solid_faces[solid]), 32768)
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)
    squared_extent = Fraction(0)
    for face in model.topology.solid_faces[solid]:
        box = model.patches[face].bounding_box(bounds[face], budget=budget)
        if not np.all(np.isfinite(box)):
            raise refuse(
                "unsupported-entity",
                "placement",
                "A finite native model extent is required.",
            )
        squared = sum(
            (
                max(
                    abs(Fraction(float(box[0, axis]))), abs(Fraction(float(box[1, axis])))
                )
                ** 2
                for axis in range(3)
            ),
            Fraction(0),
        )
        squared_extent = max(squared_extent, squared)
    return square_root_interval(squared_extent)[1]


def certify_placed_model(
    model: BRepModel,
    policy: CadImportPolicy,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> BRepModel:
    """Retain zero-defect occurrences and publish bounded source displacement."""
    if budget is None:
        raise ValueError(
            "STEP.coordinate_budget must be supplied for near-rigid import; got None."
        )
    geometry = model.require_geometry()
    occurrences = []
    extents: dict[int, float] = {}
    changed = False
    for occurrence in geometry.occurrences:
        budget.reserve(32768, 32768)
        rotation = np.asarray(occurrence.rotation, dtype=np.float64)
        deviation = gram_deviation_bound(rotation)
        translation_deviation = 0.0
        if occurrence.placement_evidence is not None:
            deviation = max(
                deviation, occurrence.placement_evidence.rotation_deviation_bound
            )
            translation_deviation = (
                occurrence.placement_evidence.translation_deviation_bound
            )
        if exact_determinant(rotation) <= 0:
            raise refuse(
                "unsupported-entity",
                "placement",
                "A placement must have certified positive determinant.",
            )
        if deviation == 0.0 and translation_deviation == 0.0:
            occurrences.append(occurrence)
            continue
        if occurrence.solid not in extents:
            extents[occurrence.solid] = _solid_extent(model, occurrence.solid, budget)
        extent = extents[occurrence.solid]
        point_bound = fraction_up(
            Fraction(deviation) * Fraction(extent) + Fraction(translation_deviation)
        )
        if point_bound > policy.maximum_placement_deviation:
            raise refuse(
                "unsupported-entity",
                "/".join(occurrence.path),
                "Certified placement point deviation exceeds its declared limit.",
            )
        occurrences.append(
            BRepOccurrence(
                occurrence.path,
                occurrence.solid,
                occurrence.rotation,
                occurrence.translation,
                placement_evidence=BRepPlacementEvidence(
                    deviation,
                    extent,
                    point_bound,
                    translation_deviation,
                    entrywise_deviation_bounds=(
                        None
                        if occurrence.placement_evidence is None
                        else occurrence.placement_evidence.entrywise_deviation_bounds
                    ),
                ),
            )
        )
        changed = True
    if not changed:
        return model
    placed = BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        occurrences=tuple(occurrences),
        assembly_containers=geometry.assembly_containers,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
        budget=budget,
    )
    return model.evolve(geometry=placed, budget=budget)
