#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Carry CAD source-pose uncertainty onto each certified assembly occurrence."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..geometry._rational import fraction_up
from ..geometry.brep._assembly import (
    BRepAssemblyGraph,
    BRepPlacementEvidence,
)
from ..geometry.brep._constructors import BRepTessellationPolicy
from ..geometry.brep._exact import square_root_interval
from ..geometry.brep._model import BRepGeometry, BRepModel, BRepTessellation
from ..geometry.brep._tessellation import tessellate_brep_model
from ._cad import CadImportPolicy, refuse
from ._cad_carriers import RigidPlacement
from ._cad_placement_frame import pose_entry_error_bounds


def source_placement_evidence(
    placement: RigidPlacement, /
) -> BRepPlacementEvidence | None:
    """Retain source-frame errors before the solid extent is known."""
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


def _solid_extent(model: BRepModel, solid: int, /) -> float:
    """Bound the local solid's distance from the placement origin."""
    geometry = model.geometry
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)
    squared_extent = Fraction(0)
    for face in geometry.topology.solid_faces[solid]:
        box = np.asarray(model.patches[face].bounding_box(bounds[face]), dtype=np.float64)
        if box.shape != (2, 3) or not np.all(np.isfinite(box)) or np.any(box[0] > box[1]):
            raise refuse(
                "unsupported-entity",
                "placement",
                "A finite native model extent is required for placement-deviation certification.",
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


def _model_with_geometry(
    model: BRepModel,
    geometry: BRepGeometry,
    tessellation: BRepTessellation | None,
    /,
) -> BRepModel:
    return BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
        tessellation=tessellation,
    )


def _with_occurrence_evidence(
    model: BRepModel,
    records: tuple[tuple[tuple[str, ...], BRepPlacementEvidence], ...],
    /,
) -> BRepModel:
    geometry = model.geometry
    graph = geometry.assembly_graph
    certified_graph = BRepAssemblyGraph(
        graph.roots,
        graph.definitions,
        graph.solid_count,
        occurrence_evidence=records,
    )
    certified_geometry = BRepGeometry(
        vertex_points=geometry.vertex_points,
        vertex_exact_points=geometry.vertex_exact_points,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=geometry.edge_face_gap_bounds,
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
        assembly_graph=certified_graph,
    )
    return _model_with_geometry(model, certified_geometry, None)


def _tessellation_policy(
    model: BRepModel, policy: CadImportPolicy, /
) -> BRepTessellationPolicy | None:
    if policy.tessellation is not None:
        return policy.tessellation
    if model.tessellation is None:
        return None
    return BRepTessellationPolicy(
        linear_deflection=model.tessellation.linear_deflection,
        angular_deflection=model.tessellation.angular_deflection,
    )


def _realize_tessellation(
    model: BRepModel, policy: BRepTessellationPolicy | None, /
) -> BRepModel:
    if policy is None:
        return model
    tessellation = tessellate_brep_model(model, policy)
    return _model_with_geometry(model, model.geometry, tessellation)


def certify_placed_model(model: BRepModel, policy: CadImportPolicy, /) -> BRepModel:
    """Certify every rounded occurrence pose against its own native solid extent."""
    if policy.maximum_placement_deviation <= 0.0:
        raise ValueError("Placement certification requires a positive deviation limit.")
    extents: dict[int, float] = {}
    evidence_rows: list[tuple[tuple[str, ...], BRepPlacementEvidence]] = []
    for occurrence in model.geometry.iter_occurrences():
        source = occurrence.placement_evidence
        if source is None:
            continue
        rotation = source.rotation_deviation_bound
        translation = source.translation_deviation_bound
        if rotation == 0.0 and translation == 0.0:
            continue
        extent = extents.get(occurrence.solid_index)
        if extent is None:
            extent = _solid_extent(model, occurrence.solid_index)
            extents[occurrence.solid_index] = extent
        point_bound = fraction_up(
            Fraction(rotation) * Fraction(extent) + Fraction(translation)
        )
        if point_bound > policy.maximum_placement_deviation:
            raise refuse(
                "unsupported-entity",
                "/".join(occurrence.path),
                "Certified placement point deviation exceeds its declared limit.",
                occurrence.path,
            )
        evidence_rows.append(
            (
                occurrence.path,
                BRepPlacementEvidence(
                    rotation,
                    extent,
                    point_bound,
                    translation,
                    entrywise_deviation_bounds=source.entrywise_deviation_bounds,
                ),
            )
        )
    certified = (
        model
        if not evidence_rows
        else _with_occurrence_evidence(model, tuple(sorted(evidence_rows)))
    )
    return _realize_tessellation(certified, _tessellation_policy(model, policy))


__all__ = ["certify_placed_model", "source_placement_evidence"]
