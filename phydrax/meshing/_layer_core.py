#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable advancing-layer/PLC composition, not a second tetrahedralizer."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

import phydrax.discretization._cell_complex as cell_complex_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.geometry._mapped_reference_domain as mapped_domain_owner
import phydrax.geometry._source_identity as source_identity_owner
import phydrax.linalg._exact as exact_linalg_owner
import phydrax.meshing._canonical as canonical_owner
import phydrax.meshing._contracts as contracts_owner
import phydrax.meshing._layer_core_association as association_composition
import phydrax.meshing._layer_core_controls as controls_owner
import phydrax.meshing._layer_core_mapped as mapped_core_owner
import phydrax.meshing._layer_core_periodic as periodic_owner
import phydrax.meshing._layer_core_regions as regions_owner
import phydrax.meshing._layer_core_subdivision as subdivision_owner
import phydrax.meshing._layer_core_topology as layer_topology
import phydrax.meshing._organization as organization_owner
import phydrax.meshing._plc_source as plc_source_owner
import phydrax.meshing._polyhedral_generation as polyhedral_generation_owner
import phydrax.meshing._sizing as sizing_owner
import phydrax.meshing._volume_generation as volume_generation_owner
import phydrax.meshing.providers._native_sources as sources_owner

from ..discretization import CellBlock, CellMesh
from ..typing import checked
from ._contracts import VolumeMeshingSpec


if TYPE_CHECKING:
    from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource
    from ._layer_core_execution import LayerCoreExecution


def _fixed_cap_edge_obstruction(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    cap_polygon_ids: np.ndarray,
    radius_edge_bound: float,
    /,
) -> tuple[int, int, tuple[int, int], Fraction] | None:
    """Prove a fixed cap edge forces a bad adjacent fixed boundary triangle.

    For a triangle on an unsplit edge of length L and altitude h, its
    circumradius divided by its shortest edge is at least L/(4h). Every
    tetrahedron incident to the triangle has at least that ratio. Arithmetic
    uses the exact binary64 input coordinates and compares squared quantities.
    """
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    cap = {int(row) for row in cap_polygon_ids}
    edge_caps: dict[tuple[int, int], int] = {}
    for row in sorted(cap):
        polygon = polygons[row]
        for index in range(3):
            first, second = int(polygon[index]), int(polygon[(index + 1) % 3])
            edge = (min(first, second), max(first, second))
            edge_caps.setdefault(edge, row)
    points = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(complex_.vertices, dtype=np.float64)
    )
    bound = Fraction(float(radius_edge_bound))
    for row, polygon in enumerate(polygons):
        if row in cap:
            continue
        for index in range(3):
            first, second, third = (
                int(polygon[(index + offset) % 3]) for offset in range(3)
            )
            edge = (min(first, second), max(first, second))
            cap_row = edge_caps.get(edge)
            if cap_row is None:
                continue
            direction = tuple(
                b - a for a, b in zip(points[first], points[second], strict=True)
            )
            offset = tuple(
                b - a for a, b in zip(points[first], points[third], strict=True)
            )
            gram = exact_linalg_owner.matmul_exact(
                (direction, offset), tuple(zip(direction, offset, strict=True))
            )
            length_squared = gram[0][0]
            cross_squared = length_squared * gram[1][1] - gram[0][1] * gram[1][0]
            if cross_squared == 0:
                continue
            lower_squared = length_squared * length_squared / (16 * cross_squared)
            if lower_squared > bound * bound:
                return cap_row, row, edge, lower_squared
    return None


def _require_fixed(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    core: volume_generation_owner.VolumeConstruction,
    /,
) -> np.ndarray:
    """Require exact fixed PLC vertices, triangles, oriented sides and closure."""
    if complex_.boundary != "fixed":
        raise ValueError("Layer/core fill requires a fixed PLC boundary.")
    input_vertices = np.asarray(core.input_vertices, dtype=np.int64)
    core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    vertex_count = complex_.vertices.shape[0]
    if (
        input_vertices.shape != (core_points.shape[0],)
        or np.any(input_vertices < -1)
        or np.any(input_vertices >= vertex_count)
    ):
        raise ValueError(
            "Native volume input-vertex ancestry is incompatible with the PLC."
        )
    retained = input_vertices >= 0
    if not np.array_equal(np.sort(input_vertices[retained]), np.arange(vertex_count)):
        raise ValueError("Fixed PLC fill must retain every input vertex exactly once.")
    if not layer_topology.bitwise_equal_points(
        core_points[retained],
        np.asarray(complex_.vertices, dtype=np.float64)[input_vertices[retained]],
    ):
        raise ValueError("Fixed PLC fill moved an authored input vertex.")
    core_regions = np.asarray(core.cell_regions, dtype=np.int64)
    if (
        core_regions.shape
        != np.sort(np.asarray(core.mesh.entity_set(3).entity_ids)).shape
    ):
        raise ValueError("Native volume regions must cover every recovered core cell.")
    if np.any(core_regions < 0) or np.any(core_regions >= complex_.region_count):
        raise ValueError("Native volume fill introduced an undeclared core region.")
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    boundary_faces = np.asarray(core.boundary_faces, dtype=np.int64)
    face_source_facets = np.asarray(core.face_source_facets, dtype=np.int64)
    face_orientations = np.asarray(core.face_orientations, dtype=np.int8)
    if (
        boundary_faces.ndim != 2
        or boundary_faces.shape[1] != 3
        or face_source_facets.shape != (boundary_faces.shape[0],)
        or face_orientations.shape != face_source_facets.shape
    ):
        raise ValueError(
            "Native volume boundary ancestry arrays have incompatible shapes."
        )
    if (
        np.any(boundary_faces < 0)
        or np.any(boundary_faces >= core_points.shape[0])
        or np.any(face_source_facets < 0)
        or np.any(face_source_facets >= complex_.facet_count)
        or np.any(~np.isin(face_orientations, (-1, 1)))
    ):
        raise ValueError(
            "Native volume boundary ancestry contains undeclared identities."
        )
    polygon_by_key: dict[tuple[int, ...], int] = {}
    for polygon_index, polygon in enumerate(polygons):
        key = tuple(sorted(int(value) for value in polygon))
        if key in polygon_by_key:
            raise ValueError("The fixed PLC contains duplicate oriented polygons.")
        polygon_by_key[key] = polygon_index
    polygon_rows = np.full(polygons.shape[0], -1, dtype=np.int64)
    for face_row, face in enumerate(boundary_faces):
        inherited = input_vertices[face]
        if np.any(inherited < 0):
            raise ValueError(
                "Fixed PLC boundary facets cannot contain generated vertices."
            )
        polygon_index = polygon_by_key.get(
            tuple(sorted(int(value) for value in inherited))
        )
        if polygon_index is None or polygon_rows[polygon_index] >= 0:
            raise ValueError("A recovered boundary triangle changed its source polygon.")
        if int(face_source_facets[face_row]) != int(
            complex_.polygon_facets[polygon_index]
        ):
            raise ValueError("A recovered boundary triangle changed its source facet.")
        polygon = polygons[polygon_index]
        oriented = tuple(int(value) for value in inherited)
        original = tuple(int(value) for value in polygon)
        if layer_topology.cycle(oriented) == layer_topology.cycle(original):
            sign = 1
        elif layer_topology.cycle(oriented) == layer_topology.cycle(
            tuple(reversed(original))
        ):
            sign = -1
        else:
            raise ValueError("A recovered boundary triangle changed its source vertices.")
        if sign != int(face_orientations[face_row]):
            raise ValueError(
                "Native boundary orientation disagrees with its source facet."
            )
        polygon_rows[polygon_index] = face_row
    if np.any(polygon_rows < 0):
        raise ValueError("Fixed PLC fill must retain every source polygon exactly once.")
    input_rows = np.full(vertex_count, -1, dtype=np.int64)
    input_rows[input_vertices[retained]] = np.flatnonzero(retained)
    core_regions = np.asarray(core.cell_regions, dtype=np.int64)
    face_table = layer_topology.faces(core.mesh, core_regions)
    source_keys: set[tuple[int, ...]] = set()
    for polygon_index, polygon in enumerate(polygons):
        core_vertices = tuple(int(input_rows[int(vertex)]) for vertex in polygon)
        key = tuple(sorted(core_vertices))
        source_keys.add(key)
        incidents = face_table.get(key, [])
        facet = int(complex_.polygon_facets[polygon_index])
        expected = np.asarray(complex_.facet_regions[facet], dtype=np.int64)
        if len(incidents) != int(np.count_nonzero(expected >= 0)):
            raise ValueError("Native core recovery split or dropped a fixed PLC facet.")
        for side, region in enumerate(expected):
            if region < 0:
                continue
            oriented = (
                tuple(polygon.tolist()) if side == 1 else tuple(polygon[::-1].tolist())
            )
            if not any(
                face.region == int(region)
                and layer_topology.cycle(
                    tuple(int(input_vertices[vertex]) for vertex in face.vertices)
                )
                == layer_topology.cycle(oriented)
                for face in incidents
            ):
                raise ValueError("Native core recovery changed an oriented PLC side.")
    if any(
        len(incidents) == 1 and key not in source_keys
        for key, incidents in face_table.items()
    ):
        raise ValueError(
            "Native core recovery introduced an undeclared external boundary."
        )
    return input_vertices


def _require_selectively_conforming(
    complex_: plc_source_owner.PiecewiseLinearComplex,
    core: volume_generation_owner.VolumeConstruction,
    protected_facets: np.ndarray,
    /,
) -> np.ndarray:
    """Prove protected execution facets and witnessed conforming source sides."""
    protected = np.asarray(protected_facets)
    if (
        complex_.boundary != "conforming"
        or protected.shape != (complex_.facet_count,)
        or protected.dtype != np.dtype(np.bool_)
    ):
        raise ValueError("Selective core recovery needs a conforming PLC and facet mask.")
    points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    ancestry = np.asarray(core.input_vertices, dtype=np.int64)
    count = complex_.vertices.shape[0]
    if (
        ancestry.shape != (len(points),)
        or np.any(ancestry < -1)
        or np.any(ancestry >= count)
    ):
        raise ValueError("Selective recovery has invalid execution-vertex ancestry.")
    retained = ancestry >= 0
    if not np.array_equal(np.sort(ancestry[retained]), np.arange(count)):
        raise ValueError(
            "Selective recovery must retain every execution vertex exactly once."
        )
    if not layer_topology.bitwise_equal_points(
        points[retained],
        np.asarray(complex_.vertices, dtype=np.float64)[ancestry[retained]],
    ):
        raise ValueError("Selective recovery moved an execution input vertex.")
    faces = np.asarray(core.boundary_faces, dtype=np.int64)
    sources = np.asarray(core.face_source_facets, dtype=np.int64)
    orientations = np.asarray(core.face_orientations, dtype=np.int8)
    strata = np.asarray(core.witness_strata, dtype=np.int8)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or sources.shape != (len(faces),)
        or orientations.shape != sources.shape
        or strata.shape != ancestry.shape
        or np.any(faces < 0)
        or np.any(faces >= len(points))
        or np.any(sources < 0)
        or np.any(sources >= complex_.facet_count)
        or np.any(~np.isin(orientations, (-1, 1)))
    ):
        raise ValueError(
            "Selective recovery has invalid boundary or source witness extents."
        )
    if (
        len(core.root_vertex_stencils) != len(points)
        or len(core.root_exact_points) != len(points)
        or core.candidate_quality_id is None
    ):
        raise ValueError(
            "Selective recovery requires re-evaluated original-root candidate evidence."
        )
    for row in np.unique(faces):
        if (
            core.root_vertex_stencils[int(row)] is None
            or core.root_exact_points[int(row)] is None
        ):
            raise ValueError(
                "A conforming boundary vertex lacks its original-root witness."
            )
    regions = np.asarray(core.cell_regions, dtype=np.int64)
    if (
        regions.shape != np.sort(np.asarray(core.mesh.entity_set(3).entity_ids)).shape
        or np.any(regions < 0)
        or np.any(regions >= complex_.region_count)
    ):
        raise ValueError("Selective recovery changed declared core material regions.")
    table = layer_topology.faces(core.mesh, regions)
    keys: set[tuple[int, ...]] = set()
    protected_rows: dict[int, list[int]] = {}
    for row, face in enumerate(faces.tolist()):
        key = tuple(sorted(int(vertex) for vertex in face))
        if key in keys:
            raise ValueError("Selective recovery repeated a boundary triangle.")
        keys.add(key)
        facet = int(sources[row])
        if protected[facet]:
            protected_rows.setdefault(facet, []).append(row)
        expected = np.asarray(complex_.facet_regions[facet], dtype=np.int64)
        incidents = table.get(key, ())
        if len(incidents) != int(np.count_nonzero(expected >= 0)):
            raise ValueError(
                "Selective recovery changed a boundary facet's material incidence."
            )
        face_vertices = tuple(int(vertex) for vertex in face)
        for side, region in enumerate(expected):
            if region < 0:
                continue
            forward = (side == 1) == (int(orientations[row]) == 1)
            desired = face_vertices if forward else tuple(reversed(face_vertices))
            if not any(
                incident.region == int(region)
                and layer_topology.cycle(incident.vertices)
                == layer_topology.cycle(desired)
                for incident in incidents
            ):
                raise ValueError(
                    "Selective recovery changed an oriented source region side."
                )
    polygons = np.asarray(complex_.polygon_vertices, dtype=np.int64).reshape(-1, 3)
    polygon_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)
    for facet in np.flatnonzero(protected):
        original = np.flatnonzero(polygon_facets == facet)
        recovered = protected_rows.get(int(facet), [])
        if len(original) != len(recovered):
            raise ValueError(
                "Native recovery split or dropped a protected execution facet."
            )
        expected = {
            tuple(sorted(int(vertex) for vertex in polygons[row])) for row in original
        }
        actual = set()
        for row in recovered:
            inherited = ancestry[faces[row]]
            if np.any(inherited < 0):
                raise ValueError("A protected facet contains a generated vertex.")
            actual.add(tuple(sorted(int(vertex) for vertex in inherited)))
        if actual != expected:
            raise ValueError("Native recovery changed a protected execution facet.")
    if any(len(incidents) == 1 and key not in keys for key, incidents in table.items()):
        raise ValueError("Selective recovery introduced an undeclared exterior boundary.")
    return ancestry


def core_request(
    specification: VolumeMeshingSpec,
    layer_controls: tuple[contracts_owner.BoundaryLayerControl, ...] = (),
    /,
    *,
    physical_scale: float = 1.0,
) -> VolumeMeshingSpec:
    """Combine requested core caps without rewriting the source size semantics."""
    if not np.isfinite(physical_scale) or physical_scale <= 0.0:
        raise ValueError("physical_scale must be positive and finite.")
    bounds = tuple(
        control.core_maximum_size / physical_scale
        for control in (*specification.layer_controls, *layer_controls)
        if control.core_maximum_size is not None
    )
    if not bounds:
        return specification
    original = specification.size_controls[0]
    match original:
        case sizing_owner.UniformSizeControl() if len(specification.size_controls) == 1:
            pass
        case _:
            raise ValueError(
                "A layer/core request requires one uniform core size control."
            )
    cap = min(bounds)
    if original.strength is sizing_owner.SizeControlStrength.HARD and (
        (original.minimum_size is not None and original.minimum_size > cap)
        or original.target_size
        > cap
        + specification.size_compliance.absolute_tolerance
        + specification.size_compliance.relative_tolerance * original.target_size
    ):
        raise contracts_owner.MeshingFailure(
            contracts_owner.MeshingFailureCategory.CONTROL_CONFLICT,
            "The hard core size request conflicts with the layer control's core maximum size.",
        )
    target = min(original.target_size, cap)
    minimum = original.minimum_size
    if (
        original.strength is sizing_owner.SizeControlStrength.SOFT
        and minimum is not None
        and minimum > target
    ):
        minimum = None
    size = sizing_owner.UniformSizeControl(
        original.scope,
        target,
        minimum_size=minimum,
        maximum_size=min(cap, original.maximum_size)
        if original.maximum_size is not None
        else cap,
        maximum_growth_rate=original.maximum_growth_rate,
        strength=original.strength,
        priority=original.priority,
    )
    return VolumeMeshingSpec(
        specification.target,
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=(size,),
        protected_features=specification.protected_features,
        region_controls=specification.region_controls,
        patch_controls=specification.patch_controls,
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=specification.layer_controls,
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


def _execution_core_request(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    execution: LayerCoreExecution,
    /,
) -> VolumeMeshingSpec:
    """Lower authored controls to an internal execution carrier, without changing them."""
    from ._controls import ProtectedFeature
    from ._scope import MeshingScope

    complex_ = execution.complex
    original = source.complex
    boundary = specification.boundary_scope
    scope = MeshingScope(
        boundary.source_id,
        boundary.source_revision,
        boundary.entity_kind,
        2,
        f"{complex_.complex_id}:facets",
        np.asarray(complex_.facet_entity_ids, dtype=np.int64),
    )
    if len(specification.size_controls) != 1 or not isinstance(
        specification.size_controls[0], sizing_owner.UniformSizeControl
    ):
        raise ValueError(
            "Refined layer/core execution requires its original uniform size control."
        )
    control = specification.size_controls[0]
    size = sizing_owner.UniformSizeControl(
        scope,
        control.target_size,
        minimum_size=control.minimum_size,
        maximum_size=control.maximum_size,
        maximum_growth_rate=control.maximum_growth_rate,
        strength=control.strength,
        priority=control.priority,
    )
    original_facet_rows = np.asarray(original.polygon_facets, dtype=np.int64)
    original_facet_ids = np.asarray(original.facet_entity_ids, dtype=np.int64)
    features = []
    for feature in specification.protected_features:
        requested = {int(value) for value in np.asarray(feature.scope.entity_ids)}
        dimension = feature.scope.entity_dimension
        if dimension == 2:
            selected = []
            for row, candidates in enumerate(execution.source_polygon_candidates):
                identities = {
                    int(original_facet_ids[original_facet_rows[index]])
                    for index in candidates
                }
                if identities & requested and not identities <= requested:
                    raise ValueError(
                        "A protected source facet is crossed by an execution facet."
                    )
                if identities and identities <= requested:
                    selected.append(int(complex_.facet_entity_ids[row]))
            set_id = f"{complex_.complex_id}:facets"
        elif dimension == 1:
            original_edges = {
                int(identifier): tuple(edge)
                for identifier, edge in zip(
                    original.plc_edge_entity_ids, original.plc_edge_vertices, strict=True
                )
            }
            if not requested <= set(original_edges):
                raise ValueError("A protected curve is absent from the original PLC.")
            selected = []
            for row, edge in enumerate(complex_.plc_edge_vertices):
                supports = {
                    root
                    for vertex in edge
                    for root, weight in execution.vertex_stencils[vertex]
                    if weight
                }
                if any(
                    supports <= set(original_edges[identifier])
                    for identifier in requested
                ):
                    selected.append(int(complex_.plc_edge_entity_ids[row]))
            set_id = f"{complex_.complex_id}:segments"
        elif dimension == 0:
            original_ids = np.asarray(original.vertex_entity_ids, dtype=np.int64)
            if not requested <= set(original_ids.tolist()):
                raise ValueError("A protected vertex is absent from the original PLC.")
            selected = [
                int(complex_.vertex_entity_ids[row])
                for row, terms in enumerate(execution.vertex_stencils)
                if len(terms) == 1
                and terms[0][1] == 1
                and int(original_ids[terms[0][0]]) in requested
            ]
            set_id = f"{complex_.complex_id}:vertices"
        else:
            raise ValueError("Unsupported protected execution feature dimension.")
        if not selected:
            raise ValueError("Execution carrier omitted an original protected feature.")
        translated = MeshingScope(
            boundary.source_id,
            boundary.source_revision,
            boundary.entity_kind,
            dimension,
            set_id,
            np.asarray(sorted(set(selected)), dtype=np.int64),
        )
        features.append(
            ProtectedFeature(
                translated,
                feature.feature_kind,
                maximum_deviation=feature.maximum_deviation,
                hard=feature.hard,
            )
        )
    return VolumeMeshingSpec(
        specification.target,
        scope,
        specification.fill_strategy,
        size_controls=(size,),
        protected_features=tuple(features),
        region_controls=specification.region_controls,
        patch_controls=(),
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=(),
        periodic_constraints=(),
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


def _compose_polyhedral_layer_topology(
    layers: sources_owner.NativeLayerCoreSource,
    core: polyhedral_generation_owner.PolyhedralConstruction,
    mapping: np.ndarray,
    initial_points: np.ndarray,
    input_vertices: np.ndarray,
    source_revision: str,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compose exact polyhedral core loops beside immutable standard layer blocks."""
    core_mesh = core.mesh
    connectivity = core_mesh.connectivity
    if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
        raise TypeError("Native restricted-power core lost polyhedral connectivity.")
    core_points = np.asarray(core_mesh.coordinates, dtype=np.float64)
    vertex_carriers = np.asarray(core.restricted_cells.vertex_carriers, dtype=np.int64)
    carrier_ids = np.asarray(core.carrier.mesh.vertex_global_ids, dtype=np.int64)
    if (
        vertex_carriers.shape != (core_points.shape[0], 4)
        or input_vertices.shape != carrier_ids.shape
        or mapping.shape != (layers.complex.vertices.shape[0],)
        or np.any(mapping < 0)
    ):
        raise ValueError(
            "Polyhedral source-carrier ancestry does not align with the PLC."
        )
    carrier_rows = {int(identifier): row for row, identifier in enumerate(carrier_ids)}
    source_vertex_rows = np.full((layers.complex.vertices.shape[0],), -1, dtype=np.int64)
    core_vertex_map = np.full((core_points.shape[0],), -1, dtype=np.int64)
    for core_row in range(vertex_carriers.shape[0]):
        carriers = vertex_carriers[core_row]
        active = carriers[carriers >= 0]
        if active.size != 1:
            continue
        carrier_row = carrier_rows.get(int(active[0]))
        if carrier_row is None:
            raise ValueError("A power vertex names an absent fixed-carrier vertex.")
        source_vertex = int(input_vertices[carrier_row])
        if source_vertex < 0:
            continue
        if source_vertex_rows[source_vertex] >= 0:
            raise ValueError("A fixed PLC vertex has duplicate power-cell descendants.")
        source_vertex_rows[source_vertex] = core_row
        core_vertex_map[core_row] = mapping[source_vertex]
    if np.any(source_vertex_rows < 0):
        raise ValueError("Restricted-power core omitted a fixed PLC source vertex.")
    layer_mesh = layers.layers.mesh
    generated = np.flatnonzero(core_vertex_map < 0)
    core_vertex_map[generated] = initial_points.shape[0] + np.arange(
        generated.size, dtype=np.int64
    )
    points = np.concatenate((initial_points, core_points[generated]))
    layer_cells = np.sort(np.asarray(layer_mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_cells = np.sort(np.asarray(core_mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_start = int(layer_cells[-1]) + 1
    if core_cells.size and core_start + core_cells.size - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer cell identity range cannot contain core cells."
        )
    initial_extra = initial_points.shape[0] - layer_mesh.coordinates.shape[0]
    if initial_extra < 0:
        raise ValueError("Layer/core identity preparation lost accepted layer vertices.")
    next_vertex = (
        int(np.max(np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64))) + 1
        if layer_mesh.vertex_global_ids.size
        else 0
    )
    extra_vertices = initial_extra + generated.size
    if extra_vertices and next_vertex + extra_vertices - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer vertex identity range cannot contain core vertices."
        )
    vertex_ids = np.concatenate(
        (
            np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64),
            next_vertex + np.arange(extra_vertices, dtype=np.int64),
        )
    )
    standard_blocks = tuple(
        CellBlock(
            f"layer:{block.name}",
            block.cell_kind,
            np.asarray(block.vertices, dtype=np.int64),
            global_ids=np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in layer_mesh.blocks
    )
    core_cell_ids = core_start + np.arange(core_cells.size, dtype=np.int64)
    cell_rows = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    oriented_loops = layer_topology.polyhedral_cell_faces(core_mesh)
    loops_by_id = {
        int(identifier): oriented_loops[row]
        for row, identifier in enumerate(cell_rows.tolist())
    }
    polyhedral_cells = tuple(
        tuple(
            np.asarray([core_vertex_map[vertex] for vertex in loop], dtype=np.int64)
            for loop in loops_by_id[int(identifier)]
        )
        for identifier in core_cells.tolist()
    )
    if len(core_mesh.blocks) == 0 or any(
        block.cell_kind != "polyhedron" for block in core_mesh.blocks
    ):
        raise TypeError("Restricted-power core blocks must all be polyhedral.")
    mesh = CellMesh.from_mixed_3d(
        points,
        standard_blocks,
        polyhedra={"core:power": polyhedral_cells},
        vertex_global_ids=vertex_ids,
        polyhedral_cell_global_ids={"core:power": core_cell_ids},
        numeric_version=source_revision,
    )
    layer_regions = np.asarray(layers.layer_regions, dtype=np.int64)
    core_region_by_id = {
        int(identifier): int(region)
        for identifier, region in zip(
            np.asarray(core_mesh.entity_set(3).entity_ids, dtype=np.int64),
            np.asarray(core.cell_regions, dtype=np.int64),
            strict=True,
        )
    }
    core_regions = np.asarray(
        [core_region_by_id[int(identifier)] for identifier in core_cells],
        dtype=np.int64,
    )
    region_map = np.asarray(layers.core_region_map, dtype=np.int64)
    combined_regions = np.concatenate((layer_regions, region_map[core_regions]))
    if (
        combined_regions.shape
        != np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)).shape
    ):
        raise ValueError("Layer and core region records do not cover the combined mesh.")
    return mesh, core_vertex_map, combined_regions, layer_cells, core_cell_ids


def _polyhedral_periodic_ancestry(
    source: sources_owner.NativeLayerCoreSource,
    core: polyhedral_generation_owner.PolyhedralConstruction,
    mapping: np.ndarray,
    initial_points: np.ndarray,
    input_vertices: np.ndarray,
    core_vertex_map: np.ndarray,
    source_ancestry: tuple[np.ndarray, np.ndarray] | None,
    /,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align the restricted-power quotient with authored layer/core seam winding."""
    core_periodic = core.mesh.periodic_topology
    if source_ancestry is None:
        if core_periodic is not None:
            raise ValueError(
                "Polyhedral core introduced periodic topology without layer ancestry."
            )
        return None
    layer_periodic = source.layers.mesh.periodic_topology
    if layer_periodic is None or core_periodic is None:
        raise ValueError(
            "Periodic polyhedral layer/core fill requires exact layer and core quotient topology."
        )
    base_roots, base_shifts = source_ancestry
    point_count = max(
        initial_points.shape[0], int(np.max(core_vertex_map, initial=-1)) + 1
    )
    if base_roots.shape != (initial_points.shape[0],) or base_shifts.shape != (
        initial_points.shape[0],
        layer_periodic.cell.rank,
    ):
        raise ValueError(
            "Authored layer/core periodic ancestry has incompatible extents."
        )
    if periodic_exact_owner.identification_id(
        core_periodic.cell
    ) != periodic_exact_owner.identification_id(layer_periodic.cell):
        raise ValueError(
            "Polyhedral core and accepted layers use different periodic cells."
        )
    roots = np.arange(point_count, dtype=np.int64)
    shifts = np.zeros((point_count, layer_periodic.cell.rank), dtype=np.int64)
    roots[: base_roots.size] = base_roots
    shifts[: base_shifts.shape[0]] = base_shifts
    connectivity = core.mesh.connectivity
    if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
        raise TypeError("Periodic power core must retain polyhedral connectivity.")
    carrier_ids = np.asarray(core.carrier.mesh.vertex_global_ids, dtype=np.int64)
    carrier_rows = {int(identifier): row for row, identifier in enumerate(carrier_ids)}
    source_vertex_rows = np.full((source.complex.vertices.shape[0],), -1, dtype=np.int64)
    periodic_carriers = np.asarray(core.restricted_cells.vertex_carriers, dtype=np.int64)
    for core_row in range(periodic_carriers.shape[0]):
        carriers = periodic_carriers[core_row]
        active = carriers[carriers >= 0]
        if active.size != 1:
            continue
        carrier_row = carrier_rows.get(int(active[0]))
        if carrier_row is None:
            raise ValueError(
                "A periodic power vertex names an absent PLC carrier vertex."
            )
        source_vertex = int(input_vertices[carrier_row])
        if source_vertex >= 0:
            if source_vertex_rows[source_vertex] >= 0:
                raise ValueError("A periodic PLC vertex has duplicate power descendants.")
            source_vertex_rows[source_vertex] = core_row
    if np.any(source_vertex_rows < 0):
        raise ValueError("Periodic power core omitted an authored PLC vertex.")
    core_roots = np.asarray(core_periodic.vertex_representatives, dtype=np.int64)
    core_shifts = np.asarray(core_periodic.vertex_shifts, dtype=np.int64)
    if (
        core_roots.shape != core_vertex_map.shape
        or core_shifts.shape != (core_vertex_map.size, layer_periodic.cell.rank)
        or np.any(core_roots < 0)
        or np.any(core_roots >= core_vertex_map.size)
    ):
        raise ValueError("Periodic power-core quotient arrays do not match its vertices.")
    orders = periodic_exact_owner.identification_orders(layer_periodic.cell)
    offsets: dict[int, np.ndarray] = {}
    for source_vertex, core_row in enumerate(source_vertex_rows.tolist()):
        base_row = int(mapping[source_vertex])
        core_root = int(core_roots[core_row])
        target_root = int(core_vertex_map[core_root])
        if not 0 <= base_row < base_roots.size or not 0 <= target_root < base_roots.size:
            raise ValueError(
                "A periodic PLC seam root is absent from the authored layer bank."
            )
        if base_roots[base_row] != base_roots[target_root]:
            raise ValueError("Power-core periodic roots contradict authored layer seams.")
        offset = periodic_exact_owner.reduce_periodic_exponents(
            base_shifts[base_row] - core_shifts[core_row], orders
        )
        previous = offsets.get(core_root)
        if previous is not None and not np.array_equal(previous, offset):
            raise ValueError(
                "Power-core periodic winding contradicts authored source ancestry."
            )
        offsets[core_root] = offset
    for core_row, core_root_value in enumerate(core_roots.tolist()):
        target_row = int(core_vertex_map[core_row])
        target_root = int(core_vertex_map[int(core_root_value)])
        roots[target_row] = roots[target_root]
        offset = offsets.get(
            int(core_root_value), np.zeros((layer_periodic.cell.rank,), dtype=np.int64)
        )
        shifts[target_row] = periodic_exact_owner.reduce_periodic_exponents(
            core_shifts[core_row] + offset, orders
        )
    for source_vertex, core_row in enumerate(source_vertex_rows.tolist()):
        target_row = int(core_vertex_map[core_row])
        base_row = int(mapping[source_vertex])
        if roots[target_row] != base_roots[base_row] or not np.array_equal(
            shifts[target_row], base_shifts[base_row]
        ):
            raise ValueError("Power-core quotient changed an authored PLC vertex orbit.")
    return roots, shifts


def _compose_layer_topology(
    layers: sources_owner.NativeLayerCoreSource,
    core: (
        volume_generation_owner.VolumeConstruction
        | polyhedral_generation_owner.PolyhedralConstruction
    ),
    mapping: np.ndarray,
    initial_points: np.ndarray,
    input_vertices: np.ndarray,
    source_revision: str,
    /,
    *,
    layer_partition: subdivision_owner.NestedLayerPartition | None = None,
) -> tuple[CellMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep accepted layer topology and PLC identities, appending only native interior nodes."""
    if isinstance(core, polyhedral_generation_owner.PolyhedralConstruction):
        if layer_partition is not None:
            raise ValueError("Nested layer subdivision requires a simplex core.")
        return _compose_polyhedral_layer_topology(
            layers,
            core,
            mapping,
            initial_points,
            input_vertices,
            source_revision,
        )
    core_points = np.asarray(core.mesh.coordinates, dtype=np.float64)
    generated = np.flatnonzero(input_vertices < 0)
    core_vertex_map = np.empty(core_points.shape[0], dtype=np.int64)
    retained = input_vertices >= 0
    core_vertex_map[retained] = mapping[input_vertices[retained]]
    core_vertex_map[generated] = initial_points.shape[0] + np.arange(
        generated.size, dtype=np.int64
    )
    points = np.concatenate((initial_points, core_points[generated]))
    layer_mesh = layers.layers.mesh
    if layer_partition is not None:
        layer_mesh = CellMesh(
            layer_partition.coordinates,
            (
                CellBlock(
                    "refined-prisms",
                    "prism",
                    layer_partition.prism_vertices,
                    global_ids=layer_partition.prism_ids,
                ),
            ),
            vertex_global_ids=layer_partition.vertex_ids,
            numeric_version=source_revision,
        )
        if not layer_topology.bitwise_equal_points(
            np.asarray(
                initial_points[: layer_partition.coordinates.shape[0]], dtype=np.float64
            ),
            np.asarray(layer_partition.coordinates, dtype=np.float64),
        ):
            raise ValueError(
                "The execution PLC changed an accepted refined layer vertex."
            )
    layer_cells = np.sort(np.asarray(layer_mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_cells = np.sort(np.asarray(core.mesh.entity_set(3).entity_ids, dtype=np.int64))
    core_start = int(layer_cells[-1]) + 1
    if core_cells.size and core_start + core_cells.size - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer cell identity range cannot contain core cells."
        )
    initial_extra = initial_points.shape[0] - layer_mesh.coordinates.shape[0]
    if initial_extra < 0:
        raise ValueError("Layer/core identity preparation lost accepted layer vertices.")
    if layer_mesh.vertex_global_ids.size:
        next_vertex = (
            int(np.max(np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64))) + 1
        )
    else:
        next_vertex = 0
    extra_vertices = initial_extra + generated.size
    if extra_vertices and next_vertex + extra_vertices - 1 > np.iinfo(np.int64).max:
        raise ValueError(
            "The accepted layer vertex identity range cannot contain core vertices."
        )
    vertex_ids = np.concatenate(
        (
            np.asarray(layer_mesh.vertex_global_ids, dtype=np.int64),
            next_vertex + np.arange(extra_vertices, dtype=np.int64),
        )
    )
    blocks = [
        CellBlock(
            f"layer:{block.name}",
            block.cell_kind,
            np.asarray(block.vertices, dtype=np.int64),
            global_ids=np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in layer_mesh.blocks
    ]
    core_cell_ids = core_start + np.arange(core_cells.size, dtype=np.int64)
    for block in core.mesh.blocks:
        old_ids = np.asarray(block.global_ids, dtype=np.int64)
        positions = np.searchsorted(core_cells, old_ids)
        blocks.append(
            CellBlock(
                f"core:{block.name}",
                block.cell_kind,
                core_vertex_map[np.asarray(block.vertices, dtype=np.int64)],
                global_ids=core_cell_ids[positions],
            )
        )
    mesh = CellMesh(
        points,
        tuple(blocks),
        vertex_global_ids=vertex_ids,
        numeric_version=source_revision,
    )
    layer_regions = np.asarray(layers.layer_regions, dtype=np.int64)
    if layer_partition is not None:
        original_ids = np.sort(
            np.asarray(layers.layers.mesh.entity_set(3).entity_ids, dtype=np.int64)
        )
        parent_rows = np.searchsorted(original_ids, layer_partition.parent_cell_ids)
        if np.any(parent_rows >= original_ids.size) or not np.array_equal(
            original_ids[parent_rows], layer_partition.parent_cell_ids
        ):
            raise ValueError("Refined layer cells lost their original material parents.")
        layer_regions = layer_regions[parent_rows]
    core_map = np.asarray(layers.core_region_map, dtype=np.int64)
    combined_regions = np.concatenate(
        (layer_regions, core_map[np.asarray(core.cell_regions, dtype=np.int64)])
    )
    if combined_regions.shape != np.sort(np.asarray(mesh.entity_set(3).entity_ids)).shape:
        raise ValueError("Layer and core region records do not cover the combined mesh.")
    return mesh, core_vertex_map, combined_regions, layer_cells, core_cell_ids


def _compose_organization(
    mesh: CellMesh,
    regions: np.ndarray,
    layer_ids: np.ndarray,
    core_ids: np.ndarray,
    layers: sources_owner.NativeLayerCoreSource,
    complex_: plc_source_owner.PiecewiseLinearComplex,
    mapping: np.ndarray,
    cap_ids: np.ndarray,
    polygons: np.ndarray,
    layer_faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    combined_faces: dict[tuple[int, ...], list[layer_topology.LayerCoreFace]],
    region_ids: tuple[str, ...],
    /,
    *,
    polyhedral_core: polyhedral_generation_owner.PolyhedralConstruction | None = None,
    core_vertex_map: np.ndarray | None = None,
) -> tuple[
    tuple[organization_owner.MeshZone, ...],
    tuple[organization_owner.MeshPatch, ...],
    tuple[organization_owner.MeshLabel, ...],
    tuple[organization_owner.MeshAttribute, ...],
]:
    """Bind region zones, exact core patches, layer intervals and combined interfaces."""
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    order = np.argsort(regions, kind="stable")
    bounds = np.searchsorted(
        regions[order], np.arange(len(region_ids) + 1, dtype=np.int64)
    )
    zones = tuple(
        organization_owner.MeshZone(
            name,
            organization_owner.MeshZoneRole.REGION,
            layer_topology.mesh_entity_scope(mesh, 3, cells[order[start:stop]]),
        )
        for name, start, stop in zip(region_ids, bounds[:-1], bounds[1:], strict=True)
        if stop > start
    )
    labels = (
        organization_owner.MeshLabel(
            "boundary-layer", layer_topology.mesh_entity_scope(mesh, 3, layer_ids)
        ),
        organization_owner.MeshLabel(
            "core", layer_topology.mesh_entity_scope(mesh, 3, core_ids)
        ),
    )
    layer_by_id = dict(
        zip(
            np.sort(
                np.asarray(layers.layers.mesh.entity_set(3).entity_ids, dtype=np.int64)
            ).tolist(),
            np.asarray(layers.layers.layer_index, dtype=np.int64).tolist(),
            strict=True,
        )
    )
    layer_values = np.asarray(
        [layer_by_id[int(identifier)] for identifier in layer_ids], dtype=np.int64
    )
    column_values = np.asarray(layers.layers.column_index, dtype=np.int64)
    if column_values.shape != layer_ids.shape:
        raise ValueError(
            "Source column identities must identify each accepted layer cell."
        )
    attributes = (
        organization_owner.MeshAttribute(
            "layer_index",
            organization_owner.MeshAttributeRole.MARKER,
            layer_topology.mesh_entity_scope(mesh, 3, cells),
            np.concatenate((layer_values, np.full(core_ids.size, -1, dtype=np.int64))),
        ),
        organization_owner.MeshAttribute(
            "layer_column",
            organization_owner.MeshAttributeRole.MARKER,
            layer_topology.mesh_entity_scope(mesh, 3, cells),
            np.concatenate((column_values, np.full(core_ids.size, -1, dtype=np.int64))),
        ),
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_keys = dict(
        zip(canonical_owner._entity_vertex_keys(mesh, 2), face_ids.tolist(), strict=True)
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)

    def face_id(row_vertices: tuple[int, ...], /) -> int:
        vertices = np.asarray(row_vertices, dtype=np.int64)
        key = tuple(sorted(int(value) for value in vertex_ids[vertices]))
        return face_keys[key]

    faces_by_facet: dict[int, set[int]] = {
        facet: set() for facet in range(complex_.facet_count)
    }
    if polyhedral_core is None:
        for polygon, facet_value in zip(polygons, complex_.polygon_facets, strict=True):
            facet = int(facet_value)
            faces_by_facet[facet].add(face_id(tuple(mapping[polygon].tolist())))
    else:
        if core_vertex_map is None:
            raise ValueError(
                "Polyhedral organization requires exact core vertex ancestry."
            )
        connectivity = polyhedral_core.mesh.connectivity
        if not isinstance(connectivity, cell_complex_owner.PolyhedralConnectivity):
            raise TypeError(
                "Polyhedral organization requires canonical face connectivity."
            )
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        source_facets = np.asarray(polyhedral_core.face_source_facets, dtype=np.int64)
        if source_facets.shape != (connectivity.face_count,):
            raise ValueError(
                "Power-core face ancestry does not align with its connectivity."
            )
        for face, facet_value in enumerate(source_facets.tolist()):
            if facet_value < 0:
                continue
            first, last = offsets[face : face + 2]
            loop = tuple(
                int(value) for value in core_vertex_map[values[first:last]].tolist()
            )
            faces_by_facet[facet_value].add(face_id(loop))
    cap_keys = {tuple(sorted(mapping[polygons[index]].tolist())) for index in cap_ids}
    # An open wall also exposes layer side strips. The wall patch is only the
    # exterior layer faces lying wholly on the wall; a strip rising from rim
    # edge (a, b) joins the facet of the fixed PLC polygon that holds the tops
    # of the columns grown from a and b.
    layer_count = int(np.asarray(layers.layers.mesh.coordinates).shape[0])
    column_sources = np.asarray(layers.layers.vertex_source_vertex_ids, dtype=np.int64)
    wall_vertices = np.asarray(layers.layers.wall_vertices, dtype=np.int64)
    on_wall = frozenset(wall_vertices[wall_vertices >= 0].tolist())
    cap_polygons = frozenset(np.asarray(cap_ids, dtype=np.int64).tolist())
    rim_facets: dict[tuple[int, int], set[int]] = {}
    for index, (polygon, facet_value) in enumerate(
        zip(polygons, complex_.polygon_facets, strict=True)
    ):
        if index in cap_polygons:
            continue
        rows = mapping[polygon].tolist()
        for corner, row in enumerate(rows):
            following = rows[(corner + 1) % len(rows)]
            if row >= layer_count or following >= layer_count:
                continue
            ends = (int(column_sources[row]), int(column_sources[following]))
            if min(ends) >= 0 and ends[0] != ends[1]:
                rim_facets.setdefault((min(ends), max(ends)), set()).add(int(facet_value))
    wall_rows: list[int] = []
    side_rows: list[int] = []
    for key, incidents in layer_faces.items():
        if len(incidents) != 1 or key in cap_keys:
            continue
        if on_wall.issuperset(key):
            wall_rows.append(face_id(key))
            continue
        ends = sorted({int(column_sources[row]) for row in key})
        facets = rim_facets.get((ends[0], ends[-1]), set()) if len(ends) == 2 else set()
        if len(facets) == 1 and ends[0] >= 0:
            faces_by_facet[next(iter(facets))].add(face_id(key))
        else:
            side_rows.append(face_id(key))
    patches: list[organization_owner.MeshPatch] = [
        organization_owner.MeshPatch(
            "facet:"
            + source_identity_owner.source_entity_id(
                layers.source_id,
                layers.source_revision,
                "face",
                complex_.facet_ids[facet],
                occurrence_path=complex_.facet_occurrence_paths[facet],
            ),
            layer_topology.mesh_entity_scope(
                mesh, 2, np.asarray(sorted(face_rows), dtype=np.int64)
            ),
            connected=False,
        )
        for facet, face_rows in sorted(faces_by_facet.items())
        if face_rows
    ]
    if cap_keys:
        patches.append(
            organization_owner.MeshPatch(
                "layer-core-interface",
                layer_topology.mesh_entity_scope(
                    mesh,
                    2,
                    np.asarray(
                        [face_id(key) for key in sorted(cap_keys)], dtype=np.int64
                    ),
                ),
                connected=False,
            )
        )
    for name, rows in (("wall", wall_rows), ("layer-side", side_rows)):
        if rows:
            patches.append(
                organization_owner.MeshPatch(
                    name,
                    layer_topology.mesh_entity_scope(
                        mesh, 2, np.asarray(rows, dtype=np.int64)
                    ),
                    connected=False,
                )
            )
    interface_ids = np.asarray(
        [
            face_id(key)
            for key, incidents in combined_faces.items()
            if len(incidents) == 2 and incidents[0].region != incidents[1].region
        ],
        dtype=np.int64,
    )
    if interface_ids.size:
        labels += (
            organization_owner.MeshLabel(
                "interface", layer_topology.mesh_entity_scope(mesh, 2, interface_ids)
            ),
        )
    return zones, tuple(patches), labels, attributes


def _compose_refined_organization(
    mesh: CellMesh,
    regions: np.ndarray,
    source: sources_owner.NativeLayerCoreSource,
    partition: subdivision_owner.NestedLayerPartition,
    columns: subdivision_owner.RefinedColumnCertification,
    core_cell_ids: np.ndarray,
    /,
) -> tuple[
    tuple[organization_owner.MeshZone, ...],
    tuple[organization_owner.MeshPatch, ...],
    tuple[organization_owner.MeshLabel, ...],
    tuple[organization_owner.MeshAttribute, ...],
]:
    """Organize refined layer cells without relabeling execution facets as source."""
    cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    layer_ids = np.asarray(partition.prism_ids, dtype=np.int64)
    core_ids = np.asarray(core_cell_ids, dtype=np.int64)
    if regions.shape != cells.shape or not np.array_equal(
        cells, np.concatenate((layer_ids, core_ids))
    ):
        raise ValueError(
            "Refined layer/core organization requires aligned canonical cells."
        )
    order = np.argsort(regions, kind="stable")
    offsets = np.searchsorted(regions[order], np.arange(len(source.region_ids) + 1))
    zones = tuple(
        organization_owner.MeshZone(
            name,
            organization_owner.MeshZoneRole.REGION,
            layer_topology.mesh_entity_scope(mesh, 3, cells[order[start:stop]]),
        )
        for name, start, stop in zip(
            source.region_ids, offsets[:-1], offsets[1:], strict=True
        )
        if stop > start
    )
    source_ids = np.sort(
        np.asarray(source.layers.mesh.entity_set(3).entity_ids, dtype=np.int64)
    )
    parent_rows = np.searchsorted(source_ids, partition.parent_cell_ids)
    if np.any(parent_rows >= source_ids.size) or not np.array_equal(
        source_ids[parent_rows], partition.parent_cell_ids
    ):
        raise ValueError("Refined layer organization lost original parent cells.")
    intervals = np.asarray(source.layers.layer_index, dtype=np.int64)[parent_rows]
    parent_columns = np.asarray(source.layers.column_index, dtype=np.int64)[parent_rows]
    keys = tuple(
        (int(column), triangle)
        for column, triangle in zip(
            parent_columns, partition.child_reference_triangles, strict=True
        )
    )
    ranks = {key: index for index, key in enumerate(sorted(set(keys)))}
    column_ids = np.asarray([ranks[key] for key in keys], dtype=np.int64)
    scope = layer_topology.mesh_entity_scope(mesh, 3, cells)
    attributes = (
        organization_owner.MeshAttribute(
            "layer_index",
            organization_owner.MeshAttributeRole.MARKER,
            scope,
            np.concatenate((intervals, np.full(core_ids.size, -1, dtype=np.int64))),
        ),
        organization_owner.MeshAttribute(
            "layer_column",
            organization_owner.MeshAttributeRole.MARKER,
            scope,
            np.concatenate((column_ids, np.full(core_ids.size, -1, dtype=np.int64))),
        ),
    )
    labels = (
        organization_owner.MeshLabel(
            "boundary-layer", layer_topology.mesh_entity_scope(mesh, 3, layer_ids)
        ),
        organization_owner.MeshLabel(
            "core", layer_topology.mesh_entity_scope(mesh, 3, core_ids)
        ),
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    face_keys = dict(
        zip(canonical_owner._entity_vertex_keys(mesh, 2), face_ids.tolist(), strict=True)
    )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    wall_ids = {column.target_vertices[0] for column in columns.columns}
    core_set, layer_set = set(core_ids.tolist()), set(layer_ids.tolist())
    cap_faces, wall_faces, side_faces, material_faces = [], [], [], []
    for key, incident in layer_topology.faces(mesh, regions).items():
        global_vertices = tuple(sorted(int(vertex_ids[row]) for row in key))
        identifier = int(face_keys[global_vertices])
        owners = {int(cells[face.cell]) for face in incident}
        if len(incident) == 2 and owners & core_set and owners & layer_set:
            cap_faces.append(identifier)
        elif len(incident) == 1 and owners <= layer_set:
            (wall_faces if set(global_vertices) <= wall_ids else side_faces).append(
                identifier
            )
        if len(incident) == 2 and incident[0].region != incident[1].region:
            material_faces.append(identifier)
    patches = tuple(
        organization_owner.MeshPatch(
            name,
            layer_topology.mesh_entity_scope(
                mesh, 2, np.asarray(sorted(rows), dtype=np.int64)
            ),
            connected=False,
        )
        for name, rows in (
            ("layer-core-interface", cap_faces),
            ("wall", wall_faces),
            ("layer-side", side_faces),
        )
        if rows
    )
    if material_faces:
        labels += (
            organization_owner.MeshLabel(
                "interface",
                layer_topology.mesh_entity_scope(
                    mesh, 2, np.asarray(sorted(material_faces), dtype=np.int64)
                ),
            ),
        )
    return zones, patches, labels, attributes


def _overlap_geometry(
    source: sources_owner.NativeLayerCoreSource,
    partition: subdivision_owner.NestedLayerPartition,
    execution: LayerCoreExecution,
    core: volume_generation_owner.VolumeConstruction,
    core_vertex_map: np.ndarray,
    mesh: CellMesh,
    /,
) -> tuple[
    ExactPlcCellGeometrySource,
    tuple[tuple[int, tuple[tuple[str, int, Fraction], ...]], ...],
]:
    """Compose native source-point witnesses through exact original-root stencils."""
    from ..discretization._exact_plc_geometry import ExactPlcCellGeometrySource

    original_points = tuple(
        (Fraction(float(point[0])), Fraction(float(point[1])), Fraction(float(point[2])))
        for point in np.asarray(source.complex.vertices, dtype=np.float64)
    )
    original_ids = source.complex.vertex_entity_ids
    target_points = np.asarray(mesh.coordinates, dtype=np.float64)
    target_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    exact: list[tuple[Fraction, ...] | None] = [None] * len(target_ids)
    stencils: dict[int, tuple[tuple[str, int, Fraction], ...]] = {}
    for row, terms in enumerate(partition.stencils):
        exact[row] = partition.exact_coordinates[row]
        stencils[row] = tuple(
            ("layer", int(identifier), weight) for identifier, weight in terms
        )
    ancestry = np.asarray(core.input_vertices, dtype=np.int64)
    mapping = np.asarray(core_vertex_map, dtype=np.int64)
    if len(core.root_vertex_stencils) != len(ancestry) or len(
        core.root_exact_points
    ) != len(ancestry):
        raise ValueError("Core candidate lacks its original-root coordinate evidence.")
    boundary = set(
        int(value)
        for value in np.asarray(core.boundary_faces, dtype=np.int64).reshape(-1)
    )
    for row, execution_row in enumerate(ancestry.tolist()):
        target = int(mapping[row])
        terms = core.root_vertex_stencils[row]
        point = core.root_exact_points[row]
        if terms is None or point is None:
            if (
                execution_row >= 0
                or row in boundary
                or terms is not None
                or point is not None
            ):
                raise ValueError(
                    "A core source vertex lacks its exact original-root witness."
                )
            exact[target] = tuple(
                Fraction(float(value)) for value in target_points[target]
            )
            continue
        recomposed, recomputed = volume_generation_owner.compose_plc_root_point(
            original_points, (terms,), (0,), (Fraction(1),)
        )
        if recomposed != terms or recomputed != point:
            raise ValueError(
                "A core root witness does not reproduce its exact source point."
            )
        if execution_row >= 0 and terms != execution.vertex_stencils[execution_row]:
            raise ValueError(
                "A retained execution vertex changed its original-root stencil."
            )
        if exact[target] is not None and exact[target] != point:
            raise ValueError(
                "Execution PLC and accepted layer stencils disagree exactly."
            )
        exact[target] = point
        stencils.setdefault(
            target,
            tuple(
                sorted(("plc", int(original_ids[root]), weight) for root, weight in terms)
            ),
        )
    if any(point is None for point in exact):
        raise ValueError("Exact overlap geometry omitted a combined vertex.")
    exact_rows = tuple(point for point in exact if point is not None)
    rounded = np.asarray(
        [[float(value) for value in point] for point in exact_rows], dtype=np.float64
    )
    if not layer_topology.bitwise_equal_points(rounded, target_points):
        raise ValueError(
            "Native carriers do not correctly round their composed original-source points."
        )
    carrier = ExactPlcCellGeometrySource(
        source.source_id,
        source.source_revision,
        target_ids,
        exact_rows,
        vertex_source_entity_ids=(None,) * len(target_ids),
        vertex_occurrence_paths=((),) * len(target_ids),
    )
    rows = tuple(sorted((int(target_ids[row]), terms) for row, terms in stencils.items()))
    return carrier, rows


def _generate_conforming_layer_core(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    core_specification: VolumeMeshingSpec,
    schedule: volume_generation_owner.NativeVolumeSchedule,
    /,
    *,
    frame_id: str,
    mapped_source: sources_owner.NativeLayerCoreSource | None = None,
) -> layer_topology.LayerCoreConstruction:
    """Fill an exact source-bound, conformingly subdivided layer/core carrier."""
    from ._layer_core_execution import build_layer_core_execution

    if source.mapped_domain is not None or source.reference_source is not None:
        raise ValueError(
            "Conforming layer/core execution requires an unmapped source authority."
        )
    if core_specification.fill_strategy is not contracts_owner.VolumeFillStrategy.SIMPLEX:
        raise ValueError("Conforming layer/core execution requires a simplex core.")
    if source.layer_control is None or source.layer_policy is None:
        raise ValueError(
            "Conforming layer/core execution requires its original layer control and policy."
        )
    source.validate_restored()
    mapped_domain = None
    if mapped_source is not None:
        from ..geometry._mapped_reference_boundary import MappedReferenceBoundarySource
        from .providers._native_sources import _mapped_layer_core_facet_relation

        domain_owner = mapped_source.mapped_domain
        query = mapped_source.fidelity_source
        if (
            mapped_source.reference_source is not source
            or not isinstance(domain_owner, mapped_domain_owner.MappedReferenceDomain)
            or not isinstance(query, MappedReferenceBoundarySource)
            or not isinstance(
                query.authority,
                (sources_owner.NativePlcSource, sources_owner.NativeMappedHexSource),
            )
        ):
            raise ValueError(
                "Conforming mapped execution requires its original root authority."
            )
        _mapped_layer_core_facet_relation(mapped_source, query.authority)
        mapped_domain = domain_owner
        domain = domain_owner.reference_domain
    elif source.source_boundary_charts is not None and source.cap_refinement == "fixed":
        from ._layer_core_staged_domain import validate_brep_staged_layer_domain

        domain = validate_brep_staged_layer_domain(source)
    else:
        domain = layer_topology.validate_staged_layer_domain(source)
    divisions = (
        subdivision_owner.source_derived_nested_divisions(source)
        if source.cap_refinement == "conforming"
        else 1
    )
    partition = subdivision_owner.subdivide_nested_prism_columns(
        source.layers.mesh, divisions
    )
    column_certification = subdivision_owner.certify_nested_layer_columns(
        source, partition, specification
    )
    layer_partition = subdivision_owner.nested_prism_reference_partition(partition)
    execution = build_layer_core_execution(
        source, partition, source_authority=mapped_source
    )
    if source.cap_refinement == "fixed":
        original_polygons = np.asarray(
            source.complex.polygon_vertices, dtype=np.int64
        ).reshape(-1, 3)
        execution_polygons = np.asarray(
            execution.complex.polygon_vertices, dtype=np.int64
        ).reshape(-1, 3)
        execution_facets = np.asarray(execution.complex.polygon_facets, dtype=np.int64)
        recovered_caps: dict[int, tuple[int, ...]] = {}
        for polygon in execution.cap_polygon_ids:
            row = int(polygon)
            facet = int(execution_facets[row])
            candidates = execution.source_polygon_candidates[facet]
            if len(candidates) != 1 or not execution.protected_facets[facet]:
                raise ValueError("A fixed cap lost its original protected source facet.")
            roots = []
            for vertex in execution_polygons[row]:
                terms = execution.vertex_stencils[int(vertex)]
                if len(terms) != 1 or terms[0][1] != 1:
                    raise ValueError("A fixed cap contains a generated execution vertex.")
                roots.append(int(terms[0][0]))
            source_row = int(candidates[0])
            if source_row in recovered_caps:
                raise ValueError("A fixed cap source polygon was subdivided.")
            recovered_caps[source_row] = tuple(sorted(roots))
        expected_caps = {
            int(row): tuple(sorted(int(vertex) for vertex in original_polygons[int(row)]))
            for row in source.cap_polygon_ids
        }
        if recovered_caps != expected_caps:
            raise ValueError("Execution changed the original fixed cap triangulation.")
    execution_specification = _execution_core_request(
        source, core_specification, execution
    )
    carrier = volume_generation_owner.generate_plc_volume(
        execution.complex,
        execution_specification,
        schedule,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id=(
            ""
            if mapped_domain is None
            else (mapped_domain.source_geometry.source_map_id or "")
        ),
        frame_id=frame_id,
        protected_facets=execution.protected_facets,
        source_domain=mapped_domain,
        boundary_collapse_quality=source.noncap_refinement == "conforming",
        root_points=source.complex.vertices,
        root_vertex_stencils=execution.vertex_stencils,
    )
    ancestry = _require_selectively_conforming(
        execution.complex, carrier, execution.protected_facets
    )
    mapping = np.asarray(execution.vertex_layer_ids, dtype=np.int64).copy()
    if mapping.shape != (execution.complex.vertices.shape[0],):
        raise ValueError("Execution PLC vertex ancestry has incompatible extents.")
    if np.any(mapping < -1) or np.any(mapping >= partition.coordinates.shape[0]):
        raise ValueError("Execution PLC references an absent refined layer vertex.")
    free = np.flatnonzero(mapping < 0)
    mapping[free] = partition.coordinates.shape[0] + np.arange(free.size, dtype=np.int64)
    initial_points = np.concatenate(
        (
            partition.coordinates,
            np.asarray(execution.complex.vertices, dtype=np.float64)[free],
        )
    )
    mesh, core_vertex_map, regions, layer_ids, core_ids = _compose_layer_topology(
        source,
        carrier,
        mapping,
        initial_points,
        ancestry,
        source.source_revision,
        layer_partition=partition,
    )
    mesh = canonical_owner.canonicalize_cell_mesh(mesh)
    exact_carrier, source_rows = _overlap_geometry(
        source, partition, execution, carrier, core_vertex_map, mesh
    )
    periodic_topology = None
    if source.layers.mesh.periodic_topology is not None:
        periodic_topology = periodic_owner.bind_refined_core_periodic(
            source,
            partition,
            execution.vertex_stencils,
            ancestry,
            core_vertex_map,
            mesh,
            exact_carrier,
            execution.periodic_seams,
        )
        mesh = CellMesh(
            mesh.coordinates,
            mesh.blocks,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids={
                degree: np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
                for degree in range(1, mesh.topological_dimension)
            },
            numeric_version=mesh.numeric_version,
            periodic_topology=periodic_topology,
        )
    faces = layer_topology.faces(mesh, regions)
    core_set, layer_set = set(core_ids.tolist()), set(layer_ids.tolist())
    sorted_cells = np.sort(np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64))
    polygons = np.asarray(execution.complex.polygon_vertices, dtype=np.int64).reshape(
        -1, 3
    )
    for row in execution.cap_polygon_ids:
        key = tuple(sorted(int(value) for value in mapping[polygons[int(row)]]))
        incident = faces.get(key, ())
        owners = {int(sorted_cells[face.cell]) for face in incident}
        if (
            len(incident) != 2
            or len(owners & core_set) != 1
            or len(owners & layer_set) != 1
        ):
            raise ValueError(
                "A refined cap triangle is not shared by one layer and one core cell."
            )
    stencils = subdivision_owner.OverlapSourceStencils(
        source.binding_id, mesh.mesh_id, source_rows
    )
    geometry = cell_geometry_owner.CellGeometrySpec.power(mesh, exact_carrier)
    subdivision_owner.require_refined_column_certification(
        source, mesh, geometry, specification, column_certification
    )
    zones, patches, labels, attributes = _compose_refined_organization(
        mesh, regions, source, partition, column_certification, core_ids
    )
    requested, achieved = periodic_owner.core_periodic_constraint_evidence(
        source, specification
    )

    def construction(
        current_zones: tuple, current_patches: tuple
    ) -> layer_topology.LayerCoreConstruction:
        return layer_topology.LayerCoreConstruction(
            mesh,
            domain,
            regions,
            current_zones,
            current_patches,
            labels,
            attributes,
            (),
            geometry,
            carrier,
            core_vertex_map,
            core_ids,
            source.region_ids,
            source.source_id,
            source.source_revision,
            periodic_topology,
            requested,
            achieved,
            (),
            None,
            None,
            overlap_exact_geometry=exact_carrier,
            execution_carrier=execution,
            overlap_source_stencils=stencils,
            layer_core_column_certification=column_certification,
            layer_root_partition=layer_partition,
        )

    preliminary = construction(zones, patches)
    subdivision_owner.require_overlap_source_provenance(source, preliminary)
    zones, patches = controls_owner.compose_layer_controls(
        source, specification, preliminary
    )
    return construction(zones, patches)


@checked
def _generate_layer_core_unmapped(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    core_specification: VolumeMeshingSpec,
    schedule: volume_generation_owner.NativeVolumeSchedule,
    /,
    *,
    frame_id: str,
    source_domain: mapped_domain_owner.MappedReferenceDomain | None = None,
    mapped_source: sources_owner.NativeLayerCoreSource | None = None,
    polyhedral_schedule: polyhedral_generation_owner.NativePolyhedralSchedule
    | None = None,
) -> layer_topology.LayerCoreConstruction:
    """Fill the remaining fixed PLC and return the complete combined construction."""
    layers = source.layers
    complex_ = source.complex
    if source.conforming_execution:
        if (source_domain is None) != (mapped_source is None):
            raise ValueError(
                "Mapped conforming execution requires both source and root authority."
            )
        if mapped_source is not None and mapped_source.mapped_domain is not source_domain:
            raise ValueError(
                "Mapped conforming execution changed its original root domain."
            )
        return _generate_conforming_layer_core(
            source,
            specification,
            core_specification,
            schedule,
            frame_id=frame_id,
            mapped_source=mapped_source,
        )
    layer_count = sum(block.cell_count for block in layers.mesh.blocks)
    layer_regions = np.asarray(source.layer_regions, dtype=np.int64)
    if layer_regions.shape != (layer_count,):
        raise ValueError("Native layer regions must identify every accepted layer cell.")
    if any(
        controls_owner.realized_layer_control(source, control).control_id
        != layers.control_id
        for control in specification.layer_controls
    ):
        raise ValueError(
            "Layer controls must describe the accepted immutable layer mesh."
        )
    region_ids, core_region_map = regions_owner.prepare_layer_region_identity(
        complex_, layer_regions, source.region_ids, source.core_region_map
    )
    power_request: (
        tuple[
            sources_owner.NativePolyhedralSource,
            polyhedral_generation_owner.NativePolyhedralSchedule,
        ]
        | None
    ) = None
    if core_specification.fill_strategy is contracts_owner.VolumeFillStrategy.POLYHEDRAL:
        if source_domain is not None:
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Mapped polyhedral layer/core composition requires a certified mixed-cell root map.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        if polyhedral_schedule is None:
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.UNSUPPORTED_COMBINATION,
                "Polyhedral layer/core fill requires an explicit power schedule.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        power_source = source.core_power_source
        if power_source is None:
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "Polyhedral layer/core fill requires an authored core power source.",
                provider_code="native_layer_core_polyhedral",
                stage="route_execution",
            )
        power_request = (
            power_source,
            polyhedral_generation_owner.effective_polyhedral_schedule(
                core_specification, polyhedral_schedule
            ),
        )
    mapping, cap_ids, polygons, initial_points = layer_topology.prepare_identity(
        layers, complex_, source.vertex_layer_ids, source.cap_polygon_ids
    )
    if (
        power_request is None
        and source_domain is None
        and source.mapped_domain is None
        and complex_.geometry is None
        and complex_.boundary == "fixed"
        and schedule.radius_edge_bound is not None
    ):
        obstruction = _fixed_cap_edge_obstruction(
            complex_, cap_ids, schedule.radius_edge_bound
        )
        if obstruction is not None:
            cap_row, face_row, edge, lower_squared = obstruction
            facet = int(complex_.polygon_facets[face_row])
            edge_row = complex_.plc_edge_vertices.index(edge)
            finding = (
                f"Fixed cap polygon {cap_row}, adjacent polygon {face_row}, "
                f"source edge {complex_.plc_edge_ids[edge_row]!r} occurrence "
                f"{complex_.plc_edge_occurrence_paths[edge_row]!r}, source facet "
                f"{complex_.facet_ids[facet]!r} occurrence "
                f"{complex_.facet_occurrence_paths[facet]!r}: exact squared "
                f"radius-edge lower bound {lower_squared} exceeds "
                f"{Fraction(float(schedule.radius_edge_bound)) ** 2}."
            )
            raise contracts_owner.MeshingFailure(
                contracts_owner.MeshingFailureCategory.QUALITY_REJECTED,
                "The fixed cap and adjacent fixed PLC facet make the requested "
                "radius-edge quality mathematically infeasible.",
                provider_code="native_layer_core",
                stage="boundary_quality_preflight",
                entity_ids=(
                    int(complex_.plc_edge_entity_ids[edge_row]),
                    int(complex_.facet_entity_ids[facet]),
                ),
                evidence=contracts_owner.MeshingFailureEvidence(
                    requested_quantities={
                        "radius_edge_bound": schedule.radius_edge_bound,
                    },
                    checkpoint="boundary_quality_preflight",
                    logical_findings=(finding,),
                ),
            )
    if power_request is not None:
        unmapped_vertices = mapping < 0
        if np.any(unmapped_vertices):
            mapping = mapping.copy()
            mapping[unmapped_vertices] = layers.mesh.coordinates.shape[0] + np.arange(
                np.count_nonzero(unmapped_vertices), dtype=np.int64
            )
    periodic_ancestry = periodic_owner.prepare_core_periodic(
        source, mapping, polygons, initial_points
    )
    periodic_requested, periodic_achieved = (
        periodic_owner.core_periodic_constraint_evidence(source, specification)
    )
    coordinate_map_domain = (
        source_domain if source_domain is not None else source.mapped_domain
    )
    mapped_geometry = (
        None if coordinate_map_domain is None else coordinate_map_domain.source_geometry
    )
    source_map_id = (
        ""
        if mapped_geometry is None or mapped_geometry.source_map_id is None
        else mapped_geometry.source_map_id
    )
    polyhedral_core = None
    if power_request is not None:
        power_source, power_schedule = power_request
        polyhedral_core = polyhedral_generation_owner.generate_polyhedral_volume(
            power_source,
            core_specification,
            power_schedule,
            frame_id=frame_id,
            volume_schedule=schedule,
        )
        carrier_core = polyhedral_core.carrier
        construction_core: (
            volume_generation_owner.VolumeConstruction
            | polyhedral_generation_owner.PolyhedralConstruction
        ) = polyhedral_core
    else:
        carrier_core = volume_generation_owner.generate_plc_volume(
            complex_,
            core_specification,
            schedule,
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_map_id=source_map_id,
            frame_id=frame_id,
            source_domain=source_domain,
        )
        construction_core = carrier_core
    input_vertices = _require_fixed(complex_, carrier_core)
    mesh, core_vertex_map, combined_regions, layer_ids, core_cell_ids = (
        _compose_layer_topology(
            source,
            construction_core,
            mapping,
            initial_points,
            input_vertices,
            source.source_revision,
        )
    )
    if polyhedral_core is not None:
        periodic_ancestry = _polyhedral_periodic_ancestry(
            source,
            polyhedral_core,
            mapping,
            initial_points,
            input_vertices,
            core_vertex_map,
            periodic_ancestry,
        )
    mesh = canonical_owner.canonicalize_cell_mesh(mesh)
    periodic_topology = periodic_owner.bind_core_periodic(source, mesh, periodic_ancestry)
    if periodic_topology is not None:
        mesh = CellMesh(
            mesh.coordinates,
            mesh.blocks,
            vertex_global_ids=mesh.vertex_global_ids,
            entity_global_ids={
                degree: np.asarray(mesh.entity_set(degree).entity_ids, dtype=np.int64)
                for degree in range(1, mesh.topological_dimension)
            },
            numeric_version=mesh.numeric_version,
            periodic_topology=periodic_topology,
        )
    domain, layer_faces = layer_topology.combined_domain(
        layers,
        complex_,
        layer_regions,
        mapping,
        cap_ids,
        polygons,
        np.asarray(mesh.coordinates, dtype=np.float64)[: initial_points.shape[0]],
        source.source_id,
        region_ids,
        core_region_map,
        source.region_occurrence_paths,
    )
    combined_faces = layer_topology.faces(mesh, combined_regions)
    layer_face_keys = set(layer_faces)
    for index in cap_ids:
        key = tuple(sorted(mapping[polygons[index]].tolist()))
        incidents = combined_faces.get(key, [])
        if key not in layer_face_keys or len(incidents) != 2:
            raise ValueError(
                "The fixed layer cap must be shared by one layer and one core cell."
            )
        incident_ids = {
            int(np.sort(np.asarray(mesh.entity_set(3).entity_ids))[face.cell])
            for face in incidents
        }
        if not incident_ids.intersection(
            set(layer_ids.tolist())
        ) or not incident_ids.intersection(set(core_cell_ids.tolist())):
            raise ValueError(
                "A fixed cap facet is not shared across the layer/core transition."
            )
    zones, patches, labels, attributes = _compose_organization(
        mesh,
        combined_regions,
        layer_ids,
        core_cell_ids,
        source,
        complex_,
        mapping,
        cap_ids,
        polygons,
        layer_faces,
        combined_faces,
        region_ids,
        polyhedral_core=polyhedral_core,
        core_vertex_map=core_vertex_map,
    )
    associations = (
        *association_composition.layer_wall_associations(source, mesh),
        *association_composition.layer_source_associations(
            layers, mesh, layer_regions, region_ids
        ),
        *association_composition.core_source_associations(
            source,
            polyhedral_core if polyhedral_core is not None else carrier_core,
            mesh,
            core_vertex_map,
            combined_regions,
            region_ids,
            core_cell_ids,
        ),
    )
    preliminary = layer_topology.LayerCoreConstruction(
        mesh,
        domain,
        combined_regions,
        zones,
        patches,
        labels,
        attributes,
        associations,
        cell_geometry_owner.CellGeometrySpec.affine(mesh),
        carrier_core,
        core_vertex_map,
        core_cell_ids,
        region_ids,
        source.source_id,
        source.source_revision,
        periodic_topology,
        periodic_requested,
        periodic_achieved,
        (),
        None,
        None,
        polyhedral_core=polyhedral_core,
    )
    zones, patches = controls_owner.compose_layer_controls(
        source, specification, preliminary
    )
    return layer_topology.LayerCoreConstruction(
        mesh,
        domain,
        combined_regions,
        zones,
        patches,
        labels,
        attributes,
        associations,
        cell_geometry_owner.CellGeometrySpec.affine(mesh),
        carrier_core,
        core_vertex_map,
        core_cell_ids,
        region_ids,
        source.source_id,
        source.source_revision,
        periodic_topology,
        periodic_requested,
        periodic_achieved,
        (),
        None,
        None,
        polyhedral_core=polyhedral_core,
    )


def _mapped_reference_specification(
    specification: VolumeMeshingSpec, /
) -> VolumeMeshingSpec:
    """Reconstruct the reference fill request from the original physical request."""
    return VolumeMeshingSpec(
        specification.target,
        specification.boundary_scope,
        specification.fill_strategy,
        size_controls=specification.size_controls,
        protected_features=(),
        region_controls=(),
        patch_controls=(),
        region_seeds=specification.region_seeds,
        hole_seeds=specification.hole_seeds,
        layer_controls=(),
        periodic_constraints=specification.periodic_constraints,
        size_combination=specification.size_combination,
        size_compliance=specification.size_compliance,
        maximum_deviation=specification.maximum_deviation,
        deterministic=specification.deterministic,
    )


@checked
def generate_layer_core(
    source: sources_owner.NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    core_specification: VolumeMeshingSpec,
    schedule: volume_generation_owner.NativeVolumeSchedule,
    /,
    *,
    frame_id: str,
    polyhedral_schedule: polyhedral_generation_owner.NativePolyhedralSchedule
    | None = None,
) -> layer_topology.LayerCoreConstruction:
    """Fill a fixed PLC directly or compose its reference fill through exact source roots."""
    reference = source.reference_source
    if reference is None:
        if source.mapped_domain is not None:
            raise ValueError(
                "A mapped layer/core source must retain its exact reference source."
            )
        return _generate_layer_core_unmapped(
            source,
            specification,
            core_specification,
            schedule,
            frame_id=frame_id,
            polyhedral_schedule=polyhedral_schedule,
        )
    domain = source.mapped_domain
    if not isinstance(domain, mapped_domain_owner.MappedReferenceDomain):
        raise TypeError("Mapped layer/core execution requires MappedReferenceDomain.")
    if (
        reference.reference_source is not None
        or reference.mapped_domain is not None
        or (reference.source_id, reference.source_revision)
        != (source.source_id, source.source_revision)
        or (domain.source_id, domain.source_revision)
        != (source.source_id, source.source_revision)
        or domain.region_ids != source.region_ids
        or source.complex.region_ids != reference.complex.region_ids
        or source.complex.vertex_ids != reference.complex.vertex_ids
        or source.complex.vertex_occurrence_paths
        != reference.complex.vertex_occurrence_paths
        or source.complex.segment_ids != reference.complex.segment_ids
        or source.complex.segment_occurrence_paths
        != reference.complex.segment_occurrence_paths
        or source.complex.facet_ids != reference.complex.facet_ids
        or source.complex.facet_occurrence_paths
        != reference.complex.facet_occurrence_paths
        or source.complex.region_ids != reference.complex.region_ids
        or source.complex.region_entity_ids != reference.complex.region_entity_ids
        or source.complex.region_occurrence_paths
        != reference.complex.region_occurrence_paths
        or source.region_ids != reference.region_ids
        or source.region_entity_ids.shape != reference.region_entity_ids.shape
        or not np.array_equal(source.region_entity_ids, reference.region_entity_ids)
        or source.region_occurrence_paths != reference.region_occurrence_paths
    ):
        raise ValueError(
            "Mapped layer/core source and reference PLC must retain the same authored identities, occurrences, and regions."
        )
    reference_specification = _mapped_reference_specification(specification)
    physical_scale = mapped_core_owner.mapped_source_lipschitz_bound(domain)
    reference_core_specification = core_request(
        core_specification,
        specification.layer_controls,
        physical_scale=physical_scale,
    )
    reference_construction = _generate_layer_core_unmapped(
        reference,
        reference_specification,
        reference_core_specification,
        schedule,
        source_domain=domain,
        mapped_source=source,
        frame_id=frame_id,
        polyhedral_schedule=polyhedral_schedule,
    )
    mapped = mapped_core_owner.compose_mapped_layer_core(source, reference_construction)
    zones, patches = controls_owner.compose_layer_controls(source, specification, mapped)
    return layer_topology.LayerCoreConstruction(
        mapped.mesh,
        mapped.domain,
        mapped.cell_regions,
        zones,
        patches,
        mapped.labels,
        mapped.attributes,
        mapped.associations,
        mapped.geometry,
        mapped.core,
        mapped.core_vertex_map,
        mapped.core_cell_ids,
        mapped.region_ids,
        mapped.source_id,
        mapped.source_revision,
        mapped.periodic_topology,
        mapped.periodic_requested,
        mapped.periodic_achieved,
        mapped.reference_maps,
        mapped.source_mesh,
        mapped.source_geometry,
        reference_mesh=mapped.reference_mesh,
        reference_geometry=mapped.reference_geometry,
        polyhedral_core=reference_construction.polyhedral_core,
        overlap_exact_geometry=mapped.overlap_exact_geometry,
        overlap_source_stencils=mapped.overlap_source_stencils,
        layer_core_column_certification=mapped.layer_core_column_certification,
        layer_root_partition=mapped.layer_root_partition,
        execution_carrier=mapped.execution_carrier,
    )


__all__ = ["core_request", "generate_layer_core"]
