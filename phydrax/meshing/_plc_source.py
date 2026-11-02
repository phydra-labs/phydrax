#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Oriented piecewise-linear source contracts for volume construction."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, final, Literal

import equinox as eqx
import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import normalized_identifier
from ..discretization import CellGeometrySpec
from ..geometry._global_embedding import GlobalEmbeddingCertificate
from ..geometry._source_fidelity_certificate import SourceFidelityCertificate
from ..geometry._surface_source_support import SurfaceSourceCharts
from ..typing import (
    checked,
    ConvertibleToArray,
    Dim,
    HostFloat64,
    HostInt64,
    parse,
    Scope,
)

if TYPE_CHECKING:
    from ..geometry.brep._model import BRepGeometry


def _occurrence_path_bank(
    values: tuple[tuple[str, ...], ...], count: int, name: str, /
) -> tuple[tuple[str, ...], ...]:
    if len(values) != count:
        raise ValueError(f"{name} must contain one authored path per source entity.")
    paths = tuple(tuple(value) for value in values)
    if any(
        any(
            not isinstance(component, str)
            or not component
            or component != component.strip()
            for component in path
        )
        for path in paths
    ):
        raise ValueError(f"{name} components must be non-empty normalized identifiers.")
    return paths


def _identifier_bank(
    values: tuple[str, ...],
    occurrence_paths: tuple[tuple[str, ...], ...],
    count: int,
    name: str,
    /,
) -> tuple[str, ...]:
    identifiers = tuple(normalized_identifier(value, name) for value in values)
    if len(identifiers) != count:
        raise ValueError(f"{name} must contain exactly {count} authored identifiers.")
    if len(set(zip(identifiers, occurrence_paths, strict=True))) != count:
        raise ValueError(
            f"{name} and occurrence paths must uniquely identify each source entity."
        )
    return identifiers


def _numeric_entity_ids(
    kind: str,
    values: tuple[str, ...],
    occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> tuple[int, ...]:
    identifiers = tuple(
        int(
            canonical_fingerprint({"kind": kind, "value": (value, path)})[:16],
            16,
        )
        & ((1 << 63) - 1)
        for value, path in zip(values, occurrence_paths, strict=True)
    )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(
            f"Authored {kind} identity/occurrence pairs collide in the int64 scope projection."
        )
    return identifiers


def _plc_edge_records(
    offsets: np.ndarray,
    polygon_vertices: np.ndarray,
    polygon_facets: np.ndarray,
    segments: np.ndarray,
    vertex_ids: tuple[str, ...],
    vertex_paths: tuple[tuple[str, ...], ...],
    facet_ids: tuple[str, ...],
    facet_paths: tuple[tuple[str, ...], ...],
    segment_ids: tuple[str, ...],
    segment_paths: tuple[tuple[str, ...], ...],
    verified_segments: frozenset[int],
    /,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    tuple[int, ...],
]:
    boundary: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for polygon, facet in enumerate(polygon_facets.tolist()):
        start, stop = offsets[polygon : polygon + 2]
        loop = polygon_vertices[start:stop]
        for offset, first in enumerate(loop.tolist()):
            second = int(loop[(offset + 1) % loop.size])
            first_vertex = int(first)
            key = (min(first_vertex, second), max(first_vertex, second))
            boundary.setdefault(key, []).append((facet, int(first), second))

    records: list[tuple[tuple[int, int], tuple[int, ...], str, tuple[str, ...]]] = []
    explicit_keys: set[tuple[int, int]] = set()
    for row, segment in enumerate(np.asarray(segments, dtype=np.int64).tolist()):
        first, second = int(segment[0]), int(segment[1])
        key = (min(first, second), max(first, second))
        if key in explicit_keys:
            raise ValueError("Authored PLC segments cannot duplicate one geometric edge.")
        explicit_keys.add(key)
        facets = tuple(sorted({entry[0] for entry in boundary.get(key, ())}))
        facet_paths_on_edge = {facet_paths[facet] for facet in facets}
        if (
            row not in verified_segments
            and facet_paths_on_edge
            and (
                len(facet_paths_on_edge) != 1
                or segment_paths[row] not in facet_paths_on_edge
            )
        ):
            raise ValueError(
                "An explicit PLC segment must preserve one unique incident facet occurrence path."
            )
        records.append(((first, second), facets, segment_ids[row], segment_paths[row]))

    boundary_keys = set(boundary) - explicit_keys
    ordered_boundary_keys = sorted(
        boundary_keys,
        key=lambda key: (
            tuple(
                sorted(
                    (
                        vertex_ids[vertex],
                        vertex_paths[vertex],
                    )
                    for vertex in key
                )
            ),
            tuple(
                sorted(
                    (
                        facet_ids[facet],
                        facet_paths[facet],
                    )
                    for facet in {entry[0] for entry in boundary[key]}
                )
            ),
        ),
    )
    for key in ordered_boundary_keys:
        facets = tuple(
            sorted(
                {entry[0] for entry in boundary[key]},
                key=lambda facet: (facet_ids[facet], facet_paths[facet]),
            )
        )
        paths = {facet_paths[facet] for facet in facets}
        if len(paths) != 1:
            raise ValueError(
                "A PLC edge joining distinct source occurrences requires an authored segment identity and path."
            )
        path = next(iter(paths))
        directed_by_facet = {
            facet: tuple(
                (first, second)
                for entry_facet, first, second in boundary[key]
                if entry_facet == facet
            )
            for facet in facets
        }
        if len(facets) == 1:
            directed = directed_by_facet[facets[0]]
            if len(directed) == 1:
                first, second = directed[0]
            elif len(directed) == 2 and directed[0] == tuple(reversed(directed[1])):
                first, second = sorted(
                    key,
                    key=lambda vertex: (vertex_ids[vertex], vertex_paths[vertex]),
                )
            else:
                raise ValueError(
                    "A PLC facet must orient each boundary edge exactly once."
                )
        else:
            if any(len(directed_by_facet[facet]) != 1 for facet in facets):
                raise ValueError(
                    "Each incident PLC facet must orient a shared boundary edge once."
                )
            first, second = directed_by_facet[facets[0]][0]
        identity = canonical_fingerprint(
            {
                "kind": "plc-facet-edge",
                "vertices": tuple(
                    sorted((vertex_ids[vertex], vertex_paths[vertex]) for vertex in key)
                ),
                "facets": tuple(
                    sorted((facet_ids[facet], facet_paths[facet]) for facet in facets)
                ),
                "occurrence_path": path,
            }
        )
        records.append(((first, second), facets, identity, path))

    vertices = tuple(record[0] for record in records)
    facets = tuple(record[1] for record in records)
    identities = tuple(record[2] for record in records)
    paths = tuple(record[3] for record in records)
    entity_ids = _numeric_entity_ids("plc-edge", identities, paths)
    return vertices, facets, identities, paths, entity_ids


def _require_unambiguous_cad_copies(
    geometry: BRepGeometry, source_id: str, source_revision: str, /
) -> None:
    geometry.require_current()
    occurrences = {
        value.path: value for value in geometry.assembly_graph.iter_occurrences()
    }
    placed: dict[tuple, dict[int, set[tuple[str, ...]]]] = {}
    for use in geometry.qualified_entity_incidence(source_id, source_revision):
        if use.container.kind != "solid" or use.member.kind != "face":
            continue
        occurrence = occurrences.get(use.container.occurrence_path)
        if occurrence is None:
            raise ValueError("CAD face incidence lacks its assembly occurrence.")
        key = (use.member.index, occurrence.placement.matrix)
        placed.setdefault(key, {}).setdefault(use.container.index, set()).add(
            use.container.occurrence_path
        )
    if any(
        len(solids) > 1 and any(len(paths) > 1 for paths in solids.values())
        for solids in placed.values()
    ):
        raise ValueError(
            "Repeated coincident solid uses lack exact shared-copy ancestry."
        )


def _certify_cad_surface(
    surface: SurfaceSourceCharts,
    fidelity_tolerance: float,
    maximum_normal_deviation: float | None,
    junction_vertices: np.ndarray | None,
    /,
) -> tuple[GlobalEmbeddingCertificate, SourceFidelityCertificate]:
    """Compute the source's mathematical certificates without retaining a report."""
    from ._certification import MeshCertificationReport
    from ._certification_inputs import MeshCertificationInputs
    from ._certification_schedule import MeshCertificationSchedule
    from ._native_construction import NativeMeshOrganization

    domain = surface.domain
    faces = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in surface.mesh.blocks]
    )
    cells = np.asarray(surface.cell_global_ids, dtype=np.int64)
    topology = domain.multiregion_surface(
        surface.mesh.coordinates,
        faces,
        surface.root_patches,
        vertex_global_ids=surface.mesh.vertex_global_ids,
        cell_global_ids=cells,
    )
    inputs = MeshCertificationInputs(
        surface.mesh,
        surface.geometry,
        MeshCertificationSchedule("surface"),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        source=surface.boundary_source,
        fidelity_tolerance=fidelity_tolerance,
        maximum_normal_deviation=maximum_normal_deviation,
        junction_vertices=junction_vertices,
        protected_features=(),
        fidelity_witness=surface.chart_witness,
        feature_source=domain,
    )
    report = MeshCertificationReport(
        inputs,
        organization=NativeMeshOrganization(
            associations=surface.associations,
            multiregion_surface=topology,
        ),
    )
    report.require_passed()
    if report.fidelity is None:
        raise ValueError("CAD surface certification must retain source fidelity.")
    return report.embedding, report.fidelity


def _cad_segment_incidence(
    surface: SurfaceSourceCharts,
    fidelity_tolerance: float,
    maximum_normal_deviation: float | None,
    embedding: GlobalEmbeddingCertificate,
    fidelity: SourceFidelityCertificate,
    vertex_rows: np.ndarray,
    segment_edge_ids: np.ndarray,
    polygon_cell_ids: np.ndarray,
    points: np.ndarray,
    loops: tuple[np.ndarray, ...],
    polygon_facets: np.ndarray,
    segments: np.ndarray,
    vertex_ids: tuple[str, ...],
    vertex_paths: tuple[tuple[str, ...], ...],
    facet_ids: tuple[str, ...],
    facet_paths: tuple[tuple[str, ...], ...],
    facet_regions: np.ndarray,
    region_ids: tuple[str, ...],
    region_paths: tuple[tuple[str, ...], ...],
    segment_ids: tuple[str, ...],
    segment_paths: tuple[tuple[str, ...], ...],
    /,
) -> frozenset[int]:
    """Replay exact CAD surface and source incidence before joining occurrences."""
    from ..geometry._mesh_certificates import _same_tree_content
    from ..geometry._meshing_domain import PatchCurveUse
    from ..geometry._surface_source_support import validate_cad_curve_ancestry
    from ._association import _ordered_vertices
    from .providers._native_cad_volume import (
        _association_for,
        _association_row_map,
        _BRepRowResolver,
        _exact_surface_points,
    )

    if not isinstance(surface, SurfaceSourceCharts):
        raise TypeError("CAD PLC incidence requires canonical SurfaceSourceCharts.")
    surface.require_current()
    domain = surface.domain
    domain.require_current(domain.source_id, domain.source_revision)
    if domain.brep_authority is None:
        raise ValueError("CAD PLC incidence requires B-Rep authority.")
    expected_regions = tuple(
        domain.entity_id(3, row) for row in range(len(domain.regions))
    )
    if region_ids != expected_regions or region_paths != domain.region_source_occurrences:
        raise ValueError("CAD PLC regions differ from authoritative solid occurrences.")
    expected_sides = np.full((len(domain.patches), 2), -1, dtype=np.int64)
    for region_row, region in enumerate(domain.regions):
        for patch, sign in region.boundary:
            side = 1 if sign > 0 else 0
            if expected_sides[patch, side] >= 0:
                raise ValueError("CAD source patch has ambiguous region-side incidence.")
            expected_sides[patch, side] = region_row
    if not np.array_equal(facet_regions, expected_sides):
        raise ValueError(
            "CAD PLC facet regions differ from authoritative oriented incidence."
        )
    if not isinstance(embedding, GlobalEmbeddingCertificate) or not isinstance(
        fidelity, SourceFidelityCertificate
    ):
        raise TypeError(
            "CAD PLC incidence requires retained mathematical surface proofs."
        )
    embedding.require_current(surface.mesh, surface.geometry)
    GlobalEmbeddingCertificate.from_stored_fields(embedding)
    fidelity.require_inputs(
        surface.mesh,
        surface.geometry,
        surface.boundary_source,
        tolerance=fidelity_tolerance,
        maximum_normal_deviation=maximum_normal_deviation,
        witness=surface.chart_witness,
    )
    fidelity.revalidate(surface.mesh, surface.geometry, surface.boundary_source)
    if embedding.status != "certified" or fidelity.status != "certified":
        raise ValueError(
            "CAD PLC surface embedding and source fidelity must be certified."
        )
    fresh_embedding, fresh_fidelity = _certify_cad_surface(
        surface,
        fidelity_tolerance,
        maximum_normal_deviation,
        np.asarray(embedding.binding.junction_vertices, dtype=np.int64),
    )
    if not _same_tree_content(fresh_embedding, embedding) or not _same_tree_content(
        fresh_fidelity, fidelity
    ):
        raise ValueError("CAD PLC mathematical proofs differ from their source replay.")
    curve_ancestry = validate_cad_curve_ancestry(surface, fidelity_tolerance)
    if set(int(value) for value in segment_edge_ids) != set(curve_ancestry):
        raise ValueError(
            "CAD PLC segments must cover every verified source curve edge exactly."
        )
    model = domain.brep_authority
    _require_unambiguous_cad_copies(
        model.geometry, model.source_id, model.source_revision
    )
    mesh = surface.mesh
    if (
        len(vertex_rows) != mesh.coordinates.shape[0]
        or len(segment_edge_ids) != len(segments)
        or len(polygon_cell_ids) != len(loops)
        or len(set(segment_edge_ids)) != len(segment_edge_ids)
        or len(set(polygon_cell_ids)) != len(polygon_cell_ids)
        or set(int(value) for value in polygon_cell_ids) != set(surface.cell_global_ids)
        or any(row < 0 or row >= len(points) for row in vertex_rows)
    ):
        raise ValueError("CAD PLC surface incidence maps are incomplete or invalid.")
    _, exact_points = _exact_surface_points(surface)
    if any(
        not np.array_equal(points[target], np.asarray([float(x) for x in point]))
        for target, point in zip(vertex_rows, exact_points, strict=True)
    ):
        raise ValueError("CAD PLC vertices differ from certified exact surface points.")
    vertex_assoc = _association_for(surface, 0, required=True)
    edge_assoc = _association_for(surface, 1, required=True)
    face_assoc = _association_for(surface, 2, required=True)
    assert vertex_assoc is not None and edge_assoc is not None and face_assoc is not None
    if any(
        (assoc.source_id, assoc.source_revision)
        != (domain.source_id, domain.source_revision)
        for assoc in (vertex_assoc, edge_assoc, face_assoc)
    ):
        raise ValueError(
            "CAD PLC associations must bind the authoritative source revision."
        )
    resolve = _BRepRowResolver(domain)
    vertex_assoc_rows = _association_row_map(vertex_assoc)
    for row, global_id in enumerate(np.asarray(mesh.vertex_global_ids, dtype=np.int64)):
        assoc_row = vertex_assoc_rows[int(global_id)]
        dim = int(np.asarray(vertex_assoc.source_dimensions)[assoc_row])
        index = int(np.asarray(vertex_assoc.source_indices)[assoc_row])
        path = vertex_assoc.source_occurrence_paths[assoc_row]
        _, source_key, brep_key = resolve(dim, index, path)
        if vertex_assoc.source_entity_ids[assoc_row] != brep_key:
            raise ValueError("CAD PLC vertex association changed its source occurrence.")
        parameters = np.asarray(vertex_assoc.parameters)[assoc_row]
        parameter_key = (
            ()
            if dim == 0
            else tuple(
                float(value).hex() if np.isfinite(value) else None for value in parameters
            )
        )
        identity = canonical_fingerprint(
            {
                "kind": "cad-plc-source-vertex",
                "source_entity": source_key,
                "source_dimension": dim,
                "source_parameters": parameter_key,
            }
        )
        target = vertex_rows[row]
        if (vertex_ids[target], vertex_paths[target]) != (identity, path):
            raise ValueError(
                "CAD PLC vertex identity differs from its surface association."
            )
    face_assoc_rows = _association_row_map(face_assoc)
    surface_cells = {
        int(identifier): tuple(int(value) for value in vertices)
        for block in mesh.blocks
        for identifier, vertices in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        )
    }
    for polygon, identifier in enumerate(polygon_cell_ids):
        try:
            assoc_row = face_assoc_rows[identifier]
            cell = surface_cells[identifier]
        except KeyError as exc:
            raise ValueError("CAD PLC polygon lacks its certified surface cell.") from exc
        dim = int(np.asarray(face_assoc.source_dimensions)[assoc_row])
        index = int(np.asarray(face_assoc.source_indices)[assoc_row])
        path = face_assoc.source_occurrence_paths[assoc_row]
        patch, key, brep_key = resolve(dim, index, path)
        orientation = int(np.asarray(face_assoc.orientations)[assoc_row])
        mapped = tuple(vertex_rows[value] for value in cell)
        if orientation == -1:
            mapped = (mapped[0], mapped[2], mapped[1])
        if (
            dim != 2
            or orientation not in (-1, 1)
            or face_assoc.source_entity_ids[assoc_row] != brep_key
            or int(polygon_facets[polygon]) != patch
            or (facet_ids[patch], facet_paths[patch]) != (key, path)
            or tuple(int(value) for value in loops[polygon]) != mapped
        ):
            raise ValueError("CAD PLC polygon differs from certified face incidence.")
    edge_vertices = _ordered_vertices(mesh, 1)
    if edge_vertices is None:
        raise ValueError("CAD surface must retain exact edge incidence.")
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_lookup = {int(identifier): row for row, identifier in enumerate(edge_ids)}
    edge_assoc_rows = _association_row_map(edge_assoc)
    boundary: dict[tuple[int, int], set[int]] = {}
    for loop, facet in zip(loops, polygon_facets, strict=True):
        for index, first in enumerate(loop):
            second = int(loop[(index + 1) % len(loop)])
            boundary.setdefault(
                (min(int(first), second), max(int(first), second)), set()
            ).add(int(facet))
    for row, identifier in enumerate(segment_edge_ids):
        try:
            assoc_row = edge_assoc_rows[identifier]
            edge = edge_vertices[edge_lookup[identifier]]
        except KeyError as exc:
            raise ValueError("CAD PLC segment lacks its certified surface edge.") from exc
        dim = int(np.asarray(edge_assoc.source_dimensions)[assoc_row])
        index = int(np.asarray(edge_assoc.source_indices)[assoc_row])
        path = edge_assoc.source_occurrence_paths[assoc_row]
        curve, key, brep_key = resolve(dim, index, path)
        orientation = int(np.asarray(edge_assoc.orientations)[assoc_row])
        mapped = tuple(vertex_rows[int(vertex)] for vertex in edge)
        if orientation == -1:
            mapped = mapped[::-1]
        identity = canonical_fingerprint(
            {
                "kind": "cad-plc-source-segment",
                "source_curve": key,
                "occurrence_path": path,
                "oriented_source_vertices": (
                    vertex_ids[mapped[0]],
                    vertex_ids[mapped[1]],
                ),
            }
        )
        if (
            dim != 1
            or orientation not in (-1, 1)
            or edge_assoc.source_entity_ids[assoc_row] != brep_key
            or tuple(int(value) for value in segments[row]) != mapped
            or (segment_ids[row], segment_paths[row]) != (identity, path)
        ):
            raise ValueError("CAD PLC segment differs from certified curve incidence.")
        for patch in boundary.get(tuple(sorted(mapped)), ()):
            if not any(
                isinstance(use, PatchCurveUse) and use.curve == curve
                for loop in domain.patches[patch].loops
                for use in loop
            ):
                raise ValueError(
                    "CAD segment is not an authored use of each incident patch."
                )
    return frozenset(range(len(segments)))


type PlcBoundaryPolicy = Literal["fixed", "conforming"]


class PlcVertexDim(Dim, minimum=4):
    """Vertices of one three-dimensional piecewise-linear complex."""


class PlcPolygonDim(Dim, minimum=1):
    """Oriented polygons of one piecewise-linear complex."""


class PlcPolygonOffsetDim(Dim, minimum=2):
    """Loop offsets of the polygons of one complex (polygons plus one)."""


class PlcLoopDim(Dim, minimum=3):
    """Concatenated polygon loop vertices of one complex."""


class PlcFacetDim(Dim, minimum=1):
    """Facets (region-incidence groups of polygons) of one complex."""


class PlcSegmentDim(Dim):
    """Explicit internal curve segments of one complex."""


def _integer_array(value: ConvertibleToArray, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices.")
    if array.size and (
        np.any(array < np.iinfo(np.int64).min) or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} indices must be representable as signed int64.")
    return np.asarray(array, dtype=np.int64)


def _declared_facet_planes(
    facet_planes: ConvertibleToArray | None,
    facet_plane_tolerances: ConvertibleToArray | None,
    scope: Scope,
    /,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Validate declared facet plane rows and their tolerances (default 0)."""
    if facet_planes is None:
        if facet_plane_tolerances is not None:
            raise ValueError("facet_plane_tolerances require facet_planes.")
        return None, None
    planes = parse(
        np.array(facet_planes, dtype=np.float64, copy=True, order="C"),
        HostFloat64[PlcFacetDim, Literal[3], Literal[3]],
        "facet_planes",
        scope=scope,
    )
    tolerances = parse(
        np.zeros((planes.shape[0],), dtype=np.float64)
        if facet_plane_tolerances is None
        else np.array(facet_plane_tolerances, dtype=np.float64, copy=True, order="C"),
        HostFloat64[PlcFacetDim],
        "facet_plane_tolerances",
        scope=scope,
    )
    declared = planes[~np.all(np.isnan(planes), axis=(1, 2))]
    if not np.all(np.isfinite(declared)):
        raise ValueError(
            "A declared facet plane row must be finite; an undeclared row is all NaN."
        )
    for _origin, first, second in declared.tolist():
        u = [Fraction(value) for value in first]
        v = [Fraction(value) for value in second]
        if (
            u[1] * v[2] == u[2] * v[1]
            and u[2] * v[0] == u[0] * v[2]
            and u[0] * v[1] == u[1] * v[0]
        ):
            raise ValueError("Declared facet plane axes must be exactly independent.")
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances < 0.0):
        raise ValueError("facet_plane_tolerances must be finite and nonnegative.")
    return planes, tolerances


@final
class PiecewiseLinearComplex(StrictModule, NonTrainableState):
    """Oriented three-dimensional piecewise-linear source for solid filling.

    ``polygon_facets`` groups polygon loops into source facets. Each row of
    ``facet_regions`` is ``(positive, negative)``: the first region lies on
    the right-hand-normal side of the oriented facet, and the second on the
    opposite side. ``-1`` denotes void; equal region indices declare an
    internal sheet. Every declared region must occur in the incidence table.

    ``boundary="fixed"`` preserves every source vertex and facet exactly;
    ``boundary="conforming"`` permits constrained subdivision while retaining
    its source ancestry. ``geometry`` may retain a source coordinate map whose
    rows correspond to ``vertices``; its rounded execution coordinates do not
    strengthen the represented source's geometric claims.

    Each source vertex, facet and explicit segment carries an authored stable
    identity. Polygon-loop edges derive identity from their source endpoint and
    facet keys, including seams between polygons of the same facet. Recovery-only
    triangulation diagonals remain facet-supported carrier edges, not fabricated
    source entities. Missing or ambiguous source identity is refused; coordinates
    and array positions never establish entity identity.

    Occurrence paths are explicit source-entity identity components. A repeated
    entity ID is valid only when its authored occurrence path distinguishes it.

    ``facet_planes`` optionally declares an exact plane per facet as rows
    ``(origin, first_axis, second_axis)`` of shape ``(facet_count, 3, 3)``; an
    all-NaN row declares none, a declared row is finite with exactly
    independent axes. ``facet_plane_tolerances`` (``(facet_count,)``, finite and
    nonnegative, default 0) bounds how far a vertex may lie from its facet's
    plane beyond rounding. A declared facet's triangles form one face group
    whatever their binary64 rounding (its internal edges are not PLC edges)
    once the plane admits each of its vertices: the plane meets the vertex's
    rounding box or lies within the tolerance; recovery refuses otherwise.
    """

    __strict_contract__ = True

    vertices: HostFloat64[PlcVertexDim, Literal[3]]
    polygon_offsets: HostInt64[PlcPolygonOffsetDim]
    polygon_vertices: HostInt64[PlcLoopDim]
    polygon_facets: HostInt64[PlcPolygonDim]
    facet_regions: HostInt64[PlcFacetDim, Literal[2]]
    segments: HostInt64[PlcSegmentDim, Literal[2]]
    geometry: CellGeometrySpec | None
    facet_planes: HostFloat64[PlcFacetDim, Literal[3], Literal[3]] | None
    facet_plane_tolerances: HostFloat64[PlcFacetDim] | None
    vertex_ids: tuple[str, ...] = eqx.field(static=True)
    vertex_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    facet_ids: tuple[str, ...] = eqx.field(static=True)
    facet_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    segment_ids: tuple[str, ...] = eqx.field(static=True)
    segment_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    vertex_entity_ids: tuple[int, ...] = eqx.field(static=True)
    facet_entity_ids: tuple[int, ...] = eqx.field(static=True)
    segment_entity_ids: tuple[int, ...] = eqx.field(static=True)
    plc_edge_vertices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    plc_edge_facets: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plc_edge_ids: tuple[str, ...] = eqx.field(static=True)
    plc_edge_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    plc_edge_entity_ids: tuple[int, ...] = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    region_entity_ids: tuple[int, ...] = eqx.field(static=True)
    boundary: PlcBoundaryPolicy = eqx.field(static=True)
    cad_source_charts: SurfaceSourceCharts | None
    cad_embedding: GlobalEmbeddingCertificate | None
    cad_fidelity: SourceFidelityCertificate | None
    cad_fidelity_tolerance: float | None = eqx.field(static=True)
    cad_maximum_normal_deviation: float | None = eqx.field(static=True)
    cad_vertex_rows: np.ndarray | None
    cad_segment_edge_ids: np.ndarray | None
    cad_polygon_cell_ids: np.ndarray | None
    complex_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        vertices: ConvertibleToArray,
        polygons: Sequence[ConvertibleToArray] | np.ndarray,
        polygon_facets: ConvertibleToArray,
        facet_regions: ConvertibleToArray,
        region_ids: tuple[str, ...],
        /,
        *,
        segments: ConvertibleToArray | None = None,
        vertex_ids: tuple[str, ...],
        vertex_occurrence_paths: tuple[tuple[str, ...], ...],
        facet_ids: tuple[str, ...],
        facet_occurrence_paths: tuple[tuple[str, ...], ...],
        segment_ids: tuple[str, ...],
        segment_occurrence_paths: tuple[tuple[str, ...], ...],
        region_occurrence_paths: tuple[tuple[str, ...], ...],
        boundary: PlcBoundaryPolicy = "conforming",
        geometry: CellGeometrySpec | None = None,
        facet_planes: ConvertibleToArray | None = None,
        facet_plane_tolerances: ConvertibleToArray | None = None,
        cad_source_charts: SurfaceSourceCharts | None = None,
        cad_embedding: GlobalEmbeddingCertificate | None = None,
        cad_fidelity: SourceFidelityCertificate | None = None,
        cad_fidelity_tolerance: float | None = None,
        cad_maximum_normal_deviation: float | None = None,
        cad_vertex_rows: ConvertibleToArray | None = None,
        cad_segment_edge_ids: ConvertibleToArray | None = None,
        cad_polygon_cell_ids: ConvertibleToArray | None = None,
    ) -> None:
        if isinstance(polygons, np.ndarray) and polygons.ndim != 2:
            raise TypeError("polygons must be a sequence of one-dimensional index loops.")
        loops = tuple(_integer_array(loop, "polygons") for loop in polygons)
        if not loops:
            raise ValueError("polygons must contain at least one oriented loop.")
        if any(loop.ndim != 1 or loop.shape[0] < 3 for loop in loops):
            raise ValueError(
                "Every polygon loop must be one-dimensional with at least three vertices."
            )
        total_loop_vertices = sum(loop.shape[0] for loop in loops)
        if total_loop_vertices > np.iinfo(np.int64).max:
            raise OverflowError("The packed polygon incidence extent exceeds int64.")

        scope = Scope()
        points = parse(
            np.array(vertices, dtype=np.float64, copy=True, order="C"),
            HostFloat64[PlcVertexDim, Literal[3]],
            "vertices",
            scope=scope,
        )
        offsets_raw = np.empty((len(loops) + 1,), dtype=np.int64)
        offsets_raw[0] = 0
        for row, loop in enumerate(loops, start=1):
            offsets_raw[row] = offsets_raw[row - 1] + loop.shape[0]
        offsets = parse(
            offsets_raw,
            HostInt64[PlcPolygonOffsetDim],
            "polygon_offsets",
            scope=scope,
        )
        flat = parse(
            np.concatenate(loops),
            HostInt64[PlcLoopDim],
            "polygon_vertices",
            scope=scope,
        )
        owners = parse(
            _integer_array(polygon_facets, "polygon_facets"),
            HostInt64[PlcPolygonDim],
            "polygon_facets",
            scope=scope,
        )
        incidence = parse(
            _integer_array(facet_regions, "facet_regions"),
            HostInt64[PlcFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        curves = parse(
            np.zeros((0, 2), dtype=np.int64)
            if segments is None
            else _integer_array(segments, "segments"),
            HostInt64[PlcSegmentDim, Literal[2]],
            "segments",
            scope=scope,
        )
        region_names = tuple(
            normalized_identifier(identifier, "region_id") for identifier in region_ids
        )
        if not region_names:
            raise ValueError(
                "region_ids must contain at least one authored source region."
            )
        policy = parse(boundary, PlcBoundaryPolicy, "boundary")

        if owners.shape[0] != len(loops):
            raise ValueError("polygon_facets must name one facet for each polygon loop.")
        if not np.all(np.isfinite(points)):
            raise ValueError("vertices must contain only finite coordinates.")
        vertex_count = points.shape[0]
        if np.any((flat < 0) | (flat >= vertex_count)) or np.any(
            (curves < 0) | (curves >= vertex_count)
        ):
            raise ValueError(
                "Polygon and segment incidence must index declared vertices."
            )
        facet_count = incidence.shape[0]
        if np.any((owners < 0) | (owners >= facet_count)):
            raise ValueError("polygon_facets must index declared facets.")
        if np.setdiff1d(np.arange(facet_count), owners).size:
            raise ValueError("Every declared facet must own at least one polygon.")
        if np.any((incidence < -1) | (incidence >= len(region_names))) or np.any(
            np.all(incidence < 0, axis=1)
        ):
            raise ValueError(
                "Each facet must separate at least one declared region from a region or void."
            )
        if np.setdiff1d(np.arange(len(region_names)), incidence).size:
            raise ValueError(
                "Every declared region must occur in the facet incidence table."
            )
        if geometry is not None and geometry.coordinates.shape != points.shape:
            raise ValueError(
                "geometry must bind one source coordinate row per PLC vertex."
            )
        planes, plane_tolerances = _declared_facet_planes(
            facet_planes, facet_plane_tolerances, scope
        )
        vertex_paths = _occurrence_path_bank(
            vertex_occurrence_paths, vertex_count, "vertex_occurrence_paths"
        )
        facet_paths = _occurrence_path_bank(
            facet_occurrence_paths, facet_count, "facet_occurrence_paths"
        )
        segment_paths = _occurrence_path_bank(
            segment_occurrence_paths, curves.shape[0], "segment_occurrence_paths"
        )
        region_paths = _occurrence_path_bank(
            region_occurrence_paths, len(region_names), "region_occurrence_paths"
        )
        if len(set(zip(region_names, region_paths, strict=True))) != len(region_names):
            raise ValueError(
                "Region IDs and occurrence paths must uniquely identify each source region."
            )
        vertex_names = _identifier_bank(
            vertex_ids, vertex_paths, vertex_count, "vertex_ids"
        )
        facet_names = _identifier_bank(facet_ids, facet_paths, facet_count, "facet_ids")
        segment_names = _identifier_bank(
            segment_ids, segment_paths, curves.shape[0], "segment_ids"
        )
        vertex_entities = _numeric_entity_ids("plc-vertex", vertex_names, vertex_paths)
        facet_entities = _numeric_entity_ids("plc-facet", facet_names, facet_paths)
        segment_entities = _numeric_entity_ids(
            "plc-segment", segment_names, segment_paths
        )
        region_entities = _numeric_entity_ids("plc-region", region_names, region_paths)
        cad_values = (
            cad_source_charts,
            cad_fidelity_tolerance,
            cad_embedding,
            cad_fidelity,
            cad_vertex_rows,
            cad_segment_edge_ids,
            cad_polygon_cell_ids,
        )
        if any(value is not None for value in cad_values) and any(
            value is None for value in cad_values
        ):
            raise ValueError("CAD PLC incidence evidence must be supplied together.")
        if cad_source_charts is None and cad_maximum_normal_deviation is not None:
            raise ValueError("CAD normal tolerance requires source charts.")
        verified_segments: frozenset[int] = frozenset()
        normalized_vertex_rows: np.ndarray | None = None
        normalized_segment_edge_ids: np.ndarray | None = None
        normalized_polygon_cell_ids: np.ndarray | None = None
        if cad_source_charts is not None:
            assert cad_fidelity_tolerance is not None and cad_vertex_rows is not None
            assert cad_embedding is not None and cad_fidelity is not None
            assert cad_segment_edge_ids is not None and cad_polygon_cell_ids is not None
            normalized_vertex_rows = _integer_array(
                cad_vertex_rows, "cad_vertex_rows"
            ).copy()
            normalized_segment_edge_ids = _integer_array(
                cad_segment_edge_ids, "cad_segment_edge_ids"
            ).copy()
            normalized_polygon_cell_ids = _integer_array(
                cad_polygon_cell_ids, "cad_polygon_cell_ids"
            ).copy()
            for values in (
                normalized_vertex_rows,
                normalized_segment_edge_ids,
                normalized_polygon_cell_ids,
            ):
                if values.ndim != 1 or np.any(values < 0):
                    raise ValueError(
                        "CAD PLC incidence maps must be non-negative integer vectors."
                    )
                values.setflags(write=False)
            verified_segments = _cad_segment_incidence(
                cad_source_charts,
                cad_fidelity_tolerance,
                cad_maximum_normal_deviation,
                cad_embedding,
                cad_fidelity,
                normalized_vertex_rows,
                normalized_segment_edge_ids,
                normalized_polygon_cell_ids,
                points,
                loops,
                owners,
                curves,
                vertex_names,
                vertex_paths,
                facet_names,
                facet_paths,
                incidence,
                region_names,
                region_paths,
                segment_names,
                segment_paths,
            )
        (
            plc_edge_vertices,
            plc_edge_facets,
            plc_edge_ids,
            plc_edge_paths,
            plc_edge_entities,
        ) = _plc_edge_records(
            offsets,
            flat,
            owners,
            curves,
            vertex_names,
            vertex_paths,
            facet_names,
            facet_paths,
            segment_names,
            segment_paths,
            verified_segments,
        )

        for array in (points, offsets, flat, owners, incidence, curves):
            array.setflags(write=False)
        if planes is not None and plane_tolerances is not None:
            planes.setflags(write=False)
            plane_tolerances.setflags(write=False)
        self.vertices = points
        self.polygon_offsets = offsets
        self.polygon_vertices = flat
        self.polygon_facets = owners
        self.facet_regions = incidence
        self.segments = curves
        self.geometry = geometry
        self.facet_planes = planes
        self.facet_plane_tolerances = plane_tolerances
        self.vertex_ids = vertex_names
        self.vertex_occurrence_paths = vertex_paths
        self.facet_ids = facet_names
        self.facet_occurrence_paths = facet_paths
        self.segment_ids = segment_names
        self.segment_occurrence_paths = segment_paths
        self.vertex_entity_ids = vertex_entities
        self.facet_entity_ids = facet_entities
        self.segment_entity_ids = segment_entities
        self.plc_edge_vertices = plc_edge_vertices
        self.plc_edge_facets = plc_edge_facets
        self.plc_edge_ids = plc_edge_ids
        self.plc_edge_occurrence_paths = plc_edge_paths
        self.plc_edge_entity_ids = plc_edge_entities
        self.region_occurrence_paths = region_paths
        self.region_ids = region_names
        self.region_entity_ids = region_entities
        self.boundary = policy
        self.cad_source_charts = cad_source_charts
        self.cad_embedding = cad_embedding
        self.cad_fidelity = cad_fidelity
        self.cad_fidelity_tolerance = cad_fidelity_tolerance
        self.cad_maximum_normal_deviation = cad_maximum_normal_deviation
        self.cad_vertex_rows = normalized_vertex_rows
        self.cad_segment_edge_ids = normalized_segment_edge_ids
        self.cad_polygon_cell_ids = normalized_polygon_cell_ids
        self.complex_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-complex",
                "vertices": array_tree_fingerprint(points),
                "polygon_offsets": array_tree_fingerprint(offsets),
                "polygon_vertices": array_tree_fingerprint(flat),
                "polygon_facets": array_tree_fingerprint(owners),
                "facet_regions": array_tree_fingerprint(incidence),
                "segments": array_tree_fingerprint(curves),
                "vertex_ids": vertex_names,
                "vertex_occurrence_paths": vertex_paths,
                "facet_ids": facet_names,
                "facet_occurrence_paths": facet_paths,
                "segment_ids": segment_names,
                "segment_occurrence_paths": segment_paths,
                "plc_edge_vertices": plc_edge_vertices,
                "plc_edge_ids": plc_edge_ids,
                "plc_edge_occurrence_paths": plc_edge_paths,
                "region_ids": region_names,
                "region_occurrence_paths": region_paths,
                "boundary": policy,
                **(
                    {
                        "cad_incidence": (
                            cad_source_charts.source_chart_id,
                            cad_fidelity_tolerance,
                            cad_maximum_normal_deviation,
                            cad_embedding.certificate_id,
                            cad_fidelity.certificate_id,
                            array_tree_fingerprint(normalized_vertex_rows),
                            array_tree_fingerprint(normalized_segment_edge_ids),
                            array_tree_fingerprint(normalized_polygon_cell_ids),
                        )
                    }
                    if cad_source_charts is not None
                    and cad_embedding is not None
                    and cad_fidelity is not None
                    else {}
                ),
                **(
                    {"geometry": array_tree_fingerprint(geometry)}
                    if geometry is not None
                    else {}
                ),
                **(
                    {
                        "facet_planes": array_tree_fingerprint(planes),
                        "facet_plane_tolerances": array_tree_fingerprint(
                            plane_tolerances
                        ),
                    }
                    if planes is not None and plane_tolerances is not None
                    else {}
                ),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PiecewiseLinearComplex],
        value: PiecewiseLinearComplex,
        /,
        *,
        cad_source_charts: SurfaceSourceCharts | None = None,
        cad_embedding: GlobalEmbeddingCertificate | None = None,
        cad_fidelity: SourceFidelityCertificate | None = None,
    ) -> PiecewiseLinearComplex:
        """Rebuild packed loops and verify their retained scientific identity."""
        if type(value) is not cls:
            raise TypeError(
                "Stored PLC fields must have the registered PiecewiseLinearComplex type."
            )
        from .._model._structure import SurfaceSourceChartsArchiveRecipe

        stored_charts = value.cad_source_charts
        if cad_source_charts is not None:
            if type(stored_charts) is SurfaceSourceChartsArchiveRecipe:
                stored_chart_id = stored_charts.expected_source_chart_id
            elif type(stored_charts) is SurfaceSourceCharts:
                stored_chart_id = stored_charts.source_chart_id
            else:
                raise ValueError("Stored PLC has no source-chart authority to restore.")
            if cad_source_charts.source_chart_id != stored_chart_id:
                raise ValueError(
                    "Restored PLC source charts differ from their stored authority."
                )
        else:
            if (
                stored_charts is not None
                and type(stored_charts) is not SurfaceSourceCharts
            ):
                raise ValueError(
                    "Restore the source-chart archive recipe before the PLC."
                )
            cad_source_charts = stored_charts
        if cad_embedding is not None:
            if (
                value.cad_embedding is None
                or cad_embedding.certificate_id != value.cad_embedding.certificate_id
            ):
                raise ValueError(
                    "Restored PLC embedding differs from its stored certificate."
                )
        else:
            cad_embedding = value.cad_embedding
        if cad_fidelity is not None:
            if (
                value.cad_fidelity is None
                or cad_fidelity.certificate_id != value.cad_fidelity.certificate_id
            ):
                raise ValueError(
                    "Restored PLC fidelity differs from its stored certificate."
                )
        else:
            cad_fidelity = value.cad_fidelity
        offsets = np.asarray(value.polygon_offsets, dtype=np.int64)
        vertices = np.asarray(value.polygon_vertices, dtype=np.int64)
        if (
            offsets.ndim != 1
            or offsets.shape[0] < 2
            or offsets.shape[0] != value.polygon_facets.shape[0] + 1
            or offsets[0] != 0
            or offsets[-1] != vertices.size
            or np.any(offsets[1:] - offsets[:-1] < 3)
        ):
            raise ValueError(
                "Stored polygon offsets must partition loops of at least three indices."
            )
        loops = tuple(
            vertices[start:end]
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        )
        fresh = cls(
            value.vertices,
            loops,
            value.polygon_facets,
            value.facet_regions,
            value.region_ids,
            segments=value.segments,
            vertex_ids=value.vertex_ids,
            vertex_occurrence_paths=value.vertex_occurrence_paths,
            facet_ids=value.facet_ids,
            segment_ids=value.segment_ids,
            facet_occurrence_paths=value.facet_occurrence_paths,
            boundary=value.boundary,
            geometry=value.geometry,
            facet_planes=value.facet_planes,
            facet_plane_tolerances=value.facet_plane_tolerances,
            segment_occurrence_paths=value.segment_occurrence_paths,
            region_occurrence_paths=value.region_occurrence_paths,
            cad_source_charts=cad_source_charts,
            cad_embedding=cad_embedding,
            cad_fidelity=cad_fidelity,
            cad_fidelity_tolerance=value.cad_fidelity_tolerance,
            cad_maximum_normal_deviation=value.cad_maximum_normal_deviation,
            cad_vertex_rows=value.cad_vertex_rows,
            cad_segment_edge_ids=value.cad_segment_edge_ids,
            cad_polygon_cell_ids=value.cad_polygon_cell_ids,
        )
        if fresh.complex_id != value.complex_id:
            raise ValueError("Stored PLC fields do not match their complex identity.")
        return fresh

    @property
    def facet_count(self) -> int:
        return self.facet_regions.shape[0]

    @property
    def region_count(self) -> int:
        return len(self.region_ids)


__all__ = ["PiecewiseLinearComplex", "PlcBoundaryPolicy"]
