#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact source-cell corner closure for convex balanced-grid cut pieces.

This owner does not project vertices, weld coordinates, or publish meshes. A
prepared closure retains the original coordinate authority and scientific
incidence; publication still requires independent map, embedding, source and
coverage certificates. Nonsimple links have explicit witnesses and are not
misrepresented as impossible all-hex domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, TYPE_CHECKING

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._meshcore import current_native_execution_budget, current_native_host_workspace
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_enclosure import rounded_point
from ..discretization._exact_plc_geometry import ExactPlcCellGeometryConvexSource
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._mesh_certificates import (
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)
from ._contracts import MeshingLimits
from ._quad_generation import _budget, _failure, _family_host_array


if TYPE_CHECKING:
    from ._hex_generation import NativeHexGridSchedule


@dataclass(frozen=True, slots=True)
class CutCornerLinkWitness:
    cell_id: int
    vertex_id: int
    edge_ids: tuple[int, ...]
    face_ids: tuple[int, ...]


class CutCornerLinkError(ValueError):
    """An actual source-cell link needs a nonsimple conforming template.

    This is a capability obstruction, not a mathematical nonexistence theorem.
    """

    cut_failure_prefix: tuple[Any, ...] | None

    def __init__(self, witness: CutCornerLinkWitness) -> None:
        self.witness = witness
        self.cut_failure_prefix = None
        super().__init__(
            f"Cut cell {witness.cell_id}, vertex {witness.vertex_id} has "
            f"{len(witness.edge_ids)} incident edges and {len(witness.face_ids)} "
            "incident faces; the three-direction corner template does not close this link."
        )


@dataclass(frozen=True, slots=True)
class CutCornerHexPreparation:
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    coordinate_source: CellGeometrySpec | ExactPlcCellGeometryConvexSource
    exact_vertices: tuple[tuple[Fraction, ...], ...]
    # Original-source coefficient combinations, never rounded coordinate keys.
    support_offsets: np.ndarray
    support_vertices: np.ndarray
    support_denominators: np.ndarray
    vertex_parent_dimensions: np.ndarray
    vertex_parent_rows: np.ndarray
    hexes: np.ndarray
    parent_cells: np.ndarray
    corner_orientations: np.ndarray
    work_units: int
    preparation_id: str


def _cross(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _difference(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _cut_host_scratch(
    connectivity: PolyhedralConnectivity,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
) -> int:
    nv, ne, nf, nc = (
        connectivity.vertex_count,
        connectivity.edge_count,
        connectivity.face_count,
        connectivity.cell_count,
    )
    corner_count = int(connectivity.cell_vertex_values.size)
    supports_count = (
        nv + 2 * ne + int(connectivity.face_vertex_values.size) + corner_count
    )
    bit_bound = (
        geometry.maximum_bits
        if isinstance(geometry, ExactPlcCellGeometryConvexSource)
        else 1075
        if geometry.exact_source is None
        else geometry.exact_source.maximum_bits
    )
    maximum_support = max(
        2, connectivity.maximum_face_arity, connectivity.maximum_cell_vertices
    )
    rational_bits = maximum_support * bit_bound + (maximum_support + 1).bit_length()
    rational_bytes = 128 + 2 * ((rational_bits + 7) // 8)
    return (
        3 * (nv + ne + nf + nc) * rational_bytes
        + 128 * (supports_count + 8 * corner_count)
        + 24 * rational_bytes
    )


def _prepare_cut_corner_hexes(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    limits: MeshingLimits,
    /,
    *,
    immutable_face_ids: tuple[int, ...] = (),
) -> CutCornerHexPreparation:
    """Close each simple convex source cut polyhedron into actual corner hexes.

    Every source face is quadrangulated identically by its authoritative edge
    and face nodes. Thus cavities and material interfaces use the same nodes
    on both sides without a coordinate tolerance. Source-face immutability is
    explicit: this template subdivides every face and cannot retain it unchanged.
    Exact convexity and corner orientation are prerequisites, not sampled fits.
    """
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Cut corner closure requires canonical polyhedral incidence.")
    if mesh.topology.dimension != 3 or mesh.coordinates.shape[1] != 3:
        raise ValueError("Cut corner closure requires a three-dimensional volume.")
    if mesh.periodic_topology is not None:
        raise _failure(
            "Cut corner closure requires retained quotient source-cell incidence."
        )
    nv, ne, nf, nc = (
        connectivity.vertex_count,
        connectivity.edge_count,
        connectivity.face_count,
        connectivity.cell_count,
    )
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    if immutable_face_ids:
        unknown = set(immutable_face_ids).difference(face_ids.tolist())
        if unknown:
            raise ValueError(f"Immutable source faces are undeclared: {sorted(unknown)}.")
        offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        nonquad = tuple(
            (int(identifier), int(offsets[row + 1] - offsets[row]))
            for row, identifier in enumerate(face_ids)
            if identifier in immutable_face_ids and offsets[row + 1] - offsets[row] != 4
        )
        if nonquad:
            raise _failure(
                f"Immutable source faces (scientific ID, arity) {nonquad} cannot remain unchanged faces of a pure hex complex."
            )
        raise _failure(
            f"This corner template subdivides immutable source faces {immutable_face_ids}; a boundary-preserving alternative template is required."
        )
    # Admission precedes the source bank, topology dictionaries and output banks.
    face_entries = int(connectivity.face_vertex_values.size)
    corner_count = int(connectivity.cell_vertex_values.size)
    output_vertices = nv + ne + nf + nc
    supports_count = nv + 2 * ne + face_entries + corner_count
    pair_work = nc * connectivity.maximum_cell_faces * connectivity.maximum_cell_vertices
    work_bound = 256 * (face_entries + corner_count + ne + nc + pair_work)
    scratch = _cut_host_scratch(connectivity, geometry)
    _budget(limits, corner_count, output_vertices, 8 * corner_count, scratch, work_bound)
    if (
        12 * corner_count > limits.maximum_edges
        or 6 * corner_count > limits.maximum_faces
    ):
        raise _failure(
            "Cut corner closure edge/face admission exceeds original entity bounds.",
            resource=True,
        )
    data_bound = 8 * (
        3 * output_vertices + 8 * corner_count + 3 * supports_count + 4 * output_vertices
    )
    if data_bound > limits.maximum_data_bytes:
        raise _failure(
            "Cut corner closure exceeds original publication bytes.", resource=True
        )
    source: tuple[tuple[Fraction, ...], ...]
    if isinstance(geometry, ExactPlcCellGeometryConvexSource):
        parent_mesh, parent_geometry, target_mesh = geometry._owners()
        if target_mesh.topology_id != mesh.topology_id:
            raise ValueError(
                "Cut coordinate authority names a different scientific source topology."
            )
        source = geometry.prepare(mesh.coordinates).vertices
        original_geometry = parent_geometry
        source_binding = geometry.source_id
    elif isinstance(geometry, CellGeometrySpec):
        source = tuple(
            tuple(Fraction(value) for value in point)
            for point in geometry.source_coordinates()
        )
        original_geometry = geometry
        source_binding = cell_geometry_id(geometry)
    else:
        raise TypeError("Cut closure requires an actual owning coordinate source.")
    if len(source) != nv or any(len(point) != 3 for point in source):
        raise ValueError(
            "Cut source geometry must bind its complete vertex coefficient bank."
        )
    if any(
        not np.array_equal(rounded_point(point), np.asarray(mesh.coordinates)[row])
        for row, point in enumerate(source)
    ):
        raise ValueError(
            "Cut source carrier does not agree with its correctly rounded original coefficients."
        )
    edges = np.asarray(connectivity.edges, dtype=np.int64)
    fo = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    fv = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    co = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cf = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    vo = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    vv = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    cell_ids = np.asarray(mesh.entity_set(3).entity_ids, dtype=np.int64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    supports: list[tuple[int, ...]] = [(row,) for row in range(nv)]
    supports.extend(tuple(int(v) for v in edge) for edge in edges)
    supports.extend(tuple(int(v) for v in fv[fo[row] : fo[row + 1]]) for row in range(nf))
    supports.extend(tuple(int(v) for v in vv[vo[row] : vo[row + 1]]) for row in range(nc))
    exact: list[tuple[Fraction, ...]] = list(source)
    work = 0
    execution = current_native_execution_budget()

    def charge(amount: int) -> None:
        nonlocal work
        if work + amount > limits.maximum_work_units:
            raise _failure(
                "Cut corner preparation exhausts original work.", resource=True
            )
        if execution is not None:
            execution.charge(work=amount)
        work += amount

    for support in supports[nv:]:
        charge(6 * len(support))
        exact.append(
            (
                sum((source[v][0] for v in support), Fraction()) / len(support),
                sum((source[v][1] for v in support), Fraction()) / len(support),
                sum((source[v][2] for v in support), Fraction()) / len(support),
            )
        )
    edge_lookup = {
        tuple(sorted((int(edges[row, 0]), int(edges[row, 1])))): row
        for row in range(edges.shape[0])
    }
    signs = _family_host_array((corner_count,), np.int8)
    output = 0
    for cell in range(nc):
        vertices = vv[vo[cell] : vo[cell + 1]].tolist()
        faces = cf[co[cell] : co[cell + 1]].tolist()
        incident_faces = {vertex: [] for vertex in vertices}
        incident_edges = {vertex: set() for vertex in vertices}
        pair_faces: dict[tuple[int, int, int], int] = {}
        for face in faces:
            loop = fv[fo[face] : fo[face + 1]].tolist()
            origin = exact[loop[0]]
            normal = (Fraction(0),) * 3
            for position in range(1, len(loop) - 1):
                charge(30)
                normal = _cross(
                    _difference(exact[loop[position]], origin),
                    _difference(exact[loop[position + 1]], origin),
                )
                if any(normal):
                    break
            if not any(normal):
                raise _failure(
                    f"Cut source cell {cell_ids[cell]} has a degenerate face {face_ids[face]}."
                )
            side = 0
            for vertex in vertices:
                charge(12)
                distance = sum(
                    (
                        a * b
                        for a, b in zip(
                            normal, _difference(exact[vertex], origin), strict=True
                        )
                    ),
                    Fraction(0),
                )
                if vertex in loop and distance:
                    raise _failure(f"Cut face {face_ids[face]} is not exactly planar.")
                sign = (distance > 0) - (distance < 0)
                if sign and side and side != sign:
                    raise _failure(
                        f"Cut cell {cell_ids[cell]} is not convex at face {face_ids[face]}."
                    )
                side = sign or side
            for position, vertex in enumerate(loop):
                charge(8)
                before, after = loop[position - 1], loop[(position + 1) % len(loop)]
                first = edge_lookup[tuple(sorted((vertex, before)))]
                second = edge_lookup[tuple(sorted((vertex, after)))]
                incident_edges[vertex].update((first, second))
                incident_faces[vertex].append(face)
                key = (vertex, min(first, second), max(first, second))
                if key in pair_faces:
                    raise _failure(
                        "Cut source corner link repeats an incident edge pair."
                    )
                pair_faces[key] = face
        for vertex in vertices:
            edge_rows = sorted(incident_edges[vertex])
            face_rows = incident_faces[vertex]
            if len(edge_rows) != 3 or len(face_rows) != 3:
                raise CutCornerLinkError(
                    CutCornerLinkWitness(
                        int(cell_ids[cell]),
                        int(vertex_ids[vertex]),
                        tuple(int(edge_ids[row]) for row in edge_rows),
                        tuple(int(face_ids[row]) for row in face_rows),
                    )
                )
            directions = []
            for edge in edge_rows:
                a, b = edges[edge]
                other = int(b if a == vertex else a)
                directions.append(_difference(exact[other], exact[vertex]))
            charge(40)
            determinant = sum(
                (
                    a * b
                    for a, b in zip(
                        directions[0], _cross(directions[1], directions[2]), strict=True
                    )
                ),
                Fraction(0),
            )
            if not determinant:
                raise _failure(
                    f"Cut cell {cell_ids[cell]} has a rank-deficient corner at vertex {vertex_ids[vertex]}."
                )
            signs[output] = 1 if determinant > 0 else -1
            for first, second in (
                (edge_rows[0], edge_rows[1]),
                (edge_rows[0], edge_rows[2]),
                (edge_rows[1], edge_rows[2]),
            ):
                if (vertex, min(first, second), max(first, second)) not in pair_faces:
                    raise _failure("Cut source corner link is not a closed three-cycle.")
            output += 1
    from .._meshcore import polyhedron_corner_hexes

    remaining = limits.maximum_work_units - work
    if remaining <= 0:
        raise _failure("Cut corner topology exhausts original work.", resource=True)
    hexes, parents, native_work = polyhedron_corner_hexes(
        edges,
        fo,
        fv,
        co,
        cf,
        vo,
        vv,
        signs,
        nv,
        maximum_cells=limits.maximum_cells,
        maximum_scratch_bytes=limits.maximum_scratch_bytes,
        maximum_work_units=remaining,
    )
    work += native_work
    offsets = _family_host_array((output_vertices + 1,), np.int64)
    values = _family_host_array((supports_count,), np.int64)
    denominators = _family_host_array((output_vertices,), np.int64)
    dimensions = _family_host_array((output_vertices,), np.int8)
    rows = _family_host_array((output_vertices,), np.int64)
    offsets[0] = 0
    cursor = 0
    for row, support in enumerate(supports):
        values[cursor : cursor + len(support)] = support
        cursor += len(support)
        offsets[row + 1] = cursor
        denominators[row] = len(support)
    for dimension, start, count in (
        (0, 0, nv),
        (1, nv, ne),
        (2, nv + ne, nf),
        (3, nv + ne + nf, nc),
    ):
        dimensions[start : start + count] = dimension
        rows[start : start + count] = np.arange(count, dtype=np.int64)
    identity = canonical_fingerprint(
        {
            "kind": "exact-cut-corner-preparation",
            "source_topology": mesh.topology_id,
            "source_geometry": source_binding,
            "supports": tuple(supports),
            "hexes": tuple(tuple(int(v) for v in row) for row in hexes),
        }
    )
    for bank in (offsets, values, denominators, dimensions, rows, hexes, parents, signs):
        bank.flags.writeable = False
    return CutCornerHexPreparation(
        mesh,
        original_geometry,
        geometry,
        tuple(exact),
        offsets,
        values,
        denominators,
        dimensions,
        rows,
        hexes,
        parents,
        signs,
        work,
        identity,
    )


@dataclass(frozen=True, slots=True)
class CutCornerHexConstruction:
    preparation: CutCornerHexPreparation
    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray


def compose_original_plc_supports(
    prepared: CutCornerHexPreparation,
    /,
) -> tuple[np.ndarray, np.ndarray, tuple[Fraction, ...]]:
    """Compose actual cut barycenters into original PLC SCI corner coefficients."""
    authority = prepared.coordinate_source
    if not isinstance(authority, ExactPlcCellGeometryConvexSource):
        raise TypeError(
            "Original PLC support composition requires its actual cut-coordinate owner."
        )
    parent_offsets = np.asarray(authority.support_offsets, dtype=np.int64)
    parent_vertices = np.asarray(authority.support_vertices, dtype=np.int64)
    parent_coefficients = authority.support_coefficients
    count = len(prepared.exact_vertices)
    offsets = _family_host_array((count + 1,), np.int64)
    vertices = _family_host_array((4 * count,), np.int64)
    coefficients: list[Fraction] = []
    offsets[0] = 0
    cursor = 0
    for row in range(count):
        support = prepared.support_vertices[
            prepared.support_offsets[row] : prepared.support_offsets[row + 1]
        ]
        denominator = int(prepared.support_denominators[row])
        weights: dict[int, Fraction] = {}
        for cut_vertex in support:
            for slot in range(
                int(parent_offsets[cut_vertex]), int(parent_offsets[cut_vertex + 1])
            ):
                original = int(parent_vertices[slot])
                weights[original] = (
                    weights.get(original, Fraction(0))
                    + parent_coefficients[slot] / denominator
                )
        if (
            len(weights) > 4
            or sum(weights.values(), Fraction(0)) != 1
            or min(weights.values()) <= 0
        ):
            raise ValueError(
                "Actual cut barycenter has no convex original P1 source-cell support."
            )
        for original, coefficient in sorted(weights.items()):
            vertices[cursor] = original
            coefficients.append(coefficient)
            cursor += 1
        offsets[row + 1] = cursor
    return offsets, vertices[:cursor], tuple(coefficients)


def realize_cut_corner_hexes(
    prepared: CutCornerHexPreparation,
    limits: MeshingLimits,
    /,
    *,
    certificate_limits: MeshCertificateLimits,
) -> CutCornerHexConstruction:
    """Realize exact corner maps and certify positivity/global embedding.

    This does not certify original-domain coverage. The caller must compose the
    independently retained cut/source-cell partition theorem with original PLC
    source association and coverage before publication; no candidate-only route
    success is returned here.
    """
    from ..discretization._cell_geometry_validity import certify_cell_geometry_validity
    from ..discretization._cell_mesh import CellBlock
    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometryLinearActionSource,
        ExactPowerCellGeometryRestrictionSource,
        ExactPowerCellGeometrySource,
    )
    from ..geometry._mesh_certificates import certify_global_embedding
    from ._quad_generation import _vertex_ids

    if not isinstance(certificate_limits, MeshCertificateLimits):
        raise TypeError(
            "Cut realization requires the original explicit certificate limits."
        )
    authority = prepared.coordinate_source
    offsets = prepared.support_offsets
    count = len(prepared.exact_vertices)
    width = int(np.max(np.diff(offsets)))
    scratch = count * width * (8 + 256) + count * 24
    _budget(
        limits,
        prepared.hexes.shape[0],
        count,
        prepared.hexes.size,
        scratch,
        16 * count * width,
    )
    coordinates = _family_host_array((count, 3), np.float64)
    for row, point in enumerate(prepared.exact_vertices):
        coordinates[row] = rounded_point(point)
    added = count - prepared.source_mesh.coordinates.shape[0]
    mesh = CellMesh(
        coordinates,
        (CellBlock("cut_hexes", "hexahedron", prepared.hexes),),
        vertex_global_ids=_vertex_ids(prepared.source_mesh, added),
        numeric_version=prepared.source_mesh.numeric_version,
    )
    if isinstance(authority, ExactPlcCellGeometryConvexSource):
        (
            parent_mesh,
            parent_geometry,
            _,
        ) = authority._owners()
        original_offsets, original_vertices, original_coefficients = (
            compose_original_plc_supports(prepared)
        )
        original_parents = np.asarray(authority.cell_parent_ids, dtype=np.int64)[
            prepared.parent_cells
        ]
        source = ExactPlcCellGeometryConvexSource(
            parent_mesh,
            parent_geometry,
            original_offsets,
            original_vertices,
            original_coefficients,
            mesh,
            original_parents,
        )
        geometry = CellGeometrySpec.plc(mesh, source)
    else:
        parent = authority.exact_source
        if not isinstance(
            parent,
            (
                ExactPowerCellGeometrySource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometryLinearActionSource,
            ),
        ):
            raise TypeError(
                "Cut realization requires an actual exact cut-cell source, not an authored point proxy."
            )
        parents = _family_host_array((count, width), np.int64)
        parents.fill(-1)
        coefficients = []
        for row in range(count):
            support = prepared.support_vertices[offsets[row] : offsets[row + 1]]
            parents[row, : support.size] = support
            coefficient = Fraction(1, int(prepared.support_denominators[row]))
            coefficients.append(
                (coefficient,) * support.size + (Fraction(0),) * (width - support.size)
            )
        actions = _family_host_array((count, 0), np.int64)
        source = ExactPowerCellGeometryLinearActionSource(
            parent, parents, tuple(coefficients), actions, periodic_preparation=None
        )
        geometry = CellGeometrySpec.power(mesh, source)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    if validity.invalid_count or validity.unresolved_count:
        raise _failure(
            "Exact cut-corner maps have invalid or unresolved whole-cell Jacobians."
        )
    embedding = certify_global_embedding(
        mesh, geometry, validity, limits=certificate_limits
    )
    if embedding.status != "certified":
        codes = tuple(finding.check for finding in embedding.findings)
        raise _failure(f"Exact cut-corner global embedding is not certified: {codes}.")
    from ._hex_generation import exact_mapped_volume_quality

    scaled, mean, aspect = exact_mapped_volume_quality(mesh, geometry)
    for bank in (scaled, mean, aspect):
        bank.flags.writeable = False
    return CutCornerHexConstruction(
        prepared, mesh, geometry, validity, embedding, scaled, mean, aspect
    )


def prepare_cut_corner_hexes(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    limits: MeshingLimits,
    /,
    *,
    immutable_face_ids: tuple[int, ...] = (),
) -> CutCornerHexPreparation:
    """Prepare exact source corner closure inside the original host-storage owner."""
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise TypeError("Cut corner closure requires canonical polyhedral incidence.")
    execution = current_native_execution_budget()
    if execution is None:
        return _prepare_cut_corner_hexes(
            mesh, geometry, limits, immutable_face_ids=immutable_face_ids
        )
    workspace = current_native_host_workspace()
    if workspace is None:
        raise _failure(
            "Cut corner preparation requires the original native host-storage owner.",
            resource=True,
        )
    scratch = _cut_host_scratch(mesh.connectivity, geometry)
    if scratch > limits.maximum_scratch_bytes:
        raise _failure(
            "Cut corner preparation exceeds original host scratch.", resource=True
        )
    starting = workspace.bound
    workspace.set_bound(starting + scratch)
    try:
        prepared = _prepare_cut_corner_hexes(
            mesh, geometry, limits, immutable_face_ids=immutable_face_ids
        )
    finally:
        workspace.set_bound(starting)
    workspace.retain_owner(prepared)
    return prepared


def certify_cut_corner_coverage(
    construction: CutCornerHexConstruction,
    original_domain: PiecewiseLinearDomain | MappedReferenceDomain,
    source_cell_regions: np.ndarray,
    /,
    *,
    certificate_limits: MeshCertificateLimits,
) -> tuple[DomainCoverageCertificate, np.ndarray]:
    """Certify the real hex maps against an independently retained original domain."""
    from ..geometry._mesh_certificates import certify_domain_coverage

    regions = np.asarray(source_cell_regions, dtype=np.int64)
    connectivity = construction.preparation.source_mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise RuntimeError("Cut coverage source lost polyhedral connectivity.")
    source_count = connectivity.cell_count
    if regions.shape != (source_count,):
        raise ValueError(
            "Original cut-source regions must name every source cell exactly once."
        )
    target_regions = _family_host_array(
        construction.preparation.parent_cells.shape, np.int64
    )
    target_regions[:] = regions[construction.preparation.parent_cells]
    coverage = certify_domain_coverage(
        construction.mesh,
        construction.geometry,
        original_domain,
        target_regions,
        embedding=construction.embedding,
        limits=certificate_limits,
    )
    if coverage.status != "certified":
        checks = tuple(finding.check for finding in coverage.findings)
        raise _failure(f"Original-domain cut-corner coverage is not certified: {checks}.")
    target_regions.flags.writeable = False
    return coverage, target_regions


def require_cut_corner_quality(
    construction: CutCornerHexConstruction,
    schedule: NativeHexGridSchedule,
) -> None:
    """Enforce the unchanged selected portfolio's whole-map quality requirements."""
    from ._hex_generation import NativeHexGridSchedule

    if not isinstance(schedule, NativeHexGridSchedule):
        raise TypeError("Cut quality requires the original explicit hex grid schedule.")
    for name, bank, bound, lower in (
        (
            "scaled Jacobian",
            construction.scaled_jacobian_lower,
            schedule.minimum_scaled_jacobian,
            True,
        ),
        ("mean ratio", construction.mean_ratio_lower, schedule.minimum_mean_ratio, True),
        (
            "aspect ratio",
            construction.aspect_ratio_upper,
            schedule.maximum_aspect_ratio,
            False,
        ),
    ):
        failures = np.flatnonzero(bank < bound if lower else bank > bound)
        if failures.size:
            cell_ids = np.asarray(
                construction.mesh.entity_set(3).entity_ids, dtype=np.int64
            )
            evidence = tuple((int(cell_ids[row]), float(bank[row])) for row in failures)
            raise _failure(
                f"Exact cut-corner {name} conflicts with original bound {bound}: {evidence}."
            )
