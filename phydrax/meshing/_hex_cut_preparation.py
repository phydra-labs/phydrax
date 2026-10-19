#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepare face-conforming corner hexes from convex balanced-grid cut pieces.

Source coefficients and scientific incidence determine every generated node.
Publication requires independent map, embedding, fidelity, and coverage
certificates. Nonsimple links retain their incident scientific entity IDs.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import equinox as eqx
import jax
import numpy as np

from phydrax.discretization._reference_cell_types import CellBlock
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.linalg._exact import determinant_exact
from phydrax.meshing._host_banks import failure, family_host_array

from .._fingerprint import canonical_fingerprint
from .._meshcore import (
    NativeExecutionScope,
    NativeHostStorageWorkspace,
    polyhedron_corner_hexes,
)
from .._strict import StrictModule
from ..discretization._cell_complex import PolyhedralConnectivity
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import rounded_point
from ..discretization._exact_plc_geometry import ExactPlcCellGeometryConvexSource
from ..discretization._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ..geometry._domain_coverage import (
    certify_domain_coverage,
    DomainCoverageCertificate,
    PiecewiseLinearDomain,
)
from ..geometry._global_embedding import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
)
from ..geometry._mapped_reference_domain import MappedReferenceDomain
from ..geometry._rational import rational_cross
from ..typing import checked
from ._contracts import MeshingLimits
from ._failures import CutCornerLinkIncidence
from ._family_execution import check_family_limits, coordinate_phase, vertex_ids
from ._hex_generation import exact_mapped_volume_quality, NativeHexGridSchedule


class CutCornerLinkError(ValueError):
    """Report the source incidence that needs a nonsimple corner template."""

    @checked
    def __init__(self, witness: CutCornerLinkIncidence) -> None:
        self.incidence = witness
        super().__init__(
            f"CutCornerLinkError.link must have three incident edges and faces; got "
            f"cell {witness.cell_id}, vertex {witness.vertex_id}, "
            f"{len(witness.edge_ids)} edges and {len(witness.face_ids)} faces"
        )


class CutCornerHexPreparation(StrictModule):
    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    coordinate_source: CellGeometrySpec | ExactPlcCellGeometryConvexSource
    exact_vertices: tuple[tuple[Fraction, ...], ...]
    # PLC source-coefficient combinations determine nodes without rounded point keys.
    support_offsets: np.ndarray
    support_vertices: np.ndarray
    support_denominators: np.ndarray
    vertex_parent_dimensions: np.ndarray
    vertex_parent_rows: np.ndarray
    hexes: np.ndarray
    parent_cells: np.ndarray
    corner_orientations: np.ndarray
    work_units: int
    preparation_id: str = eqx.field(static=True)


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
    match geometry:
        case ExactPlcCellGeometryConvexSource():
            bit_bound = geometry.maximum_bits
        case CellGeometrySpec():
            source = geometry.exact_source
            bit_bound = 1075 if source is None else source.maximum_bits
        case _:
            raise TypeError(
                f"cut_preparation.geometry must be a cut coordinate source; got {type(geometry).__name__}"
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


class _CutAdmission(StrictModule):
    connectivity: PolyhedralConnectivity
    face_ids: np.ndarray
    corner_count: int = eqx.field(static=True)
    output_vertices: int = eqx.field(static=True)
    supports_count: int = eqx.field(static=True)
    work_bound: int = eqx.field(static=True)


class _CutSource(StrictModule):
    vertices: tuple[tuple[Fraction, ...], ...]
    geometry: CellGeometrySpec
    binding: str = eqx.field(static=True)


class _CutIncidence(StrictModule):
    edges: np.ndarray
    face_offsets: np.ndarray
    face_vertices: np.ndarray
    cell_offsets: np.ndarray
    cell_faces: np.ndarray
    vertex_offsets: np.ndarray
    cell_vertices: np.ndarray
    cell_ids: np.ndarray
    vertex_ids: np.ndarray
    edge_ids: np.ndarray
    supports: tuple[tuple[int, ...], ...]


class _CutTopology(StrictModule):
    vertices: tuple[tuple[Fraction, ...], ...]
    signs: np.ndarray
    hexes: np.ndarray
    parents: np.ndarray
    work: int = eqx.field(static=True)


def _admit_cut_corner_closure(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    limits: MeshingLimits,
    immutable_face_ids: tuple[int, ...],
) -> _CutAdmission:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "mesh.connectivity must be PolyhedralConnectivity; got another connectivity"
        )
    if mesh.topology.dimension != 3 or mesh.coordinates.shape[1] != 3:
        raise ValueError(
            "mesh.dimension must be a three-dimensional volume in three coordinates; got incompatible dimensions"
        )
    if mesh.periodic_topology is not None:
        raise failure(
            "mesh.periodic_topology must have a supported quotient cut template; got unsupported periodic incidence"
        )
    nv, ne, nf, nc = (
        connectivity.vertex_count,
        connectivity.edge_count,
        connectivity.face_count,
        connectivity.cell_count,
    )
    face_ids = np.asarray(jax.device_get(mesh.entity_set(2).entity_ids), dtype=np.int64)
    if immutable_face_ids:
        _refuse_immutable_cut_faces(connectivity, face_ids, immutable_face_ids)
    face_entries = connectivity.face_vertex_values.size
    corner_count = connectivity.cell_vertex_values.size
    output_vertices = nv + ne + nf + nc
    supports_count = nv + 2 * ne + face_entries + corner_count
    pair_work = nc * connectivity.maximum_cell_faces * connectivity.maximum_cell_vertices
    work_bound = 256 * (face_entries + corner_count + ne + nc + pair_work)
    scratch = _cut_host_scratch(connectivity, geometry)
    check_family_limits(
        limits, corner_count, output_vertices, 8 * corner_count, scratch, work_bound
    )
    if (
        12 * corner_count > limits.maximum_edges
        or 6 * corner_count > limits.maximum_faces
    ):
        raise failure(
            "cut_closure.entities must respect limits.maximum_edges and limits.maximum_faces; got oversized incidence",
            resource=True,
        )
    data_bound = 8 * (
        3 * output_vertices + 8 * corner_count + 3 * supports_count + 4 * output_vertices
    )
    if data_bound > limits.maximum_data_bytes:
        raise failure(
            "cut_closure.data_bytes must respect limits.maximum_data_bytes; got oversized publication banks",
            resource=True,
        )
    return _CutAdmission(
        connectivity, face_ids, corner_count, output_vertices, supports_count, work_bound
    )


def _refuse_immutable_cut_faces(
    connectivity: PolyhedralConnectivity,
    face_ids: np.ndarray,
    immutable_face_ids: tuple[int, ...],
) -> None:
    """Refuse unchanged faces before any coordinate preparation or subdivision."""
    unknown = set(immutable_face_ids).difference(face_ids.tolist())
    if unknown:
        raise ValueError(
            f"cut_preparation.immutable_face_ids must name declared source faces; got {sorted(unknown)}"
        )
    offsets = np.asarray(jax.device_get(connectivity.face_vertex_offsets), dtype=np.int64)
    nonquad = tuple(
        (int(identifier), int(offsets[row + 1] - offsets[row]))
        for row, identifier in enumerate(face_ids)
        if identifier in immutable_face_ids and offsets[row + 1] - offsets[row] != 4
    )
    if nonquad:
        raise failure(
            f"cut_preparation.immutable_face_ids must name quadrilateral faces for unchanged pure-hex closure; got {nonquad}"
        )
    raise failure(
        f"cut_preparation.immutable_face_ids must permit face subdivision for the selected corner template; got unchanged faces {immutable_face_ids}"
    )


def _prepare_cut_source(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    vertex_count: int,
    *,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> _CutSource:
    with coordinate_phase(scope=scope, budget=budget):
        match geometry:
            case ExactPlcCellGeometryConvexSource():
                _, original_geometry, target_mesh = geometry.owners()
                if target_mesh.topology_id != mesh.topology_id:
                    raise ValueError(
                        "geometry.target_mesh must match mesh.topology_id; got another source topology"
                    )
                source = geometry.prepare(mesh.coordinates, budget=budget).vertices
                source_binding = geometry.source_id
            case CellGeometrySpec():
                source = geometry.rational_source_coordinates(budget=budget)
                original_geometry = geometry
                source_binding = cell_geometry_id(geometry)
            case _:
                raise TypeError(
                    f"cut_preparation.geometry must be a supported cut coordinate source; got {type(geometry).__name__}"
                )
        if len(source) != vertex_count or any(len(point) != 3 for point in source):
            raise ValueError(
                "geometry.vertices must bind every three-coordinate source vertex; got incompatible coefficients"
            )
        coordinates = np.asarray(jax.device_get(mesh.coordinates))
        if any(
            not np.array_equal(rounded_point(point), coordinates[row])
            for row, point in enumerate(source)
        ):
            raise ValueError(
                "mesh.coordinates must equal the rounded source coefficients; got conflicting carrier points"
            )
    return _CutSource(source, original_geometry, source_binding)


def _prepare_cut_incidence(mesh: CellMesh, admission: _CutAdmission) -> _CutIncidence:
    connectivity = admission.connectivity
    banks = jax.device_get(
        (
            connectivity.edges,
            connectivity.face_vertex_offsets,
            connectivity.face_vertex_values,
            connectivity.cell_face_offsets,
            connectivity.cell_face_values,
            connectivity.cell_vertex_offsets,
            connectivity.cell_vertex_values,
            mesh.entity_set(3).entity_ids,
            mesh.vertex_global_ids,
            mesh.entity_set(1).entity_ids,
        )
    )
    edges, fo, fv, co, cf, vo, vv, cell_ids, vertex_ids_, edge_ids = (
        np.asarray(bank, dtype=np.int64) for bank in banks
    )
    supports: list[tuple[int, ...]] = [(row,) for row in range(connectivity.vertex_count)]
    supports.extend(tuple(int(v) for v in edge) for edge in edges)
    supports.extend(
        tuple(int(v) for v in fv[fo[row] : fo[row + 1]])
        for row in range(connectivity.face_count)
    )
    supports.extend(
        tuple(int(v) for v in vv[vo[row] : vo[row + 1]])
        for row in range(connectivity.cell_count)
    )
    return _CutIncidence(
        edges, fo, fv, co, cf, vo, vv, cell_ids, vertex_ids_, edge_ids, tuple(supports)
    )


def _certify_cut_face(
    exact: list[tuple[Fraction, ...]],
    loop: list[int],
    vertices: list[int],
    cell_id: int,
    face_id: int,
    charge: Callable[[int], None],
) -> None:
    origin = exact[loop[0]]
    normal = (Fraction(0),) * 3
    for position in range(1, len(loop) - 1):
        charge(30)
        normal = rational_cross(
            _difference(exact[loop[position]], origin),
            _difference(exact[loop[position + 1]], origin),
        )
        if any(normal):
            break
    if not any(normal):
        raise failure(
            f"cut_face.normal must be nonzero; got cell {cell_id}, degenerate face {face_id}"
        )
    side = 0
    for vertex in vertices:
        charge(12)
        distance = sum(
            (
                a * b
                for a, b in zip(normal, _difference(exact[vertex], origin), strict=True)
            ),
            Fraction(0),
        )
        if vertex in loop and distance:
            raise failure(
                f"cut_face.vertices must be rationally planar; got nonplanar face {face_id}"
            )
        sign = (distance > 0) - (distance < 0)
        if sign and side and side != sign:
            raise failure(
                f"cut_cell.faces must bound a convex cell; got cell {cell_id}, conflicting face {face_id}"
            )
        side = sign or side


def _certify_cut_cell_links(
    cell: int,
    incidence: _CutIncidence,
    admission: _CutAdmission,
    exact: list[tuple[Fraction, ...]],
    edge_lookup: dict[tuple[int, int], int],
    charge: Callable[[int], None],
) -> tuple[int, ...]:
    vertices = incidence.cell_vertices[
        incidence.vertex_offsets[cell] : incidence.vertex_offsets[cell + 1]
    ].tolist()
    faces = incidence.cell_faces[
        incidence.cell_offsets[cell] : incidence.cell_offsets[cell + 1]
    ].tolist()
    incident_faces: dict[int, list[int]] = {vertex: [] for vertex in vertices}
    incident_edges: dict[int, set[int]] = {vertex: set() for vertex in vertices}
    pair_faces: dict[tuple[int, int, int], int] = {}
    for face in faces:
        loop = incidence.face_vertices[
            incidence.face_offsets[face] : incidence.face_offsets[face + 1]
        ].tolist()
        _certify_cut_face(
            exact,
            loop,
            vertices,
            int(incidence.cell_ids[cell]),
            int(admission.face_ids[face]),
            charge,
        )
        for position, vertex in enumerate(loop):
            charge(8)
            before, after = loop[position - 1], loop[(position + 1) % len(loop)]
            first = edge_lookup[(min(vertex, before), max(vertex, before))]
            second = edge_lookup[(min(vertex, after), max(vertex, after))]
            incident_edges[vertex].update((first, second))
            incident_faces[vertex].append(face)
            key = (vertex, min(first, second), max(first, second))
            if key in pair_faces:
                raise failure(
                    "cut_link.edge_pairs must occur once; got repeated incident edges"
                )
            pair_faces[key] = face
    signs = []
    for vertex in vertices:
        edge_rows = sorted(incident_edges[vertex])
        face_rows = incident_faces[vertex]
        if len(edge_rows) != 3 or len(face_rows) != 3:
            raise CutCornerLinkError(
                CutCornerLinkIncidence(
                    int(incidence.cell_ids[cell]),
                    int(incidence.vertex_ids[vertex]),
                    tuple(int(incidence.edge_ids[row]) for row in edge_rows),
                    tuple(int(admission.face_ids[row]) for row in face_rows),
                )
            )
        directions = []
        for edge in edge_rows:
            a, b = incidence.edges[edge]
            other = int(b if a == vertex else a)
            directions.append(_difference(exact[other], exact[vertex]))
        charge(40)
        determinant = determinant_exact(directions)
        if not determinant:
            raise failure(
                f"cut_corner.directions must have rank three; got cell {incidence.cell_ids[cell]}, vertex {incidence.vertex_ids[vertex]}"
            )
        signs.append(1 if determinant > 0 else -1)
        for first, second in (
            (edge_rows[0], edge_rows[1]),
            (edge_rows[0], edge_rows[2]),
            (edge_rows[1], edge_rows[2]),
        ):
            if (vertex, min(first, second), max(first, second)) not in pair_faces:
                raise failure(
                    "cut_link.edge_pairs must form a closed three-cycle; got missing pairs"
                )
    return tuple(signs)


def _execute_cut_corner_topology(
    admission: _CutAdmission,
    source: _CutSource,
    incidence: _CutIncidence,
    limits: MeshingLimits,
    *,
    scope: NativeExecutionScope,
) -> _CutTopology:
    exact = list(source.vertices)
    work = 0
    host_ticket = scope.batch(
        admission.work_bound,
        bytes_upper=min(
            limits.maximum_scratch_bytes, scope.remaining().remaining_managed_bytes
        ),
    )
    with host_ticket:

        def charge(amount: int) -> None:
            nonlocal work
            if work + amount > limits.maximum_work_units:
                raise failure(
                    "cut_preparation.work must respect limits.maximum_work_units; got exhausted work",
                    resource=True,
                )
            host_ticket.consume(amount)
            work += amount

        for support in incidence.supports[admission.connectivity.vertex_count :]:
            charge(6 * len(support))
            exact.append(
                (
                    sum((source.vertices[v][0] for v in support), Fraction())
                    / len(support),
                    sum((source.vertices[v][1] for v in support), Fraction())
                    / len(support),
                    sum((source.vertices[v][2] for v in support), Fraction())
                    / len(support),
                )
            )
        edge_lookup: dict[tuple[int, int], int] = {}
        for row, edge in enumerate(incidence.edges):
            first, second = int(edge[0]), int(edge[1])
            edge_lookup[(min(first, second), max(first, second))] = row
        # Allocation belongs to the admitted ticket, not the unreserved root pool.
        with host_ticket.native_target():
            signs = family_host_array((admission.corner_count,), np.int8, scope=scope)
        output = 0
        for cell in range(admission.connectivity.cell_count):
            local_signs = _certify_cut_cell_links(
                cell, incidence, admission, exact, edge_lookup, charge
            )
            signs[output : output + len(local_signs)] = local_signs
            output += len(local_signs)
        remaining = limits.maximum_work_units - work
        if remaining <= 0:
            raise failure(
                "cut_topology.work must respect limits.maximum_work_units; got exhausted work",
                resource=True,
            )
        with host_ticket.native_target():
            hexes, parents, native_work = polyhedron_corner_hexes(
                incidence.edges,
                incidence.face_offsets,
                incidence.face_vertices,
                incidence.cell_offsets,
                incidence.cell_faces,
                incidence.vertex_offsets,
                incidence.cell_vertices,
                signs,
                admission.connectivity.vertex_count,
                maximum_cells=limits.maximum_cells,
                maximum_scratch_bytes=limits.maximum_scratch_bytes,
                maximum_work_units=remaining,
            )
        work += native_work
    return _CutTopology(tuple(exact), signs, hexes, parents, work)


def _assemble_cut_corner_preparation(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    admission: _CutAdmission,
    source: _CutSource,
    incidence: _CutIncidence,
    topology: _CutTopology,
    *,
    scope: NativeExecutionScope,
) -> CutCornerHexPreparation:
    offsets = family_host_array((admission.output_vertices + 1,), np.int64, scope=scope)
    values = family_host_array((admission.supports_count,), np.int64, scope=scope)
    denominators = family_host_array((admission.output_vertices,), np.int64, scope=scope)
    dimensions = family_host_array((admission.output_vertices,), np.int8, scope=scope)
    rows = family_host_array((admission.output_vertices,), np.int64, scope=scope)
    offsets[0] = 0
    cursor = 0
    for row, support in enumerate(incidence.supports):
        values[cursor : cursor + len(support)] = support
        cursor += len(support)
        offsets[row + 1] = cursor
        denominators[row] = len(support)
    nv, ne, nf, nc = (
        admission.connectivity.vertex_count,
        admission.connectivity.edge_count,
        admission.connectivity.face_count,
        admission.connectivity.cell_count,
    )
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
            "source_geometry": source.binding,
            "supports": incidence.supports,
            "hexes": tuple(tuple(int(v) for v in row) for row in topology.hexes),
        }
    )
    for bank in (
        offsets,
        values,
        denominators,
        dimensions,
        rows,
        topology.hexes,
        topology.parents,
        topology.signs,
    ):
        bank.flags.writeable = False
    return CutCornerHexPreparation(
        mesh,
        source.geometry,
        geometry,
        topology.vertices,
        offsets,
        values,
        denominators,
        dimensions,
        rows,
        topology.hexes,
        topology.parents,
        topology.signs,
        topology.work,
        identity,
    )


def _prepare_cut_corner_hexes(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    limits: MeshingLimits,
    /,
    *,
    immutable_face_ids: tuple[int, ...] = (),
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> CutCornerHexPreparation:
    """Prepare face-conforming corner hexes; unchanged source faces are refused."""
    admission = _admit_cut_corner_closure(mesh, geometry, limits, immutable_face_ids)
    source = _prepare_cut_source(
        mesh, geometry, admission.connectivity.vertex_count, scope=scope, budget=budget
    )
    incidence = _prepare_cut_incidence(mesh, admission)
    topology = _execute_cut_corner_topology(
        admission, source, incidence, limits, scope=scope
    )
    return _assemble_cut_corner_preparation(
        mesh, geometry, admission, source, incidence, topology, scope=scope
    )


class CutCornerHexConstruction(StrictModule):
    preparation: CutCornerHexPreparation
    mesh: CellMesh
    geometry: CellGeometrySpec
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    scaled_jacobian_lower: np.ndarray
    mean_ratio_lower: np.ndarray
    aspect_ratio_upper: np.ndarray


@checked
def compose_original_plc_supports(
    prepared: CutCornerHexPreparation,
    /,
    *,
    scope: NativeExecutionScope,
) -> tuple[np.ndarray, np.ndarray, tuple[Fraction, ...]]:
    """Compose cut barycenters into PLC source-corner coefficients."""
    authority = prepared.coordinate_source
    if not isinstance(authority, ExactPlcCellGeometryConvexSource):
        raise TypeError(
            "prepared.coordinate_source must be ExactPlcCellGeometryConvexSource; got another source kind"
        )
    parent_offsets = np.asarray(authority.support_offsets, dtype=np.int64)
    parent_vertices = np.asarray(authority.support_vertices, dtype=np.int64)
    parent_coefficients = authority.support_coefficients
    count = len(prepared.exact_vertices)
    offsets = family_host_array((count + 1,), np.int64, scope=scope)
    vertices = family_host_array((4 * count,), np.int64, scope=scope)
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
                "cut_barycenter.support must be positive normalized P1 weights on at most four corners; got incompatible weights"
            )
        for original, coefficient in sorted(weights.items()):
            vertices[cursor] = original
            coefficients.append(coefficient)
            cursor += 1
        offsets[row + 1] = cursor
    return offsets, vertices[:cursor], tuple(coefficients)


@checked
def realize_cut_corner_hexes(
    prepared: CutCornerHexPreparation,
    limits: MeshingLimits,
    /,
    *,
    certificate_limits: MeshCertificateLimits,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> CutCornerHexConstruction:
    """Realize corner maps and certify positivity, embedding, and map quality.

    Domain coverage is certified against the retained source partition separately.
    """

    authority = prepared.coordinate_source
    offsets = prepared.support_offsets
    count = len(prepared.exact_vertices)
    width = int(np.max(np.diff(offsets)))
    scratch = count * width * (8 + 256) + count * 24
    check_family_limits(
        limits,
        prepared.hexes.shape[0],
        count,
        prepared.hexes.size,
        scratch,
        16 * count * width,
    )
    coordinates = family_host_array((count, 3), np.float64, scope=scope)
    for row, point in enumerate(prepared.exact_vertices):
        coordinates[row] = rounded_point(point)
    added = count - prepared.source_mesh.coordinates.shape[0]
    mesh = CellMesh(
        coordinates,
        (CellBlock("cut_hexes", "hexahedron", prepared.hexes),),
        vertex_global_ids=vertex_ids(prepared.source_mesh, added),
        numeric_version=prepared.source_mesh.numeric_version,
    )
    with coordinate_phase(scope=scope, budget=budget):
        if isinstance(authority, ExactPlcCellGeometryConvexSource):
            (
                parent_mesh,
                parent_geometry,
                _,
            ) = authority.owners()
            original_offsets, original_vertices, original_coefficients = (
                compose_original_plc_supports(prepared, scope=scope)
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
                budget=budget,
            )
            geometry = CellGeometrySpec.plc(mesh, source, budget=budget)
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
                    "coordinate_source.exact_source must be a supported power cut source; got another source kind"
                )
            parents = family_host_array((count, width), np.int64, scope=scope)
            parents.fill(-1)
            coefficients = []
            for row in range(count):
                support = prepared.support_vertices[offsets[row] : offsets[row + 1]]
                parents[row, : support.size] = support
                coefficient = Fraction(1, int(prepared.support_denominators[row]))
                coefficients.append(
                    (coefficient,) * support.size
                    + (Fraction(0),) * (width - support.size)
                )
            actions = family_host_array((count, 0), np.int64, scope=scope)
            source = ExactPowerCellGeometryLinearActionSource(
                parent, parents, tuple(coefficients), actions, periodic_preparation=None
            )
            geometry = CellGeometrySpec.power(mesh, source, budget=budget)
    with coordinate_phase(scope=scope, budget=budget):
        validity = certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
    if validity.invalid_count or validity.unresolved_count:
        raise failure(
            "cut_geometry.validity must certify every whole-cell Jacobian; got invalid or unresolved cells"
        )
    with coordinate_phase(scope=scope, budget=budget):
        embedding = certify_global_embedding(
            mesh, geometry, validity, limits=certificate_limits, budget=budget
        )
    if embedding.status != "certified":
        codes = tuple(finding.check for finding in embedding.findings)
        raise failure(f"cut_embedding.status must be certified; got findings {codes}")

    scaled, mean, aspect = exact_mapped_volume_quality(
        mesh, geometry, scope=scope, budget=budget
    )
    for bank in (scaled, mean, aspect):
        bank.flags.writeable = False
    return CutCornerHexConstruction(
        prepared, mesh, geometry, validity, embedding, scaled, mean, aspect
    )


@checked
def prepare_cut_corner_hexes(
    mesh: CellMesh,
    geometry: CellGeometrySpec | ExactPlcCellGeometryConvexSource,
    limits: MeshingLimits,
    /,
    *,
    immutable_face_ids: tuple[int, ...] = (),
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
    workspace: NativeHostStorageWorkspace,
) -> CutCornerHexPreparation:
    """Prepare source corner closure inside an admitted host-storage workspace.

    This template subdivides every face. A nonempty ``immutable_face_ids``
    therefore refuses: nonquadrilateral faces cannot remain pure-hex faces,
    and quadrilateral faces require a boundary-preserving template.
    Undeclared scientific IDs raise ``ValueError``. Refusal preserves the source
    mesh and restores the caller's workspace bound.
    """
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise TypeError(
            "mesh.connectivity must be PolyhedralConnectivity; got another connectivity"
        )
    scratch = _cut_host_scratch(mesh.connectivity, geometry)
    if scratch > limits.maximum_scratch_bytes:
        raise failure(
            "cut_preparation.scratch_bytes must respect limits.maximum_scratch_bytes; got oversized host storage",
            resource=True,
        )
    starting = workspace.bound
    workspace.set_bound(starting + scratch)
    try:
        prepared = _prepare_cut_corner_hexes(
            mesh,
            geometry,
            limits,
            immutable_face_ids=immutable_face_ids,
            scope=scope,
            budget=budget,
        )
    finally:
        workspace.set_bound(starting)
    workspace.retain_owner(prepared)
    return prepared


@checked
def certify_cut_corner_coverage(
    construction: CutCornerHexConstruction,
    original_domain: PiecewiseLinearDomain | MappedReferenceDomain,
    source_cell_regions: np.ndarray,
    /,
    *,
    certificate_limits: MeshCertificateLimits,
    scope: NativeExecutionScope,
    budget: CoordinateEnclosureBudget,
) -> tuple[DomainCoverageCertificate, np.ndarray]:
    """Certify hex maps against the independently retained source domain."""

    regions = np.asarray(source_cell_regions, dtype=np.int64)
    connectivity = construction.preparation.source_mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError(
            "construction.source_mesh.connectivity must be polyhedral; got another connectivity"
        )
    source_count = connectivity.cell_count
    if regions.shape != (source_count,):
        raise ValueError(
            f"cut_coverage.source_cell_regions must name every cut source cell once; got shape {regions.shape}"
        )
    target_regions = family_host_array(
        construction.preparation.parent_cells.shape, np.int64, scope=scope
    )
    target_regions[:] = regions[construction.preparation.parent_cells]
    with coordinate_phase(scope=scope, budget=budget):
        coverage = certify_domain_coverage(
            construction.mesh,
            construction.geometry,
            original_domain,
            target_regions,
            embedding=construction.embedding,
            limits=certificate_limits,
            budget=budget,
        )
    if coverage.status != "certified":
        checks = tuple(finding.check for finding in coverage.findings)
        raise failure(f"cut_coverage.status must be certified; got findings {checks}")
    target_regions.flags.writeable = False
    return coverage, target_regions


@checked
def require_cut_corner_quality(
    construction: CutCornerHexConstruction,
    schedule: NativeHexGridSchedule,
) -> None:
    """Enforce the selected portfolio's whole-map quality requirements."""

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
            raise failure(
                f"cut_quality.{name} must respect bound {bound}; got cell values {evidence}"
            )
