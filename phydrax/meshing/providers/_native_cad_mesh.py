"""Native CAD solid route: one ``cad_mesh_solid`` call to the candidate.

CAD volume plans construct here at every proof level: curves, faces and the
fixed-boundary volume (geometry order 1 or 2) come from one native call and
become the candidate mesh as exported. ``prove()`` certifies that candidate
with the native verifiers (:mod:`phydrax.meshing._native_cad_certificate`)
on the boundary chart realization the call exported.

The native stage already adopts its PLC recovery as a tetrahedral mesh,
refines and improves it, so the exported final arrays become a
:class:`CellMesh` directly (as the PLC route's ``_make_construction`` does)
rather than through ``TetMesh3D.from_recovery``.

The candidate's compliance reports no achieved source deviation: the bound is
V3's, in the certificate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._meshcore import cad_mesh_solid, CadSolidResult, MeshcoreStatus
from ...discretization._cell_geometry import CellGeometrySpec
from ...discretization._cell_mesh import CellMesh
from ...discretization._reference_cell import reference_cell_topology
from ...discretization._topology import sorted_unique_rows
from ...discretization.fem._reference import lagrange_element
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import MeshingFailure, MeshingFailureCategory
from .._native_cad_certificate import NativeCadSurfaceWitness
from .._native_construction import NativeCertificationRequest, NativeMeshConstruction
from .._result import MeshingComplianceReport
from .._sizing import SizeControlStrength, UniformSizeControl
from ._native_cad_bundle import cad_bundle_support_issues, cad_geometry_bundle


if TYPE_CHECKING:
    from ._native import _PreparedDirectBRepCadVolume


# Native cell edge order (phx_mc_cad_quadratic_nodes): 01 02 03 12 13 23.
_NATIVE_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_BLOCK = "native-cad-tetrahedra"


def cad_solid_support_issues(
    prepared: _PreparedDirectBRepCadVolume, /
) -> tuple[str, ...]:
    """Why a prepared CAD volume cannot take the native route."""
    issues: list[str] = []
    specification = prepared.specification
    if specification.layer_controls:
        issues.append("layer controls take the layer/core route")
    # The native candidate carries no organization records: patch and region
    # controls are honoured by the existing route only.
    if specification.patch_controls or specification.region_controls:
        issues.append("patch and region controls need source organization records")
    # The native call has no seeded region classification, and one deviation
    # on every face and curve: seeds and protected features (their own
    # deviation bounds and feature evidence) are honoured by the existing route.
    if specification.region_seeds or specification.hole_seeds:
        issues.append("region and hole seeds need seeded region classification")
    if specification.protected_features:
        issues.append("protected features need their own deviation and evidence")
    if prepared.boundary_request.selected_patches != tuple(
        range(len(prepared.domain.patches))
    ):
        issues.append("the boundary request does not size every source face")
    issues.extend(cad_bundle_support_issues(prepared.domain))
    return tuple(issues)


def _quadratic_geometry(mesh: CellMesh, solid: CadSolidResult, /) -> CellGeometrySpec:
    """P2 map of the native edge nodes: vertex DOFs name points, edge DOFs nodes."""
    volume = solid.volume
    element = lagrange_element(mesh.blocks[0].cell_kind, 2)
    vertices = np.asarray(
        reference_cell_topology(mesh.blocks[0].cell_kind).vertices, dtype=np.float64
    )
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    point_count = volume.points.shape[0]
    cells = np.asarray(volume.tetrahedra, dtype=np.int64)
    cell_edges = np.asarray(volume.cell_edges, dtype=np.int64)
    route = np.empty((cells.shape[0], nodes.shape[0]), dtype=np.int32)
    for dof, node in enumerate(nodes):
        corner = np.flatnonzero(np.all(vertices == node, axis=1))
        if corner.size:
            route[:, dof] = cells[:, int(corner[0])]
            continue
        slot = next(
            row
            for row, (a, b) in enumerate(_NATIVE_EDGES)
            if np.array_equal(0.5 * (vertices[a] + vertices[b]), node)
        )
        route[:, dof] = point_count + cell_edges[:, slot]
    coordinates = np.concatenate(
        (np.asarray(volume.points, dtype=np.float64), volume.nodes), axis=0
    )
    return CellGeometrySpec({_BLOCK: element}, {_BLOCK: route}, coordinates)


def _size_compliance(
    prepared: _PreparedDirectBRepCadVolume,
    solid: CadSolidResult,
    issues: list[str],
    requested: list[tuple[str, float]],
    achieved: list[tuple[str, float]],
    /,
) -> None:
    """Boundary edge-length statistics of each face size control (PLC route's rows)."""
    from ._native_cad_volume import _domain_scope_rows

    specification = prepared.specification
    policy = specification.size_compliance
    surface = solid.surface
    points = np.asarray(solid.volume.points, dtype=np.float64)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    patches = np.asarray(surface.triangle_patches, dtype=np.int64)
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append(f"size_control:{control.control_id}:unsupported")
            continue
        rows = _domain_scope_rows(prepared.domain, control.scope, 2)
        faces = triangles[np.isin(patches, rows)]
        if faces.shape[0] == 0:
            issues.append(f"size_control:{control.control_id}:no_boundary_faces")
            continue
        edges = sorted_unique_rows(
            np.sort(faces[:, ((0, 1), (1, 2), (2, 0))], axis=2).reshape((-1, 2))
        )
        lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
        for statistic in policy.target_statistics:
            value = float(np.quantile(lengths, 0.5 if statistic == "p50" else 0.95))
            key = f"size:{control.control_id}:{statistic}"
            requested.append((key, control.target_size))
            achieved.append((key, value))
            allowed = (
                control.target_size * (1.0 + policy.relative_tolerance)
                + policy.absolute_tolerance
            )
            if control.strength is SizeControlStrength.HARD and value > allowed:
                issues.append(key)
        if control.maximum_size is not None:
            maximum = float(np.max(lengths))
            key = f"size:{control.control_id}:maximum"
            requested.append((key, control.maximum_size))
            achieved.append((key, maximum))
            if (
                control.strength is SizeControlStrength.HARD
                and maximum > control.maximum_size
            ):
                issues.append(key)


def construct_cad_solid(prepared: _PreparedDirectBRepCadVolume, /) -> CadSolidResult:
    """One native ``cad_mesh_solid`` call for the prepared CAD volume.

    Face sizes are the boundary preparation's per-face sizes, the source
    tolerance is the request's ``maximum_deviation`` on every face and curve,
    and the volume targets are the route's volume schedule. A native refusal
    (UNRESOLVED without cells) raises; unmet quality targets are compliance
    issues.
    """
    domain = prepared.domain
    specification = prepared.specification
    schedule = prepared.options.volume_schedule
    deviation = specification.maximum_deviation
    boundary = prepared.boundary_request
    if schedule is None or deviation is None or cad_solid_support_issues(prepared):
        raise ValueError("The prepared CAD volume has no native unproved route.")
    domain.require_current(domain.source_id, domain.source_revision)
    sizes = np.asarray(boundary.size_values, dtype=np.float64)
    model = cad_geometry_bundle(domain).model
    order = specification.target.geometry_order
    return cad_mesh_solid(
        model,
        sizes,
        np.full(sizes.shape, deviation, dtype=np.float64),
        np.full((model.curve_nodes.size,), deviation, dtype=np.float64),
        radius_edge_bound=(
            np.inf if schedule.radius_edge_bound is None else schedule.radius_edge_bound
        ),
        min_dihedral_degrees=schedule.minimum_dihedral_degrees,
        min_relative_determinant=schedule.minimum_relative_determinant,
        geometry_order=order,
    )


def execute_cad_solid(
    prepared: _PreparedDirectBRepCadVolume, /
) -> NativeMeshConstruction:
    """Construct the CAD solid natively; ``prove()`` certifies the candidate."""
    domain = prepared.domain
    specification = prepared.specification
    schedule = prepared.options.volume_schedule
    deviation = specification.maximum_deviation
    boundary = prepared.boundary_request
    assert schedule is not None and deviation is not None
    order = specification.target.geometry_order
    solid = construct_cad_solid(prepared)
    volume = solid.volume
    stats = volume.stats[0]
    if volume.tetrahedra.shape[0] == 0:
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native CAD volume recovery refused the surface after its boundary "
            f"split (status {solid.status.name}, recovery status "
            f"{int(stats['recovery_status'])}, reason "
            f"{int(stats['recovery_reason'])}).",
            provider_code="native_cad_volume_unresolved",
            stage="volume_meshing",
        )
    mesh = CellMesh.from_simplices(
        np.asarray(volume.points, dtype=np.float64),
        np.asarray(volume.tetrahedra, dtype=np.int64),
        dimension=3,
        block_name=_BLOCK,
        numeric_version=domain.source_revision,
    )
    geometry = (
        CellGeometrySpec.affine(mesh) if order == 1 else _quadratic_geometry(mesh, solid)
    )
    issues: list[str] = []
    if solid.status is MeshcoreStatus.UNRESOLVED:
        issues.append(f"native_unmet_quality_cells:{int(stats['unmet_cells'])}")
    requested: list[tuple[str, float]] = [
        ("maximum_deviation", float(deviation)),
        ("minimum_dihedral_degrees", float(schedule.minimum_dihedral_degrees)),
    ]
    if schedule.radius_edge_bound is not None:
        requested.append(("radius_edge_bound", float(schedule.radius_edge_bound)))
    achieved: list[tuple[str, float]] = [
        ("unmet_quality_cells", float(stats["unmet_cells"]))
    ]
    _size_compliance(prepared, solid, issues, requested, achieved)
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )
    certification = NativeCertificationRequest(
        MeshCertificationSchedule("volume_cad"),
        domain.source_id,
        domain.source_revision,
        domain=domain,
        cell_regions=np.asarray(volume.tetrahedron_regions, dtype=np.int64),
        # The boundary preparation's query selects every source face (support).
        fidelity_source=boundary.boundary_source,
        fidelity_tolerance=float(deviation),
        protected_features=specification.protected_features,
        feature_source=domain,
        native_cad_surface=NativeCadSurfaceWitness(
            mesh, solid.surface, len(cad_geometry_bundle(domain).model.loop_surfaces)
        ),
    )
    return NativeMeshConstruction(
        mesh,
        prepared.coordinate_contract,
        compliance,
        certification,
        geometry=geometry,
    )


__all__ = [
    "cad_solid_support_issues",
    "construct_cad_solid",
    "execute_cad_solid",
]
