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

Patch and region controls and region/hole seeds are honored on the native
candidate: boundary triangles group by their native source face and zones by
tetrahedron region through the PLC route's organization substrate
(:func:`~._native_volume.source_organization`), and seeds are classified by an
exact point-in-shell test against the candidate surface. Multiple solids
require the exact planar regional source proof. Protected features, layer
controls, and unsupported sources keep the existing route
(:func:`cad_solid_support_issues`).

The candidate's compliance reports no achieved source deviation: the bound is
V3's, in the certificate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..._meshcore import (
    cad_geometry_evaluate,
    cad_mesh_solid,
    CadEntityKind,
    CadGeometryQuantity,
    CadModel,
    CadProximityControl,
    CadSolidResult,
    exact_orient3d,
    MeshcoreStatus,
)
from ...discretization._cell_geometry import CellGeometrySpec
from ...discretization._cell_mesh import CellMesh
from ...discretization._reference_cell import reference_cell_topology
from ...discretization._topology import sorted_unique_rows
from ...discretization.fem._reference import (
    cubic_tetrahedron_geometry_element,
    lagrange_element,
)
from .._association import GeometryAssociation
from .._certification_schedule import MeshCertificationSchedule
from .._contracts import MeshingFailure, MeshingFailureCategory, MeshingFailureEvidence
from .._native_cad_certificate import NativeCadSurfaceWitness
from .._native_cad_records import _native_cad_regional_surface, native_cad_records
from .._native_cad_shell import source_shell
from .._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from .._result import MeshingComplianceReport
from .._sizing import (
    CAD_CHART_SURFACE_SIZE,
    ProximitySizeControl,
    size_statistic,
    SizeControlStrength,
    stored_circumradius_measures,
    triangle_circumradii,
    UniformSizeControl,
)
from .._volume_generation import _shape_targets_gated
from ._native_cad_bundle import cad_bundle_support_issues, cad_geometry_bundle
from ._native_cad_volume import (
    _domain_scope_rows,
    _region_row,
    cad_facet_regions,
    cad_patch_control_rows,
    cad_region_seeds,
)
from ._native_volume import source_organization, SourceOrganizationLayout


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
    # The native call has one deviation on every face and curve: protected
    # features (their own deviation bounds and feature evidence) are honored
    # by the existing route.
    if specification.protected_features:
        issues.append("protected features need their own deviation and evidence")
    issues.extend(cad_bundle_support_issues(prepared.domain))
    if len(prepared.domain.regions) > 1 and not issues:
        if specification.target.geometry_order != 1:
            issues.append("regional native CAD proof requires affine tetrahedra")
        else:
            refusal = source_shell(prepared.domain).refusal
            if refusal is not None:
                issues.append(f"regional native CAD source proof refused: {refusal}")
    return tuple(issues)


def _quadratic_geometry(cell_kind: str, solid: CadSolidResult, /) -> CellGeometrySpec:
    """P2 map of the native edge nodes: vertex DOFs name points, edge DOFs nodes."""
    volume = solid.volume
    element = lagrange_element(cell_kind, 2)
    vertices = np.asarray(reference_cell_topology(cell_kind).vertices, dtype=np.float64)
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


def _cubic_geometry(cell_kind: str, solid: CadSolidResult, /) -> CellGeometrySpec:
    volume = solid.volume
    extra = volume.cubic_cell_nodes
    count = volume.points.shape[0]
    additional = volume.cubic_nodes.shape[0]
    if (
        additional == 0
        or count + additional > np.iinfo(np.int32).max
        or extra.shape != (volume.tetrahedra.shape[0], 16)
        or np.any(extra < 0)
        or np.any(extra >= additional)
    ):
        raise ValueError("Native cubic CAD coordinates have an incompatible node bank.")
    route = np.empty((volume.tetrahedra.shape[0], 20), dtype=np.int32)
    route[:, :4] = volume.tetrahedra
    route[:, 4:] = count + extra
    return CellGeometrySpec(
        {_BLOCK: cubic_tetrahedron_geometry_element(cell_kind)},
        {_BLOCK: route},
        np.concatenate((volume.points, volume.cubic_nodes)),
    )


def _size_compliance(
    prepared: _PreparedDirectBRepCadVolume,
    solid: CadSolidResult,
    issues: list[str],
    requested: list[tuple[str, float]],
    achieved: list[tuple[str, float]],
    /,
) -> None:
    """Size statistics of each face size control on its boundary triangles.

    The target statistics judge the circumradius the CAD face stage bounds
    on the stored coordinates (``CAD_CHART_SURFACE_SIZE``: R <= h / sqrt(2)
    up to the rounding allowance); ``maximum_size`` bounds the longest
    boundary edge.
    """
    from ._native_cad_volume import _domain_scope_rows

    specification = prepared.specification
    policy = specification.size_compliance
    surface = solid.surface
    points = np.asarray(solid.volume.points, dtype=np.float64)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    patches = np.asarray(surface.triangle_patches, dtype=np.int64)
    declaration = CAD_CHART_SURFACE_SIZE
    for control in specification.size_controls:
        # Proximity controls report their own rows (_proximity_compliance).
        if isinstance(control, ProximitySizeControl):
            continue
        if not isinstance(control, UniformSizeControl):
            issues.append(f"size_control:{control.control_id}:unsupported")
            continue
        rows = _domain_scope_rows(prepared.domain, control.scope, 2)
        faces = triangles[np.isin(patches, rows)]
        if faces.shape[0] == 0:
            issues.append(f"size_control:{control.control_id}:no_boundary_faces")
            continue
        radii = triangle_circumradii(points, faces)
        measures = stored_circumradius_measures(points, faces)
        for statistic in policy.target_statistics:
            value = size_statistic(radii, statistic)
            key = f"size:{control.control_id}:{statistic}_circumradius"
            requested.append((key, declaration.bound_factor * control.target_size))
            achieved.append((key, value))
            if control.strength is SizeControlStrength.HARD and declaration.exceeds(
                measures, statistic, control.target_size, policy
            ):
                issues.append(key)
        if control.maximum_size is not None:
            edges = sorted_unique_rows(
                np.sort(faces[:, ((0, 1), (1, 2), (2, 0))], axis=2).reshape((-1, 2))
            )
            lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
            maximum = float(np.max(lengths))
            key = f"size:{control.control_id}:maximum"
            requested.append((key, control.maximum_size))
            achieved.append((key, maximum))
            if (
                control.strength is SizeControlStrength.HARD
                and maximum > control.maximum_size
            ):
                issues.append(key)


def _triangle_distances(
    queries: np.ndarray, corners: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each query to the nearest triangle and that triangle's row.

    The closest point of each triangle by its Voronoi region (Ericson,
    Real-Time Collision Detection 5.1.5), queries in blocks.
    """
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    ab, ac, bc = b - a, c - a, c - b
    distance = np.empty((queries.shape[0],), dtype=np.float64)
    nearest = np.empty((queries.shape[0],), dtype=np.int64)
    block = max(1, 2**22 // max(1, corners.shape[0]))
    for start in range(0, queries.shape[0], block):
        p = queries[start : start + block, None, :]
        ap, bp, cp = p - a, p - b, p - c
        d1, d2 = np.sum(ab * ap, axis=2), np.sum(ac * ap, axis=2)
        d3, d4 = np.sum(ab * bp, axis=2), np.sum(ac * bp, axis=2)
        d5, d6 = np.sum(ab * cp, axis=2), np.sum(ac * cp, axis=2)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            total = va + vb + vc
            v, w = vb / total, vc / total
            closest = a + v[..., None] * ab + w[..., None] * ac
            edge_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            closest = np.where(
                ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None],
                b + edge_bc[..., None] * bc,
                closest,
            )
            closest = np.where(
                ((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
                a + (d2 / (d2 - d6))[..., None] * ac,
                closest,
            )
            closest = np.where(
                ((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
                a + (d1 / (d1 - d3))[..., None] * ab,
                closest,
            )
        closest = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, closest)
        closest = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, closest)
        closest = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, closest)
        lengths = np.linalg.norm(p - closest, axis=2)
        rows = np.argmin(lengths, axis=1)
        nearest[start : start + block] = rows
        distance[start : start + block] = lengths[np.arange(rows.size), rows]
    return distance, nearest


def _triangle_normals(points: np.ndarray, triangles: np.ndarray, /) -> np.ndarray:
    corners = points[triangles]
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def _proxy_deviations(
    model: CadModel, solid: CadSolidResult, tolerance: float, /
) -> np.ndarray:
    """Each boundary triangle's deviation bound from its face.

    The sag of its circumcircle (radius R) on the osculating sphere of the
    largest principal curvature kappa at its defined corners,
    kappa R^2 / (1 + sqrt(1 - kappa^2 R^2)), at most the tolerance
    (cad_proximity.hpp, proxy_deviation): the native field's own bound.
    """
    surface = solid.surface
    points = np.asarray(solid.volume.points, dtype=np.float64)
    triangles = np.asarray(surface.triangles, dtype=np.int64)
    patches = np.asarray(surface.triangle_patches, dtype=np.int64)
    uv = np.asarray(surface.triangle_uv, dtype=np.float64).reshape((-1, 3, 2))
    kappa = np.zeros((triangles.shape[0],), dtype=np.float64)
    defined = np.zeros((triangles.shape[0],), dtype=bool)
    for face in np.unique(patches):
        rows = np.flatnonzero(patches == face)
        corners = uv[rows].reshape((-1, 2))

        def evaluate(parameters: np.ndarray, face: int = int(face)) -> np.ndarray:
            return cad_geometry_evaluate(
                model,
                CadEntityKind.SURFACE,
                face,
                parameters,
                CadGeometryQuantity.DIFFERENTIAL,
            )

        # Model corners (dimension 0, poles among them) are evaluated one by
        # one: one the evaluation refuses (a pole's chart edge) is undefined,
        # like a non-finite curvature.
        single = np.asarray(surface.vertex_dimensions)[triangles[rows]].reshape(-1) == 0
        differential = np.full((corners.shape[0], 8), np.nan)
        if (~single).any():
            differential[~single] = evaluate(corners[~single])
        for row in np.flatnonzero(single):
            try:
                differential[row] = evaluate(corners[row : row + 1])[0]
            except ValueError:
                continue
        curvature = np.abs(differential[:, 6:8]).max(axis=1).reshape((-1, 3))
        finite = np.isfinite(curvature)
        kappa[rows] = np.where(finite, curvature, 0.0).max(axis=1)
        defined[rows] = finite.any(axis=1)
    radius = triangle_circumradii(points, triangles)
    kr = kappa * radius
    with np.errstate(invalid="ignore"):
        sag = kappa * radius**2 / (1.0 + np.sqrt(1.0 - kr**2))
    return np.where(defined & (kr < 1.0), np.minimum(sag, tolerance), tolerance)


def _face_corners(model: CadModel, rows: np.ndarray, /) -> set[int]:
    """The model corners on the given faces (curve ends and poles)."""
    corners: set[int] = set()
    offsets = np.asarray(model.loop_use_offsets, dtype=np.int64)
    for loop, face in enumerate(np.asarray(model.loop_surfaces, dtype=np.int64)):
        if face not in rows:
            continue
        for use in range(int(offsets[loop]), int(offsets[loop + 1])):
            entity = int(model.use_entities[use])
            if int(model.use_kinds[use]) == 0:
                corners.update(int(x) for x in model.curve_corners[entity])
            else:
                corners.add(entity)
    return corners


def _proximity_controls(
    prepared: _PreparedDirectBRepCadVolume, model: CadModel, /
) -> tuple[tuple[ProximitySizeControl, CadProximityControl], ...]:
    """The specification's proximity controls over model face rows.

    Scopes that touch (share a corner or curve) have a gap reaching zero, so
    a control on them must declare ``minimum_size`` (Gmsh refuses touching
    scopes outright).
    """
    from ._native_cad_volume import _domain_scope_rows

    result = []
    for control in prepared.specification.size_controls:
        if not isinstance(control, ProximitySizeControl):
            continue
        source = _domain_scope_rows(prepared.domain, control.source_scope, 2)
        target = _domain_scope_rows(prepared.domain, control.target_scope, 2)
        if control.minimum_size is None and _face_corners(model, source) & _face_corners(
            model, target
        ):
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SPECIFICATION,
                "ProximitySizeControl scopes touch, so their gap reaches zero; "
                "declare minimum_size for the native CAD solid route.",
                stage="size_field_resolution",
            )
        result.append(
            (
                control,
                CadProximityControl(
                    tuple(int(x) for x in source),
                    tuple(int(x) for x in target),
                    control.elements_per_gap,
                    control.minimum_size,
                    control.maximum_size,
                    control.opposite_normals_only,
                ),
            )
        )
    return tuple(result)


def _proximity_compliance(
    solid: CadSolidResult,
    controls: tuple[tuple[ProximitySizeControl, CadProximityControl], ...],
    deviations: np.ndarray,
    requested: list[tuple[str, float]],
    achieved: list[tuple[str, float]],
    /,
) -> None:
    """Each proximity control's statement on its scoped boundary triangles.

    The control's size at a triangle is the local gap at its centroid (the
    distance to the other scope's boundary triangles less that triangle's
    deviation bound, the native field's lower bound) over elements_per_gap,
    clamped to its bounds; the CAD face stage's statement is the
    circumradius bound R <= size / sqrt(2) (``CAD_CHART_SURFACE_SIZE``), so a
    triangle's ratio is R over that bound. Rows: the requested
    elements_per_gap and bounds; the smallest gap estimate, the largest ratio
    and the elements per gap it achieves. The gap is an estimate, not a
    certified bound: the sag bound takes the curvature at the proxy
    triangle's corners only, so on a curved scope it can understate the
    deviation. The rows are therefore report-only and raise no issue, even
    for a HARD control.
    """
    factor = CAD_CHART_SURFACE_SIZE.bound_factor
    points = np.asarray(solid.volume.points, dtype=np.float64)
    triangles = np.asarray(solid.surface.triangles, dtype=np.int64)
    patches = np.asarray(solid.surface.triangle_patches, dtype=np.int64)
    normals = _triangle_normals(points, triangles)
    radii = triangle_circumradii(points, triangles)
    for control, native in controls:
        key = f"proximity:{control.control_id}"
        requested.append((f"{key}:elements_per_gap", float(control.elements_per_gap)))
        for name, bound in (
            ("minimum_size", control.minimum_size),
            ("maximum_size", control.maximum_size),
        ):
            if bound is not None:
                requested.append((f"{key}:{name}", bound))
        gaps, ratios = [], []
        for own, other in (
            (native.source_faces, native.target_faces),
            (native.target_faces, native.source_faces),
        ):
            mine = np.flatnonzero(np.isin(patches, own))
            theirs = np.flatnonzero(np.isin(patches, other))
            centroids = points[triangles[mine]].mean(axis=1)
            distance, nearest = _triangle_distances(centroids, points[triangles[theirs]])
            gap = np.maximum(0.0, distance - deviations[theirs[nearest]])
            if control.opposite_normals_only:
                facing = np.sum(normals[mine] * normals[theirs[nearest]], axis=1) < 0.0
                gap = np.where(facing, gap, np.inf)
            size = np.clip(
                gap / control.elements_per_gap,
                0.0 if control.minimum_size is None else control.minimum_size,
                np.inf if control.maximum_size is None else control.maximum_size,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios.append(radii[mine] / (factor * size))
            gaps.append(gap)
        gap = np.concatenate(gaps)
        ratio = np.concatenate(ratios)
        largest = float(np.nanmax(ratio)) if ratio.size else 0.0
        achieved.extend(
            (
                (f"{key}:gap_estimate", float(gap.min())),
                (f"{key}:maximum_circumradius_ratio", largest),
                (
                    f"{key}:minimum_elements_per_gap",
                    control.elements_per_gap / largest if largest > 0 else np.inf,
                ),
            )
        )


def _cell_quality(points: np.ndarray, cells: np.ndarray, /) -> tuple[np.ndarray, ...]:
    """Per cell: the smallest dihedral angle (degrees) and the radius-edge ratio."""
    corners = points[cells]
    smallest = np.full((cells.shape[0],), np.inf)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        k, m = (x for x in range(4) if x not in (i, j))
        e = corners[:, j] - corners[:, i]
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        u = corners[:, k] - corners[:, i]
        v = corners[:, m] - corners[:, i]
        u -= np.sum(u * e, axis=1, keepdims=True) * e
        v -= np.sum(v * e, axis=1, keepdims=True) * e
        cosine = np.sum(u * v, axis=1) / (
            np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        )
        smallest = np.minimum(smallest, np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    edges = corners[:, 1:] - corners[:, :1]
    rhs = 0.5 * np.sum(edges**2, axis=2)
    center = np.linalg.solve(edges, rhs[..., None])[..., 0]
    radius = np.linalg.norm(center, axis=1)
    shortest = np.min(
        np.linalg.norm(
            corners[:, (0, 0, 0, 1, 1, 2)] - corners[:, (1, 2, 3, 2, 3, 3)], axis=2
        ),
        axis=1,
    )
    return smallest, radius / shortest


def _thin_region_findings(
    model: CadModel,
    solid: CadSolidResult,
    sizes: np.ndarray,
    deviations: np.ndarray,
    radius_edge_bound: float | None,
    minimum_dihedral_degrees: float,
    /,
) -> list[str]:
    """Input findings: unmet cells in a region thinner than the targets allow at h.

    A cell in a slab of thickness t that stands on a boundary triangle of
    circumradius rho has its apex within t of the triangle's plane. Its best
    placement, an equilateral triangle with the apex over its circumcenter
    at height t, has radius-edge ratio sqrt(rho^2 + t^2) / (2 t) and
    dihedral atan(2 t / rho) at the base edges; the radius-edge bound B and
    the dihedral target theta then need rho <= t sqrt(4 B^2 - 1) and
    rho <= 2 t / tan(theta), i.e. t >= rho / k with
    k = min(sqrt(4 B^2 - 1), 2 / tan(theta)) (B = 2, theta = 15.5 deg: k =
    sqrt(15)). At the requested size h the face stage's triangles reach
    rho = h / sqrt(2) (``CAD_CHART_SURFACE_SIZE``), so the targets allow a
    thickness t >= h / (sqrt(2) k) at h. An unmet cell's thickness is taken
    through its centroid c: with A the nearest boundary face and B the
    nearest other face whose nearest triangle faces A's, the two faces come
    within d(c, A) + d(c, B) of each other plus the two triangles' deviation
    bounds (_proxy_deviations), an upper bound t+ of the local thickness. A
    face pair is reported when some unmet cell there has t+ below the
    thickness its faces' smaller h allows.
    """
    bound = np.inf if radius_edge_bound is None else float(radius_edge_bound)
    theta = np.radians(float(minimum_dihedral_degrees))
    k = min(
        np.sqrt(4.0 * bound**2 - 1.0),
        2.0 / np.tan(theta) if theta > 0.0 else np.inf,
    )
    if not np.isfinite(k):
        return []
    volume = solid.volume
    points = np.asarray(volume.points, dtype=np.float64)
    cells = np.asarray(volume.tetrahedra, dtype=np.int64)
    dihedral, ratio = _cell_quality(points, cells)
    unmet = (dihedral < float(minimum_dihedral_degrees)) | (ratio > bound)
    if not unmet.any():
        return []
    centroids = points[cells[unmet]].mean(axis=1)
    triangles = np.asarray(solid.surface.triangles, dtype=np.int64)
    patches = np.asarray(solid.surface.triangle_patches, dtype=np.int64)
    normals = _triangle_normals(points, triangles)
    factor = float(CAD_CHART_SURFACE_SIZE.bound_factor)
    faces = np.unique(patches)
    distance = np.empty((centroids.shape[0], faces.size))
    nearest = np.empty((centroids.shape[0], faces.size), dtype=np.int64)
    for column, face in enumerate(faces):
        rows = np.flatnonzero(patches == face)
        found, row = _triangle_distances(centroids, points[triangles[rows]])
        distance[:, column], nearest[:, column] = found, rows[row]
    regions: dict[tuple[int, int], list[float]] = {}
    for cell in range(centroids.shape[0]):
        order = np.argsort(distance[cell], kind="stable")
        first = nearest[cell, order[0]]
        for column in order[1:]:
            other = nearest[cell, column]
            if np.dot(normals[first], normals[other]) >= 0.0:
                continue
            thickness = (
                distance[cell, order[0]]
                + distance[cell, column]
                + deviations[first]
                + deviations[other]
            )
            low, high = sorted((int(faces[order[0]]), int(faces[column])))
            pair = (low, high)
            needed = factor * float(min(sizes[pair[0]], sizes[pair[1]])) / k
            if thickness < needed:
                entry = regions.setdefault(pair, [0.0, 0.0, needed])
                entry[0] += 1.0
                entry[1] = max(entry[1], thickness)
            break
    findings = []
    for (a, b), (count, thickness, required) in sorted(regions.items()):
        findings.append(
            f"input_thin_region_below_volume_targets:faces={a},{b}:"
            f"unmet_cells={int(count)}:thickness_at_most={float(thickness)!r}:"
            f"thickness_needed={float(required)!r}:"
            f"size={float(min(sizes[a], sizes[b]))!r}:"
            "the volume targets are unattainable in this region at the requested "
            "size; a ProximitySizeControl on these faces or relaxed targets "
            "resolve it"
        )
    return findings


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
        raise ValueError("The prepared CAD volume has no native route.")
    domain.require_current(domain.source_id, domain.source_revision)
    sizes = np.asarray(boundary.size_values, dtype=np.float64)
    model = cad_geometry_bundle(domain).model
    order = specification.target.geometry_order
    proximity = _proximity_controls(prepared, model)
    try:
        return cad_mesh_solid(
            model,
            sizes,
            np.full(sizes.shape, deviation, dtype=np.float64),
            np.full((model.curve_nodes.size,), deviation, dtype=np.float64),
            radius_edge_bound=(
                np.inf
                if schedule.radius_edge_bound is None
                else schedule.radius_edge_bound
            ),
            min_dihedral_degrees=schedule.minimum_dihedral_degrees,
            min_relative_determinant=schedule.minimum_relative_determinant,
            geometry_order=order,
            proximity_controls=tuple(native for _, native in proximity),
            sampled_minimum_scaled_jacobian=prepared.options.sampled_minimum_scaled_jacobian,
            sampled_minimum_determinant_ratio=prepared.options.sampled_minimum_determinant_ratio,
        )
    except ValueError as error:
        if not proximity:
            raise
        # The native field refuses a zero size: a gap lower bound reaching
        # zero (the scopes within the deviation of each other) without a
        # minimum_size.
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SPECIFICATION,
            "The native CAD solid refused its proximity size field (a "
            "ProximitySizeControl gap bound reaches zero without minimum_size): "
            f"{error}",
            stage="size_field_resolution",
        ) from error


def _seed_regions(solid: CadSolidResult, seeds: np.ndarray, /) -> np.ndarray:
    """Source region row holding each seed point, -1 outside the solid.

    The tetrahedra fill exactly the shell of the certified surface
    triangles, so exact orientation predicates against their closed cells
    classify each seed; a seed on a surface triangle has no unambiguous
    side and is refused (as the PLC recovery refuses a seed on a constrained
    face).
    """
    volume = solid.volume
    coordinates = np.asarray(volume.points, dtype=np.float64)
    cells = np.asarray(volume.tetrahedra, dtype=np.int64)
    regions = np.asarray(volume.tetrahedron_regions, dtype=np.int64)
    surface = {
        tuple(sorted(triangle))
        for triangle in np.asarray(solid.surface.triangles, dtype=np.int64).tolist()
    }
    faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    corners = [coordinates[cells[:, list(face)]] for face in faces]
    # Sign of each cell's opposite corner: a seed is in a closed cell when no
    # face sign disagrees with it.
    expected = [
        exact_orient3d(
            corner[:, 0], corner[:, 1], corner[:, 2], coordinates[cells[:, slot]]
        ).astype(np.int64)
        for slot, corner in enumerate(corners)
    ]
    result = np.full((seeds.shape[0],), -1, dtype=np.int64)
    for row, seed in enumerate(seeds):
        point = np.broadcast_to(seed, (cells.shape[0], 3))
        signs = np.stack(
            [
                exact_orient3d(corner[:, 0], corner[:, 1], corner[:, 2], point).astype(
                    np.int64
                )
                * sign
                for corner, sign in zip(corners, expected, strict=True)
            ],
            axis=1,
        )
        holders = np.flatnonzero(np.all(signs >= 0, axis=1))
        if holders.size == 0:
            continue
        on_surface = any(
            tuple(sorted(cells[cell, list(faces[slot])].tolist())) in surface
            for cell in holders.tolist()
            for slot in np.flatnonzero(signs[cell] == 0).tolist()
        )
        if on_surface or np.unique(regions[holders]).size != 1:
            raise MeshingFailure(
                MeshingFailureCategory.REGION_RESOLUTION_FAILED,
                f"Native CAD seed {seed.tolist()!r} lies on the certified surface.",
                # The PLC recovery refuses a seed on a constrained face.
                provider_code="plc_source_invalid_seed",
                stage="volume_meshing",
            )
        result[row] = int(regions[holders[0]])
    return result


def _check_seeds(
    prepared: _PreparedDirectBRepCadVolume, solid: CadSolidResult, /
) -> None:
    """Region seeds must lie in their region and hole seeds outside the solid.

    A single-region seed classification is the source's own: a hole seed in
    a source region would leak through its bounding faces, and a region seed
    elsewhere contradicts them, both refused as the PLC recovery refuses
    them (REGION_LEAK, INCONSISTENT_REGIONS). Outside the solid every seed
    is in the void, as on the existing route: a hole seed there is a no-op
    and a region seed leaks, wherever it lies relative to the hull.
    """
    domain = prepared.domain
    specification = prepared.specification
    for seed in specification.hole_seeds:
        _domain_scope_rows(domain, seed.scope, 3)
    points = [
        np.asarray(seed.point, dtype=np.float64)
        for seed in (*specification.region_seeds, *specification.hole_seeds)
    ]
    if not points:
        return
    if any(point.shape != (3,) for point in points):
        raise ValueError("CAD volume seeds must be three-dimensional points.")
    found = _seed_regions(solid, np.stack(points))
    expected = [
        _region_row(domain, seed.region_name) for seed in specification.region_seeds
    ] + [-1] * len(specification.hole_seeds)
    for point, region, wanted in zip(points, found.tolist(), expected, strict=True):
        if region == wanted:
            continue
        located = "outside the solid" if region < 0 else f"in region {region}"
        raise MeshingFailure(
            MeshingFailureCategory.REGION_RESOLUTION_FAILED,
            f"Native CAD {'hole' if wanted < 0 else 'region'} seed "
            f"{point.tolist()!r} lies {located}"
            + ("" if wanted < 0 else f", not in its source region {wanted}")
            + ".",
            # The PLC recovery's codes: a void label meeting a region leaks,
            # two regions are inconsistent.
            provider_code=(
                "plc_source_region_leak"
                if region < 0 or wanted < 0
                else "plc_source_inconsistent_regions"
            ),
            stage="volume_meshing",
        )


def _organization(
    prepared: _PreparedDirectBRepCadVolume,
    mesh: CellMesh,
    solid: CadSolidResult,
    associations: tuple[GeometryAssociation, ...],
    /,
) -> NativeMeshOrganization:
    """Source-face patches, region zones and region evidence (PLC substrate).

    Boundary triangles group by their native source face; zones are the
    tetrahedron regions. Controls and seeds are the existing route's
    (``_native_cad_volume``), mapped to the same source rows.
    """
    domain = prepared.domain
    specification = prepared.specification
    _check_seeds(prepared, solid)
    facet_regions = cad_facet_regions(domain)
    surface = solid.surface
    patches, zones, labels, evidence = source_organization(
        domain,
        SourceOrganizationLayout(
            mesh,
            tuple(domain.entity_id(3, row) for row in range(len(domain.regions))),
            tuple(tuple(path) for path in domain.region_source_occurrences),
            np.asarray(solid.volume.tetrahedron_regions, dtype=np.int64),
            tuple(domain.entity_id(2, row) for row in range(len(domain.patches))),
            tuple(tuple(path) for path in domain.source_occurrences[2]),
            facet_regions,
            np.asarray(surface.triangles, dtype=np.int64),
            np.asarray(surface.triangle_patches, dtype=np.int64),
            np.asarray(surface.curve_edges, dtype=np.int64).reshape((-1, 2)),
        ),
        tuple(
            zip(
                (control.name for control in specification.patch_controls),
                cad_patch_control_rows(domain, specification, facet_regions),
                strict=True,
            )
        ),
        specification.region_controls,
        cad_region_seeds(domain, specification),
    )
    return NativeMeshOrganization(
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        region_boundary_evidence=evidence,
    )


def _cad_failure_evidence(
    solid: CadSolidResult, deviation: float, /, *, volume_recovery: bool
) -> MeshingFailureEvidence:
    curves = solid.surface.curve_failures
    faces = solid.surface.face_failures
    initial_faces = solid.surface.initial_face_failures
    achieved = {
        "unresolved_curve_intervals": float(curves.size),
        "face_findings": float(faces.size),
    }
    if solid.surface.surface_refinement_retries:
        achieved["surface_refinement_retries"] = float(
            solid.surface.surface_refinement_retries
        )
        achieved["initial_face_findings"] = float(initial_faces.size)
    findings: list[str] = []
    for kind, field, bank in (
        ("curve", "curve", curves),
        ("face", "face", faces),
        ("initial_face", "face", initial_faces),
    ):
        fields = bank.dtype.names
        if fields is None:
            raise RuntimeError("Native CAD failure banks require named fields.")
        for row in bank:
            count_key = f"{kind}_{int(row[field])}_findings"
            achieved[count_key] = achieved.get(count_key, 0.0) + 1.0
            findings.append(
                f"{kind}_finding:"
                + ":".join(f"{name}={row[name].tolist()!r}" for name in fields)
            )
    if volume_recovery:
        stats = solid.volume.stats[0]
        for name in ("recovery_attempts", "recovery_status", "recovery_reason"):
            achieved[name] = float(stats[name])
    return MeshingFailureEvidence(
        requested_quantities={"maximum_deviation": deviation},
        achieved_quantities=achieved,
        checkpoint="cad_volume_recovery"
        if volume_recovery
        else "cad_surface_construction",
        logical_findings=findings,
    )


def _cad_surface_failure(solid: CadSolidResult, deviation: float, /) -> MeshingFailure:
    curves = solid.surface.curve_failures
    faces = solid.surface.face_failures
    if faces.size:
        row = faces[0]
        first = (
            f"; first face {int(row['face'])}, reason {int(row['reason'])}, "
            f"triangle {int(row['triangle'])}, refusal {int(row['refusal_kind'])}"
        )
    elif curves.size:
        first = f"; first unresolved interval on curve {int(curves[0]['curve'])}"
    else:
        first = ""
    return MeshingFailure(
        MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
        "Native CAD surface construction did not reach volume recovery "
        f"(status {solid.status.name}, {curves.size} unresolved curve "
        f"intervals and {faces.size} face findings{first}).",
        provider_code="native_cad_surface_unresolved",
        stage="surface_meshing",
        evidence=_cad_failure_evidence(solid, deviation, volume_recovery=False),
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
    if not all(control.meshing_enabled for control in specification.region_controls):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Native CAD volume meshing meshes every source region: region "
            "controls must keep meshing enabled.",
            stage="volume_meshing",
        )
    order = specification.target.geometry_order
    sizes = np.asarray(boundary.size_values, dtype=np.float64)
    solid = construct_cad_solid(prepared)
    model = cad_geometry_bundle(domain).model
    proximity = _proximity_controls(prepared, model)
    volume = solid.volume
    stats = volume.stats[0]
    if volume.tetrahedra.shape[0] == 0:
        if int(stats["recovery_attempts"]) == 0:
            raise _cad_surface_failure(solid, deviation)
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            "Native CAD volume recovery refused the surface (status "
            f"{solid.status.name}, recovery status "
            f"{int(stats['recovery_status'])}, reason "
            f"{int(stats['recovery_reason'])}, entity kind "
            f"{int(stats['failure_kind'])} row {int(stats['failure_entity'])}, "
            f"smallest-angle sine {float(stats['failure_measure'])!r}).",
            provider_code="native_cad_volume_unresolved",
            stage="volume_meshing",
            evidence=(
                _cad_failure_evidence(solid, deviation, volume_recovery=True)
                if solid.surface.surface_refinement_retries
                else None
            ),
        )
    mesh = CellMesh.from_simplices(
        np.asarray(volume.points, dtype=np.float64),
        np.asarray(volume.tetrahedra, dtype=np.int64),
        dimension=3,
        block_name=_BLOCK,
        numeric_version=domain.source_revision,
    )
    match order:
        case 1:
            geometry = CellGeometrySpec.affine(mesh)
        case 2:
            geometry = _quadratic_geometry(mesh.blocks[0].cell_kind, solid)
        case 3:
            geometry = _cubic_geometry(mesh.blocks[0].cell_kind, solid)
        case _:
            raise ValueError(f"Unsupported native CAD geometry order {order!r}.")
    # Unmet cells split per cell: forced by the input (a rim wedge below the
    # dihedral target explains every failure of the cell) or not; only the
    # unforced ones are an issue.
    forced = int(stats["unmet_forced_cells"])
    unforced = int(stats["unmet_unforced_cells"])
    issues: list[str] = []
    gated = _shape_targets_gated(schedule)
    mapped_shape = (
        (
            "sampled_minimum_scaled_jacobian",
            prepared.options.sampled_minimum_scaled_jacobian,
            "unmet_scaled_jacobian_cells",
        ),
        (
            "sampled_minimum_determinant_ratio",
            prepared.options.sampled_minimum_determinant_ratio,
            "unmet_determinant_ratio_cells",
        ),
    )
    mapped_miss = any(int(stats[count]) for _, _, count in mapped_shape)
    affine_unresolved = solid.status is MeshcoreStatus.UNRESOLVED and (
        unforced > 0 or not mapped_miss
    )
    deviations = (
        _proxy_deviations(model, solid, float(deviation))
        if proximity or (gated and affine_unresolved)
        else None
    )
    requested: list[tuple[str, float]] = [
        ("maximum_deviation", float(deviation)),
        ("minimum_dihedral_degrees", float(schedule.minimum_dihedral_degrees)),
    ]
    if schedule.radius_edge_bound is not None:
        requested.append(("radius_edge_bound", float(schedule.radius_edge_bound)))
    achieved: list[tuple[str, float]] = [
        ("unmet_quality_cells", float(int(stats["unmet_cells"]))),
        ("unmet_forced_quality_cells", float(forced)),
        ("unmet_unforced_quality_cells", float(unforced)),
    ]
    for name, target, count in mapped_shape:
        if target > 0.0:
            requested.append((name, target))
            missed = int(stats[count])
            achieved.append((count, float(missed)))
            if missed:
                issues.append(f"native_{count}:{missed}")
    if solid.surface.surface_refinement_retries:
        achieved.extend(
            (
                (
                    "surface_refinement_retries",
                    float(solid.surface.surface_refinement_retries),
                ),
                (
                    "initial_face_findings",
                    float(solid.surface.initial_face_failures.size),
                ),
            )
        )
    if gated:
        if affine_unresolved:
            issues.append(f"native_unmet_quality_cells:{unforced}")
            # Input finding: a rim whose true wedge is below the dihedral
            # target makes that target unattainable in a neighborhood of the
            # rim for any mesh (cad_mesh.cpp, count_forced_cells); relaxing
            # minimum_dihedral_degrees below the wedge is the remedy.
            target = float(schedule.minimum_dihedral_degrees)
            wedges = solid.volume.curve_rim_wedges
            for curve in np.flatnonzero(wedges < target):
                issues.append(
                    f"input_rim_wedge_below_dihedral_target:curve={int(curve)}:"
                    f"wedge_degrees={float(wedges[curve])!r}:"
                    f"target_degrees={target!r}:"
                    "the dihedral target is unattainable near this rim for any mesh"
                )
            # Input finding: unmet cells where the solid is thinner than the
            # targets allow at its boundary size (_thin_region_findings); the
            # request's size is kept, a ProximitySizeControl is the remedy.
            assert deviations is not None
            issues.extend(
                _thin_region_findings(
                    model,
                    solid,
                    sizes,
                    deviations,
                    schedule.radius_edge_bound,
                    target,
                )
            )
    else:
        # A soft schedule publishes unmet shape rows; validity, determinant,
        # construction and P2 rows of unforced cells stay hard (as the Python
        # route). A forced cell is never an issue (cad_mesh.cpp).
        hard = int(stats["unmet_hard_cells"])
        quadratic = stats["quadratic"]
        curved = int(quadratic["final_invalid"]) + int(quadratic["final_unresolved"])
        if hard:
            issues.append(f"native_unmet_quality_cells:{hard}")
        if curved:
            issues.append(f"native_unresolved_quadratic_cells:{curved}")
        achieved.append(
            ("soft_shape_target_unmet_count", float(stats["unmet_shape_rows"]))
        )
    _size_compliance(prepared, solid, issues, requested, achieved)
    if proximity:
        assert deviations is not None
        _proximity_compliance(solid, proximity, deviations, requested, achieved)
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )
    regions = np.asarray(volume.tetrahedron_regions, dtype=np.int64)
    witness = NativeCadSurfaceWitness(
        mesh, solid.surface, len(cad_geometry_bundle(domain).model.loop_surfaces)
    )
    if len(domain.regions) > 1:
        _native_cad_regional_surface(mesh, domain, witness, native_surface=solid.surface)
    records, associations = native_cad_records(
        mesh, domain, regions, witness, float(deviation)
    )
    certification = NativeCertificationRequest(
        MeshCertificationSchedule("volume_cad"),
        domain.source_id,
        domain.source_revision,
        domain=domain,
        cell_regions=regions,
        # The boundary preparation's query selects every source face (support).
        fidelity_source=boundary.boundary_source,
        fidelity_tolerance=float(deviation),
        protected_features=specification.protected_features,
        feature_source=domain,
        native_cad_surface=witness,
        native_cad_records=records,
    )
    return NativeMeshConstruction(
        mesh,
        prepared.coordinate_contract,
        compliance,
        certification,
        geometry=geometry,
        organization=_organization(prepared, mesh, solid, associations),
    )


__all__ = [
    "cad_solid_support_issues",
    "construct_cad_solid",
    "execute_cad_solid",
]
