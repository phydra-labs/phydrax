# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Source-bound boundary queries across surface lineage and chart changes."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_geometry_validity as cell_validity_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._sphere_chart_records as sphere_deformation_owner
import phydrax.discretization._surface_chart_records as chart_deformation_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._meshing_domain as domain_owner
import phydrax.geometry._rational as rational_owner
import phydrax.geometry._source_boundary_records as boundary_records_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry._surface_source_support as surface_support_owner
import phydrax.meshing._lineage as lineage_owner
import phydrax.meshing._result as result_owner
import phydrax.meshing._surface_boundary_operations as boundary_operations_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import PolygonalConnectivity
from ..typing import checked


_REFERENCE_EDGES = (
    ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
    ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
    ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
)


def _edge_parameter(
    point: tuple[Fraction, Fraction],
    edge: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    /,
) -> Fraction | None:
    """Return an exact parameter when a point lies on an oriented unit edge."""
    start, end = edge
    delta = (end[0] - start[0], end[1] - start[1])
    axis = 0 if delta[0] else 1
    if not delta[axis]:
        return None
    parameter = (point[axis] - start[axis]) / delta[axis]
    if not 0 <= parameter <= 1:
        return None
    other = 1 - axis
    if start[other] + parameter * delta[other] != point[other]:
        return None
    return parameter


def _tiles_reference_triangle(
    children: list[tuple[tuple[Fraction, Fraction], ...]], /
) -> bool:
    """Prove positive child triangles tile the oriented unit reference triangle."""
    directed: Counter[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]] = (
        Counter()
    )
    double_area = Fraction(0)
    for corners in children:
        if len(corners) != 3:
            return False
        first, second, third = corners
        area = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
            third[0] - first[0]
        )
        if area <= 0:
            return False
        double_area += area
        for start, end in zip(corners, (*corners[1:], corners[0]), strict=True):
            reverse = (end, start)
            if directed[reverse]:
                directed[reverse] -= 1
                if directed[reverse] == 0:
                    del directed[reverse]
            else:
                directed[(start, end)] += 1
    if double_area != 1:
        return False

    sides: list[list[tuple[Fraction, Fraction]]] = [[], [], []]
    for (start, end), count in directed.items():
        if count != 1:
            return False
        match = None
        for side, edge in enumerate(_REFERENCE_EDGES):
            first = _edge_parameter(start, edge)
            last = _edge_parameter(end, edge)
            if first is not None and last is not None:
                match = (side, first, last)
                break
        if match is None:
            return False
        side, first, last = match
        if last <= first:
            return False
        sides[side].append((first, last))

    for intervals in sides:
        intervals.sort()
        position = Fraction(0)
        for first, last in intervals:
            if first != position or last <= first:
                return False
            position = last
        if position != 1:
            return False
    return True


def _triangle_side_ids(mesh: cell_mesh_owner.CellMesh) -> np.ndarray:
    """Resolve ordered triangle sides through nominal vertex incidence."""
    if mesh.topological_dimension != 2 or any(
        block.cell_kind != "triangle" for block in mesh.blocks
    ):
        raise ValueError(
            "SurfaceBoundarySource.mesh must retain triangular surface incidence; got another cell family."
        )
    match mesh.connectivity:
        case cell_mesh_owner.SimplicialConnectivity() as connectivity:
            edges = np.asarray(connectivity.entities[1], dtype=np.int64)
        case PolygonalConnectivity() as connectivity:
            edges = np.asarray(connectivity.edges, dtype=np.int64)
        case _:
            raise ValueError(
                "SurfaceBoundarySource.mesh must expose canonical triangle-side incidence; got unsupported topology."
            )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            "SurfaceBoundarySource.mesh edge incidence must contain vertex pairs."
        )
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64).reshape(-1)
    if edge_ids.shape != (edges.shape[0],):
        raise ValueError(
            "SurfaceBoundarySource.mesh edge IDs must align with edge vertex rows."
        )
    lookup: dict[tuple[int, int], int] = {}
    for row, identifier in enumerate(edge_ids):
        first, last = int(edges[row, 0]), int(edges[row, 1])
        lookup[(min(first, last), max(first, last))] = int(identifier)
    triangles = np.concatenate(
        tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks), axis=0
    )
    try:
        return np.asarray(
            [
                [
                    lookup[
                        (
                            min(int(cell[first]), int(cell[last])),
                            max(int(cell[first]), int(cell[last])),
                        )
                    ]
                    for first, last in ((0, 1), (1, 2), (2, 0))
                ]
                for cell in triangles
            ],
            dtype=np.int64,
        ).reshape((-1, 3))
    except KeyError as error:
        raise ValueError(
            "SurfaceBoundarySource.mesh triangle sides must name declared edges; got incomplete topology."
        ) from error


def _boundary_occurrence_identity(
    source: domain_boundary_owner.MeshingDomainBoundarySource, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "meshing-domain-boundary-occurrences",
            "source": (source.source_id, source.source_revision),
            "domain": source.domain.content_domain_id(),
            "scope": source.source_scope_id,
            "patches": source.patches,
            "resolution": source.resolution,
            "arrays": array_tree_fingerprint(
                (
                    source.boundary_cell_ids,
                    source.boundary_patch_ids,
                    source.boundary_occurrences,
                )
            ),
        }
    )


def _subdivide_boundary_occurrences(
    source: domain_boundary_owner.MeshingDomainBoundarySource,
    mesh: cell_mesh_owner.CellMesh,
    lineage: lineage_owner.MeshLineage,
    target: cell_mesh_owner.CellMesh,
    /,
) -> domain_boundary_owner.MeshingDomainBoundarySource:
    """Transport authored coedge occurrences only through exact cell/edge lineage."""
    if (
        source.boundary_cell_ids is None
        or source.boundary_patch_ids is None
        or source.boundary_occurrences is None
    ):
        raise ValueError(
            "SurfaceSubdivisionBoundarySource requires authored root boundary occurrences."
        )
    if (lineage.source_topology_id, lineage.target_topology_id) != (
        mesh.topology_id,
        target.topology_id,
    ):
        raise ValueError(
            "SurfaceSubdivisionBoundarySource.lineage must bind its source and target meshes; got stale topology."
        )
    before_edges = _triangle_side_ids(mesh)
    after_edges = _triangle_side_ids(target)
    cell_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    target_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in target.blocks)
    )
    source_rows = {
        int(value): row
        for row, value in enumerate(np.asarray(source.boundary_cell_ids, dtype=np.int64))
    }
    mesh_rows = {int(value): row for row, value in enumerate(cell_ids.tolist())}
    if set(source_rows) != set(mesh_rows):
        raise ValueError(
            "SurfaceSubdivisionBoundarySource.boundary_cells must exactly cover the root mesh; got missing or foreign cells."
        )
    cells = lineage.entity_lineage(2)
    edges = lineage.entity_lineage(1)
    cell_parents: dict[int, set[int]] = {}
    edge_parents: dict[int, set[int]] = {}
    for record, parents in ((cells, cell_parents), (edges, edge_parents)):
        for parent, child in zip(
            np.asarray(record.source_global_ids, dtype=np.int64),
            np.asarray(record.target_global_ids, dtype=np.int64),
            strict=True,
        ):
            parents.setdefault(int(child), set()).add(int(parent))
    if set(cell_parents) != set(target_ids.tolist()):
        raise ValueError(
            "SurfaceSubdivisionBoundarySource.cell_lineage must explicitly identify every target cell; got incomplete lineage."
        )
    bank = np.full((target_ids.size, 3, 3), -1, dtype=np.int32)
    patches = np.empty(target_ids.size, dtype=np.int32)
    covered: set[tuple[int, int]] = set()
    for row, identifier in enumerate(target_ids.tolist()):
        parents = cell_parents.get(int(identifier), set())
        if len(parents) != 1:
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.cells must retain one explicit parent occurrence; got merged or absent ancestry."
            )
        parent = next(iter(parents))
        if parent not in source_rows:
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.cells must reference a declared root cell; got foreign parent."
            )
        source_row = source_rows[parent]
        patches[row] = np.asarray(source.boundary_patch_ids, dtype=np.int32)[source_row]
        for side, edge in enumerate(after_edges[row]):
            inherited = edge_parents.get(int(edge), set())
            candidates = [
                slot
                for slot, original_edge in enumerate(before_edges[mesh_rows[parent]])
                if int(original_edge) in inherited
            ]
            if len(candidates) > 1:
                raise ValueError(
                    "SurfaceSubdivisionBoundarySource.side must retain one parent side identity; got ambiguous edge ancestry."
                )
            if candidates:
                bank[row, side] = source.boundary_occurrences[source_row, candidates[0]]
                covered.add((source_row, candidates[0]))
            elif inherited:
                raise ValueError(
                    "SurfaceSubdivisionBoundarySource.side must inherit from its parent cell; got foreign parent edge."
                )
    required = {
        (int(row), int(side))
        for row, side in np.argwhere(
            np.any(np.asarray(source.boundary_occurrences) != -1, axis=2)
        )
    }
    if not required.issubset(covered):
        raise ValueError(
            "SurfaceSubdivisionBoundarySource.lineage must cover every authored root side; got missing boundary descendants."
        )
    return domain_boundary_owner.MeshingDomainBoundarySource(
        source.domain,
        source.patches,
        resolution=source.resolution,
        boundary_cell_ids=target_ids.astype(np.int64),
        boundary_patch_ids=patches,
        boundary_occurrences=bank,
    )


def _boundary_sources_by_patch(
    source: domain_boundary_owner.MeshingDomainBoundarySource,
    /,
) -> tuple[tuple[int, domain_boundary_owner.MeshingDomainBoundarySource], ...]:
    """Give each source patch an independent cover query without losing ancestry."""
    output: list[tuple[int, domain_boundary_owner.MeshingDomainBoundarySource]] = []
    patch_ids = (
        None
        if source.boundary_patch_ids is None
        else np.asarray(source.boundary_patch_ids, dtype=np.int32)
    )
    for patch in source.patches:
        bank = cell_ids = owners = None
        if patch_ids is not None:
            rows = np.flatnonzero(patch_ids == patch)
            if source.boundary_cell_ids is None or source.boundary_occurrences is None:
                raise ValueError(
                    "SurfaceBoundarySource.boundary_occurrences must be complete when patch rows are present."
                )
            cell_ids = np.asarray(source.boundary_cell_ids, dtype=np.int64)[rows]
            owners = patch_ids[rows]
            bank = np.asarray(source.boundary_occurrences, dtype=np.int32)[rows]
        query = domain_boundary_owner.MeshingDomainBoundarySource(
            source.domain,
            (patch,),
            resolution=source.resolution,
            boundary_occurrences=bank,
            boundary_cell_ids=cell_ids,
            boundary_patch_ids=owners,
        )
        output.append((patch, query))
    return tuple(output)


def _combined_cover(
    sources: tuple[tuple[int, domain_boundary_owner.MeshingDomainBoundarySource], ...],
    patch_deviations: dict[int, float],
    source_id: str,
    source_revision: str,
    /,
) -> boundary_records_owner.SourceBoundaryChartCover:
    """Combine per-patch source covers with proved target-map deviations."""
    simplices: list[np.ndarray] = []
    bounds: list[np.ndarray] = []
    findings: list[embedding_state_owner.MeshCertificateFinding] = []
    restriction_ids: set[str] = set()
    complete = bool(sources)
    for patch, query in sources:
        deviation = patch_deviations.get(patch)
        if deviation is None or not np.isfinite(deviation) or deviation < 0:
            complete = False
            findings.append(
                embedding_state_owner.MeshCertificateFinding(
                    "source_trim_coverage_premise", "unresolved", "mesh"
                )
            )
            deviation = np.inf
        cover = query.boundary_chart_cover()
        if (cover.source_id, cover.source_revision) != (source_id, source_revision):
            raise ValueError(
                "SurfaceBoundarySource.cover must retain source identity and revision; got foreign authority."
            )
        complete &= cover.complete and cover.semantics == "certified"
        if (not cover.complete or cover.semantics != "certified") and not cover.findings:
            findings.append(
                embedding_state_owner.MeshCertificateFinding(
                    "source_trim_coverage_premise", "unresolved", "mesh"
                )
            )
        findings.extend(cover.findings)
        restriction_ids.update(cover.restriction_ids)
        simplices.append(np.asarray(cover.simplices, dtype=np.float64))
        source_bounds = np.asarray(cover.deviation_bounds, dtype=np.float64)
        if source_bounds.shape != (cover.simplices.shape[0],):
            raise ValueError(
                "SurfaceBoundarySource.cover bounds must align with source simplices; got mismatched evidence."
            )
        patch_bounds = np.empty(source_bounds.shape, dtype=np.float64)
        for row, bound in enumerate(source_bounds.tolist()):
            if np.isfinite(bound) and np.isfinite(deviation):
                exact_bound = Fraction(float(bound)) + Fraction(float(deviation))
                if exact_bound > Fraction(float(np.finfo(np.float64).max)):
                    patch_bounds[row] = np.inf
                    complete = False
                else:
                    patch_bounds[row] = rational_owner.fraction_up(exact_bound)
            else:
                patch_bounds[row] = np.inf
                complete = False
        bounds.append(patch_bounds)
    if not simplices:
        return boundary_records_owner.SourceBoundaryChartCover(
            np.empty((0, 3, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            "sampled",
            False,
            source_id,
            source_revision,
            findings=(
                embedding_state_owner.MeshCertificateFinding(
                    "source_trim_coverage_premise", "unresolved", "mesh"
                ),
            ),
        )
    values = np.concatenate(simplices, axis=0)
    deviations = np.concatenate(bounds)
    return boundary_records_owner.SourceBoundaryChartCover(
        values,
        deviations,
        "certified" if complete else "sampled",
        complete,
        source_id,
        source_revision,
        findings=tuple(findings),
        restriction_ids=tuple(sorted(restriction_ids)),
    )


@final
class SurfaceSubdivisionBoundarySource(StrictModule, NonTrainableState):
    """Retain the source boundary through a serial, lineage-proved subdivision."""

    root: domain_boundary_owner.MeshingDomainBoundarySource
    support: surface_support_owner.PreparedSurfaceSourceSupport
    lineages: tuple[lineage_owner.MeshLineage, ...]
    vertex_stencils: tuple[lineage_owner.VertexInterpolationStencil, ...]
    targets: tuple[cell_mesh_owner.CellMesh, ...]
    current_boundary_source: domain_boundary_owner.MeshingDomainBoundarySource
    geometry: cell_geometry_owner.CellGeometrySpec
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: domain_boundary_owner.MeshingDomainBoundarySource,
        support: surface_support_owner.PreparedSurfaceSourceSupport,
        lineages: tuple[lineage_owner.MeshLineage, ...],
        targets: tuple[cell_mesh_owner.CellMesh, ...],
        vertex_stencils: tuple[lineage_owner.VertexInterpolationStencil, ...],
        geometry: cell_geometry_owner.CellGeometrySpec,
        /,
    ) -> None:
        if (
            any(not isinstance(value, lineage_owner.MeshLineage) for value in lineages)
            or any(not isinstance(value, cell_mesh_owner.CellMesh) for value in targets)
            or any(
                not isinstance(value, lineage_owner.VertexInterpolationStencil)
                for value in vertex_stencils
            )
        ):
            raise TypeError(
                "SurfaceSubdivisionBoundarySource.epochs must retain lineage, mesh, and vertex-stencil records."
            )
        original = support.original
        if not isinstance(original, result_owner.CellMeshingResult):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.support.original must be an accepted serial surface result."
            )
        if (
            (root.source_id, root.source_revision)
            != (support.domain.source_id, support.domain.source_revision)
            or root.domain.content_domain_id() != support.domain.content_domain_id()
            or set(root.patches)
            != set(np.asarray(support.root_patches, dtype=np.int64).tolist())
        ):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.root must bind the support source, revision, domain, and patches."
            )
        if not lineages or not (len(lineages) == len(targets) == len(vertex_stencils)):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.epochs require one mesh and exact vertex stencil per lineage."
            )
        topology = support.root_topology_id
        current_source = root
        current_mesh = original.mesh
        for lineage, target, vertex_stencil in zip(
            lineages, targets, vertex_stencils, strict=True
        ):
            if (
                lineage.source_topology_id != topology
                or lineage.target_topology_id != target.topology_id
                or vertex_stencil.source_entity_set_id
                != current_mesh.entity_set(0).entity_set_id
                or vertex_stencil.target_entity_set_id
                != target.entity_set(0).entity_set_id
            ):
                raise ValueError(
                    "SurfaceSubdivisionBoundarySource epochs must chain source and target topology and vertex-stencil entities."
                )
            current_source = _subdivide_boundary_occurrences(
                current_source, current_mesh, lineage, target
            )
            current_mesh = target
            topology = target.topology_id
        restriction = geometry.restriction_source
        if restriction is not None and (
            restriction.source_topology_id != original.mesh.topology_id
            or restriction.source_geometry_id
            != cell_validity_owner.cell_geometry_id(original.geometry)
        ):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.geometry must retain root topology and geometry ancestry."
            )
        self.root = root
        self.support = support
        self.lineages = tuple(lineages)
        self.vertex_stencils = tuple(vertex_stencils)
        self.targets = tuple(targets)
        self.current_boundary_source = current_source
        self.geometry = geometry
        self.source_id = root.source_id
        self.source_revision = root.source_revision

    @property
    def domain(self) -> domain_owner.MeshingDomain:
        return self.root.domain

    @property
    def patches(self) -> tuple[int, ...]:
        return self.root.patches

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    def boundary_distance(
        self, points: np.ndarray, /
    ) -> boundary_records_owner.SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_cover(
        self, maximum_radius: float, /
    ) -> boundary_records_owner.SourceBoundarySamples:
        return self.root.boundary_cover(maximum_radius)

    def require_current(self) -> None:
        """Revalidate source authority and exact nominal occurrence lineage."""
        self.support.require_current()
        self.root.require_current()
        self.current_boundary_source.require_current()
        rebuilt = SurfaceSubdivisionBoundarySource(
            self.root,
            self.support,
            self.lineages,
            self.targets,
            self.vertex_stencils,
            self.geometry,
        )
        if (rebuilt.source_id, rebuilt.source_revision) != (
            self.source_id,
            self.source_revision,
        ) or (
            _boundary_occurrence_identity(rebuilt.current_boundary_source)
            != _boundary_occurrence_identity(self.current_boundary_source)
        ):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.source must retain its exact root occurrence identity."
            )

    @checked
    def boundary_chart_cover(
        self,
        /,
    ) -> boundary_records_owner.SourceBoundaryChartCover:
        self.require_current()
        root = self.support.original
        if not isinstance(root, result_owner.CellMeshingResult):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.support.original must retain the accepted root result."
            )
        proof, _ = boundary_operations_owner.subdivision_proof(
            self.support,
            self.lineages,
            self.targets,
            self.vertex_stencils,
            self.geometry,
        )
        root_ids = set(np.asarray(self.support.root_cell_ids, dtype=np.int64).tolist())
        children: dict[int, list[tuple[tuple[Fraction, Fraction], ...]]] = {}
        for root_id, corners in zip(
            proof.root_cell_ids.tolist(), proof.reference_corners, strict=True
        ):
            children.setdefault(int(root_id), []).append(corners)
        complete = set(children) == root_ids and all(
            _tiles_reference_triangle(rows) for rows in children.values()
        )
        patches = np.asarray(proof.patches, dtype=np.int64)
        deviations = np.asarray(proof.restriction_deviations, dtype=np.float64)
        if patches.shape != deviations.shape:
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.proof must align patch and restriction deviations."
            )
        patch_deviations = {
            int(patch): float(np.max(deviations[patches == patch]))
            for patch in np.unique(patches).tolist()
        }
        if not complete:
            return _combined_cover(
                _boundary_sources_by_patch(self.root),
                {},
                self.source_id,
                self.source_revision,
            )
        return _combined_cover(
            _boundary_sources_by_patch(self.root),
            patch_deviations,
            self.source_id,
            self.source_revision,
        )


def _validate_surface_chart_boundary_binding(
    root: domain_boundary_owner.MeshingDomainBoundarySource
    | SurfaceSubdivisionBoundarySource,
    support: surface_support_owner.PreparedSurfaceSourceSupport,
    deformation: chart_deformation_owner.PreparedSurfaceChartDeformation
    | sphere_deformation_owner.PreparedSphereChartDeformation,
    /,
) -> None:
    if (
        (root.source_id, root.source_revision)
        != (support.domain.source_id, support.domain.source_revision)
        or root.domain.content_domain_id() != support.domain.content_domain_id()
        or set(root.patches)
        != set(np.asarray(support.root_patches, dtype=np.int64).tolist())
    ):
        raise ValueError(
            "SurfaceChartBoundarySource.root must bind the support source, domain, and patches."
        )
    if isinstance(deformation, sphere_deformation_owner.PreparedSphereChartDeformation):
        support.require_root(deformation.source_mesh, deformation.source_geometry)
        source_profile = deformation.source_atlas.source_profile
        if (
            (source_profile.source_id, source_profile.source_revision)
            != (support.domain.source_id, support.domain.source_revision)
            or source_profile.coordinate_contract.spatial_id
            != support.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "Sphere material source profile must bind the support source identity, revision, and coordinate contract."
            )
    elif deformation.domain_id != support.domain.domain_id:
        raise ValueError(
            "SurfaceChartBoundarySource.chart deformation must bind the exact support domain."
        )


def _surface_boundary_root_identity(
    root: domain_boundary_owner.MeshingDomainBoundarySource
    | SurfaceSubdivisionBoundarySource,
    /,
) -> str:
    if isinstance(root, SurfaceSubdivisionBoundarySource):
        return canonical_fingerprint(
            {
                "kind": "surface-subdivision-boundary-root",
                "source": (root.source_id, root.source_revision),
                "domain": root.domain.content_domain_id(),
                "support": root.support.support_id,
                "root_boundary": _boundary_occurrence_identity(root.root),
                "lineages": tuple(value.lineage_id for value in root.lineages),
                "targets": tuple(value.mesh_id for value in root.targets),
                "geometry": cell_validity_owner.cell_geometry_id(root.geometry),
                "current_boundary": _boundary_occurrence_identity(
                    root.current_boundary_source
                ),
            }
        )
    return _boundary_occurrence_identity(root)


def _surface_chart_boundary_identity(
    root: domain_boundary_owner.MeshingDomainBoundarySource
    | SurfaceSubdivisionBoundarySource,
    support: surface_support_owner.PreparedSurfaceSourceSupport,
    deformation: chart_deformation_owner.PreparedSurfaceChartDeformation
    | sphere_deformation_owner.PreparedSphereChartDeformation,
    predecessor_fidelity: source_fidelity_owner.SourceFidelityCertificate,
    /,
) -> str:
    sphere_identity = None
    if isinstance(deformation, sphere_deformation_owner.PreparedSphereChartDeformation):
        profile = deformation.source_atlas.source_profile
        sphere_identity = {
            "material_domain": deformation.domain_id,
            "source_atlas": deformation.source_atlas.atlas_id,
            "target_atlas": deformation.target_atlas.atlas_id,
            "source_profile": (
                profile.source_id,
                profile.source_revision,
                profile.profile_id,
                profile.state_id,
                profile.kernel_id,
                profile.represented_geometry_id,
                profile.coordinate_contract.spatial_id,
            ),
        }
    return canonical_fingerprint(
        {
            "kind": "surface-chart-boundary-source",
            "root": _surface_boundary_root_identity(root),
            "support": support.support_id,
            "support_domain": (
                support.domain.domain_id,
                support.domain.source_id,
                support.domain.source_revision,
            ),
            "support_frame": support.coordinate_contract.spatial_id,
            "deformation": deformation.deformation_id,
            "deformation_domain": deformation.domain_id,
            "source_mesh": deformation.source_mesh.mesh_id,
            "source_geometry": cell_validity_owner.cell_geometry_id(
                deformation.source_geometry
            ),
            "target_mesh": deformation.target_mesh.mesh_id,
            "target_geometry": cell_validity_owner.cell_geometry_id(
                deformation.target_geometry
            ),
            "sphere": sphere_identity,
            "predecessor_fidelity": predecessor_fidelity.certificate_id,
        }
    )


@final
class SurfaceChartBoundarySource(StrictModule, NonTrainableState):
    """Bind a prepared whole-domain chart deformation to source boundary queries."""

    root: (
        domain_boundary_owner.MeshingDomainBoundarySource
        | SurfaceSubdivisionBoundarySource
    )
    support: surface_support_owner.PreparedSurfaceSourceSupport
    deformation: (
        chart_deformation_owner.PreparedSurfaceChartDeformation
        | sphere_deformation_owner.PreparedSphereChartDeformation
    )
    predecessor_fidelity: source_fidelity_owner.SourceFidelityCertificate
    source_query_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: domain_boundary_owner.MeshingDomainBoundarySource
        | SurfaceSubdivisionBoundarySource,
        support: surface_support_owner.PreparedSurfaceSourceSupport,
        deformation: chart_deformation_owner.PreparedSurfaceChartDeformation
        | sphere_deformation_owner.PreparedSphereChartDeformation,
        predecessor_fidelity: source_fidelity_owner.SourceFidelityCertificate,
        /,
    ) -> None:
        root.require_current()
        support.require_current()
        deformation.require_bound(
            deformation.source_mesh,
            deformation.source_geometry,
            deformation.target_mesh,
            deformation.target_geometry,
        )
        if (
            predecessor_fidelity.status != "certified"
            or not np.isfinite(predecessor_fidelity.mesh_to_source_upper)
            or not np.isfinite(predecessor_fidelity.source_to_mesh_upper)
        ):
            raise ValueError(
                "SurfaceChartBoundarySource.predecessor_fidelity must prove both continuous source directions."
            )
        predecessor_fidelity.require_current(
            deformation.source_mesh, deformation.source_geometry, root
        )
        _validate_surface_chart_boundary_binding(root, support, deformation)
        if (
            deformation.source_domain_coverage != "certified"
            or deformation.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "SurfaceChartBoundarySource.deformation must retain complete source and target chart coverage."
            )
        self.root = root
        self.support = support
        self.deformation = deformation
        self.predecessor_fidelity = predecessor_fidelity
        self.source_query_id = _surface_chart_boundary_identity(
            root, support, deformation, predecessor_fidelity
        )

    @property
    def domain(self) -> domain_owner.MeshingDomain:
        return self.support.domain

    @property
    def patches(self) -> tuple[int, ...]:
        return self.root.patches

    @property
    def source_id(self) -> str:
        return self.root.source_id

    @property
    def source_revision(self) -> str:
        return self.root.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    def require_current(self) -> None:
        self.root.require_current()
        self.support.require_current()
        self.deformation.require_bound(
            self.deformation.source_mesh,
            self.deformation.source_geometry,
            self.deformation.target_mesh,
            self.deformation.target_geometry,
        )
        self.predecessor_fidelity.require_current(
            self.deformation.source_mesh,
            self.deformation.source_geometry,
            self.root,
        )
        _validate_surface_chart_boundary_binding(
            self.root, self.support, self.deformation
        )
        if self.source_query_id != _surface_chart_boundary_identity(
            self.root,
            self.support,
            self.deformation,
            self.predecessor_fidelity,
        ):
            raise ValueError("SurfaceChartBoundarySource source query identity changed.")

    def require_target_binding(
        self,
        mesh: cell_mesh_owner.CellMesh,
        geometry: cell_geometry_owner.CellGeometrySpec,
        /,
    ) -> None:
        self.require_current()
        self.deformation.require_bound(
            self.deformation.source_mesh,
            self.deformation.source_geometry,
            mesh,
            geometry,
        )

    def boundary_distance(
        self, points: np.ndarray, /
    ) -> boundary_records_owner.SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_cover(
        self, maximum_radius: float, /
    ) -> boundary_records_owner.SourceBoundarySamples:
        return self.root.boundary_cover(maximum_radius)

    @checked
    def boundary_chart_cover(
        self,
        /,
    ) -> boundary_records_owner.SourceBoundaryChartCover:
        self.require_current()
        deformation = self.deformation
        target_bounds = np.asarray(deformation.target_fidelity_bounds, dtype=np.float64)
        target_bound = float(np.max(target_bounds)) if target_bounds.size else 0.0
        predecessor_bound = max(
            float(self.predecessor_fidelity.mesh_to_source_upper),
            float(self.predecessor_fidelity.source_to_mesh_upper),
        ) + float(deformation.maximum_displacement_bound)
        global_bound = max(target_bound, predecessor_bound)
        patch_deviations = {patch: global_bound for patch in self.root.patches}
        boundary = (
            self.root
            if isinstance(self.root, domain_boundary_owner.MeshingDomainBoundarySource)
            else self.root.current_boundary_source
        )
        return _combined_cover(
            _boundary_sources_by_patch(boundary),
            patch_deviations,
            self.source_id,
            self.source_revision,
        )


__all__ = ["SurfaceChartBoundarySource", "SurfaceSubdivisionBoundarySource"]
