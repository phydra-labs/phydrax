# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bind continuous boundary coverage to surface subdivision and chart epochs.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "source_chart_cover_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Subdivision cell count plus source cover simplex count, or chart target cell count, does not exceed maximum_patches before chart expansion."
      }
    ],
    "conclusion": "Exceeding the chart count bound returns sampled coverage with an unresolved capacity finding.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.meshing._surface_boundary_sources.SurfaceSubdivisionBoundarySource.boundary_chart_cover",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ],
    "owner": "phydrax.meshing._surface_boundary_sources.SurfaceSubdivisionBoundarySource.boundary_chart_cover"
  },
  {
    "checks": [
      "source_trim_coverage_premise"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Source pullbacks have positive reference orientation and bounded coordinate-map deviations."
      },
      {
        "tag": "X",
        "statement": "Root child triangles cancel directed interior edges, retain one boundary segment chain on each unit-triangle side, and cover every source root; collapsed wedges follow the retained split lineage."
      },
      {
        "tag": "T",
        "statement": "Local positive degrees sum to boundary degree one, so an oriented planar subdivision with that boundary covers the reference triangle once."
      }
    ],
    "conclusion": "An unproved root tiling, missing source corner/cell identity, or unrepresented collapsed chain leaves source trim coverage unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Allen Hatcher, Algebraic Topology (2002)",
      "locator": "Section 2.2, Proposition 2.30 (sum of local degrees)",
      "url": "https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"
    },
    "failure_statuses": [
      "unresolved"
    ],
    "owner": "phydrax.meshing._surface_boundary_sources.SurfaceSubdivisionBoundarySource.boundary_chart_cover"
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._mapped_geometry_measure as mapped_geometry_owner
import phydrax.discretization._sphere_chart_records as sphere_deformation_owner
import phydrax.discretization._surface_chart_records as chart_deformation_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
import phydrax.geometry._surface_source_support as surface_source_support_owner
import phydrax.meshing._lineage_records as lineage_owner
import phydrax.meshing._result as result_owner
import phydrax.meshing._surface_boundary_operations as boundary_operations_owner

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..geometry._embedding_state import MeshCertificateFinding
from ..geometry._source_boundary_records import (
    SourceBoundaryChartCover,
    SourceBoundaryDistance,
    SourceBoundarySamples,
)
from ..typing import checked


_REFERENCE_EDGES = (
    ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))),
    ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
    ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
)


def _edge_parameter(
    edge: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    point: tuple[Fraction, Fraction],
    /,
) -> Fraction | None:
    """Rational parameter of ``point`` on one oriented unit-triangle edge, or None off it."""
    (x0, y0), (x1, y1) = edge
    dx, dy, px, py = x1 - x0, y1 - y0, point[0] - x0, point[1] - y0
    if px * dy - py * dx != 0:
        return None
    parameter = (px * dx + py * dy) / (dx * dx + dy * dy)
    return parameter if 0 <= parameter <= 1 else None


def _tiles_reference_triangle(
    children: list[tuple[tuple[Fraction, Fraction], ...]], /
) -> bool:
    """Rational oriented-chain proof that positively oriented children tile the unit triangle.

    Interior edges must cancel and the remaining boundary chain must be exactly
    the oriented unit-triangle boundary. A planar 2-chain with zero boundary
    vanishes, so the positively oriented children cover the root exactly once.
    """
    net: dict[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]], int] = {}
    for corners in children:
        for k in range(3):
            first, second = corners[k], corners[(k + 1) % 3]
            net[(first, second)] = net.get((first, second), 0) + 1
            net[(second, first)] = net.get((second, first), 0) - 1
    if any(count > 1 for count in net.values()):
        return False
    spans: tuple[list[tuple[Fraction, Fraction]], ...] = ([], [], [])
    for (first, second), count in net.items():
        if count != 1:
            continue
        for edge, edge_spans in zip(_REFERENCE_EDGES, spans, strict=True):
            start, end = _edge_parameter(edge, first), _edge_parameter(edge, second)
            if start is not None and end is not None and start < end:
                edge_spans.append((start, end))
                break
        else:
            return False
    for edge_spans in spans:
        position = Fraction(0)
        for start, end in sorted(edge_spans):
            if start != position:
                return False
            position = end
        if position != 1:
            return False
    return True


def _l1_upper(first: np.ndarray, second: tuple[Fraction, ...], /) -> Fraction:
    return sum(
        (abs(Fraction(float(a)) - b) for a, b in zip(first, second, strict=True)),
        Fraction(0),
    )


class _SubdivisionCoverPhase(StrictModule):
    root_coordinates: np.ndarray
    root_vertex_ids: list[int]
    cell_bounds: dict[int, float]
    collapsed: list[tuple[tuple[int, ...], float]]
    complete: bool = eqx.field(static=True)


class _ChartCoverPhase(StrictModule):
    simplices: np.ndarray
    deviation_bounds: np.ndarray


def _prepare_subdivision_cover(
    root: result_owner.CellMeshingResult,
    cover: SourceBoundaryChartCover,
    proof: surface_source_support_owner.SurfaceRestrictionRows,
) -> _SubdivisionCoverPhase:
    root_coordinates = np.asarray(root.mesh.coordinates, dtype=np.float64)
    root_vertex_ids = np.asarray(root.mesh.vertex_global_ids, dtype=np.int64).tolist()
    # Chart-cover pieces carry only physical corners; they bind root mesh
    # entities exactly as the fidelity certifier binds them to mesh maps.
    vertex_keys: dict[tuple[float, ...], int | None] = {}
    for identifier, point in zip(root_vertex_ids, root_coordinates.tolist(), strict=True):
        key = tuple(point)
        vertex_keys[key] = identifier if key not in vertex_keys else None
    cell_keys: dict[tuple[tuple[float, ...], ...], int | None] = {}
    for block in root.mesh.blocks:
        for identifier, vertices in zip(
            np.asarray(block.global_ids, dtype=np.int64).tolist(),
            np.asarray(block.vertices),
            strict=True,
        ):
            key = tuple(sorted(tuple(row) for row in root_coordinates[vertices].tolist()))
            cell_keys[key] = identifier if key not in cell_keys else None
    cell_bounds: dict[int, float] = {}
    collapsed: list[tuple[tuple[int, ...], float]] = []
    complete = True
    for simplex, bound in zip(
        np.asarray(cover.simplices).tolist(),
        np.asarray(cover.deviation_bounds).tolist(),
        strict=True,
    ):
        corners = sorted({tuple(row) for row in simplex})
        if len(corners) == 3:
            cell = cell_keys.get(tuple(corners))
            if cell is None:
                complete = False
                continue
            cell_bounds[cell] = max(bound, cell_bounds.get(cell, 0.0))
        else:
            ends = tuple(vertex_keys.get(corner) for corner in corners)
            if any(end is None for end in ends):
                complete = False
                continue
            collapsed.append((tuple(end for end in ends if end is not None), bound))
    children: dict[int, list[tuple[tuple[Fraction, Fraction], ...]]] = {}
    for root_id, corners in zip(
        proof.root_cell_ids.tolist(), proof.reference_corners, strict=True
    ):
        children.setdefault(root_id, []).append(corners)
    complete &= set(children) == set(cell_bounds) and all(
        _tiles_reference_triangle(value) for value in children.values()
    )
    return _SubdivisionCoverPhase(
        root_coordinates=root_coordinates,
        root_vertex_ids=root_vertex_ids,
        cell_bounds=cell_bounds,
        collapsed=collapsed,
        complete=complete,
    )


def _prepare_chart_cover_images(
    source: SurfaceChartBoundarySource,
    budget: CoordinateEnclosureBudget,
) -> _ChartCoverPhase:
    self = source
    mesh, geometry = self.deformation.target_mesh, self.deformation.target_geometry
    proxies = [
        source_fidelity_owner.fidelity_proxy(
            source_fidelity_owner.FidelityMap(
                mapped_geometry_owner.mapped_coordinate_expressions(
                    element, local, budget=budget
                ),
                2,
                row,
            ),
            budget=budget,
        )
        for row, (element, local) in enumerate(
            mapped_geometry_owner.mapped_geometry_cells(mesh, geometry, budget=budget)
        )
    ]
    # whole material-chart coverage carries every old map point to a
    # target map point within the certified displacement bound. Compose
    # that correspondence with the predecessor's two-sided source result
    # and the rational target-map-minus-chord enclosures; do not claim equality
    # of the old and new physical surfaces.
    rounding = max((Fraction(proxy[2]) for proxy in proxies), default=Fraction(0))
    chord = max((Fraction(proxy[1]) for proxy in proxies), default=Fraction(0))
    forward = max(
        (
            Fraction(float(value))
            for value in np.asarray(self.deformation.target_fidelity_bounds)
        ),
        default=Fraction(0),
    )
    if isinstance(
        self.deformation, sphere_deformation_owner.PreparedSphereChartDeformation
    ):
        # The declared target radial atlas independently covers the source
        # declared sphere. Its pointwise full-map fidelity bound is
        # two-sided; old-to-new displacement is a different result and
        # must not be added to that independent source error.
        atlas = self.deformation.target_atlas
        atlas.require_bound(atlas.domain, mesh, geometry, atlas.coordinate_contract)
        if (
            atlas.coverage_status != "certified"
            or self.deformation.target_domain_coverage != "certified"
        ):
            raise ValueError(
                "SurfaceChartBoundarySource.deformation.target_atlas must certify the whole source material domain; got uncertified atlas or target coverage"
            )
        bound = surface_source_support_owner.upper(forward + chord + rounding)
    else:
        backward = Fraction(self.predecessor_fidelity.source_to_mesh_upper) + Fraction(
            self.deformation.maximum_displacement_bound
        )
        bound = surface_source_support_owner.upper(
            max(forward, backward) + chord + rounding
        )
    return _ChartCoverPhase(
        simplices=np.asarray([proxy[0] for proxy in proxies], dtype=np.float64),
        deviation_bounds=np.full((len(proxies),), bound, dtype=np.float64),
    )


@final
class SurfaceSubdivisionBoundarySource(StrictModule, NonTrainableState):
    """Bind a lineage subdivision chain to source chart coverage and coordinate bounds."""

    root: domain_boundary_owner.MeshingDomainBoundarySource
    support: surface_source_support_owner.PreparedSurfaceSourceSupport
    lineages: tuple[lineage_owner.MeshLineage, ...]
    targets: tuple[cell_mesh_owner.CellMesh, ...]
    geometry: cell_geometry_owner.CellGeometrySpec
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        root: domain_boundary_owner.MeshingDomainBoundarySource,
        support: surface_source_support_owner.PreparedSurfaceSourceSupport,
        lineages: tuple[lineage_owner.MeshLineage, ...],
        targets: tuple[cell_mesh_owner.CellMesh, ...],
        geometry: cell_geometry_owner.CellGeometrySpec,
        /,
    ) -> None:
        # General tuple annotations are static-only at checked boundaries.
        if any(
            not isinstance(value, lineage_owner.MeshLineage) for value in lineages
        ) or any(not isinstance(value, cell_mesh_owner.CellMesh) for value in targets):
            raise TypeError(
                "SurfaceSubdivisionBoundarySource.epochs must contain lineage and mesh records; got incompatible epoch members"
            )

        original = support.original
        if (
            not isinstance(original, result_owner.CellMeshingResult)
            or original.mesh.storage is not None
        ):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.support.original must be an accepted serial surface result; got an incompatible root carrier"
            )
        if (root.source_id, root.source_revision) != (
            support.domain.source_id,
            support.domain.source_revision,
        ) or root.domain.content_domain_id() != support.domain.content_domain_id():
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.root must bind support source identity, revision, and domain; got a different source"
            )
        if not lineages or len(lineages) != len(targets):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.targets must contain one mesh for each nonempty lineage epoch; got misaligned or empty epochs"
            )
        topology = support.root_topology_id
        for lineage, target in zip(lineages, targets, strict=True):
            if (
                lineage.source_topology_id != topology
                or lineage.target_topology_id != target.topology_id
                or target.storage is not None
            ):
                raise ValueError(
                    "SurfaceSubdivisionBoundarySource.lineages must chain serial root and target topologies; got an incompatible epoch"
                )
            topology = target.topology_id
        restriction = geometry.restriction_source
        if restriction is not None:
            if (
                restriction.source_topology_id != original.mesh.topology_id
                or restriction.source_geometry_id != cell_geometry_id(original.geometry)
            ):
                raise ValueError(
                    "SurfaceSubdivisionBoundarySource.geometry.restriction_source must retain root topology and geometry IDs; got different root ancestry"
                )
        self.root, self.support, self.lineages, self.targets, self.geometry = (
            root,
            support,
            tuple(lineages),
            tuple(targets),
            geometry,
        )
        self.source_id, self.source_revision = root.source_id, root.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.root.ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.root.boundary_samples(maximum_samples)

    def require_current(self) -> None:
        """Re-establish source support identity and every retained binding after restoration."""
        self.support.require_current()
        rebuilt = SurfaceSubdivisionBoundarySource(
            self.root, self.support, self.lineages, self.targets, self.geometry
        )
        if (rebuilt.source_id, rebuilt.source_revision) != (
            self.source_id,
            self.source_revision,
        ):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.source must retain root identity and revision; got different source metadata"
            )

    @checked
    def boundary_chart_cover(
        self,
        maximum_patches: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> SourceBoundaryChartCover:
        cover = self.root.boundary_chart_cover(maximum_patches, budget=budget)
        if cover.semantics != "certified" or not cover.complete:
            return cover

        root = self.support.original
        if not isinstance(root, result_owner.CellMeshingResult):
            raise ValueError(
                "SurfaceSubdivisionBoundarySource.support.original must retain an accepted serial result; got a different root kind"
            )
        proof, epochs = boundary_operations_owner.subdivision_proof(
            self.support,
            self.lineages,
            self.targets,
            self.geometry,
            budget=budget,
        )
        findings = list(cover.findings)
        if proof.cell_ids.size + cover.simplices.shape[0] > maximum_patches:
            findings.append(
                MeshCertificateFinding(
                    "source_chart_cover_capacity", "unresolved", "mesh"
                )
            )
            return SourceBoundaryChartCover(
                cover.simplices,
                cover.deviation_bounds,
                "sampled",
                False,
                self.source_id,
                self.source_revision,
                findings=tuple(findings),
                resource_counts=cover.resource_counts,
            )
        prepared = _prepare_subdivision_cover(root, cover, proof)
        root_coordinates, root_vertex_ids = (
            prepared.root_coordinates,
            prepared.root_vertex_ids,
        )
        cell_bounds, collapsed = prepared.cell_bounds, prepared.collapsed
        complete = prepared.complete
        target = self.targets[-1]
        target_coordinates = np.asarray(target.coordinates, dtype=np.float64)
        target_rows = {
            identifier: row
            for row, identifier in enumerate(
                np.asarray(target.vertex_global_ids, dtype=np.int64).tolist()
            )
        }
        images: list[np.ndarray] = []
        bounds: list[float] = []
        for vertices, root_id, deviation in zip(
            proof.vertex_ids,
            proof.root_cell_ids.tolist(),
            proof.restriction_deviations.tolist(),
            strict=True,
        ):
            if root_id not in cell_bounds:
                continue
            images.append(
                target_coordinates[[target_rows[vertex] for vertex in vertices]]
            )
            bounds.append(
                surface_source_support_owner.upper(
                    Fraction(cell_bounds[root_id]) + Fraction(deviation)
                )
            )
        for ends, bound in collapsed:
            chain = self._collapsed_chain(
                ends,
                epochs,
                root_coordinates,
                root_vertex_ids,
                target_coordinates,
                target_rows,
            )
            if chain is None:
                complete = False
                continue
            for first, second, deviation in chain:
                images.append(
                    np.stack(
                        (
                            target_coordinates[first],
                            target_coordinates[first],
                            target_coordinates[second],
                        )
                    )
                )
                bounds.append(
                    surface_source_support_owner.upper(Fraction(bound) + deviation)
                )
        if not complete:
            findings.append(
                MeshCertificateFinding(
                    "source_trim_coverage_premise", "unresolved", "mesh"
                )
            )
        return SourceBoundaryChartCover(
            np.stack(images) if images else np.empty((0, 3, 3)),
            np.asarray(bounds, dtype=np.float64),
            "certified" if complete else "sampled",
            complete,
            self.source_id,
            self.source_revision,
            findings=tuple(findings),
            resource_counts=cover.resource_counts,
        )

    @staticmethod
    def _collapsed_chain(
        ends: tuple[int, ...],
        epochs: tuple[Mapping[int, boundary_operations_owner.SurfaceVertexWeights], ...],
        root_coordinates: np.ndarray,
        root_vertex_ids: list[int],
        target_coordinates: np.ndarray,
        target_rows: Mapping[int, int],
        /,
    ) -> list[tuple[int, int, Fraction]] | None:
        """Target rows and endpoint deviations of a collapsed root wedge's split segment."""
        if any(end not in target_rows for end in ends):
            return None
        if len(ends) == 1:
            row = target_rows[ends[0]]
            return [(row, row, Fraction(0))]
        chain: list[tuple[Fraction, int]] = [
            (Fraction(0), ends[0]),
            (Fraction(1), ends[1]),
        ]
        for weights in epochs:
            splits: dict[frozenset[int], list[tuple[int, dict[int, Fraction]]]] = {}
            for vertex, support in weights.items():
                shares = dict(support)
                splits.setdefault(frozenset(shares), []).append((vertex, shares))
            inserted = list(chain)
            for (first_t, first), (second_t, second) in zip(
                chain[:-1], chain[1:], strict=True
            ):
                for vertex, shares in splits.get(frozenset((first, second)), ()):
                    inserted.append(
                        (first_t + shares[second] * (second_t - first_t), vertex)
                    )
            chain = sorted(inserted)
        if any(vertex not in target_rows for _, vertex in chain):
            return None
        start, end = (root_coordinates[root_vertex_ids.index(value)] for value in ends)
        points: list[tuple[int, Fraction]] = []
        for parameter, vertex in chain:
            exact = tuple(
                (1 - parameter) * Fraction(float(a)) + parameter * Fraction(float(b))
                for a, b in zip(start, end, strict=True)
            )
            row = target_rows[vertex]
            points.append((row, _l1_upper(target_coordinates[row], exact)))
        return [
            (first[0], second[0], max(first[1], second[1]))
            for first, second in zip(points[:-1], points[1:], strict=True)
        ]


@final
class SurfaceChartBoundarySource(StrictModule, NonTrainableState):
    """Bind a surface chart epoch to predecessor source fidelity and map correspondence."""

    root: (
        domain_boundary_owner.MeshingDomainBoundarySource
        | SurfaceSubdivisionBoundarySource
    )
    support: surface_source_support_owner.PreparedSurfaceSourceSupport
    deformation: (
        chart_deformation_owner.PreparedSurfaceChartDeformation
        | sphere_deformation_owner.PreparedSphereChartDeformation
    )
    predecessor_fidelity: source_fidelity_owner.SourceFidelityCertificate

    @checked
    def __init__(
        self,
        root: domain_boundary_owner.MeshingDomainBoundarySource
        | SurfaceSubdivisionBoundarySource,
        support: surface_source_support_owner.PreparedSurfaceSourceSupport,
        deformation: chart_deformation_owner.PreparedSurfaceChartDeformation
        | sphere_deformation_owner.PreparedSphereChartDeformation,
        predecessor_fidelity: source_fidelity_owner.SourceFidelityCertificate,
        /,
    ) -> None:

        support.require_current()
        deformation.require_bound(
            deformation.source_mesh,
            deformation.source_geometry,
            deformation.target_mesh,
            deformation.target_geometry,
        )
        predecessor_fidelity.binding.require(
            deformation.source_mesh, deformation.source_geometry
        )
        if (
            predecessor_fidelity.status != "certified"
            or predecessor_fidelity.mesh_to_source_semantics != "certified"
            or predecessor_fidelity.source_to_mesh_semantics != "certified"
        ):
            raise ValueError(
                "SurfaceChartBoundarySource.predecessor_fidelity must certify both continuous source bounds; got uncertified evidence"
            )
        if (root.source_id, root.source_revision) != (
            support.domain.source_id,
            support.domain.source_revision,
        ) or deformation.domain_id != support.domain.domain_id:
            raise ValueError(
                "SurfaceChartBoundarySource.deformation must bind support source identity, revision, and domain; got a different source"
            )
        self.root, self.support, self.deformation = root, support, deformation
        self.predecessor_fidelity = predecessor_fidelity

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
        self.support.require_current()
        self.deformation.require_bound(
            self.deformation.source_mesh,
            self.deformation.source_geometry,
            self.deformation.target_mesh,
            self.deformation.target_geometry,
        )
        self.predecessor_fidelity.binding.require(
            self.deformation.source_mesh, self.deformation.source_geometry
        )

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.root.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.root.boundary_samples(maximum_samples)

    @checked
    def boundary_chart_cover(
        self,
        maximum_patches: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> SourceBoundaryChartCover:
        self.require_current()

        original = self.predecessor_fidelity.chart_coverage
        exact_predecessor = (
            original is None
            and self.predecessor_fidelity.mesh_to_source_upper == 0.0
            and self.predecessor_fidelity.mesh_to_source_lower == 0.0
            and self.predecessor_fidelity.source_to_mesh_upper == 0.0
            and self.predecessor_fidelity.source_to_mesh_lower == 0.0
            and self.deformation.source_domain_coverage == "certified"
            and self.deformation.target_domain_coverage == "certified"
        )
        if not exact_predecessor:
            if original is None:
                original = self.root.boundary_chart_cover(maximum_patches, budget=budget)
            elif (
                original.source_id,
                original.source_revision,
            ) != (self.source_id, self.source_revision):
                raise ValueError(
                    "SurfaceChartBoundarySource.predecessor_fidelity.chart_coverage must match source identity and revision; got a different source cover"
                )
            if not original.complete or original.semantics != "certified":
                return original
            findings = original.findings
            resource_counts = original.resource_counts
        else:
            findings = ()
            resource_counts = ()
        mesh = self.deformation.target_mesh

        if mesh.entity_set(2).count > maximum_patches:
            return SourceBoundaryChartCover(
                np.empty((0, 3, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                "sampled",
                False,
                self.source_id,
                self.source_revision,
                findings=(
                    MeshCertificateFinding(
                        "source_chart_cover_capacity", "unresolved", "mesh"
                    ),
                ),
            )
        prepared = _prepare_chart_cover_images(self, budget)
        return SourceBoundaryChartCover(
            prepared.simplices,
            prepared.deviation_bounds,
            "certified",
            True,
            self.source_id,
            self.source_revision,
            findings=findings,
            resource_counts=resource_counts,
        )


__all__ = [
    "SurfaceChartBoundarySource",
    "SurfaceSubdivisionBoundarySource",
]
