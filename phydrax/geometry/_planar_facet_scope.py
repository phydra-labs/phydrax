#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify selected target facets against labeled planar source fragments.

Mesh certificate checks
-----------------------
[
  {
    "owner": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
    "checks": [
      "scope_extra_source_fragments",
      "scope_incomplete_source_fragments"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every positive physical overlap incident to a selected target or source fragment remains inside the selected bipartite support."
      }
    ],
    "conclusion": "Extra source support or omitted target support violates scope closure.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "owner": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
    "checks": [
      "scope_target_overlap_premise",
      "scope_source_overlap_premise",
      "scope_domain_coverage_premise",
      "scope_owner_local_premise",
      "scope_physical_overlap_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Full-domain coverage is certified and exhaustive; source and target supports are present, physical, and represented by the supported global mesh route."
      }
    ],
    "conclusion": "Missing binding support or an unsupported coverage/storage route leaves the scoped equality unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
    "checks": [
      "scope_source_overlap"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Distinct selected coplanar source fragments have disjoint relative interiors."
      }
    ],
    "conclusion": "Positive projected intersection area violates the source-fragment partition.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "owner": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
    "checks": [
      "scope_candidate_capacity",
      "scope_coordinate_source"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Scoped candidate counts fit the request allowance."
      },
      {
        "tag": "X",
        "statement": "Each target cell supplies a supported rational coordinate law."
      }
    ],
    "conclusion": "Exceeded candidate capacity or an unsupported coordinate source leaves the scoped equality unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
    "checks": [
      "scope_continuous_containment",
      "scope_exact_projected_measure"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each oriented facet chart lies in a selected planar source union, and its projected integral is assigned once."
      },
      {
        "tag": "X",
        "statement": "The assigned projected integrals sum to every source group measure."
      }
    ],
    "conclusion": "Unproved continuous containment or unequal projected measure leaves source-scope equality unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._planar_facet_scope.certify_planar_facet_source_scope",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
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
from jax.typing import ArrayLike

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    facet_entity_ids,
    mesh_facets,
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from phydrax.geometry._planar_primitives import (
    plane_key,
    project,
    rational_points,
    signed_measure,
)
from phydrax.typing import checked

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ._domain_coverage import DomainCoverageCertificate, PiecewiseLinearDomain
from ._mapped_coverage import face_charts, SubdivisionLedger
from ._mesh_certificates import (
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._planar_coverage import mapped_containment, source_groups, SourceGroup


@final
class PlanarFacetSourceScopeCertificate(StrictModule, NonTrainableState):
    """Bind zero deviation to an explicitly labeled source-facet scope.

    Physical overlap support transfers the domain area partition to the selected
    facets. Candidate support is checked by continuous containment and projected
    integration in the source plane.
    """

    binding: MeshCertificateBinding
    domain_id: str = eqx.field(static=True)
    coverage_certificate_id: str = eqx.field(static=True)
    source_face_entity_set_id: str = eqx.field(static=True)
    source_face_fragments: tuple[tuple[int, tuple[int, ...]], ...] = eqx.field(
        static=True
    )
    target_facet_global_ids: tuple[int, ...] = eqx.field(static=True)
    source_fragment_projected_measures: tuple[
        tuple[int, tuple[int, ...], int, int], ...
    ] = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    distance_upper: float | None = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        domain_id: str,
        coverage: DomainCoverageCertificate,
        source_face_entity_set_id: str,
        source_face_fragments: tuple[tuple[int, tuple[int, ...]], ...],
        target_facet_global_ids: tuple[int, ...],
        source_fragment_projected_measures: tuple[
            tuple[int, tuple[int, ...], int, int], ...
        ],
        findings: tuple[MeshCertificateFinding, ...],
        /,
    ) -> None:
        self.binding = binding
        self.domain_id = domain_id
        self.coverage_certificate_id = coverage.certificate_id
        self.source_face_entity_set_id = source_face_entity_set_id
        self.source_face_fragments = source_face_fragments
        self.target_facet_global_ids = target_facet_global_ids
        self.source_fragment_projected_measures = source_fragment_projected_measures
        self.status = certificate_status(findings)
        self.findings = findings
        self.distance_upper = 0.0 if self.status == "certified" else None
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "planar-facet-source-scope-certificate",
                "binding": binding.binding_id,
                "domain": domain_id,
                "coverage": coverage.certificate_id,
                "source_face_entity_set": source_face_entity_set_id,
                "source_fragment_projected_measures": source_fragment_projected_measures,
                "source_face_fragments": source_face_fragments,
                "target_facets": target_facet_global_ids,
                "status": self.status,
                "findings": tuple(value.finding_id for value in findings),
                "distance_upper": self.distance_upper,
            }
        )


def _source_map(
    domain: PiecewiseLinearDomain, mapping: Mapping[int, tuple[int, ...]]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not isinstance(mapping, Mapping):
        raise TypeError(
            f"certify_planar_facet_source_scope.source_face_fragments must be a source-label mapping; got {type(mapping).__name__}"
        )
    result = []
    for tag, rows in mapping.items():
        if isinstance(tag, bool) or not isinstance(tag, (int, np.integer)):
            raise TypeError(
                f"certify_planar_facet_source_scope.source_face_fragments must use integer source tags; got tag={tag!r}"
            )
        if tag < 0:
            raise ValueError(
                f"certify_planar_facet_source_scope.source_face_fragments must use nonnegative source tags; got tag={tag}"
            )
        if not isinstance(rows, tuple):
            raise TypeError(
                f"certify_planar_facet_source_scope.source_face_fragments must use tuple row IDs; got {type(rows).__name__}"
            )
        if not rows or any(
            isinstance(row, bool) or not isinstance(row, (int, np.integer))
            for row in rows
        ):
            raise ValueError(
                f"certify_planar_facet_source_scope.source_face_fragments must provide nonempty integer row IDs; got tag={tag!r}, rows={rows!r}"
            )
        fragments = tuple(sorted(set(int(row) for row in rows)))
        if fragments[0] < 0 or fragments[-1] >= domain.facets.shape[0]:
            raise ValueError(
                f"certify_planar_facet_source_scope.source_face_fragments must index the bound domain; got row range={fragments[0]}..{fragments[-1]}"
            )
        result.append((int(tag), fragments))
    if not result:
        raise ValueError(
            "certify_planar_facet_source_scope.source_face_fragments must be nonempty; got empty mapping"
        )
    return tuple(sorted(result))


def _exact_support(
    state: EmbeddingState,
    coverage: DomainCoverageCertificate,
    selected: set[int],
    fragments: set[int],
) -> bool:
    rows = tuple(
        row
        for row in coverage.facet_source_overlaps
        if row[0] in selected or row[1] in fragments
    )
    if any(row[5:] != ("exact", "physical") for row in rows):
        return False
    targets = {row[0] for row in rows if row[1] in fragments}
    sources = {row[1] for row in rows if row[0] in selected}
    extra = sources - fragments
    missing = targets - selected
    absent_targets = selected - {row[0] for row in rows}
    absent_sources = fragments - {row[1] for row in rows}
    if extra:
        state.add(
            "scope_extra_source_fragments", "violated", "source_facet", tuple(extra)
        )
    if missing:
        state.add(
            "scope_incomplete_source_fragments", "violated", "facet", tuple(missing)
        )
    if absent_targets:
        state.add(
            "scope_target_overlap_premise", "unresolved", "facet", tuple(absent_targets)
        )
    if absent_sources:
        state.add(
            "scope_source_overlap_premise",
            "unresolved",
            "source_facet",
            tuple(absent_sources),
        )
    # Positive physical overlaps partition each covered source fragment.
    # Closing this support on the selected targets transfers that partition
    # without summing outward-rounded reports.
    return True


class _ScopeCell(StrictModule):
    kind: str
    coordinates: tuple[algebra.Expression, ...] | None
    vertices: tuple[int, ...]


class _ScopeCells(StrictModule):
    cells: dict[int, _ScopeCell]


def _prepare_scope_cells(
    mesh: CellMesh, geometry: CellGeometrySpec, *, budget: CoordinateEnclosureBudget
) -> _ScopeCells:
    elements, routes, _ = geometry.resolve(mesh)
    values = geometry.source_coordinates(budget=budget)
    cells: dict[int, _ScopeCell] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for row, local in zip(np.asarray(block.vertices), local_values, strict=True):
            cells[cursor] = _ScopeCell(
                block.cell_kind,
                algebra.coordinate_expressions(element, local, budget=budget),
                tuple(int(value) for value in row),
            )
            cursor += 1
    return _ScopeCells(cells)


def _scope_chart_measure(
    chart: tuple[algebra.Expression, ...],
    chart_domain: algebra.BernsteinDomain,
    source: tuple[SourceGroup, ...],
    covered: list[Fraction],
    limits: MeshCertificateLimits,
    work: SubdivisionLedger,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    for index, group in enumerate(source):
        for own, other in (group.regions, group.regions[::-1]):
            outcome, measure, _ = mapped_containment(
                chart,
                chart_domain,
                group,
                own,
                other,
                limits.maximum_bernstein_nodes,
                limits.maximum_subdivision_depth,
                limits.maximum_subdivision_pieces,
                limits.maximum_candidate_pairs,
                work,
                budget=budget,
            )
            if outcome == "proven":
                covered[index] += measure
                return True
    return False


def _mapped_scope(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    selected: set[int],
    fragments: set[int],
    occurrences: dict[int, int],
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    ordered = tuple(sorted(fragments))
    source, overlaps, used, exceeded = source_groups(
        domain.vertices,
        domain.facets[np.asarray(ordered)],
        domain.facet_regions[np.asarray(ordered)],
        limits.maximum_candidate_pairs,
        budget=budget,
    )
    if overlaps:
        state.add(
            "scope_source_overlap",
            "violated",
            "source_facet",
            tuple(ordered[index] for pair in overlaps for index in pair),
        )
    if exceeded:
        state.add("scope_candidate_capacity", "unresolved", "mesh")
        return
    facets = mesh_facets(mesh, budget=budget)
    prepared = _prepare_scope_cells(mesh, geometry, budget=budget)
    covered = [Fraction(0) for _ in source]
    work = SubdivisionLedger(candidate_pairs=used)
    for entity in sorted(selected):
        occurrence = occurrences[entity]
        cell = prepared.cells[int(facets.cells[occurrence])]
        kind, polynomial, vertices = cell.kind, cell.coordinates, cell.vertices
        if polynomial is None:
            state.add("scope_coordinate_source", "unresolved", "facet", (entity,))
            continue
        face = tuple(
            vertices.index(int(vertex))
            for vertex in facets.rows[occurrence]
            if vertex >= 0
        )
        for chart, chart_domain in face_charts(polynomial, kind, face, budget=budget):
            if not _scope_chart_measure(
                chart, chart_domain, source, covered, limits, work, budget=budget
            ):
                state.add(
                    "scope_continuous_containment", "unresolved", "facet", (entity,)
                )
    for actual, group in zip(covered, source, strict=True):
        expected = sum(group.measures, Fraction(0))
        if actual != expected:
            state.add(
                "scope_exact_projected_measure",
                "unresolved",
                "source_facet",
                tuple(ordered[index] for index in group.members),
            )


@checked
def certify_planar_facet_source_scope(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    target_facet_global_ids: ArrayLike,
    source_face_fragments: Mapping[int, tuple[int, ...]],
    /,
    *,
    source_face_entity_set_id: str,
    coverage: DomainCoverageCertificate,
    limits: MeshCertificateLimits | None = None,
    budget: CoordinateEnclosureBudget,
) -> PlanarFacetSourceScopeCertificate:
    """Certify equality of selected facet images and labeled source fragments."""
    coverage.binding.require(mesh, geometry)
    if (
        coverage.domain_id != domain.domain_id
        or coverage.binding.source_id != domain.source_id
        or coverage.binding.source_revision != domain.source_revision
    ):
        raise ValueError(
            "certify_planar_facet_source_scope.coverage must bind the supplied source domain; got mismatched binding"
        )
    limits_ = MeshCertificateLimits() if limits is None else limits
    if limits_.limits_id != coverage.binding.limits_id:
        raise ValueError(
            "certify_planar_facet_source_scope.limits must match the domain coverage binding; got mismatched limits"
        )
    namespace = canonical_identifier(
        source_face_entity_set_id, "source_face_entity_set_id"
    )
    source_map = _source_map(domain, source_face_fragments)
    requested = np.asarray(target_facet_global_ids)
    if (
        requested.ndim != 1
        or requested.size == 0
        or not np.issubdtype(requested.dtype, np.integer)
    ):
        raise ValueError(
            f"certify_planar_facet_source_scope.target_facet_global_ids must be a nonempty integer vector; got shape={requested.shape}, dtype={requested.dtype}"
        )
    target = tuple(sorted(set(int(value) for value in requested)))
    state = EmbeddingState([], ["planar_facet_source_scope"])
    if (
        coverage.status != "certified"
        or coverage.covered_source_facet_count != coverage.source_facet_count
    ):
        state.add("scope_domain_coverage_premise", "unresolved", "mesh")
    elif mesh.storage is not None:
        state.add("scope_owner_local_premise", "unresolved", "mesh")
    else:
        facets = mesh_facets(mesh, budget=budget)
        entities = facet_entity_ids(mesh, facets.rows)
        occurrences = {int(entity): index for index, entity in enumerate(entities)}
        if any(entity not in occurrences for entity in target):
            raise ValueError(
                "certify_planar_facet_source_scope.target_facet_global_ids must belong to the bound mesh; got unknown facet IDs"
            )
        selected = set(target)
        fragments = {row for _, rows in source_map for row in rows}
        known = {row[0] for row in coverage.facet_source_overlaps if row[6] == "physical"}
        if selected - known:
            state.add(
                "scope_target_overlap_premise",
                "unresolved",
                "facet",
                tuple(selected - known),
            )
        elif not _exact_support(state, coverage, selected, fragments):
            if any(row[6] != "physical" for row in coverage.facet_source_overlaps):
                state.add("scope_physical_overlap_premise", "unresolved", "mesh")
            else:
                _mapped_scope(
                    state,
                    mesh,
                    geometry,
                    domain,
                    selected,
                    fragments,
                    occurrences,
                    limits_,
                    budget=budget,
                )
    areas = []
    for source in sorted({row for _, rows in source_map for row in rows}):
        points = rational_points(domain.vertices[domain.facets[source]], budget=budget)
        key = plane_key(points, budget=budget)
        if key is None:
            raise ValueError(
                "certify_planar_facet_source_scope.source_face_fragments must define nondegenerate planes; got degenerate fragment"
            )
        _, axes, _, _ = key
        measure = rational_coefficient(
            abs(signed_measure(project(points, axes, budget=budget), budget=budget)),
            role="planar_facet_scope.source_measure",
        )
        areas.append((source, axes, measure.numerator, measure.denominator))
    return PlanarFacetSourceScopeCertificate(
        coverage.binding,
        domain.domain_id,
        coverage,
        namespace,
        source_map,
        target,
        tuple(areas),
        tuple(state.findings),
    )
