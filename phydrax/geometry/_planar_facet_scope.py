#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Selected target-facet equality to explicitly labeled planar source fragments."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ._mapped_coverage import face_charts, SubdivisionLedger
from ._mesh_certificates import (
    _certificate_status,
    _EmbeddingState,
    _facet_entity_ids,
    _mesh_facets,
    DomainCoverageCertificate,
    MeshCertificateBinding,
    MeshCertificateFinding,
    MeshCertificateLimits,
    MeshCertificateStatus,
    PiecewiseLinearDomain,
)
from ._planar_coverage import (
    mapped_containment,
    plane_key,
    project,
    rational_points,
    signed_measure,
    source_groups,
)


@final
class PlanarFacetSourceScopeCertificate(StrictModule, NonTrainableState):
    """Zero deviation only for a complete, independently bound source-facet scope.

    Source tags are explicit labels in ``source_face_entity_set_id``; they are
    not inferred from equal integers in CAD and declared-fragment namespaces.
    Exact physical overlap support transfers the whole-domain exact-area proof.
    Candidate overlap rows instead require a new continuous scoped containment
    and exact projected-integral proof. No original CAD equivalence is implied.
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
        if not isinstance(binding, MeshCertificateBinding) or not isinstance(
            coverage, DomainCoverageCertificate
        ):
            raise TypeError(
                "Facet scope evidence requires mesh binding and domain coverage."
            )
        if not all(isinstance(value, MeshCertificateFinding) for value in findings):
            raise TypeError("findings must contain MeshCertificateFinding values.")
        self.binding = binding
        self.domain_id = domain_id
        self.coverage_certificate_id = coverage.certificate_id
        self.source_face_entity_set_id = source_face_entity_set_id
        self.source_face_fragments = source_face_fragments
        self.target_facet_global_ids = target_facet_global_ids
        self.source_fragment_projected_measures = source_fragment_projected_measures
        self.status = _certificate_status(findings)
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
        raise TypeError("source_face_fragments must be an explicit source-label mapping.")
    result = []
    for tag, rows in mapping.items():
        if isinstance(tag, bool) or not isinstance(tag, (int, np.integer)):
            raise TypeError(
                "Source face tags must be integer identifiers in the declared source entity set."
            )
        if tag < 0:
            raise ValueError(
                "Generated caps and unknown source faces cannot be original source tags."
            )
        if not isinstance(rows, tuple):
            raise TypeError("Source fragment row IDs must be canonical tuples.")
        if not rows or any(
            isinstance(row, bool) or not isinstance(row, (int, np.integer))
            for row in rows
        ):
            raise ValueError(
                "Every source face tag requires explicit authoritative fragment row IDs."
            )
        fragments = tuple(sorted(set(int(row) for row in rows)))
        if fragments[0] < 0 or fragments[-1] >= domain.facets.shape[0]:
            raise ValueError("Source fragment row IDs must belong to the bound domain.")
        result.append((int(tag), fragments))
    if not result:
        raise ValueError("A selected source face scope cannot be empty.")
    return tuple(sorted(result))


def _exact_support(
    state: _EmbeddingState,
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
    # Every exact overlap row represents a strictly positive clipped measure.
    # The certified full-domain ledger proves exact exhaustive fragment area,
    # nonoverlap and containment. Excluding all unselected support transfers
    # that exact partition to this scope; outward float reports are not summed.
    return True


def _mapped_scope(
    state: _EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    selected: set[int],
    fragments: set[int],
    occurrences: dict[int, int],
    limits: MeshCertificateLimits,
) -> None:
    ordered = tuple(sorted(fragments))
    source, overlaps, used, exceeded = source_groups(
        domain.vertices,
        domain.facets[np.asarray(ordered)],
        domain.facet_regions[np.asarray(ordered)],
        limits.maximum_candidate_pairs,
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
    facets = _mesh_facets(mesh)
    elements, routes, _ = geometry.resolve(mesh)
    values = geometry.source_coordinates()
    cells: dict[
        int, tuple[str, tuple[algebra.Expression, ...] | None, tuple[int, ...]]
    ] = {}
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for row, local in zip(np.asarray(block.vertices), local_values, strict=True):
            cells[cursor] = (
                block.cell_kind,
                algebra.coordinate_expressions(element, local),
                tuple(int(value) for value in row),
            )
            cursor += 1
    covered = [Fraction(0) for _ in source]
    work = SubdivisionLedger(candidate_pairs=used)
    for entity in sorted(selected):
        occurrence = occurrences[entity]
        kind, polynomial, vertices = cells[int(facets.cells[occurrence])]
        if polynomial is None:
            state.add("scope_coordinate_source", "unresolved", "facet", (entity,))
            continue
        face = tuple(
            vertices.index(int(vertex))
            for vertex in facets.rows[occurrence]
            if vertex >= 0
        )
        for chart, chart_domain in face_charts(polynomial, kind, face):
            matched = False
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
                    )
                    if outcome == "proven":
                        covered[index] += measure
                        matched = True
                        break
                if matched:
                    break
            if not matched:
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
) -> PlanarFacetSourceScopeCertificate:
    """Prove selected actual facet images equal explicitly labeled source fragments."""
    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("Facet scopes require CellMesh and CellGeometrySpec.")
    if not isinstance(domain, PiecewiseLinearDomain) or not isinstance(
        coverage, DomainCoverageCertificate
    ):
        raise TypeError(
            "Facet scopes require PiecewiseLinearDomain and DomainCoverageCertificate."
        )
    coverage.binding.require(mesh, geometry)
    if (
        coverage.domain_id != domain.domain_id
        or coverage.binding.source_id != domain.source_id
        or coverage.binding.source_revision != domain.source_revision
    ):
        raise ValueError(
            "Facet scope coverage must bind the actual independent source domain."
        )
    limits_ = MeshCertificateLimits() if limits is None else limits
    if not isinstance(limits_, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits or None.")
    if limits_.limits_id != coverage.binding.limits_id:
        raise ValueError(
            "Facet scope limits must match the whole-domain coverage binding."
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
        raise ValueError("Target facet global IDs must be one nonempty integer vector.")
    target = tuple(sorted(set(int(value) for value in requested)))
    state = _EmbeddingState([], ["planar_facet_source_scope"])
    if (
        coverage.status != "certified"
        or coverage.covered_source_facet_count != coverage.source_facet_count
    ):
        state.add("scope_domain_coverage_premise", "unresolved", "mesh")
    elif mesh.storage is not None:
        state.add("scope_owner_local_premise", "unresolved", "mesh")
    else:
        facets = _mesh_facets(mesh)
        entities = _facet_entity_ids(mesh, facets.rows)
        occurrences = {int(entity): index for index, entity in enumerate(entities)}
        if any(entity not in occurrences for entity in target):
            raise ValueError("Target facet IDs must belong to the actual bound mesh.")
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
                )
    areas = []
    for source in sorted({row for _, rows in source_map for row in rows}):
        points = rational_points(domain.vertices[domain.facets[source]])
        key = plane_key(points)
        if key is None:
            raise ValueError(
                "Source scope fragments must define nondegenerate exact planes."
            )
        _, axes, _, _ = key
        measure = abs(signed_measure(project(points, axes)))
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
