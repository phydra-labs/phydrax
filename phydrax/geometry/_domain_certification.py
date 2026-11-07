#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Input-bound producer for exact oriented source-domain coverage."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import final, Literal, NamedTuple

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import ExactScalar
from ..typing import Dim, HostInt64
from ._affine_plc_domain import AffinePlcDomain
from ._approximate_source_coverage import ExactCadSourceBoundary
from ._domain_coverage import PiecewiseLinearDomain
from ._domain_preparation import CellCoverageAssignment, DomainCoveragePreparation
from ._embedding_state import (
    MeshCertificateEntityKind,
    MeshCertificateFinding,
    MeshFindingStatus,
)
from ._global_embedding import GlobalEmbeddingCertificate, ValidatedGlobalEmbedding
from ._mapped_reference_domain import MappedReferenceDomain
from ._mesh_certificates import (
    _same_tree_content,
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._meshing_domain import MeshingDomain
from ._planar_coverage import FacetOverlap, PlcSourcePoints, PlcSourcePointWitness
from ._rational import fraction_up
from ._source_boundary_records import SourceBoundaryDistanceWitness, SourceBoundaryQuery
from ._source_fidelity import _AdaptiveImplicitBoundaryCarrier
from ._source_fidelity_certificate import SourceFidelityCertificate
from .implicit._adaptive_discovery import AdaptiveImplicitSurface
from .implicit._analytic_profile import AnalyticImplicitProfile
from .implicit._restricted_delaunay_coverage import (
    RestrictedDelaunayBoundaryWitness,
    RestrictedDelaunayCoverageTransportWitness,
    RestrictedDelaunayCoverageUnavailable,
    RestrictedDelaunayCoverageWitness,
    validate_restricted_delaunay_coverage_witness,
)


type CoverageDomain = (
    PiecewiseLinearDomain
    | AffinePlcDomain
    | MappedReferenceDomain
    | MeshingDomain
    | AdaptiveImplicitSurface
    | AnalyticImplicitProfile
)
type SourceCoverageWitness = RestrictedDelaunayCoverageWitness | PlcSourcePointWitness
type CoverageMethod = Literal[
    "exact_oriented_facets", "mapped_reference_charts", "source_boundary_homotopy"
]


def require_coverage_domain(value: object, /) -> CoverageDomain:
    """Admit only concrete source owners with revalidatable coverage premises."""
    if isinstance(value, AffinePlcDomain):
        value.require_current()
    if isinstance(
        value,
        (
            PiecewiseLinearDomain,
            AffinePlcDomain,
            MappedReferenceDomain,
            MeshingDomain,
            AdaptiveImplicitSurface,
            AnalyticImplicitProfile,
        ),
    ):
        return value
    raise TypeError(
        "domain must be a PLC, mapped reference, meshing, adaptive implicit, "
        "or analytic implicit domain."
    )


def coverage_region_occurrence_paths(
    domain: CoverageDomain, /
) -> tuple[tuple[str, ...], ...]:
    """Return current occurrence paths aligned with unchanged raw region IDs."""
    region_ids = tuple(domain.region_ids)
    if isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain)):
        paths = domain.region_occurrence_paths
    elif isinstance(domain, MappedReferenceDomain):
        paths = domain.reference_domain.region_occurrence_paths
    elif isinstance(domain, MeshingDomain):
        paths = domain.region_source_occurrences
    elif isinstance(domain, AnalyticImplicitProfile):
        paths = domain.region_occurrence_paths
    elif isinstance(domain, AdaptiveImplicitSurface):
        if region_ids not in ((), ("interior",)):
            raise ValueError(
                "Adaptive implicit coverage supports one uninstanced interior region."
            )
        paths = tuple(() for _ in region_ids)
    else:
        raise TypeError("domain must be a canonical CoverageDomain.")
    if len(paths) != len(region_ids):
        raise ValueError(
            "Coverage region occurrence paths must align with source region IDs."
        )
    result = tuple(tuple(path) for path in paths)
    if len(set(zip(region_ids, result, strict=True))) != len(region_ids):
        raise ValueError(
            "Coverage region IDs and occurrence paths must uniquely identify regions."
        )
    return result


def _coverage_source_revision(
    domain: CoverageDomain, source: SourceBoundaryQuery | None, /
) -> str:
    if isinstance(domain, AdaptiveImplicitSurface):
        if source is None or not isinstance(source, SourceBoundaryQuery):
            raise ValueError(
                "Adaptive implicit coverage requires its revision-bound source boundary query."
            )
        expected = domain.boundary_source
        if isinstance(source, _AdaptiveImplicitBoundaryCarrier):
            # A source carrier republishes the discovery under its own revision;
            # it binds this domain only through the discovery's boundary query.
            source.require_current()
            adaptive = source.adaptive_source
            if (adaptive.source_id, adaptive.source_revision) != (
                expected.source_id,
                expected.source_revision,
            ) or source.source_id != expected.source_id:
                raise ValueError(
                    "Implicit source query does not bind the current discovery revision."
                )
            return source.source_revision
        if (source.source_id, source.source_revision) != (
            expected.source_id,
            expected.source_revision,
        ):
            raise ValueError(
                "Implicit source query does not bind the current discovery revision."
            )
        return expected.source_revision
    if isinstance(domain, AnalyticImplicitProfile):
        if source is None or not isinstance(source, SourceBoundaryQuery):
            raise ValueError(
                "Analytic implicit coverage requires its revision-bound source boundary query."
            )
        if (source.source_id, source.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError("Analytic profile and source query identities differ.")
        return source.source_revision
    revision = domain.source_revision
    if source is not None and (
        source.source_id != domain.source_id or source.source_revision != revision
    ):
        raise ValueError("Coverage source query must bind the actual domain revision.")
    return revision


class _CoverageCellDim(Dim, minimum=1):
    """Target cells retained by one domain coverage certificate."""


class _CoverageProof(NamedTuple):
    binding: MeshCertificateBinding
    cell_regions: HostInt64[_CoverageCellDim]
    requested_region_measures: tuple[ExactScalar | None, ...]
    achieved_region_measures: tuple[ExactScalar | None, ...]
    facet_source_overlaps: tuple[FacetOverlap, ...]
    source_facet_count: int
    target_facet_count: int
    covered_source_facet_count: int
    findings: tuple[MeshCertificateFinding, ...]
    chart_certificate_id: str | None
    coverage_method: CoverageMethod
    source_coverage_id: str | None
    source_distance_upper: float | None
    source_region_degree_ids: tuple[str, ...]


class _BoundaryCoverageProof(NamedTuple):
    facet_overlaps: tuple[FacetOverlap, ...]
    target_facet_count: int
    chart_certificate_id: str | None
    source_facet_count: int
    covered_source_facet_count: int
    coverage_method: CoverageMethod
    source_coverage_id: str | None
    source_distance_upper: float | None
    source_region_degree_ids: tuple[str, ...]
    cell_measures: Mapping[int, Fraction] | None = None


def _coverage_finding(
    check: str,
    status: MeshFindingStatus,
    entity_kind: MeshCertificateEntityKind,
    entity_ids: tuple[int, ...] = (),
    /,
    *,
    detail: str,
) -> MeshCertificateFinding:
    return MeshCertificateFinding(
        check,
        status,
        entity_kind,
        entity_ids,
        detail=detail,
    )


def _reprove_embedding(
    embedding: GlobalEmbeddingCertificate,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
) -> GlobalEmbeddingCertificate:
    fresh = GlobalEmbeddingCertificate.from_stored_fields(embedding)
    if fresh.certificate_id != embedding.certificate_id or not _same_tree_content(
        embedding, fresh
    ):
        raise ValueError(
            "Global embedding does not reproduce its current mathematical witness."
        )
    fresh.binding.require(mesh, geometry)
    return fresh


def _overlap_payload(
    values: tuple[FacetOverlap, ...], /
) -> tuple[tuple[int, int, tuple[int, int], str, str, str], ...]:
    return tuple(
        (
            value.source_facet_id,
            value.target_facet_id,
            value.region_pair,
            str(value.measure),
            value.semantics,
            value.coordinate_space,
        )
        for value in values
    )


@final
class DomainCoverageCertificate(StrictModule, NonTrainableState):
    """Input-bound exact oriented chains or witnessed source homotopies.

    Region occurrence paths align with raw region IDs and distinguish repeated
    names without renaming source-authored identities.
    """

    __strict_contract__ = True

    binding: MeshCertificateBinding
    domain: CoverageDomain
    source: SourceBoundaryQuery | None
    source_coverage_witness: SourceCoverageWitness | None
    fidelity: SourceFidelityCertificate | None
    embedding: GlobalEmbeddingCertificate
    domain_id: str
    source_id: str
    source_revision: str
    region_ids: tuple[str, ...]
    region_occurrence_paths: tuple[tuple[str, ...], ...]
    cell_regions: HostInt64[_CoverageCellDim]
    embedding_certificate_id: str
    requested_region_measures: tuple[ExactScalar | None, ...] = eqx.field(static=True)
    achieved_region_measures: tuple[ExactScalar | None, ...] = eqx.field(static=True)
    facet_source_overlaps: tuple[FacetOverlap, ...] = eqx.field(static=True)
    source_facet_count: int
    target_facet_count: int
    covered_source_facet_count: int
    chart_certificate_id: str | None
    coverage_method: CoverageMethod
    source_coverage_id: str | None
    source_distance_upper: float | None
    source_region_degree_ids: tuple[str, ...]
    findings: tuple[MeshCertificateFinding, ...]
    status: MeshCertificateStatus
    certificate_id: str

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        domain: CoverageDomain,
        cell_regions: ArrayLike,
        /,
        *,
        embedding: GlobalEmbeddingCertificate,
        source: SourceBoundaryQuery | None = None,
        fidelity: SourceFidelityCertificate | None = None,
        source_coverage_witness: SourceCoverageWitness | None = None,
        chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
        _validated_parent_embedding: ValidatedGlobalEmbedding | None = None,
    ) -> None:
        if fidelity is not None and source is None:
            source = fidelity.source
        proof = _prove_coverage(
            mesh,
            geometry,
            domain,
            cell_regions,
            embedding=embedding,
            source=source,
            fidelity=fidelity,
            source_coverage_witness=source_coverage_witness,
            chart_parent=chart_parent,
            _validated_parent_embedding=_validated_parent_embedding,
        )
        self.binding = proof.binding
        self.domain = domain
        self.source = source
        self.fidelity = fidelity
        self.source_coverage_witness = source_coverage_witness
        self.embedding = embedding
        self.domain_id = domain.domain_id
        self.source_id = domain.source_id
        self.source_revision = _coverage_source_revision(domain, source)
        self.region_ids = tuple(domain.region_ids)
        self.region_occurrence_paths = coverage_region_occurrence_paths(domain)
        self.cell_regions = proof.cell_regions
        self.embedding_certificate_id = embedding.certificate_id
        self.requested_region_measures = proof.requested_region_measures
        self.achieved_region_measures = proof.achieved_region_measures
        self.facet_source_overlaps = proof.facet_source_overlaps
        self.source_facet_count = proof.source_facet_count
        self.target_facet_count = proof.target_facet_count
        self.covered_source_facet_count = proof.covered_source_facet_count
        self.chart_certificate_id = proof.chart_certificate_id
        self.coverage_method = proof.coverage_method
        self.source_coverage_id = proof.source_coverage_id
        self.source_distance_upper = proof.source_distance_upper
        self.source_region_degree_ids = proof.source_region_degree_ids
        self.findings = proof.findings
        self.status = certificate_status(proof.findings)
        if self.status == "certified":
            if len(self.source_region_degree_ids) != len(self.region_ids) or any(
                not identifier for identifier in self.source_region_degree_ids
            ):
                raise ValueError(
                    "Certified coverage must retain one computed material-degree identity per source region."
                )
            if self.coverage_method == "source_boundary_homotopy" and (
                self.source_coverage_id is None
                or self.source_distance_upper is None
                or not np.isfinite(self.source_distance_upper)
                or self.source_distance_upper < 0.0
            ):
                raise ValueError(
                    "Certified bounded coverage must retain its source homotopy and finite distance bound."
                )
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "domain-coverage-certificate",
                "binding": proof.binding.binding_id,
                "domain": self.domain_id,
                "source": (self.source_id, self.source_revision),
                "embedding": self.embedding_certificate_id,
                "regions": array_tree_fingerprint(proof.cell_regions),
                "region_ids": self.region_ids,
                "region_occurrence_paths": self.region_occurrence_paths,
                "requested_measures": tuple(
                    str(value) for value in self.requested_region_measures
                ),
                "achieved_measures": tuple(
                    None if value is None else str(value)
                    for value in self.achieved_region_measures
                ),
                "facet_overlaps": _overlap_payload(self.facet_source_overlaps),
                "source_facet_count": self.source_facet_count,
                "target_facet_count": self.target_facet_count,
                "source_query": None
                if self.source is None
                else (self.source.source_id, self.source.source_revision),
                "source_fidelity": None
                if self.fidelity is None
                else self.fidelity.certificate_id,
                "covered_source_facet_count": self.covered_source_facet_count,
                "chart_certificate": self.chart_certificate_id,
                "coverage_method": self.coverage_method,
                "source_coverage": self.source_coverage_id,
                "source_distance_upper": self.source_distance_upper,
                "source_region_degree_ids": self.source_region_degree_ids,
                "source_coverage_witness": None
                if self.source_coverage_witness is None
                else self.source_coverage_witness.witness_id,
                "findings": tuple(value.finding_id for value in proof.findings),
                "status": self.status,
            }
        )

    def require_bound(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        domain: CoverageDomain,
        cell_regions: ArrayLike,
        /,
        *,
        embedding: GlobalEmbeddingCertificate,
        source: SourceBoundaryQuery | None = None,
        fidelity: SourceFidelityCertificate | None = None,
        source_coverage_witness: SourceCoverageWitness | None = None,
    ) -> None:
        """Cheaply require the same retained inputs for normal certificate composition."""
        require_coverage_domain(domain)
        if not isinstance(embedding, GlobalEmbeddingCertificate):
            raise TypeError("embedding must be GlobalEmbeddingCertificate.")
        source_ = self.source if source is None else source
        fidelity_ = self.fidelity if fidelity is None else fidelity
        if fidelity_ is not None and source_ is None:
            source_ = fidelity_.source
        source_coverage_witness_ = (
            self.source_coverage_witness
            if source_coverage_witness is None
            else source_coverage_witness
        )
        self.binding.require(mesh, geometry)
        if (
            not _same_tree_content(domain, self.domain)
            or domain.domain_id != self.domain_id
            or domain.source_id != self.source_id
            or _coverage_source_revision(domain, source_) != self.source_revision
            or coverage_region_occurrence_paths(domain) != self.region_occurrence_paths
        ):
            raise ValueError(
                "DomainCoverageCertificate must retain the actual current source domain."
            )
        if source_ is not None:
            if not isinstance(source_, SourceBoundaryQuery):
                raise TypeError("source must implement SourceBoundaryQuery.")
            if (source_.source_id, source_.source_revision) != (
                self.source_id,
                self.source_revision,
            ):
                raise ValueError(
                    "Domain coverage source query changed its bound identity."
                )
            if self.source is None or not _same_tree_content(source_, self.source):
                raise ValueError(
                    "Domain coverage source query differs from its retained premise."
                )
        if fidelity_ is not None:
            if not isinstance(fidelity_, SourceFidelityCertificate):
                raise TypeError("fidelity must be SourceFidelityCertificate.")
            if (fidelity_.binding.source_id, fidelity_.binding.source_revision) != (
                self.source_id,
                self.source_revision,
            ):
                raise ValueError(
                    "Domain coverage fidelity binds a different source revision."
                )
            if (
                self.fidelity is None
                or fidelity_.certificate_id != self.fidelity.certificate_id
            ):
                raise ValueError(
                    "Domain coverage fidelity differs from its retained premise."
                )
        if source_coverage_witness_ is not None:
            if not isinstance(
                source_coverage_witness_,
                (
                    RestrictedDelaunayBoundaryWitness,
                    RestrictedDelaunayCoverageTransportWitness,
                    PlcSourcePointWitness,
                ),
            ):
                raise TypeError(
                    "source_coverage_witness must be a restricted-Delaunay or PLC source-point witness or None."
                )
            if (
                self.source_coverage_witness is None
                or source_coverage_witness_.witness_id
                != self.source_coverage_witness.witness_id
            ):
                raise ValueError(
                    "Domain coverage source witness differs from its retained topology premise."
                )
        regions = np.asarray(cell_regions)
        if regions.dtype.kind not in "iu" or not np.array_equal(
            regions, self.cell_regions
        ):
            raise ValueError(
                "DomainCoverageCertificate cell_regions differ from their bound dense assignment."
            )
        if embedding.certificate_id != self.embedding_certificate_id:
            raise ValueError(
                "DomainCoverageCertificate uses a different global embedding premise."
            )
        embedding.binding.require(mesh, geometry)

    def require_current(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        domain: CoverageDomain,
        cell_regions: ArrayLike,
        /,
        *,
        embedding: GlobalEmbeddingCertificate | None = None,
        source: SourceBoundaryQuery | None = None,
        fidelity: SourceFidelityCertificate | None = None,
        source_coverage_witness: SourceCoverageWitness | None = None,
        chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> None:
        """Recompute embedding and coverage before accepting a restored report."""
        current_embedding = self.embedding if embedding is None else embedding
        source_ = self.source if source is None else source
        fidelity_ = self.fidelity if fidelity is None else fidelity
        source_coverage_witness_ = (
            self.source_coverage_witness
            if source_coverage_witness is None
            else source_coverage_witness
        )
        self.require_bound(
            mesh,
            geometry,
            domain,
            cell_regions,
            embedding=current_embedding,
            source=source_,
            fidelity=fidelity_,
            source_coverage_witness=source_coverage_witness_,
        )
        fresh_embedding = _reprove_embedding(current_embedding, mesh, geometry)
        fresh = certify_domain_coverage(
            mesh,
            geometry,
            domain,
            cell_regions,
            embedding=fresh_embedding,
            source=source_,
            fidelity=fidelity_,
            source_coverage_witness=source_coverage_witness_,
            chart_parent=chart_parent,
        )
        if any(
            finding.check == "mapped_chart_parent_premise" for finding in fresh.findings
        ):
            raise ValueError(
                "Mapped child coverage requires its actual parent mesh and geometry."
            )
        if fresh.certificate_id != self.certificate_id or not _same_tree_content(
            self, fresh
        ):
            raise ValueError(
                "Domain coverage does not reproduce its current mathematical witness."
            )

    @classmethod
    def from_stored_fields(
        cls,
        value: DomainCoverageCertificate,
        domain: CoverageDomain | None = None,
        /,
        *,
        chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> DomainCoverageCertificate:
        """Cold-recompute all claims from retained mesh, map, source, and embedding."""
        if not isinstance(value, cls):
            raise TypeError("value must be DomainCoverageCertificate.")
        domain_ = value.domain if domain is None else domain
        if not _same_tree_content(domain_, value.domain):
            raise ValueError(
                "Stored coverage cannot be rebound to another source domain."
            )
        value.require_bound(
            value.binding.mesh,
            value.binding.geometry,
            domain_,
            value.cell_regions,
            embedding=value.embedding,
            source=value.source,
            fidelity=value.fidelity,
            source_coverage_witness=value.source_coverage_witness,
        )
        fresh_embedding = _reprove_embedding(
            value.embedding, value.binding.mesh, value.binding.geometry
        )
        fresh = certify_domain_coverage(
            value.binding.mesh,
            value.binding.geometry,
            domain_,
            value.cell_regions,
            embedding=fresh_embedding,
            source_coverage_witness=value.source_coverage_witness,
            source=value.source,
            fidelity=value.fidelity,
            chart_parent=chart_parent,
        )
        if any(
            finding.check == "mapped_chart_parent_premise" for finding in fresh.findings
        ):
            raise ValueError(
                "Mapped child coverage requires its actual parent mesh and geometry."
            )
        if fresh.certificate_id != value.certificate_id or not _same_tree_content(
            value, fresh
        ):
            raise ValueError(
                "Domain coverage does not reproduce its current mathematical witness."
            )
        return fresh

    @classmethod
    def from_restored_inputs(
        cls,
        value: DomainCoverageCertificate,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        embedding: GlobalEmbeddingCertificate,
        domain: CoverageDomain,
        /,
        *,
        source: SourceBoundaryQuery | None = None,
        chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    ) -> DomainCoverageCertificate:
        """Recompute stored coverage over independently restored owners.

        ``mesh``, ``geometry``, ``embedding``, ``domain``, and ``source`` replace
        the stored archive views of those inputs; every claim is re-proved and
        the recomputed certificate must reproduce the stored identity.
        """
        if not isinstance(value, cls):
            raise TypeError("value must be DomainCoverageCertificate.")
        if (
            mesh.mesh_id != value.binding.mesh_id
            or embedding.certificate_id != value.embedding_certificate_id
        ):
            raise ValueError("Restored coverage inputs bind another mesh or embedding.")
        fresh = certify_domain_coverage(
            mesh,
            geometry,
            domain,
            value.cell_regions,
            embedding=embedding,
            source_coverage_witness=value.source_coverage_witness,
            source=source,
            fidelity=value.fidelity,
            chart_parent=chart_parent,
        )
        if any(
            finding.check == "mapped_chart_parent_premise" for finding in fresh.findings
        ):
            raise ValueError(
                "Mapped child coverage requires its actual parent mesh and geometry."
            )
        if fresh.certificate_id != value.certificate_id:
            raise ValueError(
                "Domain coverage does not reproduce its current mathematical witness."
            )
        return fresh


def _partition_findings(
    values: tuple[tuple[str, int, int], ...],
    /,
    *,
    status: MeshFindingStatus = "violated",
) -> tuple[MeshCertificateFinding, ...]:
    result: list[MeshCertificateFinding] = []
    for check, first, second in values:
        source_check = check in (
            "overlapping_source_facets",
            "uncovered_boundary",
            "double_or_partial_source_coverage",
        )
        entity_kind = "source_facet" if source_check else "facet"
        entity_ids = (first,) if second < 0 else (first, second)
        result.append(
            _coverage_finding(
                check,
                status,
                entity_kind,
                entity_ids,
                detail="The candidate oriented facet partition does not reproduce the source boundary/interface chain.",
            )
        )
    return tuple(result)


def _source_region_measures(
    domain: CoverageDomain,
    findings: list[MeshCertificateFinding],
    /,
    *,
    exact_cad: ExactCadSourceBoundary | None,
) -> tuple[ExactScalar | None, ...]:
    """Integrate exact source measures; bounded source proofs retain no fake volume."""
    from ._domain_numerics import affine_cell_measure, cell_coordinate_expressions

    if isinstance(domain, AdaptiveImplicitSurface):
        return tuple(None for _ in domain.region_ids)
    if isinstance(domain, MeshingDomain):
        if exact_cad is None:
            return tuple(None for _ in domain.regions)
        try:
            return tuple(exact_cad.chain.exact_region_measures())
        except (TypeError, ValueError, ZeroDivisionError) as error:
            findings.append(
                _coverage_finding(
                    "source_region_measure_unresolved",
                    "unresolved",
                    "mesh",
                    detail=f"Exact authored CAD source integration is unavailable: {error}.",
                )
            )
            return tuple(None for _ in domain.regions)
    try:
        if isinstance(domain, MappedReferenceDomain):
            source_maps = cell_coordinate_expressions(
                domain.reference_mesh, domain.source_geometry
            )
            source_blocks = tuple(
                block
                for block in domain.reference_mesh.blocks
                for _ in range(block.cell_count)
            )
            source_regions = np.asarray(domain.cell_regions, dtype=np.int64)
            source_cell_ids = tuple(
                int(identifier)
                for block in domain.reference_mesh.blocks
                for identifier in np.asarray(block.global_ids, dtype=np.int64)
            )
            source_totals: list[ExactScalar] = [Fraction(0) for _ in domain.region_ids]
            unresolved_source_regions: set[int] = set()
            for source_row, (expressions, block) in enumerate(
                zip(source_maps, source_blocks, strict=True)
            ):
                region = int(source_regions[source_row])
                try:
                    if expressions is None:
                        raise ValueError(
                            "Mapped source root cell has no reference coordinate map."
                        )
                    measure = affine_cell_measure(expressions, block.cell_kind)
                except (TypeError, ValueError, ZeroDivisionError) as error:
                    unresolved_source_regions.add(region)
                    findings.append(
                        _coverage_finding(
                            "source_region_measure_unresolved",
                            "unresolved",
                            "source_cell",
                            (source_cell_ids[source_row],),
                            detail=f"Exact mapped source-cell integration is unavailable: {error}.",
                        )
                    )
                    continue
                if measure <= 0:
                    unresolved_source_regions.add(region)
                    findings.append(
                        _coverage_finding(
                            "source_cell_orientation",
                            "violated",
                            "source_cell",
                            (source_cell_ids[source_row],),
                            detail="A mapped source root cell has nonpositive exact oriented measure.",
                        )
                    )
                    continue
                source_totals[region] += measure
            return tuple(
                None if region in unresolved_source_regions else measure
                for region, measure in enumerate(source_totals)
            )
        if isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain)):
            return tuple(domain.exact_region_measures())
    except (TypeError, ValueError, ZeroDivisionError) as error:
        findings.append(
            _coverage_finding(
                "source_region_measure_unresolved",
                "unresolved",
                "mesh",
                detail=f"Exact source-region integration is unavailable: {error}.",
            )
        )
    return tuple(None for _ in domain.region_ids)


def _target_region_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: CoverageDomain,
    preparation: DomainCoveragePreparation,
    findings: list[MeshCertificateFinding],
    /,
    *,
    required: bool = True,
    cell_measures: Mapping[int, Fraction] | None = None,
) -> tuple[ExactScalar | None, ...]:
    """Integrate target cells under their explicit dense region assignments.

    ``cell_measures`` replaces the measures of cells whose corners were
    substituted by exact source points, keyed by cell order.
    """
    from ._domain_numerics import affine_cell_measure, polyhedral_cell_measures

    polyhedral_measures: tuple[Fraction | None, ...] | None = None
    polyhedral_measure_error: Exception | None = None
    if any(expressions is None for expressions in preparation.coordinate_maps):
        try:
            polyhedral_measures = polyhedral_cell_measures(mesh, geometry)
        except (TypeError, ValueError, IndexError, ArithmeticError) as error:
            polyhedral_measure_error = error
    totals: list[ExactScalar] = [Fraction(0) for _ in domain.region_ids]
    unresolved_regions: set[int] = set()
    cell_kinds = tuple(
        mesh_block.cell_kind
        for mesh_block in mesh.blocks
        for _ in range(mesh_block.cell_count)
    )
    for cell, (assignment, expressions, cell_kind) in enumerate(
        zip(
            preparation.assignments,
            preparation.coordinate_maps,
            cell_kinds,
            strict=True,
        )
    ):
        try:
            if cell_measures is not None and cell in cell_measures:
                measure = cell_measures[cell]
            elif expressions is None:
                if polyhedral_measures is None:
                    raise ValueError(
                        "Exact polyhedral cell integration is unavailable: "
                        f"{polyhedral_measure_error}"
                    )
                polyhedral_measure = polyhedral_measures[assignment.row]
                if polyhedral_measure is None:
                    raise ValueError(
                        "A vertex-defined cell has no exact polyhedral measure."
                    )
                measure = polyhedral_measure
            else:
                measure = affine_cell_measure(expressions, cell_kind)
        except (TypeError, ValueError, IndexError, ArithmeticError) as error:
            unresolved_regions.add(assignment.region)
            if required:
                findings.append(
                    _coverage_finding(
                        "region_measure_unresolved",
                        "unresolved",
                        "cell",
                        (assignment.global_id,),
                        detail=f"Exact target-cell measure integration is unavailable: {error}.",
                    )
                )
            continue
        if measure <= 0:
            findings.append(
                _coverage_finding(
                    "target_cell_orientation",
                    "violated",
                    "cell",
                    (assignment.global_id,),
                    detail="A target cell has nonpositive exact signed measure.",
                )
            )
        totals[assignment.region] += measure
    achieved = tuple(None for _ in totals) if unresolved_regions else tuple(totals)
    return achieved


def _exact_region_degree_ids(
    source_domain: PiecewiseLinearDomain | AffinePlcDomain,
    region_ids: tuple[str, ...],
    region_occurrence_paths: tuple[tuple[str, ...], ...],
    overlaps: tuple[FacetOverlap, ...],
    assignments: tuple[CellCoverageAssignment, ...],
    /,
    *,
    domain_id: str,
    source_id: str,
    source_revision: str,
    embedding_id: str,
    premise_id: str,
) -> tuple[str, ...]:
    """Bind degree-one IDs only after exact oriented chain coverage is complete."""
    if (
        tuple(source_domain.region_ids) != region_ids
        or source_domain.region_occurrence_paths != region_occurrence_paths
        or not overlaps
    ):
        return ()
    source_pairs: np.ndarray = np.asarray(source_domain.facet_regions, dtype=np.int64)
    if source_pairs.ndim != 2 or source_pairs.shape[1] != 2:
        return ()
    result: list[str] = []
    for region, region_id in enumerate(region_ids):
        region_facets = tuple(
            facet
            for facet in range(source_pairs.shape[0])
            if region
            in (
                int(source_pairs[facet, 0]),
                int(source_pairs[facet, 1]),
            )
        )
        region_overlaps = tuple(
            value for value in overlaps if region in value.region_pair
        )
        if (
            not region_facets
            or not region_overlaps
            or not set(region_facets).issubset(
                {value.source_facet_id for value in region_overlaps}
            )
        ):
            return ()
        target_cells = tuple(
            (int(value.global_id), int(value.row))
            for value in assignments
            if int(value.region) == region
        )
        if not target_cells:
            return ()
        result.append(
            canonical_fingerprint(
                {
                    "kind": "oriented-material-boundary-degree-one",
                    "domain": domain_id,
                    "source": (source_id, source_revision),
                    "region": (region, region_id, region_occurrence_paths[region]),
                    "source_boundary_facets": region_facets,
                    "oriented_overlap_chain": _overlap_payload(region_overlaps),
                    "target_cells": target_cells,
                    "embedding": embedding_id,
                    "exact_chain_premise": premise_id,
                    "degree": 1,
                }
            )
        )
    return tuple(result)


def _coverage_boundary_partition(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: CoverageDomain,
    preparation: DomainCoveragePreparation,
    findings: list[MeshCertificateFinding],
    /,
    *,
    exact_cad: ExactCadSourceBoundary | None,
    embedding: GlobalEmbeddingCertificate,
    source: SourceBoundaryQuery | None,
    fidelity: SourceFidelityCertificate | None,
    source_coverage_witness: SourceCoverageWitness | None,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None,
    validated_parent_embedding: ValidatedGlobalEmbedding | None,
) -> _BoundaryCoverageProof:
    """Prove exact chain equality or an explicit source-boundary homotopy."""
    from ..discretization._coordinate_enclosure import affine_map
    from ._planar_coverage import partition_overlaps

    facet_overlaps: tuple[FacetOverlap, ...] = ()
    target_facet_count = 0
    chart_certificate_id: str | None = None
    source_distance_upper: float | None = None
    source_region_degree_ids: tuple[str, ...] = ()
    source_points: PlcSourcePoints | None = None
    if isinstance(domain, AnalyticImplicitProfile):
        if not isinstance(source, AnalyticImplicitProfile) or not _same_tree_content(
            source, domain
        ):
            findings.append(
                _coverage_finding(
                    "analytic_source_identity",
                    "violated",
                    "source_region",
                    detail="Analytic profile coverage requires the profile itself as its revision-bound source query.",
                )
            )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        if (
            fidelity is None
            or fidelity.status != "certified"
            or not isinstance(fidelity.witness, SourceBoundaryDistanceWitness)
            or fidelity.coverage is not None
        ):
            findings.append(
                _coverage_finding(
                    "analytic_source_distance_witness",
                    "unresolved",
                    "source_region",
                    detail="Analytic restricted coverage requires a current distance-only SourceBoundaryDistanceWitness.",
                )
            )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        if not isinstance(
            source_coverage_witness,
            (
                RestrictedDelaunayBoundaryWitness,
                RestrictedDelaunayCoverageTransportWitness,
            ),
        ):
            findings.append(
                _coverage_finding(
                    "restricted_delaunay_coverage_witness",
                    "unresolved",
                    "source_region",
                    detail="Analytic source coverage requires retained restricted-Delaunay intersections, orientation, and whole-volume classification.",
                )
            )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        try:
            proof = validate_restricted_delaunay_coverage_witness(
                mesh,
                geometry,
                domain,
                preparation.cell_regions,
                fidelity.witness,
                source_coverage_witness,
                embedding=embedding,
            )
        except RestrictedDelaunayCoverageUnavailable as error:
            findings.append(
                _coverage_finding(
                    error.check,
                    error.status,
                    "mesh",
                    detail=str(error),
                )
            )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        except (TypeError, ValueError, IndexError) as error:
            findings.append(
                _coverage_finding(
                    "restricted_delaunay_witness",
                    "violated",
                    "mesh",
                    detail=f"Retained restricted-Delaunay witness is structurally invalid: {error}",
                )
            )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        except (ArithmeticError, FloatingPointError) as error:
            findings.append(
                _coverage_finding(
                    "restricted_delaunay_interval",
                    "unresolved",
                    "mesh",
                    detail=f"Restricted-Delaunay interval revalidation is unresolved: {error}",
                )
            )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        distance_upper = max(fidelity.mesh_to_source_upper, fidelity.source_to_mesh_upper)
        return _BoundaryCoverageProof(
            proof.facet_overlaps,
            proof.target_facet_count,
            None,
            proof.source_facet_count,
            proof.covered_source_facet_count,
            "source_boundary_homotopy",
            proof.degree_id,
            distance_upper,
            (proof.degree_id,),
        )
    if source_coverage_witness is not None and not (
        isinstance(source_coverage_witness, PlcSourcePointWitness)
        and isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain))
    ):
        findings.append(
            _coverage_finding(
                "source_coverage_witness_domain",
                "violated",
                "source_region",
                detail="PLC source-point evidence applies only to piecewise-linear source domains."
                if isinstance(source_coverage_witness, PlcSourcePointWitness)
                else "Restricted-Delaunay analytic coverage evidence cannot be applied to this source-domain type.",
            )
        )
    if isinstance(domain, MappedReferenceDomain):
        from ._domain_chart_chain import certify_child_chart_partition

        chart_proof = certify_child_chart_partition(
            domain, mesh, geometry, preparation.cell_regions, chart_parent=chart_parent,
            _validated_parent_embedding=validated_parent_embedding,
        )
        findings.extend(chart_proof.findings)
        facet_overlaps = chart_proof.facet_overlaps
        target_facet_count = chart_proof.target_facet_count
        chart_certificate_id = canonical_fingerprint(
            {
                "parent_cell_ids": chart_proof.parent_cell_ids,
                "chart_ids": chart_proof.chart_ids,
                "facet_overlaps": _overlap_payload(facet_overlaps),
                "findings": tuple(value.finding_id for value in chart_proof.findings),
            }
        )
        source_count = len(domain.facets)
        covered_count = len({value.source_facet_id for value in facet_overlaps})
        source_coverage_id = chart_certificate_id
        if (
            not chart_proof.findings
            and len(chart_proof.parent_cell_ids) == len(preparation.assignments)
            and len(chart_proof.chart_ids) == len(preparation.assignments)
            and embedding.status == "certified"
        ):
            source_region_degree_ids = _exact_region_degree_ids(
                domain.reference_domain,
                tuple(domain.region_ids),
                domain.reference_domain.region_occurrence_paths,
                facet_overlaps,
                preparation.assignments,
                domain_id=domain.domain_id,
                source_id=domain.source_id,
                source_revision=domain.source_revision,
                embedding_id=embedding.certificate_id,
                premise_id=chart_certificate_id,
            )
        coverage_method: CoverageMethod = "mapped_reference_charts"
    elif isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain, MeshingDomain)) and (
        isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain))
        or exact_cad is not None
    ):
        if isinstance(domain, (PiecewiseLinearDomain, AffinePlcDomain)):
            source_domain = domain
            cad_boundary_id = None
        else:
            if exact_cad is None:
                findings.append(
                    _coverage_finding(
                        "source_cad_boundary_missing",
                        "unresolved",
                        "source_region",
                        detail="Exact CAD coverage has no current authored source boundary chain.",
                    )
                )
                return _BoundaryCoverageProof(
                    (), 0, None, 0, 0, "exact_oriented_facets", None, None, ()
                )
            source_domain = exact_cad.chain
            cad_boundary_id = exact_cad.boundary_id
        dimension = source_domain.ambient_dimension
        affine_cells = tuple(
            None if expressions is None else affine_map(expressions, dimension)
            for expressions in preparation.coordinate_maps
        )
        if (
            isinstance(domain, MeshingDomain)
            and any(value is None for value in affine_cells)
            and not any(
                expressions is None for expressions in preparation.coordinate_maps
            )
        ):
            from ._approximate_source_coverage import approximate_source_coverage

            attempt = approximate_source_coverage(
                mesh,
                geometry,
                domain,
                preparation.cell_regions,
                source=source,
                fidelity=fidelity,
                embedding=embedding,
            )
            if attempt.bounded is not None:
                bounded = attempt.bounded
                return _BoundaryCoverageProof(
                    (),
                    bounded.target_facet_count,
                    None,
                    bounded.source_facet_count,
                    bounded.covered_source_facet_count,
                    "source_boundary_homotopy",
                    bounded.witness_id,
                    bounded.distance_upper,
                    bounded.boundary_degree_ids,
                )
            for failure in attempt.failures:
                findings.append(
                    _coverage_finding(
                        failure.check,
                        failure.status,
                        "mesh",
                        failure.entity_ids,
                        detail=failure.detail,
                    )
                )
            return _BoundaryCoverageProof(
                (),
                0,
                None,
                len(source_domain.facets),
                0,
                "source_boundary_homotopy",
                None,
                None,
                (),
            )
        else:
            from ._domain_planar_boundary import oriented_target_facets

            if isinstance(source_coverage_witness, PlcSourcePointWitness) and isinstance(
                domain, (PiecewiseLinearDomain, AffinePlcDomain)
            ):
                source_points = _plc_source_points(
                    mesh, domain, preparation, source_coverage_witness, findings
                )
                if source_points is None:
                    return _BoundaryCoverageProof(
                        (),
                        0,
                        None,
                        len(source_domain.facets),
                        0,
                        "exact_oriented_facets",
                        None,
                        None,
                        (),
                    )
            target_facets, issues = oriented_target_facets(
                mesh,
                preparation.cell_regions,
                preparation.coordinate_maps,
                source_domain=source_domain,
                geometry=geometry,
                exact_face_images=True,
                source_points=source_points,
            )
            target_facet_count = len(target_facets)
            facet_overlaps, partition_issues = partition_overlaps(
                source_domain,
                target_facets,
                sources=None if source_points is None else source_points.sources,
            )
            exact_issues = bool(issues or partition_issues)
            if (
                isinstance(domain, MeshingDomain)
                and fidelity is not None
                and fidelity.tolerance > 0.0
                and exact_issues
            ):
                from ._approximate_source_coverage import approximate_source_coverage

                attempt = approximate_source_coverage(
                    mesh,
                    geometry,
                    domain,
                    preparation.cell_regions,
                    source=source,
                    fidelity=fidelity,
                    embedding=embedding,
                )
                if attempt.bounded is not None:
                    bounded = attempt.bounded
                    return _BoundaryCoverageProof(
                        (),
                        bounded.target_facet_count,
                        None,
                        bounded.source_facet_count,
                        bounded.covered_source_facet_count,
                        "source_boundary_homotopy",
                        bounded.witness_id,
                        bounded.distance_upper,
                        bounded.boundary_degree_ids,
                    )
                for failure in attempt.failures:
                    findings.append(
                        _coverage_finding(
                            failure.check,
                            failure.status,
                            "mesh",
                            failure.entity_ids,
                            detail=failure.detail,
                        )
                    )
                return _BoundaryCoverageProof(
                    (),
                    target_facet_count,
                    None,
                    len(source_domain.facets),
                    0,
                    "source_boundary_homotopy",
                    None,
                    None,
                    (),
                )
            for issue in issues:
                findings.append(
                    _coverage_finding(
                        issue.check,
                        issue.status,
                        "facet",
                        issue.entity_ids,
                        detail=issue.detail,
                    )
                )
            partition_status = (
                "unresolved"
                if any(issue.status == "unresolved" for issue in issues)
                else "violated"
            )
            findings.extend(
                _partition_findings(partition_issues, status=partition_status)
            )
        source_count = len(source_domain.facets)
        covered_count = len({value.source_facet_id for value in facet_overlaps})
        if isinstance(domain, MeshingDomain):
            if cad_boundary_id is None:
                findings.append(
                    _coverage_finding(
                        "source_cad_boundary_missing",
                        "unresolved",
                        "source_region",
                        detail="Exact CAD source identity has no bound authored boundary ID.",
                    )
                )
                return _BoundaryCoverageProof(
                    (),
                    target_facet_count,
                    None,
                    source_count,
                    covered_count,
                    "exact_oriented_facets",
                    None,
                    None,
                    (),
                )
            chart_certificate_id = cad_boundary_id
            source_coverage_id = cad_boundary_id
        elif source_points is None:
            source_coverage_id = canonical_fingerprint(
                {
                    "kind": "exact-oriented-domain-boundary",
                    "domain": domain.domain_id,
                    "overlaps": _overlap_payload(facet_overlaps),
                    "target_facet_count": target_facet_count,
                }
            )
        else:
            # Exact source points decide the chain; the carriers round them, and
            # declared-plane charts identify source triangles with their plane.
            source_distance_upper = _upper_sum(
                source_points.carrier_deviation_upper,
                source_points.chart_deviation_upper,
            )
            source_coverage_id = canonical_fingerprint(
                {
                    "kind": "witnessed-oriented-domain-boundary",
                    "domain": domain.domain_id,
                    "source_points": source_coverage_witness.witness_id
                    if isinstance(source_coverage_witness, PlcSourcePointWitness)
                    else None,
                    "charts": tuple(sorted(source_points.charts)),
                    "overlaps": _overlap_payload(facet_overlaps),
                    "target_facet_count": target_facet_count,
                    "distance_upper": source_distance_upper,
                }
            )
        coverage_method = (
            "source_boundary_homotopy"
            if source_points is not None and source_points.charts
            else "exact_oriented_facets"
        )
        if not issues and not partition_issues and embedding.status == "certified":
            source_region_degree_ids = _exact_region_degree_ids(
                source_domain,
                tuple(domain.region_ids),
                source_domain.region_occurrence_paths,
                facet_overlaps,
                preparation.assignments,
                domain_id=domain.domain_id,
                source_id=domain.source_id,
                source_revision=_coverage_source_revision(domain, source),
                embedding_id=embedding.certificate_id,
                premise_id=source_coverage_id,
            )
    else:
        from ._approximate_source_coverage import approximate_source_coverage

        attempt = approximate_source_coverage(
            mesh,
            geometry,
            domain,
            preparation.cell_regions,
            source=source,
            fidelity=fidelity,
            embedding=embedding,
        )
        if attempt.bounded is None:
            for failure in attempt.failures:
                findings.append(
                    _coverage_finding(
                        failure.check,
                        failure.status,
                        "mesh",
                        failure.entity_ids,
                        detail=failure.detail,
                    )
                )
            return _BoundaryCoverageProof(
                (), 0, None, 0, 0, "source_boundary_homotopy", None, None, ()
            )
        bounded = attempt.bounded
        target_facet_count = bounded.target_facet_count
        source_count = bounded.source_facet_count
        covered_count = bounded.covered_source_facet_count
        source_coverage_id = bounded.witness_id
        source_distance_upper = bounded.distance_upper
        source_region_degree_ids = bounded.boundary_degree_ids
        coverage_method = "source_boundary_homotopy"
    return _BoundaryCoverageProof(
        tuple(facet_overlaps),
        target_facet_count,
        chart_certificate_id,
        source_count,
        covered_count,
        coverage_method,
        source_coverage_id,
        source_distance_upper,
        source_region_degree_ids,
        None if source_points is None else source_points.cell_measures,
    )


def _plc_source_points(
    mesh: CellMesh,
    domain: PiecewiseLinearDomain | AffinePlcDomain,
    preparation: DomainCoveragePreparation,
    witness: PlcSourcePointWitness,
    findings: list[MeshCertificateFinding],
    /,
) -> PlcSourcePoints | None:
    """Recompute the exact source points a PLC mesh's carriers round."""
    from ._domain_planar_boundary import affine_cell_corners
    from ._planar_coverage import resolve_plc_source_points, SourcePointRefusal

    try:
        cells = affine_cell_corners(mesh, preparation.coordinate_maps)
    except (TypeError, ValueError, KeyError, ArithmeticError) as error:
        findings.append(
            _coverage_finding(
                "source_point_cells",
                "unresolved",
                "mesh",
                detail=f"Exact source points need exact affine cell corners: {error}",
            )
        )
        return None
    try:
        return resolve_plc_source_points(
            witness,
            domain,
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            cells,
        )
    except SourcePointRefusal as error:
        findings.append(
            _coverage_finding(
                error.check,
                error.status,
                "vertex",
                error.entity_ids,
                detail=str(error),
            )
        )
        return None


def _upper_sum(first: float, second: float, /) -> float:
    return fraction_up(Fraction(first) + Fraction(second))


def _prove_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: CoverageDomain,
    cell_regions: ArrayLike,
    /,
    *,
    embedding: GlobalEmbeddingCertificate,
    source: SourceBoundaryQuery | None,
    fidelity: SourceFidelityCertificate | None,
    source_coverage_witness: SourceCoverageWitness | None,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None,
    _validated_parent_embedding: ValidatedGlobalEmbedding | None,
) -> _CoverageProof:
    """Recompute exact source chains or a witnessed source-boundary homotopy."""
    from ._approximate_source_coverage import (
        exact_planar_cad_boundary,
        SourceCoverageUnavailable,
    )
    from ._domain_preparation import prepare_domain_coverage

    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError(
            "certify_domain_coverage requires CellMesh and CellGeometrySpec inputs."
        )
    domain = require_coverage_domain(domain)
    if _validated_parent_embedding is not None and (
        type(_validated_parent_embedding) is not ValidatedGlobalEmbedding
        or not isinstance(domain, MappedReferenceDomain)
        or chart_parent is None
    ):
        raise ValueError("Validated parent embedding requires its mapped chart parent.")
    if not domain.region_ids:
        raise ValueError(
            "Domain coverage requires an established source material region."
        )
    if not isinstance(embedding, GlobalEmbeddingCertificate):
        raise TypeError("embedding must be a GlobalEmbeddingCertificate.")
    if fidelity is not None and source is None:
        source = fidelity.source
    if source is not None and not isinstance(source, SourceBoundaryQuery):
        raise TypeError("source must implement SourceBoundaryQuery.")
    if source_coverage_witness is not None and not isinstance(
        source_coverage_witness,
        (
            RestrictedDelaunayBoundaryWitness,
            RestrictedDelaunayCoverageTransportWitness,
            PlcSourcePointWitness,
        ),
    ):
        raise TypeError(
            "source_coverage_witness must be a restricted-Delaunay or PLC source-point witness or None."
        )
    source_revision = _coverage_source_revision(domain, source)
    findings: list[MeshCertificateFinding] = []
    fidelity_for_coverage = fidelity
    if fidelity is not None:
        if not isinstance(fidelity, SourceFidelityCertificate):
            raise TypeError("fidelity must be SourceFidelityCertificate.")
        if source is None:
            raise ValueError(
                "A source fidelity certificate requires its source boundary query."
            )
        if (fidelity.binding.source_id, fidelity.binding.source_revision) != (
            domain.source_id,
            source_revision,
        ):
            raise ValueError(
                "Source fidelity must bind the actual coverage domain revision."
            )
        try:
            fidelity.require_current(mesh, geometry, source)
        except (
            TypeError,
            ValueError,
            IndexError,
            ArithmeticError,
            FloatingPointError,
        ) as error:
            fidelity_for_coverage = None
            findings.append(
                _coverage_finding(
                    "source_fidelity_premise",
                    "violated",
                    "mesh",
                    detail=f"Retained source fidelity does not reproduce its witness: {error}",
                )
            )
    embedding.binding.require(mesh, geometry)
    dimension = (
        3
        if isinstance(
            domain, (MeshingDomain, AdaptiveImplicitSurface, AnalyticImplicitProfile)
        )
        else domain.ambient_dimension
    )
    if mesh.topological_dimension != dimension or mesh.ambient_dimension != dimension:
        raise ValueError(
            "Domain coverage requires matching full-dimensional mesh and source domain."
        )
    preparation = prepare_domain_coverage(
        mesh, geometry, cell_regions, len(domain.region_ids)
    )
    exact_cad = None
    if isinstance(domain, MeshingDomain):
        try:
            exact_cad = exact_planar_cad_boundary(domain)
        except SourceCoverageUnavailable as error:
            exact_cad = None
            if error.status == "violated":
                findings.append(
                    _coverage_finding(
                        "source_cad_boundary_authority",
                        "violated",
                        "source_facet",
                        detail=str(error),
                    )
                )
    requested = _source_region_measures(domain, findings, exact_cad=exact_cad)
    if embedding.status != "certified":
        premise_status = "violated" if embedding.status == "violated" else "unresolved"
        findings.append(
            _coverage_finding(
                "global_embedding_premise",
                premise_status,
                "mesh",
                detail="Domain coverage cannot be certified without a current certified global embedding.",
            )
        )
    boundary = _coverage_boundary_partition(
        mesh,
        geometry,
        domain,
        preparation,
        findings,
        exact_cad=exact_cad,
        embedding=embedding,
        source=source,
        fidelity=fidelity_for_coverage,
        source_coverage_witness=source_coverage_witness,
        chart_parent=chart_parent,
        validated_parent_embedding=_validated_parent_embedding,
    )
    achieved = _target_region_measures(
        mesh,
        geometry,
        domain,
        preparation,
        findings,
        required=boundary.coverage_method != "source_boundary_homotopy",
        cell_measures=boundary.cell_measures,
    )
    if boundary.coverage_method != "source_boundary_homotopy":
        for region, (source_measure, target_measure) in enumerate(
            zip(requested, achieved, strict=True)
        ):
            if (
                source_measure is not None
                and target_measure is not None
                and target_measure != source_measure
            ):
                findings.append(
                    _coverage_finding(
                        "region_measure",
                        "violated",
                        "source_region",
                        (region,),
                        detail="Exact target-cell measure differs from the source-authoritative region measure.",
                    )
                )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        embedding.binding.coordinate_scope,
        source_id=domain.source_id,
        source_revision=source_revision,
    )
    return _CoverageProof(
        binding,
        preparation.cell_regions,
        requested,
        achieved,
        boundary.facet_overlaps,
        boundary.source_facet_count,
        boundary.target_facet_count,
        boundary.covered_source_facet_count,
        tuple(findings),
        boundary.chart_certificate_id,
        boundary.coverage_method,
        boundary.source_coverage_id,
        boundary.source_distance_upper,
        boundary.source_region_degree_ids,
    )


def certify_domain_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: CoverageDomain,
    cell_regions: ArrayLike,
    /,
    *,
    embedding: GlobalEmbeddingCertificate,
    source: SourceBoundaryQuery | None = None,
    fidelity: SourceFidelityCertificate | None = None,
    source_coverage_witness: SourceCoverageWitness | None = None,
    chart_parent: tuple[CellMesh, CellGeometrySpec] | None = None,
    _validated_parent_embedding: ValidatedGlobalEmbedding | None = None,
) -> DomainCoverageCertificate:
    """Compute exact or witnessed bounded coverage from source authority."""
    return DomainCoverageCertificate(
        mesh,
        geometry,
        domain,
        cell_regions,
        embedding=embedding,
        source=source,
        fidelity=fidelity,
        source_coverage_witness=source_coverage_witness,
        chart_parent=chart_parent,
        _validated_parent_embedding=_validated_parent_embedding,
    )


__all__ = [
    "CoverageDomain",
    "coverage_region_occurrence_paths",
    "CoverageMethod",
    "DomainCoverageCertificate",
    "certify_domain_coverage",
    "SourceCoverageWitness",
]
