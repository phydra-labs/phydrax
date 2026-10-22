#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify mapped-volume coverage through retained source reference charts.

Independent nonperiodic P1 triangulations may instead use a common-affine
specialization: every actual source root polynomial must equal one invertible
exact affine image of its declared referencephysical polynomial coefficientwise.
The actual target maps are pulled back into the canonical rational source owner,
then independently checked for validity, embedding, material interfaces and
ordinary reference PLC coverage. Source and target physical embedding premises
remain mandatory; neither rounded corner fitting nor invented root ancestry is
evidence. Unsupported laws retain the strict source-root route below.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "mapped_domain_restriction_source"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "An unrestricted target has the source geometry ID and reference topology ID before it is treated as an identity root chart."
      }
    ],
    "conclusion": "A missing restriction record without the source identities leaves root ancestry unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_source_identity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The restriction record retains the declared source geometry and reference topology IDs."
      }
    ],
    "conclusion": "A differing restriction source identity violates mapped source binding.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_region_identity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Identity roots retain source cell regions, and each restricted child retains its source root region."
      }
    ],
    "conclusion": "A differing source region assignment violates material identity.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_parent_identity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every child parent ID names a root in the independent source root bank."
      }
    ],
    "conclusion": "An absent source parent violates root ancestry.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_root_expression"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A retained affine or composed chart reaches the explicitly identified source root reference cell; ordered source vertex IDs agree and either its nodal control bank agrees or full rational expression equality proves the exact interpolated root law."
      }
    ],
    "conclusion": "A differing coordinate law or source corner order, or an unproved change of nodal representation, violates root restriction equality.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_periodic_source_identity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Source and target both declare periodic topology or both omit it; periodic cells have equal cell IDs or isometry groups have equal group IDs."
      }
    ],
    "conclusion": "A differing quotient declaration violates mapped source identity.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_periodic_trace_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Both source and target embedding certificates list periodic_mapped_trace_equivariance in evaluated_checks."
      }
    ],
    "conclusion": "A missing trace-equivalence check leaves the periodic coverage dependency unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_source_embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The independent physical source embedding certificate reports certified status."
      }
    ],
    "conclusion": "Source embedding refusal propagates its violated or unresolved status into mapped coverage.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_reference_coverage_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The independent source reference partition coverage certificate reports certified status for its declared domain and cell regions."
      }
    ],
    "conclusion": "Source reference partition refusal propagates its violated or unresolved status into mapped coverage.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_missing_root"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every independently declared source root has at least one retained target chart."
      }
    ],
    "conclusion": "A root with no target charts violates source coverage.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_chart_representation"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Retained reference charts share vertex images and block coordinate elements; an affine carrier reproduces its polynomial chart without binary rounding changes."
      }
    ],
    "conclusion": "An unrepresentable chart carrier or differing shared vertex image leaves root coverage unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  },
  {
    "checks": [
      "mapped_domain_root_coverage_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each reconstructed root chart mesh has a certified embedding and a certified coverage certificate for the source reference cell boundary."
      }
    ],
    "conclusion": "Root chart partition refusal propagates its violated or unresolved status before any root-integral substitution.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped source coverage evidence contract",
      "locator": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ],
    "owner": "phydrax.geometry._mapped_reference_coverage.certify_mapped_reference_coverage"
  }
]
End mesh certificate checks
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from typing import assert_never, cast, final

import equinox as eqx
import jax
import numpy as np

import phydrax.discretization._periodic_cell as periodic_cell_owner
import phydrax.discretization._periodic_exact as periodic_exact_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
import phydrax.geometry._domain_coverage as domain_coverage_owner
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_reference_domain as mapped_reference_domain_owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._reference_cell_types import (
    CellBlock,
    CellGeometryElement,
    PolyhedralBlock,
)
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    facet_entity_ids,
    mesh_facets,
    MeshCertificateLimits,
)
from phydrax.geometry._mapped_corners import mapped_source_corner_coordinates
from phydrax.geometry._planar_primitives import project, rational_points, signed_measure

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._identity import strict_module_payload
from .._strict import StrictModule
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    coordinate_lagrange_element,
    equivalent_coordinate_root,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    require_full_p1_source,
    RestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_rational_geometry import (
    ExactRationalCellGeometrySource,
    rational_source_layouts,
)
from ..discretization._reference_cell import reference_cell_topology
from ..linalg._hermitian_spectral import fraction_matrix_profile, reserve_fraction_work
from ..linalg._small_batched import prepare_exact_small_linear_actions
from ..typing import checked
from ._mapped_coverage import integrate, map_domain
from ._mesh_certificates import active_expression_work, MeshCertificateBinding


@dataclass(frozen=True)
class _Root:
    cell_id: int
    kind: str
    region: int
    vertices: tuple[int, ...]
    element: CellGeometryElement
    controls: algebra.CoordinateSourceBank
    polynomial: tuple[algebra.Polynomial, ...]


class AffineReferenceChart(StrictModule):
    """A cell's reference chart with a retained affine transform."""

    block: str = eqx.field(static=True)
    cell_id: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    vertices: tuple[int, ...] = eqx.field(static=True)
    matrix: tuple[tuple[Fraction, ...], ...]
    offset: tuple[Fraction, ...]


class ComposedReferenceChart(StrictModule):
    """A cell's reference chart with its coordinate element and coefficient bank."""

    block: str = eqx.field(static=True)
    cell_id: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    vertices: tuple[int, ...] = eqx.field(static=True)
    reference_element: CellGeometryElement
    reference_controls: np.ndarray


type Chart = AffineReferenceChart | ComposedReferenceChart


class _ChartSource(StrictModule):
    element: CellGeometryElement
    controls: np.ndarray


@checked
def source_roots(
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[int, _Root]:
    elements, routes, _ = domain.source_geometry.resolve(domain.reference_mesh, budget=budget)
    values = algebra.prepared_coordinate_source_bank(
        domain.source_geometry, budget=budget
    )
    vertex_ids = np.asarray(domain.reference_mesh.vertex_global_ids, dtype=np.int64)
    result = {}
    cursor = 0
    for block, element, route in zip(
        domain.reference_mesh.blocks, elements, routes, strict=True
    ):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for local, cell_id, row in zip(
            local_values,
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            strict=True,
        ):
            polynomial = algebra.coordinate_polynomials(element, local, budget=budget)
            if polynomial is None:
                raise ValueError(
                    "MappedReferenceCoverage.root must resolve coordinate polynomials; got absent root expressions"
                )
            result[int(cell_id)] = _Root(
                int(cell_id),
                block.cell_kind,
                int(domain.cell_regions[cursor]),
                tuple(int(value) for value in vertex_ids[row]),
                element,
                local,
                polynomial,
            )
            cursor += 1
    return result


@checked
def physical_source_mesh(
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    *,
    budget: CoordinateEnclosureBudget,
) -> CellMesh:
    """An rational corner view of the independently declared source coordinate maps."""
    points = mapped_source_corner_coordinates(
        domain.reference_mesh, domain.source_geometry, budget=budget
    )
    return domain.reference_mesh.with_coordinates(
        points, numeric_version=domain.source_revision
    )


def _source_owner_binding(value: object) -> tuple[tuple[int, ...], str]:
    """Bind an issued graph's owners and all numerical fields, including static banks."""
    pending = [value]
    seen: dict[int, object] = {}
    arrays = []
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen[identity] = item
        if isinstance(item, (np.ndarray, jax.Array)):
            arrays.append(item)
        elif is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, field.name) for field in fields(item))
        elif isinstance(item, Mapping):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return tuple(seen), canonical_fingerprint(array_tree_fingerprint(tuple(arrays)))


def _source_snapshot_id(
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
) -> str:
    # Root owners define semantics; the full numerical tree also catches stale IDs.
    return canonical_fingerprint(
        {
            "domain": domain.domain_id,
            "source": (domain.source_id, domain.source_revision),
            "reference_domain": domain.reference_domain.domain_id,
            "region_ids": domain.region_ids,
            "reference_mesh": domain.reference_mesh.mesh_id,
            "reference_blocks": tuple(
                (block.name, block.cell_kind, block.block_id)
                for block in domain.reference_mesh.blocks
            ),
            "source_geometry": cell_geometry_id(domain.source_geometry),
            "numerical_banks": array_tree_fingerprint(domain),
        }
    )


@final
class MappedReferenceSourceEvidence(StrictModule):
    """Independent source theorems, distinct from every target's coverage proof."""

    domain: mapped_reference_domain_owner.MappedReferenceDomain
    limits: MeshCertificateLimits
    physical_mesh: CellMesh
    reference_validity: CellValidityCertificate
    reference_embedding: global_embedding_owner.GlobalEmbeddingCertificate
    reference_coverage: domain_coverage_owner.DomainCoverageCertificate
    physical_validity: CellValidityCertificate
    physical_embedding: global_embedding_owner.GlobalEmbeddingCertificate
    source_snapshot_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: mapped_reference_domain_owner.MappedReferenceDomain,
        limits: MeshCertificateLimits,
        physical_mesh: CellMesh,
        reference_validity: CellValidityCertificate,
        reference_embedding: global_embedding_owner.GlobalEmbeddingCertificate,
        reference_coverage: domain_coverage_owner.DomainCoverageCertificate,
        physical_validity: CellValidityCertificate,
        physical_embedding: global_embedding_owner.GlobalEmbeddingCertificate,
    ) -> None:
        self.domain = domain
        self.limits = limits
        self.physical_mesh = physical_mesh
        self.reference_validity = reference_validity
        self.reference_embedding = reference_embedding
        self.reference_coverage = reference_coverage
        self.physical_validity = physical_validity
        self.physical_embedding = physical_embedding
        self.source_snapshot_id = _source_snapshot_id(domain)
        self.evidence_id = _source_evidence_identity(self)
        self.require_current(domain, limits)

    @checked
    def require_current(
        self,
        domain: mapped_reference_domain_owner.MappedReferenceDomain,
        limits: MeshCertificateLimits,
    ) -> None:
        """Bind live banks, material/periodic topology, policy and all proof premises."""
        if (
            not tree_content_equal(self.domain, domain)
            or not tree_content_equal(self.limits, limits)
            or self.source_snapshot_id != _source_snapshot_id(domain)
            or self.evidence_id != _source_evidence_identity(self)
        ):
            raise ValueError(
                "MappedReferenceSourceEvidence.source must retain its exact source "
                "banks, topology, material assignments and certificate limits."
            )
        reference_geometry = CellGeometrySpec.affine(domain.reference_mesh)
        self.reference_validity.require_bound(
            reference_geometry, mesh=domain.reference_mesh
        )
        self.physical_validity.require_bound(
            domain.source_geometry, mesh=self.physical_mesh
        )
        self.reference_embedding.binding.require(domain.reference_mesh, reference_geometry)
        self.reference_coverage.binding.require(domain.reference_mesh, reference_geometry)
        self.physical_embedding.binding.require(self.physical_mesh, domain.source_geometry)
        if (
            not self.reference_validity.all_certified
            or not self.physical_validity.all_certified
            or any(
                certificate.status != "certified"
                or certificate.binding.limits_id != limits.limits_id
                for certificate in (
                    self.reference_embedding,
                    self.reference_coverage,
                    self.physical_embedding,
                )
            )
            or self.reference_embedding.validity_certificate_id
            != self.reference_validity.certificate_id
            or self.physical_embedding.validity_certificate_id
            != self.physical_validity.certificate_id
            or self.reference_coverage.embedding_certificate_id
            != self.reference_embedding.certificate_id
            or self.reference_coverage.domain_id != domain.reference_domain.domain_id
        ):
            raise ValueError(
                "MappedReferenceSourceEvidence.certificates must independently certify "
                "the bound reference partition and physical source."
            )

    @checked
    def require_live(
        self,
        domain: mapped_reference_domain_owner.MappedReferenceDomain,
        limits: MeshCertificateLimits,
    ) -> None:
        """Require actual issuing-process proof authority, not constructor-valid IDs."""
        issued = _ISSUED_SOURCE_EVIDENCE.get(id(self))
        if (
            type(self) is not MappedReferenceSourceEvidence
            or issued is None
            or issued[0]() is not self
            or issued[1] != self.evidence_id
            or issued[2] != _source_owner_binding(self)
        ):
            raise ValueError(
                "MappedReferenceSourceEvidence.authority must be issued by an actual "
                "independent source proof; constructor-valid records are not authority."
            )
        self.require_current(domain, limits)


_ISSUED_SOURCE_EVIDENCE: dict[
    int,
    tuple[
        weakref.ReferenceType[MappedReferenceSourceEvidence],
        str,
        tuple[tuple[int, ...], str],
    ],
] = {}


def _issue_source_evidence(
    evidence: MappedReferenceSourceEvidence,
    *,
    budget: CoordinateEnclosureBudget,
) -> MappedReferenceSourceEvidence:
    identity = id(evidence)
    reference = weakref.ref(
        evidence, lambda _: _ISSUED_SOURCE_EVIDENCE.pop(identity, None)
    )
    binding = _source_owner_binding(evidence)
    budget.retain_basis(binding)
    # The global seal retains no numerical owners, only a weak reference and IDs.
    _ISSUED_SOURCE_EVIDENCE[identity] = reference, evidence.evidence_id, binding
    return evidence


def _source_evidence_identity(evidence: MappedReferenceSourceEvidence) -> str:
    certificates = (
        evidence.reference_validity,
        evidence.reference_embedding,
        evidence.reference_coverage,
        evidence.physical_validity,
        evidence.physical_embedding,
    )
    return canonical_fingerprint(
        {
            "kind": "mapped-reference-source-evidence",
            "source_snapshot": evidence.source_snapshot_id,
            "limits": strict_module_payload(evidence.limits)["numeric_content_id"],
            "certificates": tuple(value.certificate_id for value in certificates),
            "physical_mesh": evidence.physical_mesh.mesh_id,
            "proof_banks": array_tree_fingerprint((evidence.physical_mesh, certificates)),
        }
    )


def retained_mapped_reference_source_evidence(
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> MappedReferenceSourceEvidence | None:
    key = (domain.domain_id, limits.limits_id, _source_snapshot_id(domain))
    evidence = budget.mapped_reference_source_evidence.get(key)
    if evidence is not None:
        evidence.require_live(domain, limits)
    return evidence


def retain_mapped_reference_source_evidence(
    evidence: MappedReferenceSourceEvidence,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Admit an authenticated live source proof into the consuming request."""
    evidence.require_live(domain, limits)
    key = (domain.domain_id, limits.limits_id, evidence.source_snapshot_id)
    budget.retain_owner(evidence)
    budget.retain_basis(key)
    budget.mapped_reference_source_evidence[key] = evidence


@checked
def certify_mapped_reference_source(
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    CellMesh,
    global_embedding_owner.GlobalEmbeddingCertificate,
    domain_coverage_owner.DomainCoverageCertificate,
]:
    """Independently establish the source reference partition and physical embedding."""
    retained = retained_mapped_reference_source_evidence(domain, limits, budget=budget)
    if retained is not None:
        return (
            retained.physical_mesh,
            retained.physical_embedding,
            retained.reference_coverage,
        )
    reference_geometry = CellGeometrySpec.affine(domain.reference_mesh, budget=budget)
    reference_validity = certify_cell_geometry_validity(
        reference_geometry, mesh=domain.reference_mesh, budget=budget
    )
    reference_embedding = global_embedding_owner.certify_global_embedding(
        domain.reference_mesh,
        reference_geometry,
        reference_validity,
        limits=limits,
        budget=budget,
    )
    reference_coverage = domain_coverage_owner.certify_domain_coverage(
        domain.reference_mesh,
        reference_geometry,
        domain.reference_domain,
        domain.cell_regions,
        embedding=reference_embedding,
        limits=limits,
        budget=budget,
    )
    physical_mesh = physical_source_mesh(domain, budget=budget)
    physical_validity = certify_cell_geometry_validity(
        domain.source_geometry, mesh=physical_mesh, budget=budget
    )
    physical_embedding = global_embedding_owner.certify_global_embedding(
        physical_mesh,
        domain.source_geometry,
        physical_validity,
        limits=limits,
        budget=budget,
    )
    if (
        reference_validity.all_certified
        and physical_validity.all_certified
        and reference_embedding.status
        == reference_coverage.status
        == physical_embedding.status
        == "certified"
    ):
        retain_mapped_reference_source_evidence(
            _issue_source_evidence(
                MappedReferenceSourceEvidence(
                    domain,
                    limits,
                    physical_mesh,
                    reference_validity,
                    reference_embedding,
                    reference_coverage,
                    physical_validity,
                    physical_embedding,
                ),
                budget=budget,
            ),
            domain,
            limits,
            budget=budget,
        )
    return physical_mesh, physical_embedding, reference_coverage


def _source_scientific_content(evidence: MappedReferenceSourceEvidence) -> tuple[object, ...]:
    receipts = {
        "source_expression_work_units",
        "source_expression_required_work_units",
        "source_expression_peak_bytes",
    }
    return (
        evidence.domain,
        evidence.limits,
        evidence.physical_mesh,
        evidence.reference_validity,
        evidence.physical_validity,
        *(
            {
                field.name: getattr(certificate, field.name)
                for field in fields(certificate)
                if field.name not in receipts
            }
            for certificate in (
                evidence.reference_embedding,
                evidence.reference_coverage,
                evidence.physical_embedding,
            )
        ),
    )


@checked
def reprove_mapped_reference_source_evidence(
    stored: MappedReferenceSourceEvidence,
    *,
    budget: CoordinateEnclosureBudget,
) -> MappedReferenceSourceEvidence:
    """Cold-rebuild once per request, then compare the complete scientific identity.

    The caller must invoke this before seeding any imported evidence into this
    request. Cache entries originate only from an actual proof or authenticated
    live-owner retention; archive deserialization itself is not such authority.
    """
    stored.require_current(stored.domain, stored.limits)
    certify_mapped_reference_source(stored.domain, stored.limits, budget=budget)
    fresh = retained_mapped_reference_source_evidence(
        stored.domain, stored.limits, budget=budget
    )
    if (
        fresh is None
        or fresh.evidence_id != stored.evidence_id
        or not tree_content_equal(
            _source_scientific_content(stored), _source_scientific_content(fresh)
        )
    ):
        raise ValueError(
            "MappedReferenceSourceEvidence.cold_proof must reproduce the retained "
            "source theorems and numerical proof banks."
        )
    # Postorder readers still hold this exact stored object in their parent records.
    # Its authority is issued only after the independent proof and content match.
    _issue_source_evidence(stored, budget=budget)
    return fresh


@checked
def restore_mapped_reference_source_evidence(
    stored: MappedReferenceSourceEvidence,
    *,
    budget: CoordinateEnclosureBudget,
) -> MappedReferenceSourceEvidence:
    """Admit a live proof or cold-reprove and seal the exact postorder stored owner."""
    stored.require_current(stored.domain, stored.limits)
    issued = _ISSUED_SOURCE_EVIDENCE.get(id(stored))
    if issued is not None and issued[0]() is stored:
        stored.require_live(stored.domain, stored.limits)
        retain_mapped_reference_source_evidence(
            stored, stored.domain, stored.limits, budget=budget
        )
        return stored
    return reprove_mapped_reference_source_evidence(stored, budget=budget)


class _AffineTransform(StrictModule):
    matrix: tuple[tuple[Fraction, ...], ...]
    offset: tuple[Fraction, ...]


def _polynomial_affine_transform(
    element: PolynomialComposedCellGeometryElement,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineTransform | None:
    ancestor, expressions = algebra.coordinate_reference_chain(
        element, ancestor=element.source_element, budget=budget
    )
    if (
        ancestor.element_id != element.source_element.element_id
        or len(expressions) != dimension
        or any(isinstance(value, algebra.RationalPolynomial) for value in expressions)
    ):
        return None
    arguments = tuple(
        value
        for value in expressions
        if not isinstance(value, algebra.RationalPolynomial)
    )
    if any(
        len(exponent) != dimension or sum(exponent) > 1
        for argument in arguments
        for exponent in argument
    ):
        return None
    transform = tuple(
        tuple(
            algebra.require_rational_reference_scalar(
                argument.get(
                    tuple(int(axis == column) for axis in range(dimension)),
                    Fraction(),
                )
            )
            for column in range(dimension)
        )
        for argument in arguments
    )
    translation = tuple(
        algebra.require_rational_reference_scalar(
            argument.get((0,) * dimension, Fraction())
        )
        for argument in arguments
    )
    return _AffineTransform(transform, translation)


def _affine_reference_step(
    element: RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineTransform | None:
    match element:
        case RationalComposedCellGeometryElement():
            return None
        case RestrictedCellGeometryElement():
            transform = tuple(
                tuple(Fraction(float(value)) for value in row)
                for row in np.asarray(element.matrix)
            )
            translation = tuple(
                Fraction(float(value)) for value in np.asarray(element.offset)
            )
            return _AffineTransform(transform, translation)
        case PolynomialComposedCellGeometryElement():
            return _polynomial_affine_transform(element, dimension, budget=budget)
        case invalid:
            assert_never(invalid)


def _restriction(
    element: CellGeometryElement,
    root_element: CellGeometryElement,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]] | None:
    matrix = tuple(
        tuple(Fraction(int(i == j)) for j in range(dimension)) for i in range(dimension)
    )
    offset = (Fraction(0),) * dimension
    while element.element_id != root_element.element_id and isinstance(
        element,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        step = _affine_reference_step(element, dimension, budget=budget)
        if step is None:
            return None
        transform, translation = step.matrix, step.offset
        matrix = tuple(
            tuple(
                sum(
                    (transform[i][k] * matrix[k][j] for k in range(dimension)),
                    Fraction(0),
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        offset = tuple(
            sum((transform[i][k] * offset[k] for k in range(dimension)), translation[i])
            for i in range(dimension)
        )
        element = element.source_element
    if (
        element.element_id != root_element.element_id
        or algebra.coordinate_source_signature(element)
        != algebra.coordinate_source_signature(root_element)
    ):
        return None
    return matrix, offset


type PolynomialReferenceChart = tuple[
    reference_types_owner.FiniteElementSpec
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement,
    np.ndarray,
]


@checked
def polynomial_reference_chart(
    element: CellGeometryElement,
    root_element: CellGeometryElement,
    *,
    budget: CoordinateEnclosureBudget,
) -> PolynomialReferenceChart | None:
    """Retain each rational chart owner and coefficients, replacing only the root by identity."""

    if element.element_id == root_element.element_id:
        identity = budget.degree_one_coordinate_element(root_element.cell_kind)
        return identity, np.asarray(identity.reference_nodes, dtype=np.float64)
    if not isinstance(
        element,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        return None
    source = polynomial_reference_chart(
        element.source_element, root_element, budget=budget
    )
    if source is None:
        return None
    reference, controls = source
    if isinstance(element, RestrictedCellGeometryElement):
        return RestrictedCellGeometryElement(
            reference, element.cell_kind, element.matrix, element.offset
        ), controls
    numerator_dtype = (
        np.int64
        if any(a < 0 for row in element.chart_coefficients for a, _ in row)
        else np.uint64
    )
    numerators = np.asarray(
        [[a for a, _ in row] for row in element.chart_coefficients], dtype=numerator_dtype
    )
    denominators = np.asarray(
        [[b for _, b in row] for row in element.chart_coefficients], dtype=np.uint64
    )
    match element:
        case PolynomialComposedCellGeometryElement():
            return PolynomialComposedCellGeometryElement(
                reference, element.chart_element, numerators, denominators, budget=budget
            ), controls
        case RationalComposedCellGeometryElement():
            return RationalComposedCellGeometryElement(
                require_full_p1_source(reference),
                element.chart_element,
                numerators,
                denominators,
            ), controls
        case invalid:
            assert_never(invalid)


class PreparedPolynomialReferenceCharts:
    """Request-local reference charts bound to freshly hashed full element inputs."""

    def __init__(self) -> None:
        self._charts: dict[
            tuple[tuple[str, str], tuple[str, str]], PolynomialReferenceChart
        ] = {}

    @checked
    def prepare(
        self,
        element: CellGeometryElement,
        root_element: CellGeometryElement,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> PolynomialReferenceChart | None:
        """Reuse a composed reference element only after full current input binding."""
        key = algebra.element_identity(element, budget=budget), algebra.element_identity(root_element, budget=budget)
        prepared = self._charts.get(key)
        if prepared is None:
            prepared = polynomial_reference_chart(element, root_element, budget=budget)
            if prepared is None:
                return None
            budget.reserve(1, 128)
            prepared[1].setflags(write=False)
            self._charts[key] = prepared
        # A different consuming ledger must admit its own retained chart graph.
        budget.retain_owner(prepared[0])
        budget.retain_basis((key, prepared[1]))
        return prepared


def _target_cell_chart(
    state: EmbeddingState,
    block: CellBlock | PolyhedralBlock,
    element: CellGeometryElement,
    local: algebra.CoordinateSourceBank,
    cell_id: int,
    vertices: tuple[int, ...],
    parent_vertices: np.ndarray,
    region: int,
    root: _Root,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> Chart | None:
    equivalent = (
        None
        if tuple(int(value) for value in parent_vertices if value >= 0) != root.vertices
        else equivalent_coordinate_root(
            element, local, root.element, root.controls, budget=budget
        )
    )
    if equivalent is None:
        state.add("mapped_domain_root_expression", "violated", "cell", (int(cell_id),))
        return None
    chart = _restriction(element, equivalent, dimension, budget=budget)
    chain = element
    composed = False
    while chain.element_id != equivalent.element_id and isinstance(
        chain,
        (
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
        ),
    ):
        composed |= isinstance(
            chain,
            (
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
            ),
        )
        chain = chain.source_element
    if composed:
        chart = None
    reference = (
        polynomial_reference_chart(element, equivalent, budget=budget)
        if chart is None
        else None
    )
    if chart is None and reference is None:
        state.add("mapped_domain_root_expression", "violated", "cell", (int(cell_id),))
        return None
    elif region != root.region:
        state.add("mapped_domain_region_identity", "violated", "cell", (int(cell_id),))
        return None
    elif chart is not None:
        return AffineReferenceChart(
            block.name, int(cell_id), block.cell_kind, vertices, *chart
        )
    elif reference is not None:
        return ComposedReferenceChart(
            block.name, int(cell_id), block.cell_kind, vertices, *reference
        )
    return None


def _target_charts(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    roots: dict[int, _Root],
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[int, list[Chart]] | None:
    record = geometry.restriction_source
    if record is None:
        # rational independently declared identity is a root restriction result,
        # not an inference from matching dimensions, nodes or fitted values.
        if (
            cell_geometry_id(geometry) != cell_geometry_id(domain.source_geometry)
            or mesh.topology_id != domain.reference_mesh.topology_id
        ):
            state.add("mapped_domain_restriction_source", "unresolved", "mesh")
            return None
        if not np.array_equal(regions, domain.cell_regions):
            state.add("mapped_domain_region_identity", "violated", "mesh")
            return None
        dimension = domain.ambient_dimension
        matrix = tuple(
            tuple(Fraction(i == j) for j in range(dimension)) for i in range(dimension)
        )
        offset = (Fraction(0),) * dimension
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        return {
            int(cell_id): [
                AffineReferenceChart(
                    block.name,
                    int(cell_id),
                    block.cell_kind,
                    tuple(int(value) for value in vertex_ids[row]),
                    matrix,
                    offset,
                )
            ]
            for block in mesh.blocks
            for cell_id, row in zip(
                np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
            )
        }
    if (
        record.source_geometry_id != cell_geometry_id(domain.source_geometry)
        or record.source_topology_id != domain.reference_mesh.topology_id
    ):
        state.add("mapped_domain_source_identity", "violated", "mesh")
        return None
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    values = geometry.source_coordinates(budget=budget)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    parents = record.block_parent_cell_ids
    corners = record.block_parent_vertex_ids
    result: dict[int, list[Chart]] = {cell_id: [] for cell_id in roots}
    cursor = 0
    valid = True
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        local_values = tuple(
            tuple(values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for local, cell_id, row, parent, parent_vertices in zip(
            local_values,
            np.asarray(block.global_ids),
            np.asarray(block.vertices),
            np.asarray(parents[block.name]),
            np.asarray(corners[block.name]),
            strict=True,
        ):
            root = roots.get(int(parent))
            if root is None:
                state.add(
                    "mapped_domain_parent_identity", "violated", "cell", (int(cell_id),)
                )
                valid = False
                cursor += 1
                continue
            vertices = tuple(int(value) for value in vertex_ids[row])
            chart = _target_cell_chart(
                state,
                block,
                element,
                local,
                int(cell_id),
                vertices,
                parent_vertices,
                int(regions[cursor]),
                root,
                domain.ambient_dimension,
                budget=budget,
            )
            if chart is None:
                valid = False
            else:
                result[root.cell_id].append(chart)
            cursor += 1
    return result if valid else None


def _chart_mesh(
    charts: list[Chart],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec] | None:
    affine: list[AffineReferenceChart] = []
    for chart in charts:
        match chart:
            case ComposedReferenceChart():
                return exact_chart_mesh(charts, dimension, budget=budget)
            case AffineReferenceChart():
                affine.append(chart)
            case invalid:
                assert_never(invalid)
    return _affine_chart_mesh(affine, dimension, budget=budget)


class _AffineChartPositions(StrictModule):
    positions: dict[int, tuple[Fraction, ...]]
    grouped: dict[str, list[AffineReferenceChart]]


def _affine_chart_positions(
    charts: list[AffineReferenceChart],
) -> _AffineChartPositions | None:
    positions: dict[int, tuple[Fraction, ...]] = {}
    grouped: dict[str, list[AffineReferenceChart]] = {}
    for chart in charts:
        grouped.setdefault(chart.block, []).append(chart)
        for vertex, corner in zip(
            chart.vertices, reference_cell_topology(chart.kind).vertices, strict=True
        ):
            point = tuple(
                sum(
                    (
                        weight * Fraction(value)
                        for weight, value in zip(row, corner, strict=True)
                    ),
                    offset,
                )
                for row, offset in zip(chart.matrix, chart.offset, strict=True)
            )
            if any(Fraction(float(value)) != value for value in point) or (
                vertex in positions and positions[vertex] != point
            ):
                return None
            positions[vertex] = point
    return _AffineChartPositions(positions, grouped)


def _affine_chart_expressions_match(
    blocks: tuple[CellBlock, ...],
    grouped: dict[str, list[AffineReferenceChart]],
    points: np.ndarray,
    elements: Mapping[str, CellGeometryElement],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    variables = algebra.axes(dimension, budget=budget)
    for block in blocks:
        for chart, row in zip(
            grouped[block.name], np.asarray(block.vertices), strict=True
        ):
            arguments = variables
            if chart.kind == "pyramid":
                u, v, w = variables
                collapse = algebra.add(
                    algebra.constant(1, 3, budget=budget),
                    algebra.scale(w, -1, budget=budget),
                    budget=budget,
                )
                arguments = (
                    algebra.add(
                        algebra.multiply(u, collapse, budget=budget),
                        algebra.scale(w, Fraction(1, 2), budget=budget),
                        budget=budget,
                    ),
                    algebra.add(
                        algebra.multiply(v, collapse, budget=budget),
                        algebra.scale(w, Fraction(1, 2), budget=budget),
                        budget=budget,
                    ),
                    w,
                )
            expected = tuple(
                algebra.add(
                    algebra.constant(offset, dimension, budget=budget),
                    algebra.sum_polynomials(
                        tuple(
                            algebra.scale(variable, weight, budget=budget)
                            for variable, weight in zip(
                                arguments, row_matrix, strict=True
                            )
                        ),
                        budget=budget,
                    ),
                    budget=budget,
                )
                for row_matrix, offset in zip(chart.matrix, chart.offset, strict=True)
            )
            actual = algebra.coordinate_polynomials(
                elements[block.name], points[row], budget=budget
            )
            if actual != expected:
                return False
    return True


def _affine_chart_mesh(
    charts: list[AffineReferenceChart],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec] | None:
    prepared = _affine_chart_positions(charts)
    if prepared is None:
        return None
    positions, grouped = prepared.positions, prepared.grouped
    ids = tuple(sorted(positions))
    index = {vertex: row for row, vertex in enumerate(ids)}
    points = np.asarray(
        tuple(tuple(float(value) for value in positions[vertex]) for vertex in ids),
        dtype=np.float64,
    )
    blocks = tuple(
        CellBlock(
            name,
            members[0].kind,
            np.asarray(
                tuple(
                    tuple(index[vertex] for vertex in chart.vertices) for chart in members
                ),
                dtype=np.int32,
            ),
            global_ids=np.asarray(
                tuple(chart.cell_id for chart in members), dtype=np.int64
            ),
        )
        for name, members in sorted(
            grouped.items(),
            key=lambda item: (
                len(reference_cell_topology(item[1][0].kind).vertices),
                item[0],
            ),
        )
    )
    mesh = CellMesh(points, blocks, vertex_global_ids=np.asarray(ids, dtype=np.int64))
    elements = {
        block.name: coordinate_lagrange_element(block.cell_kind, 1) for block in blocks
    }
    geometry = CellGeometrySpec(
        elements, {block.name: block.vertices for block in blocks}, points
    )
    if not _affine_chart_expressions_match(
        blocks, grouped, points, elements, dimension, budget=budget
    ):
        return None
    return mesh, geometry


def _pyramid_reference_source(
    identity: reference_types_owner.FiniteElementSpec,
    nodal: reference_types_owner.FiniteElementSpec,
    coefficients: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _ChartSource:
    """Return a collapsed chart with coefficients inside its source simplex."""
    frame_scale: Fraction | None = None
    if any(min(row) < 0 or sum(row, Fraction(0)) > 1 for row in coefficients):
        # A reference-space pyramid need not fit the unit tetrahedron. An
        # enclosing binary frame puts normalized components in [0, 1/4];
        # their three-component sum is at most 3/4.
        magnitude = max(abs(value) for row in coefficients for value in row)
        exponent = max(
            0,
            magnitude.numerator.bit_length() - magnitude.denominator.bit_length(),
        )
        frame_scale = Fraction(1 << exponent)
        if frame_scale < magnitude:
            frame_scale *= 2
        step = 8 * frame_scale
        # Each reference scalar earns one addition and one division.
        budget.reserve(2 * sum(len(row) for row in coefficients))
        coefficients = tuple(
            tuple((value + frame_scale) / step for value in row) for row in coefficients
        )
    element = RationalComposedCellGeometryElement(
        identity,
        nodal,
        np.asarray(
            [[value.numerator for value in row] for row in coefficients],
            dtype=np.int64,
        ),
        np.asarray(
            [[value.denominator for value in row] for row in coefficients],
            dtype=np.uint64,
        ),
    )
    values = np.asarray(identity.reference_nodes, dtype=np.float64)
    if frame_scale is not None:
        values = values * float(8 * frame_scale) - float(frame_scale)
    return _ChartSource(element, values)


def _reference_chart_source(
    chart: Chart, dimension: int, *, budget: CoordinateEnclosureBudget
) -> _ChartSource:
    match chart:
        case ComposedReferenceChart():
            return _ChartSource(chart.reference_element, chart.reference_controls)
        case AffineReferenceChart():
            identity = coordinate_lagrange_element(
                "triangle" if dimension == 2 else "tetrahedron", 1
            )
            nodal = coordinate_lagrange_element(chart.kind, 1)
            coefficients = tuple(
                tuple(
                    sum(
                        (
                            weight * Fraction(float(x))
                            for weight, x in zip(row, corner, strict=True)
                        ),
                        shift,
                    )
                    for row, shift in zip(chart.matrix, chart.offset, strict=True)
                )
                for corner in np.asarray(nodal.reference_nodes)
            )
            if chart.kind == "pyramid":
                return _pyramid_reference_source(
                    identity, nodal, coefficients, budget=budget
                )
            element = PolynomialComposedCellGeometryElement(
                identity,
                nodal,
                np.asarray(
                    [[value.numerator for value in row] for row in coefficients],
                    dtype=np.int64,
                ),
                np.asarray(
                    [[value.denominator for value in row] for row in coefficients],
                    dtype=np.uint64,
                ),
                budget=budget,
            )
            values = np.asarray(identity.reference_nodes, dtype=np.float64)
            return _ChartSource(element, values)
        case invalid:
            assert_never(invalid)


@checked
def exact_chart_mesh(
    charts: list[Chart], dimension: int, *, budget: CoordinateEnclosureBudget
) -> tuple[CellMesh, CellGeometrySpec] | None:
    """RNE carrier plus the measured source-coordinate chart, never interpolation."""
    positions: dict[int, tuple[Fraction, ...]] = {}
    grouped: dict[str, list[Chart]] = {}
    elements: dict[str, CellGeometryElement] = {}
    controls: list[np.ndarray] = []
    routes: dict[str, np.ndarray] = {}
    count = 0
    for chart in charts:
        source = _reference_chart_source(chart, dimension, budget=budget)
        element, values = source.element, source.controls
        if chart.block in elements:
            if elements[chart.block].element_id != element.element_id:
                return None
        else:
            elements[chart.block] = element
        images = algebra.coordinate_corner_images(element, values, budget=budget)
        if images is None:
            return None
        for vertex, point in zip(chart.vertices, images, strict=True):
            if vertex in positions and positions[vertex] != point:
                return None
            positions[vertex] = tuple(
                algebra.require_rational_reference_scalar(value) for value in point
            )

        grouped.setdefault(chart.block, []).append(chart)
        route = np.arange(count, count + values.shape[0], dtype=np.int64)
        routes[chart.block] = (
            np.vstack((routes[chart.block], route))
            if chart.block in routes
            else route[None]
        )
        controls.append(values)
        count += values.shape[0]
    ids = tuple(sorted(positions))
    index = {vertex: row for row, vertex in enumerate(ids)}
    points = np.asarray(
        tuple(tuple(float(value) for value in positions[vertex]) for vertex in ids),
        dtype=np.float64,
    )
    blocks = tuple(
        CellBlock(
            name,
            members[0].kind,
            np.asarray(
                tuple(
                    tuple(index[vertex] for vertex in chart.vertices) for chart in members
                ),
                dtype=np.int32,
            ),
            global_ids=np.asarray(
                tuple(chart.cell_id for chart in members), dtype=np.int64
            ),
        )
        for name, members in sorted(
            grouped.items(),
            key=lambda item: (
                len(reference_cell_topology(item[1][0].kind).vertices),
                item[0],
            ),
        )
    )
    mesh = CellMesh(points, blocks, vertex_global_ids=np.asarray(ids, dtype=np.int64))
    return mesh, CellGeometrySpec(elements, routes, np.concatenate(controls))


def _root_domain(
    domain: mapped_reference_domain_owner.MappedReferenceDomain, root: _Root
) -> tuple[domain_coverage_owner.PiecewiseLinearDomain, tuple[int, ...]]:
    topology = reference_cell_topology(root.kind)
    facets = []
    owners = []
    for index, face in enumerate(topology.entities[domain.ambient_dimension - 1]):
        pieces = (
            (face,)
            if len(face) <= 3
            else ((face[0], face[1], face[2]), (face[0], face[2], face[3]))
        )
        facets.extend(pieces)
        owners.extend(index for _ in pieces)
    declared = domain_coverage_owner.PiecewiseLinearDomain(
        np.asarray(topology.vertices, dtype=np.float64),
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (len(facets), 1)),
        ("root",),
        source_id=f"{domain.source_id}:root:{root.cell_id}",
    )
    return declared, tuple(owners)


def _facet_identities(
    mesh: CellMesh,
    *,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[dict[int, tuple[int, ...]], dict[tuple[int, ...], int]]:
    facets = mesh_facets(mesh, geometry=geometry, budget=budget)
    entities = facet_entity_ids(mesh, facets.rows)
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    identities = {
        int(entity): tuple(sorted(int(value) for value in vertices[row[row >= 0]]))
        for entity, row in zip(entities, facets.rows, strict=True)
    }
    return identities, {key: entity for entity, key in identities.items()}


def _compose_fragment_relations(
    relations: dict[domain_coverage_owner.CoverageOverlapKey, Fraction],
    chart_mesh: CellMesh,
    chart_coverage: domain_coverage_owner.DomainCoverageCertificate,
    face_owners: tuple[int, ...],
    root: _Root,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    source_coverage: domain_coverage_owner.DomainCoverageCertificate,
    source_facets: dict[tuple[int, ...], int],
    target_facets: dict[tuple[int, ...], int],
    covering: set[int],
    *,
    budget: CoordinateEnclosureBudget,
    vertex_images: tuple[int, ...] | None = None,
) -> None:
    chart_facets, _ = _facet_identities(chart_mesh, budget=budget)
    root_faces = reference_cell_topology(root.kind).entities[domain.ambient_dimension - 1]
    source_rows: dict[int, list[tuple[int, tuple[int, ...], float]]] = {}
    source_overlaps = source_coverage.facet_source_overlaps
    for facet, source, axes, upper in zip(
        source_overlaps.facets.tolist(),
        source_overlaps.sources.tolist(),
        source_overlaps.axes.tolist(),
        source_overlaps.upper.tolist(),
        strict=True,
    ):
        source_rows.setdefault(facet, []).append((source, tuple(axes), upper))
    chart_overlaps = chart_coverage.facet_source_overlaps
    for chart_facet, piece in zip(
        chart_overlaps.facets.tolist(), chart_overlaps.sources.tolist(), strict=True
    ):
        vertices = chart_facets[chart_facet]
        target = target_facets[
            vertices if vertex_images is None else tuple(sorted(vertex_images[v] for v in vertices))
        ]
        if target not in covering:
            continue
        source_face = tuple(
            sorted(root.vertices[vertex] for vertex in root_faces[face_owners[piece]])
        )
        for source, axes, upper in source_rows.get(source_facets[source_face], ()):
            total = abs(
                signed_measure(
                    project(
                        rational_points(
                            np.asarray(domain.reference_domain.vertices)[
                                domain.reference_domain.facets[source]
                            ],
                            budget=budget,
                        ),
                        axes,
                        budget=budget,
                    ),
                    budget=budget,
                )
            )
            key: domain_coverage_owner.CoverageOverlapKey = (
                target,
                source,
                axes,
                "candidate",
                "reference",
            )
            relations[key] = min(
                algebra.require_rational_reference_scalar(total),
                relations.get(key, Fraction(0)) + Fraction(upper),
            )


def _add_measure(
    totals: list[Fraction | None], region: int, measure: Fraction | None
) -> None:
    current = totals[region]
    totals[region] = None if current is None or measure is None else current + measure


class _RootPartition(StrictModule):
    mesh: CellMesh
    coverage: domain_coverage_owner.DomainCoverageCertificate
    face_owners: tuple[int, ...] = eqx.field(static=True)
    vertex_images: tuple[int, ...] | None = eqx.field(static=True)


def _periodic_source_premises(
    state: EmbeddingState,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    mesh: CellMesh,
    source_embedding: global_embedding_owner.GlobalEmbeddingCertificate,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate,
) -> None:
    source_periodic = domain.reference_mesh.periodic_topology
    target_periodic = mesh.periodic_topology
    if (source_periodic is None) != (target_periodic is None):
        state.add("mapped_domain_periodic_source_identity", "violated", "mesh")
    elif source_periodic is not None and target_periodic is not None:
        source_cell, target_cell = source_periodic.cell, target_periodic.cell
        same_group = (
            isinstance(source_cell, periodic_cell_owner.PeriodicCell)
            and isinstance(target_cell, periodic_cell_owner.PeriodicCell)
            and source_cell.cell_id == target_cell.cell_id
        ) or (
            isinstance(source_cell, periodic_exact_owner.PeriodicIsometryGroup)
            and isinstance(target_cell, periodic_exact_owner.PeriodicIsometryGroup)
            and source_cell.group_id == target_cell.group_id
        )
        if not same_group:
            state.add("mapped_domain_periodic_source_identity", "violated", "mesh")
        for certificate in (source_embedding, embedding):
            if "periodic_mapped_trace_equivariance" not in certificate.evaluated_checks:
                state.add("mapped_domain_periodic_trace_premise", "unresolved", "mesh")
        # The whole source reference partition below covers every declared
        # facet, including seam gauges. Physical walls/material traces and
        # quotient-glued seams are not silently omitted from continuous bounds.


def _direct_root_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    roots: dict[int, _Root],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[int, Fraction | None]:
    target_elements, target_routes, _ = geometry.resolve(mesh, budget=budget)
    target_values = geometry.source_coordinates(budget=budget)
    direct: dict[int, Fraction | None] = {cell_id: Fraction(0) for cell_id in roots}
    record = geometry.restriction_source
    parents = (
        {block.name: np.asarray(block.global_ids) for block in mesh.blocks}
        if record is None
        else record.block_parent_cell_ids
    )
    for block, element, route in zip(
        mesh.blocks, target_elements, target_routes, strict=True
    ):
        local_values = tuple(
            tuple(target_values[index] for index in row)
            for row in np.asarray(route, dtype=np.int64)
        )
        for parent, local in zip(
            np.asarray(parents[block.name]), local_values, strict=True
        ):
            measure = domain_coverage_owner.polynomial_cell_measure(
                element, local, block.cell_kind, dimension, budget=budget
            )
            current = direct[int(parent)]
            direct[int(parent)] = (
                None if measure is None or current is None else current + measure
            )
    return direct


def _root_partition(
    state: EmbeddingState,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    root: _Root,
    charts: list[Chart],
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
    partitions: dict[tuple[object, ...], _RootPartition],
) -> _RootPartition | None:
    cell_id = root.cell_id
    if not charts:
        state.add("mapped_domain_missing_root", "violated", "source_cell", (cell_id,))
        return None
    # Coverage of a reference partition is independent of global cell/vertex
    # names and of the physical root map. Normalize only those names; retain
    # every exact affine chart, its incidence and its block grouping.
    vertex_images: tuple[int, ...] | None = None
    partition_key: tuple[object, ...] | None = None
    if all(isinstance(chart, AffineReferenceChart) for chart in charts):
        affine = [chart for chart in charts if isinstance(chart, AffineReferenceChart)]
        vertex_images = tuple(sorted({vertex for chart in affine for vertex in chart.vertices}))
        vertices = {vertex: row for row, vertex in enumerate(vertex_images)}
        blocks = {name: f"chart_{row}" for row, name in enumerate(sorted({chart.block for chart in affine}))}
        normalized = [
            AffineReferenceChart(
                blocks[chart.block], row, chart.kind,
                tuple(vertices[vertex] for vertex in chart.vertices),
                chart.matrix, chart.offset,
            )
            for row, chart in enumerate(affine)
        ]
        partition_key = (
            root.kind,
            tuple(
                (chart.block, chart.kind, chart.vertices, chart.matrix, chart.offset)
                for chart in normalized
            ),
        )
        budget.retain_basis((partition_key,))
        cached = partitions.get(partition_key)
        if cached is not None:
            return _RootPartition(cached.mesh, cached.coverage, cached.face_owners, vertex_images)
        charts = list(normalized)
    chart_view = _chart_mesh(charts, domain.ambient_dimension, budget=budget)
    if chart_view is None:
        state.add(
            "mapped_domain_chart_representation",
            "unresolved",
            "cell",
            tuple(chart.cell_id for chart in charts),
        )
        return None
    chart_mesh, chart_geometry = chart_view
    chart_validity = certify_cell_geometry_validity(
        chart_geometry, mesh=chart_mesh, budget=budget
    )
    chart_embedding = global_embedding_owner.certify_global_embedding(
        chart_mesh, chart_geometry, chart_validity, limits=limits, budget=budget
    )
    declared, face_owners = _root_domain(domain, root)
    chart_coverage = domain_coverage_owner.certify_domain_coverage(
        chart_mesh,
        chart_geometry,
        declared,
        np.zeros(sum(block.cell_count for block in chart_mesh.blocks), dtype=np.int64),
        embedding=chart_embedding,
        limits=limits,
        budget=budget,
    )
    partition = _RootPartition(chart_mesh, chart_coverage, face_owners, vertex_images)
    if partition_key is not None and chart_coverage.status == "certified":
        budget.retain_owner(partition)
        partitions[partition_key] = partition
    return partition


def _affine_polynomial_image(
    coordinates: tuple[algebra.Polynomial, ...],
    transform: _AffineTransform,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[algebra.Polynomial, ...]:
    dimension = len(coordinates)
    return tuple(
        algebra.add(
            algebra.constant(offset, dimension, budget=budget),
            algebra.sum_polynomials(
                tuple(
                    algebra.scale(value, weight, budget=budget)
                    for value, weight in zip(coordinates, row, strict=True)
                ),
                budget=budget,
            ),
            budget=budget,
        )
        for row, offset in zip(transform.matrix, transform.offset, strict=True)
    )


def _common_affine_source(
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    roots: dict[int, _Root],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_AffineTransform, Fraction] | None:
    """Prove one law against every coefficient, never against fitted corner samples."""
    dimension = domain.ambient_dimension
    reference = CellGeometrySpec.affine(domain.reference_mesh, budget=budget)
    elements, routes, _ = reference.resolve(domain.reference_mesh, budget=budget)
    bank = cast(
        tuple[tuple[Fraction, ...], ...],
        algebra.prepared_coordinate_source_bank(reference, budget=budget),
    )
    source_elements, source_routes, _ = domain.source_geometry.resolve(
        domain.reference_mesh, budget=budget
    )
    standard_source = all(
        isinstance(element, reference_types_owner.FiniteElementSpec)
        and algebra.coordinate_source_signature(element)
        == algebra.coordinate_source_signature(coordinate_lagrange_element(block.cell_kind, 1))
        for block, element in zip(domain.reference_mesh.blocks, source_elements, strict=True)
    )
    forward: _AffineTransform | None = None
    inverse: _AffineTransform | None = None
    determinant = Fraction(0)
    zero = (0,) * dimension
    linear = tuple(
        tuple(int(axis == column) for axis in range(dimension))
        for column in range(dimension)
    )
    for block, element, route in zip(
        domain.reference_mesh.blocks, elements, routes, strict=True
    ):
        for cell_id, indices in zip(
            np.asarray(block.global_ids), np.asarray(route), strict=True
        ):
            actual = roots[int(cell_id)].polynomial
            declared = algebra.coordinate_polynomials(
                element, tuple(bank[int(index)] for index in indices), budget=budget
            )
            if declared is None or any(
                not isinstance(value, Fraction)
                for polynomial in (*actual, *declared)
                for value in polynomial.values()
            ):
                return None
            if forward is None:
                # Solve R_linear.T @ A.T = P_linear.T. The later full
                # polynomial equality is the premise, not this candidate solve.
                coefficients = tuple(
                    tuple(cast(Fraction, value.get(exponent, Fraction(0))) for value in declared)
                    for exponent in linear
                )
                right = tuple(
                    tuple(cast(Fraction, value.get(exponent, Fraction(0))) for value in actual)
                    for exponent in linear
                )
                candidate = prepare_exact_small_linear_actions(
                    coefficients, right, budget=budget
                )
                if candidate.actions is None:
                    return None
                matrix = tuple(zip(*candidate.actions, strict=True))
                untranslated = _affine_polynomial_image(
                    declared,
                    _AffineTransform(matrix, (Fraction(0),) * dimension),
                    budget=budget,
                )
                budget.reserve(dimension)
                offset = tuple(
                    physical.get(zero, Fraction(0)) - mapped.get(zero, Fraction(0))
                    for physical, mapped in zip(actual, untranslated, strict=True)
                )
                if any(not isinstance(value, Fraction) for value in offset):
                    return None
                forward = _AffineTransform(matrix, cast(tuple[Fraction, ...], offset))
                solved = prepare_exact_small_linear_actions(
                    matrix,
                    tuple(
                        tuple(Fraction(row == column) for column in range(dimension))
                        for row in range(dimension)
                    ),
                    budget=budget,
                )
                if solved.actions is None:
                    return None
                determinant = solved.determinant
                translated = _affine_polynomial_image(
                    tuple(
                        algebra.constant(-value, dimension, budget=budget)
                        for value in offset
                    ),
                    _AffineTransform(solved.actions, (Fraction(0),) * dimension),
                    budget=budget,
                )
                inverse_offset = tuple(value.get(zero, Fraction(0)) for value in translated)
                if any(not isinstance(value, Fraction) for value in inverse_offset):
                    return None
                inverse = _AffineTransform(
                    solved.actions, cast(tuple[Fraction, ...], inverse_offset)
                )
                if standard_source:
                    source_bank = algebra.prepared_coordinate_source_bank(
                        domain.source_geometry, budget=budget
                    )
                    if all(isinstance(value, Fraction) for point in source_bank for value in point):
                        # Both actual routes use the identical standard nodal
                        # basis. Prove every routed coefficient, not merely the
                        # geometric corners: linearity and partition of unity
                        # then lift A*r+b to the complete coordinate polynomial.
                        pairs: set[tuple[int, int]] = set()
                        for declared_route, actual_route in zip(routes, source_routes, strict=True):
                            if np.asarray(declared_route).shape != np.asarray(actual_route).shape:
                                return None
                            for declared_row, actual_row in zip(
                                np.asarray(declared_route), np.asarray(actual_route), strict=True
                            ):
                                budget.reserve(len(declared_row))
                                for declared_index, actual_index in zip(
                                    declared_row, actual_row, strict=True
                                ):
                                    pair = (int(declared_index), int(actual_index))
                                    budget.retain_basis((pair,))
                                    pairs.add(pair)
                        numeric_bank = cast(tuple[tuple[Fraction, ...], ...], source_bank)
                        numerator_bits, denominator_bits = fraction_matrix_profile(
                            (bank, numeric_bank, forward.matrix, (forward.offset,)), budget
                        )
                        reserve_fraction_work(
                            budget, len(pairs) * dimension * (2 * dimension + 1),
                            len(pairs) * (dimension + 1),
                            (dimension + 2) * (numerator_bits + denominator_bits + 2),
                        )
                        for declared_index, actual_index in pairs:
                            expected_row = tuple(
                                sum(
                                    (weight * value for weight, value in zip(
                                        row, bank[declared_index], strict=True
                                    )),
                                    offset,
                                )
                                for row, offset in zip(forward.matrix, forward.offset, strict=True)
                            )
                            budget.reserve(dimension)
                            if numeric_bank[actual_index] != expected_row:
                                return None
                        return inverse, abs(determinant)
            expected = _affine_polynomial_image(declared, forward, budget=budget)
            budget.reserve(sum(len(value) for value in (*actual, *expected)))
            if actual != expected:
                return None
    return None if inverse is None else (inverse, abs(determinant))


def _inverse_affine_target(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    inverse: _AffineTransform,
    *,
    limits: MeshCertificateLimits,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec] | None:
    """Retain rational inverse images under the declared coverage theorem limits.

    The exact source includes its work bound in scientific identity. The caller's
    ledger pays every operation, but its request-wide allowance is not that bound.
    """
    dimension = domain.ambient_dimension
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    if any(
        not isinstance(element, reference_types_owner.FiniteElementSpec)
        or algebra.coordinate_source_signature(element)
        != algebra.coordinate_source_signature(coordinate_lagrange_element(block.cell_kind, 1))
        for block, element in zip(mesh.blocks, elements, strict=True)
    ):
        return None
    source_bank = algebra.prepared_coordinate_source_bank(geometry, budget=budget)
    if any(not isinstance(value, Fraction) for point in source_bank for value in point):
        return None
    bank = cast(tuple[tuple[Fraction, ...], ...], source_bank)
    numerator_bits, denominator_bits = fraction_matrix_profile(
        (bank, inverse.matrix, (inverse.offset,)), budget
    )
    bits = (dimension + 2) * (numerator_bits + denominator_bits + 2)
    reserve_fraction_work(
        budget,
        len(bank) * dimension * (2 * dimension + 1),
        len(bank) * (dimension + 1),
        bits,
    )
    pulled = tuple(
        tuple(
            sum(
                (weight * value for weight, value in zip(row, point, strict=True)),
                offset,
            )
            for row, offset in zip(inverse.matrix, inverse.offset, strict=True)
        )
        for point in bank
    )
    positions: dict[int, tuple[Fraction, ...]] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, indices in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            if block.cell_kind in ("triangle", "tetrahedron"):
                # On the admitted standard simplex P1 basis there are no
                # higher-degree coefficients. Every actual bank row was
                # transformed above, and sum(N_i)=1, hence interpolation
                # commutes exactly with this affine map on the entire cell.
                # Tensor-product P1 retains the generic degree check below:
                # its mixed coefficients need not vanish.
                budget.reserve(len(indices))
                for vertex, index in zip(vertices, indices, strict=True):
                    position = pulled[int(index)]
                    if int(vertex) in positions and positions[int(vertex)] != position:
                        return None
                    positions[int(vertex)] = position
                continue
            # Tensor-product P1 also commutes with the bank transformation.
            # Retain only its independent affine-degree admission; unlike
            # simplex P1, standard Q1 can contain nonzero mixed coefficients.
            with budget.temporary_scope():
                original = algebra.coordinate_polynomials(
                    element, tuple(bank[int(index)] for index in indices), budget=budget
                )
                if original is None:
                    return None
                budget.reserve(sum(len(value) for value in original))
                matches = not any(
                    sum(exponent) > 1 for value in original for exponent in value
                )
                del original
                if not matches:
                    return None
                for vertex, index in zip(vertices, indices, strict=True):
                    position = pulled[int(index)]
                    if int(vertex) in positions and positions[int(vertex)] != position:
                        return None
                    positions[int(vertex)] = position
    if len(positions) != mesh.coordinates.shape[0]:
        return None
    # A literal simplex frame is only the existing rational source's coefficient
    # carrier. Its identity binds both actual source and target, not historical IDs.
    identity = canonical_fingerprint(
        (
            "inverse-common-affine",
            _source_snapshot_id(domain),
            cell_geometry_id(geometry),
            tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in inverse.matrix
            ),
            tuple((value.numerator, value.denominator) for value in inverse.offset),
        )
    )
    name = f"inverse-affine-frame-{identity}"
    frame_kind = "triangle" if dimension == 2 else "tetrahedron"
    frame = np.vstack((np.eye(dimension), np.zeros((1, dimension))))
    parent = CellGeometrySpec(
        {name: coordinate_lagrange_element(frame_kind, 1)},
        {name: np.arange(dimension + 1, dtype=np.int32)[None]},
        frame,
        budget=budget,
    )
    reserve_fraction_work(budget, len(pulled) * (dimension + 1), len(pulled), bits * 2)
    actions = tuple(
        tuple(
            (index, value.numerator, value.denominator)
            for index, value in enumerate((*point, 1 - sum(point, Fraction(0))))
            if value
        )
        for point in pulled
    )
    element_map = dict(zip((block.name for block in mesh.blocks), elements, strict=True))
    route_map = dict(zip((block.name for block in mesh.blocks), routes, strict=True))
    exact = ExactRationalCellGeometrySource(
        parent,
        actions,
        rational_source_layouts(element_map, route_map),
        maximum_work=limits.maximum_work_units,
        maximum_bits=bits * 2,
        budget=budget,
    )
    pulled_geometry = CellGeometrySpec(
        element_map,
        route_map,
        np.asarray(tuple(tuple(float(value) for value in point) for point in pulled)),
        exact_source=exact,
        budget=budget,
    )
    pulled_mesh = mesh.with_coordinates(
        np.asarray(
            tuple(tuple(float(value) for value in positions[index]) for index in range(len(positions)))
        ),
        numeric_version=identity,
    )
    return pulled_mesh, pulled_geometry


def _independent_affine_coverage(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    roots: dict[int, _Root],
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate,
    requested: tuple[Fraction, ...],
    premises: list[str],
    *,
    budget: CoordinateEnclosureBudget,
) -> domain_coverage_owner.DomainCoverageCertificate | None:
    # Explicit root-parent declarations and every periodic path retain the
    # original strict route. Unsupported affine laws do not add positive evidence.
    if (
        geometry.restriction_source is not None
        or domain.ambient_dimension not in (2, 3)
        or mesh.periodic_topology is not None
        or domain.reference_mesh.periodic_topology is not None
    ):
        return None
    common = _common_affine_source(domain, roots, budget=budget)
    if common is None:
        return None
    inverse, determinant = common
    target = _inverse_affine_target(
        mesh, geometry, domain, inverse, limits=limits, budget=budget
    )
    if target is None:
        return None
    pulled_mesh, pulled_geometry = target
    validity = certify_cell_geometry_validity(
        pulled_geometry, mesh=pulled_mesh, budget=budget
    )
    pulled_embedding = global_embedding_owner.certify_global_embedding(
        pulled_mesh, pulled_geometry, validity, limits=limits, budget=budget
    )
    premises.extend((embedding.certificate_id, pulled_embedding.certificate_id))
    coverage = domain_coverage_owner.certify_domain_coverage(
        pulled_mesh,
        pulled_geometry,
        domain.reference_domain,
        regions,
        embedding=pulled_embedding,
        limits=limits,
        budget=budget,
    )
    premises.append(coverage.certificate_id)
    # Preserve the actual failed checks (including resource receipts), not only
    # a generic dependency failure. Target entity IDs survive the pullback.
    state.findings.extend(pulled_embedding.findings)
    state.findings.extend(coverage.findings)
    for check, certificate in (
        ("embedding_premise", pulled_embedding),
        ("mapped_domain_reference_coverage_premise", coverage),
    ):
        if certificate.status != "certified":
            state.add(check, certificate.status, "mesh")
    measured, _ = domain_coverage_owner.mapped_coverage_measures(
        state, pulled_mesh, pulled_geometry, regions, budget=budget
    )
    budget.reserve(len(measured))
    achieved = tuple(None if value is None else determinant * value for value in measured)
    achieved = (*achieved, *(Fraction(0) for _ in range(len(requested) - len(achieved))))
    overlaps = coverage.facet_source_overlaps
    budget.reserve(len(overlaps.facets))
    relations: dict[domain_coverage_owner.CoverageOverlapKey, Fraction] = {
        (int(facet), int(source), tuple(int(axis) for axis in axes), "candidate", "reference"):
        Fraction(float(upper))
        for facet, source, axes, upper in zip(
            overlaps.facets, overlaps.sources, overlaps.axes, overlaps.upper, strict=True
        )
    }
    complete = (
        not state.findings
        and embedding.status == pulled_embedding.status == coverage.status == "certified"
    )
    return domain_coverage_owner.coverage_result(
        binding,
        embedding,
        domain,
        state,
        requested,
        achieved,
        domain.facets.shape[0] if complete else 0,
        relations,
        tuple(premises),
        expression_work=active_expression_work(budget=budget),
    )


@checked
def certify_mapped_reference_coverage(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: mapped_reference_domain_owner.MappedReferenceDomain,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: global_embedding_owner.GlobalEmbeddingCertificate,
    *,
    budget: CoordinateEnclosureBudget,
) -> domain_coverage_owner.DomainCoverageCertificate:
    requested = domain.exact_region_measures(budget=budget)
    _, source_embedding, reference_coverage = certify_mapped_reference_source(
        domain, limits, budget=budget
    )
    premises = [source_embedding.certificate_id, reference_coverage.certificate_id]
    _periodic_source_premises(state, domain, mesh, source_embedding, embedding)
    for check, certificate in (
        ("mapped_domain_source_embedding_premise", source_embedding),
        ("mapped_domain_reference_coverage_premise", reference_coverage),
    ):
        if certificate.status != "certified":
            state.add(check, certificate.status, "mesh")
    roots = source_roots(domain, budget=budget)
    try:
        # Close this real proof phase before constructing a refusal finding;
        # the enclosing owner's measurement has not unwound at this catch.
        with budget.measurement():
            independent = _independent_affine_coverage(
                state, mesh, geometry, domain, roots, regions, limits, binding, embedding,
                requested, premises, budget=budget,
            )
    except CoordinateEnclosureResourceError as error:
        try:
            state.add(
                "mapped_coverage_resource_budget",
                "unresolved",
                "mesh",
                resource_error=error,
                expression_budget=budget,
            )
            return domain_coverage_owner.coverage_result(
                binding, embedding, domain, state, requested,
                tuple(None for _ in requested), 0, {}, tuple(premises),
                expression_work=active_expression_work(budget=budget),
            )
        except CoordinateEnclosureResourceError as reporting_error:
            error.add_note(
                f"Completed mapped coverage premise certificates: {tuple(premises)!r}"
            )
            raise error from reporting_error
    if independent is not None:
        return independent
    charts = _target_charts(state, mesh, geometry, regions, domain, roots, budget=budget)
    achieved: list[Fraction | None] = [Fraction(0) for _ in domain.region_ids]
    relations: dict[domain_coverage_owner.CoverageOverlapKey, Fraction] = {}
    _, source_facets = _facet_identities(domain.reference_mesh, budget=budget)
    _, target_facets = _facet_identities(mesh, geometry=geometry, budget=budget)
    facets = mesh_facets(mesh, geometry=geometry, budget=budget)
    entities = facet_entity_ids(mesh, facets.rows)
    covering = set(int(value) for value in entities[facets.boundary])
    for group in np.unique(facets.group):
        rows = np.flatnonzero(facets.group == group)
        if (
            rows.size == 2
            and regions[facets.cells[rows[0]]] != regions[facets.cells[rows[1]]]
        ):
            covering.add(int(entities[rows[0]]))
    if charts is None:
        reported, _ = domain_coverage_owner.mapped_coverage_measures(
            state, mesh, geometry, regions, budget=budget
        )
        actual = (
            *reported,
            *(Fraction(0) for _ in range(len(domain.region_ids) - len(reported))),
        )
        return domain_coverage_owner.coverage_result(
            binding,
            embedding,
            domain,
            state,
            requested,
            actual,
            0,
            relations,
            tuple(premises),
            expression_work=active_expression_work(budget=budget),
        )
    direct = _direct_root_measures(
        mesh, geometry, roots, domain.ambient_dimension, budget=budget
    )
    complete = (
        source_embedding.status
        == reference_coverage.status
        == embedding.status
        == "certified"
    )
    partitions: dict[tuple[object, ...], _RootPartition] = {}
    for cell_id, root in sorted(roots.items()):
        partition = _root_partition(
            state, domain, root, charts[cell_id], limits, budget=budget, partitions=partitions
        )
        if partition is None:
            _add_measure(achieved, root.region, direct[cell_id])
            complete = False
            continue
        chart_mesh, chart_coverage, face_owners = (
            partition.mesh,
            partition.coverage,
            partition.face_owners,
        )
        premises.append(chart_coverage.certificate_id)
        if chart_coverage.status != "certified":
            state.add(
                "mapped_domain_root_coverage_premise",
                chart_coverage.status,
                "cell",
                tuple(chart.cell_id for chart in charts[cell_id]),
            )
            _add_measure(achieved, root.region, direct[cell_id])
            complete = False
            continue
        _compose_fragment_relations(
            relations,
            chart_mesh,
            chart_coverage,
            face_owners,
            root,
            domain,
            reference_coverage,
            source_facets,
            target_facets,
            covering,
            budget=budget,
            vertex_images=partition.vertex_images,
        )
        actual = direct[cell_id]
        if actual is None:
            # rational coefficient/root identity plus the independently certified
            # reference partition permits integration over the source root,
            # including rational source restrictions whose child polynomial is
            # absent. This is the source partition/change-of-variables result.
            jacobian = tuple(
                tuple(
                    algebra.derivative(value, axis, budget=budget)
                    for axis in range(domain.ambient_dimension)
                )
                for value in root.polynomial
            )
            actual = integrate(
                algebra.determinant(jacobian, budget=budget),
                map_domain(root.kind),
                domain.ambient_dimension,
                budget=budget,
            )
        _add_measure(achieved, root.region, actual)
    return domain_coverage_owner.coverage_result(
        binding,
        embedding,
        domain,
        state,
        requested,
        tuple(achieved),
        domain.facets.shape[0] if complete else 0,
        relations,
        tuple(premises),
        expression_work=active_expression_work(budget=budget),
    )
