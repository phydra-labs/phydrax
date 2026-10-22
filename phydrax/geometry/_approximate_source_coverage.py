#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify bounded CAD-region deviation from oriented boundary homotopies.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "approximate_source_embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The supplied mesh-bound global embedding certificate is not certified."
      }
    ],
    "conclusion": "Carry its violated or unresolved verdict as the missing disjointness obligation.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ]
  },
  {
    "checks": [
      "approximate_source_fidelity_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The source-revision-bound fidelity certificate must certify both directions over the complete mesh boundary."
      }
    ],
    "conclusion": "An unproved or scoped fidelity certificate cannot imply whole-region coverage.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
      "url": ""
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ]
  },
  {
    "checks": [
      "approximate_source_single_region_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "This route requires exactly one independently declared source region."
      }
    ],
    "conclusion": "A multiregion source remains unresolved rather than inferring exclusive region assignments.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "approximate_source_region_assignment"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each mesh cell must have an explicit source region index; -1 denotes an unassigned cell."
      }
    ],
    "conclusion": "Unassigned identified cells leave the region coverage unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "owner_local_approximate_source_coverage_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The mesh has owner-local storage without the global oriented boundary required by this route."
      }
    ],
    "conclusion": "A local partition cannot conclude whole-region deviation.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "approximate_source_boundary_theorem_premise"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "The continuous source-distance upper bound encloses the complete straight boundary homotopy within the requested tolerance."
      },
      {
        "tag": "X",
        "statement": "The independently recomputed source projection must have signed degree one on every component."
      },
      {
        "tag": "T",
        "statement": "Degree is invariant off the swept homotopy image, so the symmetric difference is contained in the swept boundary tube."
      }
    ],
    "conclusion": "An unavailable boundary projection argument leaves approximate source coverage unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Allen Hatcher, Algebraic Topology (2002)",
      "locator": "Section 2.2, Degree; Section 2.B, Jordan separation",
      "url": "https://math.uchicago.edu/~farb/algtop2023/Hatcher.pdf"
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "approximate_source_boundary_degree_binding"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The recomputed oriented boundary degree ID must equal the complete source-fidelity degree ID."
      }
    ],
    "conclusion": "Unbound degree evidence leaves approximate source coverage unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "approximate_source_expression_resource"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The coordinate budget refuses the guarded homotopy phase at its work or storage allowance."
      }
    ],
    "conclusion": "Approximate source coverage remains unresolved with the refusing phase quantities.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_approximate_source_coverage.certify_approximate_source_coverage",
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

import math
from typing import final, Self

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._domain_planar_boundary as domain_planar_boundary_owner

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ..typing import checked, HostInt64, parse
from ._embedding_state import (
    coordinate_certificate_request,
    EmbeddingState,
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from ._global_embedding import (
    cell_global_ids,
    coordinate_scope,
    GlobalEmbeddingCertificate,
)
from ._mesh_certificates import (
    certificate_status,
    MeshCellDim,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._source_boundary_records import SourceBoundaryQuery
from . import _source_fidelity as source_fidelity_owner


@final
class ApproximateSourceCoverageCertificate(StrictModule, NonTrainableState):
    """Bound Hausdorff deviation of a mesh region from an independently declared CAD region.

    An oriented homotopy of the entire boundary, positive cell orientation and
    global embedding put the symmetric difference in the certified source-boundary
    tube. ``distance_upper`` bounds region Hausdorff deviation; region
    measures and boundary-equality evidence use separate certificates.
    """

    __strict_contract__ = True

    binding: MeshCertificateBinding
    embedding: GlobalEmbeddingCertificate
    fidelity: source_fidelity_owner.SourceFidelityCertificate
    domain: domain_planar_boundary_owner.CadSourceDomain
    region_ids: tuple[str, ...] = eqx.field(static=True)
    cell_regions: HostInt64[MeshCellDim]
    boundary_degree_id: str | None = eqx.field(static=True)
    distance_upper: float = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    status: MeshCertificateStatus = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        embedding: GlobalEmbeddingCertificate,
        fidelity: source_fidelity_owner.SourceFidelityCertificate,
        domain: domain_planar_boundary_owner.CadSourceDomain,
        cell_regions: ArrayLike,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        boundary_degree_id: str | None,
        distance_upper: float,
    ) -> None:
        raw_regions = np.asarray(cell_regions)
        if raw_regions.ndim != 1 or raw_regions.dtype.kind not in "iu":
            raise ValueError(
                "ApproximateSourceCoverageCertificate.cell_regions must be an integer vector; "
                f"got shape {raw_regions.shape}, dtype {raw_regions.dtype}"
            )
        regions = parse(
            np.asarray(raw_regions, dtype=np.int64),
            HostInt64[MeshCellDim],
            "ApproximateSourceCoverageCertificate.cell_regions",
        )
        bound = float(distance_upper)
        if math.isnan(bound) or bound < 0.0:
            raise ValueError(
                "ApproximateSourceCoverageCertificate.distance_upper must be nonnegative; "
                f"got {bound!r}"
            )
        for premise in (embedding.binding, fidelity.binding):
            if (
                premise.mesh_id,
                premise.topology_id,
                premise.geometry_layout_id,
                premise.geometry_id,
                premise.coordinate_scope,
            ) != (
                binding.mesh_id,
                binding.topology_id,
                binding.geometry_layout_id,
                binding.geometry_id,
                binding.coordinate_scope,
            ):
                raise ValueError(
                    "ApproximateSourceCoverageCertificate.binding must bind both premises; "
                    f"got {premise.binding_id!r}"
                )
        if (binding.source_id, binding.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ) or (fidelity.binding.source_id, fidelity.binding.source_revision) != (
            domain.source_id,
            domain.source_revision,
        ):
            raise ValueError(
                "ApproximateSourceCoverageCertificate.source must match the CAD revision; "
                f"got {binding.source_revision!r}, {fidelity.binding.source_revision!r}"
            )
        findings_ = tuple(findings)
        if certificate_status(findings_) == "certified" and (
            embedding.status != "certified"
            or fidelity.status != "certified"
            or fidelity.target_facet_ids is not None
            or boundary_degree_id is None
            or boundary_degree_id != fidelity.boundary_degree_id
            or not math.isfinite(bound)
            or np.any(regions != 0)
            or len(domain.region_ids) != 1
            or regions.size != embedding.cell_count
            or bound < max(fidelity.mesh_to_source_upper, fidelity.source_to_mesh_upper)
        ):
            raise ValueError(
                "ApproximateSourceCoverageCertificate.premises must prove bounded coverage "
                "of the entire single region; got missing or inconsistent evidence"
            )
        self.binding = binding
        self.embedding = embedding
        self.fidelity = fidelity
        self.domain = domain
        self.region_ids = domain.region_ids
        self.cell_regions = regions
        self.boundary_degree_id = boundary_degree_id
        self.distance_upper = bound
        self.findings = findings_
        self.status = certificate_status(findings_)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "approximate-source-coverage-certificate",
                "binding": binding.binding_id,
                "domain": domain.domain_id,
                "embedding": embedding.certificate_id,
                "fidelity": fidelity.certificate_id,
                "regions": domain.region_ids,
                "cell_regions": array_tree_fingerprint(regions),
                "boundary_degree": boundary_degree_id,
                "distance_upper": repr(bound),
                "findings": tuple(value.finding_id for value in findings_),
                "status": self.status,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[Self],
        value: ApproximateSourceCoverageCertificate,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Self:
        """Rebuild every physical premise; reconstruction never issues a live theorem."""
        if type(value) is not cls:
            raise TypeError("ApproximateSourceCoverageCertificate.stored_fields must retain its owner type")
        domain = domain_planar_boundary_owner.CadSourceDomain.from_stored_fields(
            value.domain, budget=budget
        )
        embedding = GlobalEmbeddingCertificate.from_stored_fields(
            value.embedding, budget=budget
        )
        fidelity = source_fidelity_owner.SourceFidelityCertificate.from_stored_fields(value.fidelity)
        rebuilt = cls(
            value.binding, embedding, fidelity, domain, value.cell_regions, value.findings,
            boundary_degree_id=value.boundary_degree_id,
            distance_upper=value.distance_upper,
        )
        if not tree_content_equal(value, rebuilt):
            raise ValueError("ApproximateSourceCoverageCertificate.stored_fields must retain its physical premises")
        return rebuilt


def _bound_boundary_source(
    domain: domain_planar_boundary_owner.CadSourceDomain,
    source: SourceBoundaryQuery | None,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> SourceBoundaryQuery:
    """Retain the supplied proof chain only for the exact authored boundary."""
    authored = domain.boundary_source(budget=budget)
    if source is None:
        return authored
    match source, authored:
        case (
            domain_boundary_owner.MeshingDomainBoundarySource(),
            domain_boundary_owner.MeshingDomainBoundarySource(),
        ):
            if source.patches == authored.patches and tree_content_equal(
                source.domain, authored.domain
            ):
                return source
        case (
            domain_planar_boundary_owner.EmbeddedPlanarBoundarySource(),
            domain_planar_boundary_owner.EmbeddedPlanarBoundarySource(),
        ):
            if tree_content_equal(source, authored):
                return source
    raise ValueError(
        "certify_approximate_source_coverage.source must retain the complete "
        "authored CAD boundary, authority, source revision and trim bindings"
    )


@checked
def certify_approximate_source_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: domain_planar_boundary_owner.CadSourceDomain,
    cell_regions: ArrayLike,
    /,
    *,
    embedding: GlobalEmbeddingCertificate,
    fidelity: source_fidelity_owner.SourceFidelityCertificate,
    source: SourceBoundaryQuery | None = None,
    limits: MeshCertificateLimits | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> ApproximateSourceCoverageCertificate:
    """Certify bounded CAD-region deviation using the entire oriented boundary.

    Each cell belongs to exactly one explicit source region. Global embedding
    supplies positive orientation and disjoint cell interiors. The same
    boundary-degree argument used by source fidelity proves a bounded, oriented
    boundary homotopy. Its swept image contains the symmetric difference and
    is within ``distance_upper`` of both regions.

    A toleranced B-Rep additionally needs authenticated oriented incidence
    ribbons to a closed representative, retaining the original native vertex,
    edge and face atoms and their admitted gap envelopes. The ribbon
    displacement is included in both directed fidelity bounds. A scalar gap
    allowance or bounded surface fidelity alone never supplies this degree
    evidence, and a positive gap is never relabeled exact coincidence.

    ``source`` retains the chain-bound boundary used by source fidelity. Its
    complete authored domain is checked independently of the derived chart
    witness; the witness is never stored in ``CadSourceDomain`` or substituted
    into that domain's identity. Omitting it selects the authored, chain-free
    boundary and does not manufacture a general curved-BRep degree proof.
    """
    embedding.binding.require(mesh, geometry)
    fidelity.binding.require(mesh, geometry)
    if (
        mesh.topological_dimension != mesh.ambient_dimension
        or mesh.ambient_dimension != domain.ambient_dimension
    ):
        raise ValueError(
            "certify_approximate_source_coverage.mesh must be a volume of the source dimension; "
            f"got {mesh.topological_dimension}, {mesh.ambient_dimension}, {domain.ambient_dimension}"
        )
    raw_regions = np.asarray(cell_regions)
    if raw_regions.ndim != 1 or raw_regions.dtype.kind not in "iu":
        raise ValueError(
            "certify_approximate_source_coverage.cell_regions must be an integer vector; "
            f"got shape {raw_regions.shape}, dtype {raw_regions.dtype}"
        )
    regions = parse(
        np.asarray(raw_regions, dtype=np.int64), HostInt64[MeshCellDim], "cell_regions"
    )
    cell_ids = cell_global_ids(mesh)
    if (
        regions.size != cell_ids.size
        or np.any(regions < -1)
        or np.any(regions >= len(domain.region_ids))
    ):
        raise ValueError(
            "certify_approximate_source_coverage.cell_regions must assign each cell a region or -1; "
            f"got {regions.size} assignments for {cell_ids.size} cells"
        )
    limits_ = MeshCertificateLimits() if limits is None else limits
    state = EmbeddingState([], ["approximate_source_coverage"])
    if embedding.status != "certified":
        state.add(
            "approximate_source_embedding_premise",
            "violated" if embedding.status == "violated" else "unresolved",
            "mesh",
        )
    if fidelity.status != "certified" or fidelity.target_facet_ids is not None:
        state.add(
            "approximate_source_fidelity_premise",
            "violated" if fidelity.status == "violated" else "unresolved",
            "mesh",
        )
    if len(domain.region_ids) != 1:
        state.add("approximate_source_single_region_premise", "unresolved", "mesh")
    if np.any(regions == -1):
        state.add(
            "approximate_source_region_assignment",
            "unresolved",
            "cell",
            cell_ids[regions == -1],
        )
    if mesh.storage is not None:
        state.add("owner_local_approximate_source_coverage_premise", "unresolved", "mesh")
    with coordinate_certificate_request(limits_, budget=budget) as budget:
        scope = embedding.binding.coordinate_scope
        proof = None
        imported_closure = None
        try:
            with (
                budget.measurement(),
                budget.activate(),
                budget.bound_stage(
                    limits_.maximum_work_units, limits_.maximum_scratch_bytes
                ),
            ):
                source = _bound_boundary_source(domain, source, budget=budget)
                if (fidelity.binding.source_id, fidelity.binding.source_revision) != (
                    source.source_id,
                    source.source_revision,
                ):
                    raise ValueError(
                        "certify_approximate_source_coverage.fidelity must bind the CAD source revision; "
                        f"got {fidelity.binding.source_id!r}, {fidelity.binding.source_revision!r}"
                    )
                scope = coordinate_scope(mesh, geometry, budget=budget).coordinate_scope
                if not state.findings:
                    closure = fidelity.cad_closure
                    if closure is not None:
                        if not isinstance(source, domain_boundary_owner.MeshingDomainBoundarySource):
                            raise ValueError(
                                "approximate_source_coverage.cad_closure requires its actual native chart source"
                            )
                        source_fidelity_owner.require_issued_cad_boundary_closure(
                            mesh, geometry, closure, limits_, source=source, budget=budget
                        )
                        if (
                            closure.authority_id != domain.model.model_id
                            or closure.source_scope_id != source.source_scope_id
                            or fidelity.boundary_degree_id is None
                            or closure.homotopy_distance_upper > fidelity.tolerance
                        ):
                            raise ValueError(
                                "approximate_source_coverage.cad_closure must retain the "
                                "issued complete authored boundary, source scope and tolerance"
                            )
                        budget.retain_owner(closure)
                        imported_closure = closure
                    else:
                        proof = source_fidelity_owner.boundary_degree_proof(
                            mesh, geometry, source, fidelity.tolerance, limits_, budget=budget
                        )
                        if proof is None:
                            state.add(
                                "approximate_source_boundary_theorem_premise",
                                "unresolved",
                                "mesh",
                            )
                        else:
                            state.findings.extend(proof.findings)
                            if (
                                proof.boundary_degree_id is None
                                or proof.boundary_degree_id != fidelity.boundary_degree_id
                            ):
                                state.add(
                                    "approximate_source_boundary_degree_binding",
                                    "unresolved",
                                    "mesh",
                                )
        except CoordinateEnclosureResourceError as error:
            state.add(
                "approximate_source_expression_resource",
                "unresolved",
                "mesh",
                resource_error=error,
                expression_budget=budget,
            )
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            scope,
            limits_,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
        degree_id = (
            fidelity.boundary_degree_id
            if imported_closure is not None
            else None if proof is None else proof.boundary_degree_id
        )
        proof_upper = (
            imported_closure.homotopy_distance_upper
            if imported_closure is not None
            else math.inf if proof is None else proof.distance_upper
        )
        upper = (
            math.inf
            if state.findings or degree_id is None
            else max(
                proof_upper,
                fidelity.mesh_to_source_upper,
                fidelity.source_to_mesh_upper,
            )
        )
        return ApproximateSourceCoverageCertificate(
            binding,
            embedding,
            fidelity,
            domain,
            regions,
            tuple(state.findings),
            boundary_degree_id=degree_id,
            distance_upper=upper,
        )


__all__ = ["ApproximateSourceCoverageCertificate", "certify_approximate_source_coverage"]
