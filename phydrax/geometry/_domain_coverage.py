#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify domain and interface coverage from independent source partitions.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The supplied embedding certificate is bound to this mesh and geometry and has certified status."
      }
    ],
    "conclusion": "Coverage relies on the bound embedding certificate for disjoint cell interiors; uncertified embedding leaves that dependency unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Domain coverage evidence dependency",
      "locator": "phydrax.geometry.certify_domain_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "owner_local_domain_coverage_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The mesh uses owner-local storage without a global boundary and region partition supplied to this operation."
      }
    ],
    "conclusion": "Local storage alone does not admit this global coverage route.",
    "citation": {
      "kind": "definition",
      "reference": "Global domain coverage admission",
      "locator": "phydrax.geometry.certify_domain_coverage",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "unassigned_cell"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every cell assignment follows mesh block order and lies in the declared region index range or equals -1."
      }
    ],
    "conclusion": "Every assigned cell belongs to exactly one declared region; a -1 assignment violates the coverage requirement.",
    "citation": {
      "kind": "definition",
      "reference": "Declared region assignment",
      "locator": "phydrax.geometry.certify_domain_coverage.cell_regions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "coverage_candidate_capacity",
      "mapped_coverage_resource_budget"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Candidate evaluations and exact coordinate operations are charged in execution order to the configured limits and request ledger."
      }
    ],
    "conclusion": "Exhausting candidate capacity or coordinate work/storage admission leaves coverage unresolved and retains the measured refusal.",
    "citation": {
      "kind": "definition",
      "reference": "Coverage resource admission",
      "locator": "phydrax.geometry.MeshCertificateLimits and CoordinateEnclosureBudget",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "nonplanar_covering_facet",
      "nonplanar_cell_facet",
      "declared_projection_predicates"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Every polygon triangulation reports nonplanar=False and undecided=False, and every declared simplex has a nonzero orientation sign with known=True on its selected projection."
      }
    ],
    "conclusion": "Unproved triangulation or projected orientation prevents the affine facet or region measure argument.",
    "citation": {
      "kind": "definition",
      "reference": "Affine coverage projection admission",
      "locator": "phydrax.geometry._domain_coverage._covering_pieces, _cell_measure_pieces, _projection_axes",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "overlapping_source_facets"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Coplanar oriented source facets have explicit region pairs; exact projected intersections are evaluated for admitted box candidates."
      }
    ],
    "conclusion": "Positive-measure intersections of compatible source facets violate the independent source partition.",
    "citation": {
      "kind": "mathematical",
      "reference": "John K. Hunter, Measure Theory (2011)",
      "locator": "Definition 1.9 (countable additivity)",
      "url": "https://www.math.ucdavis.edu/~hunter/measure_theory/measure_notes.pdf"
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "unmatched_boundary_facet",
      "unmatched_interface_facet"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every target facet occurrence has an explicit oriented region pair and exact physical projected intersections with the independent source partition."
      }
    ],
    "conclusion": "A boundary or interface facet whose image is not fully matched to the same oriented source region pair violates coverage.",
    "citation": {
      "kind": "definition",
      "reference": "Declared boundary and interface correspondence",
      "locator": "phydrax.geometry._domain_coverage._partition_covering_pieces and _certify_mapped_chart",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "mapped_boundary_source",
      "mapped_facet_containment"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Coordinate source face charts and their polynomial plane identities are available; source groups retain normalized planes, oriented region pairs, and source facet row identities."
      },
      {
        "tag": "N",
        "statement": "A Bernstein image hull is contained in the source union and the signed projected Jacobian has a certified nonnegative bound; subdivision completes within the configured limits."
      }
    ],
    "conclusion": "Unavailable source charts or unfinished chart containment leave mapped facet coverage unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Mapped facet containment admission",
      "locator": "phydrax.geometry._domain_coverage._mapped_domain_coverage and _certify_mapped_chart",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "uncovered_boundary",
      "omitted_interface",
      "double_covered_boundary",
      "double_covered_interface"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Source groups are finite compact nondegenerate facet unions; contained target images have disjoint interiors under the bound embedding certificate and preserve oriented region pairs."
      },
      {
        "tag": "N",
        "statement": "Projected source measures and signed mapped image measures have rational integrals with zero integration error."
      },
      {
        "tag": "T",
        "statement": "A finite union of compact images is closed in the source facet union, so a nonempty complement contains a relatively open neighborhood within a source facet."
      }
    ],
    "conclusion": "Equal projected measure of contained compact images establishes coverage of every source member; a smaller or larger measure reports missing or repeated boundary/interface coverage.",
    "citation": {
      "kind": "mathematical",
      "reference": "John K. Hunter, Measure Theory (2011)",
      "locator": "Definitions 1.1 and 1.9; sections 2.2 and 2.6 (rectangle measure and open-set rectangle covers)",
      "url": "https://www.math.ucdavis.edu/~hunter/measure_theory/measure_notes.pdf"
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "mapped_measure_source",
      "mapped_rational_boundary_measure"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each assigned source map supplies a supported polynomial Jacobian integral or rational boundary-flux representation for its oriented region chain."
      }
    ],
    "conclusion": "An unavailable signed integral remains unknown with no achieved enclosure or integration error.",
    "citation": {
      "kind": "definition",
      "reference": "Supported source integral representation",
      "locator": "phydrax.geometry._domain_coverage.mapped_coverage_measures",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "mapped_signed_boundary_region_measure"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The nonperiodic three-dimensional source chain has at most two owners per facet and a bound certified embedding that evaluated facet_pairing and mapped_trace_continuity. The resolved maps form a finite piecewise-smooth oriented chain whose internal paired traces have opposite signs."
      },
      {
        "tag": "N",
        "statement": "Each resolved source-face flux is integrated as a fraction with zero integration error; unknown region fluxes are reported separately."
      }
    ],
    "conclusion": "The oriented boundary chain supplies signed region volumes while retaining unresolved integrals as unknown.",
    "citation": {
      "kind": "mathematical",
      "reference": "John K. Hunter, Notes on Partial Differential Equations",
      "locator": "Theorem 1.40 and section 1.12, including the piecewise C1 boundary extension",
      "url": "https://www.math.ucdavis.edu/~hunter/m218a_09/pde_notes.pdf"
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "region_measure"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Declared source facets and mesh cells have explicit region identities, outward orientation, piecewise-smooth maps, and the bound embedding and independent source partition hypotheses."
      },
      {
        "tag": "N",
        "statement": "Known requested and achieved signed region integrals are fractions with zero integration error, compared before outward-rounded binary64 reporting."
      }
    ],
    "conclusion": "Unequal known requested and achieved region integrals violate domain coverage in declared region order.",
    "citation": {
      "kind": "mathematical",
      "reference": "John K. Hunter, Notes on Partial Differential Equations",
      "locator": "Theorems 1.38 (change of variables) and 1.40 (divergence theorem), sections 1.11-1.12",
      "url": "https://www.math.ucdavis.edu/~hunter/m218a_09/pde_notes.pdf"
    },
    "failure_statuses": [
      "violated"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import fields
from fractions import Fraction
from typing import assert_never, final, Literal, Protocol, TypeAlias

import equinox as eqx
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax._host_storage as host_storage_owner
import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.discretization._exact_plc_geometry as exact_plc_geometry_owner
import phydrax.discretization._exact_power_geometry as exact_power_geometry_owner
import phydrax.discretization._exact_rational_geometry as exact_rational_geometry_owner
import phydrax.geometry._exact_polyhedral_geometry as exact_polyhedral_geometry_owner
import phydrax.geometry._mapped_coverage as mapped_coverage_owner
import phydrax.geometry._mapped_reference_domain as mapped_reference_domain_owner
import phydrax.geometry._planar_coverage as planar_coverage_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
from phydrax.discretization._coordinate_budget import reserve_polynomial
from phydrax.discretization._coordinate_identity import coordinate_scope_key
from phydrax.discretization._reference_cell_types import (
    CellBlock,
    CellGeometryElement,
    PolyhedralBlock,
)

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer, unique_identifiers
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ..discretization._coordinate_enclosure import (
    CoordinateCoefficients,
    CoordinateSourceBank,
    Expression,
    Polynomial,
)
from ..discretization._exact_algebraic_geometry import (
    ExactAlgebraicCellGeometrySource,
    rational_coefficient,
)
from ..linalg._small_batched import cofactor_determinant_2, cofactor_determinant_3
from ..typing import checked, Dim, HostFloat64, HostInt64, parse, Scope
from ._embedding_state import (
    boxes,
    candidate_pairs,
    coordinate_certificate_request,
    CoordinateMapScope,
    EmbeddingState,
    facet_entity_ids,
    Facets,
    mesh_facets,
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from ._global_embedding import (
    cell_global_ids,
    coordinate_scope,
    degenerate_simplices,
    exact_source_predicate_view,
    GlobalEmbeddingCertificate,
    mesh_orient2d,
)
from ._mesh_certificates import (
    active_expression_work,
    certificate_status,
    MeshCellDim,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._polygon_triangulation import loop_triangles
from ._rational import dyadic_integers


class _MappedReferenceProofs(Protocol):
    """Read-only request capabilities for recursive mapped-reference proofs."""

    @staticmethod
    def certify_mapped_reference_source(
        domain: mapped_reference_domain_owner.MappedReferenceDomain,
        limits: MeshCertificateLimits,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[CellMesh, GlobalEmbeddingCertificate, DomainCoverageCertificate]: ...

    @staticmethod
    def certify_mapped_reference_coverage(
        state: EmbeddingState,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        domain: mapped_reference_domain_owner.MappedReferenceDomain,
        regions: np.ndarray,
        limits: MeshCertificateLimits,
        binding: MeshCertificateBinding,
        embedding: GlobalEmbeddingCertificate,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> DomainCoverageCertificate: ...


def mapped_reference_proofs() -> _MappedReferenceProofs:
    # Independent source proofs recurse through certificate evidence; resolve the proof owner once at the request boundary.
    from . import _mapped_reference_coverage

    return _mapped_reference_coverage


CoverageOverlapSemantics: TypeAlias = Literal["exact", "candidate"]


CoverageCoordinateSpace: TypeAlias = Literal["physical", "reference"]


type CoverageOverlapKey = tuple[
    int, int, tuple[int, ...], CoverageOverlapSemantics, CoverageCoordinateSpace
]


type _CoverageOverlap = tuple[
    int,
    int,
    tuple[int, ...],
    float,
    float,
    CoverageOverlapSemantics,
    CoverageCoordinateSpace,
]


class _DomainVertexDim(Dim, minimum=1):
    """Vertices of one declared piecewise-linear domain."""


class _DomainFacetDim(Dim, minimum=1):
    """Oriented facets of one declared piecewise-linear domain."""


class _DomainAmbientDim(Dim, minimum=2):
    """Ambient dimension of a declared domain (facet arity)."""


@final
class PiecewiseLinearDomain(StrictModule, NonTrainableState):
    """Declared affine domain: oriented boundary and interface facets.

    ``facets`` are segments (2-D) or triangles (3-D) whose induced normal points
    from region ``facet_regions[:, 0]`` into region ``facet_regions[:, 1]``;
    ``-1`` denotes the exterior. Region measures follow from the Gauss-Green
    divergence identity over these facets.
    """

    __strict_contract__ = True

    vertices: HostFloat64[_DomainVertexDim, _DomainAmbientDim]
    facets: HostInt64[_DomainFacetDim, _DomainAmbientDim]
    facet_regions: HostInt64[_DomainFacetDim, Literal[2]]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        facets: ArrayLike,
        facet_regions: ArrayLike,
        region_ids: tuple[str, ...],
        /,
        *,
        source_id: str,
    ) -> None:
        scope = Scope()
        points = parse(
            np.asarray(vertices, dtype=np.float64),
            HostFloat64[_DomainVertexDim, _DomainAmbientDim],
            "vertices",
            scope=scope,
        )
        rows = parse(
            np.asarray(facets, dtype=np.int64),
            HostInt64[_DomainFacetDim, _DomainAmbientDim],
            "facets",
            scope=scope,
        )
        pairs = parse(
            np.asarray(facet_regions, dtype=np.int64),
            HostInt64[_DomainFacetDim, Literal[2]],
            "facet_regions",
            scope=scope,
        )
        ids = unique_identifiers(region_ids, "region_ids")
        source = canonical_identifier(source_id, "source_id")
        if points.shape[1] not in (2, 3):
            raise ValueError(
                f"PiecewiseLinearDomain.vertices must have dimension 2 or 3; got {points.shape[1]}"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError(
                "PiecewiseLinearDomain.vertices must be finite; got nonfinite coordinates"
            )
        if np.any(rows < 0) or np.any(rows >= points.shape[0]):
            raise ValueError(
                "PiecewiseLinearDomain.facets must index declared vertices; got out-of-range indices"
            )
        if (
            np.any(pairs < -1)
            or np.any(pairs >= len(ids))
            or np.any(pairs[:, 0] == pairs[:, 1])
        ):
            raise ValueError(
                "PiecewiseLinearDomain.facet_regions must contain distinct region indices or -1; got invalid pairs"
            )
        if np.any(degenerate_simplices(points, rows)):
            raise ValueError(
                "PiecewiseLinearDomain.facets must be nondegenerate; got degenerate facets"
            )
        self.vertices = points
        self.facets = rows
        self.facet_regions = pairs
        self.region_ids = ids
        self.source_id = source
        self.source_revision = canonical_fingerprint(
            {
                "vertices": array_tree_fingerprint(points),
                "facets": array_tree_fingerprint(rows),
                "facet_regions": array_tree_fingerprint(pairs),
                "region_ids": ids,
            }
        )
        self.domain_id = canonical_fingerprint(
            {
                "kind": "piecewise-linear-domain",
                "source": source,
                "revision": self.source_revision,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PiecewiseLinearDomain], value: PiecewiseLinearDomain, /
    ) -> PiecewiseLinearDomain:
        """Authenticate stored facets against their retained source revision.

        Facet nondegeneracy is earned by fresh construction; restoration checks
        content bindings without repeating predicates under the caller's budget.
        """
        # Lazy: source registration resolves geometry and meshing declarations.
        from .._model._structure import restore_registered_fields
        from ..lifecycle._meshing_source_records import register_meshing_source_types

        register_meshing_source_types()
        if type(value) is not cls:
            raise TypeError(
                f"PiecewiseLinearDomain.stored_fields must match the registered type; got {type(value).__name__}"
            )
        stored = {member.name: getattr(value, member.name) for member in fields(value)}
        stored["region_ids"] = unique_identifiers(value.region_ids, "region_ids")
        stored["source_id"] = canonical_identifier(value.source_id, "source_id")
        stored["source_revision"] = canonical_fingerprint(
            {
                "vertices": array_tree_fingerprint(value.vertices),
                "facets": array_tree_fingerprint(value.facets),
                "facet_regions": array_tree_fingerprint(value.facet_regions),
                "region_ids": stored["region_ids"],
            }
        )
        stored["domain_id"] = canonical_fingerprint(
            {
                "kind": "piecewise-linear-domain",
                "source": stored["source_id"],
                "revision": stored["source_revision"],
            }
        )
        rebuilt = restore_registered_fields(cls, stored)
        for member in fields(value):
            if not tree_content_equal(
                getattr(rebuilt, member.name), getattr(value, member.name)
            ):
                raise ValueError(
                    f"PiecewiseLinearDomain.{member.name} must match retained source bindings; got changed stored content"
                )
        return rebuilt

    @property
    def ambient_dimension(self) -> int:
        return self.vertices.shape[1]


@final
class CoverageMeasures(StrictModule):
    """Group reported region integrals, enclosures, and integration errors."""

    requested: tuple[float, ...] = eqx.field(static=True)
    achieved: tuple[float | None, ...] = eqx.field(static=True)
    requested_bounds: tuple[tuple[float, float], ...] = eqx.field(static=True)
    achieved_bounds: tuple[tuple[float, float] | None, ...] = eqx.field(static=True)
    integration_errors: tuple[float | None, ...] = eqx.field(static=True)


@final
class CoverageEvidence(StrictModule):
    """Group facet relations, proof dependencies, and direct subdivision work."""

    covered_source_facets: int = eqx.field(static=True)
    facet_source_overlaps: tuple[_CoverageOverlap, ...] = eqx.field(static=True)
    premise_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    candidate_pairs: int = eqx.field(static=True)
    subdivision_pieces: int = eqx.field(static=True)
    maximum_subdivision_depth: int = eqx.field(static=True)


@final
class CoverageExpressionWork(StrictModule):
    """Group charged work, admission requirement, and phase-local peak storage."""

    charged: int | None = eqx.field(static=True)
    required: int | None = eqx.field(static=True)
    peak_bytes: int | None = eqx.field(static=True)


def _validated_expression_work(work: CoverageExpressionWork) -> CoverageExpressionWork:
    if not (
        (work.charged is None) == (work.required is None) == (work.peak_bytes is None)
    ):
        raise ValueError(
            "DomainCoverageCertificate.expression_work must be measured together; got partially measured work"
        )
    charged = (
        None
        if work.charged is None
        else nonnegative_integer(work.charged, "source_expression_work_units")
    )
    required = (
        None
        if work.required is None
        else nonnegative_integer(work.required, "source_expression_required_work_units")
    )
    if charged is not None and required is not None and charged > required:
        raise ValueError(
            "DomainCoverageCertificate.expression_work.charged must not exceed required; got charged work above requirement"
        )
    peak = (
        None
        if work.peak_bytes is None
        else nonnegative_integer(work.peak_bytes, "source_expression_peak_bytes")
    )
    return CoverageExpressionWork(charged, required, peak)


def _validated_region_measures(
    measures: CoverageMeasures,
    region_count: int,
    status: MeshCertificateStatus,
) -> CoverageMeasures:
    if not (
        len(measures.requested)
        == len(measures.achieved)
        == len(measures.requested_bounds)
        == len(measures.achieved_bounds)
        == len(measures.integration_errors)
        == region_count
    ):
        raise ValueError(
            "DomainCoverageCertificate.measures must follow declared regions; got inconsistent lengths"
        )
    for requested, achieved, requested_bounds, achieved_bounds, error in zip(
        measures.requested,
        measures.achieved,
        measures.requested_bounds,
        measures.achieved_bounds,
        measures.integration_errors,
        strict=True,
    ):
        if not (
            math.isfinite(requested)
            and math.isfinite(requested_bounds[0])
            and math.isfinite(requested_bounds[1])
            and requested_bounds[0] <= requested <= requested_bounds[1]
        ):
            raise ValueError(
                "DomainCoverageCertificate.measures.requested_bounds must contain finite reports; got invalid enclosure"
            )
        if achieved is None or achieved_bounds is None or error is None:
            if not (achieved is None and achieved_bounds is None and error is None):
                raise ValueError(
                    "DomainCoverageCertificate.measures.achieved must be absent together with bounds and error; got partial integral"
                )
        elif not (
            math.isfinite(achieved)
            and math.isfinite(error)
            and error >= 0.0
            and math.isfinite(achieved_bounds[0])
            and math.isfinite(achieved_bounds[1])
            and achieved_bounds[0] <= achieved <= achieved_bounds[1]
        ):
            raise ValueError(
                "DomainCoverageCertificate.measures.achieved_bounds must contain finite reports with nonnegative errors; got invalid enclosure"
            )
    if status == "certified" and None in measures.achieved:
        raise ValueError(
            "DomainCoverageCertificate.measures.achieved must include every certified region integral; got absent integral"
        )
    return measures


def _validated_overlap_evidence(
    evidence: CoverageEvidence,
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
) -> CoverageEvidence:
    overlaps: list[_CoverageOverlap] = []
    for (
        facet,
        source,
        axes,
        lower,
        upper,
        semantics,
        space,
    ) in evidence.facet_source_overlaps:
        semantics = parse(semantics, CoverageOverlapSemantics, "overlap semantics")
        space = parse(space, CoverageCoordinateSpace, "overlap coordinate space")
        if (
            source < 0
            or source >= domain.facets.shape[0]
            or len(axes) != domain.ambient_dimension - 1
            or len(set(axes)) != len(axes)
            or any(axis < 0 or axis >= domain.ambient_dimension for axis in axes)
            or not 0.0 <= lower <= upper
        ):
            raise ValueError(
                "DomainCoverageCertificate.facet_source_overlaps must contain valid source rows, axes, and ordered bounds; got malformed overlap"
            )
        overlaps.append((facet, source, axes, lower, upper, semantics, space))
    rows = tuple(overlaps)
    if tuple(sorted(rows)) != rows:
        raise ValueError(
            "DomainCoverageCertificate.facet_source_overlaps must follow canonical ordering; got unsorted overlaps"
        )
    return CoverageEvidence(
        evidence.covered_source_facets,
        rows,
        tuple(evidence.premise_certificate_ids),
        evidence.candidate_pairs,
        evidence.subdivision_pieces,
        evidence.maximum_subdivision_depth,
    )


@final
class DomainCoverageCertificate(StrictModule, NonTrainableState):
    """Report coverage of a declared domain, its boundary, interfaces and regions.

    Region measures integrate source coordinate expressions in rational arithmetic.
    Reporting bounds include binary64 rounding; integration errors are zero
    for these rational integrals. An absent integral (unresolved or refused source
    expression) reports ``None`` for its achieved measure, achieved bounds and
    integration error: ``None`` means unknown/unbounded, never a computed zero.

    ``facet_source_overlaps`` retains declared source facet row identities.
    Rows contain target facet global ID, source row ID, projection axes, lower
    and upper projected overlap bounds, ``CoverageOverlapSemantics``, and the
    coordinate space. Physical rows describe planar source overlap; reference
    rows describe the independent parameter charts of a mapped source.
    Candidate bounds describe conservative relations rather than measured overlaps.
    ``premise_certificate_ids`` binds independently computed source embedding
    and reference/root partition evidence for mapped reference domains.
    ``candidate_pair_count`` counts this owner's evaluated source-source and
    target-source closed-box candidates; nested dependency work is not guessed
    or added to this direct counter. ``subdivision_piece_count`` and
    ``maximum_subdivision_depth_reached`` record this phase's charged
    subdivision pieces and deepest examined level, including a refused
    over-budget piece; zero means no subdivision was performed.
    ``source_expression_work_units`` is the charged work increment of this
    certificate's coordinate-expression phase. ``source_expression_required_work_units``
    is its largest phase-local dry-admission requirement and the minimum allowance
    for this coverage phase with the same prepared state. It excludes predecessor
    work and does not state the request-wide work allowance: a shared request ledger
    must also admit its earlier work and other operations. ``source_expression_peak_bytes``
    is its peak additional live storage above the phase's entry footprint.
    Request-wide counters, predecessor storage, and earlier peaks remain on the
    request ledger. Refused phases retain completed and required work. Zero means
    no coordinate expression; ``None`` means unmeasured.
    ``certificate_id`` excludes these invocation-specific expression receipts
    while retaining the domain, proof dependencies, measures, and coverage verdict.
    """

    binding: MeshCertificateBinding
    embedding_certificate_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    region_ids: tuple[str, ...] = eqx.field(static=True)
    requested_region_measures: tuple[float, ...] = eqx.field(static=True)
    achieved_region_measures: tuple[float | None, ...] = eqx.field(static=True)
    requested_region_measure_bounds: tuple[tuple[float, float], ...] = eqx.field(
        static=True
    )
    achieved_region_measure_bounds: tuple[tuple[float, float] | None, ...] = eqx.field(
        static=True
    )
    integration_error_bounds: tuple[float | None, ...] = eqx.field(static=True)
    facet_source_overlaps: tuple[_CoverageOverlap, ...] = eqx.field(static=True)
    premise_certificate_ids: tuple[str, ...] = eqx.field(static=True)
    source_facet_count: int = eqx.field(static=True)
    covered_source_facet_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    subdivision_piece_count: int = eqx.field(static=True)
    maximum_subdivision_depth_reached: int = eqx.field(static=True)
    source_expression_work_units: int | None = eqx.field(static=True)
    source_expression_required_work_units: int | None = eqx.field(static=True)
    source_expression_peak_bytes: int | None = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        embedding_certificate_id: str,
        domain: PiecewiseLinearDomain
        | mapped_reference_domain_owner.MappedReferenceDomain,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        measures: CoverageMeasures,
        evidence: CoverageEvidence,
        expression_work: CoverageExpressionWork,
    ) -> None:
        counts = tuple(
            nonnegative_integer(count, name)
            for name, count in (
                ("candidate_pair_count", evidence.candidate_pairs),
                ("subdivision_piece_count", evidence.subdivision_pieces),
                ("maximum_subdivision_depth_reached", evidence.maximum_subdivision_depth),
            )
        )
        work = _validated_expression_work(expression_work)
        # General containers remain value-validated at the certificate boundary.
        for finding in findings:
            match finding:
                case MeshCertificateFinding():
                    pass
                case invalid:
                    raise TypeError(
                        f"DomainCoverageCertificate.findings must contain MeshCertificateFinding values; got {type(invalid).__name__}"
                    )
        status = certificate_status(findings)
        measures = _validated_region_measures(measures, len(domain.region_ids), status)
        evidence = _validated_overlap_evidence(evidence, domain)
        self.binding = binding
        self.embedding_certificate_id = str(embedding_certificate_id)
        self.domain_id = domain.domain_id
        self.status = status
        self.findings = tuple(findings)
        self.region_ids = domain.region_ids
        self.requested_region_measures = tuple(measures.requested)
        self.achieved_region_measures = tuple(measures.achieved)
        self.requested_region_measure_bounds = tuple(measures.requested_bounds)
        self.achieved_region_measure_bounds = tuple(measures.achieved_bounds)
        self.integration_error_bounds = tuple(measures.integration_errors)
        self.source_facet_count = domain.facets.shape[0]
        self.covered_source_facet_count = evidence.covered_source_facets
        self.facet_source_overlaps = evidence.facet_source_overlaps
        self.premise_certificate_ids = evidence.premise_certificate_ids
        (
            self.candidate_pair_count,
            self.subdivision_piece_count,
            self.maximum_subdivision_depth_reached,
        ) = counts
        self.source_expression_work_units = work.charged
        self.source_expression_required_work_units = work.required
        self.source_expression_peak_bytes = work.peak_bytes
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "domain-coverage-certificate",
                "binding": binding.binding_id,
                "embedding": self.embedding_certificate_id,
                "domain": domain.domain_id,
                "status": self.status,
                "findings": [value.finding_id for value in findings],
                "requested": self.requested_region_measures,
                "achieved": self.achieved_region_measures,
                "requested_bounds": self.requested_region_measure_bounds,
                "achieved_bounds": self.achieved_region_measure_bounds,
                "integration_errors": self.integration_error_bounds,
                "facet_source_overlaps": self.facet_source_overlaps,
                "premises": self.premise_certificate_ids,
                "covered": self.covered_source_facet_count,
                "candidate_pairs": self.candidate_pair_count,
                "subdivision_pieces": self.subdivision_piece_count,
                "maximum_subdivision_depth": self.maximum_subdivision_depth_reached,
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[DomainCoverageCertificate],
        value: DomainCoverageCertificate,
        domain: PiecewiseLinearDomain
        | mapped_reference_domain_owner.MappedReferenceDomain,
        /,
    ) -> DomainCoverageCertificate:
        """Rebuild stored evidence against its declared domain and recompute its ID."""
        return cls(
            value.binding,
            value.embedding_certificate_id,
            domain,
            value.findings,
            measures=CoverageMeasures(
                value.requested_region_measures,
                value.achieved_region_measures,
                value.requested_region_measure_bounds,
                value.achieved_region_measure_bounds,
                value.integration_error_bounds,
            ),
            evidence=CoverageEvidence(
                value.covered_source_facet_count,
                value.facet_source_overlaps,
                value.premise_certificate_ids,
                value.candidate_pair_count,
                value.subdivision_piece_count,
                value.maximum_subdivision_depth_reached,
            ),
            expression_work=CoverageExpressionWork(
                value.source_expression_work_units,
                value.source_expression_required_work_units,
                value.source_expression_peak_bytes,
            ),
        )


@final
class _CoveringPieces(StrictModule):
    """Affine pieces of mesh boundary/interface facets with expected sides."""

    pieces: np.ndarray
    facet_entities: np.ndarray
    own_region: np.ndarray
    other_region: np.ndarray


def _covering_pieces(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoveringPieces:
    order = np.argsort(facets.group, kind="stable")
    first_occurrence = np.ones((order.size,), dtype=np.bool_)
    first_occurrence[1:] = facets.group[order[1:]] != facets.group[order[:-1]]
    partner = np.full((order.size,), -1, dtype=np.int64)
    same = ~first_occurrence[1:]
    partner[order[:-1][same]] = order[1:][same]
    own = regions[facets.cells]
    other = np.where(partner >= 0, regions[facets.cells[np.maximum(partner, 0)]], -1)
    selected = np.zeros((order.size,), dtype=np.bool_)
    selected[order[first_occurrence]] = True
    covering = selected & (facets.counts[facets.group] <= 2) & (own != other)
    rows = facets.rows[covering]
    rows = rows[:, : int(np.max(np.sum(rows >= 0, axis=1), initial=1))]
    entities = (
        facet_entity_ids(mesh, rows) if rows.size else np.empty((0,), dtype=np.int64)
    )
    if mesh.topological_dimension == 2:
        owners = np.arange(rows.shape[0])
        pieces = rows
    else:
        pieces, owners, nonplanar, undecided = loop_triangles(
            points, rows, limits.maximum_candidate_pairs, budget=budget
        )
        if np.any(nonplanar | undecided):
            state.add(
                "nonplanar_covering_facet",
                "unresolved",
                "facet",
                entities[nonplanar | undecided],
            )
    return _CoveringPieces(
        pieces, entities[owners], own[covering][owners], other[covering][owners]
    )


def _projection_axes(
    corners: np.ndarray, /, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinate pair on which each declared facet projects injectively.

    Returns ``(axes, orientation, certain)``; the orientation of the projected
    declared facet is proved correct and nonzero where ``certain``.
    """

    count = corners.shape[0]
    choices = np.asarray(((1, 2), (0, 2), (0, 1)))
    if corners.shape[1] == 2:
        axis = np.where(corners[:, 0, 0] != corners[:, 1, 0], 0, 1)
        sign = np.sign(
            corners[np.arange(count), 1, axis] - corners[np.arange(count), 0, axis]
        )
        return axis[:, None], sign.astype(np.int16), np.ones((count,), dtype=np.bool_)
    normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    preference = np.argsort(-np.abs(normal), axis=1, kind="stable")
    axes = np.zeros((count, 2), dtype=np.int64)
    orientation = np.zeros((count,), dtype=np.int16)
    certain = np.zeros((count,), dtype=np.bool_)
    for rank in range(3):
        pending = orientation == 0
        candidate = choices[preference[:, rank]]
        projected = np.take_along_axis(corners, candidate[:, None, :], axis=2)
        sign, known = mesh_orient2d(
            projected[:, 0], projected[:, 1], projected[:, 2], budget=budget
        )
        take = pending & known & (sign != 0)
        axes[take] = candidate[take]
        orientation[take] = sign[take]
        certain |= take
    return axes, orientation, certain


def _projected_measures(
    integers: np.ndarray, rows: np.ndarray, axes: np.ndarray, /
) -> np.ndarray:
    """Return absolute projected simplex measures as scaled integers."""

    corners = integers[rows]
    index = np.arange(rows.shape[0])
    if rows.shape[1] == 2:
        axis = axes[:, 0]
        values = corners[index, 1, axis] - corners[index, 0, axis]
    else:
        first = np.stack(
            [
                corners[index, 1, axes[:, k]] - corners[index, 0, axes[:, k]]
                for k in range(2)
            ],
            axis=1,
        )
        second = np.stack(
            [
                corners[index, 2, axes[:, k]] - corners[index, 0, axes[:, k]]
                for k in range(2)
            ],
            axis=1,
        )
        values = np.asarray(cofactor_determinant_2(first, second))
    return np.asarray([abs(value) for value in values.tolist()], dtype=object)


def _coverage_rational_polygon(
    polygon: planar_primitives_owner.Polygon, /
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(rational_coefficient(value, role="coverage.polygon") for value in point)
        for point in polygon
    )


def _partition_covering_pieces(
    state: EmbeddingState,
    covering: _CoveringPieces,
    points: np.ndarray,
    domain: PiecewiseLinearDomain,
    axes: np.ndarray,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[list[Fraction], np.ndarray, bool, dict[CoverageOverlapKey, Fraction]]:
    """Partition affine facets across the independent source triangulation."""
    with budget.live_storage() as prepared_storage:
        with budget.temporary_scope():
            groups, overlaps, used, exceeded = planar_coverage_owner.source_groups(
                domain.vertices,
                domain.facets,
                domain.facet_regions,
                limits.maximum_candidate_pairs - state.candidate_pairs,
                budget=budget,
            )
            prepared_bound = 256 + sum(
                host_storage_owner.declared_object_storage_bytes(group) + 128
                for group in groups
            )
            prepared_storage.set_bound(prepared_bound)
        state.candidate_pairs += used
        covered = [Fraction(0) for _ in range(domain.facets.shape[0])]
        unmatched = np.zeros(covering.pieces.shape[0], dtype=np.bool_)
        relations: dict[CoverageOverlapKey, Fraction] = {}
        for first, second in overlaps:
            state.add(
                "overlapping_source_facets", "violated", "source_facet", (first, second)
            )
        if exceeded:
            state.add("coverage_candidate_capacity", "unresolved", "mesh")
            return covered, unmatched, True, relations
        lookup = {(group.plane, group.regions): group for group in groups}
        compatible: dict[
            tuple[planar_primitives_owner.Point, tuple[int, int]], list[int]
        ] = {}
        polygons: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
        whole: dict[int, Fraction] = {}
        piece_preparation: dict[
            int, planar_primitives_owner.PolygonIntersectionPreparation
        ] = {}
        source_preparation: dict[
            int, planar_primitives_owner.PolygonIntersectionPreparation
        ] = {}
        target_projections: dict[int, tuple[tuple[int, ...], Fraction]] = {}
        accumulated = [Fraction(0) for _ in range(covering.pieces.shape[0])]
        for piece, row in enumerate(covering.pieces):
            with budget.temporary_scope():
                polygon = planar_primitives_owner.rational_points(
                    points[row], budget=budget
                )
                key = planar_primitives_owner.plane_key(polygon, budget=budget)
                if key is None:
                    unmatched[piece] = True
                    continue
                plane, _, _, sign = key
                regions = (
                    int(covering.own_region[piece]),
                    int(covering.other_region[piece]),
                )
                canonical_regions = regions if sign > 0 else regions[::-1]
                group = lookup.get((plane, canonical_regions))
                if group is None:
                    unmatched[piece] = True
                    continue
                pair = (group.plane, group.regions)
                polygons[piece] = _coverage_rational_polygon(
                    planar_primitives_owner.project(polygon, group.axes, budget=budget)
                )
                signed = rational_coefficient(
                    planar_primitives_owner.signed_measure(
                        polygons[piece], budget=budget
                    ),
                    role="coverage.projected_measure",
                )
                whole[piece] = abs(signed)
                piece_preparation[piece] = (
                    planar_primitives_owner.prepare_polygon_intersection(
                        polygons[piece], 1 if signed > 0 else -1, budget=budget
                    )
                )
                compatible.setdefault(pair, []).append(piece)
                prepared_bound += budget.live_object_storage_upper(
                    (
                        piece,
                        piece_preparation[piece],
                        piece_preparation[piece].polygon,
                        piece_preparation[piece].lower,
                        piece_preparation[piece].upper,
                        piece_preparation[piece].halfplanes,
                        piece_preparation[piece].bits,
                        whole[piece],
                    )
                )
                prepared_storage.set_bound(prepared_bound)
            del polygon, key
        for pair, pieces in compatible.items():
            group = lookup[pair]
            chosen = np.asarray(pieces, dtype=np.int64)
            source_rows = np.asarray(group.members, dtype=np.int64)
            firsts, seconds, exceeded = candidate_pairs(
                *boxes(points, covering.pieces[chosen]),
                limits.maximum_candidate_pairs - state.candidate_pairs,
                other=boxes(domain.vertices, domain.facets[source_rows]),
            )
            state.candidate_pairs += firsts.size
            for first, local in zip(firsts.tolist(), seconds.tolist(), strict=True):
                with budget.temporary_scope():
                    piece = pieces[first]
                    target = group.members[local]
                    budget.reserve(1)
                    if target not in source_preparation:
                        prepared = planar_primitives_owner.prepare_polygon_intersection(
                            group.simplices[local],
                            group.orientations[local],
                            budget=budget,
                        )
                        prepared_bound += budget.live_object_storage_upper(
                            (
                                target,
                                prepared,
                                prepared.polygon,
                                prepared.lower,
                                prepared.upper,
                                prepared.halfplanes,
                                prepared.bits,
                            )
                        )
                        prepared_storage.set_bound(prepared_bound)
                        source_preparation[target] = prepared
                    measure = rational_coefficient(
                        planar_primitives_owner.intersection(
                            polygons[piece],
                            group.simplices[local],
                            budget=budget,
                            subject_preparation=piece_preparation[piece],
                            target_preparation=source_preparation[target],
                        ),
                        role="coverage.intersection_measure",
                    )
                    if measure == 0:
                        continue
                    accumulated[piece] += measure
                    budget.reserve(1)
                    projection = target_projections.get(target)
                    if projection is None:
                        target_axes = tuple(int(axis) for axis in axes[target])
                        # An identical projection already earned its rational measure
                        # during source-group preparation.
                        if target_axes == group.axes:
                            target_measure = group.measures[local]
                        else:
                            target_measure = abs(
                                planar_primitives_owner.signed_measure(
                                    planar_primitives_owner.project(
                                        planar_primitives_owner.rational_points(
                                            domain.vertices[domain.facets[target]],
                                            budget=budget,
                                        ),
                                        target_axes,
                                        budget=budget,
                                    ),
                                    budget=budget,
                                )
                            )
                        target_measure = rational_coefficient(
                            target_measure, role="coverage.declared_projected_measure"
                        )
                        projection = (target_axes, target_measure)
                        prepared_bound += budget.live_object_storage_upper(
                            (target, projection)
                        )
                        prepared_storage.set_bound(prepared_bound)
                        target_projections[target] = projection
                    target_axes, target_measure = projection
                    actual = measure * target_measure / group.measures[local]
                    covered[target] += actual
                    relation: CoverageOverlapKey = (
                        int(covering.facet_entities[piece]),
                        target,
                        target_axes,
                        "exact",
                        "physical",
                    )
                    relations[relation] = relations.get(relation, Fraction(0)) + actual
            if exceeded:
                state.add("coverage_candidate_capacity", "unresolved", "mesh")
                return covered, unmatched, True, relations
            for piece in pieces:
                unmatched[piece] = accumulated[piece] != whole[piece]
        return covered, unmatched, False, relations


def _region_measures(
    integers: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    count: int,
    /,
) -> list[int]:
    """Exact divergence-theorem measures ``sum weight * det`` per group."""

    corners = integers[rows]
    if rows.shape[1] == 2:
        values = cofactor_determinant_2(corners[:, 0], corners[:, 1])
    else:
        values = cofactor_determinant_3(corners[:, 0], corners[:, 1], corners[:, 2])
    totals = [0] * count
    for group, weight, value in zip(
        groups.tolist(), weights.tolist(), values.tolist(), strict=True
    ):
        if group >= 0:
            totals[group] += weight * value
    return totals


def _cell_measure_pieces(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine pieces of every outward facet occurrence and their cells."""

    if mesh.topological_dimension == 2:
        return facets.rows[:, :2], facets.cells
    pieces, owners, nonplanar, undecided = loop_triangles(
        points, facets.rows, limits.maximum_candidate_pairs, budget=budget
    )
    if np.any(nonplanar | undecided):
        state.add(
            "nonplanar_cell_facet",
            "unresolved",
            "facet",
            facet_entity_ids(mesh, facets.rows[nonplanar | undecided]),
        )
    return pieces, facets.cells[owners]


@checked
def coverage_result(
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
    state: EmbeddingState,
    requested: tuple[Fraction, ...],
    achieved: tuple[Fraction | None, ...],
    covered: int,
    overlaps: dict[CoverageOverlapKey, Fraction],
    premises: tuple[str, ...],
    /,
    *,
    expression_work: tuple[int, int, int] | None,
) -> DomainCoverageCertificate:
    """Build coverage reports while retaining rational-integral mismatches and work evidence."""

    mismatched = tuple(
        index
        for index, (actual, target) in enumerate(zip(achieved, requested, strict=True))
        if actual is not None and actual != target
    )
    if mismatched:
        state.add("region_measure", "violated", "source_region", mismatched)
    return DomainCoverageCertificate(
        binding,
        embedding.certificate_id,
        domain,
        tuple(state.findings),
        measures=CoverageMeasures(
            tuple(float(value) for value in requested),
            tuple(None if value is None else float(value) for value in achieved),
            tuple(mapped_coverage_owner.enclosure(value) for value in requested),
            tuple(
                None if value is None else mapped_coverage_owner.enclosure(value)
                for value in achieved
            ),
            tuple(None if value is None else 0.0 for value in achieved),
        ),
        evidence=CoverageEvidence(
            covered,
            tuple(
                (
                    facet,
                    source,
                    axes,
                    0.0
                    if semantics == "candidate"
                    else mapped_coverage_owner.enclosure(value)[0],
                    mapped_coverage_owner.enclosure(value)[1],
                    semantics,
                    space,
                )
                for (facet, source, axes, semantics, space), value in sorted(
                    overlaps.items()
                )
            ),
            premises,
            state.candidate_pairs,
            state.subdivision_pieces,
            state.subdivision_depth,
        ),
        expression_work=CoverageExpressionWork(
            None if expression_work is None else expression_work[0],
            None if expression_work is None else expression_work[2],
            None if expression_work is None else expression_work[1],
        ),
    )


def _box_jacobian_determinant(
    coordinates: tuple[Polynomial, ...], /, *, budget: CoordinateEnclosureBudget
) -> tuple[dict[tuple[int, ...], int], int]:
    """Return a 2D/3D box-map Jacobian determinant as integers over one denominator.

    This polynomial equals ``determinant(jacobian)`` in rational arithmetic; every coefficient
    visit and product term is charged to the active coordinate ledger.
    """

    dimension = len(coordinates)
    budget.reserve(sum(len(polynomial) for polynomial in coordinates) * (dimension + 1))
    common = math.lcm(
        *(
            rational_coefficient(value, role="coverage.box_coordinate").denominator
            for polynomial in coordinates
            for value in polynomial.values()
        )
    )
    columns: list[list[dict[tuple[int, ...], int]]] = []
    for polynomial in coordinates:
        row = []
        for axis in range(dimension):
            derivative: dict[tuple[int, ...], int] = {}
            for index, value in polynomial.items():
                value = rational_coefficient(value, role="coverage.box_coordinate")
                if index[axis]:
                    lowered = (*index[:axis], index[axis] - 1, *index[axis + 1 :])
                    derivative[lowered] = derivative.get(lowered, 0) + index[
                        axis
                    ] * value.numerator * (common // value.denominator)
            row.append(derivative)
        columns.append(row)

    def bits(values: dict[tuple[int, ...], int]) -> int:
        return max((abs(value).bit_length() for value in values.values()), default=1)

    def product(
        first: dict[tuple[int, ...], int], second: dict[tuple[int, ...], int]
    ) -> dict[tuple[int, ...], int]:
        coordinate_enclosure_owner.reserve_polynomial(
            len(first) * len(second),
            len(first) * len(second),
            dimension,
            bits(first) + bits(second) + max(len(first) * len(second), 1).bit_length(),
            budget=budget,
        )
        result: dict[tuple[int, ...], int] = {}
        for a, x in first.items():
            for b, y in second.items():
                index = tuple(i + j for i, j in zip(a, b, strict=True))
                result[index] = result.get(index, 0) + x * y
        return result

    def combine(
        terms: tuple[tuple[int, dict[tuple[int, ...], int]], ...],
    ) -> dict[tuple[int, ...], int]:
        count = sum(len(term) for _, term in terms)
        coordinate_enclosure_owner.reserve_polynomial(
            count,
            count,
            dimension,
            max((bits(term) for _, term in terms), default=1) + len(terms).bit_length(),
            budget=budget,
        )
        result: dict[tuple[int, ...], int] = {}
        for sign, term in terms:
            for index, value in term.items():
                result[index] = result.get(index, 0) + sign * value
        return result

    j = columns
    if dimension == 2:
        determinant = combine(
            ((1, product(j[0][0], j[1][1])), (-1, product(j[0][1], j[1][0])))
        )
    else:
        minors = (
            combine(((1, product(j[1][1], j[2][2])), (-1, product(j[1][2], j[2][1])))),
            combine(((1, product(j[1][2], j[2][0])), (-1, product(j[1][0], j[2][2])))),
            combine(((1, product(j[1][0], j[2][1])), (-1, product(j[1][1], j[2][0])))),
        )
        determinant = combine(
            tuple((1, product(j[0][axis], minor)) for axis, minor in enumerate(minors))
        )
    return {
        index: value for index, value in determinant.items() if value
    }, common**dimension


@checked
def box_jacobian_measure(
    coordinates: tuple[Polynomial, ...], /, *, budget: CoordinateEnclosureBudget
) -> Fraction:
    """Integrate a 2D/3D map's Jacobian determinant over the unit box in rational arithmetic."""

    determinant, denominator = _box_jacobian_determinant(coordinates, budget=budget)

    budget.reserve(2 * len(determinant))
    weights = {index: math.prod(i + 1 for i in index) for index in determinant}
    scale_ = math.lcm(1, *weights.values())
    total = sum(
        value * (scale_ // weights[index]) for index, value in determinant.items()
    )
    return Fraction(total, scale_ * denominator)


@final
class _PolynomialRootChartPhase(StrictModule):
    """Retain a direct root and its polynomial reference composition."""

    root: CellGeometryElement
    reference: tuple[Polynomial, ...]


def _polynomial_root_chart(
    element: CellGeometryElement,
    kind: str | None,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialRootChartPhase | None:
    source = element
    while True:
        match source:
            case (
                cell_geometry_owner.RestrictedCellGeometryElement()
                | cell_geometry_owner.PolynomialComposedCellGeometryElement()
                | cell_geometry_owner.RationalComposedCellGeometryElement()
            ) as composed:
                source = composed.source_element
            case _:
                break
    match source:
        case cell_geometry_owner.BarycentricCellGeometryElement():
            return None
        case _:
            pass
    root, reference = coordinate_enclosure_owner.coordinate_reference_chain(
        element, budget=budget
    )
    match root:
        case (
            cell_geometry_owner.RestrictedCellGeometryElement()
            | cell_geometry_owner.PolynomialComposedCellGeometryElement()
            | cell_geometry_owner.RationalComposedCellGeometryElement()
            | cell_geometry_owner.SplineCellGeometryElement()
        ):
            return None
        case _:
            pass
    if (
        (root.cell_kind == "pyramid" if kind is None else root.cell_kind != kind)
        or len(reference) != dimension
        or any(not isinstance(value, dict) for value in reference)
    ):
        return None
    return _PolynomialRootChartPhase(
        root, tuple(value for value in reference if isinstance(value, dict))
    )


def _axis_aligned_root_chart(
    element: CellGeometryElement,
    kind: str,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellGeometryElement, tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Direct box root, origin and signed steps of an axis-aligned affine chart chain.

    The chain composes the target element's reference maps; unsupported charts,
    root kinds, and rational sources return ``None``.
    """

    if kind not in ("quadrilateral", "hexahedron") or dimension not in (2, 3):
        return None
    phase = _polynomial_root_chart(element, kind, dimension, budget=budget)
    if phase is None:
        return None
    root, polynomial_reference = phase.root, phase.reference
    origin, steps = [], []
    for axis, value in enumerate(polynomial_reference):
        unit = tuple(int(axis == other) for other in range(dimension))
        if any(index not in ((0,) * dimension, unit) for index in value) or not value.get(
            unit
        ):
            return None
        origin.append(value.get((0,) * dimension, Fraction(0)))
        steps.append(value[unit])
    return root, tuple(origin), tuple(steps)


def _affine_jacobian_determinant(
    polynomials: tuple[Polynomial, ...],
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Fraction | None:
    """Constant Jacobian determinant of an exactly affine map, else ``None``.

    Only total-degree-one maps qualify; their Jacobian and determinant use the
    coordinate-enclosure polynomial algebra in rational arithmetic, charged to the active ledger.
    """

    zero = (0,) * dimension
    if len(polynomials) != dimension or any(
        sum(index) > 1 for value in polynomials for index in value
    ):
        return None
    jacobian = tuple(
        tuple(
            coordinate_enclosure_owner.derivative(value, axis, budget=budget)
            for axis in range(dimension)
        )
        for value in polynomials
    )
    return rational_coefficient(
        coordinate_enclosure_owner.determinant(jacobian, budget=budget).get(
            zero, Fraction(0)
        ),
        role="coverage.affine_reference_jacobian",
    )


def _affine_root_chart(
    element: CellGeometryElement,
    kind: str,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellGeometryElement, Fraction] | None:
    """Return the direct root and rational chart determinant of an affine reference chain.

    An affine restriction of an affine root is affine; its measure is the
    product of both constant Jacobian determinants and the reference measure.
    Pyramid charts, composed, spline or nonaffine chains return ``None``.
    """

    if kind == "pyramid" or dimension not in (2, 3):
        return None
    phase = _polynomial_root_chart(element, None, dimension, budget=budget)
    if phase is None:
        return None
    root, polynomial_reference = phase.root, phase.reference
    determinant = _affine_jacobian_determinant(
        polynomial_reference, dimension, budget=budget
    )
    return None if determinant is None else (root, determinant)


def _signed_boundary_region_measures(
    state: EmbeddingState,
    regions: np.ndarray,
    cells: list[tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]],
    maps: Sequence[tuple[Expression, ...] | None],
    totals: list[Fraction | None],
    rational_regions: set[int],
    facets: Facets,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Integrate the oriented source chain using its certified internal trace equality."""

    budget.reserve(1)
    groups = facets.groups
    unresolved: set[int] = set()
    for group in groups:
        budget.reserve(group.size)
        if group.size > 2:
            raise ValueError(
                f"mapped_coverage_measures.facets must have at most two owners per group; got {group.size} owners"
            )
        occurrence = int(group[0])
        owner = int(facets.cells[occurrence])
        own = int(regions[owner])
        other = int(regions[facets.cells[group[1]]]) if group.size == 2 else -1
        if own == other:
            continue
        affected = tuple(region for region in (own, other) if region >= 0)
        if not affected:
            continue
        expression = maps[owner]
        kind, element, controls, vertices = cells[owner]
        if expression is None:
            unresolved.update(affected)
            continue
        with budget.temporary_scope():
            face = tuple(
                vertices.index(int(vertex))
                for vertex in facets.rows[occurrence]
                if vertex >= 0
            )
            flux = mapped_coverage_owner.prepared_face_flux(
                element, controls, kind, face, budget=budget
            )
            if flux is None:
                unresolved.update(affected)
                continue
            budget.reserve(len(affected))
            if own >= 0:
                current = totals[own]
                if current is not None:
                    totals[own] = current + flux
            if other >= 0:
                current = totals[other]
                if current is not None:
                    totals[other] = current - flux
    for region in sorted(unresolved):
        totals[region] = None
        state.add(
            "mapped_rational_boundary_measure"
            if region in rational_regions
            else "mapped_measure_source",
            "unresolved",
            "source_region",
            (region,),
        )


type _CoverageCell = tuple[
    str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]
]
type _RootMeasureKey = tuple[str, str, CoordinateSourceBank]


@final
class _RegionIntegralPhase(StrictModule):
    """Retain signed region integrals and cell source descriptors."""

    totals: tuple[Fraction | None, ...]
    cells: list[_CoverageCell]


@final
class _CellIntegralPhase(StrictModule):
    """Describe a root integral route with rational-arithmetic accumulation."""

    handled: bool
    measure: Fraction | None


@final
class _RationalRegionPhase(StrictModule):
    """Retain the resolved boundary flux of one rational source region."""

    resolved: bool
    measure: Fraction


@final
class _RootIntegralCache(StrictModule):
    """Retain root determinants for this invocation's live source identities."""

    box: dict[_RootMeasureKey, tuple[dict[tuple[int, ...], int], int] | None]
    affine: dict[_RootMeasureKey, Fraction | None]


def _cached_boundary_integrals(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    facets: Facets | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RegionIntegralPhase | None:
    prepared = budget.prepared_cell_cache.get(coordinate_scope_key(mesh, geometry))
    if prepared is not None:
        count = sum(block.cell_count for block in mesh.blocks)
        budget.reserve(2 * count, 256 + 24 * count)
        ids = tuple(
            int(identifier)
            for block in mesh.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        )
        if (
            prepared.cell_ids != ids
            or len(prepared.descriptors) != count
            or len(prepared.coordinates) != count
        ):
            raise ValueError(
                "mapped_coverage_measures.prepared_cells must match the mesh cell bank; got inconsistent identifiers or counts"
            )
        totals_: list[Fraction | None] = [
            Fraction(0) for _ in range(int(np.max(regions, initial=-1)) + 1)
        ]
        cells_: list[
            tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]
        ] = list(prepared.descriptors)
        maps_ = list(prepared.coordinates)
        rational_: set[int] = set()
        for owner, polynomial in enumerate(maps_):
            budget.reserve(len(polynomial))
            if any(
                isinstance(
                    value,
                    coordinate_enclosure_owner.RationalPolynomial,
                )
                for value in polynomial
            ):
                region = int(regions[owner])
                if region >= 0:
                    rational_.add(region)
        _signed_boundary_region_measures(
            state,
            regions,
            cells_,
            maps_,
            totals_,
            rational_,
            mesh_facets(mesh, geometry=geometry, budget=budget)
            if facets is None
            else facets,
            budget=budget,
        )
        state.checks.append("mapped_signed_boundary_region_measure")
        return _RegionIntegralPhase(tuple(totals_), cells_)
    return None


def _affine_root_integral(
    affine: tuple[CellGeometryElement, Fraction],
    controls: CoordinateSourceBank,
    dimension: int,
    reference_measure: Fraction,
    active: bool,
    cache: _RootIntegralCache,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CellIntegralPhase:
    root, chart_determinant = affine
    key = (
        *coordinate_enclosure_owner.element_identity(root),
        controls,
    )
    if key not in cache.affine:
        expression = coordinate_enclosure_owner.coordinate_expressions(
            root, controls, budget=budget
        )
        cache.affine[key] = (
            None
            if expression is None
            or any(
                isinstance(
                    value,
                    coordinate_enclosure_owner.RationalPolynomial,
                )
                for value in expression
            )
            else _affine_jacobian_determinant(
                tuple(
                    value
                    for value in expression
                    if not isinstance(
                        value,
                        coordinate_enclosure_owner.RationalPolynomial,
                    )
                ),
                dimension,
                budget=budget,
            )
        )
        budget.retain_basis((key, cache.affine[key]))
    root_determinant = cache.affine[key]
    if root_determinant is None:
        return _CellIntegralPhase(False, None)
    if not active:
        return _CellIntegralPhase(True, None)
    budget.reserve(2)
    return _CellIntegralPhase(
        True, root_determinant * chart_determinant * reference_measure
    )


def _box_root_integral(
    chart: tuple[CellGeometryElement, tuple[Fraction, ...], tuple[Fraction, ...]],
    controls: CoordinateSourceBank,
    dimension: int,
    active: bool,
    cache: _RootIntegralCache,
    powers: dict[tuple[int, int], Fraction],
    *,
    budget: CoordinateEnclosureBudget,
) -> _CellIntegralPhase:
    root, origin, steps = chart
    key = (
        *coordinate_enclosure_owner.element_identity(root),
        controls,
    )
    if key not in cache.box:
        expression = coordinate_enclosure_owner.coordinate_expressions(
            root, controls, budget=budget
        )
        polynomials = (
            None
            if expression is None
            or any(
                isinstance(
                    value,
                    coordinate_enclosure_owner.RationalPolynomial,
                )
                for value in expression
            )
            else tuple(
                value
                for value in expression
                if not isinstance(
                    value,
                    coordinate_enclosure_owner.RationalPolynomial,
                )
            )
        )
        root_determinant = (
            None
            if polynomials is None
            else _box_jacobian_determinant(polynomials, budget=budget)
        )
        budget.retain_basis((key, root_determinant))
        cache.box[key] = root_determinant
    prepared = cache.box[key]
    if prepared is None:
        return _CellIntegralPhase(False, None)
    if not active:
        return _CellIntegralPhase(True, None)
    determinant, denominator = prepared
    budget.reserve((dimension + 2) * len(determinant))
    measure = Fraction(0)
    for exponents, value in determinant.items():
        weight = Fraction(value, denominator)
        for axis, exponent in enumerate(exponents):
            integral = powers.get((axis, exponent))
            if integral is None:
                integral = powers[axis, exponent] = (
                    (origin[axis] + steps[axis]) ** (exponent + 1)
                    - origin[axis] ** (exponent + 1)
                ) / (exponent + 1)
            weight *= integral
        measure += weight
    return _CellIntegralPhase(True, measure)


def _polynomial_cell_integral(
    polynomial: tuple[Expression, ...],
    kind: str,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CellIntegralPhase:
    polynomial_coordinates = tuple(
        value
        for value in polynomial
        if not isinstance(
            value,
            coordinate_enclosure_owner.RationalPolynomial,
        )
    )
    if mapped_coverage_owner.map_domain(kind) == "box" and len(
        polynomial_coordinates
    ) == dimension in (2, 3):
        measure = box_jacobian_measure(polynomial_coordinates, budget=budget)
    else:
        jacobian = tuple(
            tuple(
                coordinate_enclosure_owner.derivative(value, axis, budget=budget)
                for axis in range(dimension)
            )
            for value in polynomial_coordinates
        )
        measure = mapped_coverage_owner.integrate(
            coordinate_enclosure_owner.determinant(jacobian, budget=budget),
            mapped_coverage_owner.map_domain(kind),
            dimension,
            budget=budget,
        )
    return _CellIntegralPhase(True, measure)


def _rational_region_integral(
    region: int,
    regions: np.ndarray,
    cells: list[_CoverageCell],
    facets: Facets,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RationalRegionPhase:
    total = Fraction(0)
    resolved = True
    for occurrence, owner in enumerate(facets.cells.tolist()):
        if regions[owner] != region:
            continue
        group = facets.group[occurrence]
        incident = facets.cells[facets.group == group]
        if incident.size == 2 and np.all(regions[incident] == region):
            continue
        kind, element, controls, vertices = cells[owner]
        with budget.temporary_scope():
            expression = coordinate_enclosure_owner.coordinate_expressions(
                element, controls, budget=budget
            )
            if expression is None or dimension != 3:
                resolved = False
                break
            face = tuple(
                vertices.index(int(vertex))
                for vertex in facets.rows[occurrence]
                if vertex >= 0
            )
            charts = mapped_coverage_owner.prepared_face_charts(
                element, controls, kind, face, budget=budget
            )
            if charts is None:
                resolved = False
                break
            for chart_expression, chart_domain in charts:
                flux = mapped_coverage_owner.chart_flux(
                    chart_expression, chart_domain, budget=budget
                )
                if flux is None:
                    resolved = False
                    break
                total += flux
        if not resolved:
            break
    return _RationalRegionPhase(resolved, total)


@final
class _MappedBlockPhase(StrictModule):
    """Retain controls, chart routes, and the signed reference measure of a block."""

    controls: tuple[CoordinateSourceBank, ...]
    affine: tuple[CellGeometryElement, Fraction] | None
    chart: tuple[CellGeometryElement, tuple[Fraction, ...], tuple[Fraction, ...]] | None
    reference_measure: Fraction


def _prepare_mapped_block(
    block: CellBlock | PolyhedralBlock,
    element: CellGeometryElement,
    route: Array,
    coordinates: CoordinateSourceBank,
    dimension: int,
    ambient_dimension: int,
    boundary_measure: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedBlockPhase:
    if boundary_measure:
        budget.reserve(
            route.size + block.vertices.size,
            256
            + block.cell_count
            * (128 + 8 * route.shape[1] + 48 * block.vertices.shape[1]),
        )
    local = tuple(
        tuple(coordinates[row] for row in indices)
        for indices in np.asarray(route, dtype=np.int64)
    )
    with budget.temporary_scope():
        affine = (
            _affine_root_chart(element, block.cell_kind, dimension, budget=budget)
            if ambient_dimension == dimension and not boundary_measure
            else None
        )
        chart = (
            _axis_aligned_root_chart(element, block.cell_kind, dimension, budget=budget)
            if ambient_dimension == dimension and not boundary_measure
            else None
        )
    reference_measure = mapped_coverage_owner.integrate(
        {(0,) * dimension: Fraction(1)},
        mapped_coverage_owner.map_domain(block.cell_kind),
        dimension,
        budget=budget,
    )
    return _MappedBlockPhase(local, affine, chart, reference_measure)


@final
class _DirectCellIntegralPhase(StrictModule):
    """Retain a direct source integral or its unresolved/rational classification."""

    measure: Fraction | None
    rational: bool
    unresolved: bool


def _direct_cell_integral(
    state: EmbeddingState,
    element: CellGeometryElement,
    controls: CoordinateCoefficients,
    block: CellBlock | PolyhedralBlock,
    dimension: int,
    index: int,
    assigned: bool,
    active: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> _DirectCellIntegralPhase:
    polynomial = coordinate_enclosure_owner.coordinate_expressions(
        element, controls, budget=budget
    )
    if polynomial is None:
        state.add(
            "mapped_measure_source", "unresolved", "cell", (int(block.global_ids[index]),)
        )
        return _DirectCellIntegralPhase(None, False, True)
    if assigned and any(
        isinstance(value, coordinate_enclosure_owner.RationalPolynomial)
        for value in polynomial
    ):
        return _DirectCellIntegralPhase(None, True, False)
    if not active:
        return _DirectCellIntegralPhase(None, False, False)
    phase = _polynomial_cell_integral(
        polynomial, block.cell_kind, dimension, budget=budget
    )
    return _DirectCellIntegralPhase(phase.measure, False, False)


@final
class _BoundaryCellPhase(StrictModule):
    """Retain rational or unknown classification after storing a boundary source."""

    rational: bool
    unresolved: bool


def _prepare_boundary_cell(
    state: EmbeddingState,
    element: CellGeometryElement,
    controls: CoordinateCoefficients,
    block: CellBlock | PolyhedralBlock,
    index: int,
    assigned: bool,
    boundary_maps: list[tuple[Expression, ...] | None],
    *,
    budget: CoordinateEnclosureBudget,
) -> _BoundaryCellPhase:
    polynomial = coordinate_enclosure_owner.coordinate_expressions(
        element, controls, budget=budget
    )
    boundary_maps.append(polynomial)
    if polynomial is None:
        state.add(
            "mapped_measure_source", "unresolved", "cell", (int(block.global_ids[index]),)
        )
        return _BoundaryCellPhase(False, True)
    rational = assigned and any(
        isinstance(value, coordinate_enclosure_owner.RationalPolynomial)
        for value in polynomial
    )
    return _BoundaryCellPhase(rational, False)


def _execute_mapped_cell_integral(
    state: EmbeddingState,
    element: CellGeometryElement,
    controls: CoordinateSourceBank,
    block: CellBlock | PolyhedralBlock,
    index: int,
    dimension: int,
    assigned: bool,
    active: bool,
    prepared: _MappedBlockPhase,
    cache: _RootIntegralCache,
    powers: dict[tuple[int, int], Fraction],
    *,
    budget: CoordinateEnclosureBudget,
) -> _DirectCellIntegralPhase:
    if prepared.affine is not None:
        phase = _affine_root_integral(
            prepared.affine,
            controls,
            dimension,
            prepared.reference_measure,
            active,
            cache,
            budget=budget,
        )
        if phase.handled:
            return _DirectCellIntegralPhase(phase.measure, False, False)
    if prepared.chart is not None:
        phase = _box_root_integral(
            prepared.chart, controls, dimension, active, cache, powers, budget=budget
        )
        if phase.handled:
            return _DirectCellIntegralPhase(phase.measure, False, False)
    return _direct_cell_integral(
        state, element, controls, block, dimension, index, assigned, active, budget=budget
    )


@checked
def mapped_coverage_measures(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    regions: np.ndarray,
    *,
    embedding: GlobalEmbeddingCertificate | None = None,
    facets: Facets | None = None,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    tuple[Fraction | None, ...],
    list[tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]],
]:
    """Compute rational region measures, retaining cell elements and controls.

    A bound, fully traced nonperiodic source chain uses physical boundary
    flux, cancelling only earned internal traces. Otherwise each source
    Jacobian is integrated over its reference chart. Both signed integrals
    share the active ledger.
    """

    dimension = mesh.topological_dimension
    boundary_measure = (
        dimension == mesh.ambient_dimension == 3
        and mesh.periodic_topology is None
        and embedding is not None
        and embedding.status == "certified"
        and {"mapped_trace_continuity", "facet_pairing"}
        <= set(embedding.evaluated_checks)
    )
    if boundary_measure and embedding is not None:
        embedding.binding.require(mesh, geometry)
        budget.reserve(1)
        cached = _cached_boundary_integrals(
            state, mesh, geometry, regions, facets, budget=budget
        )
        if cached is not None:
            return cached.totals, cached.cells
    elements, routes, _ = geometry.resolve(mesh)
    coordinates = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    totals: list[Fraction | None] = [
        Fraction(0) for _ in range(int(np.max(regions, initial=-1)) + 1)
    ]
    cells: list[
        tuple[str, CellGeometryElement, CoordinateCoefficients, tuple[int, ...]]
    ] = []
    rational_regions: set[int] = set()
    boundary_maps: list[tuple[Expression, ...] | None] = []
    cache = _RootIntegralCache({}, {})
    cursor = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        block_phase = _prepare_mapped_block(
            block,
            element,
            route,
            coordinates,
            dimension,
            mesh.ambient_dimension,
            boundary_measure,
            budget=budget,
        )
        local = block_phase.controls
        # Signed one-dimensional integrals of the chart's source interval powers.
        powers: dict[tuple[int, int], Fraction] = {}
        block_vertices = np.asarray(block.vertices, dtype=np.int64)
        for index in range(block.cell_count):
            cells.append(
                (
                    block.cell_kind,
                    element,
                    local[index],
                    tuple(block_vertices[index].tolist()),
                )
            )
            region = int(regions[cursor])
            cursor += 1
            if boundary_measure:
                boundary_phase = _prepare_boundary_cell(
                    state,
                    element,
                    local[index],
                    block,
                    index,
                    region >= 0,
                    boundary_maps,
                    budget=budget,
                )
                if region >= 0:
                    if boundary_phase.unresolved:
                        totals[region] = None
                    elif boundary_phase.rational:
                        rational_regions.add(region)
                continue
            with budget.temporary_scope():
                active = region >= 0 and totals[region] is not None
                phase = _execute_mapped_cell_integral(
                    state,
                    element,
                    local[index],
                    block,
                    index,
                    dimension,
                    region >= 0,
                    active,
                    block_phase,
                    cache,
                    powers,
                    budget=budget,
                )
                if region >= 0:
                    if phase.unresolved:
                        totals[region] = None
                    elif phase.rational:
                        rational_regions.add(region)
                    elif phase.measure is not None:
                        current = totals[region]
                        if current is not None:
                            totals[region] = current + phase.measure
    if boundary_measure:
        _signed_boundary_region_measures(
            state,
            regions,
            cells,
            boundary_maps,
            totals,
            rational_regions,
            mesh_facets(mesh, geometry=geometry, budget=budget)
            if facets is None
            else facets,
            budget=budget,
        )
        state.checks.append("mapped_signed_boundary_region_measure")
        return tuple(totals), cells
    if rational_regions:
        facets = mesh_facets(mesh, geometry=geometry, budget=budget)
        for region in sorted(rational_regions):
            phase = _rational_region_integral(
                region, regions, cells, facets, dimension, budget=budget
            )
            totals[region] = phase.measure if phase.resolved else None
            if not phase.resolved:
                state.add(
                    "mapped_rational_boundary_measure",
                    "unresolved",
                    "source_region",
                    (region,),
                )
    return tuple(totals), cells


@checked
def declared_coverage_measures(
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
) -> tuple[Fraction, ...]:
    """Return signed measures in declared region order."""
    match domain:
        case mapped_reference_domain_owner.MappedReferenceDomain():
            return domain.exact_region_measures()
        case PiecewiseLinearDomain():
            pass
        case invalid:
            assert_never(invalid)
    integers, exponent = dyadic_integers(domain.vertices)
    dimension = domain.ambient_dimension
    factor = Fraction(2) ** (dimension * exponent) / (2 if dimension == 2 else 6)
    weights = np.ones(domain.facets.shape[0], dtype=np.int64)
    return tuple(
        (back + front) * factor
        for back, front in zip(
            _region_measures(
                integers,
                domain.facets,
                weights,
                domain.facet_regions[:, 0],
                len(domain.region_ids),
            ),
            _region_measures(
                integers,
                domain.facets,
                -weights,
                domain.facet_regions[:, 1],
                len(domain.region_ids),
            ),
            strict=True,
        )
    )


def _exact_chart_plane(
    chart: tuple[Expression, ...], /, *, budget: CoordinateEnclosureBudget
) -> tuple[Fraction, ...] | Literal["nonplanar"] | None:
    """Return a polynomial facet chart's rational plane, ``"nonplanar"``, or ``None``.

    The images of the parameter origin and unit points lie on the chart's
    polynomial image. When they span a unique plane, the rational-arithmetic plane
    expression of the chart decides: zero means the whole image lies in that
    plane, keyed with the source groups' rational normalization; nonzero means no
    plane contains the image, so every declared plane rejects it. Rational,
    degenerate or non-hypersurface charts return ``None`` for the full search.
    """

    polynomials = tuple(
        value
        for value in chart
        if not isinstance(value, coordinate_enclosure_owner.RationalPolynomial)
    )
    if len(polynomials) != len(chart) or not polynomials:
        return None
    dimension = next((len(index) for value in polynomials for index in value), None)
    if dimension is None or dimension + 1 != len(polynomials):
        return None
    points = tuple(
        tuple(
            coordinate_enclosure_owner.evaluate(
                value,
                tuple(Fraction(int(axis == unit)) for axis in range(dimension)),
                budget=budget,
            )
            for value in polynomials
        )
        for unit in (-1, *range(dimension))
    )
    key = planar_primitives_owner.plane_key(points, budget=budget)
    if key is None:
        return None
    plane = tuple(
        rational_coefficient(value, role="coverage.facet_plane") for value in key[0]
    )
    if all(sum(index) <= 1 for value in polynomials for index in value):
        # An affine image is the affine hull of these spanning points.
        return plane
    residual = coordinate_enclosure_owner.add(
        coordinate_enclosure_owner.sum_polynomials(
            tuple(
                coordinate_enclosure_owner.scale(value, weight, budget=budget)
                for value, weight in zip(polynomials, plane[:-1], strict=True)
            ),
            budget=budget,
        ),
        coordinate_enclosure_owner.constant(plane[-1], dimension, budget=budget),
        budget=budget,
    )
    return "nonplanar" if residual else plane


@final
class _MappedCoveragePhase(StrictModule):
    """Retain source partitions, rational overlap accumulation, and subdivision work."""

    source: tuple[planar_coverage_owner.SourceGroup, ...]
    by_plane: dict[tuple[Fraction, ...], list[int]]
    every_target: tuple[int, ...]
    covered: list[Fraction]
    uncertain_sources: set[int]
    subdivision_work: mapped_coverage_owner.SubdivisionLedger
    relations: dict[CoverageOverlapKey, Fraction]
    exceeded: bool


@final
class _MappedChartCoveragePhase(StrictModule):
    """Retain new overlap keys after one chart's containment verdict."""

    retained_keys: tuple[CoverageOverlapKey, ...]


def _prepare_mapped_coverage(
    state: EmbeddingState,
    domain: PiecewiseLinearDomain,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedCoveragePhase:
    source, source_overlaps, used, exceeded = planar_coverage_owner.source_groups(
        domain.vertices,
        domain.facets,
        domain.facet_regions,
        limits.maximum_candidate_pairs - state.candidate_pairs,
        budget=budget,
    )
    state.candidate_pairs += used
    for first, second in source_overlaps:
        state.add(
            "overlapping_source_facets", "violated", "source_facet", (first, second)
        )
    if exceeded:
        state.add("coverage_candidate_capacity", "unresolved", "mesh")
    # Source groups follow normalized rational planes, in source order. A chart
    # whose image lies in another plane, or whose region pair differs, is
    # "absent" for that group by definition of the containment proof.
    by_plane: dict[tuple[Fraction, ...], list[int]] = {}
    budget.reserve(len(source), 128 * len(source))
    for target, declared in enumerate(source):
        plane = tuple(
            rational_coefficient(value, role="coverage.source_plane")
            for value in declared.plane
        )
        by_plane.setdefault(plane, []).append(target)
    every_target = tuple(range(len(source)))
    covered = [Fraction(0) for _ in source]
    uncertain_sources: set[int] = set()
    subdivision_work = mapped_coverage_owner.SubdivisionLedger(
        candidate_pairs=state.candidate_pairs
    )
    relations: dict[CoverageOverlapKey, Fraction] = {}
    return _MappedCoveragePhase(
        source,
        by_plane,
        every_target,
        covered,
        uncertain_sources,
        subdivision_work,
        relations,
        exceeded,
    )


def _certify_mapped_chart(
    state: EmbeddingState,
    chart: tuple[Expression, ...],
    chart_domain: str,
    own: int,
    other: int,
    entity: int,
    phase: _MappedCoveragePhase,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedChartCoveragePhase:
    source, by_plane, every_target = phase.source, phase.by_plane, phase.every_target
    covered, uncertain_sources = phase.covered, phase.uncertain_sources
    subdivision_work, relations, exceeded = (
        phase.subdivision_work,
        phase.relations,
        phase.exceeded,
    )
    facet_keys: list[CoverageOverlapKey] = []
    matched = False
    unresolved = exceeded
    plane = _exact_chart_plane(chart, budget=budget)
    targets = (
        every_target
        if plane is None
        else ()
        if plane == "nonplanar"
        else tuple(
            target
            for target in by_plane.get(plane, ())
            if source[target].regions in ((own, other), (other, own))
        )
    )
    for target in targets:
        declared = source[target]
        outcome, measure, candidates = planar_coverage_owner.mapped_containment(
            chart,
            chart_domain,
            declared,
            own,
            other,
            limits.maximum_bernstein_nodes,
            limits.maximum_subdivision_depth,
            limits.maximum_subdivision_pieces,
            limits.maximum_candidate_pairs,
            subdivision_work,
            budget=budget,
        )
        if outcome == "proven":
            covered[target] += measure
            for local in candidates:
                member = declared.members[local]
                source_measure = declared.measures[local]
                key: CoverageOverlapKey = (
                    entity,
                    member,
                    declared.axes,
                    "candidate",
                    "physical",
                )
                if key not in relations:
                    facet_keys.append(key)
                relations[key] = min(
                    source_measure,
                    relations.get(key, Fraction(0)) + min(measure, source_measure),
                )
            matched = True
            break
        if outcome == "unresolved":
            uncertain_sources.add(target)
            unresolved = True
    if not matched:
        state.add(
            "mapped_facet_containment"
            if unresolved
            else "unmatched_interface_facet"
            if other >= 0
            else "unmatched_boundary_facet",
            "unresolved" if unresolved else "violated",
            "facet",
            (entity,),
        )
    return _MappedChartCoveragePhase(tuple(facet_keys))


def _mapped_domain_coverage(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
    *,
    budget: CoordinateEnclosureBudget,
) -> DomainCoverageCertificate:

    facets = mesh_facets(mesh, geometry=geometry, budget=budget)
    totals, cells = mapped_coverage_measures(
        state, mesh, geometry, regions, embedding=embedding, facets=facets, budget=budget
    )
    achieved = (
        *totals,
        *(Fraction(0) for _ in range(len(domain.region_ids) - len(totals))),
    )
    phase = _prepare_mapped_coverage(state, domain, limits, budget=budget)
    source, covered = phase.source, phase.covered
    uncertain_sources, relations = phase.uncertain_sources, phase.relations
    subdivision_work = phase.subdivision_work
    groups = facets.groups
    entity_ids = facet_entity_ids(mesh, facets.rows)
    for group in groups:
        occurrence = int(group[0])
        own = int(regions[facets.cells[occurrence]])
        other = int(regions[facets.cells[group[1]]]) if group.size == 2 else -1
        if group.size > 2 or own == other:
            continue
        kind, element, controls, vertices = cells[int(facets.cells[occurrence])]
        rows = facets.rows[occurrence]
        entity = int(entity_ids[occurrence])
        facet_keys: list[CoverageOverlapKey] = []
        # The source map remains ledger-owned; only this facet's workspace ends.
        with budget.temporary_scope():
            face = tuple(vertices.index(int(vertex)) for vertex in rows if vertex >= 0)
            charts = mapped_coverage_owner.prepared_face_charts(
                element, controls, kind, face, budget=budget
            )
            if charts is None:
                state.add("mapped_boundary_source", "unresolved", "facet", (entity,))
                uncertain_sources.update(range(len(source)))
                continue
            for chart, chart_domain in charts:
                chart_phase = _certify_mapped_chart(
                    state,
                    chart,
                    chart_domain,
                    own,
                    other,
                    entity,
                    phase,
                    limits,
                    budget=budget,
                )
                facet_keys.extend(chart_phase.retained_keys)
        if facet_keys:
            # Overlap relations outlive the facet workspace: their rational keys and
            # measures stay charged to this owner's storage for the whole proof.
            bits = max(
                abs(relations[key].numerator).bit_length()
                + relations[key].denominator.bit_length()
                for key in facet_keys
            )
            reserve_polynomial(len(facet_keys), len(facet_keys), 5, bits, budget=budget)
    state.candidate_pairs = subdivision_work.candidate_pairs
    state.subdivision_pieces = subdivision_work.pieces
    state.subdivision_depth = subdivision_work.maximum_depth
    requested = declared_coverage_measures(domain)
    equal = 0
    for target, (actual, declared) in enumerate(zip(covered, source, strict=True)):
        expected = sum(declared.measures, Fraction(0))
        boundary = -1 in declared.regions
        if actual == expected:
            # Compact contained images, interior-disjointness and equal rational union
            # measure imply coverage of every declared member, not merely
            # almost-everywhere coverage of a merged scientific face.
            equal += len(declared.members)
        elif target not in uncertain_sources:
            check = (
                ("uncovered_boundary" if boundary else "omitted_interface")
                if actual < expected
                else (
                    "double_covered_boundary" if boundary else "double_covered_interface"
                )
            )
            state.add(check, "violated", "source_facet", declared.members)
    return coverage_result(
        binding,
        embedding,
        domain,
        state,
        requested,
        achieved,
        equal,
        relations,
        (),
        expression_work=active_expression_work(budget=budget),
    )


@final
class _CoverageRegionPhase(StrictModule):
    """Bind cell assignments and identifiers to one checked host extent."""

    __strict_contract__ = True
    regions: HostInt64[MeshCellDim]
    cell_ids: HostInt64[MeshCellDim]


@final
class _CoverageScopePhase(StrictModule):
    """Retain coordinate classification or its measured resource refusal."""

    scope: CoordinateMapScope = eqx.field(static=True)
    prepared_source: CoordinateSourceBank
    resource: CoordinateEnclosureResourceError | None


@final
class _CoverageExecutionPhase(StrictModule):
    """Retain the certificate produced by one physical coverage route."""

    certificate: DomainCoverageCertificate


@final
class _AffineCoveragePhase(StrictModule):
    """Retain affine region integrals and source-facet relations in rational arithmetic."""

    requested: tuple[Fraction, ...]
    achieved: tuple[Fraction, ...]
    covered: int
    relations: dict[CoverageOverlapKey, Fraction]


def _prepare_coverage_regions(
    mesh: CellMesh,
    cell_regions: ArrayLike,
    region_count: int,
) -> _CoverageRegionPhase:
    scope = Scope()
    regions = parse(
        np.asarray(cell_regions, dtype=np.int64),
        HostInt64[MeshCellDim],
        "cell_regions",
        scope=scope,
    )
    cell_ids = cell_global_ids(mesh)
    if regions.shape[0] != cell_ids.size:
        raise ValueError(
            f"certify_domain_coverage.cell_regions must assign every mesh cell; got {regions.shape[0]} assignments for {cell_ids.size} cells"
        )
    if np.any(regions < -1) or np.any(regions >= region_count):
        raise ValueError(
            "certify_domain_coverage.cell_regions must index declared regions or be -1; got out-of-range indices"
        )
    return _CoverageRegionPhase(
        regions,
        parse(cell_ids, HostInt64[MeshCellDim], "cell_ids", scope=scope),
    )


def _prepare_coverage_scope(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    embedding: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits,
    preparation: ExitStack,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoverageScopePhase:
    resource: CoordinateEnclosureResourceError | None = None
    try:
        # Admission and classification share the phase-local refusal measurement.
        with budget.measurement():
            preparation.enter_context(budget.activate())
            preparation.enter_context(
                budget.bound_stage(
                    limits.maximum_work_units, limits.maximum_scratch_bytes
                )
            )
            prepared_scope = coordinate_scope(mesh, geometry, budget=budget)
            scope = prepared_scope.coordinate_scope
            prepared_source = prepared_scope.source_values
    except CoordinateEnclosureResourceError as error:
        scope = embedding.binding.coordinate_scope
        prepared_source = ()
        resource = error
    return _CoverageScopePhase(scope, prepared_source, resource)


def _coverage_resource_refusal(
    state: EmbeddingState,
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
    error: CoordinateEnclosureResourceError,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoverageExecutionPhase:
    state.add(
        "mapped_coverage_resource_budget",
        "unresolved",
        "mesh",
        resource_error=error,
        expression_budget=budget,
    )
    return _CoverageExecutionPhase(
        coverage_result(
            binding,
            embedding,
            domain,
            state,
            declared_coverage_measures(domain),
            tuple(None for _ in domain.region_ids),
            0,
            {},
            (),
            expression_work=active_expression_work(budget=budget),
        )
    )


def _execute_mapped_coverage(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
    regions: np.ndarray,
    limits: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoverageExecutionPhase:
    try:
        with budget.measurement(), budget.activate():
            match domain:
                case mapped_reference_domain_owner.MappedReferenceDomain():
                    certificate = (
                        mapped_reference_proofs().certify_mapped_reference_coverage(
                            state,
                            mesh,
                            geometry,
                            domain,
                            regions,
                            limits,
                            binding,
                            embedding,
                            budget=budget,
                        )
                    )
                case PiecewiseLinearDomain():
                    certificate = _mapped_domain_coverage(
                        state,
                        mesh,
                        geometry,
                        domain,
                        regions,
                        limits,
                        binding,
                        embedding,
                        budget=budget,
                    )
                case invalid:
                    assert_never(invalid)
            return _CoverageExecutionPhase(certificate)
    except CoordinateEnclosureResourceError as error:
        return _coverage_resource_refusal(
            state, binding, embedding, domain, error, budget=budget
        )


def _execute_affine_coverage(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    regions: np.ndarray,
    limits_: MeshCertificateLimits,
    binding: MeshCertificateBinding,
    embedding: GlobalEmbeddingCertificate,
    prepared_source: CoordinateSourceBank,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoverageExecutionPhase:
    expression_work: tuple[int, int, int]
    match geometry.exact_source:
        case None:
            # Binary64 mesh-coordinate predicates form no coordinate expression.

            terms = _affine_coverage_terms(
                state,
                mesh,
                domain,
                regions,
                np.asarray(mesh.coordinates, dtype=np.float64),
                limits_,
                budget=budget,
            )
            expression_work = active_expression_work(budget=budget)
        case ExactAlgebraicCellGeometrySource():
            raise ValueError(
                "certify_domain_coverage.geometry.exact_source must support rational polyhedral predicates; got an algebraic nodal source"
            )
        case (
            exact_rational_geometry_owner.ExactRationalCellGeometrySource() as source
        ) if source.field_degree != 1:
            raise ValueError(
                "certify_domain_coverage.geometry.exact_source must support rational polyhedral predicates; got algebraic interpolation ancestry"
            )
        case (
            exact_power_geometry_owner.ExactPowerCellGeometrySource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
            | exact_plc_geometry_owner.ExactPlcCellGeometrySource()
            | exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
        ):
            # The same request ledger owns source vertices and the proof.
            try:
                with budget.measurement(), budget.activate():
                    exact_points = np.asarray(prepared_source, dtype=object)
                    predicate_points, _ = exact_source_predicate_view(
                        exact_points, budget=budget
                    )
                    terms = _affine_coverage_terms(
                        state,
                        mesh,
                        domain,
                        regions,
                        predicate_points,
                        limits_,
                        budget=budget,
                    )
            except CoordinateEnclosureResourceError as error:
                return _coverage_resource_refusal(
                    state, binding, embedding, domain, error, budget=budget
                )
            expression_work = active_expression_work(budget=budget)
        case invalid:
            assert_never(invalid)
    return _CoverageExecutionPhase(
        coverage_result(
            binding,
            embedding,
            domain,
            state,
            terms.requested,
            terms.achieved,
            terms.covered,
            terms.relations,
            (),
            expression_work=expression_work,
        )
    )


@checked
def certify_domain_coverage(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
    cell_regions: ArrayLike,
    /,
    *,
    embedding: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> DomainCoverageCertificate:
    """Certify coverage of an independently declared planar or mapped volume.

    ``cell_regions`` assigns every cell (mesh block order) exclusively to one
    index of ``domain.region_ids``; ``-1`` marks an unassigned cell. The
    embedding certificate establishes the required interior disjointness.
    """

    embedding.binding.require(mesh, geometry)
    limits_ = MeshCertificateLimits() if limits is None else limits
    dimension = mesh.topological_dimension
    if dimension != mesh.ambient_dimension or dimension != domain.ambient_dimension:
        raise ValueError(
            f"certify_domain_coverage.mesh must be a volume of domain dimension {domain.ambient_dimension}; got topological dimension {dimension} and ambient dimension {mesh.ambient_dimension}"
        )
    if mesh.storage is not None:
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            embedding.binding.coordinate_scope,
            limits_,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
        )
        state = EmbeddingState([], ["domain_coverage"])
        state.add("owner_local_domain_coverage_premise", "unresolved", "mesh")
        return coverage_result(
            binding,
            embedding,
            domain,
            state,
            declared_coverage_measures(domain),
            tuple(None for _ in domain.region_ids),
            0,
            {},
            (),
            expression_work=(0, 0, 0),
        )
    assignment = _prepare_coverage_regions(mesh, cell_regions, len(domain.region_ids))
    regions, cell_ids = assignment.regions, assignment.cell_ids
    with coordinate_certificate_request(limits_, budget=budget) as budget:
        with ExitStack() as preparation:
            phase = _prepare_coverage_scope(
                mesh, geometry, embedding, limits_, preparation, budget=budget
            )
            binding = MeshCertificateBinding(
                mesh,
                geometry,
                phase.scope,
                limits_,
                source_id=domain.source_id,
                source_revision=domain.source_revision,
            )
            state = EmbeddingState([], ["domain_coverage"])
            if phase.resource is not None:
                return _coverage_resource_refusal(
                    state,
                    binding,
                    embedding,
                    domain,
                    phase.resource,
                    budget=budget,
                ).certificate
            if embedding.status != "certified":
                state.add("embedding_premise", "unresolved", "mesh")
            unassigned = regions < 0
            if np.any(unassigned):
                state.add("unassigned_cell", "violated", "cell", cell_ids[unassigned])
            match domain:
                case mapped_reference_domain_owner.MappedReferenceDomain():
                    result = _execute_mapped_coverage(
                        state,
                        mesh,
                        geometry,
                        domain,
                        regions,
                        limits_,
                        binding,
                        embedding,
                        budget=budget,
                    )
                case PiecewiseLinearDomain():
                    match phase.scope:
                        case "mapped":
                            result = _execute_mapped_coverage(
                                state,
                                mesh,
                                geometry,
                                domain,
                                regions,
                                limits_,
                                binding,
                                embedding,
                                budget=budget,
                            )
                        case "affine":
                            result = _execute_affine_coverage(
                                state,
                                mesh,
                                geometry,
                                domain,
                                regions,
                                limits_,
                                binding,
                                embedding,
                                phase.prepared_source,
                                budget=budget,
                            )
                        case invalid_scope:
                            assert_never(invalid_scope)
                case invalid:
                    assert_never(invalid)
            return result.certificate


def _affine_coverage_terms(
    state: EmbeddingState,
    mesh: CellMesh,
    domain: PiecewiseLinearDomain,
    regions: np.ndarray,
    points: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineCoveragePhase:
    """Compute affine coverage of binary64 or rational source points.

    Every rational-point operation charges the active coordinate ledger.
    Returns requested and achieved region measures, covered source facets and
    overlap relations.
    """
    dimension = mesh.topological_dimension
    facets = mesh_facets(mesh, budget=budget)
    covering = _covering_pieces(
        state, mesh, points, facets, regions, limits, budget=budget
    )
    declared_corners = domain.vertices[domain.facets]
    axes, orientation, certain = _projection_axes(declared_corners, budget=budget)
    if not np.all(certain):
        state.add(
            "declared_projection_predicates",
            "unresolved",
            "source_facet",
            np.flatnonzero(~certain),
        )
    covered, unmatched, incomplete, relations = _partition_covering_pieces(
        state, covering, points, domain, axes, limits, budget=budget
    )
    interface = covering.other_region >= 0
    for check, rows in (
        ("unmatched_boundary_facet", unmatched & ~interface),
        ("unmatched_interface_facet", unmatched & interface),
    ):
        if np.any(rows):
            state.add(check, "violated", "facet", covering.facet_entities[rows])
    stacked = np.concatenate((points, domain.vertices))

    integers, coordinate_scale = exact_polyhedral_geometry_owner.coordinate_integers(
        stacked, budget=budget
    )
    mesh_integers = integers[: points.shape[0]]
    declared_integers = integers[points.shape[0] :]
    projected_factor = coordinate_scale ** (dimension - 1)
    if dimension == 3:
        projected_factor /= 2
    declared_measure = tuple(
        value * projected_factor
        for value in _projected_measures(declared_integers, domain.facets, axes).tolist()
    )
    boundary_facet = np.any(domain.facet_regions < 0, axis=1)
    deficit = np.asarray(
        [value < total for value, total in zip(covered, declared_measure, strict=True)],
        dtype=np.bool_,
    )
    excess = np.asarray(
        [value > total for value, total in zip(covered, declared_measure, strict=True)],
        dtype=np.bool_,
    )
    for check, rows in (
        ("uncovered_boundary", deficit & boundary_facet),
        ("omitted_interface", deficit & ~boundary_facet),
        ("double_covered_boundary", excess & boundary_facet),
        ("double_covered_interface", excess & ~boundary_facet),
    ):
        if np.any(rows) and not incomplete:
            state.add(check, "violated", "source_facet", np.flatnonzero(rows))
    region_count = len(domain.region_ids)
    declared_totals = [
        back + front
        for back, front in zip(
            _region_measures(
                declared_integers,
                domain.facets,
                np.ones((domain.facets.shape[0],), dtype=np.int64),
                domain.facet_regions[:, 0],
                region_count,
            ),
            _region_measures(
                declared_integers,
                domain.facets,
                -np.ones((domain.facets.shape[0],), dtype=np.int64),
                domain.facet_regions[:, 1],
                region_count,
            ),
            strict=True,
        )
    ]
    pieces, owners = _cell_measure_pieces(
        state, mesh, points, facets, limits, budget=budget
    )
    achieved_totals = _region_measures(
        mesh_integers,
        pieces,
        np.ones((owners.size,), dtype=np.int64),
        regions[owners],
        region_count,
    )
    denominator = 2 if dimension == 2 else 6
    factor = coordinate_scale**dimension / denominator
    return _AffineCoveragePhase(
        tuple(value * factor for value in declared_totals),
        tuple(value * factor for value in achieved_totals),
        int(np.count_nonzero(~deficit & ~excess)),
        relations,
    )


__all__ = [
    "CoverageCoordinateSpace",
    "CoverageEvidence",
    "CoverageExpressionWork",
    "CoverageMeasures",
    "CoverageOverlapKey",
    "CoverageOverlapSemantics",
    "DomainCoverageCertificate",
    "PiecewiseLinearDomain",
    "box_jacobian_measure",
    "certify_domain_coverage",
    "coverage_result",
    "declared_coverage_measures",
    "mapped_coverage_measures",
    "mapped_reference_proofs",
]
