#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify global cell embedding from contact and oriented boundary-degree invariants.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "cell_validity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The bound cell validity status vector is read in mesh block order."
      }
    ],
    "conclusion": "Record the local orientation decisions as the input obligation, without inferring global injectivity.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._validity_findings",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "invalid_cell"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The cell validity status is INVALID for the identified cell."
      }
    ],
    "conclusion": "A failed local orientation decision violates the embedding request.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._validity_findings",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "unresolved_cell_validity",
      "cell_validity_embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "At least one bound cell does not have CERTIFIED_VALID status."
      }
    ],
    "conclusion": "Global embedding remains unresolved until every cell has a local orientation decision.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._validity_findings; certify_global_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "facet_pairing"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Facet occurrences are grouped by shared vertex identity and read in their induced orientations."
      }
    ],
    "conclusion": "Record the combinatorial pairing obligation independently of geometric contacts.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.pairing_findings",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "nonmanifold_facet",
      "inconsistent_facet_orientation"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "An undeclared facet has more than two incident cells, or two incident cells induce equal rather than opposite facet orientations."
      }
    ],
    "conclusion": "The assembled cell chain fails its stated manifold or oriented-pairing contract.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.pairing_findings",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "cell_contact",
      "boundary_contact"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Closed-box candidates contain every potential segment or triangle contact."
      },
      {
        "tag": "X",
        "statement": "Orientation predicates decide contact beyond shared mesh vertex identities in rational arithmetic."
      }
    ],
    "conclusion": "A decided nonshared contact violates cell or boundary embedding.",
    "citation": {
      "kind": "mathematical",
      "reference": "Jonathan Richard Shewchuk, Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates (1997)",
      "locator": "orientation determinants and exact sign decisions, Discrete & Computational Geometry 18:305\u2013363",
      "url": "https://www.cs.cmu.edu/~quake/robust.html"
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "cell_contact_capacity",
      "boundary_contact_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The number of closed-box candidate pairs exceeds the remaining maximum_candidate_pairs allowance."
      }
    ],
    "conclusion": "The contact proof is unresolved rather than discarding unvisited candidates.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.pairwise_contacts",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "cell_contact_predicates",
      "boundary_contact_predicates"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A candidate contact predicate has no certain sign decision."
      }
    ],
    "conclusion": "The identified candidate entities remain unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.pairwise_contacts",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "interval_overlap"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Sorted closed intervals retain their declared endpoint vertex identities."
      }
    ],
    "conclusion": "A positive overlap of interval interiors violates embedding beyond a shared endpoint.",
    "citation": {
      "kind": "mathematical",
      "reference": "Jonathan Richard Shewchuk, Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates (1997)",
      "locator": "orientation determinants and exact sign decisions, Discrete & Computational Geometry 18:305\u2013363",
      "url": "https://www.cs.cmu.edu/~quake/robust.html"
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "nonplanar_cell",
      "undecided_cell_triangulation"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The affine surface route requires a planar, decided triangulation of each complete cell loop."
      }
    ],
    "conclusion": "An unavailable affine decomposition leaves the listed surface cells unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.surface_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "curved_boundary_facet",
      "undecided_boundary_facet"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The affine boundary route requires a decided planar triangulation of each boundary facet."
      }
    ],
    "conclusion": "An unavailable facet decomposition leaves the boundary proof unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.boundary_pieces",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "nonmanifold_boundary_ridge"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each oriented boundary ridge must cancel between exactly paired boundary facets."
      }
    ],
    "conclusion": "An unbalanced boundary shell has no closed-chain degree conclusion.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.boundary_shells",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "exterior_degree"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "The cell validity certificate bounds each closed-cell Jacobian determinant strictly above zero."
      },
      {
        "tag": "X",
        "statement": "Interior facets cancel with opposite orientations and the boundary has no nonshared contacts."
      },
      {
        "tag": "T",
        "statement": "Degree is the signed preimage count and changes by one across an embedded oriented shell."
      }
    ],
    "conclusion": "Record the exterior-degree obligation whose zero value on every shell bounds the covering degree by one.",
    "citation": {
      "kind": "mathematical",
      "reference": "Allen Hatcher, Algebraic Topology (2002)",
      "locator": "Section 2.2, Degree; Section 2.B, Jordan separation",
      "url": "https://math.uchicago.edu/~farb/algtop2023/Hatcher.pdf"
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "nested_component"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Rational signed shell measure fixes its orientation and symbolic ray crossings count other shells."
      },
      {
        "tag": "T",
        "statement": "Positive local degrees add to the winding degree of the oriented boundary chain."
      }
    ],
    "conclusion": "A nonzero exterior degree proves an extra covering component and violates embedding.",
    "citation": {
      "kind": "mathematical",
      "reference": "Allen Hatcher, Algebraic Topology (2002)",
      "locator": "Section 2.2, Degree; Section 2.B, Jordan separation",
      "url": "https://math.uchicago.edu/~farb/algtop2023/Hatcher.pdf"
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "checks": [
      "exterior_degree_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A prior boundary contact, pairing or triangulation obligation is unresolved or violated."
      }
    ],
    "conclusion": "Exterior degree cannot conclude global embedding without those decisions.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.volume_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "exterior_degree_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The shell-to-boundary ray-test count exceeds maximum_ray_tests."
      }
    ],
    "conclusion": "The exterior-degree request is unresolved before excess ray work executes.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._exterior_degrees",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "degenerate_shell"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A boundary shell has zero signed measure or no vertex exclusive to that shell."
      }
    ],
    "conclusion": "This exterior-degree route has no admissible shell orientation or ray origin.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._exterior_degrees",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "exterior_degree_predicates"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "At least one symbolic ray crossing lacks a certain orientation sign."
      }
    ],
    "conclusion": "The shell exterior degree remains unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._exterior_degrees",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "exact_plc_source_coordinates"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The authenticated PLC coordinate bank supplies rational source vertex values."
      }
    ],
    "conclusion": "The contact and degree proof uses those source values rather than only the rounded mesh carrier.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._exact_source_embedding",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "exact_source_binary64_value_equivalence"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every rational coordinate in the predicate input bank equals Fraction(float(source_coordinate))."
      }
    ],
    "conclusion": "The exported binary64 predicate view represents the same source values.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.exact_source_predicate_view",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "exact_source_bit_budget"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The integer-scaling profile cannot fit the source maximum_bits allowance."
      }
    ],
    "conclusion": "Source embedding is unresolved before inadmissible integer scaling is formed.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding._exact_source_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "exact_source_resource_budget",
      "mapped_source_expression_resource_budget"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The coordinate budget refuses a guarded source or predicate phase at its charged work or storage allowance."
      }
    ],
    "conclusion": "Embedding remains unresolved and carries the refusing phase quantities.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.certify_global_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "owner_local_global_embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The mesh has owner-local storage without the global cell bank required by this route."
      }
    ],
    "conclusion": "This global proof route remains unresolved rather than treating a local subset as the domain.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.certify_global_embedding",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "unsupported_embedding_dimension"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The topological and ambient dimension pair has no admitted interval, surface or volume route."
      }
    ],
    "conclusion": "The unsupported embedding request remains unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Phydrax mesh certificate contract",
      "locator": "_global_embedding.certify_global_embedding",
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
import sys
from collections.abc import Iterator
from contextlib import (
    AbstractContextManager,
    contextmanager,
    ExitStack,
    nullcontext,
)
from dataclasses import dataclass
from fractions import Fraction
from typing import assert_never, cast, final, Literal

import equinox as eqx
import jax
import numpy as np
from jax.typing import ArrayLike
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.discretization._coordinate_identity as coordinate_identity_owner
import phydrax.discretization._exact_algebraic_geometry as exact_algebraic_geometry_owner
import phydrax.discretization._exact_plc_geometry as exact_plc_geometry_owner
import phydrax.discretization._exact_power_geometry as exact_power_geometry_owner
import phydrax.discretization._exact_rational_geometry as exact_rational_geometry_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
import phydrax.geometry._exact_polyhedral_geometry as exact_polyhedral_geometry_owner
import phydrax.geometry._exact_power_certificates as exact_power_certificates_owner
import phydrax.meshing._audit_topology as audit_topology_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    bigint_bytes,
    exact_bits,
    exact_charge,
    exact_reserve,
    HostPredicateResult,
    orient2d_host,
    orient3d_host,
    PredicateMode,
    resolve_host_predicate_mode,
    segment_intersections_2d,
    SegmentIntersectionStatus,
)
from .._issued_proofs import IssuedProofs
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer
from ..discretization._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    ExactCellGeometrySource,
)
from ..discretization._cell_geometry_predecessor import CellGeometryPredecessor
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityStatus,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ..discretization._coordinate_enclosure import CoordinateSourceBank, Polynomial
from ..discretization._reference_cell import reference_cell_topology
from ..discretization._reference_cell_types import CellGeometryElement
from ..linalg._small_batched import cofactor_determinant_2, cofactor_determinant_3
from ..typing import checked, parse
from ._embedding_state import (
    boxes,
    candidate_pairs,
    coordinate_certificate_request,
    CoordinateMapScope,
    CoordinateScope,
    EmbeddingState,
    facet_entity_ids,
    Facets,
    MappedEmbeddingCertifier,
    MappedRestrictionRootEmbeddingEvidence,
    mesh_facets,
    MeshCertificateEntityKind,
    MeshCertificateFinding,
    MeshCertificateLimits,
    padded_facet_rows,
    PeriodicAffineQuotientEvidence,
    PeriodicAffineTorusDegreeEvidence,
    PeriodicRestrictionPartitionEvidence,
)
from ._mesh_certificates import (
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._mesh_check_names import MeshCertificateCheck
from ._periodic_mapped_restriction import PeriodicMappedRestrictionEvidence
from ._periodic_restriction import PeriodicRestrictionPremise
from ._polygon_triangulation import loop_triangles
from ._rational import dyadic_integers, rational_vector
from ._source_boundary_records import CadBoundaryClosureEvidence


type MappedBoundaryDegreeStatus = Literal["embedded", "violated", "premise_unproven"]


@final
class MappedBoundaryDegreeEvidence(StrictModule, NonTrainableState):
    """Premises and outcome of the boundary-degree theorem for mapped volume meshes.

    Boundary-degree statement. Let ``M`` be a mesh of triangles/quadrilaterals in ``R^2`` or of
    tetrahedra/prisms/hexahedra in ``R^3`` whose interior facets pair exactly
    two cells with opposite induced orientations, and let ``F`` be its piecewise
    polynomial coordinate map. Suppose

    (a) ``det DF_c > 0`` on every closed reference cell (the bound validity
        certificate), so each polynomial map extends to an orientation-preserving
        local diffeomorphism near each point of its closed reference cell;
    (b) cells that share a vertex, an edge or a facet have identical
        restricted expressions in rational arithmetic, so ``F`` is well defined on ``|M|``;
    (c) ``F`` is injective on the union of boundary facets;
    (d) every boundary shell ``j`` (a ridge-connected, balanced boundary
        component) has exterior degree ``e_j = a_j + c_j - 1 = 0``, where
        ``a_j`` is the winding number of all other shells at a point of
        shell ``j`` and ``c_j`` is ``1`` for a positively and ``0`` for a
        negatively oriented shell.

    Then ``F`` is injective on ``|M|``: cell interiors are pairwise disjoint
    and closed cells meet only on combinatorially shared entities.

    Proof outline. Off the image of the facet skeleton, the preimage count is
    the sum of positive per-cell local degrees, even if a cell is not known
    injective on its whole closed reference domain. Interior facet traces
    cancel, so this count equals the winding of the oriented boundary cycle.
    Every complement component of the embedded boundary borders some shell;
    (d) bounds this nonnegative count by one almost everywhere.
    To include exceptional skeleton images, take a sufficiently small closed
    star about an interior source point. Its oriented facets cancel locally,
    and its remaining boundary avoids that point's image. Its local degree is
    positive: one incident cell contributes an open sector by (a), all regular
    contributions are nonnegative, and degree is constant in the avoiding ball.
    Two distinct interior preimages would therefore give count at least two
    on an open set. An interior preimage of a boundary image would give positive
    count on the degree-zero exterior side. Both contradict the same bound.
    Boundary preimages are unique by (c). Thus no separate closed-cell
    injectivity premise is necessary. Generic callers additionally prove it;
    the CAD route instead imports an independently proved whole-boundary isotopy.
    Several shells take ``a_j`` from exact symbolic-ray winding of affine
    proxies, with certified query-avoiding deviation homotopies, and ``c_j``
    from exact signed shell measures. A single embedded shell has exterior
    degree zero because its adjacent positive cell occupies its covered side.

    ``status`` is ``embedded`` when every hypothesis is proved, ``violated`` when
    proved hypotheses imply a nonconforming map, a boundary self-contact or a
    degree above one (``failed_premise`` names the deciding check), and
    ``premise_unproven`` when the route falls back to pairwise cell contacts.
    """

    status: MappedBoundaryDegreeStatus = eqx.field(static=True)
    failed_premise: str | None = eqx.field(static=True)
    oriented_cell_count: int = eqx.field(static=True)
    conforming_entity_count: int = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    boundary_pair_count: int = eqx.field(static=True)
    tangent_cone_contact_count: int = eqx.field(static=True)
    shell_count: int = eqx.field(static=True)
    shell_exterior_degrees: tuple[int, ...] = eqx.field(static=True)
    maximum_degree: int | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: MappedBoundaryDegreeStatus,
        failed_premise: str | None,
        /,
        *,
        oriented_cell_count: int,
        conforming_entity_count: int,
        boundary_facet_count: int,
        boundary_pair_count: int,
        tangent_cone_contact_count: int,
        shell_count: int,
        shell_exterior_degrees: tuple[int, ...],
        maximum_degree: int | None,
    ) -> None:
        status_ = parse(status, MappedBoundaryDegreeStatus, "status")
        premise = (
            None
            if failed_premise is None
            else canonical_identifier(failed_premise, "failed_premise")
        )
        counts = {
            name: nonnegative_integer(value, name)
            for name, value in (
                ("oriented_cell_count", oriented_cell_count),
                ("conforming_entity_count", conforming_entity_count),
                ("boundary_facet_count", boundary_facet_count),
                ("boundary_pair_count", boundary_pair_count),
                ("tangent_cone_contact_count", tangent_cone_contact_count),
                ("shell_count", shell_count),
            )
        }
        if not isinstance(shell_exterior_degrees, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in shell_exterior_degrees
        ):
            raise TypeError(
                f"MappedBoundaryDegreeEvidence.shell_exterior_degrees must be an integer tuple; got {shell_exterior_degrees!r}"
            )
        degree = (
            None
            if maximum_degree is None
            else nonnegative_integer(maximum_degree, "maximum_degree")
        )
        if (status_ == "embedded") != (premise is None):
            raise ValueError(
                f"MappedBoundaryDegreeEvidence.failed_premise must be absent exactly for embedded status; got status={status_!r}, failed_check={premise!r}"
            )
        if status_ == "embedded" and (
            degree != 1
            or len(shell_exterior_degrees) != counts["shell_count"]
            or any(shell_exterior_degrees)
        ):
            raise ValueError(
                f"MappedBoundaryDegreeEvidence.shell_exterior_degrees must be zero on every embedded shell with maximum degree one; got {shell_exterior_degrees!r}, maximum={degree!r}"
            )
        self.status = status_
        self.failed_premise = premise
        self.oriented_cell_count = counts["oriented_cell_count"]
        self.conforming_entity_count = counts["conforming_entity_count"]
        self.boundary_facet_count = counts["boundary_facet_count"]
        self.boundary_pair_count = counts["boundary_pair_count"]
        self.tangent_cone_contact_count = counts["tangent_cone_contact_count"]
        self.shell_count = counts["shell_count"]
        self.shell_exterior_degrees = shell_exterior_degrees
        self.maximum_degree = degree
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mapped-boundary-degree-evidence",
                "status": status_,
                "failed_premise": premise,
                "counts": tuple(counts.items()),
                "shell_exterior_degrees": shell_exterior_degrees,
                "maximum_degree": degree,
            }
        )


@final
class EmbeddingCounts(StrictModule, NonTrainableState):
    """Nonnegative entity and query counts of a global embedding proof."""

    cell_count: int = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    shell_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    ray_test_count: int = eqx.field(static=True)
    subdivision_piece_count: int = eqx.field(static=True)
    periodic_image_count: int = eqx.field(static=True)

    def __init__(
        self,
        cell_count: int,
        boundary_facet_count: int,
        shell_count: int,
        candidate_pair_count: int,
        ray_test_count: int,
        subdivision_piece_count: int,
        periodic_image_count: int = 0,
    ) -> None:
        values = tuple(
            nonnegative_integer(value, f"EmbeddingCounts.{name}")
            for name, value in (
                ("cell_count", cell_count),
                ("boundary_facet_count", boundary_facet_count),
                ("shell_count", shell_count),
                ("candidate_pair_count", candidate_pair_count),
                ("ray_test_count", ray_test_count),
                ("subdivision_piece_count", subdivision_piece_count),
                ("periodic_image_count", periodic_image_count),
            )
        )
        (
            self.cell_count,
            self.boundary_facet_count,
            self.shell_count,
            self.candidate_pair_count,
            self.ray_test_count,
            self.subdivision_piece_count,
            self.periodic_image_count,
        ) = values


@final
class EmbeddingExpressionWork(StrictModule, NonTrainableState):
    """Charged rational-expression work and peak additional storage for a proof phase."""

    work_units: int = eqx.field(static=True)
    peak_bytes: int = eqx.field(static=True)

    def __init__(self, work_units: int = 0, peak_bytes: int = 0) -> None:
        self.work_units = nonnegative_integer(
            work_units, "EmbeddingExpressionWork.work_units"
        )
        self.peak_bytes = nonnegative_integer(
            peak_bytes, "EmbeddingExpressionWork.peak_bytes"
        )


@final
class GlobalEmbeddingCertificate(StrictModule, NonTrainableState):
    """Evidence that cell interiors are pairwise disjoint and contacts are shared.

    ``evaluated_checks`` lists the decided orientation and contact obligations. ``status`` is
    ``certified`` only when every hypothesis holds; findings expose the failing
    or undecided entities. ``boundary_degree`` records the hypotheses of the
    mapped boundary-degree theorem whenever a mapped volume proof attempted it.
    ``restriction_root_embedding`` binds an imported predecessor theorem after
    full root-map correspondence; child-domain and child-contact proofs remain
    independent obligations of this certificate. ``periodic_restriction``
    records the exact parent partition that transfers a complete periodic
    source embedding; every target premise is proved by this certificate.
    ``certificate_id`` binds the proof hypotheses and verdict, excluding the
    source-expression work and storage measurements reported by this invocation.
    """

    binding: MeshCertificateBinding
    validity_certificate_id: str = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    evaluated_checks: tuple[str, ...] = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    shell_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    ray_test_count: int = eqx.field(static=True)
    subdivision_piece_count: int = eqx.field(static=True)
    periodic_image_count: int = eqx.field(static=True)
    source_expression_work_units: int = eqx.field(static=True)
    source_expression_peak_bytes: int = eqx.field(static=True)
    boundary_degree: MappedBoundaryDegreeEvidence | None
    periodic_affine: PeriodicAffineQuotientEvidence | None
    periodic_affine_degree: PeriodicAffineTorusDegreeEvidence | None
    periodic_restriction: PeriodicRestrictionPartitionEvidence | None
    periodic_mapped_restriction: PeriodicMappedRestrictionEvidence | None
    restriction_root_embedding: MappedRestrictionRootEmbeddingEvidence | None
    cad_boundary_closure: CadBoundaryClosureEvidence | None
    certificate_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        validity_certificate_id: str,
        findings: tuple[MeshCertificateFinding, ...],
        evaluated_checks: tuple[str, ...],
        /,
        *,
        counts: EmbeddingCounts,
        expression_work: EmbeddingExpressionWork | None = None,
        boundary_degree: MappedBoundaryDegreeEvidence | None = None,
        periodic_affine: PeriodicAffineQuotientEvidence | None = None,
        periodic_affine_degree: PeriodicAffineTorusDegreeEvidence | None = None,
        periodic_restriction: PeriodicRestrictionPartitionEvidence | None = None,
        periodic_mapped_restriction: PeriodicMappedRestrictionEvidence | None = None,
        restriction_root_embedding: MappedRestrictionRootEmbeddingEvidence | None = None,
        cad_boundary_closure: CadBoundaryClosureEvidence | None = None,
        _periodic_mapped_budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if not all(isinstance(value, MeshCertificateFinding) for value in findings):
            raise TypeError(
                "GlobalEmbeddingCertificate.findings must contain MeshCertificateFinding values; "
                f"got {findings!r}"
            )
        if periodic_affine is not None and (
            periodic_affine.mesh_id,
            periodic_affine.geometry_id,
            periodic_affine.validity_certificate_id,
        ) != (binding.mesh_id, binding.geometry_id, validity_certificate_id):
            raise ValueError(
                "GlobalEmbeddingCertificate.periodic_affine must bind its mesh, geometry, "
                "and local validity certificate; "
                f"got mesh={periodic_affine.mesh_id!r}, "
                f"geometry={periodic_affine.geometry_id!r}, "
                f"validity={periodic_affine.validity_certificate_id!r}"
            )
        degree_checks = {
            "periodic_affine_torus_manifold", "periodic_affine_torus_trace",
            "periodic_affine_torus_orientation", "periodic_affine_torus_degree_one",
        }
        if periodic_affine_degree is not None:
            if (
                (periodic_affine_degree.mesh_id, periodic_affine_degree.geometry_id)
                != (binding.mesh_id, binding.geometry_id)
                or not degree_checks <= set(evaluated_checks)
            ):
                raise ValueError(
                    "GlobalEmbeddingCertificate.periodic_affine_degree must bind its current mesh, geometry and every decided theorem premise"
                )
            PeriodicAffineTorusDegreeEvidence.from_stored_fields(periodic_affine_degree)
        elif not findings and "periodic_affine_torus_degree_one" in evaluated_checks:
            raise ValueError(
                "GlobalEmbeddingCertificate.periodic_affine_degree must retain the successful exact degree evidence"
            )
        restriction_checks = {
            "periodic_authored_group_identity",
            "periodic_mapped_trace_equivariance",
            "periodic_restriction_source_embedding",
            "periodic_restriction_parent_partition",
            "periodic_restriction_boundary_subdivision",
        }
        if periodic_restriction is not None:
            if (
                periodic_restriction.mesh_id,
                periodic_restriction.geometry_id,
                periodic_restriction.validity_certificate_id,
            ) != (
                binding.mesh_id,
                binding.geometry_id,
                validity_certificate_id,
            ) or not restriction_checks <= set(evaluated_checks):
                raise ValueError(
                    "GlobalEmbeddingCertificate.periodic_restriction must bind its current mesh, geometry, validity and every decided theorem premise"
                )
            PeriodicRestrictionPartitionEvidence.from_stored_fields(periodic_restriction)
        elif "periodic_restriction_parent_partition" in evaluated_checks:
            raise ValueError(
                "GlobalEmbeddingCertificate.periodic_restriction must retain the evaluated exact partition evidence"
            )
        mapped_checks = {
            "periodic_authored_group_identity",
            "periodic_mapped_trace_equivariance",
            "periodic_mapped_restriction_source_embedding",
            "periodic_mapped_restriction_parent_partition",
            "periodic_mapped_restriction_common_subcomplex",
            "periodic_mapped_restriction_derived_region_assignment",
        }
        if periodic_mapped_restriction is not None:
            retained = PeriodicMappedRestrictionEvidence.from_stored_fields(
                periodic_mapped_restriction, budget=_periodic_mapped_budget
            )
            if (
                retained.predecessor.target_mesh.mesh_id != binding.mesh_id
                or cell_geometry_id(retained.predecessor.target_geometry) != binding.geometry_id
                or retained.validity.certificate_id != validity_certificate_id
                or not mapped_checks <= set(evaluated_checks)
            ):
                raise ValueError(
                    "GlobalEmbeddingCertificate.periodic_mapped_restriction must bind every current target and theorem premise"
                )
        elif mapped_checks & set(evaluated_checks) - {
            "periodic_authored_group_identity", "periodic_mapped_trace_equivariance"
        }:
            raise ValueError(
                "GlobalEmbeddingCertificate.periodic_mapped_restriction must retain its exact mapped partition"
            )
        has_root_import = "mapped_restriction_root_embedding_reuse" in evaluated_checks
        if has_root_import != (restriction_root_embedding is not None):
            raise ValueError(
                "GlobalEmbeddingCertificate.restriction_root_embedding must accompany its evaluated reuse check; "
                f"got evidence={restriction_root_embedding!r}, check={has_root_import!r}"
            )
        if restriction_root_embedding is not None and (
            restriction_root_embedding.target_binding_id,
            restriction_root_embedding.validity_certificate_id,
        ) != (binding.binding_id, validity_certificate_id):
            raise ValueError(
                "GlobalEmbeddingCertificate.restriction_root_embedding must bind its target and positive validity; "
                f"got binding={restriction_root_embedding.target_binding_id!r}, "
                f"validity={restriction_root_embedding.validity_certificate_id!r}"
            )
        has_cad_import = "mapped_cad_boundary_homotopy_reuse" in evaluated_checks
        if has_cad_import != (cad_boundary_closure is not None):
            raise ValueError(
                "GlobalEmbeddingCertificate.cad_boundary_closure must retain its "
                "evaluated whole-boundary homotopy import"
            )
        if cad_boundary_closure is not None:
            CadBoundaryClosureEvidence.from_stored_fields(cad_boundary_closure)
            source = cad_boundary_closure.binding
            if (
                source.mesh_id, source.topology_id, source.geometry_layout_id,
                source.geometry_id, source.coordinate_scope, source.limits_id,
            ) != (
                binding.mesh_id, binding.topology_id, binding.geometry_layout_id,
                binding.geometry_id, binding.coordinate_scope, binding.limits_id,
            ):
                raise ValueError(
                    "GlobalEmbeddingCertificate.cad_boundary_closure must bind "
                    "the current target mesh, geometry, topology and proof limits"
                )
        work = EmbeddingExpressionWork() if expression_work is None else expression_work
        cell_count, boundary_facet_count, shell_count = (
            counts.cell_count,
            counts.boundary_facet_count,
            counts.shell_count,
        )
        candidate_pair_count, ray_test_count = (
            counts.candidate_pair_count,
            counts.ray_test_count,
        )
        subdivision_piece_count, periodic_image_count = (
            counts.subdivision_piece_count,
            counts.periodic_image_count,
        )
        source_expression_work_units, source_expression_peak_bytes = (
            work.work_units,
            work.peak_bytes,
        )
        self.binding = binding
        self.validity_certificate_id = str(validity_certificate_id)
        self.status = certificate_status(findings)
        self.findings = tuple(findings)
        self.evaluated_checks = tuple(evaluated_checks)
        self.cell_count = cell_count
        self.boundary_facet_count = boundary_facet_count
        self.shell_count = shell_count
        self.candidate_pair_count = candidate_pair_count
        self.ray_test_count = ray_test_count
        self.subdivision_piece_count = subdivision_piece_count
        self.periodic_image_count = periodic_image_count
        self.source_expression_work_units = source_expression_work_units
        self.source_expression_peak_bytes = source_expression_peak_bytes
        self.boundary_degree = boundary_degree
        self.periodic_affine = periodic_affine
        self.periodic_affine_degree = periodic_affine_degree
        self.periodic_restriction = periodic_restriction
        self.periodic_mapped_restriction = periodic_mapped_restriction
        self.restriction_root_embedding = restriction_root_embedding
        self.cad_boundary_closure = cad_boundary_closure
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "global-embedding-certificate",
                "binding": binding.binding_id,
                "validity": self.validity_certificate_id,
                "status": self.status,
                "findings": [value.finding_id for value in findings],
                "checks": self.evaluated_checks,
                "periodic_affine": (
                    None if periodic_affine is None else periodic_affine.evidence_id
                ),
                "periodic_affine_degree": (
                    None if periodic_affine_degree is None else periodic_affine_degree.evidence_id
                ),
                **(
                    {}
                    if periodic_restriction is None
                    else {"periodic_restriction": periodic_restriction.evidence_id}
                ),
                **(
                    {}
                    if periodic_mapped_restriction is None
                    else {"periodic_mapped_restriction": periodic_mapped_restriction.evidence_id}
                ),
                "counts": (
                    cell_count,
                    boundary_facet_count,
                    shell_count,
                    candidate_pair_count,
                    ray_test_count,
                    subdivision_piece_count,
                    periodic_image_count,
                ),
                **(
                    {}
                    if boundary_degree is None
                    else {"boundary_degree": boundary_degree.evidence_id}
                ),
                **(
                    {
                        "mapped_affine_polytope_affinity": canonical_fingerprint(
                            {
                                "kind": "full-source-affine-polytope-contact",
                                "binding": binding.binding_id,
                                "validity": self.validity_certificate_id,
                            }
                        )
                    }
                    if "mapped_affine_polytope_contact" in self.evaluated_checks
                    else {}
                ),
                **(
                    {}
                    if restriction_root_embedding is None
                    else {
                        "restriction_root_embedding": restriction_root_embedding.evidence_id
                    }
                ),
                **(
                    {}
                    if cad_boundary_closure is None
                    else {"cad_boundary_closure": cad_boundary_closure.evidence_id}
                ),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[GlobalEmbeddingCertificate],
        value: GlobalEmbeddingCertificate,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> GlobalEmbeddingCertificate:
        """Reconstruct a certificate through the validated evidence phases.

        Stored proof fields determine the regenerated certificate fingerprint;
        invocation work remains outside that scientific identity.
        """
        periodic_affine = (
            None
            if value.periodic_affine is None
            else PeriodicAffineQuotientEvidence.from_stored_fields(value.periodic_affine)
        )
        periodic_affine_degree = (
            None
            if value.periodic_affine_degree is None
            else PeriodicAffineTorusDegreeEvidence.from_stored_fields(
                value.periodic_affine_degree
            )
        )
        periodic_restriction = (
            None
            if value.periodic_restriction is None
            else PeriodicRestrictionPartitionEvidence.from_stored_fields(
                value.periodic_restriction
            )
        )
        restriction_root_embedding = (
            None
            if value.restriction_root_embedding is None
            else MappedRestrictionRootEmbeddingEvidence.from_stored_fields(
                value.restriction_root_embedding
            )
        )
        rebuilt = cls(
            value.binding,
            value.validity_certificate_id,
            value.findings,
            value.evaluated_checks,
            counts=EmbeddingCounts(
                value.cell_count,
                value.boundary_facet_count,
                value.shell_count,
                value.candidate_pair_count,
                value.ray_test_count,
                value.subdivision_piece_count,
                value.periodic_image_count,
            ),
            expression_work=EmbeddingExpressionWork(
                value.source_expression_work_units,
                value.source_expression_peak_bytes,
            ),
            boundary_degree=value.boundary_degree,
            periodic_affine=periodic_affine,
            periodic_affine_degree=periodic_affine_degree,
            periodic_restriction=periodic_restriction,
            periodic_mapped_restriction=value.periodic_mapped_restriction,
            restriction_root_embedding=restriction_root_embedding,
            cad_boundary_closure=value.cad_boundary_closure,
            _periodic_mapped_budget=budget,
        )
        if (
            rebuilt.certificate_id != value.certificate_id
            or rebuilt.status != value.status
        ):
            raise ValueError(
                "GlobalEmbeddingCertificate.stored_fields must match its regenerated proof identity and status; "
                f"got certificate={value.certificate_id!r}, status={value.status!r}"
            )
        return rebuilt


def _signs(result: HostPredicateResult, /) -> tuple[np.ndarray, np.ndarray]:
    signs = result.signs.astype(np.int16, copy=False)
    certain = result.certain
    return signs, certain


def _exact_step(
    exact: bool, work: int, storage: int = 0, /, *, budget: CoordinateEnclosureBudget
) -> AbstractContextManager[None]:
    """Charge a rational-source step to the request ledger; binary64 steps are unmetered."""
    return exact_charge(work, storage, budget=budget) if exact else nullcontext()


@checked
def mesh_orient2d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient2d_host(a, b, c, mode=mode, budget=budget))


@checked
def mesh_orient3d(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    return _signs(orient3d_host(a, b, c, d, mode=mode, budget=budget))


def degenerate_simplices(
    points: np.ndarray,
    rows: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Exactly degenerate segments (2-D) or collinear triangles (3-D)."""

    corners = points[rows]
    if rows.shape[1] == 2:
        return np.all(corners[:, 0] == corners[:, 1], axis=-1)
    degenerate = np.ones((rows.shape[0],), dtype=np.bool_)
    for axes in ((0, 1), (0, 2), (1, 2)):
        projected = corners[:, :, axes]
        sign, _ = mesh_orient2d(
            projected[:, 0], projected[:, 1], projected[:, 2], budget=budget
        )
        degenerate &= sign == 0
    return degenerate


@checked
def cell_global_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )


@checked
def coordinate_scope(
    mesh: CellMesh, geometry: CellGeometrySpec, /, *, budget: CoordinateEnclosureBudget
) -> CoordinateScope:
    """Prepare coordinate scope inside the supplied request ledger."""
    with budget.activate():
        return _coordinate_scope_active(mesh, geometry, budget=budget)


@final
class _CoordinateScopeInputs(StrictModule):
    """Resolved host carriers and rational source controls for one scope proof."""

    elements: tuple[CellGeometryElement, ...]
    routes: tuple[np.ndarray, ...]
    values: np.ndarray
    points: np.ndarray
    source_values: CoordinateSourceBank
    vertices: tuple[np.ndarray, ...]
    vertex_valid: tuple[np.ndarray, ...]
    corner_images: tuple[
        tuple[coordinate_enclosure_owner.ExactScalar, ...] | None, ...
    ] | None = None


@final
class _CoordinateCellMap(StrictModule):
    """Rational corner shortcuts or source polynomials for one degree-one cell."""

    simplex_images: CoordinateSourceBank | None
    prepared_images: CoordinateSourceBank | None
    polynomials: tuple[Polynomial, ...]


@final
class _CornerBindings:
    """Rational target-vertex images bound so far by one coordinate-scope proof.

    A cardinal corner routed to its own source row binds exactly that retained
    row, so it is recorded in ``identity`` without forming a dictionary entry or
    a later bank copy. Every other binding keeps its image in ``images``; the two
    sets are disjoint and together are the scope's target corners.
    """

    __slots__ = ("identity", "images", "source_values")

    def __init__(
        self,
        source_values: CoordinateSourceBank,
        count: int,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> None:
        budget.reserve(0, 256 + count)
        self.source_values = source_values
        self.identity = np.zeros((count,), dtype=np.bool_)
        self.images: dict[int, tuple[coordinate_enclosure_owner.ExactScalar, ...]] = {}

    def get(
        self, vertex: int, /
    ) -> tuple[coordinate_enclosure_owner.ExactScalar, ...] | None:
        if self.identity[vertex]:
            return self.source_values[vertex]
        return self.images.get(vertex)

    def __len__(self) -> int:
        return int(np.count_nonzero(self.identity)) + len(self.images)



def _prepare_coordinate_scope_inputs(
    mesh: CellMesh, geometry: CellGeometrySpec, /, *, budget: CoordinateEnclosureBudget
) -> _CoordinateScopeInputs:
    if geometry.exact_source is None:
        elements, routes, coordinates = geometry.resolve(mesh, budget=budget)
    else:
        elements, routes, coordinates = geometry._resolve(
            mesh, exact_source_prepared=True, budget=budget
        )
    source_values = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    # One noniterative transfer exposes every device carrier used by the host proof.
    values, points, host_routes, vertices, valid = jax.device_get(
        (
            coordinates,
            mesh.coordinates,
            routes,
            tuple(block.vertices for block in mesh.blocks),
            tuple(block.vertex_valid for block in mesh.blocks),
        )
    )
    images = None
    if (
        coordinate_identity_owner.coordinate_scope_key(mesh, geometry)
        in budget.coordinate_vertex_image_cache
        or any(
            not isinstance(element, reference_types_owner.FiniteElementSpec)
            for element in elements
        )
    ) and all(
        block.cell_kind in ("interval", "triangle", "tetrahedron")
        and coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
        for block, element in zip(mesh.blocks, elements, strict=True)
    ):
        images = coordinate_enclosure_owner.prepared_coordinate_vertex_images(
            mesh,
            geometry,
            mismatch="coordinate_scope.shared_vertex must have one rational image; got conflicting affine chart images",
            budget=budget,
        )
    return _CoordinateScopeInputs(
        tuple(elements),
        tuple(np.asarray(route, dtype=np.int64) for route in host_routes),
        np.asarray(values, dtype=np.float64),
        np.asarray(points, dtype=np.float64),
        source_values,
        tuple(np.asarray(row, dtype=np.int64) for row in vertices),
        tuple(np.asarray(row, dtype=np.bool_) for row in valid),
        images,
    )


def _prepare_coordinate_cell_map(
    element: CellGeometryElement,
    kind: str,
    local: CoordinateSourceBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CoordinateCellMap | None:
    simplex_images = (
        coordinate_enclosure_owner.coordinate_corner_images(element, local, budget=budget)
        if kind in ("interval", "triangle", "tetrahedron")
        else None
    )
    prepared = None
    polynomials: tuple[Polynomial, ...] = ()
    if simplex_images is None:
        prepared = coordinate_enclosure_owner.multi_affine_coordinates(
            element, local, budget=budget
        )
        if prepared is None:
            coordinate_values = coordinate_enclosure_owner.coordinate_polynomials(
                element, local, budget=budget
            )
            if coordinate_values is None:
                return None
            polynomials = coordinate_values
        else:
            polynomials = prepared[0]
    return _CoordinateCellMap(
        simplex_images, None if prepared is None else prepared[1], polynomials
    )


def _bind_coordinate_cell_corners(
    prepared: _CoordinateCellMap,
    kind: str,
    vertices: np.ndarray,
    reference: CoordinateSourceBank,
    inputs: _CoordinateScopeInputs,
    target_corners: _CornerBindings,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    nonbinary_corner = False
    for corner, (vertex, point) in enumerate(zip(vertices, reference, strict=True)):
        parameter = (
            (Fraction(1, 2), Fraction(1, 2), Fraction(1))
            if kind == "pyramid" and point[2] == 1
            else point
        )
        if prepared.simplex_images is not None:
            image = prepared.simplex_images[corner]
        elif prepared.prepared_images is not None:
            image = prepared.prepared_images[corner]
        else:
            image = tuple(
                coordinate_enclosure_owner.evaluate(value, parameter, budget=budget)
                for value in prepared.polynomials
            )
        budget.reserve(len(image))
        if not np.array_equal(
            np.asarray(tuple(float(value) for value in image), dtype=np.float64).view(
                np.uint64
            ),
            inputs.points[vertex].view(np.uint64),
        ):
            match geometry.exact_source:
                case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
                    restriction = geometry.restriction_source
                    if (
                        restriction is None
                        or restriction.carrier_publication != "published_affine_roots"
                    ) and image != inputs.source_values[int(vertex)]:
                        raise ValueError(
                            "coordinate_scope.PLC_image must agree with its certified source point; "
                            f"got vertex {vertex}"
                        )
                    # resolve checked the separate ideal affine-root publication
                    # law; source predicates still consume their rational source image.
                case _:
                    restriction = geometry.restriction_source
                    if (
                        restriction is None
                        or restriction.carrier_publication != "published_affine_roots"
                    ):
                        raise ValueError(
                            "coordinate_scope.carrier must be the correctly rounded source image; "
                            f"got vertex {vertex}"
                        )
        prior = target_corners.get(int(vertex))
        if prior is not None and prior != image:
            raise ValueError(
                "coordinate_scope.shared_vertex must have one rational image; "
                f"got conflicting images at vertex {vertex}"
            )
        if prior is None:
            budget.reserve(1, 256 + 128 * len(image))
            target_corners.images[int(vertex)] = image
        nonbinary_corner |= any(Fraction(float(value)) != value for value in image)
    return nonbinary_corner


def _coordinate_cell_is_mapped(
    prepared: _CoordinateCellMap,
    kind: str,
    nonbinary_corner: bool,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    element: CellGeometryElement | None = None,
) -> bool:
    if prepared.simplex_images is not None:
        if nonbinary_corner:
            match geometry.exact_source:
                case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
                    restriction = geometry.restriction_source
                    return not (
                        isinstance(element, reference_types_owner.FiniteElementSpec)
                        or (
                            restriction is not None
                            and restriction.carrier_publication
                            == "published_affine_roots"
                        )
                    )
                case _:
                    return True
        return False
    physical = (
        coordinate_enclosure_owner.restrict_coordinates(
            prepared.polynomials,
            "pyramid",
            np.zeros((3,), dtype=np.float64),
            np.eye(3, dtype=np.float64),
            budget=budget,
        )
        if kind == "pyramid"
        else prepared.polynomials
    )
    if physical is not None:
        budget.reserve(sum(len(polynomial) for polynomial in physical))
    return (
        nonbinary_corner
        or physical is None
        or any(sum(index) > 1 for polynomial in physical for index in polynomial)
    )


def _bind_cardinal_simplex_block(
    element: reference_types_owner.FiniteElementSpec,
    kind: str,
    route: np.ndarray,
    vertices: np.ndarray,
    inputs: _CoordinateScopeInputs,
    target_corners: _CornerBindings,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray | None:
    """Bind a degree-one simplex block whose corners are cardinal coefficient rows.

    Each corner image is the source-bank row its unit corner weight selects, so
    every bank row whose rounding can decide a refusal or classification is
    rounded and classified once, and each such corner is compared with its
    carrier vertex bits. A PLC corner routed to its own row is its certified
    source point, so its rounding decides nothing: PLC refusals require another
    row and PLC cells are never mapped. Source-image work is charged once per
    unique row actually classified; newly bound vertices retain their creation
    charge, and a vertex bound to its own row forms no image entry.
    Mismatch/shared-vertex refusals and the affine/mapped classification match
    the per-cell rule. Returns mapped cells, or ``None`` where the cardinal rule does not
    describe the block.
    """
    count = route.shape[0]
    if (
        kind not in ("interval", "triangle", "tetrahedron")
        or element.cell_kind != kind
        or count == 0
        or route.ndim != 2
        or route.shape[1] != element.local_dof_count
        or vertices.shape != (count, element.local_dof_count)
    ):
        return None
    nodes = cell_geometry_owner._cardinal_corner_nodes(element, budget=budget)
    ambient = inputs.points.shape[1]
    source_values = inputs.source_values
    if nodes is None or len(nodes) != vertices.shape[1]:
        return None
    corner_rows = route[:, list(nodes)].reshape(-1)
    corner_vertices = vertices.reshape(-1)
    used, images = np.unique(corner_rows, return_inverse=True)
    if np.any(used < 0) or np.any(used >= len(source_values)) or any(
        len(source_values[row]) != ambient for row in used.tolist()
    ):
        return None
    restriction = geometry.restriction_source
    published = (
        restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    )
    plc = isinstance(
        geometry.exact_source, exact_plc_geometry_owner.ExactPlcCellGeometrySource
    )
    identity_corners = corner_rows == corner_vertices
    with budget.temporary_scope():
        if not plc:
            classified = np.ones((used.size,), dtype=np.bool_)
        else:
            classified = np.zeros((used.size,), dtype=np.bool_)
            if not published:
                classified[images[~identity_corners]] = True
        rows = np.flatnonzero(classified)
        budget.reserve(rows.size * ambient)
        rounded = np.zeros((used.size, ambient), dtype=np.float64)
        nonbinary = np.zeros((used.size,), dtype=np.bool_)
        for position in rows.tolist():
            image = source_values[int(used[position])]
            rounded[position] = tuple(float(value) for value in image)
            nonbinary[position] = any(Fraction(float(value)) != value for value in image)
        compared = classified[images]
        mismatched = np.zeros((corner_rows.size,), dtype=np.bool_)
        mismatched[compared] = ~np.all(
            rounded.view(np.uint64)[images[compared]]
            == inputs.points[corner_vertices[compared]].view(np.uint64),
            axis=1,
        )
        _, first, local_vertices = np.unique(
            corner_vertices, return_index=True, return_inverse=True
        )
        first_rows = corner_rows[first][local_vertices]
        prebound = target_corners.identity[corner_vertices]
        if target_corners.images:
            prebound |= np.fromiter(
                (
                    int(vertex) in target_corners.images
                    for vertex in corner_vertices.tolist()
                ),
                dtype=np.bool_,
                count=corner_vertices.size,
            )
        for position in np.flatnonzero(
            mismatched | prebound | (corner_rows != first_rows)
        ).tolist():
            vertex = int(corner_vertices[position])
            image = source_values[int(corner_rows[position])]
            if mismatched[position] and not published:
                if not plc:
                    raise ValueError(
                        "coordinate_scope.carrier must be the correctly rounded source image; "
                        f"got vertex {vertex}"
                    )
                if image != source_values[vertex]:
                    raise ValueError(
                        "coordinate_scope.PLC_image must agree with its certified source point; "
                        f"got vertex {vertex}"
                    )
            prior = (
                target_corners.get(vertex)
                if prebound[position]
                else source_values[int(first_rows[position])]
            )
            if prior != image:
                raise ValueError(
                    "coordinate_scope.shared_vertex must have one rational image; "
                    f"got conflicting images at vertex {vertex}"
                )
        # The per-cell rule admits each newly bound vertex in its cell's
        # transient scope; the largest cell bound is that rule's peak.
        bound = ~prebound & (np.arange(corner_vertices.size) == first[local_vertices])
        per_cell = np.count_nonzero(bound.reshape(count, -1), axis=1)
        foreign = bound & ~identity_corners
        budget.reserve(
            int(np.count_nonzero(foreign)),
            int(np.max(per_cell, initial=0)) * (256 + 128 * ambient),
        )
        target_corners.identity[corner_vertices[bound & identity_corners]] = True
        for vertex, row in zip(
            corner_vertices[foreign].tolist(), corner_rows[foreign].tolist(), strict=True
        ):
            target_corners.images[vertex] = source_values[row]
        mapped = np.any(nonbinary[images].reshape(count, -1), axis=1)
        if plc:
            mapped[:] = False
    return np.flatnonzero(mapped).astype(np.int64)


def _bind_affine_simplex_block(
    element: CellGeometryElement,
    kind: str,
    route: np.ndarray,
    vertices: np.ndarray,
    inputs: _CoordinateScopeInputs,
    target_corners: _CornerBindings,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray | None:
    """Classify a proven affine simplex chart on its shared exact vertex images.

    The corner evaluator checks every routed corner, including distinct
    operations naming a shared vertex. Rounding and nonbinary classification
    then depend only on that vertex image, not on its incident-cell count.
    """
    if kind not in ("interval", "triangle", "tetrahedron") or not (
        coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
    ):
        return None
    if vertices.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    images = inputs.corner_images
    if images is None:
        images = coordinate_enclosure_owner.coordinate_vertex_images(
            ((element, route, vertices),),
            inputs.source_values,
            inputs.points.shape[0],
            mismatch="coordinate_scope.shared_vertex must have one rational image; got conflicting affine chart images",
            budget=budget,
        )
    if images is None:
        return None
    used = np.unique(vertices)
    ambient = inputs.points.shape[1]
    budget.reserve(
        used.size * ambient,
        1024 + used.size * (32 + 8 * ambient) + inputs.points.shape[0],
    )
    bound = []
    for vertex in used.tolist():
        image = images[vertex]
        if image is None:
            raise ValueError(
                f"coordinate_scope.exact_charts must bind each affine corner; got vertex {vertex}"
            )
        bound.append(image)
    rounded = np.asarray(
        [tuple(float(value) for value in image) for image in bound],
        dtype=np.float64,
    )
    mismatched = ~np.all(
        rounded.view(np.uint64) == inputs.points[used].view(np.uint64), axis=1
    )
    restriction = geometry.restriction_source
    published = (
        restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    )
    plc = isinstance(
        geometry.exact_source, exact_plc_geometry_owner.ExactPlcCellGeometrySource
    )
    nonbinary = np.zeros((inputs.points.shape[0],), dtype=np.bool_)
    priors = [target_corners.get(vertex) for vertex in used.tolist()]
    new = sum(prior is None for prior in priors)
    budget.reserve(new, new * (256 + 128 * ambient))
    for position, (vertex, image, prior) in enumerate(
        zip(used.tolist(), bound, priors, strict=True)
    ):
        if mismatched[position] and not published:
            if not plc:
                raise ValueError(
                    "coordinate_scope.carrier must be the correctly rounded source image; "
                    f"got vertex {vertex}"
                )
            if image != inputs.source_values[vertex]:
                raise ValueError(
                    "coordinate_scope.PLC_image must agree with its certified source point; "
                    f"got vertex {vertex}"
                )
        if prior is not None and prior != image:
            raise ValueError(
                "coordinate_scope.shared_vertex must have one rational image; "
                f"got conflicting images at vertex {vertex}"
            )
        if prior is None:
            target_corners.images[vertex] = image
        nonbinary[vertex] = any(
            Fraction(value) != exact
            for value, exact in zip(rounded[position], image, strict=True)
        )
    return (
        np.empty((0,), dtype=np.int64)
        if plc and (
            published or isinstance(element, reference_types_owner.FiniteElementSpec)
        )
        else np.flatnonzero(np.any(nonbinary[vertices], axis=1)).astype(np.int64)
    )


def _classify_coordinate_cells(
    element: CellGeometryElement,
    kind: str,
    route: np.ndarray,
    vertices: np.ndarray,
    cursor: int,
    inputs: _CoordinateScopeInputs,
    target_corners: _CornerBindings,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> list[np.ndarray]:
    """Bind each cell's exact corners and return the cells that stay mapped."""
    with budget.temporary_scope():
        affine = _bind_affine_simplex_block(
            element,
            kind,
            route,
            vertices,
            inputs,
            target_corners,
            geometry,
            budget=budget,
        )
    if affine is not None:
        return [cursor + affine] if affine.size else []
    mapped: list[np.ndarray] = []
    reference = tuple(
        rational_vector(point) for point in reference_cell_topology(kind).vertices
    )
    for cell, row in enumerate(route):
        local = tuple(inputs.source_values[index] for index in row)
        # Only classification workspace is temporary; corner bindings remain live.
        with budget.temporary_scope():
            prepared = _prepare_coordinate_cell_map(element, kind, local, budget=budget)
            if prepared is None:
                mapped.append(np.asarray((cursor + cell,), dtype=np.int64))
                continue
            nonbinary = _bind_coordinate_cell_corners(
                prepared,
                kind,
                vertices[cell],
                reference,
                inputs,
                target_corners,
                geometry,
                budget=budget,
            )
            if _coordinate_cell_is_mapped(
                prepared, kind, nonbinary, geometry, budget=budget, element=element
            ):
                mapped.append(np.asarray((cursor + cell,), dtype=np.int64))
    return mapped


def _coordinate_scope_active(
    mesh: CellMesh, geometry: CellGeometrySpec, /, *, budget: CoordinateEnclosureBudget
) -> CoordinateScope:
    """Classify affine cells after rational corner and shared-carrier validation."""
    scope_key = coordinate_identity_owner.coordinate_scope_key(mesh, geometry)
    cached = budget.scope_cache.get(scope_key)
    if cached is not None:
        budget.reserve(0)
        has_mapped, cells, source_values = cached
        return CoordinateScope("mapped" if has_mapped else "affine", cells, source_values)
    inputs = _prepare_coordinate_scope_inputs(mesh, geometry, budget=budget)
    mapped: list[np.ndarray] = []
    target_corners = _CornerBindings(
        inputs.source_values, inputs.points.shape[0], budget=budget
    )
    restriction = geometry.restriction_source
    if (
        inputs.corner_images is not None
        and isinstance(
            geometry.exact_source, exact_plc_geometry_owner.ExactPlcCellGeometrySource
        )
        and restriction is not None
        and restriction.carrier_publication == "published_affine_roots"
    ):
        # resolve admitted the published carrier law; the all-affine image
        # bank proved every routed corner and every shared-vertex equality.
        # PLC simplex images are affine even when their exact corners are not
        # binary64. Neither rounding nor nonbinary tests can change this rule.
        budget.reserve(0, 256 + 80 * len(inputs.corner_images))
        foreign = []
        for vertex, image in enumerate(inputs.corner_images):
            if image is None:
                continue
            if (
                vertex < len(inputs.source_values)
                and image is inputs.source_values[vertex]
            ):
                target_corners.identity[vertex] = True
            else:
                foreign.append((vertex, image))
        budget.reserve(
            len(foreign), sum(256 + 128 * len(image) for _, image in foreign)
        )
        target_corners.images.update(foreign)
        return _commit_coordinate_scope(
            mesh, geometry, scope_key, inputs, mapped, target_corners, budget=budget
        )
    cursor = 0
    for block, element, route, vertices, valid in zip(
        mesh.blocks,
        inputs.elements,
        inputs.routes,
        inputs.vertices,
        inputs.vertex_valid,
        strict=True,
    ):
        match element:
            case CellVertexGeometryElement():
                if route.shape[1] != vertices.shape[1]:
                    raise ValueError(
                        "coordinate_scope.routes must match the cell corner layout; "
                        f"got {route.shape[1]} routes for {vertices.shape[1]} corners"
                    )
                if not np.array_equal(
                    inputs.values[route][valid], inputs.points[vertices][valid]
                ):
                    raise ValueError(
                        "coordinate_scope.vertex_geometry must equal the mesh coordinates; "
                        f"got differing corner values in block {block.name!r}"
                    )
            case reference_types_owner.FiniteElementSpec() if element.degree == 1:
                cardinal = _bind_cardinal_simplex_block(
                    element,
                    block.cell_kind,
                    route,
                    vertices,
                    inputs,
                    target_corners,
                    geometry,
                    budget=budget,
                )
                if cardinal is not None:
                    if cardinal.size:
                        mapped.append(cursor + cardinal)
                    cursor += block.cell_count
                    continue
                mapped.extend(
                    _classify_coordinate_cells(
                        element,
                        block.cell_kind,
                        route,
                        vertices,
                        cursor,
                        inputs,
                        target_corners,
                        geometry,
                        budget=budget,
                    )
                )
            case (
                cell_geometry_owner.BarycentricCellGeometryElement()
                | cell_geometry_owner.PolynomialComposedCellGeometryElement()
            ) if block.cell_kind in (
                "interval",
                "triangle",
                "tetrahedron",
            ) and coordinate_enclosure_owner.affine_simplex_source(
                element, budget=budget
            ):
                # A proven affine chart chain is classified by its exact corner
                # images under the same rule as a degree-one Lagrange block.
                mapped.extend(
                    _classify_coordinate_cells(
                        element,
                        block.cell_kind,
                        route,
                        vertices,
                        cursor,
                        inputs,
                        target_corners,
                        geometry,
                        budget=budget,
                    )
                )
            case _:
                mapped.append(
                    np.arange(cursor, cursor + block.cell_count, dtype=np.int64)
                )
        cursor += block.cell_count
    return _commit_coordinate_scope(
        mesh, geometry, scope_key, inputs, mapped, target_corners, budget=budget
    )


def _commit_coordinate_scope(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    scope_key: tuple[str, str, str],
    inputs: _CoordinateScopeInputs,
    mapped: list[np.ndarray],
    target_corners: _CornerBindings,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateScope:
    budget.reserve(0, 128 + 8 * sum(value.size for value in mapped))
    cells = np.concatenate(mapped) if mapped else np.empty((0,), dtype=np.int64)
    source_values = inputs.source_values
    if not cells.size and target_corners and geometry.exact_source is not None:
        if len(target_corners) != inputs.points.shape[0]:
            raise ValueError(
                "coordinate_scope.exact_charts must bind every target vertex; "
                f"got {len(target_corners)} bindings for {inputs.points.shape[0]} vertices"
            )
        identity = target_corners.identity
        if not (
            len(inputs.source_values) == identity.size and bool(np.all(identity))
        ):
            if inputs.corner_images is not None:
                # This mesh-bound bank already checked every routed corner;
                # the complete target binding above proves none is unbound.
                # Preserve that exact immutable bank, not another per-vertex
                # packing/profile of the identical source images.
                source_values = cast(CoordinateSourceBank, inputs.corner_images)
            else:
                images = target_corners.images
                source_values = tuple(
                    inputs.source_values[index] if identity[index] else images[index]
                    for index in range(inputs.points.shape[0])
                )
                budget.reserve(inputs.points.size, 128 + 8 * inputs.points.size)
                budget.retain_basis(source_values)
        # Otherwise every target vertex is bound to its own source row, so the
        # target bank is the request-retained source bank itself; no copy forms.
    cells.flags.writeable = False
    prepared_scope = (bool(cells.size), cells, source_values)
    budget.retain_basis((scope_key, prepared_scope))
    budget.scope_cache[scope_key] = prepared_scope
    return CoordinateScope("mapped" if cells.size else "affine", cells, source_values)


def _validity_findings(
    state: EmbeddingState, validity: CellValidityCertificate, cell_ids: np.ndarray, /
) -> None:
    state.checks.append("cell_validity")
    state.closed_cell_orientation = validity.all_certified
    status = np.asarray(validity.status)
    invalid = status == CellValidityStatus.INVALID
    unresolved = status == CellValidityStatus.UNRESOLVED
    if np.any(invalid):
        state.add("invalid_cell", "violated", "cell", cell_ids[invalid])
    if np.any(unresolved):
        state.add("unresolved_cell_validity", "unresolved", "cell", cell_ids[unresolved])


@checked
def pairing_findings(
    state: EmbeddingState, mesh: CellMesh, facets: Facets, junctions: np.ndarray, /
) -> None:
    state.checks.append("facet_pairing")
    order = facets.order
    same = facets.group[order[1:]] == facets.group[order[:-1]]
    first = order[:-1][same]
    second = order[1:][same]
    # Declared curve-network junctions are the only vertices where more than two
    # intervals may meet; valence-two junctions keep the orientation check.
    declared = (
        np.isin(
            np.asarray(mesh.vertex_global_ids, dtype=np.int64)[facets.rows[:, 0]],
            junctions,
        )
        if junctions.size
        else np.zeros(facets.rows.shape[0], dtype=np.bool_)
    )
    nonmanifold = (facets.counts[facets.group] > 2) & ~declared
    twice = facets.counts[facets.group[first]] == 2
    first = first[twice]
    second = second[twice]
    dimension = mesh.topological_dimension
    if dimension == 1:
        consistent = facets.signs[first] != facets.signs[second]
    elif dimension == 2:
        consistent = (facets.rows[first, 0] == facets.rows[second, 1]) & (
            facets.rows[first, 1] == facets.rows[second, 0]
        )
    else:
        rows = facets.rows
        lengths = np.sum(rows >= 0, axis=1)
        position = np.argmin(np.where(rows >= 0, rows, np.iinfo(np.int64).max), axis=1)
        index = np.arange(rows.shape[0])
        successor = rows[index, (position + 1) % lengths]
        predecessor = rows[index, (position - 1) % lengths]
        consistent = successor[first] == predecessor[second]
    if np.any(nonmanifold):
        state.add(
            "nonmanifold_facet",
            "violated",
            "facet",
            facet_entity_ids(mesh, facets.rows[nonmanifold]),
        )
    if not np.all(consistent):
        state.add(
            "inconsistent_facet_orientation",
            "violated",
            "facet",
            facet_entity_ids(mesh, facets.rows[first[~consistent]]),
        )


def _shared_segment_separation(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Prove two segments meet only at their declared common endpoint.

    Along each nonconstant coordinate, an open segment stays strictly on its
    endpoint's side of the common point. Opposite sides, or one segment lying
    in a coordinate plane that the other leaves, therefore exclude every
    nonshared contact. Source comparisons are exact, including rational banks;
    no subtraction, rounded construction or tolerance is involved.
    """
    count = first.shape[0]
    with budget.temporary_scope():
        budget.reserve(0, 1024 + count * (64 + 48 * points.shape[1]))
        sharing = first[:, :, None] == second[:, None, :]
        rows = np.flatnonzero(np.sum(sharing, axis=(1, 2)) == 1)
        result = np.zeros(count, dtype=np.bool_)
        if not rows.size:
            return result
        left = np.argmax(np.any(sharing[rows], axis=2), axis=1)
        right = np.argmax(np.any(sharing[rows], axis=1), axis=1)
        origin = points[first[rows, left]]
        a = points[first[rows, 1 - left]]
        b = points[second[rows, 1 - right]]
        opposed = ((a < origin) & (b > origin)) | ((a > origin) & (b < origin))
        different_support = (a == origin) != (b == origin)
        result[rows] = np.any(opposed | different_support, axis=1)
        return result


def _segment_contacts(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide planar or spatial segment contacts with exact predicates beyond shared mesh vertices."""

    if points.shape[1] == 3:
        sign, certain = mesh_orient3d(
            points[first[:, 0]],
            points[first[:, 1]],
            points[second[:, 0]],
            points[second[:, 1]],
            budget=budget,
        )
        hit = np.zeros(first.shape[0], dtype=np.bool_)
        planar = np.flatnonzero(certain & (sign == 0))
        if planar.size:
            ledger = budget
            ledger.reserve(
                certain.size + planar.size, certain.nbytes + sys.getsizeof(certain)
            )
            # Native/JAX predicate results are immutable borrowed evidence. Only
            # this derived aggregate mask is owned and changed by the proof.
            certain = certain.copy()
            projected_hit = np.ones(planar.size, dtype=np.bool_)
            projected_certain = np.ones(planar.size, dtype=np.bool_)
            # A coplanar pair has an injective coordinate-plane projection on
            # its affine hull (also for collinear pairs). Thus contact beyond
            # the shared vertices must hold in all three projections. Skew
            # pairs are already disjoint by the four-point orientation sign.
            # These slices are views, not three copies of the vertex bank.
            for projection in (points[:, :2], points[:, 1:], points[:, ::2]):
                contact, resolved = _segment_contacts(
                    projection, first[planar], second[planar], budget=budget
                )
                # A projection constant on both segments cannot distinguish
                # overlap from their shared endpoint; it imposes no restriction
                # when both constant images are the same rational point.
                collapsed = (
                    np.all(
                        projection[first[planar, 0]] == projection[first[planar, 1]],
                        axis=1,
                    )
                    & np.all(
                        projection[second[planar, 0]] == projection[second[planar, 1]],
                        axis=1,
                    )
                    & np.all(
                        projection[first[planar, 0]] == projection[second[planar, 0]],
                        axis=1,
                    )
                )
                contact |= collapsed
                resolved |= collapsed
                projected_hit &= contact
                projected_certain &= resolved
            hit[planar] = projected_hit
            certain[planar] &= projected_certain
        return hit & certain, certain

    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    status = np.asarray(
        segment_intersections_2d(
            points[first[:, 0]],
            points[first[:, 1]],
            points[second[:, 0]],
            points[second[:, 1]],
            mode=mode,
        ).status
    )
    shared = np.sum(first[:, :, None] == second[:, None, :], axis=(1, 2))
    certain = status != SegmentIntersectionStatus.UNCERTAIN
    hit = np.where(
        shared == 0,
        status != SegmentIntersectionStatus.DISJOINT,
        np.where(
            shared == 1, status == SegmentIntersectionStatus.COLLINEAR_OVERLAP, True
        ),
    )
    return hit & certain, certain


_EXACT_CONTACT_BATCH = 4096


@contextmanager
@checked
def _exact_contact_boxes(
    points: np.ndarray,
    rows: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Build outward boxes and keep their exact storage bound through consumption.

    Binary64 rounding is monotone, so taking minima/maxima after conversion
    yields the same rounded extrema as rational comparisons before conversion.
    The original outward nextafter then gives identical conservative boxes.
    Only actual bit scans and source-coordinate conversions retain their
    existing exact-work charges; no rational comparisons are performed here.
    When every bank row is used, a preparation bound to this bank already
    scanned the same bit lengths.
    """
    with exact_charge(0, 1024 + 128 * rows.size, budget=budget):
        used, inverse = np.unique(rows, return_inverse=True)
        selected = points[used]
        bits = (
            integer_preparation.rational_bits(points, budget=budget)
            if integer_preparation is not None and used.size == points.shape[0]
            else exact_bits(selected, budget=budget)
        )
        with exact_charge(
            selected.size,
            2 * bigint_bytes(2 * bits + 1) + 8 * selected.size,
            budget=budget,
        ):
            rounded = np.asarray(selected, dtype=np.float64)
            lower, upper = boxes(rounded, inverse.reshape(rows.shape))
            result = (
                np.nextafter(lower, -np.inf),
                np.nextafter(upper, np.inf),
            )
        del used, inverse, selected, bits, rounded, lower, upper
        try:
            yield result
        finally:
            del result


def _exact_shared_edge_disjoint(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Prove edge-adjacent affine triangles meet only on their shared edge."""
    result = np.zeros((first.shape[0],), dtype=np.bool_)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        shared = tuple(sorted(set(left.tolist()).intersection(right.tolist())))
        if len(shared) != 2:
            continue
        free_left = tuple(set(left.tolist()).difference(shared))
        free_right = tuple(set(right.tolist()).difference(shared))
        if len(free_left) != 1 or len(free_right) != 1:
            continue
        with _exact_step(True, 15, budget=budget):
            a, b = points[list(shared)]
            c, d = points[free_left[0]], points[free_right[0]]
            u = tuple(y - x for x, y in zip(a, b, strict=True))
            v = tuple(y - x for x, y in zip(a, c, strict=True))
            normal = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            distance = sum(
                (
                    value * (target - source)
                    for value, source, target in zip(normal, a, d, strict=True)
                ),
                Fraction(0),
            )
        if distance:
            result[row] = True
            continue
        with _exact_step(True, 6, budget=budget):
            axis = max(range(3), key=lambda index: abs(normal[index]))
            axes = tuple(index for index in range(3) if index != axis)
            side_c = (b[axes[0]] - a[axes[0]]) * (c[axes[1]] - a[axes[1]]) - (
                b[axes[1]] - a[axes[1]]
            ) * (c[axes[0]] - a[axes[0]])
            side_d = (b[axes[0]] - a[axes[0]]) * (d[axes[1]] - a[axes[1]]) - (
                b[axes[1]] - a[axes[1]]
            ) * (d[axes[0]] - a[axes[0]])
            result[row] = side_c * side_d < 0
    return result


def _exact_shared_vertex_axis_disjoint(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Prove vertex-adjacent triangles occupy opposite open coordinate halfspaces."""
    result = np.zeros((first.shape[0],), dtype=np.bool_)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        shared = tuple(set(left.tolist()).intersection(right.tolist()))
        if len(shared) != 1:
            continue
        vertex = points[shared[0]]
        free_left = points[list(set(left.tolist()).difference(shared))]
        free_right = points[list(set(right.tolist()).difference(shared))]
        for axis in range(3):
            with _exact_step(True, 4, budget=budget):
                left_delta = tuple(point[axis] - vertex[axis] for point in free_left)
                right_delta = tuple(point[axis] - vertex[axis] for point in free_right)
            if (
                min(left_delta) > 0
                and max(right_delta) < 0
                or max(left_delta) < 0
                and min(right_delta) > 0
            ):
                result[row] = True
                break
    return result


def _exact_triangle_axis_disjoint(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Prove disjoint affine triangles with a strictly separating rational axis."""
    result = np.zeros((first.shape[0],), dtype=np.bool_)
    for row, (left, right) in enumerate(zip(first, second, strict=True)):
        a, b = points[left], points[right]
        with _exact_step(True, 18, budget=budget):
            edges_a = tuple(
                tuple(y - x for x, y in zip(a[index], a[(index + 1) % 3], strict=True))
                for index in range(3)
            )
            edges_b = tuple(
                tuple(y - x for x, y in zip(b[index], b[(index + 1) % 3], strict=True))
                for index in range(3)
            )
            normals = (
                (
                    edges_a[0][1] * edges_a[1][2] - edges_a[0][2] * edges_a[1][1],
                    edges_a[0][2] * edges_a[1][0] - edges_a[0][0] * edges_a[1][2],
                    edges_a[0][0] * edges_a[1][1] - edges_a[0][1] * edges_a[1][0],
                ),
                (
                    edges_b[0][1] * edges_b[1][2] - edges_b[0][2] * edges_b[1][1],
                    edges_b[0][2] * edges_b[1][0] - edges_b[0][0] * edges_b[1][2],
                    edges_b[0][0] * edges_b[1][1] - edges_b[0][1] * edges_b[1][0],
                ),
            )
        axes = [*normals]
        axes.extend(
            (
                left_edge[1] * right_edge[2] - left_edge[2] * right_edge[1],
                left_edge[2] * right_edge[0] - left_edge[0] * right_edge[2],
                left_edge[0] * right_edge[1] - left_edge[1] * right_edge[0],
            )
            for left_edge in edges_a
            for right_edge in edges_b
        )
        for axis in axes:
            if not any(axis):
                continue
            with _exact_step(True, 18, budget=budget):
                projection_a = tuple(
                    sum(
                        (
                            weight * value
                            for weight, value in zip(axis, point, strict=True)
                        ),
                        Fraction(0),
                    )
                    for point in a
                )
                projection_b = tuple(
                    sum(
                        (
                            weight * value
                            for weight, value in zip(axis, point, strict=True)
                        ),
                        Fraction(0),
                    )
                    for point in b
                )
            if max(projection_a) < min(projection_b) or max(projection_b) < min(
                projection_a
            ):
                result[row] = True
                break
    return result


@contextmanager
@checked
def exact_pair_contacts(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield exact pair decisions while their output and working bound is live.

    One positive common denominator preserves every orientation sign, so the
    predicate algorithm decides contact with exact predicates and signs that
    charge themselves. Per triangle pair it forms at most two projection
    normals (twenty visits, eighteen integers of at most ``2 b + 3`` bits for
    bank bits ``b``) and gathers at most 128 slots; a segment pair gathers
    eight slots.
    The integer bank contains only vertices used by this batch. Its injective
    index map preserves declared sharing while avoiding normalization of
    unrelated interior coordinates during a boundary-only contact proof.
    """
    slots = first.size + second.size
    with exact_charge(0, 1024 + 128 * slots, budget=budget):
        result = _exact_pair_contact_values(
            points,
            first,
            second,
            slots=slots,
            budget=budget,
            integer_preparation=integer_preparation,
        )
        try:
            yield result
        finally:
            del result


def _exact_pair_contact_values(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    slots: int,
    budget: CoordinateEnclosureBudget,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not slots:
        return np.zeros(first.shape[0], dtype=np.bool_), np.ones(
            first.shape[0], dtype=np.bool_
        )
    used = np.unique(np.concatenate((first.reshape(-1), second.reshape(-1))))
    if integer_preparation is not None:
        integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
            points, budget=budget, preparation=integer_preparation
        )
        if used.size < points.shape[0]:
            integers = integers[used]
    else:
        selected = points[used] if used.size < points.shape[0] else points
        integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
            selected, budget=budget
        )
    if used.size < points.shape[0]:
        first = np.searchsorted(used, first)
        second = np.searchsorted(used, second)
    if first.shape[1] == 2:
        with exact_charge(0, 8 * 8 * first.shape[0], budget=budget):
            return _segment_contacts(integers, first, second, budget=budget)
    bits = (
        integer_preparation.integer_bits
        if integer_preparation is not None
        else exact_bits(integers, budget=budget)
    )
    hit = np.zeros((first.shape[0],), dtype=np.bool_)
    certain = np.ones((first.shape[0],), dtype=np.bool_)
    for start in range(0, first.shape[0], _EXACT_CONTACT_BATCH):
        stop = min(start + _EXACT_CONTACT_BATCH, first.shape[0])
        count = stop - start
        left, right = first[start:stop], second[start:stop]
        shared = np.sum(left[:, :, None] == right[:, None, :], axis=(1, 2))
        # A welded vertex removes two plane-side inputs and four irrelevant
        # edge/triangle preparations; a welded edge leaves only its fold test.
        work = int(np.sum(np.where(shared == 2, 20, np.where(shared == 1, 28, 40))))
        with exact_charge(
            work, count * (36 * bigint_bytes(2 * bits + 3) + 8 * 128), budget=budget
        ):
            hit[start:stop], certain[start:stop] = (
                audit_topology_owner.triangle_pairs_intersect(
                    integers, left, right, budget=budget
                )
            )
    return hit, certain


@checked
def pairwise_contacts(
    state: EmbeddingState,
    points: np.ndarray,
    items: np.ndarray,
    item_entities: np.ndarray,
    kind: MeshCertificateEntityKind,
    check: MeshCertificateCheck,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    source_rows: np.ndarray | None = None,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
) -> None:
    """Decide contacts of segment/triangle items with exact predicates beyond combinatorial sharing."""

    check = parse(check, MeshCertificateCheck, "pairwise_contacts.check")
    state.checks.append(check)
    exact = points.dtype == object
    if exact:
        with _exact_contact_boxes(
            points, items, budget=budget, integer_preparation=integer_preparation
        ) as contact_boxes:
            try:
                first, second, exceeded = candidate_pairs(
                    contact_boxes[0],
                    contact_boxes[1],
                    limits.maximum_candidate_pairs - state.candidate_pairs,
                )
            finally:
                del contact_boxes
    else:
        lower, upper = boxes(points, items)
        first, second, exceeded = candidate_pairs(
            lower,
            upper,
            limits.maximum_candidate_pairs - state.candidate_pairs,
        )
    state.candidate_pairs += first.size
    if exceeded:
        state.add(f"{check}_capacity", "unresolved", "mesh")
    if source_rows is not None:
        pending = (source_rows[first] < 0) | (source_rows[second] < 0)
        first, second = first[pending], second[pending]
    if first.size and items.shape[1] == 2:
        separated = _shared_segment_separation(
            points, items[first], items[second], budget=budget
        )
        first, second = first[~separated], second[~separated]
    if not first.size:
        return
    if exact:
        hit = np.zeros(first.shape, dtype=np.bool_)
        certain = np.ones(first.shape, dtype=np.bool_)
        if items.shape[1] == 3:
            selected_first, selected_second = items[first], items[second]
            shared = np.sum(
                selected_first[:, :, None] == selected_second[:, None, :],
                axis=(1, 2),
            )
            adjacent = shared == 2
            if np.any(adjacent):
                proven = _exact_shared_edge_disjoint(
                    points,
                    selected_first[adjacent],
                    selected_second[adjacent],
                    budget=budget,
                )
                adjacent_rows = np.flatnonzero(adjacent)
                certain[adjacent_rows[proven]] = True
                pending = np.ones(first.shape, dtype=np.bool_)
                pending[adjacent_rows[proven]] = False
            else:
                pending = np.ones(first.shape, dtype=np.bool_)
            vertex_adjacent = (shared == 1) & pending
            if np.any(vertex_adjacent):
                proven = _exact_shared_vertex_axis_disjoint(
                    points,
                    selected_first[vertex_adjacent],
                    selected_second[vertex_adjacent],
                    budget=budget,
                )
                vertex_rows = np.flatnonzero(vertex_adjacent)
                pending[vertex_rows[proven]] = False
            separate = (shared == 0) & pending
            if np.any(separate):
                proven = _exact_triangle_axis_disjoint(
                    points,
                    selected_first[separate],
                    selected_second[separate],
                    budget=budget,
                )
                separate_rows = np.flatnonzero(separate)
                pending[separate_rows[proven]] = False
        else:
            pending = np.ones(first.shape, dtype=np.bool_)
        if np.any(pending):
            with exact_pair_contacts(
                points, items[first[pending]], items[second[pending]], budget=budget,
                integer_preparation=integer_preparation,
            ) as contact_results:
                try:
                    hit[pending], certain[pending] = contact_results
                finally:
                    del contact_results
    elif items.shape[1] == 2:
        hit, certain = _segment_contacts(
            points, items[first], items[second], budget=budget
        )
    else:
        hit, certain = audit_topology_owner.triangle_pairs_intersect(
            points, items[first], items[second], budget=budget
        )
    # Pieces of one facet or cell touch along their own diagonals.
    distinct = item_entities[first] != item_entities[second]
    hit &= certain & distinct
    uncertain = ~certain & distinct
    if np.any(hit):
        state.add(
            check,
            "violated",
            kind,
            np.concatenate((item_entities[first[hit]], item_entities[second[hit]])),
        )
    if np.any(uncertain):
        state.add(
            f"{check}_predicates",
            "unresolved",
            kind,
            np.concatenate(
                (item_entities[first[uncertain]], item_entities[second[uncertain]])
            ),
        )


def _interval_embedding(
    state: EmbeddingState, mesh: CellMesh, cell_ids: np.ndarray, /
) -> None:
    """Decide 1-D interval disjointness beyond shared vertices."""

    state.checks.append("interval_overlap")
    rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)[:, 0]
    left = np.minimum(coordinates[rows[:, 0]], coordinates[rows[:, 1]])
    right = np.maximum(coordinates[rows[:, 0]], coordinates[rows[:, 1]])
    left_vertex = np.where(coordinates[rows[:, 0]] <= coordinates[rows[:, 1]], 0, 1)
    order = np.lexsort((right, left))
    left, right = left[order], right[order]
    start = rows[order, left_vertex[order]]
    stop = rows[order, 1 - left_vertex[order]]
    running = np.maximum.accumulate(right)
    owner = np.maximum.accumulate(np.where(right == running, np.arange(right.size), 0))
    previous = running[:-1]
    previous_owner = owner[:-1]
    overlap = left[1:] < previous
    contact = (left[1:] == previous) & (start[1:] != stop[previous_owner])
    bad = overlap | contact
    if np.any(bad):
        state.add(
            "interval_overlap",
            "violated",
            "cell",
            np.concatenate(
                (cell_ids[order[1:][bad]], cell_ids[order[previous_owner[bad]]])
            ),
        )


@checked
def cell_loops(mesh: CellMesh, /) -> np.ndarray:
    """Counterclockwise vertex loops of 2-D cells, padded with ``-1``."""

    loops = []
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        loops.append(np.where(valid, vertices, -1))
    return padded_facet_rows(loops)


@checked
def surface_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None = None,
) -> None:
    loops = cell_loops(mesh)
    triangles, owners, nonplanar, undecided = loop_triangles(
        points, loops, limits.maximum_candidate_pairs, budget=budget
    )
    if np.any(nonplanar):
        state.add("nonplanar_cell", "unresolved", "cell", cell_ids[nonplanar])
    if np.any(undecided):
        state.add(
            "undecided_cell_triangulation", "unresolved", "cell", cell_ids[undecided]
        )
    pairwise_contacts(
        state,
        points,
        triangles,
        cell_ids[owners],
        "cell",
        "cell_contact",
        limits,
        budget=budget,
        source_rows=(
            None
            if predecessor is None
            else np.asarray(predecessor.source_rows, dtype=np.int64)[owners]
        ),
    )


@dataclass(frozen=True)
class Boundary:
    """Oriented boundary facets, their affine pieces and their entity ids."""

    rows: np.ndarray
    entities: np.ndarray
    pieces: np.ndarray
    piece_owners: np.ndarray


@checked
def boundary_pieces(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Boundary:
    rows = facets.rows[facets.boundary]
    rows = rows[:, : int(np.max(np.sum(rows >= 0, axis=1), initial=1))]
    entities = (
        facet_entity_ids(mesh, rows) if rows.size else np.empty((0,), dtype=np.int64)
    )
    state.boundary_facets = rows.shape[0]
    if mesh.topological_dimension == 2:
        return Boundary(rows, entities, rows, np.arange(rows.shape[0]))
    pieces, owners, nonplanar, undecided = loop_triangles(
        points, rows, limits.maximum_candidate_pairs, budget=budget
    )
    if np.any(nonplanar):
        state.add("curved_boundary_facet", "unresolved", "facet", entities[nonplanar])
    if np.any(undecided):
        state.add("undecided_boundary_facet", "unresolved", "facet", entities[undecided])
    return Boundary(rows, entities, pieces, owners)


@checked
def boundary_shells(
    boundary: Boundary, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary shells linked through manifold ridges, and unbalanced shells."""

    rows = boundary.rows
    count = rows.shape[0]
    if dimension == 2:
        tails = rows[:, 0]
        heads = rows[:, 1]
        facet = np.arange(count)
    else:
        lengths = np.sum(rows >= 0, axis=1)
        column = np.arange(rows.shape[1])
        following = rows[np.arange(count)[:, None], (column + 1) % lengths[:, None]]
        slots = column[None, :] < lengths[:, None]
        tails = rows[slots]
        heads = following[slots]
        facet = np.broadcast_to(np.arange(count)[:, None], rows.shape)[slots]
    if dimension == 2:
        # Ridges are vertices: one outgoing and one incoming facet link.
        vertex = np.concatenate((tails, heads))
        direction = np.concatenate((np.ones_like(tails), -np.ones_like(heads)))
        owners = np.concatenate((facet, facet))
    else:
        vertex = np.stack((np.minimum(tails, heads), np.maximum(tails, heads)), axis=1)
        direction = np.where(tails < heads, 1, -1)
        owners = facet
    keys = vertex.reshape(vertex.shape[0], -1)
    _, ridge, ridge_counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    ridge = ridge.reshape(-1)
    balance = np.zeros((ridge_counts.size,), dtype=np.int64)
    np.add.at(balance, ridge, direction)
    order = np.argsort(ridge, kind="stable")
    same = ridge[order[1:]] == ridge[order[:-1]]
    manifold = (ridge_counts[ridge[order[:-1]]] == 2) & (balance[ridge[order[:-1]]] == 0)
    link = same & manifold
    graph = coo_matrix(
        (
            np.ones(np.count_nonzero(link), dtype=np.int8),
            (owners[order[:-1][link]], owners[order[1:][link]]),
        ),
        shape=(count, count),
    )
    _, labels = connected_components(graph, directed=False)
    shell_keys = np.concatenate((labels[owners][:, None], keys), axis=1)
    _, shell_ridge = np.unique(shell_keys, axis=0, return_inverse=True)
    shell_balance = np.zeros((int(np.max(shell_ridge, initial=-1)) + 1,), np.int64)
    np.add.at(shell_balance, shell_ridge.reshape(-1), direction)
    unbalanced = np.zeros((int(np.max(labels, initial=-1)) + 1,), dtype=np.bool_)
    np.logical_or.at(
        unbalanced, labels[owners], shell_balance[shell_ridge.reshape(-1)] != 0
    )
    return labels, unbalanced


def shell_representatives(
    rows: np.ndarray, labels: np.ndarray, shell_count: int, /
) -> np.ndarray:
    """Smallest vertex of each shell that belongs to no other shell (``-1``)."""

    slots = rows >= 0
    vertex = rows[slots]
    shell = np.broadcast_to(labels[:, None], rows.shape)[slots]
    pairs = np.unique(np.stack((vertex, shell), axis=1), axis=0)
    vertices, multiplicity = np.unique(pairs[:, 0], return_counts=True)
    exclusive = np.isin(pairs[:, 0], vertices[multiplicity == 1])
    representative = np.full((shell_count,), -1, dtype=np.int64)
    chosen = pairs[exclusive]
    # Rows are sorted by vertex, so the first occurrence per shell is minimal.
    shells, first = np.unique(chosen[:, 1], return_index=True)
    representative[shells] = chosen[first, 0]
    return representative


def _shell_measures(
    integers: np.ndarray, boundary: Boundary, labels: np.ndarray, shells: int, /
) -> list[int]:
    """Return integer-scaled signed measures of every shell."""

    pieces = integers[boundary.pieces]
    if boundary.pieces.shape[1] == 2:
        values = cofactor_determinant_2(pieces[:, 0], pieces[:, 1])
    else:
        values = cofactor_determinant_3(pieces[:, 0], pieces[:, 1], pieces[:, 2])
    totals = [0] * shells
    for shell, value in zip(
        labels[boundary.piece_owners].tolist(), values.tolist(), strict=True
    ):
        totals[shell] += value
    return totals


@checked
def ray_crossings_2d(
    point: np.ndarray,
    segments: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[int, bool]:
    """Winding of oriented segments at ``point`` from a perturbed +x ray."""

    a = segments[:, 0]
    b = segments[:, 1]
    direction = np.sign(b[:, 1] - a[:, 1]).astype(np.int16)
    upward = (a[:, 1] <= point[1]) & (point[1] < b[:, 1])
    downward = (b[:, 1] <= point[1]) & (point[1] < a[:, 1])
    inside = np.where(direction > 0, upward, downward) & (direction != 0)
    if not np.any(inside):
        return 0, True
    sigma, certain = mesh_orient2d(
        a[inside], b[inside], np.broadcast_to(point, a[inside].shape), budget=budget
    )
    hit = sigma * direction[inside] > 0
    decided = bool(np.all(certain) and np.all(sigma != 0))
    return int(np.sum(direction[inside][hit])), decided


@checked
def ray_crossings_3d(
    point: np.ndarray,
    triangles: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[int, bool]:
    """Winding of oriented triangles at ``point`` from a perturbed +x ray.

    The ray starts at ``point + (0, e, e**2)`` for infinitesimal ``e``; a zero
    projected edge orientation is resolved by the first nonvanishing derivative,
    ``sign(u_z - v_z)`` then ``sign(v_y - u_y)``.
    """

    projected = triangles[:, :, 1:]
    target = np.broadcast_to(point[1:], projected[:, 0].shape)
    orientation, certain = mesh_orient2d(
        projected[:, 0], projected[:, 1], projected[:, 2], budget=budget
    )
    decided = bool(np.all(certain))
    inside = orientation != 0
    for start, stop in ((0, 1), (1, 2), (2, 0)):
        u = projected[:, start]
        v = projected[:, stop]
        sign, known = mesh_orient2d(u, v, target, budget=budget)
        perturbed = np.where(
            sign != 0,
            sign,
            np.where(
                u[:, 1] != v[:, 1],
                np.sign(u[:, 1] - v[:, 1]),
                np.sign(v[:, 0] - u[:, 0]),
            ),
        )
        inside &= perturbed == orientation
        decided &= bool(np.all(known))
    if not np.any(inside):
        return 0, decided
    selected = triangles[inside]
    sigma, known = mesh_orient3d(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        np.broadcast_to(point, selected[:, 0].shape),
        budget=budget,
    )
    hit = sigma * orientation[inside] < 0
    decided &= bool(np.all(known) and np.all(sigma != 0))
    return int(np.sum(orientation[inside][hit])), decided


def _exterior_degrees(
    state: EmbeddingState,
    points: np.ndarray,
    boundary: Boundary,
    dimension: int,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
) -> None:
    with budget.temporary_scope():
        _exterior_degrees_scoped(
            state,
            points,
            boundary,
            dimension,
            limits,
            budget=budget,
            integer_preparation=integer_preparation,
        )


def _exterior_degrees_scoped(
    state: EmbeddingState,
    points: np.ndarray,
    boundary: Boundary,
    dimension: int,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
) -> None:
    state.checks.append("exterior_degree")
    labels, unbalanced = boundary_shells(boundary, dimension)
    shells = unbalanced.size
    state.shells = shells
    if np.any(unbalanced):
        state.add(
            "nonmanifold_boundary_ridge",
            "unresolved",
            "facet",
            boundary.entities[unbalanced[labels]],
        )
        return
    representative = shell_representatives(boundary.rows, labels, shells)
    exact = points.dtype == object
    pieces = boundary.pieces.shape[0]
    if exact:
        # One positive common denominator preserves every orientation, measure
        # sign and the perturbed ray: rational source shells use integer scaling.
        # A segment's 2D cofactor uses three operations; a triangle's 3D cofactor
        # uses fourteen. Both fit the existing 2**(3 b + 3) storage bound.
        # Min/max boxes compare each later corner in each coordinate twice;
        # a ray test takes five box comparisons per piece and, per crossed
        # candidate, fifteen perturbation visits with six differences below
        # 2**(b + 1) and forty slots. Orientation signs charge themselves.
        integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
            points, budget=budget, preparation=integer_preparation
        )
        bits = (
            integer_preparation.integer_bits
            if integer_preparation is not None
            else exact_bits(integers, budget=budget)
        )
        coordinates = integers
        exact_reserve(
            2 * dimension * (boundary.pieces.shape[1] - 1) * pieces,
            8 * 15 * pieces + shells * bigint_bytes(3 * bits + 3 + pieces.bit_length()),
            budget=budget,
        )
    else:
        integers, _ = dyadic_integers(points)
        coordinates = points
        bits = 0
    with _exact_step(
        exact,
        (3 if dimension == 2 else 14) * pieces,
        14 * pieces * bigint_bytes(3 * bits + 3),
        budget=budget,
    ):
        measures = _shell_measures(integers, boundary, labels, shells)
    piece_shell = labels[boundary.piece_owners]
    corners = coordinates[boundary.pieces]
    lower = np.min(corners, axis=1)
    upper = np.max(corners, axis=1)
    if shells * boundary.pieces.shape[0] > limits.maximum_ray_tests:
        state.add("exterior_degree_capacity", "unresolved", "mesh")
        return
    # One symbolic ray query per shell; the shell count is bounded by the ray budget.
    for shell in range(shells):
        facets = boundary.entities[labels == shell]
        if representative[shell] < 0 or measures[shell] == 0:
            state.add("degenerate_shell", "unresolved", "facet", facets)
            continue
        origin = coordinates[representative[shell]]
        # Closed transverse boxes keep every piece the perturbed ray can cross.
        with _exact_step(exact, 5 * pieces, budget=budget):
            others = corners[
                (piece_shell != shell)
                & np.all(lower[:, 1:] <= origin[1:], axis=1)
                & np.all(upper[:, 1:] >= origin[1:], axis=1)
                & (upper[:, 0] >= origin[0])
            ]
        state.ray_tests += boundary.pieces.shape[0]
        rows = others.shape[0]
        with _exact_step(
            exact, 15 * rows, rows * (6 * bigint_bytes(bits + 1) + 8 * 40), budget=budget
        ):
            if dimension == 3:
                winding, decided = ray_crossings_3d(origin, others, budget=budget)
            else:
                winding, decided = ray_crossings_2d(origin, others, budget=budget)
        if not decided:
            state.add("exterior_degree_predicates", "unresolved", "facet", facets)
            continue
        degree = winding + (0 if measures[shell] > 0 else -1)
        if degree != 0:
            state.add("nested_component", "violated", "facet", facets)


@checked
def volume_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None = None,
    geometry: CellGeometrySpec | None = None,
    restriction: CellGeometryPredecessor | None = None,
) -> None:
    boundary = boundary_pieces(state, mesh, points, facets, limits, budget=budget)
    decided = _nested_boundary_import(
        state, mesh, geometry, points, limits, budget=budget, restriction=restriction
    )
    inherited = (
        None
        if predecessor is None or decided
        else _boundary_predecessor_rows(predecessor, facets, budget=budget)
    )
    _boundary_embedding(
        state, points, boundary, mesh.topological_dimension, limits,
        budget=budget, inherited=inherited, decided=decided,
    )


def _nested_boundary_import(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec | None,
    points: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    restriction: CellGeometryPredecessor | None,
) -> bool:
    """Import the source boundary-contact theorem through an exact parent partition.

    Admitted only on a clean state after facet pairing, with the authenticated
    certified source embedding of a nested restriction. The imported root
    union and the partition proof each record their own check; any absent or
    failed premise retains the full pairwise boundary-contact proof.
    """
    if restriction is None or geometry is None or not state.clean:
        return False
    from . import _restricted_embedding as restricted_embedding_owner

    with budget.temporary_scope():
        imported = restricted_embedding_owner.nested_parent_partition(
            mesh, geometry, points, limits,
            state=state, budget=budget, predecessor=restriction,
        )
    if isinstance(imported, str) or not state.clean:
        return False
    state.checks.extend(
        (
            "boundary_contact",
            "mapped_restriction_root_embedding_reuse",
            "restriction_parent_partition",
        )
    )
    state.imported_root_embedding_certificate_id = imported.certificate_id
    return True


def _boundary_embedding(
    state: EmbeddingState,
    points: np.ndarray,
    boundary: Boundary,
    dimension: int,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    inherited: np.ndarray | None,
    integer_preparation: (
        exact_polyhedral_geometry_owner.CoordinateIntegerPreparation | None
    ) = None,
    decided: bool = False,
) -> None:
    """Decide every boundary contact and exterior degree in its input bank.

    ``decided`` contacts were transferred by an admitted parent-partition
    theorem; exterior degrees are always computed on this boundary.
    """
    if not decided:
        pairwise_contacts(
            state,
            points,
            boundary.pieces,
            boundary.entities[boundary.piece_owners],
            "facet",
            "boundary_contact",
            limits,
            budget=budget,
            source_rows=(
                None if inherited is None else inherited[boundary.piece_owners]
            ),
            integer_preparation=integer_preparation,
        )
    if not state.clean:
        # Degree theory requires the preceding boundary decisions.
        state.add("exterior_degree_premise", "unresolved", "mesh")
        return
    _exterior_degrees(
        state, points, boundary, dimension, limits, budget=budget,
        integer_preparation=integer_preparation,
    )


def _exact_planar_boundary_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    limits: MeshCertificateLimits,
    maximum_bits: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None,
    geometry: CellGeometrySpec | None = None,
    restriction: CellGeometryPredecessor | None = None,
) -> None:
    """Normalize only the source points used by the planar boundary theorem.

    Full source-map validity and oriented facet pairing have already been
    checked. The remaining contact, shell measure and degree predicates use
    boundary vertices only. An injective index map preserves every incidence
    and the original facet entity IDs, without changing the physical source.
    """
    boundary = boundary_pieces(state, mesh, points, facets, limits, budget=budget)
    decided = _nested_boundary_import(
        state, mesh, geometry, points, limits, budget=budget, restriction=restriction
    )
    inherited = (
        None
        if predecessor is None or decided
        else _boundary_predecessor_rows(predecessor, facets, budget=budget)
    )
    with budget.temporary_scope():
        budget.reserve(0, 1024 + 128 * boundary.rows.size)
        used = np.unique(boundary.rows[boundary.rows >= 0])
        local_points = points[used]
        local_boundary = Boundary(
            np.searchsorted(used, boundary.rows),
            boundary.entities,
            np.searchsorted(used, boundary.pieces),
            boundary.piece_owners,
        )
        integer_preparation = exact_polyhedral_geometry_owner.CoordinateIntegerPreparation(
            local_points, maximum_bits, budget=budget
        )
        if integer_preparation.profile is None:
            state.add("exact_source_bit_budget", "unresolved", "mesh")
            return
        predicate_points, binary = exact_source_predicate_view(
            local_points, budget=budget
        )
        if binary:
            state.checks.append("exact_source_binary64_value_equivalence")
        _boundary_embedding(
            state, predicate_points, local_boundary, 2, limits,
            budget=budget, inherited=inherited,
            integer_preparation=None if binary else integer_preparation,
            decided=decided,
        )


@checked
def exact_source_predicate_view(
    points: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, bool]:
    """Export binary64 only when it represents every authenticated source value.

    The binary predicates use exact predicates with adaptive/dyadic signs. This is an
    equality-certified execution view of the source bank, not its mesh carrier.
    A nonrepresentable value retains the source bank's rational predicate path.
    Rational-subfield algebraic scalars are unwrapped into a transient exact
    Fraction view before rational predicates consume them. Genuine radicals
    remain in their authored field for the mapped-field contact path.
    """
    algebraic = False
    for value in points.flat:
        budget.reserve(1)
        if isinstance(value, exact_algebraic_geometry_owner.RadicalElement):
            algebraic = True
            if any(radicand != 1 for radicand, _ in value.terms):
                return points, False
    if algebraic:
        budget.reserve(points.size, 128 + points.size * np.dtype(object).itemsize)
        rational = np.empty(points.shape, dtype=object)
        for index, value in enumerate(points.flat):
            rational.flat[index] = exact_algebraic_geometry_owner.rational_coefficient(
                value, role="embedding.exact_source_predicate_view",
            )
        points = rational
    for value in points.flat:
        budget.reserve(2)
        try:
            numeric = float(value)
        except OverflowError:
            return points, False
        if not math.isfinite(numeric) or Fraction(numeric) != value:
            return points, False
    budget.reserve(points.size, 128 + points.size * np.dtype(np.float64).itemsize)
    exported = np.empty(points.shape, dtype=np.float64)
    for index, value in enumerate(points.flat):
        exported.flat[index] = float(value)
    return exported, True


def source_bank_binary64_rows(
    bank: CoordinateSourceBank, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, np.ndarray]:
    """Issue exact binary64-equality proofs independently for immutable rows.

    Only rows whose mask is true may be used by binary predicates. A false
    row retains its original exact source; its float slots are not a carrier
    approximation or a substitute source. The strong bank reference prevents
    identity-key recycling across this request's consumers.
    """
    from ..discretization._coordinate_operands import (
        _immutable_array_layout,
        seal_coordinate_operand_array,
    )

    cached = budget.source_binary64_rows_cache.get(id(bank))
    if cached is not None and cached[0] is bank:
        budget.reserve(2)
        if (
            _immutable_array_layout(cached[1]) != cached[3]
            or _immutable_array_layout(cached[2]) != cached[4]
        ):
            raise ValueError("Source binary64 rows must retain their actual immutable numeric layouts.")
        return cached[1], cached[2]
    ambient = len(bank[0]) if bank else 0
    with budget.temporary_scope(), budget.live_storage() as conversion_storage:
        budget.reserve(0, 256 + len(bank) * (8 * ambient + 1))
        view = np.full((len(bank), ambient), np.nan, dtype=np.float64)
        represented = np.ones((len(bank),), dtype=np.bool_)
        conversions: dict[tuple[type, int, int], tuple[float, bool]] = {}
        for row, point in enumerate(bank):
            if len(point) != ambient:
                represented[row] = False
                continue
            for column, value in enumerate(point):
                # This is only a rational binary64-equality filter. Algebraic
                # rows retain their exact-source path; no rounding is necessary.
                if type(value) is not Fraction:
                    represented[row] = False
                    break
                budget.reserve(1)
                key = (Fraction, value.numerator, value.denominator)
                known = conversions.get(key)
                if known is None:
                    budget.reserve(3)
                    conversion_storage.set_bound(
                        512 + 512 * (len(conversions) + 1)
                    )
                    try:
                        numeric = float(value)
                        exact = math.isfinite(numeric) and Fraction(numeric) == value
                    except (OverflowError, TypeError, ValueError):
                        numeric, exact = math.nan, False
                    known = numeric, exact
                    conversions[key] = known
                numeric, exact = known
                if not exact:
                    represented[row] = False
                    break
                view[row, column] = numeric
        del conversions
        # Only the last scalar/key locals remain; the full conversion table is
        # dead before copying the two immutable exported numeric banks.
        conversion_storage.set_bound(512 if bank else 0)
        view = seal_coordinate_operand_array(view, budget=budget)
        represented = seal_coordinate_operand_array(represented, budget=budget)
        layouts = _immutable_array_layout(view), _immutable_array_layout(represented)
        budget.retain_owner((view, represented, *layouts))
        budget.source_binary64_rows_cache[id(bank)] = (
            bank, view, represented, *layouts,
        )
        return view, represented


def source_bank_predicate_view(
    bank: CoordinateSourceBank, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, bool]:
    """Export one prepared immutable source bank's predicate view once per ledger.

    Embedding and coverage certify the same prepared source bank in one
    request; the second consumer reads the view the first one earned.
    """
    view, represented = source_bank_binary64_rows(bank, budget=budget)
    cached = budget.source_predicate_view_cache.get(id(bank))
    if cached is not None and cached[0] is bank:
        budget.reserve(1)
        if cached[1] is not view or cached[2] is not True:
            raise ValueError("Source predicate views must retain their actual immutable binary row bank.")
        return view, True
    binary = bool(np.all(represented))
    if binary:
        budget.source_predicate_view_cache[id(bank)] = (bank, view, True)
        return view, True
    # Object arrays cannot have immutable byte-backed object references. Never
    # retain one as a reusable authority behind a reversible NumPy write flag.
    budget.reserve(0, 128 + len(bank) * (len(bank[0]) if bank else 0) * 8)
    exact = np.asarray(bank, dtype=object)
    exact.flags.writeable = False
    return exact, False


def _exact_source_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: ExactCellGeometrySource,
    cell_ids: np.ndarray,
    junctions: np.ndarray,
    limits: MeshCertificateLimits,
    scope: CoordinateMapScope,
    exact_points: CoordinateSourceBank,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    mapped_certifier: MappedEmbeddingCertifier,
    predecessor: CellGeometryPredecessor | None = None,
    mapped_predecessor: CellGeometryPredecessor | None = None,
) -> None:
    """Certify embedding on the authenticated source coordinates.

    The binary64 carrier is only the correctly rounded image of the source;
    contacts, shell measures and exterior degrees are decided on the source.
    The PLC constructor admits only direct tetrahedral carriers in three
    dimensions, so the source runs the volume proof.
    """

    # Geometry-bound facets are shared with coverage and fidelity in this request.
    facets = mesh_facets(mesh, geometry=geometry, budget=budget)
    pairing_findings(state, mesh, facets, junctions)
    match source:
        case (
            exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource()
            | exact_plc_geometry_owner.ExactPlcCellGeometrySource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
        ) if scope == "affine":
            state.checks.append("exact_plc_source_coordinates")
            # Scope preparation proved every source map affine and retained
            # the rational target vertex bank, so no cell expression is rebuilt.
            points = np.asarray(exact_points, dtype=object)
            if mesh.topological_dimension == 2:
                _exact_planar_boundary_embedding(
                    state, mesh, points, facets, limits, source.maximum_bits,
                    budget=budget, predecessor=predecessor,
                    geometry=geometry, restriction=mapped_predecessor,
                )
            elif (
                exact_polyhedral_geometry_owner.coordinate_integer_profile(
                    points, source.maximum_bits, budget=budget
                )
                is None
            ):
                state.add("exact_source_bit_budget", "unresolved", "mesh")
            else:
                predicate_points, binary = source_bank_predicate_view(
                    exact_points, budget=budget
                )
                if binary:
                    state.checks.append("exact_source_binary64_value_equivalence")
                volume_embedding(
                    state,
                    mesh,
                    predicate_points,
                    facets,
                    limits,
                    budget=budget,
                    predecessor=predecessor,
                    geometry=geometry,
                    restriction=mapped_predecessor,
                )
            state.source_expression_work_units = budget.current_measurement.work_units
            state.source_expression_peak_bytes = (
                budget.current_measurement.peak_bytes_upper
            )
        case (
            exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
        ):
            mapped_certifier(
                state,
                mesh,
                geometry,
                cell_ids,
                limits,
                budget=budget,
                predecessor=mapped_predecessor,
            )
            state.source_expression_work_units = budget.current_measurement.work_units
            state.source_expression_peak_bytes = (
                budget.current_measurement.peak_bytes_upper
            )
        case (
            exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_plc_geometry_owner.ExactPlcCellGeometrySource()
        ) if all(block.cell_kind == "hexahedron" for block in mesh.blocks) or (
            isinstance(source, exact_plc_geometry_owner.ExactPlcCellGeometrySource)
            and any(
                isinstance(
                    element,
                    (
                        cell_geometry_owner.BarycentricCellGeometryElement,
                        cell_geometry_owner.PolynomialComposedCellGeometryElement,
                        cell_geometry_owner.RationalComposedCellGeometryElement,
                        cell_geometry_owner.RestrictedCellGeometryElement,
                    ),
                )
                for element in geometry.elements
            )
        ):
            mapped_certifier(
                state,
                mesh,
                geometry,
                cell_ids,
                limits,
                budget=budget,
                predecessor=mapped_predecessor,
            )
            state.source_expression_work_units = budget.current_measurement.work_units
            state.source_expression_peak_bytes = (
                budget.current_measurement.peak_bytes_upper
            )
        case (
            exact_power_geometry_owner.ExactPowerCellGeometrySource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
        ):
            exact_power_certificates_owner.certify_exact_power_embedding(
                state, mesh, geometry, facets, cell_ids, limits, budget=budget
            )
        case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
            state.checks.append("exact_plc_source_coordinates")
            # The request's already-active ledger owns source preparation and
            # every exact predicate; no stage renews its allowance.
            points = np.asarray(exact_points, dtype=object)
            if (
                exact_polyhedral_geometry_owner.coordinate_integer_profile(
                    points, source.maximum_bits, budget=budget
                )
                is None
            ):
                state.add("exact_source_bit_budget", "unresolved", "mesh")
            else:
                predicate_points, binary = exact_source_predicate_view(
                    points, budget=budget
                )
                if binary:
                    state.checks.append("exact_source_binary64_value_equivalence")
                volume_embedding(
                    state,
                    mesh,
                    predicate_points,
                    facets,
                    limits,
                    budget=budget,
                    predecessor=predecessor,
                )
            state.source_expression_work_units = budget.current_measurement.work_units
            state.source_expression_peak_bytes = (
                budget.current_measurement.peak_bytes_upper
            )
        case invalid:
            assert_never(invalid)


def _junction_vertices(mesh: CellMesh, values: ArrayLike | None, /) -> np.ndarray:
    """Sorted declared junction vertex global ids of an interval mesh."""

    if values is None:
        return np.zeros((0,), dtype=np.int64)
    if mesh.topological_dimension != 1:
        raise ValueError(
            f"certify_global_embedding.junction_vertices must belong to an interval mesh; got dimension={mesh.topological_dimension}"
        )
    junctions = np.unique(np.asarray(values, dtype=np.int64).reshape(-1))
    if np.setdiff1d(junctions, np.asarray(mesh.vertex_global_ids, dtype=np.int64)).size:
        raise ValueError(
            f"certify_global_embedding.junction_vertices must name mesh vertex global IDs; got {junctions.tolist()!r}"
        )
    return junctions


def _contact_predecessor(
    predecessor: CellGeometryPredecessor | None,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    limits: MeshCertificateLimits,
    junctions: tuple[int, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> CellGeometryPredecessor | None:
    """Authenticate imported evidence and select the affine contact reuse route."""
    if predecessor is None:
        return None
    source = predecessor.source_embedding
    if source is None or source.binding.coordinate_scope != "affine":
        return None
    predecessor.require_bound(mesh, geometry, budget=budget)
    source = GlobalEmbeddingCertificate.from_stored_fields(source, budget=budget)
    source.binding.require(predecessor.source_mesh, predecessor.source_geometry)
    predecessor.source_validity.require_bound(
        predecessor.source_geometry, mesh=predecessor.source_mesh
    )
    if (
        source.status != "certified"
        or not predecessor.source_validity.all_certified
        or source.validity_certificate_id != predecessor.source_validity.certificate_id
    ):
        raise ValueError(
            "certify_global_embedding.predecessor.source_embedding must certify its bound source validity; "
            f"got status={source.status!r}, validity={source.validity_certificate_id!r}"
        )
    # Mapped/periodic certificates do not retain per-contact subdivision counts
    # or check sequences. Their full routes preserve those proof identities.
    if (
        source.binding.coordinate_scope != "affine"
        or source.binding.limits_id != limits.limits_id
        or source.binding.junction_vertices != junctions
        or predecessor.source_mesh.periodic_topology is not None
        or mesh.periodic_topology is not None
    ):
        return None
    dimensions = (mesh.topological_dimension, mesh.ambient_dimension)
    check = (
        "cell_contact" if dimensions in ((1, 2), (1, 3), (2, 3)) else "boundary_contact"
    )
    return predecessor if check in source.evaluated_checks else None


def _boundary_predecessor_rows(
    predecessor: CellGeometryPredecessor,
    facets: Facets,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Map boundary occurrences by global vertex keys and unchanged source owners."""
    source_mesh = predecessor.source_mesh
    source_facets = mesh_facets(source_mesh, budget=budget)
    source_vertices = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    target_vertices = np.asarray(
        predecessor.target_mesh.vertex_global_ids, dtype=np.int64
    )
    source_keys = {
        (
            int(owner),
            tuple(sorted(source_vertices[row[row >= 0]].tolist())),
        )
        for row, owner in zip(
            source_facets.rows[source_facets.boundary],
            source_facets.cells[source_facets.boundary],
            strict=True,
        )
    }
    target_rows = np.asarray(predecessor.source_rows, dtype=np.int64)
    inherited = np.full((np.count_nonzero(facets.boundary),), -1, dtype=np.int64)
    for index, (row, owner) in enumerate(
        zip(facets.rows[facets.boundary], facets.cells[facets.boundary], strict=True)
    ):
        source_row = int(target_rows[owner])
        key = (
            source_row,
            tuple(sorted(target_vertices[row[row >= 0]].tolist())),
        )
        if source_row >= 0 and key in source_keys:
            inherited[index] = source_row
    return inherited


def _certify_affine_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    facets: Facets,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    dimensions: tuple[int, int],
    /,
    *,
    geometry: CellGeometrySpec,
    restriction: CellGeometryPredecessor | None,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None = None,
) -> None:
    """Select the affine contact and boundary-degree proof for paired facets."""
    match dimensions:
        case (1, 1):
            _interval_embedding(state, mesh, cell_ids)
        case (1, 2) | (1, 3):
            items = np.concatenate(
                [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
            )
            pairwise_contacts(
                state,
                points,
                items,
                cell_ids,
                "cell",
                "cell_contact",
                limits,
                budget=budget,
                source_rows=(
                    None
                    if predecessor is None
                    else np.asarray(predecessor.source_rows, dtype=np.int64)
                ),
            )
        case (2, 3):
            surface_embedding(
                state,
                mesh,
                points,
                cell_ids,
                limits,
                budget=budget,
                predecessor=predecessor,
            )
        case (2, 2) | (3, 3):
            volume_embedding(
                state,
                mesh,
                points,
                facets,
                limits,
                budget=budget,
                predecessor=predecessor,
                geometry=geometry,
                restriction=restriction,
            )
        case _:
            state.add("unsupported_embedding_dimension", "unresolved", "mesh")


_ISSUED_NESTED_PARTITIONS: IssuedProofs[
    CoordinateEnclosureBudget, GlobalEmbeddingCertificate
] = IssuedProofs()


def _nested_partition_authority_key(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    predecessor: CellGeometryPredecessor,
    certificate: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> str:
    predecessor.require_bound(mesh, geometry, budget=budget)
    return _nested_partition_content_key(
        mesh, geometry, predecessor, certificate, limits, budget=budget
    )


def _nested_partition_content_key(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    predecessor: CellGeometryPredecessor,
    certificate: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> str:
    """Bind content after the same operation's actual predecessor validation.

    Issuance follows the fresh closed-partition/root proof without callbacks
    or publication between validation and recording. Consumer verification
    must use ``_nested_partition_authority_key`` to authenticate current owners.
    """
    from ..lifecycle._meshing_source_records import stored_record_content_id

    budget.reserve(8 + len(certificate.evaluated_checks), 1024)
    return canonical_fingerprint(
        {
            "kind": "earned-current-nested-partition",
            "target_scope": coordinate_identity_owner.coordinate_scope_key(mesh, geometry),
            "predecessor": predecessor.content_id,
            "certificate": stored_record_content_id(certificate, excluded="", budget=budget),
            "limits": stored_record_content_id(limits, excluded="", budget=budget),
        }
    )


def _has_nested_partition_premises(
    certificate: GlobalEmbeddingCertificate, predecessor: CellGeometryPredecessor,
) -> bool:
    root = certificate.restriction_root_embedding
    source = predecessor.source_embedding
    return (
        certificate.status == "certified"
        and source is not None
        and root is not None
        and root.imported_certificate_id == source.certificate_id
        and {
            "restriction_parent_partition", "mapped_restriction_root_embedding_reuse",
            "facet_pairing", "boundary_contact",
        }.issubset(certificate.evaluated_checks)
    )


def is_issued_current_nested_partition(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    predecessor: CellGeometryPredecessor,
    certificate: GlobalEmbeddingCertificate,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Authenticate the complete partition actually earned on this request ledger."""
    if not _has_nested_partition_premises(certificate, predecessor):
        return False
    certificate.binding.require(mesh, geometry)
    return _ISSUED_NESTED_PARTITIONS.get(
        budget,
        _nested_partition_authority_key(
            mesh, geometry, predecessor, certificate, limits, budget=budget
        ),
    ) is certificate


def _embedding_request_key(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    limits: MeshCertificateLimits,
    junctions: tuple[int, ...],
    /,
) -> str:
    """Bind request-local proof reuse to current mesh arrays and every proof input."""
    return canonical_fingerprint(
        {
            "kind": "global-embedding-request",
            "mesh": mesh.mesh_id,
            "mesh_arrays": array_tree_fingerprint(mesh),
            "topology": mesh.topology_id,
            "periodic_topology": None
            if mesh.periodic_topology is None
            else mesh.periodic_topology.periodic_topology_id,
            "geometry": cell_geometry_id(geometry),
            "coordinate_scope": coordinate_identity_owner.coordinate_scope_key(
                mesh, geometry
            ),
            "validity": validity.certificate_id,
            "limits": limits.limits_id,
            "junctions": junctions,
        }
    )


_COMPLETE_PERIODIC_ROUTES = (
    frozenset(
        (
            "cell_validity",
            "periodic_authored_group_identity",
            "periodic_full_source_hulls",
            "periodic_image_sufficiency",
            "periodic_mapped_trace_equivariance",
            "periodic_mapped_self_image_embedding",
        )
    ),
    frozenset(
        (
            "cell_validity",
            "periodic_authored_group_identity",
            "periodic_mapped_trace_equivariance",
            "periodic_restriction_source_embedding",
            "periodic_restriction_parent_partition",
            "periodic_restriction_boundary_subdivision",
        )
    ),
    frozenset(
        (
            "cell_validity",
            "periodic_authored_group_identity",
            "periodic_mapped_trace_equivariance",
            "periodic_mapped_restriction_source_embedding",
            "periodic_mapped_restriction_parent_partition",
            "periodic_mapped_restriction_common_subcomplex",
            "periodic_mapped_restriction_derived_region_assignment",
        )
    ),
)


def _periodic_restriction_premise(
    premise: PeriodicRestrictionPremise | None,
    mesh: CellMesh,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> PeriodicRestrictionPremise | None:
    """Authenticate a complete current periodic source certificate for transfer."""
    if premise is None:
        return None
    if mesh.periodic_topology is None or premise.source_mesh.periodic_topology is None:
        raise ValueError(
            "certify_global_embedding.periodic_restriction must bind periodic source and target meshes; got a nonperiodic mesh"
        )
    source = GlobalEmbeddingCertificate.from_stored_fields(
        premise.source_embedding, budget=budget
    )
    source.binding.require(premise.source_mesh, premise.source_geometry)
    premise.source_validity.require_bound(
        premise.source_geometry, mesh=premise.source_mesh
    )
    checks = frozenset(source.evaluated_checks)
    if (
        source.status != "certified"
        or source.findings
        or not premise.source_validity.all_certified
        or source.validity_certificate_id != premise.source_validity.certificate_id
        or not any(route <= checks for route in _COMPLETE_PERIODIC_ROUTES)
    ):
        raise ValueError(
            "certify_global_embedding.periodic_restriction.source_embedding must be a complete certified quotient embedding of its bound source validity; "
            f"got status={source.status!r}, checks={sorted(checks)!r}"
        )
    predecessor = premise.mapped_predecessor
    if predecessor is not None and any(
        actual is not declared
        for actual, declared in (
            (predecessor.source_mesh, premise.source_mesh),
            (predecessor.source_geometry, premise.source_geometry),
            (predecessor.source_validity, premise.source_validity),
            (predecessor.source_embedding, premise.source_embedding),
        )
    ):
        raise ValueError(
            "Mapped periodic predecessor must retain the actual authenticated premise source owners"
        )
    return premise




def _certify_cad_boundary_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    closure: CadBoundaryClosureEvidence,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Consume the positive-local-degree theorem without closed-cell injectivity.

    Only the already proved whole-boundary contact obligation is imported.
    Full-cell positivity, all shared entity traces, oriented facet pairing and
    shell exterior degrees remain independent current-target obligations.
    """
    from . import _boundary_degree_embedding, _embedding_cells, _mapped_embedding

    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
        or mesh.storage is not None
        or mesh.periodic_topology is not None
    ):
        raise ValueError("cad_boundary requires the complete nonperiodic tetrahedral volume")
    budget.retain_owner(closure)
    facets = mesh_facets(mesh, geometry=geometry, budget=budget)
    pairing_findings(state, mesh, facets, np.empty((0,), dtype=np.int64))
    if not state.clean:
        return
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    values = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    budget.reserve(int(cell_ids.size), 1024 + 4096 * int(cell_ids.size))
    cells = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, dofs in zip(np.asarray(block.vertices), np.asarray(route), strict=True):
            coordinates = coordinate_enclosure_owner.coordinate_expressions(
                element, tuple(values[int(dof)] for dof in dofs), budget=budget
            )
            if coordinates is None:
                state.add("coordinate_source_enclosure", "unresolved", "mesh")
                return
            if any(
                _mapped_embedding.bernstein_node_count(value, "simplex", 3)
                > limits.maximum_bernstein_nodes
                for value in coordinates
            ):
                state.add("mapped_bernstein_node_budget", "unresolved", "mesh")
                return
            cells.append(
                _embedding_cells.Cell(
                    coordinates, "simplex", "tetrahedron", 3,
                    tuple(int(vertex) for vertex in vertices),
                    _embedding_cells.vertices("tetrahedron"),
                )
            )
    facet_traces = _mapped_embedding.trace_continuity(
        state, mesh, cells, geometry=geometry, budget=budget,
        retain_facet_traces=True,
    )
    if state.clean and not _boundary_degree_embedding.certify_boundary_degree_embedding(
        state, mesh, cells, cell_ids, limits, budget=budget,
        cad_boundary=(geometry, closure),
        facet_traces=facet_traces,
    ):
        state.add("mapped_boundary_degree", "unresolved", "mesh")


@checked
def certify_global_embedding(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    validity: CellValidityCertificate,
    /,
    *,
    limits: MeshCertificateLimits | None = None,
    junction_vertices: ArrayLike | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    predecessor: CellGeometryPredecessor | None = None,
    periodic_restriction: PeriodicRestrictionPremise | None = None,
    cad_boundary: CadBoundaryClosureEvidence | None = None,
) -> GlobalEmbeddingCertificate:
    """Certify that the mesh is globally embedded (see the module docstring).

    ``validity`` must certify the same coordinate arrays and topology; it is
    the positive-orientation hypothesis, not a substitute for this certificate.
    ``junction_vertices`` declares the vertex global ids at which an interval
    mesh may join more or fewer than two intervals (curve-network junctions);
    interval contacts beyond shared vertices are still refused there.
    ``predecessor`` authenticates request-owned unchanged-cell evidence. Affine
    contacts and source boundary facets reuse their per-contact evidence;
    changed contacts, broad-phase counts, and exterior degrees are recomputed.
    Mapped restrictions can import an embedded root union after coefficientwise
    source-map comparison, preserving their child-domain and child-contact proofs.
    Periodic proofs retain their full quotient routes unless an authenticated
    ``periodic_restriction`` premise admits the exact parent-partition theorem.
    ``cad_boundary`` imports only an independently issued whole-boundary CAD
    isotopy. The current full-cell positive Jacobians, exact skeleton conformity,
    facet pairing and exterior degrees are still proved here; fidelity status
    or a stored homotopy hash alone never authorizes this route.
    """

    # Mapped/periodic proof kernels consume affine certificate helpers; import their dispatch once per request.
    from . import (
        _mapped_embedding as mapped_embedding_owner,
        _periodic_embedding as periodic_embedding_owner,
    )

    validity.require_bound(geometry, mesh=mesh)
    limits_ = MeshCertificateLimits() if limits is None else limits
    junctions = _junction_vertices(mesh, junction_vertices)
    inherited = _contact_predecessor(
        predecessor, mesh, geometry, limits_, tuple(junctions.tolist()), budget=budget
    )
    restriction = _periodic_restriction_premise(periodic_restriction, mesh, budget=budget)
    cad_closure = cad_boundary
    if cad_closure is not None:
        from ._source_fidelity import require_issued_cad_boundary_closure

        require_issued_cad_boundary_closure(mesh, geometry, cad_closure, limits_)
    reuse_key = (
        _embedding_request_key(
            mesh, geometry, validity, limits_, tuple(junctions.tolist())
        )
        if budget is not None and mesh.storage is None
        else None
    )
    if cad_closure is not None and reuse_key is not None:
        reuse_key = canonical_fingerprint(
            {"embedding_request": reuse_key, "cad_boundary": cad_closure.evidence_id}
        )
    with coordinate_certificate_request(limits_, budget=budget) as budget:
        earned = (
            None
            if reuse_key is None
            else budget.global_embedding_certificate_cache.get(reuse_key)
        )
        if earned is not None:
            earned.binding.require(mesh, geometry)
            if earned.periodic_mapped_restriction is not None:
                PeriodicMappedRestrictionEvidence.from_stored_fields(
                    earned.periodic_mapped_restriction, budget=budget
                )
            return earned
        with ExitStack() as preparation:
            scope_resource: CoordinateEnclosureResourceError | None = None
            try:
                # Each refusal's quantities describe the phase that earned them.
                with budget.measurement():
                    preparation.enter_context(budget.activate())
                    preparation.enter_context(
                        budget.bound_stage(
                            limits_.maximum_work_units, limits_.maximum_scratch_bytes
                        )
                    )
                    prepared_scope = coordinate_scope(mesh, geometry, budget=budget)
                    scope = prepared_scope.coordinate_scope
                    prepared_source = prepared_scope.source_values
            except CoordinateEnclosureResourceError as error:
                scope = "mapped"
                prepared_source = ()
                scope_resource = error
            binding = MeshCertificateBinding(
                mesh,
                geometry,
                scope,
                limits_,
                junction_vertices=tuple(junctions.tolist()),
            )
            points = np.asarray(mesh.coordinates, dtype=np.float64)
            cell_ids = cell_global_ids(mesh)
            state = EmbeddingState([], [])
            _validity_findings(state, validity, cell_ids)
            dimensions = (mesh.topological_dimension, mesh.ambient_dimension)
            if scope_resource is not None:
                finding = (
                    "exact_source_resource_budget"
                    if geometry.exact_source is not None
                    else "mapped_source_expression_resource_budget"
                )
                state.add(
                    finding,
                    "unresolved",
                    "mesh",
                    resource_error=scope_resource,
                    expression_budget=budget,
                )
            elif not validity.all_certified:
                state.add(
                    "cell_validity_embedding_premise",
                    "unresolved",
                    "cell",
                    cell_ids[
                        np.asarray(validity.status) != CellValidityStatus.CERTIFIED_VALID
                    ],
                )
            elif mesh.storage is not None:
                state.add("owner_local_global_embedding_premise", "unresolved", "mesh")
            elif mesh.periodic_topology is not None:
                state.periodic_images = (
                    periodic_embedding_owner.certify_periodic_mapped_embedding(
                        state,
                        mesh,
                        geometry,
                        cell_ids,
                        limits_,
                        budget=budget,
                        validity=validity,
                        restriction=restriction,
                    )
                )
            elif cad_closure is not None:
                try:
                    with budget.measurement(), budget.activate(), budget.temporary_scope():
                        _certify_cad_boundary_embedding(
                            state, mesh, geometry, cell_ids, cad_closure, limits_,
                            budget=budget,
                        )
                except CoordinateEnclosureResourceError as error:
                    state.add(
                        "mapped_source_expression_resource_budget", "unresolved", "mesh",
                        resource_error=error, expression_budget=budget,
                    )
            elif geometry.exact_source is not None:
                try:
                    with budget.measurement(), budget.activate():
                        _exact_source_embedding(
                            state,
                            mesh,
                            geometry,
                            geometry.exact_source,
                            cell_ids,
                            junctions,
                            limits_,
                            scope,
                            prepared_source,
                            budget=budget,
                            mapped_certifier=mapped_embedding_owner.certify_mapped_embedding,
                            predecessor=inherited,
                            mapped_predecessor=predecessor,
                        )
                except CoordinateEnclosureResourceError as error:
                    state.add(
                        "exact_source_resource_budget",
                        "unresolved",
                        "mesh",
                        resource_error=error,
                        expression_budget=budget,
                    )
            elif scope == "mapped":
                try:
                    with budget.measurement():
                        facets = mesh_facets(mesh, geometry=geometry, budget=budget)
                        pairing_findings(state, mesh, facets, junctions)
                        with budget.activate():
                            mapped_embedding_owner.certify_mapped_embedding(
                                state,
                                mesh,
                                geometry,
                                cell_ids,
                                limits_,
                                budget=budget,
                                predecessor=predecessor,
                            )
                except CoordinateEnclosureResourceError as error:
                    state.add(
                        "mapped_source_expression_resource_budget",
                        "unresolved",
                        "mesh",
                        resource_error=error,
                        expression_budget=budget,
                    )
            else:
                try:
                    with budget.measurement():
                        facets = mesh_facets(mesh, geometry=geometry, budget=budget)
                        pairing_findings(state, mesh, facets, junctions)
                        _certify_affine_embedding(
                            state,
                            mesh,
                            points,
                            facets,
                            cell_ids,
                            limits_,
                            dimensions,
                            budget=budget,
                            predecessor=inherited,
                            geometry=geometry,
                            restriction=predecessor,
                        )
                except CoordinateEnclosureResourceError as error:
                    state.add(
                        "mapped_source_expression_resource_budget",
                        "unresolved",
                        "mesh",
                        resource_error=error,
                        expression_budget=budget,
                    )
            state.source_expression_work_units = budget.current_measurement.work_units
            state.source_expression_peak_bytes = (
                budget.current_measurement.peak_bytes_upper
            )
            certificate = GlobalEmbeddingCertificate(
                binding,
                validity.certificate_id,
                tuple(state.findings),
                tuple(state.checks),
                boundary_degree=state.boundary_degree,
                periodic_affine=state.periodic_affine,
                periodic_affine_degree=state.periodic_affine_degree,
                periodic_restriction=state.periodic_restriction,
                periodic_mapped_restriction=state.periodic_mapped_restriction,
                _periodic_mapped_budget=budget,
                restriction_root_embedding=(
                    None
                    if state.imported_root_embedding_certificate_id is None
                    else MappedRestrictionRootEmbeddingEvidence(
                        state.imported_root_embedding_certificate_id,
                        binding.binding_id,
                        validity.certificate_id,
                    )
                ),
                cad_boundary_closure=(
                    cad_closure
                    if "mapped_cad_boundary_homotopy_reuse" in state.checks
                    else None
                ),
                counts=EmbeddingCounts(
                    cell_count=cell_ids.size,
                    boundary_facet_count=state.boundary_facets,
                    shell_count=state.shells,
                    candidate_pair_count=state.candidate_pairs,
                    ray_test_count=state.ray_tests,
                    subdivision_piece_count=state.subdivision_pieces,
                    periodic_image_count=state.periodic_images,
                ),
                expression_work=EmbeddingExpressionWork(
                    work_units=state.source_expression_work_units,
                    peak_bytes=state.source_expression_peak_bytes,
                ),
            )
            if reuse_key is not None and certificate.status == "certified":
                budget.retain_basis((reuse_key,))
                budget.retain_owner(certificate)
                budget.global_embedding_certificate_cache[reuse_key] = certificate
            if predecessor is not None and _has_nested_partition_premises(certificate, predecessor):
                _ISSUED_NESTED_PARTITIONS.record(
                    budget,
                    _nested_partition_content_key(
                        mesh, geometry, predecessor, certificate, limits_, budget=budget
                    ),
                    certificate,
                )
            return certificate


__all__ = [
    "Boundary",
    "EmbeddingCounts",
    "EmbeddingExpressionWork",
    "GlobalEmbeddingCertificate",
    "MappedBoundaryDegreeEvidence",
    "MappedBoundaryDegreeStatus",
    "boundary_pieces",
    "boundary_shells",
    "cell_global_ids",
    "cell_loops",
    "certify_global_embedding",
    "coordinate_scope",
    "degenerate_simplices",
    "exact_pair_contacts",
    "exact_source_predicate_view",
    "mesh_orient2d",
    "mesh_orient3d",
    "pairing_findings",
    "pairwise_contacts",
    "ray_crossings_2d",
    "ray_crossings_3d",
    "shell_representatives",
    "source_bank_predicate_view",
    "surface_embedding",
    "volume_embedding",
]
