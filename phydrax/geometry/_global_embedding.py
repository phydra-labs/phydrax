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
        "statement": "Every rational source coordinate equals Fraction(float(source_coordinate))."
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
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from fractions import Fraction
from typing import assert_never, final, Literal

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

from .._fingerprint import canonical_fingerprint
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
)
from ._mesh_certificates import (
    certificate_status,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._mesh_check_names import MeshCertificateCheck
from ._polygon_triangulation import loop_triangles
from ._rational import dyadic_integers, rational_vector


type MappedBoundaryDegreeStatus = Literal["embedded", "violated", "premise_unproven"]


@final
class MappedBoundaryDegreeEvidence(StrictModule, NonTrainableState):
    """Premises and outcome of the boundary-degree theorem for mapped volume meshes.

    Boundary-degree statement. Let ``M`` be a mesh of triangles/quadrilaterals in ``R^2`` or of
    tetrahedra/prisms/hexahedra in ``R^3`` whose interior facets pair exactly
    two cells with opposite induced orientations, and let ``F`` be its piecewise
    polynomial coordinate map. Suppose

    (a) ``det DF_c > 0`` on every closed reference cell (the bound validity
        certificate) and every cell map is injective on its closed cell (the
        fixed-projection local injectivity proof);
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
    the sum of per-cell degrees, which is the winding number of the oriented
    boundary cycle because identical interior facet traces cancel. Every
    complement component of the embedded boundary borders some shell, the
    winding number jumps by exactly one into the side covered by the adjacent
    cell, and that side carries ``a_j + c_j = e_j + 1``; so (d) bounds the
    preimage count by one almost everywhere. A local-degree argument at
    interior points of ``|M|`` and (c) at boundary points turn this into
    injectivity. A single shell has ``a_j = 0`` and its covered side already
    carries degree at least one, so (d) holds without further work. Several
    shells take ``a_j`` from symbolic-ray winding counts of an affine proxy shell
    (vertex images, triangulated faces) joined to the curved shell by a
    straight homotopy whose certified deviation stays strictly away from the
    query point, and ``c_j`` from the sign of the rational signed shell measure.

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
    independent obligations of this certificate.
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
    restriction_root_embedding: MappedRestrictionRootEmbeddingEvidence | None
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
        restriction_root_embedding: MappedRestrictionRootEmbeddingEvidence | None = None,
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
        self.restriction_root_embedding = restriction_root_embedding
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
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[GlobalEmbeddingCertificate], value: GlobalEmbeddingCertificate, /
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
            restriction_root_embedding=restriction_root_embedding,
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


@final
class _CoordinateCellMap(StrictModule):
    """Rational corner shortcuts or source polynomials for one degree-one cell."""

    simplex_images: CoordinateSourceBank | None
    prepared_images: CoordinateSourceBank | None
    polynomials: tuple[Polynomial, ...]


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
    return _CoordinateScopeInputs(
        tuple(elements),
        tuple(np.asarray(route, dtype=np.int64) for route in host_routes),
        np.asarray(values, dtype=np.float64),
        np.asarray(points, dtype=np.float64),
        source_values,
        tuple(np.asarray(row, dtype=np.int64) for row in vertices),
        tuple(np.asarray(row, dtype=np.bool_) for row in valid),
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
    target_corners: dict[int, tuple[coordinate_enclosure_owner.ExactScalar, ...]],
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
        target_corners[int(vertex)] = image
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
) -> bool:
    if prepared.simplex_images is not None:
        if nonbinary_corner:
            match geometry.exact_source:
                case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
                    return False
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
    target_corners: dict[int, tuple[coordinate_enclosure_owner.ExactScalar, ...]] = {}
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
                reference = tuple(
                    rational_vector(point)
                    for point in reference_cell_topology(block.cell_kind).vertices
                )
                local_values = tuple(
                    tuple(inputs.source_values[index] for index in row) for row in route
                )
                for cell, local in enumerate(local_values):
                    # Only classification workspace is temporary; corner bindings remain live.
                    with budget.temporary_scope():
                        prepared = _prepare_coordinate_cell_map(
                            element, block.cell_kind, local, budget=budget
                        )
                        if prepared is None:
                            mapped.append(np.asarray((cursor + cell,), dtype=np.int64))
                            continue
                        nonbinary = _bind_coordinate_cell_corners(
                            prepared,
                            block.cell_kind,
                            vertices[cell],
                            reference,
                            inputs,
                            target_corners,
                            geometry,
                            budget=budget,
                        )
                        if _coordinate_cell_is_mapped(
                            prepared, block.cell_kind, nonbinary, geometry, budget=budget
                        ):
                            mapped.append(np.asarray((cursor + cell,), dtype=np.int64))
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
    target_corners: dict[int, tuple[coordinate_enclosure_owner.ExactScalar, ...]],
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
        source_values = tuple(
            target_corners[index] for index in range(inputs.points.shape[0])
        )
        budget.reserve(inputs.points.size, 128 + 8 * inputs.points.size)
        budget.retain_basis(source_values)
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


def _exact_box_charge(
    points: np.ndarray, count: int, /, *, budget: CoordinateEnclosureBudget
) -> tuple[int, int]:
    """Work and transient storage of ``boxes`` over ``count`` rational items.

    Per item twelve rational comparisons (each cross-multiplies two integers
    of at most ``2 b + 1`` bits, ``b`` the source bits) and six outward float
    conversions, with sixteen gathered slots; the bit scan charges itself.
    """
    return 18 * count, 2 * bigint_bytes(
        2 * exact_bits(points, budget=budget) + 1
    ) + 8 * 16 * count


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


@checked
def exact_pair_contacts(
    points: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide contact on the integer scaling of rational source coordinates.

    One positive common denominator preserves every orientation sign, so the
    predicate algorithm decides contact with exact predicates and signs that
    charge themselves. Per triangle pair it forms at most two projection
    normals (twenty visits, eighteen integers of at most ``2 b + 3`` bits for
    bank bits ``b``) and gathers at most 128 slots; a segment pair gathers
    eight slots.
    """

    with exact_charge(0, budget=budget):
        integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
            points, budget=budget
        )
        if first.shape[1] == 2:
            with exact_charge(0, 8 * 8 * first.shape[0], budget=budget):
                return _segment_contacts(integers, first, second, budget=budget)
        bits = exact_bits(integers, budget=budget)
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
) -> None:
    """Decide contacts of segment/triangle items with exact predicates beyond combinatorial sharing."""

    check = parse(check, MeshCertificateCheck, "pairwise_contacts.check")
    state.checks.append(check)
    exact = points.dtype == object
    with _exact_step(
        exact,
        *(_exact_box_charge(points, items.shape[0], budget=budget) if exact else (0, 0)),
        budget=budget,
    ):
        lower, upper = boxes(points, items)
    first, second, exceeded = candidate_pairs(
        lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
    )
    state.candidate_pairs += first.size
    if exceeded:
        state.add(f"{check}_capacity", "unresolved", "mesh")
    if source_rows is not None:
        pending = (source_rows[first] < 0) | (source_rows[second] < 0)
        first, second = first[pending], second[pending]
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
            hit[pending], certain[pending] = exact_pair_contacts(
                points, items[first[pending]], items[second[pending]], budget=budget
            )
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
        # Per piece the shell measure is one 14-visit integer determinant
        # below 2**(3 b + 3); boxes take twelve comparisons and fifteen slots;
        # a ray test takes five box comparisons per piece and, per crossed
        # candidate, fifteen perturbation visits with six differences below
        # 2**(b + 1) and forty slots. Orientation signs charge themselves.
        integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
            points, budget=budget
        )
        bits = exact_bits(integers, budget=budget)
        coordinates = integers
        exact_reserve(
            12 * pieces,
            8 * 15 * pieces + shells * bigint_bytes(3 * bits + 3 + pieces.bit_length()),
            budget=budget,
        )
    else:
        integers, _ = dyadic_integers(points)
        coordinates = points
        bits = 0
    with _exact_step(
        exact, 14 * pieces, 14 * pieces * bigint_bytes(3 * bits + 3), budget=budget
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
) -> None:
    boundary = boundary_pieces(state, mesh, points, facets, limits, budget=budget)
    inherited = (
        None
        if predecessor is None
        else _boundary_predecessor_rows(predecessor, facets, budget=budget)
    )
    pairwise_contacts(
        state,
        points,
        boundary.pieces,
        boundary.entities[boundary.piece_owners],
        "facet",
        "boundary_contact",
        limits,
        budget=budget,
        source_rows=(None if inherited is None else inherited[boundary.piece_owners]),
    )
    if not state.clean:
        # Degree theory requires the preceding boundary decisions.
        state.add("exterior_degree_premise", "unresolved", "mesh")
        return
    _exterior_degrees(
        state, points, boundary, mesh.topological_dimension, limits, budget=budget
    )


@checked
def exact_source_predicate_view(
    points: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, bool]:
    """Export binary64 only when it represents every authenticated source value.

    The binary predicates use exact predicates with adaptive/dyadic signs. This is an
    equality-certified execution view of the source bank, not its mesh carrier.
    A nonrepresentable value retains the source bank's rational predicate path.
    """
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

    facets = mesh_facets(mesh, budget=budget)
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
) -> CellGeometryPredecessor | None:
    """Authenticate imported evidence and select the affine contact reuse route."""
    if predecessor is None:
        return None
    source = predecessor.source_embedding
    if source is None or source.binding.coordinate_scope != "affine":
        return None
    predecessor.require_bound(mesh, geometry)
    source = GlobalEmbeddingCertificate.from_stored_fields(source)
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
            )
        case _:
            state.add("unsupported_embedding_dimension", "unresolved", "mesh")


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
    Periodic proofs retain their full quotient routes.
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
        predecessor, mesh, geometry, limits_, tuple(junctions.tolist())
    )
    with coordinate_certificate_request(limits_, budget=budget) as budget:
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
                    )
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
            return GlobalEmbeddingCertificate(
                binding,
                validity.certificate_id,
                tuple(state.findings),
                tuple(state.checks),
                boundary_degree=state.boundary_degree,
                periodic_affine=state.periodic_affine,
                restriction_root_embedding=(
                    None
                    if state.imported_root_embedding_certificate_id is None
                    else MappedRestrictionRootEmbeddingEvidence(
                        state.imported_root_embedding_certificate_id,
                        binding.binding_id,
                        validity.certificate_id,
                    )
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
    "surface_embedding",
    "volume_embedding",
]
