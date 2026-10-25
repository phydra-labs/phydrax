#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bound mesh/source boundary deviation with independent query evidence.

Host preparation retains source coefficient banks and bounded traversal workspaces.
Compiled implicit field batches execute in JAX and cross explicitly to the host
at the field and gradient batch boundaries. Curve representatives use NumPy plans.
Every resource-to-finding translation measures the guarded request phase.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "source_chart_authority_premise",
      "source_chart_facet_lineage_premise",
      "source_chart_orientation_premise",
      "source_chart_map_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The independently reconstructed native BRep authority, revision, single region and complete oriented patch/trim chain agree with the supplied source."
      },
      {
        "tag": "X",
        "statement": "Every source chart triangle matches exactly one target boundary facet, its corner coefficient values match the root vertex bank, and the authored region-side orientation agrees with the reference volume boundary orientation."
      },
      {
        "tag": "X",
        "statement": "Native endpoint/root, coedge correspondence and original admitted gap envelopes authenticate a continuous closed representative and oriented incidence ribbons from every original face; removable value and one-sided derivative limits of its curved-edge corrections are proved."
      }
    ],
    "conclusion": "An unsupported source map, foreign authority, incomplete facet lineage or inconsistent source side cannot establish a boundary-degree identity.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity._chart_boundary_homotopy",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_chart_homotopy_premise"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Outward native source and exact target polynomial intervals certify continuous deviation including original-face-to-representative ribbon bounds, rank-two projected Jacobians, edge-atlas monotonicity, vertex-cone separation and synchronized-time separation of every nonadjacent pair throughout the closed-boundary homotopy."
      },
      {
        "tag": "N",
        "statement": "A fixed separator on each vertex-pair time slab admits every closed reference leaf by strict one-sided directional derivatives or by a strict outward relative-image halfspace bound. Leaves containing the common vertex retain the derivative test; relative images subtract the actual completed moving vertex before time interpolation."
      },
      {
        "tag": "T",
        "statement": "Along a segment from the common vertex, derivative-certified leaves increase signed displacement and image-certified leaves prove it positive directly. Absolute continuity across the finite convex cover propagates strict positivity away from the vertex; the authenticated trace supplies the same moving vertex on both cells."
      },
      {
        "tag": "X",
        "statement": "Exact shared representative and target traces glue the complete oriented chart chain to the complete target boundary, without unidentified contacts or omitted authored patches; original nonzero incidences remain bounded ribbons under their unchanged import/model allowance."
      },
      {
        "tag": "T",
        "statement": "A continuous family of embeddings of a closed oriented boundary preserves its degree off the swept image; oriented incidence ribbons preserve the represented region side off their support. Complete oriented chart boundaries and nonzero chart Jacobians give degree-one coverage of the bounded trimmed parameter domains."
      }
    ],
    "conclusion": "Only the fully certified source-to-target isotopy establishes the two directed boundary bounds and the source-bound signed-degree identity.",
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
      "source_chart_homotopy_resource"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The shared coordinate/source budget refuses the chart map, exact trace or continuous homotopy phase at its original work or storage allowance."
      }
    ],
    "conclusion": "Retain the refusing resource phase and leave the boundary-degree proof unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity._chart_boundary_homotopy",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_trim_coverage_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "An explicitly supplied chart witness names every selected authored patch exactly once and independently certifies its oriented chart chain and complete trims."
      }
    ],
    "conclusion": "A closed-form boundary projection cannot discard an invalid supplied chart witness; the accepted witness and its continuous cover bind the degree identity.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity.analytic_boundary_proof",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "mesh_sample_capacity",
      "source_sample_capacity",
      "source_chart_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The proposed mesh lattice count, source sample count or chart simplex count is compared with maximum_source_samples before additional queries are issued."
      }
    ],
    "conclusion": "Refuse an incomplete sample/chart cover or a count exceeding the declared source sample allowance.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "distance_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Packed traversal adds each admitted node/item visit to completed distance evaluations; mapped reverse traversal starts at the measured forward evaluation count."
      }
    ],
    "conclusion": "Refuse the first proposed traversal visit or chart comparison count that exceeds maximum_distance_evaluations; retain the requested and completed counts.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_work_capacity",
      "source_scratch_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The proposed lattice evaluation count and explicit scratch bound are compared with maximum_work_units and maximum_scratch_bytes in that order after sample, distance and subdivision admission."
      }
    ],
    "conclusion": "Refuse work or storage exceeding the declared affine cover allowance.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "subdivision_capacity",
      "subdivision_depth",
      "bernstein_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Pending pieces, refinement depth and expression Bernstein-node count are compared with the matching declared maximum before expansion or evaluation."
      }
    ],
    "conclusion": "Refuse the continuous cover when its declared subdivision or Bernstein allowance is exceeded.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_affine_chain_resource_budget",
      "source_cover_expression_budget",
      "source_expression_resource_budget",
      "analytic_boundary_source_expression_resource",
      "coordinate_source_execution_error_budget"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The guarded coordinate/native operation runs inside budget.measurement; a refused admitted work/storage/deadline request retains its phase evidence."
      }
    ],
    "conclusion": "Report unresolved resource refusal without publishing finite replacement bounds.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "coordinate_source_expression",
      "analytic_boundary_source_expression_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each selected boundary cell has a supported reference topology and a retained coordinate expression over its full coefficient route."
      }
    ],
    "conclusion": "Refuse absent coordinate expressions or an analytic boundary map of the wrong dimension.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "curved_cell"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Certified polygon triangulation classifies every selected surface loop as planar and decided."
      }
    ],
    "conclusion": "Refuse a surface loop whose continuous planar simplex representation remains undecided.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "sampled_fidelity",
      "source_distance_semantics"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every reported directed distance semantics is certified; the boundary query result is checked before interpreting its bounds as a continuous enclosure."
      }
    ],
    "conclusion": "Sampled-only numerical distances cannot establish a certified source fidelity status.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "boundary_deviation"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Each certified mesh-boundary sample distance has directed lower/upper bounds; source distance is extended over its simplex by the cover radius and rounding enclosure."
      }
    ],
    "conclusion": "A lower deviation exceeding tolerance violates fidelity; an upper bound not proved within tolerance remains unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Metric-space distance to a set",
      "locator": "Triangle inequality for point-to-set distance",
      "url": "https://en.wikipedia.org/wiki/Distance#Distance_from_a_point_to_a_set"
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ]
  },
  {
    "checks": [
      "source_uncovered"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Source samples carry covering radii and source-point error; packed minima evaluate every unpruned candidate using outward objective bounds."
      }
    ],
    "conclusion": "A certified lower distance exceeding tolerance violates source coverage; a remaining upper distance exceeding tolerance is unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Hausdorff distance",
      "locator": "Directed point-to-set supremum and triangle inequality",
      "url": "https://en.wikipedia.org/wiki/Hausdorff_distance"
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ]
  },
  {
    "checks": [
      "mapped_source_boundary_equality_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Independent reference coverage and source/target embedding certificates are certified and share the source revision; retained source-coordinate restrictions partition the declared reference source."
      }
    ],
    "conclusion": "Certify zero deviation only when the recomputed domain coverage certificate establishes boundary equality; otherwise retain an unbounded reverse direction.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "owner_local_source_fidelity_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A whole-boundary source fidelity request has a global mesh rather than owner-local storage."
      }
    ],
    "conclusion": "Refuse a local storage closure as an unqualified whole-boundary source cover.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_source_chain_capacity",
      "analytic_boundary_chain_capacity",
      "analytic_boundary_piece_capacity",
      "analytic_boundary_bernstein_capacity",
      "analytic_boundary_ray_capacity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Independent source-chain pairs, target boundary facets, analytic refinement pieces, expression nodes and ray tests are compared with the corresponding declared maxima before expansion."
      }
    ],
    "conclusion": "Refuse the analytic projection when a source-chain, target-chain, subdivision, node or ray allowance is exceeded.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_volume_chain_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The selected target is a nonperiodic volume mesh with codimension-one boundary facets."
      }
    ],
    "conclusion": "Refuse an analytic closed-volume argument on a periodic carrier or a nonvolume mesh.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_polynomial_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every retained boundary coordinate expression is a polynomial rather than a rational quotient."
      }
    ],
    "conclusion": "Admit polynomial restriction cancellation and Bernstein projection bounds only for the polynomial chain.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_proxy_rounding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every rational proxy corner and projection center is exactly representable as finite binary64 before conversion to dyadic degree predicates."
      }
    ],
    "conclusion": "Refuse a proxy whose rounding could alter the degree predicate inputs.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_exact_closed_trace_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Oriented polynomial restrictions cancel pairwise on every boundary triangle edge, and no edge equals its reversal."
      }
    ],
    "conclusion": "The polynomial triangular chain has zero oriented boundary; failed cancellation remains unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_complete_loop_premise",
      "analytic_boundary_line_partition_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Rational oriented endpoints form nonbranching closed cycles; the polygon cycle partitions each independent source line into increasing gap-free intervals from zero to one."
      }
    ],
    "conclusion": "The target polygon cycle covers each independent source line with its declared orientation.",
    "citation": {
      "kind": "definition",
      "reference": "Source fidelity certificate contract",
      "locator": "phydrax.geometry._source_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_circle_oriented_degree_premise",
      "analytic_boundary_oriented_degree_one_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Rational ray crossing counts and oriented proxy edges establish the requested circle winding or closed-surface degree one."
      },
      {
        "tag": "X",
        "statement": "The independent analytic source is an oriented line/circle boundary or a closed cylinder gauge surface."
      },
      {
        "tag": "T",
        "statement": "A map of nonzero degree between the admitted closed oriented boundaries is surjective."
      }
    ],
    "conclusion": "The admitted proxy projection covers the independent analytic boundary with the declared signed degree.",
    "citation": {
      "kind": "mathematical",
      "reference": "Topological degree",
      "locator": "Nonzero degree implies surjectivity; degree of sphere maps",
      "url": "https://encyclopediaofmath.org/wiki/Topological_degree"
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_homotopy_premise"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Certified projection-coordinate intervals have positive dot product for radial homotopies; line tangential Bernstein controls stay between zero and one."
      },
      {
        "tag": "T",
        "statement": "Degree is invariant under a continuous homotopy that avoids the projection singularity."
      }
    ],
    "conclusion": "Transfer proxy projection degree to every point of the polynomial boundary map.",
    "citation": {
      "kind": "mathematical",
      "reference": "Topological degree",
      "locator": "Homotopy invariance",
      "url": "https://encyclopediaofmath.org/wiki/Topological_degree"
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "analytic_boundary_deviation",
      "analytic_boundary_projection_witness"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Outward radial or affine projection displacement bounds cover the polynomial simplex; rational interior evaluations independently refuse an insufficient displacement bound."
      }
    ],
    "conclusion": "An insufficient projection displacement bound remains unresolved rather than asserting a nearest-boundary lower distance.",
    "citation": {
      "kind": "mathematical",
      "reference": "Bernstein polynomial enclosure",
      "locator": "Convex-hull property of Bernstein coefficients and interval arithmetic",
      "url": "https://doi.org/10.1016/j.cagd.2012.03.001"
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "coordinate_source_execution_error"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "A continuous execution-carrier error enclosure is added outward to each directed upper bound and subtracted from each lower bound."
      }
    ],
    "conclusion": "Refuse source fidelity if the execution-carrier-adjusted directed upper bound exceeds tolerance.",
    "citation": {
      "kind": "mathematical",
      "reference": "Hausdorff distance",
      "locator": "Triangle inequality",
      "url": "https://en.wikipedia.org/wiki/Hausdorff_distance"
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
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

import phydrax._meshcore as meshcore_owner
import phydrax.discretization._cell_geometry_validity as cell_geometry_validity_owner
import phydrax.discretization._coordinate_budget as coordinate_budget_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.ein as ein
import phydrax.geometry._affine_chart_coverage as affine_chart_coverage_owner
import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._domain_chart_chain as domain_chart_chain_owner
import phydrax.geometry._domain_numerics as domain_numerics_owner
import phydrax.geometry._domain_planar_boundary as domain_planar_boundary_owner
import phydrax.geometry._domain_preparation as domain_preparation_owner
import phydrax.geometry._embedding_cells as embedding_cells_owner
import phydrax.geometry._exact_polyhedral_geometry as exact_polyhedral_geometry_owner
import phydrax.geometry._mapped_embedding as mapped_embedding_owner
import phydrax.geometry._mapped_reference_domain as mapped_reference_domain_owner
import phydrax.geometry._meshing_domain as meshing_domain_owner
import phydrax.geometry._radial_source_fidelity as radial_source_fidelity_owner
import phydrax.geometry._surface_source_support as surface_source_support_owner
import phydrax.geometry.brep._patches as patches_owner
import phydrax.geometry.implicit._adaptive_discovery as adaptive_discovery_owner
import phydrax.geometry.implicit._analytic_profile as analytic_profile_owner
from phydrax.discretization._coordinate_budget import RationalEnclosureError
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.geometry._rational import fraction_endpoint

from .._bvh import bvh_host_minima, prepare_bvh
from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    tree_content_equal,
)
from .._geometry_predicates import bigint_bytes
from .._host_storage import immutable_host_array
from .._issued_proofs import IssuedProofs
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_integer
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from ..discretization._coordinate_enclosure import (
    CoordinateCoefficients,
    Expression,
    Polynomial,
)
from ..discretization._coordinate_identity import coordinate_scope_key
from ..discretization._reference_cell import (
    reference_cell_topology,
    ReferenceCellTopology,
)
from ..typing import checked, Dim, HostFloat64, HostInt64, parse, Scope
from ._certificate import DistanceSemantics, SignReliability, ZeroSetAccuracy
from ._certified_implicit import implicit_state_id
from ._contracts import CompiledGeometry
from ._domain_coverage import (
    certify_domain_coverage,
    DomainCoverageCertificate,
    is_issued_domain_coverage,
    mapped_reference_proofs,
    PiecewiseLinearDomain,
)
from ._embedding_state import (
    coordinate_certificate_request,
    CoordinateMapScope,
    EmbeddingState,
    facet_entity_ids,
    mesh_facets,
    MeshCertificateEntityKind,
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from ._global_embedding import (
    boundary_pieces,
    cell_global_ids,
    cell_loops,
    certify_global_embedding,
    coordinate_scope,
    exact_pair_contacts,
    GlobalEmbeddingCertificate,
    ray_crossings_2d,
    ray_crossings_3d,
)
from ._mesh_certificates import (
    certificate_status,
    MeshCellDim,
    MeshCertificateBinding,
    MeshCertificateStatus,
)
from ._polygon_triangulation import loop_triangles
from ._rational import dyadic_integers, rational_vector
from ._source_boundary_records import (
    AffineChainResourceEvidence,
    AffineChainResourceMetric,
    CadBoundaryClosureEvidence,
    SourceAmbientDim,
    SourceBoundaryChartCover,
    SourceBoundaryChartQuery,
    SourceBoundaryDistance,
    SourceBoundaryQuery,
    SourceBoundarySamples,
    SourceBoundSemantics,
)
from .brep._analytic_source_regions import (
    complete_cylinder_region,
    CylinderSourceRegion,
    native_full_turn,
)
from .brep._carrier_pieces import admit_carrier_fingerprint, carrier_payload
from .brep._model import BRepGeometry, BRepModel
from .brep._patches import AbstractCurve


_EPSILON = float(np.finfo(np.float64).eps)


_DISTANCE_WORKING_ENTRIES = 1 << 22


class _CurveSourceDim(Dim, minimum=1):
    """Independently declared trimmed source carriers."""


class _SourceBoxCornerDim(Dim, minimum=2):
    """Lower and upper corners of a source enclosure."""


def _source_box_radius(box: np.ndarray, point: np.ndarray, /) -> float:
    squared = sum(
        max(
            abs(Fraction(float(low)) - Fraction(float(value))),
            abs(Fraction(float(high)) - Fraction(float(value))),
        )
        ** 2
        for low, high, value in zip(box[0], box[1], point, strict=True)
    )
    return float(
        np.nextafter(math.sqrt(float(np.nextafter(float(squared), math.inf))), math.inf)
    )


def _enclosed_source_distance(
    points: np.ndarray, source_boxes: np.ndarray, samples: SourceBoundarySamples, /
) -> SourceBoundaryDistance:
    """Return box lower bounds and evaluated source-point upper bounds."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != source_boxes.shape[2]:
        raise ValueError(
            f"SourceBoundaryQuery.points must have shape (N, {source_boxes.shape[2]}); got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"SourceBoundaryQuery.points must be finite; got {np.count_nonzero(~np.isfinite(values))} nonfinite values."
        )
    lower = np.zeros((values.shape[0],), dtype=np.float64)
    upper = np.full((values.shape[0],), math.inf, dtype=np.float64)
    chunk = max(
        1,
        _DISTANCE_WORKING_ENTRIES
        // max(source_boxes.shape[0], samples.points.shape[0], 1),
    )
    source_scale = max(
        float(np.max(np.abs(source_boxes))),
        float(np.max(np.abs(samples.points), initial=0.0)),
        1.0,
    )
    for start in range(0, values.shape[0], chunk):
        selected = values[start : start + chunk]
        scale = max(float(np.max(np.abs(selected), initial=0.0)), source_scale)
        slack = 64 * _EPSILON * scale
        offset = np.maximum(
            np.maximum(
                source_boxes[None, :, 0] - selected[:, None],
                selected[:, None] - source_boxes[None, :, 1],
            ),
            0.0,
        )
        lower[start : start + chunk] = np.maximum(
            np.min(np.linalg.norm(offset, axis=-1), axis=1) - slack, 0.0
        )
        if samples.points.shape[0]:
            distance = np.linalg.norm(selected[:, None] - samples.points[None], axis=-1)
            upper[start : start + chunk] = np.nextafter(
                np.min(distance + samples.point_error, axis=1) + slack, math.inf
            )
    return SourceBoundaryDistance(lower, upper, "certified")


@final
class ParametricCurveBoundarySource(StrictModule, NonTrainableState):
    """Query independently declared trimmed curves with continuous boundary bounds.

    Carrier boxes cover whole parameter intervals. Outward parameter brackets
    enclose representative-point evaluation error. Evaluated source points
    bound distances above; whole-interval boxes bound distances below.
    """

    __strict_contract__ = True

    curves: tuple[AbstractCurve, ...]
    parameter_ranges: tuple[tuple[float, float], ...] = eqx.field(static=True)
    boxes: HostFloat64[_CurveSourceDim, _SourceBoxCornerDim, SourceAmbientDim]
    samples: SourceBoundarySamples
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    covering_radius: float = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)

    def __init__(
        self,
        curves: tuple[AbstractCurve, ...],
        parameter_ranges: tuple[tuple[float, float], ...],
        /,
        *,
        source_id: str,
        source_revision: str,
        covering_radius: float = 0.01,
        maximum_samples: int = 4096,
    ) -> None:
        # General-container annotations do not establish their members' kinds.
        # Admit the carriers before checking range counts or reading geometry.
        match curves:
            case tuple():
                if not curves:
                    raise TypeError(
                        "ParametricCurveBoundarySource.curves must be nonempty; got ()."
                    )
            case invalid:
                raise TypeError(
                    "ParametricCurveBoundarySource.curves must be a nonempty tuple; "
                    f"got {type(invalid).__name__}."
                )
        for index, curve in enumerate(curves):
            match curve:
                case AbstractCurve():
                    continue
                case invalid:
                    raise TypeError(
                        "ParametricCurveBoundarySource.curves must contain AbstractCurve; "
                        f"got {type(invalid).__name__} at index {index}."
                    )
        if len(curves) != len(parameter_ranges):
            raise ValueError(
                f"ParametricCurveBoundarySource.parameter_ranges must match the curve count; got {len(parameter_ranges)} ranges for {len(curves)} curves."
            )
        dimension = curves[0].ambient_dimension
        if any(curve.ambient_dimension != dimension for curve in curves):
            raise ValueError(
                f"ParametricCurveBoundarySource.curves must share ambient dimension {dimension}; got {tuple(curve.ambient_dimension for curve in curves)}."
            )
        radius = float(covering_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"ParametricCurveBoundarySource.covering_radius must be finite and positive; got {covering_radius!r}."
            )
        budget = positive_integer(maximum_samples, "maximum_samples")
        ranges = tuple(
            curve.validate_range(*interval)
            for curve, interval in zip(curves, parameter_ranges, strict=True)
        )
        source_boxes = np.asarray(
            [
                curve.bounding_box(*interval)
                for curve, interval in zip(curves, ranges, strict=True)
            ],
            dtype=np.float64,
        )
        if source_boxes.shape != (len(curves), 2, dimension) or not np.all(
            np.isfinite(source_boxes)
        ):
            raise ValueError(
                f"ParametricCurveBoundarySource.boxes must be finite with shape ({len(curves)}, 2, {dimension}); got shape {source_boxes.shape}."
            )
        if np.any(source_boxes[:, 0] > source_boxes[:, 1]):
            raise ValueError(
                "ParametricCurveBoundarySource.boxes must have ordered lower and upper bounds; got a reversed interval."
            )
        points, radii, errors = [], [], []
        pending = [(i, first, last) for i, (first, last) in enumerate(ranges)]
        complete = True
        evaluations = 0
        plans = tuple(
            patches_owner.prepare_curve_host(curve, batch_capacity=1) for curve in curves
        )
        while pending:
            if evaluations >= budget:
                complete = False
                break
            i, first, last = pending.pop()
            middle = float((Fraction(first) + Fraction(last)) / 2)
            point = np.asarray(
                plans[i].evaluate_host_points(np.asarray(((middle,),), dtype=np.float64)),
                dtype=np.float64,
            )[0]
            box = np.asarray(curves[i].bounding_box(first, last), dtype=np.float64)
            anchor_first = max(first, float(np.nextafter(middle, -math.inf)))
            anchor_last = min(last, float(np.nextafter(middle, math.inf)))
            error_box = np.asarray(
                curves[i].bounding_box(anchor_first, anchor_last), dtype=np.float64
            )
            error = _source_box_radius(error_box, point)
            bound = _source_box_radius(box, point)
            evaluations += 1
            if bound > radius and first < middle < last:
                pending.extend(((i, first, middle), (i, middle, last)))
                continue
            complete = complete and bound <= radius
            points.append(point)
            radii.append(bound)
            errors.append(error)
        self.curves = curves
        self.parameter_ranges = ranges
        self.boxes = parse(
            source_boxes,
            HostFloat64[_CurveSourceDim, _SourceBoxCornerDim, SourceAmbientDim],
            "boxes",
        )
        self.samples = SourceBoundarySamples(
            np.asarray(points, dtype=np.float64).reshape(-1, dimension),
            np.asarray(radii, dtype=np.float64),
            np.asarray(errors, dtype=np.float64),
            "certified",
            complete=complete,
        )
        self.source_id = canonical_identifier(source_id, "source_id")
        self.source_revision = canonical_identifier(source_revision, "source_revision")
        self.covering_radius = radius
        self.maximum_samples = budget

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ParametricCurveBoundarySource],
        value: ParametricCurveBoundarySource,
        /,
    ) -> ParametricCurveBoundarySource:
        """Rebuild curve enclosures and samples from the stored carrier inputs."""
        return cls(
            value.curves,
            value.parameter_ranges,
            source_id=value.source_id,
            source_revision=value.source_revision,
            covering_radius=value.covering_radius,
            maximum_samples=value.maximum_samples,
        )

    @property
    def ambient_dimension(self) -> int:
        return self.curves[0].ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return _enclosed_source_distance(points, self.boxes, self.samples)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        budget = positive_integer(maximum_samples, "maximum_samples")
        count = self.samples.points.shape[0]
        return SourceBoundarySamples(
            self.samples.points[:budget],
            self.samples.covering_radius[:budget],
            self.samples.point_error[:budget],
            "certified",
            complete=self.samples.complete and count <= budget,
        )


@final
class MappedDomainBoundarySource(StrictModule, NonTrainableState):
    """Query declared mapped source images through retained coordinate expressions.

    Boundary equality requires independent reference coverage, source/target
    embedding and a gap-free partition of source restrictions. Point queries
    use coordinate-expression enclosures over every selected source piece.
    """

    __strict_contract__ = True

    domain: mapped_reference_domain_owner.MappedReferenceDomain
    cell_regions: HostInt64[MeshCellDim]
    limits: MeshCertificateLimits
    covering_radius: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: mapped_reference_domain_owner.MappedReferenceDomain,
        cell_regions: ArrayLike,
        /,
        *,
        limits: MeshCertificateLimits | None = None,
        covering_radius: float = 0.01,
    ) -> None:

        limits_ = domain.certificate_limits if limits is None else limits
        if not tree_content_equal(limits_, domain.certificate_limits):
            raise ValueError(
                "MappedDomainBoundarySource.limits must match domain.certificate_limits"
            )
        radius = float(covering_radius)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(
                f"MappedDomainBoundarySource.covering_radius must be finite and positive; got {covering_radius!r}."
            )
        regions = parse(
            np.asarray(cell_regions, dtype=np.int64),
            HostInt64[MeshCellDim],
            "cell_regions",
        )
        if np.any(regions < 0) or np.any(regions >= len(domain.region_ids)):
            raise ValueError(
                f"MappedDomainBoundarySource.cell_regions must index {len(domain.region_ids)} source regions; got an out-of-range index."
            )
        self.domain = domain
        self.cell_regions = regions
        self.limits = limits_
        self.covering_radius = radius

    def for_target_regions(self, regions: ArrayLike, /) -> MappedDomainBoundarySource:
        """Renew an explicit target assignment without changing the source domain."""
        return MappedDomainBoundarySource(
            self.domain,
            regions,
            limits=self.limits,
            covering_radius=self.covering_radius,
        )

    @property
    def source_id(self) -> str:
        return self.domain.source_id

    @property
    def source_revision(self) -> str:
        return self.domain.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.domain.ambient_dimension

    def boundary_samples(
        self, maximum_samples: int, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> SourceBoundarySamples:
        with coordinate_certificate_request(self.limits, budget=budget) as budget:
            maps, _ = _mapped_domain_source_maps(self, budget=budget)
            return _mapped_domain_source_samples(
                maps,
                self.ambient_dimension,
                self.covering_radius,
                positive_integer(maximum_samples, "maximum_samples"),
                self.limits,
                budget=budget,
            )

    def boundary_distance(
        self, points: np.ndarray, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> SourceBoundaryDistance:
        with coordinate_certificate_request(self.limits, budget=budget) as budget:
            maps, boxes = _mapped_domain_source_maps(self, budget=budget)
            samples = _mapped_domain_source_samples(
                maps,
                self.ambient_dimension,
                self.covering_radius,
                self.limits.maximum_source_samples,
                self.limits,
                budget=budget,
            )
            return _enclosed_source_distance(points, boxes, samples)

    def boundary_coverage(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        limits: MeshCertificateLimits,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> DomainCoverageCertificate:

        # Transformed/restored module fields must retain their source owner.
        match self.domain:
            case mapped_reference_domain_owner.MappedReferenceDomain() as domain:
                validity = cell_geometry_validity_owner.certify_cell_geometry_validity(
                    geometry, mesh=mesh, budget=budget
                )
                embedding = certify_global_embedding(
                    mesh, geometry, validity, limits=limits, budget=budget
                )
                return certify_domain_coverage(
                    mesh,
                    geometry,
                    domain,
                    self.cell_regions,
                    limits=limits,
                    embedding=embedding,
                    budget=budget,
                )
            case invalid:
                raise TypeError(
                    f"MappedDomainBoundarySource.domain must retain MappedReferenceDomain; got {type(invalid).__name__}."
                )


class _ProjectionFacetDim(Dim):
    """Facets of an independently checked closed source projection."""


class _ProjectionPieceDim(Dim):
    """Continuously enclosed facet subpieces."""


class _ProjectionFiberCandidateDim(Dim):
    """Candidate facets tested by a rational source normal fiber."""


ProjectionFiberCrossingKind: TypeAlias = Literal[
    "disjoint", "interior", "edge", "vertex", "coplanar"
]


@final
class ProjectionFacetEvidence(StrictModule, NonTrainableState):
    """Facet geometry, subdivisions and continuous source bounds."""

    facets: np.ndarray
    facet_vertices: np.ndarray
    facet_ids: np.ndarray
    facet_components: np.ndarray
    piece_triangles: np.ndarray
    piece_barycentric_vertices: np.ndarray
    piece_coordinate_bounds: np.ndarray
    piece_owners: np.ndarray
    field_bounds: np.ndarray
    gradient_bounds: np.ndarray
    normal_bounds: np.ndarray


@final
class ProjectionFiberEvidence(StrictModule, NonTrainableState):
    """Rational source fiber and classified candidate facet crossings."""

    fiber_endpoints: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    fiber_value_bounds: np.ndarray
    fiber_gradient_bounds: np.ndarray
    fiber_candidate_facets: np.ndarray
    fiber_crossing_classes: tuple[ProjectionFiberCrossingKind, ...] = eqx.field(
        static=True
    )
    fiber_crossing_parameters: tuple[tuple[int, int] | None, ...] = eqx.field(static=True)


@final
class ImplicitProjectionCoverageEvidence(StrictModule, NonTrainableState):
    """Retain source-profile, tube, normal and degree-one fiber evidence.

    Source fidelity recomputes these bounds from the geometry, topology and
    request limits. Fiber coordinates and crossing parameters retain rational
    pairs so projection coverage is independent of rounded source anchors.
    """

    __strict_contract__ = True

    binding: MeshCertificateBinding
    profile_id: str = eqx.field(static=True)
    source_state_id: str = eqx.field(static=True)
    source_schema_id: str = eqx.field(static=True)
    source_kernel_id: str = eqx.field(static=True)
    source_physical_id: str = eqx.field(static=True)
    source_family: str = eqx.field(static=True)
    source_center: HostFloat64[Literal[3]]
    source_radius: float = eqx.field(static=True)
    source_major_radius: float = eqx.field(static=True)
    source_reach_lower: float = eqx.field(static=True)
    source_tube_radius: float = eqx.field(static=True)
    source_hessian_upper: float = eqx.field(static=True)
    source_normal_lipschitz_upper: float = eqx.field(static=True)
    source_component_count: int = eqx.field(static=True)
    source_genus: int = eqx.field(static=True)
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    facets: HostFloat64[_ProjectionFacetDim, Literal[3], Literal[3]]
    facet_vertices: HostInt64[_ProjectionFacetDim, Literal[3]]
    facet_ids: HostInt64[_ProjectionFacetDim]
    facet_components: HostInt64[_ProjectionFacetDim]
    piece_triangles: HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]]
    piece_barycentric_vertices: HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]]
    piece_coordinate_bounds: HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]]
    piece_owners: HostInt64[_ProjectionPieceDim]
    field_bounds: HostFloat64[_ProjectionPieceDim, Literal[2]]
    gradient_bounds: HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]]
    normal_bounds: HostFloat64[_ProjectionPieceDim, Literal[2]]
    fiber_endpoints: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    fiber_value_bounds: HostFloat64[Literal[2], Literal[2]]
    fiber_gradient_bounds: HostFloat64[Literal[2]]
    fiber_candidate_facets: HostInt64[_ProjectionFiberCandidateDim]
    fiber_crossing_classes: tuple[ProjectionFiberCrossingKind, ...] = eqx.field(
        static=True
    )
    fiber_crossing_parameters: tuple[tuple[int, int] | None, ...] = eqx.field(static=True)
    forward_upper: float = eqx.field(static=True)
    forward_lower: float = eqx.field(static=True)
    query_count: int = eqx.field(static=True)
    ray_test_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        profile: analytic_profile_owner.AnalyticImplicitProfile,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        facet_evidence: ProjectionFacetEvidence,
        boundary_fiber: ProjectionFiberEvidence,
        forward_upper: float,
        forward_lower: float,
        query_count: int,
        ray_test_count: int,
    ) -> None:

        profile.require_bound(profile.geometry, profile.coordinate_contract)
        if (binding.source_id, binding.source_revision) != (
            profile.source_id,
            profile.source_revision,
        ):
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.binding must name source {(profile.source_id, profile.source_revision)!r}; got {(binding.source_id, binding.source_revision)!r}."
            )
        if not (float(forward_upper) >= float(forward_lower) >= 0.0):
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.forward_upper must be at least forward_lower >= 0; got {(forward_upper, forward_lower)!r}."
            )
        if query_count < 0 or ray_test_count < 0:
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.work_counts must be nonnegative; got {(query_count, ray_test_count)!r}."
            )
        facets = facet_evidence.facets
        facet_vertices = facet_evidence.facet_vertices
        facet_ids = facet_evidence.facet_ids
        facet_components = facet_evidence.facet_components
        piece_triangles = facet_evidence.piece_triangles
        piece_barycentric_vertices = facet_evidence.piece_barycentric_vertices
        piece_coordinate_bounds = facet_evidence.piece_coordinate_bounds
        piece_owners = facet_evidence.piece_owners
        field_bounds = facet_evidence.field_bounds
        gradient_bounds = facet_evidence.gradient_bounds
        normal_bounds = facet_evidence.normal_bounds
        fiber_endpoints = boundary_fiber.fiber_endpoints
        fiber_value_bounds = boundary_fiber.fiber_value_bounds
        fiber_gradient_bounds = boundary_fiber.fiber_gradient_bounds
        fiber_candidate_facets = boundary_fiber.fiber_candidate_facets
        fiber_crossing_classes = boundary_fiber.fiber_crossing_classes
        fiber_crossing_parameters = boundary_fiber.fiber_crossing_parameters
        if fiber_endpoints and (
            len(fiber_endpoints) != 2
            or any(len(point) != 3 for point in fiber_endpoints)
            or any(
                denominator <= 0 for point in fiber_endpoints for _, denominator in point
            )
        ):
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.fiber_endpoints must contain two rational three-dimensional endpoints with positive denominators; got {fiber_endpoints!r}."
            )
        count = fiber_candidate_facets.shape[0]
        if (
            len(fiber_crossing_classes) != count
            or len(fiber_crossing_parameters) != count
        ):
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.fiber_crossings must match {count} candidate facets; got {len(fiber_crossing_classes)} classes and {len(fiber_crossing_parameters)} parameters."
            )
        classes = tuple(
            parse(kind, ProjectionFiberCrossingKind, "fiber_crossing_class")
            for kind in fiber_crossing_classes
        )
        scope = Scope()
        self.binding = binding
        self.profile_id = profile.profile_id
        self.source_state_id = profile.state_id
        self.source_schema_id = profile.schema_id
        self.source_kernel_id = profile.kernel_id
        self.source_physical_id = profile.coordinate_contract.spatial_id
        self.source_family = profile.family
        self.source_center = parse(
            profile.center, HostFloat64[Literal[3]], "source_center"
        )
        self.source_radius = profile.radius
        self.source_major_radius = profile.major_radius
        self.source_reach_lower = profile.reach_lower
        self.source_tube_radius = profile.tube_radius
        self.source_hessian_upper = profile.hessian_upper
        self.source_normal_lipschitz_upper = profile.normal_lipschitz_upper
        self.source_component_count = profile.component_count
        self.source_genus = profile.genus
        self.findings = tuple(findings)
        self.status = certificate_status(self.findings)
        self.facets = parse(
            facets,
            HostFloat64[_ProjectionFacetDim, Literal[3], Literal[3]],
            "facets",
            scope=scope,
        )
        self.facet_vertices = parse(
            facet_vertices,
            HostInt64[_ProjectionFacetDim, Literal[3]],
            "facet_vertices",
            scope=scope,
        )
        self.facet_ids = parse(
            facet_ids, HostInt64[_ProjectionFacetDim], "facet_ids", scope=scope
        )
        self.facet_components = parse(
            facet_components,
            HostInt64[_ProjectionFacetDim],
            "facet_components",
            scope=scope,
        )
        self.piece_triangles = parse(
            piece_triangles,
            HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]],
            "piece_triangles",
            scope=scope,
        )
        self.piece_barycentric_vertices = parse(
            piece_barycentric_vertices,
            HostFloat64[_ProjectionPieceDim, Literal[3], Literal[3]],
            "piece_barycentric_vertices",
            scope=scope,
        )
        self.piece_coordinate_bounds = parse(
            piece_coordinate_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]],
            "piece_coordinate_bounds",
            scope=scope,
        )
        self.piece_owners = parse(
            piece_owners, HostInt64[_ProjectionPieceDim], "piece_owners", scope=scope
        )
        self.field_bounds = parse(
            field_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2]],
            "field_bounds",
            scope=scope,
        )
        self.gradient_bounds = parse(
            gradient_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2], Literal[3]],
            "gradient_bounds",
            scope=scope,
        )
        self.normal_bounds = parse(
            normal_bounds,
            HostFloat64[_ProjectionPieceDim, Literal[2]],
            "normal_bounds",
            scope=scope,
        )
        self.fiber_endpoints = fiber_endpoints
        self.fiber_value_bounds = parse(
            fiber_value_bounds, HostFloat64[Literal[2], Literal[2]], "fiber_value_bounds"
        )
        self.fiber_gradient_bounds = parse(
            fiber_gradient_bounds, HostFloat64[Literal[2]], "fiber_gradient_bounds"
        )
        self.fiber_candidate_facets = parse(
            fiber_candidate_facets,
            HostInt64[_ProjectionFiberCandidateDim],
            "fiber_candidate_facets",
        )
        # These validated proof operands are immutable numerical banks, not
        # mutable producer views. Preserve their host representation and exact
        # bytes while allowing publication to authenticate the actual backing.
        self.source_center = immutable_host_array(self.source_center)
        self.facets = immutable_host_array(self.facets)
        self.facet_vertices = immutable_host_array(self.facet_vertices)
        self.facet_ids = immutable_host_array(self.facet_ids)
        self.facet_components = immutable_host_array(self.facet_components)
        self.piece_triangles = immutable_host_array(self.piece_triangles)
        self.piece_barycentric_vertices = immutable_host_array(self.piece_barycentric_vertices)
        self.piece_coordinate_bounds = immutable_host_array(self.piece_coordinate_bounds)
        self.piece_owners = immutable_host_array(self.piece_owners)
        self.field_bounds = immutable_host_array(self.field_bounds)
        self.gradient_bounds = immutable_host_array(self.gradient_bounds)
        self.normal_bounds = immutable_host_array(self.normal_bounds)
        self.fiber_value_bounds = immutable_host_array(self.fiber_value_bounds)
        self.fiber_gradient_bounds = immutable_host_array(self.fiber_gradient_bounds)
        self.fiber_candidate_facets = immutable_host_array(self.fiber_candidate_facets)
        self.fiber_crossing_classes = classes
        self.fiber_crossing_parameters = fiber_crossing_parameters
        self.forward_upper, self.forward_lower = (
            float(forward_upper),
            float(forward_lower),
        )
        self.query_count, self.ray_test_count = query_count, ray_test_count
        self.evidence_id = _projection_evidence_identity(
            self,
            fiber_endpoints,
            classes,
            fiber_crossing_parameters,
            query_count,
            ray_test_count,
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ImplicitProjectionCoverageEvidence],
        value: ImplicitProjectionCoverageEvidence,
        /,
        *,
        profile: analytic_profile_owner.AnalyticImplicitProfile,
    ) -> ImplicitProjectionCoverageEvidence:
        """Rebuild stored facet and fiber banks against the retained source profile."""
        if value.profile_id != profile.profile_id:
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.profile_id must match retained profile {profile.profile_id!r}; got {value.profile_id!r}."
            )
        if (value.binding.source_id, value.binding.source_revision) != (
            profile.source_id,
            profile.source_revision,
        ):
            raise ValueError(
                f"ImplicitProjectionCoverageEvidence.binding must name retained source {(profile.source_id, profile.source_revision)!r}; got {(value.binding.source_id, value.binding.source_revision)!r}."
            )
        renewed = cls(
            value.binding,
            profile,
            value.findings,
            facet_evidence=ProjectionFacetEvidence(
                facets=value.facets,
                facet_vertices=value.facet_vertices,
                facet_ids=value.facet_ids,
                facet_components=value.facet_components,
                piece_triangles=value.piece_triangles,
                piece_barycentric_vertices=value.piece_barycentric_vertices,
                piece_coordinate_bounds=value.piece_coordinate_bounds,
                piece_owners=value.piece_owners,
                field_bounds=value.field_bounds,
                gradient_bounds=value.gradient_bounds,
                normal_bounds=value.normal_bounds,
            ),
            boundary_fiber=ProjectionFiberEvidence(
                fiber_endpoints=value.fiber_endpoints,
                fiber_value_bounds=value.fiber_value_bounds,
                fiber_gradient_bounds=value.fiber_gradient_bounds,
                fiber_candidate_facets=value.fiber_candidate_facets,
                fiber_crossing_classes=value.fiber_crossing_classes,
                fiber_crossing_parameters=value.fiber_crossing_parameters,
            ),
            forward_upper=value.forward_upper,
            forward_lower=value.forward_lower,
            query_count=value.query_count,
            ray_test_count=value.ray_test_count,
        )
        if not tree_content_equal(renewed, value):
            raise ValueError(
                "ImplicitProjectionCoverageEvidence.stored_fields must reproduce retained projection evidence bitwise; got changed stored fields."
            )
        return renewed


def _projection_evidence_identity(
    evidence: ImplicitProjectionCoverageEvidence,
    fiber_endpoints: tuple[tuple[tuple[int, int], ...], ...],
    classes: tuple[ProjectionFiberCrossingKind, ...],
    fiber_crossing_parameters: tuple[tuple[int, int] | None, ...],
    query_count: int,
    ray_test_count: int,
    /,
) -> str:
    """Bind projection source facts, continuous bounds, fiber crossings and work."""
    return canonical_fingerprint(
        {
            "kind": "implicit-projection-coverage-evidence",
            "binding": evidence.binding.binding_id,
            "profile": evidence.profile_id,
            "source_facts": (
                evidence.source_state_id,
                evidence.source_schema_id,
                evidence.source_kernel_id,
                evidence.source_physical_id,
                evidence.source_family,
                evidence.source_radius,
                evidence.source_major_radius,
                evidence.source_reach_lower,
                evidence.source_tube_radius,
                evidence.source_hessian_upper,
                evidence.source_normal_lipschitz_upper,
                evidence.source_component_count,
                evidence.source_genus,
            ),
            "findings": tuple(finding.finding_id for finding in evidence.findings),
            "arrays": array_tree_fingerprint(
                (
                    evidence.facets,
                    evidence.facet_vertices,
                    evidence.facet_ids,
                    evidence.facet_components,
                    evidence.source_center,
                    evidence.piece_barycentric_vertices,
                    evidence.piece_coordinate_bounds,
                    evidence.piece_triangles,
                    evidence.piece_owners,
                    evidence.field_bounds,
                    evidence.gradient_bounds,
                    evidence.normal_bounds,
                    evidence.fiber_value_bounds,
                    evidence.fiber_gradient_bounds,
                    evidence.fiber_candidate_facets,
                )
            ),
            "fiber": fiber_endpoints,
            "crossing_classes": classes,
            "crossing_parameters": fiber_crossing_parameters,
            "forward": (repr(evidence.forward_upper), repr(evidence.forward_lower)),
            "work": (query_count, ray_test_count),
        }
    )


@final
class ImplicitProjectionBoundarySource(StrictModule, NonTrainableState):
    """Nominal analytic source with independently recomputed projection coverage."""

    profile: analytic_profile_owner.AnalyticImplicitProfile
    discovery: adaptive_discovery_owner.ImplicitVolumeQuery | None

    @checked
    def __init__(
        self,
        profile: analytic_profile_owner.AnalyticImplicitProfile,
        /,
        *,
        discovery: adaptive_discovery_owner.ImplicitVolumeQuery | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:

        profile.require_bound(
            profile.geometry, profile.coordinate_contract, budget=budget
        )
        self.profile = profile
        if discovery is not None:
            discovery.require_bound(profile.geometry, budget=budget)
            domain = np.asarray(discovery.domain)
            if np.any(domain[0] > profile.boundary_bounds[0]) or np.any(
                domain[1] < profile.boundary_bounds[1]
            ):
                raise ValueError(
                    "ImplicitProjectionBoundarySource.discovery must enclose the complete analytic boundary"
                )
            if budget is not None:
                budget.retain_basis((discovery,))
        self.discovery = discovery

    @classmethod
    @checked
    def from_stored_fields(
        cls, value: ImplicitProjectionBoundarySource, /
    ) -> ImplicitProjectionBoundarySource:
        return cls(value.profile, discovery=value.discovery)

    @property
    def source_id(self) -> str:
        return self.profile.source_id

    @property
    def source_revision(self) -> str:
        return self.profile.source_revision

    @property
    def ambient_dimension(self) -> int:
        return self.profile.ambient_dimension

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        return self.profile.boundary_distance(points)

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        return self.profile.boundary_samples(maximum_samples)

    def projection_coverage(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        /,
        *,
        tolerance: float,
        limits: MeshCertificateLimits,
        budget: CoordinateEnclosureBudget,
    ) -> ImplicitProjectionCoverageEvidence:

        # The projection algorithm consumes this evidence type; load it once at the source-fidelity request.
        import phydrax.geometry.implicit._projection_cover as projection_cover_owner

        # Transformed/restored module fields must retain their source owner.
        match self.profile:
            case analytic_profile_owner.AnalyticImplicitProfile() as profile:
                profile.require_bound(profile.geometry, profile.coordinate_contract)
                return projection_cover_owner.compute_implicit_projection_cover(
                    mesh,
                    geometry,
                    profile,
                    tolerance=tolerance,
                    limits=limits,
                    budget=budget,
                )
            case invalid:
                raise TypeError(
                    f"ImplicitProjectionBoundarySource.profile must retain AnalyticImplicitProfile; got {type(invalid).__name__}."
                )


@final
class ImplicitBoundarySource(StrictModule):
    """Boundary queries of a compiled implicit geometry at one design state.

    Distance bounds are ``certified`` only for a globally reliable signed
    distance whose evaluation error is established by its owner; the boundary
    is then sampled on a grid of ``spacing`` over the kernel bounds, which are
    the declared enclosure of the boundary. Other fields report ``sampled``
    first-order distance estimates ``|phi| / |grad phi|``.
    """

    geometry: CompiledGeometry
    spacing: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)

    @checked
    def __init__(
        self, geometry: CompiledGeometry, /, *, source_id: str, spacing: float
    ) -> None:
        step = float(spacing)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(
                f"ImplicitBoundarySource.spacing must be finite and positive; got {spacing!r}."
            )
        self.geometry = geometry
        self.spacing = step
        self.source_id = canonical_identifier(source_id, "source_id")
        self.source_revision = implicit_state_id(geometry)

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[ImplicitBoundarySource], value: ImplicitBoundarySource, /
    ) -> ImplicitBoundarySource:
        """Rebuild an implicit source after validating its numerical geometry state."""
        accepted = np.asarray(jax.device_get(value.geometry.validity().accepted))
        if not bool(accepted):
            raise ValueError(
                "ImplicitBoundarySource.geometry must have accepted validity; "
                f"got accepted={accepted}."
            )
        return cls(value.geometry, source_id=value.source_id, spacing=value.spacing)

    @property
    def ambient_dimension(self) -> int:
        return self.geometry.ambient_dimension

    @property
    def certified_distance(self) -> bool:
        certificate = self.geometry.field_certificate
        return (
            certificate.distance_semantics is DistanceSemantics.EXACT
            and certificate.sign_reliability is SignReliability.RELIABLE
            and certificate.zero_set_accuracy is ZeroSetAccuracy.EXACT
            and certificate.validity_region == "all_space"
            and certificate.evaluation_error is not None
            and certificate.bound_origin == "established"
        )

    def _field(self, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
        values = jnp.asarray(points, dtype=jnp.float64)

        def scalar(point: jax.Array) -> jax.Array:
            return self.geometry.boundary_field(point[None])[0]

        field = np.asarray(
            jax.device_get(self.geometry.boundary_field(values)), dtype=np.float64
        )
        gradient = np.asarray(
            jax.device_get(jax.vmap(jax.grad(scalar))(values)), dtype=np.float64
        )
        return field.reshape(-1), gradient

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        field, gradient = self._field(points)
        if self.certified_distance:
            error = float(self.geometry.field_certificate.evaluation_error or 0.0)
            magnitude = np.abs(field)
            return SourceBoundaryDistance(
                np.maximum(magnitude - error, 0.0), magnitude + error, "certified"
            )
        norm = np.maximum(np.linalg.norm(gradient, axis=-1), np.finfo(np.float64).tiny)
        estimate = np.abs(field) / norm
        return SourceBoundaryDistance(estimate, estimate, "sampled")

    def boundary_samples(self, maximum_samples: int, /) -> SourceBoundarySamples:
        bounds = np.asarray(
            jax.device_get(self.geometry.bounds), dtype=np.float64
        ).reshape(2, -1)
        counts = np.floor((bounds[1] - bounds[0]) / self.spacing).astype(np.int64) + 2
        dimension = bounds.shape[1]
        if math.prod(int(value) for value in counts) > maximum_samples:
            empty = np.empty((0,), dtype=np.float64)
            return SourceBoundarySamples(
                np.empty((0, dimension)), empty, empty, "sampled", complete=False
            )
        axes = tuple(
            bounds[0, axis] + self.spacing * np.arange(counts[axis], dtype=np.float64)
            for axis in range(dimension)
        )
        grid = np.stack(
            [value.reshape(-1) for value in np.meshgrid(*axes, indexing="ij")], axis=-1
        )
        field, gradient = self._field(grid)
        certificate = self.geometry.field_certificate
        certified = self.certified_distance
        error = float(certificate.evaluation_error or 0.0) if certified else 0.0
        lipschitz = certificate.lipschitz_upper_bound
        half_diagonal = 0.5 * self.spacing * math.sqrt(dimension)
        # A boundary point y has a nearest node x with |x - y| <= half_diagonal,
        # so |phi(x)| <= L half_diagonal + error keeps every such node.
        band = (lipschitz if lipschitz is not None else 2.0) * half_diagonal + error
        keep = np.abs(field) <= band * (1.0 + 16.0 * _EPSILON)
        squared = np.maximum(
            np.sum(gradient[keep] ** 2, axis=-1), np.finfo(np.float64).tiny
        )
        projected = grid[keep] - (field[keep] / squared)[:, None] * gradient[keep]
        step = np.linalg.norm(grid[keep] - projected, axis=-1)
        residual, _ = self._field(projected)
        return SourceBoundarySamples(
            projected,
            (half_diagonal + step) * (1.0 + 16.0 * _EPSILON),
            np.abs(residual) + error,
            "certified" if certified else "sampled",
            complete=True,
        )


@final
class SourceFidelityBounds(StrictModule, NonTrainableState):
    """Directed bounds and query counts in mesh/source order.

    Bounds retain the upper/lower ordering used by certificate fingerprints.
    """

    semantics: tuple[SourceBoundSemantics, SourceBoundSemantics] = eqx.field(static=True)
    mesh_to_source: tuple[float, float] = eqx.field(static=True)
    source_to_mesh: tuple[float, float] = eqx.field(static=True)
    sample_counts: tuple[int, int] = eqx.field(static=True)




# Only the independent full source/homotopy proof below issues this premise.
# Reconstruction validates its stored report but never grants proof authority:
# a restored process must earn the theorem again, as for issued domain coverage.
# Key by the exact issued report and weakly retain its mesh owner. Several
# independently earned reports on the same mesh may remain valid concurrently.
_ISSUED_CAD_BOUNDARY: IssuedProofs[CadBoundaryClosureEvidence, CellMesh] = IssuedProofs()


def _cad_boundary_request_key(
    mesh: CellMesh, geometry: CellGeometrySpec, limits: MeshCertificateLimits,
    closure: CadBoundaryClosureEvidence, /
) -> str:
    return canonical_fingerprint(
        {
            "kind": "issued-cad-boundary-isotopy",
            "mesh": mesh.mesh_id,
            "mesh_arrays": array_tree_fingerprint(mesh),
            "topology": mesh.topology_id,
            "coordinate_scope": coordinate_scope_key(mesh, geometry),
            "limits": limits.limits_id,
            "issued_evidence": closure.evidence_id,
        }
    )


def _cad_boundary_source_key(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    limits: MeshCertificateLimits,
    closure: CadBoundaryClosureEvidence,
    source: domain_boundary_owner.MeshingDomainBoundarySource,
    budget: CoordinateEnclosureBudget,
    /,
) -> str:
    with budget.temporary_scope():
        admit_carrier_fingerprint(source, budget)
        return canonical_fingerprint(
            {
                "issued_boundary": _cad_boundary_request_key(mesh, geometry, limits, closure),
                "source_definition": carrier_payload(source),
            }
        )


def require_issued_cad_boundary_closure(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    closure: CadBoundaryClosureEvidence,
    limits: MeshCertificateLimits,
    /,
    *,
    source: domain_boundary_owner.MeshingDomainBoundarySource | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> None:
    """Authenticate an actually earned whole-boundary theorem, not a report/hash."""
    CadBoundaryClosureEvidence.from_stored_fields(closure)
    closure.binding.require(mesh, geometry)
    expected = MeshCertificateBinding(
        mesh, geometry, "mapped", limits,
        source_id=closure.binding.source_id,
        source_revision=closure.binding.source_revision,
    )
    if (
        closure.binding.limits_id != limits.limits_id
        or closure.binding.coordinate_scope != "mapped"
        or closure.binding.junction_vertices
        or closure.binding.binding_id != expected.binding_id
        or _ISSUED_CAD_BOUNDARY.get(
            closure, _cad_boundary_request_key(mesh, geometry, limits, closure)
        ) is not mesh
    ):
        raise ValueError(
            "cad_boundary_closure must be the independently issued whole-boundary "
            "homotopy for the current mesh, geometry, topology and limits; "
            "stored reports, foreign owners and changed inputs must prove again"
        )
    if source is not None:
        with coordinate_certificate_request(limits, budget=budget) as budget:
            if _ISSUED_CAD_BOUNDARY.get(
                closure,
                _cad_boundary_source_key(mesh, geometry, limits, closure, source, budget),
            ) is not mesh:
                raise ValueError(
                    "cad_boundary_closure.source must retain the actual issued source "
                    "carriers, root/static metadata and complete chart restrictions"
                )


@final
class SourceFidelityCoverage(StrictModule, NonTrainableState):
    """Independent coverage evidence used by a source fidelity certificate."""

    projection: ImplicitProjectionCoverageEvidence | None = None
    domain: DomainCoverageCertificate | None = None
    chart: SourceBoundaryChartCover | None = None
    cad_closure: CadBoundaryClosureEvidence | None = None
    boundary_equality: DomainBoundaryFidelityEvidence | None = None

    @checked
    def __init__(
        self,
        projection: ImplicitProjectionCoverageEvidence | None = None,
        domain: DomainCoverageCertificate | None = None,
        chart: SourceBoundaryChartCover | None = None,
        cad_closure: CadBoundaryClosureEvidence | None = None,
        boundary_equality: DomainBoundaryFidelityEvidence | None = None,
    ) -> None:
        self.projection = projection
        self.domain = domain
        self.chart = chart
        self.cad_closure = cad_closure
        self.boundary_equality = boundary_equality


@final
class SourceFidelityCertificate(StrictModule, NonTrainableState):
    """Two-sided bounded deviation between the mesh boundary and a source.

    ``mesh_to_source_upper`` bounds the distance of every mesh boundary point
    from the source boundary; ``source_to_mesh_upper`` bounds the distance of
    every source boundary point from the mesh boundary. ``*_lower`` values are
    witnessed deviations. Each direction states whether its bound is certified
    or only sampled; ``status`` is certified only for certified bounds within
    the requested ``tolerance``.
    ``boundary_degree_id`` binds the oriented analytic projection
    proof on that route; other routes leave it absent.
    """

    binding: MeshCertificateBinding
    projection_coverage: ImplicitProjectionCoverageEvidence | None
    domain_coverage: DomainCoverageCertificate | None
    chart_coverage: SourceBoundaryChartCover | None
    cad_closure: CadBoundaryClosureEvidence | None
    boundary_equality: DomainBoundaryFidelityEvidence | None
    status: MeshCertificateStatus = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    tolerance: float = eqx.field(static=True)
    mesh_to_source_semantics: SourceBoundSemantics = eqx.field(static=True)
    source_to_mesh_semantics: SourceBoundSemantics = eqx.field(static=True)
    mesh_to_source_upper: float = eqx.field(static=True)
    mesh_to_source_lower: float = eqx.field(static=True)
    source_to_mesh_upper: float = eqx.field(static=True)
    source_to_mesh_lower: float = eqx.field(static=True)
    sample_order: int = eqx.field(static=True)
    mesh_sample_count: int = eqx.field(static=True)
    source_sample_count: int = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    target_facet_ids: tuple[int, ...] | None = eqx.field(static=True)
    source_scope_id: str | None = eqx.field(static=True)
    boundary_degree_id: str | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding: MeshCertificateBinding,
        findings: tuple[MeshCertificateFinding, ...],
        /,
        *,
        tolerance: float,
        bounds: SourceFidelityBounds,
        sample_order: int,
        coverage: SourceFidelityCoverage | None = None,
        target_facet_ids: tuple[int, ...] | None = None,
        source_scope_id: str | None = None,
        boundary_degree_id: str | None = None,
    ) -> None:
        semantics = bounds.semantics
        mesh_to_source = bounds.mesh_to_source
        source_to_mesh = bounds.source_to_mesh
        sample_counts = bounds.sample_counts
        projection_coverage = None if coverage is None else coverage.projection
        domain_coverage = None if coverage is None else coverage.domain
        chart_coverage = None if coverage is None else coverage.chart
        cad_closure = None if coverage is None else coverage.cad_closure
        boundary_equality = None if coverage is None else coverage.boundary_equality
        if target_facet_ids is not None:
            if (
                not target_facet_ids
                or tuple(sorted(set(target_facet_ids))) != target_facet_ids
            ):
                raise ValueError(
                    f"SourceFidelityCertificate.target_facet_ids must be nonempty, unique and increasing; got {target_facet_ids!r}."
                )
            if not isinstance(source_scope_id, str) or not source_scope_id:
                raise ValueError(
                    f"SourceFidelityCertificate.source_scope_id must be nonempty for scoped facets; got {source_scope_id!r}."
                )
        elif source_scope_id is not None and (
            not isinstance(source_scope_id, str) or not source_scope_id
        ):
            raise ValueError(
                f"SourceFidelityCertificate.source_scope_id must identify a nonempty source scope; got {source_scope_id!r}."
            )
        if projection_coverage is not None and (
            projection_coverage.binding.binding_id != binding.binding_id
        ):
            raise ValueError(
                f"SourceFidelityCertificate.projection_coverage.binding must match {binding.binding_id!r}; got {projection_coverage.binding.binding_id!r}."
            )
        if domain_coverage is not None and (
            domain_coverage.binding.binding_id != binding.binding_id
        ):
            raise ValueError(
                f"SourceFidelityCertificate.domain_coverage.binding must match {binding.binding_id!r}; got {domain_coverage.binding.binding_id!r}."
            )
        if chart_coverage is not None and (
            (chart_coverage.source_id, chart_coverage.source_revision)
            != (binding.source_id, binding.source_revision)
        ):
            raise ValueError(
                f"SourceFidelityCertificate.chart_coverage must name source {(binding.source_id, binding.source_revision)!r}; got {(chart_coverage.source_id, chart_coverage.source_revision)!r}."
            )
        if cad_closure is not None:
            CadBoundaryClosureEvidence.from_stored_fields(cad_closure)
            if (
                cad_closure.binding.binding_id != binding.binding_id
                or boundary_degree_id is None
                or target_facet_ids is not None
                or cad_closure.gap_upper > max(mesh_to_source[0], source_to_mesh[0])
                or cad_closure.homotopy_distance_upper
                > min(mesh_to_source[0], source_to_mesh[0])
            ):
                raise ValueError(
                    "SourceFidelityCertificate.cad_closure must bind this whole-boundary "
                    "source/mesh proof and its retained physical deviation bounds"
                )
        if boundary_equality is not None:
            boundary_equality.require_binding(binding)
            previous = boundary_equality.predecessor.fidelity
            if (
                target_facet_ids is not None
                or source_scope_id is not None
                or semantics
                != (previous.mesh_to_source_semantics, previous.source_to_mesh_semantics)
                or mesh_to_source
                != (previous.mesh_to_source_upper, previous.mesh_to_source_lower)
                or source_to_mesh
                != (previous.source_to_mesh_upper, previous.source_to_mesh_lower)
                or tolerance != previous.tolerance
                or sample_order != previous.sample_order
                or sample_counts
                != (previous.mesh_sample_count, previous.source_sample_count)
            ):
                raise ValueError(
                    "SourceFidelityCertificate.boundary_equality must retain its whole-source bounds and original controls; got changed fidelity"
                )
        forward = parse(semantics[0], SourceBoundSemantics, "semantics")
        backward = parse(semantics[1], SourceBoundSemantics, "semantics")
        findings_ = tuple(findings)
        if (forward, backward) != ("certified", "certified") and not findings_:
            findings_ = (
                MeshCertificateFinding("sampled_fidelity", "unresolved", "mesh"),
            )
        self.binding = binding
        self.projection_coverage = projection_coverage
        self.domain_coverage = domain_coverage
        self.chart_coverage = chart_coverage
        self.cad_closure = cad_closure
        self.boundary_equality = boundary_equality
        self.status = certificate_status(findings_)
        self.findings = findings_
        self.tolerance = float(tolerance)
        self.mesh_to_source_semantics = forward
        self.source_to_mesh_semantics = backward
        self.mesh_to_source_lower, self.mesh_to_source_upper = (
            float(mesh_to_source[1]),
            float(mesh_to_source[0]),
        )
        self.source_to_mesh_lower, self.source_to_mesh_upper = (
            float(source_to_mesh[1]),
            float(source_to_mesh[0]),
        )
        self.sample_order = sample_order
        self.mesh_sample_count, self.source_sample_count = sample_counts
        self.target_facet_ids = target_facet_ids
        self.source_scope_id = source_scope_id
        self.boundary_degree_id = boundary_degree_id
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "source-fidelity-certificate",
                "binding": binding.binding_id,
                "status": self.status,
                "findings": [value.finding_id for value in findings_],
                "tolerance": self.tolerance,
                "semantics": (forward, backward),
                # Unresolved directions are unbounded (inf), which JSON cannot
                # carry; float repr preserves both finite and unbounded identity values.
                "mesh_to_source": tuple(repr(float(value)) for value in mesh_to_source),
                "source_to_mesh": tuple(repr(float(value)) for value in source_to_mesh),
                "sample_order": sample_order,
                "sample_counts": tuple(sample_counts),
                "projection_coverage": None
                if projection_coverage is None
                else projection_coverage.evidence_id,
                "domain_coverage": None
                if domain_coverage is None
                else domain_coverage.certificate_id,
                "chart_coverage": None
                if chart_coverage is None
                else chart_coverage.cover_id,
                "cad_closure": None if cad_closure is None else cad_closure.evidence_id,
                **(
                    {}
                    if boundary_equality is None
                    else {"boundary_equality": boundary_equality.evidence_id}
                ),
                **(
                    {}
                    if boundary_degree_id is None
                    else {"boundary_degree": boundary_degree_id}
                ),
                **(
                    {}
                    if target_facet_ids is None
                    else {
                        "target_facets": target_facet_ids,
                        "source_scope": source_scope_id,
                    }
                ),
                **(
                    {"source_scope": source_scope_id}
                    if target_facet_ids is None and source_scope_id is not None
                    else {}
                ),
            }
        )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SourceFidelityCertificate], value: SourceFidelityCertificate, /
    ) -> SourceFidelityCertificate:
        """Rebuild a stored certificate and reject changed bound or evidence identity."""
        renewed = cls(
            value.binding,
            value.findings,
            tolerance=value.tolerance,
            bounds=SourceFidelityBounds(
                semantics=(
                    value.mesh_to_source_semantics,
                    value.source_to_mesh_semantics,
                ),
                mesh_to_source=(value.mesh_to_source_upper, value.mesh_to_source_lower),
                source_to_mesh=(value.source_to_mesh_upper, value.source_to_mesh_lower),
                sample_counts=(value.mesh_sample_count, value.source_sample_count),
            ),
            sample_order=value.sample_order,
            coverage=SourceFidelityCoverage(
                projection=value.projection_coverage,
                domain=value.domain_coverage,
                chart=value.chart_coverage,
                cad_closure=value.cad_closure,
                boundary_equality=value.boundary_equality,
            ),
            target_facet_ids=value.target_facet_ids,
            source_scope_id=value.source_scope_id,
            boundary_degree_id=value.boundary_degree_id,
        )
        if (
            renewed.certificate_id != value.certificate_id
            or renewed.status != value.status
        ):
            raise ValueError(
                "SourceFidelityCertificate.certificate_id must bind its stored fields and status; "
                f"got {value.certificate_id!r} with status {value.status!r}."
            )
        return renewed


def _source_authority_content_id(source: SourceBoundaryQuery, /) -> str:
    # Reuse the closure owner's identity recipe: array fingerprints alone omit
    # static controls such as implicit sampling spacing and source chart limits.
    from ..lifecycle._meshing_source_records import stored_record_content_id

    return stored_record_content_id(source, excluded="")


@final
class SourceFidelityPredecessor(StrictModule, NonTrainableState):
    """Complete independently accepted source authority for boundary-set reuse."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    source: SourceBoundaryQuery
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain
    cell_regions: np.ndarray
    embedding: GlobalEmbeddingCertificate
    coverage: DomainCoverageCertificate
    fidelity: SourceFidelityCertificate
    limits: MeshCertificateLimits

    @property
    def content_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "domain-boundary-fidelity-source",
                "mesh": (self.mesh.mesh_id, array_tree_fingerprint(self.mesh)),
                "geometry": coordinate_scope_key(self.mesh, self.geometry),
                "source": (
                    self.source.source_id,
                    self.source.source_revision,
                    _source_authority_content_id(self.source),
                ),
                "domain": (self.domain.domain_id, array_tree_fingerprint(self.domain)),
                "regions": array_tree_fingerprint(self.cell_regions),
                "embedding": self.embedding.certificate_id,
                "coverage": self.coverage.certificate_id,
                "fidelity": self.fidelity.certificate_id,
                "limits": self.limits.limits_id,
            }
        )

    def require_current(self) -> None:
        """Validate full proof content, boundary coverage, orientation and owners."""
        self.embedding.binding.require(self.mesh, self.geometry)
        GlobalEmbeddingCertificate.from_stored_fields(self.embedding)
        self.coverage.binding.require(self.mesh, self.geometry)
        self.fidelity.binding.require(self.mesh, self.geometry)
        rebuilt = DomainCoverageCertificate.from_stored_fields(
            self.coverage, self.domain
        )
        SourceFidelityCertificate.from_stored_fields(self.fidelity)
        if (
            self.mesh.storage is not None
            or self.mesh.periodic_topology is not None
            or self.mesh.topological_dimension != self.mesh.ambient_dimension
            or self.embedding.status != "certified"
            or self.coverage.status != "certified"
            or self.fidelity.status != "certified"
            or rebuilt.certificate_id != self.coverage.certificate_id
            or self.coverage.domain_id != self.domain.domain_id
            or self.coverage.embedding_certificate_id != self.embedding.certificate_id
            or self.coverage.covered_source_facet_count
            != self.coverage.source_facet_count
            or not self.coverage.source_facet_count
            or self.coverage.binding.limits_id != self.limits.limits_id
            or self.embedding.binding.limits_id != self.limits.limits_id
            or self.fidelity.binding.limits_id != self.limits.limits_id
            or self.fidelity.target_facet_ids is not None
            or self.fidelity.source_scope_id is not None
            or (
                self.fidelity.binding.source_id,
                self.fidelity.binding.source_revision,
            )
            != (self.source.source_id, self.source.source_revision)
            or (
                isinstance(self.source, MappedDomainBoundarySource)
                and not np.array_equal(self.source.cell_regions, self.cell_regions)
            )
            or (
                self.fidelity.mesh_to_source_semantics,
                self.fidelity.source_to_mesh_semantics,
            )
            != ("certified", "certified")
            or not math.isfinite(
                max(
                    self.fidelity.mesh_to_source_upper,
                    self.fidelity.source_to_mesh_upper,
                )
            )
            or max(
                self.fidelity.mesh_to_source_upper,
                self.fidelity.source_to_mesh_upper,
            )
            > self.fidelity.tolerance
        ):
            raise ValueError(
                "SourceFidelityPredecessor must retain complete certified oriented domain and whole-source fidelity proofs; got changed or partial authority"
            )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[SourceFidelityPredecessor], value: SourceFidelityPredecessor, /
    ) -> SourceFidelityPredecessor:
        """Validate retained data only; reconstruction never issues proof authority."""
        renewed = cls(
            value.mesh, value.geometry, value.source, value.domain,
            value.cell_regions, value.embedding, value.coverage,
            value.fidelity, value.limits,
        )
        renewed.require_current()
        return renewed


_ISSUED_INDEPENDENT_FIDELITY: IssuedProofs[
    CellMesh, SourceFidelityCertificate
] = IssuedProofs()


def _independent_fidelity_key(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    certificate: SourceFidelityCertificate,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "issued-independent-source-fidelity",
            "mesh": (mesh.mesh_id, array_tree_fingerprint(mesh)),
            "geometry": coordinate_scope_key(mesh, geometry),
            "source": (
                source.source_id, source.source_revision,
                _source_authority_content_id(source),
            ),
            "certificate": certificate.certificate_id,
        }
    )


def _issue_independent_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    certificate: SourceFidelityCertificate,
    /,
) -> SourceFidelityCertificate:
    if (
        certificate.status == "certified"
        and isinstance(source, StrictModule)
        and certificate.target_facet_ids is None
        and certificate.source_scope_id is None
        and certificate.boundary_equality is None
    ):
        _ISSUED_INDEPENDENT_FIDELITY.record(
            mesh,
            _independent_fidelity_key(mesh, geometry, source, certificate),
            certificate,
        )
    return certificate


_ISSUED_DOMAIN_FIDELITY: IssuedProofs[
    CellMesh, SourceFidelityCertificate
] = IssuedProofs()


def issue_domain_fidelity_source(
    predecessor: SourceFidelityPredecessor, /, *, budget: CoordinateEnclosureBudget
) -> None:
    """Record authority only after independent full acceptance in this process."""
    if _ISSUED_INDEPENDENT_FIDELITY.get(
        predecessor.mesh,
        _independent_fidelity_key(
            predecessor.mesh, predecessor.geometry, predecessor.source,
            predecessor.fidelity,
        ),
    ) is not predecessor.fidelity:
        return
    if not is_issued_domain_coverage(
        predecessor.mesh, predecessor.geometry, predecessor.domain,
        predecessor.cell_regions, predecessor.embedding, predecessor.limits,
        predecessor.coverage, budget=budget,
    ):
        return
    predecessor.require_current()
    # This operation's actual boundary/interface/adjacency proof, not a rebuilt
    # status or a toleranced volume comparison, must have been earned here.
    _ISSUED_DOMAIN_FIDELITY.record(
        predecessor.mesh, predecessor.content_id, predecessor.fidelity
    )


@final
class DomainBoundaryFidelityEvidence(StrictModule, NonTrainableState):
    """Exact physical boundary equality through full oriented domain coverage.

    Both volume carriers cover every declared material region, exterior facet
    and interface with the declared adjacency. Equal region volumes alone are
    not a premise. This transfers only directed boundary distances; feature,
    patch, stratum and normal claims remain their independent owners' work.
    """

    predecessor: SourceFidelityPredecessor
    target_coverage: DomainCoverageCertificate
    source_execution_error: float = eqx.field(static=True)
    target_execution_error: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        predecessor: SourceFidelityPredecessor,
        target_coverage: DomainCoverageCertificate,
        source_execution_error: float,
        target_execution_error: float,
    ) -> None:
        predecessor.require_current()
        rebuilt = DomainCoverageCertificate.from_stored_fields(
            target_coverage, predecessor.domain
        )
        if (
            target_coverage.status != "certified"
            or rebuilt.certificate_id != target_coverage.certificate_id
            or target_coverage.domain_id != predecessor.domain.domain_id
            or target_coverage.covered_source_facet_count
            != target_coverage.source_facet_count
            or target_coverage.source_facet_count
            != predecessor.coverage.source_facet_count
            or predecessor.fidelity.boundary_equality is not None
            or not math.isfinite(source_execution_error)
            or not math.isfinite(target_execution_error)
            or not 0.0 <= target_execution_error <= source_execution_error
        ):
            raise ValueError(
                "DomainBoundaryFidelityEvidence must retain complete exact boundary/interface coverage and an unchained root; got partial or foreign proof"
            )
        self.predecessor = predecessor
        self.target_coverage = target_coverage
        self.source_execution_error = float(source_execution_error)
        self.target_execution_error = float(target_execution_error)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "exact-domain-boundary-fidelity",
                "source": predecessor.content_id,
                "target_coverage": target_coverage.certificate_id,
                "execution_error": (
                    self.source_execution_error, self.target_execution_error
                ),
            }
        )

    def require_binding(self, binding: MeshCertificateBinding, /) -> None:
        renewed = DomainBoundaryFidelityEvidence(
            self.predecessor, self.target_coverage,
            self.source_execution_error, self.target_execution_error,
        )
        target = self.target_coverage.binding
        root = self.predecessor.fidelity.binding
        if (
            renewed.evidence_id != self.evidence_id
            or (
                binding.mesh_id,
                binding.topology_id,
                binding.geometry_layout_id,
                binding.geometry_id,
                binding.coordinate_scope,
            )
            != (
                target.mesh_id,
                target.topology_id,
                target.geometry_layout_id,
                target.geometry_id,
                target.coordinate_scope,
            )
            or (binding.source_id, binding.source_revision)
            != (root.source_id, root.source_revision)
        ):
            raise ValueError(
                "DomainBoundaryFidelityEvidence must bind the actual target maps and original source; got changed proof or owner"
            )

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[DomainBoundaryFidelityEvidence],
        value: DomainBoundaryFidelityEvidence,
        /,
    ) -> DomainBoundaryFidelityEvidence:
        renewed = cls(
            value.predecessor, value.target_coverage,
            value.source_execution_error, value.target_execution_error,
        )
        if renewed.evidence_id != value.evidence_id:
            raise ValueError(
                "DomainBoundaryFidelityEvidence.evidence_id must bind its complete root and target proofs; got changed content"
            )
        return renewed


def _reprove_domain_fidelity_root(
    root: SourceFidelityPredecessor, /, *, budget: CoordinateEnclosureBudget
) -> SourceFidelityPredecessor | None:
    """Earn a cold root on this request's existing cumulative budget."""
    validity = cell_geometry_validity_owner.certify_cell_geometry_validity(
        root.geometry, mesh=root.mesh, budget=budget
    )
    if validity.certificate_id != root.embedding.validity_certificate_id:
        # This route cannot reconstruct another validity policy from an ID.
        return None
    embedding = certify_global_embedding(
        root.mesh, root.geometry, validity, limits=root.limits, budget=budget
    )
    coverage = certify_domain_coverage(
        root.mesh, root.geometry, root.domain, root.cell_regions,
        embedding=embedding, limits=root.limits, budget=budget,
    )
    fidelity = certify_source_fidelity(
        root.mesh, root.geometry, root.source,
        tolerance=root.fidelity.tolerance, sample_order=root.fidelity.sample_order,
        limits=root.limits, budget=budget,
    )
    if (
        embedding.status != "certified"
        or coverage.status != "certified"
        or fidelity.status != "certified"
    ):
        return None
    renewed = SourceFidelityPredecessor(
        root.mesh, root.geometry, root.source, root.domain, root.cell_regions,
        embedding, coverage, fidelity, root.limits,
    )
    issue_domain_fidelity_source(renewed, budget=budget)
    return renewed


def prepare_domain_boundary_fidelity(
    predecessor: SourceFidelityPredecessor,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    domain: PiecewiseLinearDomain | mapped_reference_domain_owner.MappedReferenceDomain,
    cell_regions: np.ndarray,
    embedding: GlobalEmbeddingCertificate,
    coverage: DomainCoverageCertificate,
    limits: MeshCertificateLimits,
    /,
    *,
    tolerance: float,
    sample_order: int,
    budget: CoordinateEnclosureBudget,
) -> SourceFidelityCertificate | None:
    """Reuse issued root bounds only after earning exact target boundary equality."""
    if isinstance(source, ImplicitProjectionBoundarySource):
        # Equal domain boundaries transfer distance bounds, not the target's
        # bound facet/normal-fiber degree certificate. Let the canonical query
        # earn its complete projection obligation on the caller's ledger.
        return None
    predecessor.require_current()
    inherited = predecessor.fidelity.boundary_equality
    root = predecessor if inherited is None else inherited.predecessor
    root.require_current()
    query = source
    if isinstance(source, MappedDomainBoundarySource) and isinstance(
        root.source, MappedDomainBoundarySource
    ):
        if not np.array_equal(source.cell_regions, cell_regions):
            return None
        # The query owner's explicit renewal changes target routing, not the
        # independently authored source or any sampling/coverage controls.
        query = source.for_target_regions(root.source.cell_regions)
    if (
        not tree_content_equal(domain, root.domain)
        or (source.source_id, source.source_revision)
        != (root.source.source_id, root.source.source_revision)
        or not tree_content_equal(query, root.source)
        or tolerance != root.fidelity.tolerance
        or sample_order != root.fidelity.sample_order
        or mesh.storage is not None
        or mesh.periodic_topology is not None
        or mesh.topological_dimension != mesh.ambient_dimension
    ):
        return None
    issued = _ISSUED_DOMAIN_FIDELITY.get(root.mesh, root.content_id)
    if issued is None:
        reproved = _reprove_domain_fidelity_root(root, budget=budget)
        if reproved is None:
            return None
        root = reproved
        issued = _ISSUED_DOMAIN_FIDELITY.get(root.mesh, root.content_id)
    if issued is None or issued.certificate_id != root.fidelity.certificate_id:
        return None
    embedding.binding.require(mesh, geometry)
    if not is_issued_domain_coverage(
        mesh, geometry, domain, cell_regions, embedding, limits, coverage, budget=budget,
    ):
        return None
    source_error = root.geometry.source_execution_error(root.mesh, budget=budget)
    target_error = geometry.source_execution_error(mesh, budget=budget)
    if target_error > source_error:
        return None
    budget.retain_owner((root, coverage))
    evidence = DomainBoundaryFidelityEvidence(root, coverage, source_error, target_error)
    binding = MeshCertificateBinding(
        mesh, geometry, coverage.binding.coordinate_scope, limits,
        source_id=source.source_id, source_revision=source.source_revision,
    )
    previous = root.fidelity
    return SourceFidelityCertificate(
        binding, (), tolerance=tolerance,
        bounds=SourceFidelityBounds(
            semantics=("certified", "certified"),
            mesh_to_source=(previous.mesh_to_source_upper, previous.mesh_to_source_lower),
            source_to_mesh=(previous.source_to_mesh_upper, previous.source_to_mesh_lower),
            sample_counts=(previous.mesh_sample_count, previous.source_sample_count),
        ),
        sample_order=sample_order,
        coverage=SourceFidelityCoverage(boundary_equality=evidence),
    )


def _fidelity_items(
    state: EmbeddingState,
    mesh: CellMesh,
    points: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine boundary pieces compared with the source and their entity ids."""

    dimension = mesh.topological_dimension
    cell_ids = cell_global_ids(mesh)
    if dimension == mesh.ambient_dimension:
        facets = mesh_facets(mesh, budget=budget)
        boundary = boundary_pieces(state, mesh, points, facets, limits, budget=budget)
        return boundary.pieces, boundary.entities[boundary.piece_owners]
    if dimension == 1:
        items = np.concatenate(
            [
                np.asarray(jax.device_get(block.vertices), dtype=np.int64)
                for block in mesh.blocks
            ]
        )
        return items, cell_ids
    triangles, owners, nonplanar, undecided = loop_triangles(
        points, cell_loops(mesh), limits.maximum_candidate_pairs, budget=budget
    )
    if np.any(nonplanar | undecided):
        state.add("curved_cell", "unresolved", "cell", cell_ids[nonplanar | undecided])
    return triangles, cell_ids[owners]


def _lattice(order: int, vertices: int, /) -> np.ndarray:
    """Barycentric lattice of spacing ``1 / order`` on a simplex."""

    rows = [
        index
        for index in np.ndindex(*((order + 1,) * (vertices - 1)))
        if sum(index) <= order
    ]
    weights = np.asarray(rows, dtype=np.float64) / order
    return np.concatenate((1.0 - np.sum(weights, axis=1, keepdims=True), weights), axis=1)


def point_simplex_distances(samples: np.ndarray, simplices: np.ndarray, /) -> np.ndarray:
    """Euclidean distances ``(samples, simplices)`` to closed segments/triangles."""

    if simplices.shape[1] == 2:
        start = simplices[None, :, 0]
        edge = simplices[None, :, 1] - start
        offset = samples[:, None, :] - start
        length = np.maximum(np.sum(edge * edge, axis=-1), np.finfo(np.float64).tiny)
        t = np.clip(np.sum(offset * edge, axis=-1) / length, 0.0, 1.0)
        return np.linalg.norm(offset - t[..., None] * edge, axis=-1)
    a = simplices[None, :, 0]
    b = simplices[None, :, 1]
    c = simplices[None, :, 2]
    p = samples[:, None, :]
    distances = np.minimum.reduce(
        [
            point_simplex_distances(samples, np.stack((u, v), axis=1))
            for u, v in (
                (simplices[:, 0], simplices[:, 1]),
                (simplices[:, 1], simplices[:, 2]),
                (simplices[:, 2], simplices[:, 0]),
            )
        ]
    )
    normal = np.cross(b - a, c - a)
    norm = np.maximum(np.linalg.norm(normal, axis=-1), np.finfo(np.float64).tiny)
    height = np.sum((p - a) * normal, axis=-1) / norm
    foot = p - (height / norm)[..., None] * normal
    # The foot of the perpendicular lies in the triangle iff it is on the inner
    # side of every edge; otherwise the nearest point is on an edge.
    inside = np.ones(distances.shape, dtype=np.bool_)
    for u, v in ((a, b), (b, c), (c, a)):
        inside &= np.sum(np.cross(v - u, foot - u) * normal, axis=-1) >= 0.0
    return np.where(inside, np.minimum(distances, np.abs(height)), distances)


def _mesh_to_source(
    state: EmbeddingState,
    source: SourceBoundaryQuery,
    points: np.ndarray,
    items: np.ndarray,
    entities: np.ndarray,
    order: int,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
) -> tuple[SourceBoundSemantics, float, float, int]:
    """Refine a continuous affine cover until tolerance or an explicit limit."""
    corners = points[items]
    edges = np.stack(
        [
            np.linalg.norm(corners[:, i] - corners[:, j], axis=-1)
            for i, j in combinations(range(items.shape[1]), 2)
        ],
        axis=1,
    )
    diameter = np.max(edges, axis=1) * (1 + 16 * _EPSILON)
    rounding = 16 * _EPSILON * np.max(np.abs(corners), axis=(1, 2))
    bounds = np.full((items.shape[0],), math.inf, dtype=np.float64)
    witnesses = np.zeros((items.shape[0],), dtype=np.float64)
    active = np.arange(items.shape[0], dtype=np.int64)
    evaluations, pieces = 0, 0
    semantics: SourceBoundSemantics = "sampled"
    for depth in range(limits.maximum_subdivision_depth + 1):
        dimension = items.shape[1] - 1
        count = active.size * math.comb(order + dimension, dimension)
        added_pieces = active.size * order**dimension
        capacity = next(
            (
                name
                for name, work, maximum in (
                    (
                        "mesh_sample_capacity",
                        evaluations + count,
                        limits.maximum_source_samples,
                    ),
                    (
                        "distance_capacity",
                        evaluations + count,
                        limits.maximum_distance_evaluations,
                    ),
                    (
                        "subdivision_capacity",
                        pieces + added_pieces,
                        limits.maximum_subdivision_pieces,
                    ),
                    (
                        "source_work_capacity",
                        evaluations + count,
                        limits.maximum_work_units,
                    ),
                    (
                        "source_scratch_capacity",
                        count * 256 + corners.nbytes * 4,
                        limits.maximum_scratch_bytes,
                    ),
                )
                if work > maximum
            ),
            None,
        )
        if capacity is not None:
            state.add(capacity, "unresolved", "facet", entities[active])
            break
        lattice = _lattice(order, items.shape[1])
        samples = ein.contract("lv,ivd->ild", lattice, corners[active]).reshape(
            -1, points.shape[1]
        )
        distance = source.boundary_distance(samples)
        if distance.lower.shape != (count,):
            raise ValueError(
                f"SourceBoundaryDistance.lower must match {count} query points; got {distance.lower.shape}."
            )
        evaluations += count
        pieces += added_pieces
        semantics = distance.semantics
        # Affine lattice sub-simplices cover the entire simplex. Their diameter
        # decreases under refinement; source distance is one-Lipschitz.
        upper = np.max(distance.upper.reshape(active.size, -1), axis=1)
        lower = np.max(distance.lower.reshape(active.size, -1), axis=1)
        bounds[active] = np.minimum(
            bounds[active],
            np.nextafter(upper + diameter[active] / order + rounding[active], math.inf),
        )
        witnesses[active] = np.maximum(
            witnesses[active],
            np.maximum(np.nextafter(lower - rounding[active], -math.inf), 0.0),
        )
        certified = semantics == "certified"
        violated = certified & (witnesses[active] > tolerance)
        if np.any(violated):
            state.add(
                "boundary_deviation", "violated", "facet", entities[active[violated]]
            )
        active = active[(bounds[active] > tolerance) & ~violated]
        if not active.size or not certified:
            break
        if depth == limits.maximum_subdivision_depth:
            state.add("subdivision_depth", "unresolved", "facet", entities[active])
            break
        order *= 2
    if active.size:
        state.add("boundary_deviation", "unresolved", "facet", entities[active])
    return semantics, float(np.max(bounds)), float(np.max(witnesses)), evaluations


class _DistanceCapacity(Exception):
    def __init__(self, requested: int) -> None:
        self.requested = requested


@dataclass
class _DistanceMeter:
    """Admit packed traversal work, storage and source-distance queries in order."""

    budget: CoordinateEnclosureBudget
    maximum_evaluations: int
    evaluations: int = 0

    def work(self, units: int) -> None:
        self.budget.reserve(units)

    def storage(self, nbytes: int) -> None:
        self.budget.reserve(0, nbytes)

    def visit(self, kind: str, units: int) -> None:
        del kind
        requested = self.evaluations + units
        if requested > self.maximum_evaluations:
            raise _DistanceCapacity(requested)
        self.budget.reserve(units)
        if units > self.budget.ticket.remaining_geometry_queries:
            self.budget.ticket.refuse(0, queries_upper=units)
        self.budget.ticket.consume(geometry_queries=units)

        self.evaluations = requested


@dataclass
class _SimplexDistanceKernel:
    corners: np.ndarray
    delta: np.ndarray
    other: np.ndarray
    slack: float

    def bounds(
        self,
        point: np.ndarray,
        node: int,
        lower: np.ndarray,
        upper: np.ndarray,
        out: np.ndarray,
    ) -> None:
        del node
        np.subtract(lower, point, out=self.delta)
        np.subtract(point, upper, out=self.other)
        np.maximum(self.delta, self.other, out=self.delta)
        np.maximum(self.delta, 0.0, out=self.delta)
        distance = math.sqrt(float(np.dot(self.delta, self.delta)))
        out[0] = max(0.0, distance * (1 - 16 * _EPSILON) - self.slack)

    def values(self, point: np.ndarray, selected: np.ndarray, out: np.ndarray) -> None:
        out[:, 0] = point_simplex_distances(point[None], self.corners[selected])[0]


@checked
def source_to_mesh(
    state: EmbeddingState,
    source: SourceBoundaryQuery,
    points: np.ndarray,
    items: np.ndarray,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[SourceBoundSemantics, float, float, int]:
    samples = source.boundary_samples(limits.maximum_source_samples)
    count = samples.points.shape[0]
    if not samples.complete:
        state.add("source_sample_capacity", "unresolved", "mesh")
        return "sampled", math.inf, 0.0, count
    # Candidate work is admitted during bounded packed traversal, not as an
    # unexecuted Cartesian product of every source sample and target facet.
    if count == 0:
        return samples.semantics, 0.0, 0.0, 0

    meter = _DistanceMeter(budget, limits.maximum_distance_evaluations, 0)
    owners: list[object] = []

    try:
        with budget.measurement(), budget.temporary_scope():
            owners.append((points, items, samples))
            meter.work(items.size * points.shape[1])
            meter.storage(items.size * points.shape[1] * 8)
            corners = points[items]
            owners.append(corners)
            meter.work(2 * corners.size)
            meter.storage(2 * corners.shape[0] * corners.shape[2] * 8)
            boxes_min, boxes_max = np.min(corners, axis=1), np.max(corners, axis=1)
            owners.append((boxes_min, boxes_max))
            tree = prepare_bvh(
                boxes_min,
                boxes_max,
                dtype=np.float64,
                _charge_work=meter.work,
                _reserve_storage=meter.storage,
            )
            owners.append(tree)
            meter.work(2 * (points.size + samples.points.size))
            meter.storage(points.nbytes + samples.points.nbytes)
            scale = max(
                float(np.max(np.abs(points))), float(np.max(np.abs(samples.points)))
            )
            slack = 32.0 * _EPSILON * scale
            dimension = points.shape[1]
            # The existing closed-simplex owner uses bounded leaf batches.
            # Its 512-byte per-pair workspace bound is admitted
            # before dispatch, alongside the explicit box-distance buffers.
            meter.storage(512 * tree.leaf_size + 16 * dimension)
            delta = np.empty((dimension,), dtype=np.float64)
            other = np.empty((dimension,), dtype=np.float64)
            owners.append((delta, other))

            kernel = _SimplexDistanceKernel(corners, delta, other, slack)

            minima, _, _, _ = bvh_host_minima(
                tree,
                samples.points,
                objective_count=1,
                node_lower_bounds=kernel.bounds,
                item_values=kernel.values,
                visit=meter.visit,
                admit_storage=meter.storage,
                retain_owner=owners.append,
            )
            meter.work(8 * count)
            meter.storage(3 * count * 8 + 4 * count)
            nearest = minima[:, 0]
            upper = nearest * (1.0 + 16.0 * _EPSILON) + slack + samples.covering_radius
            lower = np.maximum(
                nearest * (1.0 - 16.0 * _EPSILON) - slack - samples.point_error,
                0.0,
            )
            certified = samples.semantics == "certified"
            exceeded = lower > tolerance
            open_samples = (upper > tolerance) & ~(exceeded & certified)
            if np.any(exceeded & certified):
                state.add(
                    "source_uncovered",
                    "violated",
                    "source_sample",
                    np.flatnonzero(exceeded),
                )
            if np.any(open_samples):
                state.add(
                    "source_uncovered",
                    "unresolved",
                    "source_sample",
                    np.flatnonzero(open_samples),
                )
            result = samples.semantics, float(np.max(upper)), float(np.max(lower)), count
    except _DistanceCapacity as failure:
        state.findings.append(
            MeshCertificateFinding(
                "distance_capacity",
                "unresolved",
                "mesh",
                resource="distance_evaluations",
                requested=(
                    ("limit", limits.maximum_distance_evaluations),
                    ("requested", failure.requested),
                ),
                achieved=(("completed", meter.evaluations),),
            )
        )
        return samples.semantics, math.inf, 0.0, count
    except CoordinateEnclosureResourceError as failure:
        state.add(
            "source_cover_expression_budget",
            "unresolved",
            "mesh",
            resource_error=failure,
            expression_budget=budget,
        )
        return samples.semantics, math.inf, 0.0, count
    return result


@dataclass(frozen=True)
class FidelityMap:
    coordinates: tuple[Expression, ...]
    dimension: int
    entity: int
    entity_kind: MeshCertificateEntityKind = "cell"


@final
class _FidelityBoundaryScope(StrictModule):
    """Physical boundary keys and stable facet identifiers for map restrictions."""

    boundary: set[tuple[int, ...]] = eqx.field(static=True)
    facet_entities: dict[tuple[int, ...], int] = eqx.field(static=True)


def _fidelity_boundary_scope(
    mesh: CellMesh,
    volume: bool,
    target_facet_ids: tuple[int, ...] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FidelityBoundaryScope:
    boundary: set[tuple[int, ...]] = set()
    facet_entities: dict[tuple[int, ...], int] = {}
    if volume:
        facets = mesh_facets(mesh, budget=budget)
        identifiers = facet_entity_ids(mesh, facets.rows)
        facet_entities = {
            tuple(sorted(int(v) for v in row if v >= 0)): int(identifier)
            for row, identifier in zip(facets.rows, identifiers, strict=True)
        }
        if mesh.periodic_topology is not None:
            degree = mesh.topological_dimension - 1
            periodic = mesh.periodic_topology
            entity_rows = {
                int(value): index
                for index, value in enumerate(
                    np.asarray(jax.device_get(mesh.entity_set(degree).entity_ids))
                )
            }
            orbit = np.asarray(jax.device_get(periodic.orbits(degree)[0]))
            walls = np.asarray(
                jax.device_get(periodic.quotient.entities(degree).subset("boundary").mask)
            )
            physical = np.asarray(
                [walls[orbit[entity_rows[int(value)]]] for value in identifiers],
                dtype=np.bool_,
            )
            boundary_mask = facets.boundary & physical
        else:
            boundary_mask = facets.boundary
        boundary = {
            tuple(sorted(int(v) for v in row if v >= 0))
            for row in facets.rows[boundary_mask]
        }
        if target_facet_ids is not None:
            selected = set(target_facet_ids)
            if not selected <= set(facet_entities.values()):
                raise ValueError(
                    "SourceFidelityCertificate.target_facet_ids must name mesh facets; got an absent identifier."
                )
            boundary = {
                key
                for key, identifier in facet_entities.items()
                if identifier in selected
            }
    return _FidelityBoundaryScope(boundary, facet_entities)


@final
class _FidelityFaces(StrictModule):
    """Reference facet chains with stable entity IDs, including collapsed pyramids."""

    faces: list[tuple[int, ...]] = eqx.field(static=True)
    entities: list[int] = eqx.field(static=True)
    reference: np.ndarray


def _fidelity_faces(
    cell_kind: str,
    topology: ReferenceCellTopology,
    vertices: np.ndarray,
    cell: int,
    entity: int,
    reference: np.ndarray,
    boundary_scope: _FidelityBoundaryScope,
    volume: bool,
    scoped: bool,
    /,
) -> _FidelityFaces:
    boundary, facet_entities = boundary_scope.boundary, boundary_scope.facet_entities
    if volume:
        faces = [
            face
            for face in topology.entities[topology.dimension - 1]
            if tuple(sorted(int(vertices[cell, v]) for v in face)) in boundary
        ]
    else:
        faces = [tuple(range(len(topology.vertices)))]
    face_entities = (
        [
            facet_entities[tuple(sorted(int(vertices[cell, v]) for v in face))]
            for face in faces
        ]
        if scoped
        else [entity] * len(faces)
    )
    if cell_kind == "pyramid":
        # The polynomial source lives on the collapsed cube; side
        # squares collapse continuously to the physical apex.
        cube = np.asarray(
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            dtype=np.float64,
        )
        collapsed: dict[frozenset[int], tuple[int, ...]] = {
            frozenset((0, 1, 2, 3)): (0, 1, 2, 3),
            frozenset((0, 1, 4)): (0, 1, 5, 4),
            frozenset((1, 2, 4)): (1, 2, 6, 5),
            frozenset((2, 3, 4)): (2, 3, 7, 6),
            frozenset((0, 3, 4)): (3, 0, 4, 7),
        }
        faces = [collapsed[frozenset(face)] for face in faces]
        reference = cube
    return _FidelityFaces(faces, face_entities, reference)


def _prepared_fidelity_cells(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> coordinate_enclosure_owner.PreparedCoordinateCells | None:
    """Reuse only the current request's earned physical maps for every cell."""
    prepared = budget.prepared_cell_cache.get(coordinate_scope_key(mesh, geometry))
    if prepared is None:
        return None
    count = sum(block.cell_count for block in mesh.blocks)
    budget.reserve(2 * count, 256 + 24 * count)
    identifiers = tuple(
        int(identifier)
        for block in mesh.blocks
        for identifier in np.asarray(block.global_ids, dtype=np.int64)
    )
    if (
        prepared.cell_ids != identifiers
        or len(prepared.descriptors) != count
        or len(prepared.coordinates) != count
    ):
        raise ValueError(
            "_fidelity_maps.prepared_cells must match the mesh cell bank; got inconsistent identifiers or counts"
        )
    return prepared


def _fidelity_maps(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    target_facet_ids: tuple[int, ...] | None = None,
    budget: CoordinateEnclosureBudget,
) -> list[FidelityMap]:
    """Restrict only selected boundary cells to the oriented simplex chain.

    Full physical cell maps earned by embedding in this request are reused
    under the current mesh, geometry, and live-source identity. Interior volume
    cells have no requested chart and do not require coefficient expansion.
    """

    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    values = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    volume = mesh.topological_dimension == mesh.ambient_dimension
    boundary_scope = _fidelity_boundary_scope(
        mesh, volume, target_facet_ids, budget=budget
    )
    prepared = _prepared_fidelity_cells(mesh, geometry, budget=budget)
    offset = 0
    result = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind not in (
            "interval",
            "triangle",
            "quadrilateral",
            "tetrahedron",
            "prism",
            "pyramid",
            "hexahedron",
        ):
            state.add(
                "coordinate_source_expression",
                "unresolved",
                "cell",
                np.asarray(jax.device_get(block.global_ids), dtype=np.int64),
            )
            offset += block.cell_count
            continue
        # Static heterogeneous blocks cross to host once before their cell traversal.
        topology = reference_cell_topology(block.cell_kind)
        reference = np.asarray(topology.vertices, dtype=np.float64)
        vertices = np.asarray(jax.device_get(block.vertices), dtype=np.int64)
        local_routes = np.asarray(jax.device_get(route), dtype=np.int64)
        cell_ids = np.asarray(jax.device_get(block.global_ids), dtype=np.int64)
        for cell in range(local_routes.shape[0]):
            dofs = np.array(local_routes[cell], dtype=np.int64, copy=False)
            if dofs.ndim != 1:
                raise ValueError(
                    f"CellGeometrySpec.geometry_dofs must contain one-dimensional cell coefficient routes; got {dofs.shape}."
                )
            entity = int(cell_ids[cell])
            restricted = _fidelity_faces(
                block.cell_kind,
                topology,
                vertices,
                cell,
                entity,
                reference,
                boundary_scope,
                volume,
                target_facet_ids is not None,
            )
            faces, face_entities, reference = (
                restricted.faces,
                restricted.entities,
                restricted.reference,
            )
            if not faces:
                continue
            # Interior maps cannot affect this boundary decision. A preceding
            # embedding proof may already own the selected cell's rational map.
            polynomial = (
                prepared.coordinates[offset + cell]
                if prepared is not None
                else coordinate_enclosure_owner.coordinate_expressions(
                    element, tuple(values[int(dof)] for dof in dofs), budget=budget
                )
            )
            if polynomial is None:
                state.add("coordinate_source_expression", "unresolved", "cell", (entity,))
                continue
            for face, face_entity in zip(faces, face_entities, strict=True):
                simplices = (
                    [face]
                    if len(face) <= 3
                    else [
                        (face[0], face[i], face[i + 1]) for i in range(1, len(face) - 1)
                    ]
                )
                for simplex in simplices:
                    corners = reference[np.asarray(simplex)]
                    arguments = coordinate_enclosure_owner.affine_arguments(
                        corners[0], (corners[1:] - corners[0]).T, budget=budget
                    )
                    composition = coordinate_enclosure_owner.ExpressionComposition(
                        arguments, budget=budget
                    )
                    result.append(
                        FidelityMap(
                            tuple(
                                composition(value, budget=budget) for value in polynomial
                            ),
                            len(simplex) - 1,
                            face_entity,
                            "facet" if target_facet_ids is not None else "cell",
                        )
                    )
        offset += block.cell_count
    return result


def _fidelity_node_count(coordinates: tuple[Expression, ...], dimension: int) -> int:

    return max(
        (
            coordinate_enclosure_owner.expression_node_count(value, "simplex", dimension)
            for value in coordinates
        ),
        default=0,
    )


@checked
def fidelity_norm_bound(
    coordinates: tuple[Expression, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:

    controls = [
        coordinate_enclosure_owner.expression_bernstein_coefficients(
            value, "simplex", dimension, budget=budget
        )
        for value in coordinates
    ]
    squared = sum(
        (max(abs(value) for value in bank) ** 2 for bank in controls), Fraction(0)
    )
    if squared == 0:
        return 0.0
    return float(
        np.nextafter(math.sqrt(float(np.nextafter(float(squared), math.inf))), math.inf)
    )


def fidelity_affine(corners: CoordinateCoefficients, /) -> tuple[Polynomial, ...]:
    """Interpolate retained source coefficients on an affine simplex."""

    rows = coordinate_enclosure_owner.coordinate_source_rows(corners)
    dimension = len(rows) - 1
    return tuple(
        {
            (0,) * dimension: rows[0][axis],
            **{
                tuple(int(i == j) for j in range(dimension)): rows[i + 1][axis]
                - rows[0][axis]
                for i in range(dimension)
            },
        }
        for axis in range(len(rows[0]))
    )


@checked
def fidelity_proxy(
    item: FidelityMap, /, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, float, float]:
    """Return an affine proxy and a continuous coordinate-minus-proxy enclosure."""

    reference = (tuple(Fraction(0) for _ in range(item.dimension)),) + tuple(
        tuple(Fraction(int(axis == j)) for j in range(item.dimension))
        for axis in range(item.dimension)
    )
    exact = [
        [
            coordinate_enclosure_owner.expression_evaluate(value, point, budget=budget)
            for value in item.coordinates
        ]
        for point in reference
    ]
    corners = np.asarray(
        [[float(value) for value in row] for row in exact], dtype=np.float64
    )
    affine = fidelity_affine(corners)
    residual = tuple(
        coordinate_enclosure_owner.expression_add(
            value,
            coordinate_enclosure_owner.expression_scale(linear, -1, budget=budget),
            budget=budget,
        )
        for value, linear in zip(item.coordinates, affine, strict=True)
    )
    deviation = fidelity_norm_bound(residual, item.dimension, budget=budget)
    rounding = max(
        fidelity_norm_bound(
            tuple(
                {(0,) * item.dimension: value - Fraction(float(rounded))}
                for value, rounded in zip(row, floating, strict=True)
            ),
            item.dimension,
            budget=budget,
        )
        for row, floating in zip(exact, corners, strict=True)
    )
    return corners, deviation, rounding


@final
class _ChartFidelityResult(StrictModule):
    forward: tuple[SourceBoundSemantics, float, float, int] = eqx.field(static=True)
    backward: tuple[SourceBoundSemantics, float, float, int] = eqx.field(static=True)
    cover: SourceBoundaryChartCover


def _match_chart_proxies(
    state: EmbeddingState,
    maps: list[FidelityMap],
    proxies: list[tuple[np.ndarray, float, float]],
    cover: SourceBoundaryChartCover,
    limits: MeshCertificateLimits,
    /,
) -> _ChartFidelityResult | None:
    lookup: dict[tuple[tuple[float, ...], ...], list[int]] = {}
    for i, (corners, _, _) in enumerate(proxies):
        key = tuple(sorted(tuple(float(v) for v in row) for row in corners))
        lookup.setdefault(key, []).append(i)
    matched = np.zeros((len(maps),), dtype=np.bool_)
    source_bounds = []
    forward = np.full((len(maps),), math.inf, dtype=np.float64)
    comparisons = 0
    for triangle, error in zip(cover.simplices, cover.deviation_bounds, strict=True):
        triangle = np.asarray(triangle, dtype=np.float64)
        key = tuple(sorted(tuple(float(v) for v in row) for row in triangle))
        candidates = lookup.get(key, [])
        if not candidates:
            # A collapsed chart triangle can have repeated corners. Its affine
            # segment is exactly contained in a kept triangle sharing endpoints.
            endpoints = set(key)
            candidates = (
                [
                    i
                    for other, ids in lookup.items()
                    if endpoints.issubset(set(other))
                    for i in ids
                ]
                if len(endpoints) < 3
                else []
            )
        comparisons += len(proxies) if len(set(key)) < 3 else len(candidates)
        if comparisons > limits.maximum_distance_evaluations:
            state.add("distance_capacity", "unresolved", "mesh")
            return _ChartFidelityResult(
                ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
            )
        if not candidates:
            return None
        bound = min(
            float(np.nextafter(float(error) + proxies[i][1], math.inf))
            for i in candidates
        )
        source_bounds.append(bound)
        for i in candidates:
            if len(set(key)) == 3:
                matched[i] = True
                forward[i] = min(
                    forward[i],
                    float(np.nextafter(float(error) + proxies[i][1], math.inf)),
                )
    if not np.all(matched):
        return None
    forward_upper = float(np.max(forward, initial=0.0))
    backward_upper = max(source_bounds, default=0.0)
    return _ChartFidelityResult(
        ("certified", forward_upper, 0.0, len(maps)),
        ("certified", backward_upper, 0.0, len(source_bounds)),
        cover,
    )


def _chart_refusal_cover(
    cover: SourceBoundaryChartCover,
    error: CoordinateEnclosureResourceError | meshcore_owner.MeshcoreError,
    limits: MeshCertificateLimits,
    /,
) -> SourceBoundaryChartCover:
    resources: list[tuple[AffineChainResourceMetric, int, int]] = []
    match error:
        case meshcore_owner.MeshcoreError():
            if error.work_counters is not None:
                resources.append(
                    (
                        "refused_total_work",
                        int(error.work_counters.construction_work),
                        limits.maximum_work_units,
                    )
                )
            if error.memory_counters is not None:
                resources.append(
                    (
                        "refused_native_peak_bytes",
                        int(error.memory_counters.peak_bytes),
                        limits.maximum_scratch_bytes,
                    )
                )
        case CoordinateEnclosureResourceError():
            metric: AffineChainResourceMetric
            match error.resource:
                case "coefficient_work":
                    metric = "refused_coefficient_work"
                case "polynomial_storage":
                    metric = "refused_polynomial_storage"
                case "retained_basis":
                    metric = "refused_retained_basis"
                case invalid_resource:
                    assert_never(invalid_resource)
            resources.append(
                (
                    metric,
                    int(error.completed),
                    int(error.limit),
                )
            )
        case invalid:
            assert_never(invalid)
    cover = SourceBoundaryChartCover(
        cover.simplices,
        cover.deviation_bounds,
        cover.semantics,
        cover.complete,
        cover.source_id,
        cover.source_revision,
        findings=cover.findings,
        resource_counts=cover.resource_counts,
        affine_chain_resources=(
            *(row.as_source() for row in cover.affine_chain_resources),
            *(AffineChainResourceEvidence(
                metric, "target", cover.cover_id, used, maximum,
            ) for metric, used, maximum in resources),
        ),
        restriction_ids=cover.restriction_ids,
    )
    return cover


def _brep_source_gap(
    source: SourceBoundaryQuery, /, *, budget: CoordinateEnclosureBudget
) -> tuple[float, str | None]:
    """Add full-carrier and endpoint displacement by the triangle inequality.

    The source-wide maximum deliberately applies even to interior chart rows:
    Their nodal boundary constraints may share a retained seam carrier.
    The domain owner independently revalidates the rational gap proofs.
    """
    if not isinstance(source, domain_boundary_owner.MeshingDomainBoundarySource):
        return 0.0, None
    authority = source.domain.brep_authority
    geometry = authority.geometry if isinstance(authority, BRepModel) else authority
    if not isinstance(geometry, BRepGeometry):
        return 0.0, None
    face_bounds = geometry.edge_face_gap_bounds
    vertex_bounds = geometry.edge_vertex_gap_bounds
    if face_bounds is None and vertex_bounds is None:
        return 0.0, None
    domain_preparation_owner.certify_brep_gap_bounds(
        geometry,
        tuple(patch.surface for patch in source.domain.patches),
        budget=budget,
    )
    budget.reserve(len(geometry.coedge_edges) + 2 * len(geometry.edge_vertices), 256)
    face_gap = max(face_bounds, default=0.0) if face_bounds is not None else 0.0
    vertex_gap = (
        max((max(row) for row in vertex_bounds), default=0.0)
        if vertex_bounds is not None
        else 0.0
    )
    if face_gap == 0.0 and vertex_gap == 0.0:
        return 0.0, None
    gap = float(np.nextafter(face_gap + vertex_gap, math.inf))
    identity = canonical_fingerprint(
        {
            "kind": "brep-source-seam-scope",
            "scope": source.source_scope_id,
            "geometry": geometry.geometry_id,
            "edge_face_gap_bounds": face_bounds,
            "edge_vertex_gap_bounds": vertex_bounds,
        }
    )
    return gap, identity


def _include_brep_source_gap(
    certificate: SourceFidelityCertificate,
    source: SourceBoundaryQuery,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SourceFidelityCertificate:
    """Apply source seam displacement even on zero-restriction shortcuts.

    Chart rows already include their own seam displacement; adding it again
    here is conservative and also covers routes that discard a chart comparison
    in favor of direct queries. No zero-equality claim survives a positive gap.
    """
    if certificate.cad_closure is not None:
        # The retained closure already includes original face, edge and vertex
        # incidence displacement. Charging it again would double the same bound.
        return certificate
    gap, identity = _brep_source_gap(source, budget=budget)
    if identity is None:
        return certificate
    forward = float(np.nextafter(certificate.mesh_to_source_upper + gap, math.inf))
    backward = float(np.nextafter(certificate.source_to_mesh_upper + gap, math.inf))
    findings = certificate.findings
    if max(forward, backward) > certificate.tolerance:
        findings = (
            *findings,
            MeshCertificateFinding("boundary_deviation", "unresolved", "mesh"),
        )
    return SourceFidelityCertificate(
        certificate.binding,
        findings,
        tolerance=certificate.tolerance,
        bounds=SourceFidelityBounds(
            semantics=(
                certificate.mesh_to_source_semantics,
                certificate.source_to_mesh_semantics,
            ),
            mesh_to_source=(
                forward,
                max(0.0, certificate.mesh_to_source_lower - gap),
            ),
            source_to_mesh=(
                backward,
                max(0.0, certificate.source_to_mesh_lower - gap),
            ),
            sample_counts=(
                certificate.mesh_sample_count,
                certificate.source_sample_count,
            ),
        ),
        sample_order=certificate.sample_order,
        coverage=SourceFidelityCoverage(
            projection=certificate.projection_coverage,
            domain=certificate.domain_coverage,
            chart=certificate.chart_coverage,
            cad_closure=certificate.cad_closure,
        ),
        target_facet_ids=certificate.target_facet_ids,
        source_scope_id=identity,
    )


def _chart_fidelity(
    state: EmbeddingState,
    source: SourceBoundaryQuery,
    maps: list[FidelityMap],
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _ChartFidelityResult | None:
    """Bound mesh coordinate maps against an independent source chart chain."""
    if (
        not maps
        or not isinstance(source, SourceBoundaryChartQuery)
        or any(item.dimension != 2 for item in maps)
    ):
        return None
    cover = source.boundary_chart_cover(limits.maximum_source_samples, budget=budget)
    gap, gap_identity = _brep_source_gap(source, budget=budget)
    if gap_identity is not None:
        budget.reserve(cover.deviation_bounds.size, cover.deviation_bounds.nbytes)
        cover = SourceBoundaryChartCover(
            cover.simplices,
            np.nextafter(cover.deviation_bounds + gap, math.inf),
            cover.semantics,
            cover.complete,
            cover.source_id,
            cover.source_revision,
            findings=cover.findings,
            resource_counts=cover.resource_counts,
            affine_chain_resources=cover.affine_chain_resources,
            restriction_ids=(*cover.restriction_ids, gap_identity),
        )
    if (
        cover.source_id != source.source_id
        or cover.source_revision != source.source_revision
    ):
        raise ValueError(
            f"SourceBoundaryChartCover.source_revision must name source {(source.source_id, source.source_revision)!r}; got {(cover.source_id, cover.source_revision)!r}."
        )
    state.findings.extend(cover.findings)
    if (
        not cover.complete
        or cover.semantics != "certified"
        or cover.simplices.shape[0] == 0
    ):
        return _ChartFidelityResult(
            ("sampled", math.inf, 0.0, 0), ("sampled", math.inf, 0.0, 0), cover
        )
    if max(len(maps), cover.simplices.shape[0]) > limits.maximum_source_samples:
        state.add("source_chart_capacity", "unresolved", "mesh")
        return _ChartFidelityResult(
            ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
        )
    if len(maps) > limits.maximum_subdivision_pieces:
        state.add("subdivision_capacity", "unresolved", "mesh")
        return _ChartFidelityResult(
            ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
        )
    if any(
        max(3, _fidelity_node_count(item.coordinates, 2)) > limits.maximum_bernstein_nodes
        for item in maps
    ):
        state.add("bernstein_capacity", "unresolved", "mesh")
        return _ChartFidelityResult(
            ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
        )

    proxies = []
    for item in maps:
        with budget.temporary_scope():
            proxies.append(fidelity_proxy(item, budget=budget))

    earned_areas: dict[
        tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]],
        Fraction,
    ] = {}
    try:
        with budget.measurement():
            affine_bound, cover, separation = (
                affine_chart_coverage_owner.prove_affine_chart_chain(
                    proxies,
                    cover,
                    limits,
                    tolerance,
                    source_identity=(source.source_id, source.source_revision),
                    areas=budget.affine_chain_intersection_cache,
                    earned_areas=earned_areas,
                )
            )
            if earned_areas:
                # Exact pair areas are request-owned proofs reused by later
                # whole and scoped chart chains over the same rational rows.
                if budget.capture_affine_areas:
                    budget.reserve(len(earned_areas), 128 + 32 * len(earned_areas))
                budget.retain_basis(
                    tuple(earned_areas.items()), operand=budget.capture_affine_areas,
                    _kernel_operands=True,
                )
                budget.affine_chain_intersection_cache.update(earned_areas)
                if budget.capture_affine_areas:
                    # Only actual exact-kernel output enters the issued bank;
                    # no working-cache entries or coverage verdicts are copied.
                    budget.affine_intersection_operand_snapshots.update(earned_areas)
    except (
        coordinate_budget_owner.CoordinateEnclosureResourceError,
        meshcore_owner.MeshcoreError,
    ) as error:
        match error:
            case meshcore_owner.MeshcoreError():
                if error.status not in (
                    meshcore_owner.MeshcoreStatus.CAPACITY_EXCEEDED,
                    meshcore_owner.MeshcoreStatus.TIMEOUT,
                ):
                    raise
            case CoordinateEnclosureResourceError():
                pass
            case invalid:
                assert_never(invalid)
        state.add(
            "source_affine_chain_resource_budget",
            "unresolved",
            "mesh",
            resource_error=error,
            expression_budget=budget,
        )
        cover = _chart_refusal_cover(cover, error, limits)
        return _ChartFidelityResult(
            ("certified", math.inf, 0.0, 0), ("certified", math.inf, 0.0, 0), cover
        )
    if separation is not None:
        state.add(
            "boundary_deviation", "violated", maps[0].entity_kind, (maps[0].entity,)
        )
        return _ChartFidelityResult(
            ("certified", math.inf, separation, 0), ("certified", math.inf, 0.0, 0), cover
        )
    if affine_bound is not None:
        return _ChartFidelityResult(
            ("certified", affine_bound, 0.0, 0),
            ("certified", affine_bound, 0.0, 0),
            cover,
        )
    # Equal corner coordinates establish a shared affine parameterization. The
    # residual bound applies everywhere, including bowed interiors.
    return _match_chart_proxies(state, maps, proxies, cover, limits)


def _fidelity_reference_children(dimension: int, /) -> tuple[np.ndarray, ...]:
    if dimension == 1:
        children = (
            np.asarray(((0.0,), (0.5,))),
            np.asarray(((0.5,), (1.0,))),
        )
    else:
        children = tuple(
            np.asarray(row, dtype=np.float64)
            for row in (
                ((0, 0), (0.5, 0), (0, 0.5)),
                ((0.5, 0), (1, 0), (0.5, 0.5)),
                ((0, 0.5), (0.5, 0.5), (0, 1)),
                ((0.5, 0), (0.5, 0.5), (0, 0.5)),
            )
        )
    return children


@checked
def split_fidelity_polynomial(
    value: Polynomial, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Polynomial, ...]:
    """Restrict a triangle polynomial to the four ordered midpoint children."""
    return tuple(
        coordinate_enclosure_owner.compose(
            value,
            coordinate_enclosure_owner.affine_arguments(
                child[0], (child[1:] - child[0]).T, budget=budget
            ),
            budget=budget,
        )
        for child in _fidelity_reference_children(2)
    )


@checked
def split_fidelity_map(
    item: FidelityMap, /, *, budget: CoordinateEnclosureBudget
) -> list[FidelityMap]:

    children = _fidelity_reference_children(item.dimension)
    return [
        FidelityMap(
            tuple(
                coordinate_enclosure_owner.expression_compose(
                    value,
                    coordinate_enclosure_owner.affine_arguments(
                        child[0], (child[1:] - child[0]).T, budget=budget
                    ),
                    budget=budget,
                )
                for value in item.coordinates
            ),
            item.dimension,
            item.entity,
            item.entity_kind,
        )
        for child in children
    ]


def _mapped_domain_source_maps(
    source: MappedDomainBoundarySource, /, *, budget: CoordinateEnclosureBudget
) -> tuple[list[FidelityMap], np.ndarray]:

    # Transformed/restored module fields must retain their source owner.
    match source.domain:
        case mapped_reference_domain_owner.MappedReferenceDomain() as domain:
            physical, embedding, coverage = (
                mapped_reference_proofs().certify_mapped_reference_source(
                    domain, source.limits, budget=budget
                )
            )
            if embedding.status != "certified" or coverage.status != "certified":
                raise ValueError(
                    f"MappedDomainBoundarySource.domain must have certified root embedding and reference coverage; got {(embedding.status, coverage.status)!r}."
                )
            state = EmbeddingState([], ["mapped_source_boundary_cover"])
            maps = _fidelity_maps(state, physical, domain.source_geometry, budget=budget)
            if state.findings or not maps:
                raise ValueError(
                    f"MappedDomainBoundarySource.domain must provide supported boundary expressions; got {len(maps)} maps and {len(state.findings)} findings."
                )
            if len(maps) > source.limits.maximum_subdivision_pieces or any(
                _fidelity_node_count(item.coordinates, item.dimension)
                > source.limits.maximum_bernstein_nodes
                for item in maps
            ):
                raise ValueError(
                    f"MappedDomainBoundarySource.domain must fit subdivision and Bernstein limits; got {len(maps)} boundary maps."
                )
            boxes = np.asarray(
                [
                    np.asarray(
                        [
                            coordinate_enclosure_owner.expression_bounds(
                                value, "simplex", item.dimension, budget=budget
                            )
                            for value in item.coordinates
                        ],
                        dtype=np.float64,
                    ).T
                    for item in maps
                ],
                dtype=np.float64,
            )
            return maps, boxes
        case invalid:
            raise TypeError(
                f"MappedDomainBoundarySource.domain must retain MappedReferenceDomain; got {type(invalid).__name__}."
            )


def _mapped_domain_source_samples(
    maps: list[FidelityMap],
    dimension: int,
    requested_radius: float,
    maximum_samples: int,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SourceBoundarySamples:

    pending = [(item, 0, 0) for item in maps]
    points, radii, errors = [], [], []
    complete = len(maps) <= limits.maximum_subdivision_pieces
    work = 0
    if not complete:
        pending.clear()
    sample_capacity = min(
        maximum_samples,
        limits.maximum_source_samples,
        limits.maximum_distance_evaluations,
        limits.maximum_subdivision_pieces,
    )
    live_bound = 0
    with budget.live_storage() as pending_storage:
        while pending:
            if work >= sample_capacity:
                complete = False
                break
            item, depth, item_storage = pending.pop()
            if (
                _fidelity_node_count(item.coordinates, item.dimension)
                > limits.maximum_bernstein_nodes
            ):
                complete = False
                break
            next_live_bound = live_bound - item_storage
            child_records: list[tuple[FidelityMap, int, int]] = []
            # A sample's enclosures are scratch. Only pending child maps survive
            # this iteration; the initial maps belong to the enclosing caller.
            with budget.polynomial_preparation_scope():
                anchor = _fidelity_anchor(item, budget=budget)
                point, error, radius = anchor.point, anchor.error, anchor.radius
                work += 1
                if radius > requested_radius and depth < limits.maximum_subdivision_depth:
                    children = split_fidelity_map(item, budget=budget)
                    if (
                        work + len(pending) + len(children)
                        <= limits.maximum_subdivision_pieces
                    ):
                        child_records = [
                            (
                                child,
                                depth + 1,
                                budget.live_object_storage_upper(
                                    (child, child.coordinates)
                                ),
                            )
                            for child in children
                        ]
                        next_live_bound += sum(storage for _, _, storage in child_records)
                    del children
                pending_storage.set_bound(next_live_bound)
                if not child_records:
                    complete = complete and radius <= requested_radius
                    points.append(point)
                    radii.append(radius)
                    errors.append(error)
            live_bound = next_live_bound
            pending.extend(child_records)
            del item, child_records
    return SourceBoundarySamples(
        np.asarray(points, dtype=np.float64).reshape(-1, dimension),
        np.asarray(radii, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        "certified",
        complete=complete,
    )


@final
class _FidelityAnchor(StrictModule):
    point: np.ndarray
    error: float = eqx.field(static=True)
    radius: float = eqx.field(static=True)


def _fidelity_anchor(
    item: FidelityMap, /, *, budget: CoordinateEnclosureBudget
) -> _FidelityAnchor:
    reference = (Fraction(1, item.dimension + 1),) * item.dimension
    exact = tuple(
        coordinate_enclosure_owner.expression_evaluate(value, reference, budget=budget)
        for value in item.coordinates
    )
    point = np.asarray([float(value) for value in exact], dtype=np.float64)
    error = fidelity_norm_bound(
        tuple(
            coordinate_enclosure_owner.constant(
                value - Fraction(float(rounded)),
                item.dimension,
                budget=budget,
            )
            for value, rounded in zip(exact, point, strict=True)
        ),
        item.dimension,
        budget=budget,
    )
    radius = fidelity_norm_bound(
        tuple(
            coordinate_enclosure_owner.expression_add(
                value,
                coordinate_enclosure_owner.constant(
                    -Fraction(float(rounded)),
                    item.dimension,
                    budget=budget,
                ),
                budget=budget,
            )
            for value, rounded in zip(item.coordinates, point, strict=True)
        ),
        item.dimension,
        budget=budget,
    )
    return _FidelityAnchor(point, error, radius)


@final
class _MappedCoverDecision(StrictModule):
    lower: float = eqx.field(static=True)
    violated: bool = eqx.field(static=True)
    bound: float = eqx.field(static=True)
    child_records: list[tuple[FidelityMap, int, int]]
    split: bool = eqx.field(static=True)
    stop: bool = eqx.field(static=True)
    unbounded: bool = eqx.field(static=True)


def _mapped_cover_decision(
    state: EmbeddingState,
    item: FidelityMap,
    depth: int,
    anchor: _FidelityAnchor,
    distance: SourceBoundaryDistance,
    tolerance: float,
    limits: MeshCertificateLimits,
    count: int,
    pending_count: int,
    lower: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedCoverDecision:
    error, radius = anchor.error, anchor.radius
    witnessed = max(
        0.0,
        float(np.nextafter(distance.lower[0] - error, -math.inf)),
    )
    bound = float(np.nextafter(distance.upper[0] + radius, math.inf))
    lower = max(lower, witnessed)
    violated = distance.semantics == "certified" and witnessed > tolerance
    if violated:
        state.add(
            "boundary_deviation",
            "violated",
            item.entity_kind,
            (item.entity,),
        )
    needs_split = not violated and (bound > tolerance or radius > tolerance * 0.125)
    split = False
    child_records: list[tuple[FidelityMap, int, int]] = []
    stop = False
    unbounded = False
    if needs_split and depth < limits.maximum_subdivision_depth:
        children = split_fidelity_map(item, budget=budget)
        if count + pending_count + len(children) <= limits.maximum_subdivision_pieces:
            child_records = [
                (
                    child,
                    depth + 1,
                    budget.live_object_storage_upper((child, child.coordinates)),
                )
                for child in children
            ]
            split = True
        elif bound > tolerance:
            state.add(
                "subdivision_capacity",
                "unresolved",
                item.entity_kind,
                (item.entity,),
            )
            unbounded = True
            stop = True
    elif needs_split and bound > tolerance:
        state.add(
            "subdivision_depth",
            "unresolved",
            item.entity_kind,
            (item.entity,),
        )
        unbounded = True
        stop = True
    return _MappedCoverDecision(
        lower, violated, bound, child_records, split, stop, unbounded
    )


@final
class _MappedMeshCover(StrictModule):
    forward: tuple[SourceBoundSemantics, float, float, int] = eqx.field(static=True)
    centers: np.ndarray
    radii: np.ndarray
    errors: np.ndarray


def _mapped_mesh_cover(
    state: EmbeddingState,
    source: SourceBoundaryQuery,
    maps: list[FidelityMap],
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedMeshCover:

    # Initial maps belong to the enclosing coordinate preparation. Child maps
    # are temporary subdivision owners: keep only the still-pending graphs live
    # and release each one after its last bound/query use.
    pending = [(item, 0, 0) for item in maps]
    centers, radii, errors = [], [], []
    upper, lower, count = 0.0, 0.0, 0
    semantics: SourceBoundSemantics = "certified"
    if len(pending) > limits.maximum_subdivision_pieces:
        state.add(
            "subdivision_capacity",
            "unresolved",
            maps[0].entity_kind if maps else "cell",
            tuple(item.entity for item in maps),
        )
        upper = math.inf
        pending.clear()
    live_bound = 0
    with budget.live_storage() as pending_storage:
        while pending:
            item, depth, item_storage = pending.pop()
            child_records: list[tuple[FidelityMap, int, int]] = []
            next_live_bound = live_bound - item_storage
            stop = False
            with budget.temporary_scope():
                if count >= min(
                    limits.maximum_source_samples, limits.maximum_distance_evaluations
                ):
                    check = (
                        "mesh_sample_capacity"
                        if count >= limits.maximum_source_samples
                        else "distance_capacity"
                    )
                    state.add(
                        check,
                        "unresolved",
                        item.entity_kind,
                        (
                            item.entity,
                            *(value.entity for value, _, _ in pending),
                        ),
                    )
                    upper = math.inf
                    stop = True
                elif (
                    _fidelity_node_count(item.coordinates, item.dimension)
                    > limits.maximum_bernstein_nodes
                ):
                    state.add(
                        "bernstein_capacity",
                        "unresolved",
                        item.entity_kind,
                        (item.entity,),
                    )
                    upper = math.inf
                else:
                    anchor = _fidelity_anchor(item, budget=budget)
                    center, error, radius = anchor.point, anchor.error, anchor.radius
                    distance = source.boundary_distance(center[None])
                    if distance.lower.shape != (1,):
                        raise ValueError(
                            f"SourceBoundaryDistance.lower must match 1 query points; got {distance.lower.shape}."
                        )
                    count += 1
                    if distance.semantics != "certified":
                        state.add(
                            "source_distance_semantics",
                            "unresolved",
                            item.entity_kind,
                            (item.entity,),
                        )
                        ambient = source.ambient_dimension
                        return _MappedMeshCover(
                            ("sampled", math.inf, 0.0, count),
                            np.asarray([center], dtype=np.float64).reshape(-1, ambient),
                            np.asarray([radius], dtype=np.float64),
                            np.asarray([error], dtype=np.float64),
                        )
                    decision = _mapped_cover_decision(
                        state,
                        item,
                        depth,
                        anchor,
                        distance,
                        tolerance,
                        limits,
                        count,
                        len(pending),
                        lower,
                        budget=budget,
                    )
                    lower, bound, violated = (
                        decision.lower,
                        decision.bound,
                        decision.violated,
                    )
                    child_records = decision.child_records
                    next_live_bound += sum(storage for _, _, storage in child_records)
                    split, stop = decision.split, decision.stop
                    if decision.unbounded:
                        upper = math.inf
                    if not split:
                        if bound > tolerance and not violated:
                            state.add(
                                "boundary_deviation",
                                "unresolved",
                                item.entity_kind,
                                (item.entity,),
                            )
                        upper = max(upper, bound)
                        centers.append(center)
                        radii.append(radius)
                        errors.append(error)
                pending_storage.set_bound(next_live_bound)
            live_bound = next_live_bound
            pending.extend(child_records)
            if stop:
                break
    ambient = source.ambient_dimension
    return _MappedMeshCover(
        (semantics, upper, lower, count),
        np.asarray(centers, dtype=np.float64).reshape(-1, ambient),
        np.asarray(radii, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


@dataclass
class _MappedDistanceKernel:
    centers: np.ndarray
    radii: np.ndarray
    errors: np.ndarray
    delta: np.ndarray
    distance: np.ndarray
    weights: np.ndarray
    node_delta: np.ndarray
    node_other: np.ndarray
    slack: float
    minimum_error: float
    maximum_radius: float

    def bounds(
        self,
        point: np.ndarray,
        node: int,
        lower: np.ndarray,
        upper: np.ndarray,
        out: np.ndarray,
    ) -> None:
        del node
        np.subtract(lower, point, out=self.node_delta)
        np.subtract(point, upper, out=self.node_other)
        np.maximum(self.node_delta, self.node_other, out=self.node_delta)
        np.maximum(self.node_delta, 0.0, out=self.node_delta)
        box_distance = math.sqrt(float(np.dot(self.node_delta, self.node_delta)))
        # Outward arithmetic lowers the box self.distance before applying
        # the same two objectives as the exhaustive center reduction.
        box_lower = max(0.0, box_distance * (1 - 16 * _EPSILON) - self.slack)
        out[0] = box_lower * (1 + 16 * _EPSILON) + self.slack + self.minimum_error
        out[1] = box_lower * (1 - 16 * _EPSILON) - self.slack - self.maximum_radius

    def values(self, point: np.ndarray, items: np.ndarray, out: np.ndarray) -> None:
        active_delta, active_distance, active_weights = (
            self.delta[: items.size],
            self.distance[: items.size],
            self.weights[: items.size],
        )
        np.take(self.centers, items, axis=0, out=active_delta)
        np.subtract(active_delta, point, out=active_delta)
        np.square(active_delta, out=active_delta)
        np.sum(active_delta, axis=1, out=active_distance)
        np.sqrt(active_distance, out=active_distance)
        np.multiply(active_distance, 1 + 16 * _EPSILON, out=out[:, 0])
        out[:, 0] += self.slack
        np.take(self.errors, items, out=active_weights)
        out[:, 0] += active_weights
        np.multiply(active_distance, 1 - 16 * _EPSILON, out=out[:, 1])
        out[:, 1] -= self.slack
        np.take(self.radii, items, out=active_weights)
        out[:, 1] -= active_weights


def _mapped_source_cover(
    state: EmbeddingState,
    source: SourceBoundaryQuery,
    centers: np.ndarray,
    radii: np.ndarray,
    errors: np.ndarray,
    forward_upper: float,
    forward_evaluations: int,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[SourceBoundSemantics, float, float, int]:
    samples = source.boundary_samples(limits.maximum_source_samples)
    count = samples.points.shape[0]
    if not samples.complete or count == 0:
        state.add("source_sample_capacity", "unresolved", "mesh")
        return samples.semantics, math.inf, 0.0, count
    if (
        count > limits.maximum_source_samples
        or forward_evaluations > limits.maximum_distance_evaluations
    ):
        state.add("distance_capacity", "unresolved", "mesh")
        return samples.semantics, math.inf, 0.0, count
    if not math.isfinite(forward_upper) or centers.shape[0] == 0:
        return samples.semantics, math.inf, 0.0, count

    meter = _DistanceMeter(
        budget, limits.maximum_distance_evaluations, forward_evaluations
    )
    owners: list[object] = []

    try:
        with budget.measurement(), budget.temporary_scope():
            owners.append((centers, radii, errors, samples))
            # One immutable center bank; its median builder
            # admits construction visits and storage before execution.
            tree = prepare_bvh(
                centers,
                centers,
                dtype=np.float64,
                _charge_work=meter.work,
                _reserve_storage=meter.storage,
            )
            owners.append(tree)
            meter.work(2 * (samples.points.size + centers.size) + 2 * centers.shape[0])
            meter.storage(samples.points.nbytes + centers.nbytes)
            slack = (
                32.0
                * _EPSILON
                * max(
                    float(np.max(np.abs(samples.points))),
                    float(np.max(np.abs(centers))),
                    1.0,
                )
            )
            minimum_error, maximum_radius = float(np.min(errors)), float(np.max(radii))
            dimension, leaf_size = centers.shape[1], tree.leaf_size
            meter.storage(8 * (leaf_size * dimension + 2 * leaf_size + 2 * dimension))
            delta = np.empty((leaf_size, dimension), dtype=np.float64)
            distance = np.empty((leaf_size,), dtype=np.float64)
            weights = np.empty((leaf_size,), dtype=np.float64)
            node_delta = np.empty((dimension,), dtype=np.float64)
            node_other = np.empty((dimension,), dtype=np.float64)
            owners.append((delta, distance, weights, node_delta, node_other))

            kernel = _MappedDistanceKernel(
                centers,
                radii,
                errors,
                delta,
                distance,
                weights,
                node_delta,
                node_other,
                slack,
                minimum_error,
                maximum_radius,
            )

            minima, _, _, _ = bvh_host_minima(
                tree,
                samples.points,
                objective_count=2,
                node_lower_bounds=kernel.bounds,
                item_values=kernel.values,
                visit=meter.visit,
                admit_storage=meter.storage,
                retain_owner=owners.append,
            )
            meter.work(8 * count)
            meter.storage(3 * count * 8 + 4 * count)
            high = np.nextafter(minima[:, 0] + samples.covering_radius, math.inf)
            low = np.maximum(minima[:, 1] - samples.point_error, 0.0)
            upper, lower = float(np.max(high)), float(np.max(low))
            violated = (low > tolerance) & (samples.semantics == "certified")
            if np.any(violated):
                state.add(
                    "source_uncovered",
                    "violated",
                    "source_sample",
                    np.flatnonzero(violated),
                )
            unresolved = (high > tolerance) & ~violated
            if np.any(unresolved):
                state.add(
                    "source_uncovered",
                    "unresolved",
                    "source_sample",
                    np.flatnonzero(unresolved),
                )
    except _DistanceCapacity as failure:
        state.findings.append(
            MeshCertificateFinding(
                "distance_capacity",
                "unresolved",
                "mesh",
                resource="distance_evaluations",
                requested=(
                    ("limit", limits.maximum_distance_evaluations),
                    ("requested", failure.requested),
                ),
                achieved=(("completed", meter.evaluations),),
                observations=(("forward_distance_evaluations", forward_evaluations),),
            )
        )
        return samples.semantics, math.inf, 0.0, count
    except CoordinateEnclosureResourceError as failure:
        state.add(
            "source_cover_expression_budget",
            "unresolved",
            "mesh",
            resource_error=failure,
            expression_budget=budget,
        )
        return samples.semantics, math.inf, 0.0, count
    return samples.semantics, upper, lower, count


@final
class _AnalyticBoundaryProof(StrictModule):
    distance_upper: float = eqx.field(static=True)
    boundary_degree_id: str | None = eqx.field(static=True)
    findings: tuple[MeshCertificateFinding, ...]
    piece_count: int = eqx.field(static=True)
    cad_closure: CadBoundaryClosureEvidence | None = None


type _AnalyticPoint = tuple[Fraction, ...]


type _AnalyticPolynomialKey = tuple[tuple[tuple[int, ...], int, int], ...]


_AnalyticProjectionKind: TypeAlias = Literal["cylinder", "circle", "line"]


def _analytic_polynomial_key(
    value: Polynomial, /, *, budget: CoordinateEnclosureBudget
) -> _AnalyticPolynomialKey:

    budget.reserve(
        2 * len(value) * (len(value).bit_length() + 1),
        256 + 256 * len(value),
    )
    return tuple(
        (index, coefficient.numerator, coefficient.denominator)
        for index, value_ in sorted(value.items())
        for coefficient in (
            rational_coefficient(value_, role="source_fidelity.analytic_coefficient"),
        )
        if coefficient
    )


def _analytic_corners(
    item: FidelityMap, /, *, budget: CoordinateEnclosureBudget
) -> tuple[_AnalyticPoint, ...]:

    reference = ((Fraction(0),) * item.dimension,) + tuple(
        tuple(Fraction(int(axis == j)) for j in range(item.dimension))
        for axis in range(item.dimension)
    )
    return tuple(
        tuple(
            rational_coefficient(
                coordinate_enclosure_owner.expression_evaluate(
                    value, point, budget=budget
                ),
                role="source_fidelity.analytic_corner",
            )
            for value in item.coordinates
        )
        for point in reference
    )


def _analytic_exact_binary64_proxy(
    corners: tuple[tuple[_AnalyticPoint, ...], ...],
    center: _AnalyticPoint,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    """Admit dyadic degree predicates only for exactly representable proxy coordinates."""
    budget.reserve(len(center) + sum(len(point) for item in corners for point in item))
    maximum = Fraction(float(np.finfo(np.float64).max))
    return all(
        abs(value) <= maximum and Fraction(float(value)) == value
        for points in ((center,), *corners)
        for point in points
        for value in point
    )


def _analytic_closed_chain(
    maps: list[FidelityMap], /, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Check oriented polynomial edge-restriction cancellation."""

    directed: dict[tuple[_AnalyticPolynomialKey, ...], int] = {}
    for item in maps:
        reference = np.asarray(((0, 0), (1, 0), (0, 1)), dtype=np.float64)
        for first, last in ((0, 1), (1, 2), (2, 0)):
            arguments = coordinate_enclosure_owner.affine_arguments(
                reference[first],
                (reference[last] - reference[first])[:, None],
                budget=budget,
            )
            reverse = coordinate_enclosure_owner.affine_arguments(
                reference[last],
                (reference[first] - reference[last])[:, None],
                budget=budget,
            )
            keys = []
            for chart in (arguments, reverse):
                row = []
                for value in item.coordinates:
                    restricted = coordinate_enclosure_owner.expression_compose(
                        value, chart, budget=budget
                    )
                    match restricted:
                        case dict():
                            row.append(
                                _analytic_polynomial_key(restricted, budget=budget)
                            )
                        case coordinate_enclosure_owner.RationalPolynomial():
                            return False
                        case invalid:
                            assert_never(invalid)
                keys.append(tuple(row))
            if keys[0] == keys[1]:
                return False
            if directed.get(keys[1], 0):
                directed[keys[1]] -= 1
            else:
                directed[keys[0]] = directed.get(keys[0], 0) + 1
    return not any(directed.values())


def _analytic_planar_cycles(
    maps: list[FidelityMap], /, *, budget: CoordinateEnclosureBudget
) -> list[list[int]] | None:
    """Partition an oriented rational endpoint chain into nonbranching closed loops."""
    outgoing: dict[_AnalyticPoint, int] = {}
    incoming: dict[_AnalyticPoint, int] = {}
    ends = [_analytic_corners(item, budget=budget) for item in maps]
    for index, (first, last) in enumerate(ends):
        if first == last or first in outgoing or last in incoming:
            return None
        outgoing[first], incoming[last] = index, index
    if outgoing.keys() != incoming.keys():
        return None
    remaining = set(range(len(maps)))
    cycles: list[list[int]] = []
    while remaining:
        first = min(remaining)
        current = first
        cycle = []
        while current in remaining:
            remaining.remove(current)
            cycle.append(current)
            current = outgoing[ends[current][1]]
        if current != first:
            return None
        cycles.append(cycle)
    return cycles


def _analytic_squared(
    values: tuple[Expression, ...], /, *, budget: CoordinateEnclosureBudget
) -> Expression:

    return coordinate_enclosure_owner.expression_sum(
        tuple(
            coordinate_enclosure_owner.expression_multiply(value, value, budget=budget)
            for value in values
        ),
        budget=budget,
    )


def _analytic_relative(
    coordinates: tuple[Expression, ...],
    center: _AnalyticPoint,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:

    return tuple(
        coordinate_enclosure_owner.expression_add(
            value,
            coordinate_enclosure_owner.constant(-origin, dimension, budget=budget),
            budget=budget,
        )
        for value, origin in zip(coordinates, center, strict=True)
    )


def _analytic_homotopy(
    item: FidelityMap, center: _AnalyticPoint, /, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[Expression, ...], Expression]:

    corners = np.asarray(
        [
            [float(value) for value in point]
            for point in _analytic_corners(item, budget=budget)
        ],
        dtype=np.float64,
    )
    relative = _analytic_relative(item.coordinates, center, item.dimension, budget=budget)
    proxy = _analytic_relative(
        fidelity_affine(corners), center, item.dimension, budget=budget
    )
    dot = coordinate_enclosure_owner.expression_sum(
        tuple(
            coordinate_enclosure_owner.expression_multiply(first, second, budget=budget)
            for first, second in zip(relative, proxy, strict=True)
        ),
        budget=budget,
    )
    return relative, dot


def _analytic_distance_bound(
    kind: _AnalyticProjectionKind,
    coordinates: tuple[Expression, ...],
    dimension: int,
    radius: Fraction,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, bool]:
    """Outward bounds of radial/affine projection displacement and positive-dot homotopy."""

    intervals = tuple(
        coordinate_enclosure_owner.expression_bounds(
            value, "simplex", dimension, budget=budget
        )
        for value in coordinates
    )
    positive = intervals[-1][0] > 0.0
    match kind:
        case "line":
            tangent_controls = (
                coordinate_enclosure_owner.expression_bernstein_coefficients(
                    coordinates[1], "simplex", dimension, budget=budget
                )
            )
            positive = min(tangent_controls) >= 0 and max(tangent_controls) <= 1
            if not coordinates[0]:
                return 0.0, positive
            bound = float(np.nextafter(math.sqrt(max(0.0, intervals[0][1])), math.inf))
        case "circle":
            low = max(
                0.0, float(np.nextafter(math.sqrt(max(0.0, intervals[0][0])), -math.inf))
            )
            high = float(np.nextafter(math.sqrt(max(0.0, intervals[0][1])), math.inf))
            bound = float(
                np.nextafter(
                    max(
                        high - fraction_endpoint(radius, upper=False, saturate=True),
                        fraction_endpoint(radius, upper=True, saturate=True) - low,
                        0.0,
                    ),
                    math.inf,
                )
            )
        case "cylinder":
            radial_low = max(
                0.0, float(np.nextafter(math.sqrt(max(0.0, intervals[0][0])), -math.inf))
            )
            radial_high = float(
                np.nextafter(math.sqrt(max(0.0, intervals[0][1])), math.inf)
            )
            axial_low, axial_high = intervals[2]
            axial_minimum = max(axial_low, -axial_high, 0.0)
            gauge_low = max(radial_low, axial_minimum)
            gauge_high = max(radial_high, abs(axial_low), abs(axial_high))
            if gauge_low <= 0.0:
                return math.inf, positive
            norm_high = float(
                np.nextafter(math.sqrt(max(0.0, intervals[1][1])), math.inf)
            )
            factor = float(np.nextafter(norm_high / gauge_low, math.inf))
            error = float(
                np.nextafter(max(1.0 - gauge_low, gauge_high - 1.0, 0.0), math.inf)
            )
            bound = float(np.nextafter(factor * error, math.inf))
        case invalid:
            assert_never(invalid)
    return bound, positive


def _analytic_projection_witness_refusal(
    kind: _AnalyticProjectionKind,
    coordinates: tuple[Expression, ...],
    dimension: int,
    radius: Fraction,
    tolerance: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Refuse an insufficient projection bound without asserting nearest-boundary distance."""

    budget.reserve(8, 1024)
    points = (
        ((Fraction(1, 4),), (Fraction(3, 4),))
        if dimension == 1
        else ((Fraction(1, 3), Fraction(1, 3)),)
    )
    threshold = Fraction(tolerance)
    for point in points:
        values = tuple(
            rational_coefficient(
                coordinate_enclosure_owner.expression_evaluate(
                    value, point, budget=budget
                ),
                role="source_fidelity.projection_witness",
            )
            for value in coordinates
        )
        bits = max(
            1075,
            *(
                max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                for value in (threshold, radius, *values)
            ),
        )
        budget.reserve(64, 128 * bigint_bytes(8 * bits + 8))
        match kind:
            case "line":
                if values[0] > threshold**2:
                    return True
            case "circle":
                if values[0] > (radius + threshold) ** 2 or (
                    radius > threshold and values[0] < (radius - threshold) ** 2
                ):
                    return True
            case "cylinder":
                low_squared = fraction_endpoint(values[0], upper=False, saturate=True)
                high_squared = fraction_endpoint(values[0], upper=True, saturate=True)
                if not math.isfinite(high_squared):
                    continue
                radial_low = Fraction(
                    max(
                        0.0,
                        float(np.nextafter(math.sqrt(max(0.0, low_squared)), -math.inf)),
                    )
                )
                radial_high = Fraction(
                    float(np.nextafter(math.sqrt(max(0.0, high_squared)), math.inf))
                )
                gauge_low = max(radial_low, abs(values[2]))
                gauge_high = max(radial_high, abs(values[2]))
                if gauge_high > 0:
                    error_lower = max(gauge_low - 1, 1 - gauge_high, Fraction(0))
                    if values[1] * error_lower**2 > threshold**2 * gauge_high**2:
                        return True
            case invalid:
                assert_never(invalid)
    return False


def _analytic_refine(
    state: EmbeddingState,
    item: FidelityMap,
    kind: _AnalyticProjectionKind,
    radius: Fraction,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Enclose every source-polynomial point with a bounded live subdivision worklist."""
    pending = [(item, 0)]
    upper = 0.0
    while pending:
        if state.subdivision_pieces >= min(
            limits.maximum_subdivision_pieces, limits.maximum_distance_evaluations
        ):
            state.add("analytic_boundary_piece_capacity", "unresolved", "mesh")
            return math.inf
        current, depth = pending.pop()
        state.subdivision_pieces += 1
        with budget.temporary_scope():
            live = tuple(
                value
                for piece in (current, *(piece for piece, _ in pending))
                for value in piece.coordinates
            )
            budget.reserve(1, budget.live_object_storage_upper(live))
            if (
                _fidelity_node_count(current.coordinates, current.dimension)
                > limits.maximum_bernstein_nodes
            ):
                state.add(
                    "analytic_boundary_bernstein_capacity",
                    "unresolved",
                    current.entity_kind,
                    (current.entity,),
                )
                return math.inf
            bound, positive = _analytic_distance_bound(
                kind, current.coordinates, current.dimension, radius, budget=budget
            )
            if positive and bound <= tolerance:
                upper = max(upper, bound)
                continue
            if _analytic_projection_witness_refusal(
                kind,
                current.coordinates,
                current.dimension,
                radius,
                tolerance,
                budget=budget,
            ):
                state.add(
                    "analytic_boundary_projection_witness",
                    "unresolved",
                    current.entity_kind,
                    (current.entity,),
                )
                return math.inf
            if depth >= limits.maximum_subdivision_depth:
                state.add(
                    "analytic_boundary_deviation"
                    if positive
                    else "analytic_boundary_homotopy_premise",
                    "unresolved",
                    current.entity_kind,
                    (current.entity,),
                )
                upper = max(upper, bound)
                continue
            children = 2 if current.dimension == 1 else 4
            if (
                state.subdivision_pieces + len(pending) + children
                > limits.maximum_subdivision_pieces
            ):
                state.add("analytic_boundary_piece_capacity", "unresolved", "mesh")
                return math.inf
            pending.extend(
                (child, depth + 1) for child in split_fidelity_map(current, budget=budget)
            )
    return upper


def _analytic_cylinder_proof(
    state: EmbeddingState,
    maps: list[FidelityMap],
    frame: CylinderSourceRegion,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:

    if not _analytic_closed_chain(maps, budget=budget):
        state.add("analytic_boundary_exact_closed_trace_premise", "unresolved", "mesh")
        return math.inf
    exact_corners = tuple(_analytic_corners(item, budget=budget) for item in maps)
    if not _analytic_exact_binary64_proxy(exact_corners, frame.center, budget):
        state.add("analytic_boundary_proxy_rounding_premise", "unresolved", "mesh")
        return math.inf
    triangles = np.asarray(
        [
            [[float(value) for value in point] for point in corners]
            for corners in exact_corners
        ],
        dtype=np.float64,
    )
    center = np.asarray([float(value) for value in frame.center], dtype=np.float64)
    budget.reserve(20 * triangles.size, 256 * triangles.size)
    points, inverse = np.unique(triangles.reshape((-1, 3)), axis=0, return_inverse=True)
    if 2 * len(maps) > limits.maximum_ray_tests:
        state.add("analytic_boundary_ray_capacity", "unresolved", "mesh")
        return math.inf
    state.ray_tests += 2 * len(maps)
    degree, certain = ray_crossings_3d(center, triangles, budget=budget)
    if (
        not certain
        or degree != 1
        or not domain_numerics_owner.sphere_radial_degree_one(
            (center, 1.0, 1.0), points, inverse.reshape((-1, 3))
        )
    ):
        state.add("analytic_boundary_oriented_degree_one_premise", "unresolved", "mesh")
        return math.inf
    bounds = []
    for item in maps:
        with budget.temporary_scope():
            relative, dot = _analytic_homotopy(item, frame.center, budget=budget)
            local = tuple(
                coordinate_enclosure_owner.expression_sum(
                    tuple(
                        coordinate_enclosure_owner.expression_scale(
                            value, weight, budget=budget
                        )
                        for value, weight in zip(relative, axis, strict=True)
                    ),
                    budget=budget,
                )
                for axis in frame.axes
            )
            radial = coordinate_enclosure_owner.expression_scale(
                _analytic_squared(local[:2], budget=budget),
                1 / frame.radius**2,
                budget=budget,
            )
            axial = coordinate_enclosure_owner.expression_scale(
                local[2], 1 / frame.halfheight, budget=budget
            )
            scalars = FidelityMap(
                (radial, _analytic_squared(relative, budget=budget), axial, dot),
                item.dimension,
                item.entity,
                item.entity_kind,
            )
            bounds.append(
                _analytic_refine(
                    state,
                    scalars,
                    "cylinder",
                    frame.radius,
                    tolerance,
                    limits,
                    budget=budget,
                )
            )
        if state.findings:
            return math.inf
    return max(bounds, default=math.inf)


@dataclass(frozen=True)
class _PlanarCircleFrame:
    center: _AnalyticPoint
    radius: Fraction
    orientation: int


type _PlanarLineLoop = tuple[tuple[_AnalyticPoint, _AnalyticPoint], ...]


@final
class _PlanarSourcePolygon(StrictModule):
    outer: _PlanarLineLoop = eqx.field(static=True)
    points: np.ndarray
    edges: np.ndarray


def _analytic_planar_polygon(
    source: domain_planar_boundary_owner.EmbeddedPlanarBoundarySource,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PlanarSourcePolygon | None:
    outer: list[tuple[_AnalyticPoint, _AnalyticPoint]] = []
    for edge in source.loop_edges[0]:
        match source.curves[edge]:
            case patches_owner.LineCurve() as curve:
                origin = rational_vector(np.asarray(curve.origin))
                direction = rational_vector(np.asarray(curve.direction))
                first, last = source.parameter_ranges[edge]
                endpoints = (
                    tuple(
                        o + Fraction(first) * d
                        for o, d in zip(origin, direction, strict=True)
                    ),
                    tuple(
                        o + Fraction(last) * d
                        for o, d in zip(origin, direction, strict=True)
                    ),
                )
                outer.append(endpoints)
            case _:
                return None
    if len(outer) < 3 or any(
        first == last or last != outer[(index + 1) % len(outer)][0]
        for index, (first, last) in enumerate(outer)
    ):
        return None
    points = np.asarray([first for first, _ in outer], dtype=object)
    edges = np.asarray(
        [(i, (i + 1) % len(outer)) for i in range(len(outer))], dtype=np.int64
    )
    first_pairs, last_pairs = np.triu_indices(len(outer), 1)
    with exact_pair_contacts(
        points, edges[first_pairs], edges[last_pairs], budget=budget
    ) as contacts:
        try:
            if np.any(contacts[0]) or not np.all(contacts[1]):
                return None
        finally:
            del contacts
    area = sum(
        (first[0] * last[1] - first[1] * last[0] for first, last in outer), Fraction(0)
    )
    if area <= 0:
        return None
    return _PlanarSourcePolygon(tuple(outer), points, edges)


def _analytic_planar_hole_separation(
    outer: _PlanarLineLoop,
    points: np.ndarray,
    edges: np.ndarray,
    holes: list[_PlanarCircleFrame],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    for index, hole in enumerate(holes):
        integers, _ = exact_polyhedral_geometry_owner.coordinate_integers(
            np.concatenate((np.asarray(hole.center, dtype=object)[None], points)),
            budget=budget,
        )
        degree, certain = ray_crossings_2d(
            integers[0], integers[1:][edges], budget=budget
        )
        if not certain or degree != 1:
            return False
        for first, last in outer:
            direction = tuple(b - a for a, b in zip(first, last, strict=True))
            relative = tuple(c - a for c, a in zip(hole.center, first, strict=True))
            length_squared = sum((v * v for v in direction), Fraction(0))
            parameter = max(
                Fraction(0),
                min(
                    Fraction(1),
                    sum(
                        (a * b for a, b in zip(relative, direction, strict=True)),
                        Fraction(0),
                    )
                    / length_squared,
                ),
            )
            squared = sum(
                (
                    (a - parameter * b) ** 2
                    for a, b in zip(relative, direction, strict=True)
                ),
                Fraction(0),
            )
            if squared <= hole.radius**2:
                return False
        for other in holes[:index]:
            if (
                sum(
                    (
                        (a - b) ** 2
                        for a, b in zip(hole.center, other.center, strict=True)
                    ),
                    Fraction(0),
                )
                <= (hole.radius + other.radius) ** 2
            ):
                return False
    return True


def _analytic_planar_source(
    source: domain_planar_boundary_owner.EmbeddedPlanarBoundarySource,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_PlanarLineLoop, tuple[_PlanarCircleFrame, ...]] | None:
    """Admit a simple line polygon and disjoint, strictly interior full-circle holes."""

    if any(source.projection_errors) or not source.loop_edges:
        return None
    geometry = source.model.geometry
    if geometry is None:
        return None
    reversed_face = float(source.model.orientation[0]) < 0
    coedges = tuple(
        coedge
        for loop in geometry.face_loops[0]
        for coedge in (reversed(loop) if reversed_face else loop)
    )
    polygon = _analytic_planar_polygon(source, budget=budget)
    if polygon is None:
        return None
    outer, points, edges = polygon.outer, polygon.points, polygon.edges
    holes: list[_PlanarCircleFrame] = []
    for loop in source.loop_edges[1:]:
        if len(loop) != 1:
            return None
        edge = loop[0]
        match source.curves[edge]:
            case patches_owner.CircleCurve() as curve:
                first, last = source.parameter_ranges[edge]
                if not native_full_turn(source.model, coedges[edge], zero_phase=False):
                    return None
                axes = tuple(
                    rational_vector(np.asarray(axis))
                    for axis in (curve.first_axis, curve.second_axis)
                )
                if any(
                    sum((a * b for a, b in zip(x, y, strict=True)), Fraction(0))
                    != int(i == j)
                    for i, x in enumerate(axes)
                    for j, y in enumerate(axes)
                ):
                    return None
                determinant = axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0]
                orientation = 1 if determinant * Fraction(last - first) > 0 else -1
                if orientation != -1:
                    return None
                holes.append(
                    _PlanarCircleFrame(
                        rational_vector(np.asarray(curve.center)),
                        Fraction(float(curve.radius)),
                        orientation,
                    )
                )
            case _:
                return None
    if not _analytic_planar_hole_separation(outer, points, edges, holes, budget=budget):
        return None
    return tuple(outer), tuple(holes)


def _analytic_line_match(
    item: FidelityMap, source: _PlanarLineLoop, /, *, budget: CoordinateEnclosureBudget
) -> tuple[int, Fraction, Fraction] | None:
    """Match oriented rational endpoints to one independent source segment."""
    endpoints = _analytic_corners(item, budget=budget)
    matches = []
    for edge, (first, last) in enumerate(source):
        direction = tuple(b - a for a, b in zip(first, last, strict=True))
        axis = next(i for i, value in enumerate(direction) if value)
        parameters = tuple(
            (point[axis] - first[axis]) / direction[axis] for point in endpoints
        )
        if 0 <= parameters[0] < parameters[1] <= 1 and all(
            tuple(a + t * d for a, d in zip(first, direction, strict=True)) == point
            for point, t in zip(endpoints, parameters, strict=True)
        ):
            matches.append((edge, parameters[0], parameters[1]))
    return matches[0] if len(matches) == 1 else None


def _analytic_line_partition(
    maps: list[FidelityMap],
    cycle: list[int],
    source: _PlanarLineLoop,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Verify oriented rational interval partitions of every independent source segment."""
    intervals: list[list[tuple[Fraction, Fraction]]] = [[] for _ in source]
    for index in cycle:
        match = _analytic_line_match(maps[index], source, budget=budget)
        if match is None:
            return False
        edge, first, last = match
        intervals[edge].append((first, last))
    for bank in intervals:
        ordered = sorted(bank)
        if not ordered or ordered[0][0] != 0 or ordered[-1][1] != 1:
            return False
        if any(first[1] != last[0] for first, last in zip(ordered, ordered[1:])):
            return False
    return True


def _analytic_circle_degree(
    state: EmbeddingState,
    maps: list[FidelityMap],
    cycle: list[int],
    frame: _PlanarCircleFrame,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    if state.ray_tests + len(cycle) > limits.maximum_ray_tests:
        state.add("analytic_boundary_ray_capacity", "unresolved", "mesh")
        return False
    state.ray_tests += len(cycle)
    budget.reserve(32 * len(cycle), 2048 * len(cycle))
    exact_corners = tuple(
        _analytic_corners(maps[index], budget=budget) for index in cycle
    )
    if not _analytic_exact_binary64_proxy(exact_corners, frame.center, budget):
        state.add("analytic_boundary_proxy_rounding_premise", "unresolved", "mesh")
        return False
    segments = np.asarray(
        [
            [[float(value) for value in point] for point in corners]
            for corners in exact_corners
        ],
        dtype=np.float64,
    )
    center = np.asarray([float(v) for v in frame.center], dtype=np.float64)
    integers, _ = dyadic_integers(
        np.concatenate((center[None], segments.reshape((-1, 2))))
    )
    relative = integers[1:].reshape((-1, 2, 2)) - integers[0]
    signed = relative[:, 0, 0] * relative[:, 1, 1] - relative[:, 0, 1] * relative[:, 1, 0]
    if not np.all(signed * frame.orientation > 0):
        return False
    degree, certain = ray_crossings_2d(center, segments, budget=budget)
    return certain and degree == frame.orientation


def _analytic_planar_proof(
    state: EmbeddingState,
    maps: list[FidelityMap],
    source: tuple[_PlanarLineLoop, tuple[_PlanarCircleFrame, ...]],
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:

    cycles = _analytic_planar_cycles(maps, budget=budget)
    outer, holes = source
    if cycles is None or len(cycles) != 1 + len(holes):
        state.add("analytic_boundary_complete_loop_premise", "unresolved", "mesh")
        return math.inf
    matches = [
        index
        for index, cycle in enumerate(cycles)
        if _analytic_line_partition(maps, cycle, outer, budget=budget)
    ]
    if len(matches) != 1:
        state.add("analytic_boundary_line_partition_premise", "unresolved", "mesh")
        return math.inf
    polygon = matches[0]
    assigned = {polygon}
    bounds = []
    for index in cycles[polygon]:
        item = maps[index]
        with budget.temporary_scope():
            match = _analytic_line_match(item, outer, budget=budget)
            if match is None:
                state.add(
                    "analytic_boundary_line_partition_premise", "unresolved", "mesh"
                )
                return math.inf
            first, last = outer[match[0]]
            direction = tuple(b - a for a, b in zip(first, last, strict=True))
            length_squared = sum((value**2 for value in direction), Fraction(0))
            relative = _analytic_relative(item.coordinates, first, 1, budget=budget)
            tangent = coordinate_enclosure_owner.expression_scale(
                coordinate_enclosure_owner.expression_sum(
                    tuple(
                        coordinate_enclosure_owner.expression_scale(
                            value, weight, budget=budget
                        )
                        for value, weight in zip(relative, direction, strict=True)
                    ),
                    budget=budget,
                ),
                1 / length_squared,
                budget=budget,
            )
            residual = tuple(
                coordinate_enclosure_owner.expression_add(
                    value,
                    coordinate_enclosure_owner.expression_scale(
                        tangent, -weight, budget=budget
                    ),
                    budget=budget,
                )
                for value, weight in zip(relative, direction, strict=True)
            )
            scalars = FidelityMap(
                (_analytic_squared(residual, budget=budget), tangent),
                1,
                item.entity,
                item.entity_kind,
            )
            bounds.append(
                _analytic_refine(
                    state, scalars, "line", Fraction(0), tolerance, limits, budget=budget
                )
            )
        if state.findings:
            return math.inf
    for hole in holes:
        matches = [
            index
            for index, cycle in enumerate(cycles)
            if index not in assigned
            and _analytic_circle_degree(state, maps, cycle, hole, limits, budget=budget)
        ]
        if state.findings:
            return math.inf
        if len(matches) != 1:
            state.add(
                "analytic_boundary_circle_oriented_degree_premise", "unresolved", "mesh"
            )
            return math.inf
        assigned.add(matches[0])
        for index in cycles[matches[0]]:
            item = maps[index]
            with budget.temporary_scope():
                relative, dot = _analytic_homotopy(item, hole.center, budget=budget)
                scalars = FidelityMap(
                    (_analytic_squared(relative, budget=budget), dot),
                    1,
                    item.entity,
                    item.entity_kind,
                )
                bounds.append(
                    _analytic_refine(
                        state,
                        scalars,
                        "circle",
                        hole.radius,
                        tolerance,
                        limits,
                        budget=budget,
                    )
                )
            if state.findings:
                return math.inf
    return max(bounds, default=math.inf)


@final
class _AnalyticBoundaryChain(StrictModule):
    maps: list[FidelityMap]
    polynomial_coordinates: tuple[tuple[Polynomial, ...], ...]


def _analytic_boundary_chain(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AnalyticBoundaryChain:
    maps = _fidelity_maps(state, mesh, geometry, budget=budget)
    if not maps or any(item.dimension != mesh.ambient_dimension - 1 for item in maps):
        state.add(
            "analytic_boundary_source_expression_premise",
            "unresolved",
            "mesh",
        )
    if any(
        isinstance(
            value,
            coordinate_enclosure_owner.RationalPolynomial,
        )
        for item in maps
        for value in item.coordinates
    ):
        state.add("analytic_boundary_polynomial_premise", "unresolved", "mesh")
    if len(maps) > min(
        limits.maximum_source_samples,
        limits.maximum_subdivision_pieces,
        limits.maximum_ray_tests,
    ):
        state.add("analytic_boundary_chain_capacity", "unresolved", "mesh")
    polynomial_coordinates: list[tuple[Polynomial, ...]] = []
    if not state.findings:
        for item in maps:
            row: list[Polynomial] = []
            for value in item.coordinates:
                match value:
                    case dict():
                        row.append(value)
                    case coordinate_enclosure_owner.RationalPolynomial():
                        state.add(
                            "analytic_boundary_polynomial_premise", "unresolved", "mesh"
                        )
                    case invalid:
                        assert_never(invalid)
            polynomial_coordinates.append(tuple(row))
    return _AnalyticBoundaryChain(maps, tuple(polynomial_coordinates))


def _analytic_boundary_degree_identity(
    chain: _AnalyticBoundaryChain,
    source: SourceBoundaryQuery,
    source_scope_id: str,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    limits: MeshCertificateLimits,
    tolerance: float,
    upper: float,
    state: EmbeddingState,
    projection_kind: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> str:
    payload = []
    for item, polynomials in zip(chain.maps, chain.polynomial_coordinates, strict=True):
        row = []

        for value in polynomials:
            row.append(_analytic_polynomial_key(value, budget=budget))
        payload.append(
            (
                item.dimension,
                item.entity_kind,
                item.entity,
                tuple(row),
            )
        )
    budget.reserve(
        sum(len(row[-1]) for row in payload),
        budget.live_object_storage_upper(tuple(payload)),
    )
    return canonical_fingerprint(
        {
            "kind": "analytic-source-boundary-degree",
            "theorem": projection_kind,
            "source": source.source_id,
            "source_revision": source.source_revision,
            "source_scope": source_scope_id,
            "mesh": mesh.mesh_id,
            "topology": mesh.topology_id,
            "geometry": cell_geometry_id(geometry),
            "boundary_maps": tuple(payload),
            "limits": limits.limits_id,
            "tolerance": repr(tolerance),
            "distance_upper": repr(upper),
            "pieces": state.subdivision_pieces,
        }
    )


@checked
def analytic_boundary_proof(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AnalyticBoundaryProof | None:
    """Return polynomial boundary projection bounds and signed-degree evidence.

    Explicit chart witnesses are independently checked and included in the
    degree identity. Their distance ribbons do not themselves establish a
    general curved-source homotopy, embedding or signed-degree theorem.
    """

    cylinder_model: BRepModel | None = None
    match source:
        case domain_planar_boundary_owner.EmbeddedPlanarBoundarySource():
            source_scope_id = source.source_scope_id
            source_size = len(source.curves)
        case domain_boundary_owner.MeshingDomainBoundarySource():
            # The closed-volume theorem is not selected for a surface target;
            # its own chart-chain certificate owns that comparison.
            if mesh.topological_dimension != mesh.ambient_dimension:
                return None
            if len(source.patches) != 3 or source.patches != tuple(
                range(len(source.domain.patches))
            ):
                return None
            match source.domain.brep_authority:
                case BRepModel() as model:
                    if source.domain.authority_id != model.model_id:
                        return None
                    cylinder_model = model
                case _:
                    return None
            source_scope_id = source.source_scope_id
            source_size = len(source.domain.curves) + len(source.domain.patches)
        case _:
            return None
    state = EmbeddingState([], ["analytic_boundary_projection"])

    upper = math.inf
    degree_id = None
    try:
        with budget.measurement(), budget.activate(), budget.temporary_scope():
            if source_size**2 > min(
                limits.maximum_candidate_pairs, limits.maximum_ray_tests
            ):
                state.add("analytic_boundary_source_chain_capacity", "unresolved", "mesh")
                return _AnalyticBoundaryProof(math.inf, None, tuple(state.findings), 0)
            budget.reserve(2048 * source_size**2, 8192 * max(1, source_size))
            frame = (
                complete_cylinder_region(cylinder_model, budget=budget)
                if cylinder_model is not None
                else None
            )
            planar = (
                _analytic_planar_source(source, budget=budget)
                if isinstance(
                    source, domain_planar_boundary_owner.EmbeddedPlanarBoundarySource
                )
                else None
            )
            if frame is None and planar is None:
                return None
            # Missing retained gap tables are not evidence of coincidence. The
            # closed-form cylinder admission itself earns exact zero rows: it
            # requires exact rational harmonic correspondence of every ring and
            # cap coedge, an exact seam line and exact ring vertices. A gapped
            # source therefore never reaches this shortcut; its positive bounds
            # need the separately identified chart-closure representative.
            if (
                isinstance(source, domain_boundary_owner.MeshingDomainBoundarySource)
                and source.chart_triangulations
            ):
                # A supplied chart witness is part of the scientific request,
                # even when an independent closed-form projection exists.
                # Never silently discard a gapped, reversed or foreign chain.
                if tuple(sorted(row[0] for row in source.chart_triangulations)) != (
                    source.patches
                ):
                    state.add("source_trim_coverage_premise", "unresolved", "mesh")
                else:
                    cover = source.boundary_chart_cover(
                        min(
                            limits.maximum_source_samples,
                            limits.maximum_subdivision_pieces,
                        ),
                        budget=budget,
                    )
                    state.findings.extend(cover.findings)
                    if not cover.complete or cover.semantics != "certified":
                        state.add("source_trim_coverage_premise", "unresolved", "mesh")
                    source_scope_id = canonical_fingerprint(
                        {
                            "source_scope": source_scope_id,
                            "chart_cover": cover.cover_id,
                            "chart_patches": tuple(
                                row[0] for row in source.chart_triangulations
                            ),
                            "chart_witness": array_tree_fingerprint(
                                tuple(row[1:] for row in source.chart_triangulations)
                            ),
                        }
                    )
                if state.findings:
                    return _AnalyticBoundaryProof(
                        math.inf, None, tuple(state.findings), 0
                    )
            if (
                mesh.topological_dimension != mesh.ambient_dimension
                or mesh.periodic_topology is not None
            ):
                state.add("analytic_boundary_volume_chain_premise", "unresolved", "mesh")
                return _AnalyticBoundaryProof(math.inf, None, tuple(state.findings), 0)
            # Geometry-bound facets charge their build once per request and
            # are shared with the embedding and coverage certificates.
            facets = mesh_facets(mesh, geometry=geometry, budget=budget)
            count = int(np.count_nonzero(facets.boundary))
            if count > min(
                limits.maximum_source_samples,
                limits.maximum_subdivision_pieces,
                limits.maximum_ray_tests,
            ):
                state.add("analytic_boundary_chain_capacity", "unresolved", "mesh")
            else:
                chain = _analytic_boundary_chain(
                    state, mesh, geometry, limits, budget=budget
                )
                maps = chain.maps
                if not state.findings:
                    if frame is not None:
                        upper = _analytic_cylinder_proof(
                            state, maps, frame, tolerance, limits, budget=budget
                        )
                    elif planar is not None:
                        upper = _analytic_planar_proof(
                            state, maps, planar, tolerance, limits, budget=budget
                        )
                    if not state.findings:
                        degree_id = _analytic_boundary_degree_identity(
                            chain,
                            source,
                            source_scope_id,
                            mesh,
                            geometry,
                            limits,
                            tolerance,
                            upper,
                            state,
                            "cylinder-gauge"
                            if frame is not None
                            else "planar-line-circle",
                            budget=budget,
                        )
    except CoordinateEnclosureResourceError as error:
        state.add(
            "analytic_boundary_source_expression_resource",
            "unresolved",
            "mesh",
            resource_error=error,
            expression_budget=budget,
        )
    except RationalEnclosureError:
        state.add("analytic_boundary_source_expression_premise", "unresolved", "mesh")
    return _AnalyticBoundaryProof(
        upper, degree_id, tuple(state.findings), state.subdivision_pieces
    )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _PreparedChartBoundarySource:
    """Issued source-only premises retained by one live coordinate request.

    No value points back to its budget. The weak issuer therefore cannot keep
    a request alive through its retained proof. Neither target coordinate maps
    nor target correspondence or homotopy findings belong in this preparation.
    """

    cover: SourceBoundaryChartCover
    maps: tuple[tuple[domain_chart_chain_owner.SourceChartMap, ...], ...]


_ISSUED_CHART_BOUNDARY_SOURCE: IssuedProofs[
    CoordinateEnclosureBudget, _PreparedChartBoundarySource
] = IssuedProofs()


def _prepare_chart_boundary_source(
    source: domain_boundary_owner.MeshingDomainBoundarySource,
    maximum_patches: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PreparedChartBoundarySource:
    """Earn source coverage/native maps once, authenticating live content on reuse."""
    with budget.temporary_scope():
        # This includes root/static metadata, all source carriers and every
        # chart/provenance/restriction array, not their asserted identities.
        admit_carrier_fingerprint(source, budget)
        identity = canonical_fingerprint(
            {
                "source": carrier_payload(source),
                "maximum_patches": maximum_patches,
            }
        )
    issued = _ISSUED_CHART_BOUNDARY_SOURCE.get(budget, identity)
    if issued is not None:
        budget.ticket.checkpoint()
        return issued
    cover = source.boundary_chart_cover(maximum_patches, budget=budget)
    maps: tuple[tuple[domain_chart_chain_owner.SourceChartMap, ...], ...] = ()
    if cover.complete and cover.semantics == "certified" and not cover.findings:
        native = domain_chart_chain_owner.prepare_source_chart_domain(
            source.domain, budget=budget
        )
        maps = tuple(
            domain_chart_chain_owner.prepare_source_chart_maps(
                source.domain, data, budget=budget, preparations=native
            )
            for data in source.chart_triangulations
        )
    prepared = _PreparedChartBoundarySource(cover, maps)
    if maps:
        # Admit the entire retained graph before publishing reuse authority.
        # Incomplete/refused source proofs never enter the issuer.
        budget.retain_owner(prepared)
        _ISSUED_CHART_BOUNDARY_SOURCE.record(budget, identity, prepared)
    return prepared


def _chart_boundary_cells(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: domain_boundary_owner.MeshingDomainBoundarySource,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    source_maps: tuple[tuple[domain_chart_chain_owner.SourceChartMap, ...], ...],
) -> tuple[embedding_cells_owner.IntervalBoundaryCell, ...]:
    """Match every root chart triangle to one outward target boundary facet."""
    if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
        state.add("source_chart_facet_lineage_premise", "unresolved", "mesh")
        return ()
    facets = mesh_facets(mesh, geometry=geometry, budget=budget)
    occurrences = np.flatnonzero(facets.boundary)
    count = occurrences.size
    if not count or count > min(
        limits.maximum_source_samples, limits.maximum_subdivision_pieces
    ):
        state.add("source_chart_capacity", "unresolved", "mesh")
        return ()
    # Pre-admit the request-local facet, vertex and map records. This is
    # per-created-entity storage/work, not a hash/sort/instruction proxy.
    budget.reserve(int(count), 4096 + 4096 * int(count) + 8 * mesh.coordinates.size)
    points = np.asarray(jax.device_get(mesh.coordinates), dtype=np.float64)
    rows = facets.rows[occurrences, :3]
    identifiers = facet_entity_ids(mesh, rows)
    selected = tuple(sorted(int(value) for value in identifiers))
    maps = _fidelity_maps(
        state, mesh, geometry, target_facet_ids=selected, budget=budget
    )
    if state.findings or len(maps) != count:
        state.add("source_chart_facet_lineage_premise", "unresolved", "mesh")
        return ()
    by_identifier = {item.entity: item for item in maps}
    if len(by_identifier) != count:
        state.add("source_chart_facet_lineage_premise", "unresolved", "mesh")
        return ()
    vertices = np.unique(rows)
    point_rows: dict[tuple[float, ...], int] = {}
    for vertex in vertices:
        key = tuple(float(value) for value in points[vertex])
        if key in point_rows:
            state.add("source_chart_facet_lineage_premise", "unresolved", "mesh")
            return ()
        point_rows[key] = int(vertex)
    boundary = {
        tuple(sorted(int(value) for value in row)): (int(identifier), int(occurrence))
        for row, identifier, occurrence in zip(
            rows, identifiers, occurrences, strict=True
        )
    }
    reference = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    exact_reference = tuple(
        tuple(Fraction(float(value)) for value in row) for row in reference
    )
    map_vertices: dict[int, tuple[int, ...]] = {}
    for item in maps:
        corners = []
        for point in exact_reference:
            coordinate_key: list[float] = []
            for expression in item.coordinates:
                value = coordinate_enclosure_owner.expression_evaluate(
                    expression, point, budget=budget
                )
                if not isinstance(value, Fraction):
                    break
                rounded = float(value)
                if not math.isfinite(rounded) or Fraction(rounded) != value:
                    break
                coordinate_key.append(rounded)
            key = tuple(coordinate_key)
            if len(key) != 3 or key not in point_rows:
                state.add(
                    "source_chart_facet_lineage_premise",
                    "unresolved",
                    "facet",
                    (item.entity,),
                )
                return ()
            corners.append(point_rows[key])
        map_vertices[item.entity] = tuple(corners)
    stops = np.cumsum([block.cell_count for block in mesh.blocks])
    connectivity = tuple(np.asarray(block.vertices) for block in mesh.blocks)
    global_vertices = np.asarray(mesh.vertex_global_ids)
    region_sides = dict(source.domain.regions[0].boundary)
    seen: set[int] = set()
    result = []
    for data, patch_maps in zip(source.chart_triangulations, source_maps, strict=True):
        patch, _, chart_points, chart_cells, *_ = data
        if len(patch_maps) != len(chart_cells):
            state.add("source_chart_map_premise", "unresolved", "source_facet", (patch,))
            return ()
        orientation = region_sides[patch] * (
            -1 if source.domain.patches[patch].reversed else 1
        )
        for source_map, cell in zip(patch_maps, chart_cells, strict=True):
            anchor = chart_points[cell]
            keys = tuple(tuple(float(value) for value in point) for point in anchor)
            if any(key not in point_rows for key in keys):
                state.add("source_chart_facet_lineage_premise", "unresolved", "source_facet", (patch,))
                return ()
            target_rows = tuple(point_rows[key] for key in keys)
            match = boundary.get(tuple(sorted(target_rows)))
            if match is None or match[0] in seen or len(set(target_rows)) != 3:
                state.add("source_chart_facet_lineage_premise", "unresolved", "source_facet", (patch,))
                return ()
            identifier, occurrence = match
            seen.add(identifier)
            owner = int(facets.cells[occurrence])
            block = int(np.searchsorted(stops, owner, side="right"))
            start = 0 if block == 0 else int(stops[block - 1])
            parent_vertices = tuple(int(value) for value in connectivity[block][owner - start])
            local = tuple(parent_vertices.index(vertex) for vertex in target_rows)
            opposite = next(index for index in range(4) if index not in local)
            permutation = (*local, opposite)
            inversions = sum(
                permutation[i] > permutation[j]
                for i in range(4)
                for j in range(i + 1, 4)
            )
            # The reference tetrahedron is positively oriented. Its boundary
            # normal points away from the omitted reference vertex.
            outward = -1 if inversions % 2 == 0 else 1
            if orientation != outward:
                state.add("source_chart_orientation_premise", "unresolved", "facet", (identifier,))
                return ()
            item = by_identifier[identifier]
            ordered = map_vertices[identifier]
            if set(ordered) != set(target_rows):
                state.add("source_chart_facet_lineage_premise", "unresolved", "facet", (identifier,))
                return ()
            order = tuple(ordered.index(vertex) for vertex in target_rows)
            target = item.coordinates
            if order != (0, 1, 2):
                corners = reference[np.asarray(order)]
                arguments = coordinate_enclosure_owner.affine_arguments(
                    corners[0], (corners[1:] - corners[0]).T, budget=budget
                )
                composition = coordinate_enclosure_owner.ExpressionComposition(
                    arguments, budget=budget
                )
                target = tuple(composition(value, budget=budget) for value in target)
            result.append(
                embedding_cells_owner.IntervalBoundaryCell(
                    source=source_map,
                    target=target,
                    vertices=(
                        int(global_vertices[target_rows[0]]),
                        int(global_vertices[target_rows[1]]),
                        int(global_vertices[target_rows[2]]),
                    ),
                    anchor=anchor,
                    entity=identifier,
                    orientation=orientation,
                )
            )
    if seen != set(selected):
        state.add("source_chart_facet_lineage_premise", "unresolved", "mesh")
        return ()
    return tuple(result)


def _chart_boundary_homotopy(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AnalyticBoundaryProof | None:
    """Prove an oriented CAD representative, its gap ribbons and target isotopy."""
    if (
        not isinstance(source, domain_boundary_owner.MeshingDomainBoundarySource)
        or not source.chart_triangulations
        or mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or any(block.cell_kind != "tetrahedron" for block in mesh.blocks)
    ):
        return None
    state = EmbeddingState([], ["source_chart_boundary_homotopy"])
    upper, degree_id, pieces = math.inf, None, 0
    closure = None
    try:
        with budget.measurement(), budget.activate(), budget.temporary_scope():
            if coordinate_scope(mesh, geometry, budget=budget).coordinate_scope == "affine":
                # Straight publication keeps its direct certified chart-cover
                # comparison with the authored faces. The closed-representative
                # isotopy is the curved-geometry degree/coverage theorem.
                return None
            domain = source.domain
            authority = domain.brep_authority
            if (
                not isinstance(authority, BRepModel)
                or domain.authority_id != authority.model_id
                or not tree_content_equal(
                    domain,
                    meshing_domain_owner.MeshingDomain.from_brep(authority, budget=budget),
                )
                or len(domain.regions) != 1
                or source.patches != tuple(range(len(domain.patches)))
                or tuple(patch for patch, _ in domain.regions[0].boundary) != source.patches
                or mesh.periodic_topology is not None
                or mesh.storage is not None
            ):
                state.add("source_chart_authority_premise", "unresolved", "mesh")
            elif tuple(sorted(row[0] for row in source.chart_triangulations)) != source.patches:
                state.add("source_trim_coverage_premise", "unresolved", "mesh")
            else:
                prepared_source = _prepare_chart_boundary_source(
                    source,
                    min(limits.maximum_source_samples, limits.maximum_subdivision_pieces),
                    budget=budget,
                )
                cover = prepared_source.cover
                state.findings.extend(cover.findings)
                if not cover.complete or cover.semantics != "certified":
                    state.add("source_trim_coverage_premise", "unresolved", "mesh")
                if not state.findings:
                    cells = _chart_boundary_cells(
                        state, mesh, geometry, source, limits, budget=budget,
                        source_maps=prepared_source.maps,
                    )
                    if not state.findings:
                        proof = mapped_embedding_owner.certify_interval_boundary_homotopy(
                            cells, limits, tolerance=tolerance, budget=budget
                        )
                        pieces = proof.piece_count
                        if (
                            not proof.complete
                            or proof.proof_id is None
                            or not math.isfinite(proof.distance_upper)
                            or proof.distance_upper > tolerance
                        ):
                            state.add(
                                "source_chart_homotopy_premise",
                                "unresolved",
                                "facet" if proof.entities else "mesh",
                                proof.entities,
                            )
                        else:
                            upper = proof.distance_upper
                            budget.reserve(1, 2048 + 40 * len(cells))
                            representative_id = canonical_fingerprint(
                                {
                                    "kind": "bounded-cad-boundary-representative",
                                    "authority": domain.authority_id,
                                    "source_scope": source.source_scope_id,
                                    "chart_cover": cover.cover_id,
                                    "maps": tuple(cell.source.map_id for cell in cells),
                                    "orientations": tuple(cell.orientation for cell in cells),
                                }
                            )
                            binding = MeshCertificateBinding(
                                mesh,
                                geometry,
                                coordinate_scope(mesh, geometry, budget=budget).coordinate_scope,
                                limits,
                                source_id=source.source_id,
                                source_revision=source.source_revision,
                            )
                            closure = CadBoundaryClosureEvidence(
                                binding,
                                authority_id=domain.authority_id,
                                source_scope_id=source.source_scope_id,
                                representative_id=representative_id,
                                homotopy_id=proof.proof_id,
                                homotopy_distance_upper=proof.distance_upper,
                                homotopy_piece_count=proof.piece_count,
                                chart_indices=np.asarray(
                                    [
                                        (data[0], row)
                                        for data in source.chart_triangulations
                                        for row in range(len(data[3]))
                                    ],
                                    dtype=np.int64,
                                ),
                                target_facet_ids=np.asarray(
                                    [cell.entity for cell in cells], dtype=np.int64
                                ),
                                gap_bounds=np.asarray(
                                    [cell.source.authored_deviation_bound for cell in cells],
                                    dtype=np.float64,
                                ),
                                original_gap_allowance=np.asarray(
                                    authority.report.curve_surface_tolerance,
                                    dtype=np.float64,
                                ),
                            )
                            _ISSUED_CAD_BOUNDARY.record(
                                closure, _cad_boundary_request_key(
                                    mesh, geometry, limits, closure
                                ),
                                mesh,
                            )
                            _ISSUED_CAD_BOUNDARY.record(
                                closure,
                                _cad_boundary_source_key(
                                    mesh, geometry, limits, closure, source, budget
                                ),
                                mesh,
                            )
                            degree_id = canonical_fingerprint(
                                {
                                    "kind": "source-chart-boundary-degree",
                                    "source": source.source_id,
                                    "revision": source.source_revision,
                                    "authority": domain.authority_id,
                                    "domain": domain.domain_id,
                                    "source_scope": source.source_scope_id,
                                    "chart_cover": cover.cover_id,
                                    "chart_patches": tuple(row[0] for row in source.chart_triangulations),
                                    "chart_witness": array_tree_fingerprint(
                                        tuple(row[1:] for row in source.chart_triangulations)
                                    ),
                                    "mesh": mesh.mesh_id,
                                    "topology": mesh.topology_id,
                                    "geometry": cell_geometry_id(geometry),
                                    "homotopy": proof.proof_id,
                                    "cad_closure": closure.evidence_id,
                                    "limits": limits.limits_id,
                                    "tolerance": repr(tolerance),
                                    "distance_upper": repr(upper),
                                }
                            )
        if closure is not None:
            budget.reserve(0, budget.live_object_storage_upper(closure))
    except domain_chart_chain_owner.SourceChartMapError:
        state.add("source_chart_map_premise", "unresolved", "mesh")
    except CoordinateEnclosureResourceError as error:
        closure, upper, degree_id = None, math.inf, None
        state.add(
            "source_chart_homotopy_resource",
            "unresolved",
            "mesh",
            resource_error=error,
            expression_budget=budget,
        )
    except RationalEnclosureError:
        state.add("source_chart_map_premise", "unresolved", "mesh")
    return _AnalyticBoundaryProof(
        upper, degree_id, tuple(state.findings), pieces, closure
    )


def boundary_degree_proof(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    tolerance: float,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AnalyticBoundaryProof | None:
    """Use the same source-bound degree proof for fidelity and region coverage."""
    gap, _ = _brep_source_gap(source, budget=budget)
    analytic = (
        analytic_boundary_proof(
            mesh, geometry, source, tolerance, limits, budget=budget
        )
        if gap == 0.0
        else None
    )
    if analytic is not None:
        return analytic
    return _chart_boundary_homotopy(
        mesh, geometry, source, tolerance, limits, budget=budget
    )


def _certify_boundary_degree_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    binding: MeshCertificateBinding,
    tolerance: float,
    order: int,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SourceFidelityCertificate | None:
    proof = boundary_degree_proof(
        mesh, geometry, source, tolerance, limits, budget=budget
    )
    if proof is None:
        return None
    return SourceFidelityCertificate(
        binding,
        proof.findings,
        tolerance=tolerance,
        bounds=SourceFidelityBounds(
            semantics=("certified", "certified"),
            mesh_to_source=(proof.distance_upper, 0.0),
            source_to_mesh=(
                proof.distance_upper
                if proof.boundary_degree_id is not None
                else math.inf,
                0.0,
            ),
            sample_counts=(proof.piece_count, 0),
        ),
        sample_order=order,
        coverage=SourceFidelityCoverage(cad_closure=proof.cad_closure),
        boundary_degree_id=proof.boundary_degree_id,
    )


@final
class _FidelityRequest(StrictModule):
    tolerance: float = eqx.field(static=True)
    order: int = eqx.field(static=True)
    limits: MeshCertificateLimits
    selected_facets: tuple[int, ...] | None = eqx.field(static=True)
    source_scope_id: str | None = eqx.field(static=True)


def _prepare_fidelity_request(
    mesh: CellMesh,
    source: SourceBoundaryQuery,
    tolerance: float,
    sample_order: int,
    limits: MeshCertificateLimits | None,
    target_facet_ids: ArrayLike | None,
    /,
) -> _FidelityRequest:
    # Protocol annotations are static at checked boundaries. Validate source
    # metadata and both boundary-query capabilities before reading their members.
    if not isinstance(source, SourceBoundaryQuery):
        raise TypeError(
            f"certify_source_fidelity.source must satisfy SourceBoundaryQuery; got {type(source).__name__}."
        )
    if source.ambient_dimension != mesh.ambient_dimension:
        raise ValueError(
            f"certify_source_fidelity.source.ambient_dimension must equal {mesh.ambient_dimension}; got {source.ambient_dimension}."
        )
    if mesh.ambient_dimension - mesh.topological_dimension > 1 or (
        mesh.ambient_dimension == 1
    ):
        raise ValueError(
            f"certify_source_fidelity.mesh must be a volume, surface or planar curve; got topological dimension {mesh.topological_dimension} and ambient dimension {mesh.ambient_dimension}."
        )
    tolerance_ = float(tolerance)
    if not math.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError(
            f"certify_source_fidelity.tolerance must be finite and nonnegative; got {tolerance!r}."
        )
    order = positive_integer(sample_order, "sample_order")
    limits_ = MeshCertificateLimits() if limits is None else limits
    selected_facets: tuple[int, ...] | None = None
    source_scope_id: str | None = None
    if target_facet_ids is not None:
        if (
            mesh.topological_dimension != mesh.ambient_dimension
            or mesh.topological_dimension != 3
        ):
            raise ValueError(
                f"certify_source_fidelity.target_facet_ids must scope a three-dimensional volume; got dimensions {(mesh.topological_dimension, mesh.ambient_dimension)!r}."
            )
        match source:
            case domain_boundary_owner.MeshingDomainBoundarySource():
                scoped_source = source
            case _:
                raise TypeError(
                    f"certify_source_fidelity.source must be MeshingDomainBoundarySource for a facet scope; got {type(source).__name__}."
                )
        identifiers = np.asarray(jax.device_get(target_facet_ids))
        if identifiers.ndim != 1 or identifiers.size == 0:
            raise ValueError(
                f"certify_source_fidelity.target_facet_ids must be a nonempty vector; got {identifiers.shape}."
            )
        if identifiers.dtype.kind not in "iu":
            raise TypeError(
                f"certify_source_fidelity.target_facet_ids must have an integer dtype; got {identifiers.dtype}."
            )
        selected_facets = tuple(sorted(int(value) for value in identifiers))
        if len(set(selected_facets)) != len(selected_facets) or selected_facets[0] < 0:
            raise ValueError(
                f"certify_source_fidelity.target_facet_ids must be unique and nonnegative; got {selected_facets!r}."
            )
        if not set(selected_facets) <= set(
            np.asarray(jax.device_get(mesh.entity_set(2).entity_ids)).tolist()
        ):
            raise ValueError(
                "SourceFidelityCertificate.target_facet_ids must name mesh facets; got an absent identifier."
            )
        source_scope_id = scoped_source.source_scope_id
    return _FidelityRequest(tolerance_, order, limits_, selected_facets, source_scope_id)


def _certify_boundary_source_evidence(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    binding: MeshCertificateBinding,
    request: _FidelityRequest,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> SourceFidelityCertificate | None:
    tolerance_, order, limits_ = request.tolerance, request.order, request.limits
    match source:
        case MappedDomainBoundarySource():
            coverage = source.boundary_coverage(
                mesh, geometry, limits=limits_, budget=budget
            )
            if coverage.binding.binding_id != binding.binding_id:
                raise ValueError(
                    f"MappedDomainBoundarySource.boundary_coverage.binding must match {binding.binding_id!r}; got {coverage.binding.binding_id!r}."
                )
            complete = coverage.status == "certified"
            bound = 0.0 if complete else math.inf
            findings = (
                ()
                if complete
                else (
                    MeshCertificateFinding(
                        "mapped_source_boundary_equality_premise",
                        "unresolved",
                        "mesh",
                    ),
                )
            )
            return SourceFidelityCertificate(
                binding,
                findings,
                tolerance=tolerance_,
                bounds=SourceFidelityBounds(
                    semantics=("certified", "certified"),
                    mesh_to_source=(bound, 0.0),
                    source_to_mesh=(bound, 0.0),
                    sample_counts=(0, 0),
                ),
                sample_order=order,
                coverage=SourceFidelityCoverage(domain=coverage),
            )
        case ImplicitProjectionBoundarySource():
            evidence = source.projection_coverage(
                mesh, geometry, tolerance=tolerance_, limits=limits_, budget=budget
            )
            if evidence.binding.binding_id != binding.binding_id:
                raise ValueError(
                    f"ImplicitProjectionBoundarySource.projection_coverage.binding must match {binding.binding_id!r}; got {evidence.binding.binding_id!r}."
                )
            findings = evidence.findings
            if evidence.forward_upper > tolerance_ and not any(
                finding.check == "boundary_deviation" for finding in findings
            ):
                findings = (
                    *findings,
                    MeshCertificateFinding(
                        "boundary_deviation",
                        "unresolved",
                        "mesh",
                    ),
                )
            backward_upper = (
                evidence.forward_upper if evidence.status == "certified" else math.inf
            )
            return SourceFidelityCertificate(
                binding,
                findings,
                tolerance=tolerance_,
                bounds=SourceFidelityBounds(
                    semantics=("certified", "certified"),
                    mesh_to_source=(evidence.forward_upper, evidence.forward_lower),
                    source_to_mesh=(backward_upper, 0.0),
                    sample_counts=(0, 0),
                ),
                sample_order=order,
                coverage=SourceFidelityCoverage(projection=evidence),
            )
        case _:
            return None


@final
class _FidelityDirections(StrictModule):
    forward: tuple[SourceBoundSemantics, float, float, int] = eqx.field(static=True)
    backward: tuple[SourceBoundSemantics, float, float, int] = eqx.field(static=True)
    chart: SourceBoundaryChartCover | None


def _execute_fidelity_covers(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    scope: CoordinateMapScope,
    request: _FidelityRequest,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FidelityDirections:
    tolerance_, order, limits_ = request.tolerance, request.order, request.limits
    selected_facets = request.selected_facets
    points = np.asarray(jax.device_get(mesh.coordinates), dtype=np.float64)
    chart = None
    maps = []
    if (
        selected_facets is not None
        or scope == "mapped"
        or isinstance(source, SourceBoundaryChartQuery)
    ):
        maps = _fidelity_maps(
            state, mesh, geometry, target_facet_ids=selected_facets, budget=budget
        )
        chart = _chart_fidelity(state, source, maps, tolerance_, limits_, budget=budget)
    if chart is not None and (
        (chart.forward[1] <= tolerance_ and chart.backward[1] <= tolerance_)
        or any(
            finding.check == "source_affine_chain_resource_budget"
            for finding in state.findings
        )
        or any(
            finding.check == "boundary_deviation" and finding.status == "violated"
            for finding in state.findings
        )
    ):
        forward, backward = chart.forward, chart.backward
    elif selected_facets is not None or scope == "mapped":
        query_source: SourceBoundaryQuery = source

        if isinstance(source, domain_boundary_owner.MeshingDomainBoundarySource):
            query_source = source.prepare_boundary_queries()
        mesh_cover = _mapped_mesh_cover(
            state, query_source, maps, tolerance_, limits_, budget=budget
        )
        forward = mesh_cover.forward
        if not maps or any(
            value.check == "coordinate_source_expression" for value in state.findings
        ):
            forward = (forward[0], math.inf, forward[2], forward[3])
        backward = (
            ("sampled", math.inf, 0.0, 0)
            if forward[0] != "certified"
            else _mapped_source_cover(
                state,
                query_source,
                mesh_cover.centers,
                mesh_cover.radii,
                mesh_cover.errors,
                forward[1],
                forward[3],
                tolerance_,
                limits_,
                budget=budget,
            )
        )
    else:
        items, entities = _fidelity_items(state, mesh, points, limits_, budget=budget)
        forward = _mesh_to_source(
            state, source, points, items, entities, order, tolerance_, limits_
        )
        backward = source_to_mesh(
            state, source, points, items, tolerance_, limits_, budget=budget
        )
    return _FidelityDirections(forward, backward, None if chart is None else chart.cover)


def _certify_scientific_source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    /,
    *,
    tolerance: float,
    sample_order: int = 4,
    limits: MeshCertificateLimits | None = None,
    target_facet_ids: ArrayLike | None = None,
    budget: CoordinateEnclosureBudget,
) -> SourceFidelityCertificate:
    """Bound the two-sided deviation between the mesh boundary and ``source``.

    Volume meshes compare boundary facets, surface and curve meshes their
    cells. Affine maps adaptively refine a barycentric lattice cover starting
    at ``sample_order``. Mapped maps use retained source-expression Bernstein
    enclosures with bounded subdivision. Both routes honor the declared work
    and subdivision limits rather than relaxing the requested tolerance.
    Certified chart chains additionally provide continuous Taylor comparison bounds.
    """

    request = _prepare_fidelity_request(
        mesh, source, tolerance, sample_order, limits, target_facet_ids
    )
    tolerance_, order, limits_ = request.tolerance, request.order, request.limits
    selected_facets, source_scope_id = request.selected_facets, request.source_scope_id
    try:
        with budget.measurement():
            scope = coordinate_scope(mesh, geometry, budget=budget).coordinate_scope
    except CoordinateEnclosureResourceError as error:
        binding = MeshCertificateBinding(
            mesh,
            geometry,
            "mapped",
            limits_,
            source_id=source.source_id,
            source_revision=source.source_revision,
        )
        state = EmbeddingState([], ["source_fidelity"])
        state.add(
            "source_expression_resource_budget",
            "unresolved",
            "mesh",
            resource_error=error,
            expression_budget=budget,
        )
        return SourceFidelityCertificate(
            binding,
            tuple(state.findings),
            tolerance=tolerance_,
            bounds=SourceFidelityBounds(
                semantics=("sampled", "sampled"),
                mesh_to_source=(math.inf, 0.0),
                source_to_mesh=(math.inf, 0.0),
                sample_counts=(0, 0),
            ),
            sample_order=order,
            target_facet_ids=selected_facets,
            source_scope_id=source_scope_id,
        )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        scope,
        limits_,
        source_id=source.source_id,
        source_revision=source.source_revision,
    )
    state = EmbeddingState([], ["source_fidelity"])
    if mesh.storage is not None:
        state.add("owner_local_source_fidelity_premise", "unresolved", "mesh")
        return SourceFidelityCertificate(
            binding,
            tuple(state.findings),
            tolerance=tolerance_,
            bounds=SourceFidelityBounds(
                semantics=("sampled", "sampled"),
                mesh_to_source=(math.inf, 0.0),
                source_to_mesh=(math.inf, 0.0),
                sample_counts=(0, 0),
            ),
            sample_order=order,
            target_facet_ids=selected_facets,
            source_scope_id=source_scope_id,
        )

    restrictions = (
        surface_source_support_owner.certify_original_surface_restriction_fidelity(
            mesh, geometry, source, binding, tolerance_, order, limits_, budget=budget
        )
        if selected_facets is None
        else None
    )
    if restrictions is not None:
        return restrictions

    radial = (
        radial_source_fidelity_owner.certify_native_radial_fidelity(
            mesh, geometry, source, binding, tolerance_, order, limits_, budget=budget
        )
        if selected_facets is None
        else None
    )
    if radial is not None:
        return radial
    degree = (
        _certify_boundary_degree_fidelity(
            mesh, geometry, source, binding, tolerance_, order, limits_, budget=budget
        )
        if selected_facets is None
        else None
    )
    if degree is not None:
        return degree
    boundary_evidence = _certify_boundary_source_evidence(
        mesh, geometry, source, binding, request, budget=budget
    )
    if boundary_evidence is not None:
        return boundary_evidence
    directions = _execute_fidelity_covers(
        state, mesh, geometry, source, scope, request, budget=budget
    )
    forward, backward = directions.forward, directions.backward
    return SourceFidelityCertificate(
        binding,
        tuple(state.findings),
        tolerance=tolerance_,
        bounds=SourceFidelityBounds(
            semantics=(forward[0], backward[0]),
            mesh_to_source=(forward[1], forward[2]),
            source_to_mesh=(backward[1], backward[2]),
            sample_counts=(forward[3], backward[3]),
        ),
        sample_order=order,
        coverage=SourceFidelityCoverage(chart=directions.chart),
        target_facet_ids=selected_facets,
        source_scope_id=source_scope_id,
    )


@checked
def certify_source_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    /,
    *,
    tolerance: float,
    sample_order: int = 4,
    limits: MeshCertificateLimits | None = None,
    target_facet_ids: ArrayLike | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> SourceFidelityCertificate:
    """Certify directed source-cover bounds and the numerical coefficient carrier.

    A quotient source supplies group coordinate expressions. Independently
    rounded local coefficients form their execution carrier. Its continuous
    error enclosure is added to both directed upper bounds while the requested
    tolerance and seam equality criterion stay unchanged.

    ``target_facet_ids`` compares named global volume facets with selected
    ``MeshingDomainBoundarySource`` patches. The facet list and occurrence-qualified
    source scope participate in certificate identity.
    """

    limits_ = MeshCertificateLimits() if limits is None else limits
    with coordinate_certificate_request(limits_, budget=budget) as budget:
        with budget.activate():
            certificate = _certify_scientific_source_fidelity(
                mesh,
                geometry,
                source,
                budget=budget,
                tolerance=tolerance,
                sample_order=sample_order,
                limits=limits_,
                target_facet_ids=target_facet_ids,
            )
            certificate = _include_brep_source_gap(certificate, source, budget=budget)
            if geometry.periodic_source is None and geometry.exact_source is None:
                return _issue_independent_fidelity(mesh, geometry, source, certificate)
            findings = certificate.findings
            try:
                with budget.measurement(), budget.activate():
                    error = geometry.source_execution_error(mesh, budget=budget)
            except CoordinateEnclosureResourceError as failure:
                error = math.inf
                state = EmbeddingState(list(findings), [])
                state.add(
                    "coordinate_source_execution_error_budget",
                    "unresolved",
                    "mesh",
                    resource_error=failure,
                    expression_budget=budget,
                )
                findings = tuple(state.findings)
            except coordinate_budget_owner.RationalEnclosureError:
                error = math.inf
                findings = (
                    *findings,
                    MeshCertificateFinding(
                        "coordinate_source_execution_error_budget",
                        "unresolved",
                        "mesh",
                        (),
                    ),
                )
            if error == 0.0:
                return _issue_independent_fidelity(mesh, geometry, source, certificate)
            forward = float(
                np.nextafter(certificate.mesh_to_source_upper + error, math.inf)
            )
            backward = float(
                np.nextafter(certificate.source_to_mesh_upper + error, math.inf)
            )
            if max(forward, backward) > certificate.tolerance:
                findings = (
                    *findings,
                    MeshCertificateFinding(
                        "coordinate_source_execution_error", "unresolved", "mesh", ()
                    ),
                )
            certificate = SourceFidelityCertificate(
                certificate.binding,
                findings,
                tolerance=certificate.tolerance,
                bounds=SourceFidelityBounds(
                    semantics=(
                        certificate.mesh_to_source_semantics,
                        certificate.source_to_mesh_semantics,
                    ),
                    mesh_to_source=(
                        forward,
                        max(0.0, certificate.mesh_to_source_lower - error),
                    ),
                    source_to_mesh=(
                        backward,
                        max(0.0, certificate.source_to_mesh_lower - error),
                    ),
                    sample_counts=(
                        certificate.mesh_sample_count,
                        certificate.source_sample_count,
                    ),
                ),
                sample_order=certificate.sample_order,
                coverage=SourceFidelityCoverage(
                    projection=certificate.projection_coverage,
                    domain=certificate.domain_coverage,
                    chart=certificate.chart_coverage,
                    cad_closure=certificate.cad_closure,
                ),
                target_facet_ids=certificate.target_facet_ids,
                source_scope_id=certificate.source_scope_id,
                boundary_degree_id=certificate.boundary_degree_id,
            )
            return _issue_independent_fidelity(mesh, geometry, source, certificate)


__all__ = [
    "FidelityMap",
    "ImplicitBoundarySource",
    "ImplicitProjectionBoundarySource",
    "ImplicitProjectionCoverageEvidence",
    "MappedDomainBoundarySource",
    "ParametricCurveBoundarySource",
    "ProjectionFacetEvidence",
    "ProjectionFiberCrossingKind",
    "ProjectionFiberEvidence",
    "SourceFidelityBounds",
    "SourceFidelityCertificate",
    "SourceFidelityCoverage",
    "analytic_boundary_proof",
    "boundary_degree_proof",
    "certify_source_fidelity",
    "fidelity_affine",
    "fidelity_norm_bound",
    "fidelity_proxy",
    "point_simplex_distances",
    "source_to_mesh",
    "split_fidelity_map",
    "split_fidelity_polynomial",
]
