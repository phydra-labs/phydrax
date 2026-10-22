#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify source-map contacts in a bounded periodic group neighborhood.

Lifted corners retain representative identity and generator winding. Bernstein
hulls enclose source-map images, and finite linear cosets retain their periods.

Mesh certificate checks
-----------------------
[
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_authored_group_identity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Generator matrices, orders, period translations, and source preparation bind the supplied periodic group."
      }
    ],
    "conclusion": "A group identity failure leaves the quotient embedding unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_full_source_hulls",
      "periodic_image_sufficiency"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every coordinate enclosure is finite and every Bernstein control net stays within its node allowance."
      },
      {
        "tag": "X",
        "statement": "The translation dual maps period displacements to their generator exponents; finite linear cosets retain their declared windings."
      }
    ],
    "conclusion": "Bernstein control hulls enclose each source map; a full-source P1 affine simplex has its degree-one controls at its reference vertices, so exact vertex extrema give the same directed hull without another control-net preparation. Affine group actions preserve that premise. Difference hulls bound every translation exponent that can produce contact.",
    "citation": {
      "kind": "mathematical",
      "reference": "Rida T. Farouki, The Bernstein polynomial basis: a centennial retrospective",
      "locator": "slides 12 and 17: nonnegative partition of unity and convex-hull property",
      "url": "https://faculty.engineering.ucdavis.edu/farouki/wp-content/uploads/sites/51/2021/07/Bernstein-polynomial-basis.pdf"
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_mapped_trace_equivariance",
      "periodic_mapped_self_image_embedding"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Lifted corner identifiers include representative identity, winding, and fixed-generator flags."
      },
      {
        "tag": "X",
        "statement": "Every candidate source-map contact and quotient trace is checked under its group action."
      }
    ],
    "conclusion": "Quotient admission requires equal identified traces and separation of every nonincident image pair.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "cell_validity",
      "periodic_layer_column_validity_reuse",
      "periodic_layer_column_graph_source",
      "periodic_layer_column_affine_reference_contact"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A prior cell-validity check covers the layer graph map, and every fiber child has a reference carrier."
      }
    ],
    "conclusion": "The layer route reuses local injectivity and checks quotient contacts on the fiber reference cells.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_affine_validity_reuse"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The bound cell-validity certificate certifies every cell of this source geometry, and the piece has the same simplex reference kind as its mesh block and coordinate element."
      },
      {
        "tag": "N",
        "statement": "The coordinate-source owner proves the full root basis is P1 affine and every restriction is affine; this is not an interpolation of corner samples."
      },
      {
        "tag": "T",
        "statement": "An affine map with full-column-rank derivative is injective on its full reference simplex, including its boundary."
      }
    ],
    "conclusion": "The bound positive validity proof supplies the constant derivative rank, so repeating a projected-Jacobian injectivity proof is unnecessary. Quotient traces and contacts remain independently checked.",
    "citation": {
      "kind": "mathematical",
      "reference": "Sheldon Axler, Linear Algebra Done Right, fourth edition (2024)",
      "locator": "Section 3B, null spaces and injectivity: a linear map is injective exactly when its null space is zero",
      "url": "https://linear.axler.net/LADR4e.pdf"
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_affine_quotient_contact"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every full-dimensional simplex has a canonical full-source affinity proof, the bound validity certificate proves positive orientation, and every quotient facet trace has been checked with no physical boundary facets or source-facet exceptions. PeriodicAffineQuotientEvidence records the affinity, validity, and facet-trace certificate IDs."
      },
      {
        "tag": "N",
        "statement": "Exact rational source-corner separating axes prove disjoint open interiors under every admitted group image; shared source traces and ordered candidate/image bounds retain their separate checks."
      },
      {
        "tag": "T",
        "statement": "A positive affine local star on a closed oriented quotient has positive integer degree. An additional nonincident closed contact would force overlapping open interiors in a neighborhood of that contact."
      }
    ],
    "conclusion": "The closed oriented affine quotient and open-interior separation prove the closed-contact obligation without enumerating intersection-plane vertices. Missing premises retain the generic exact contact proof.",
    "citation": {
      "kind": "mathematical",
      "reference": "Lipman, Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing",
      "locator": "Theorems 1 and 4; Section 3, cycle degree",
      "url": "https://arxiv.org/abs/1310.0955"
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_inverse_pair_proof_reuse"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The contact pair and its inverse have the same unordered orbit key within this request's immutable source bank, no source facet exception is active, and the earlier classification includes matching source traces."
      }
    ],
    "conclusion": "The exact invertible group action transports the earlier trace equality and contact classification to the inverse pair. Both are reused before repeating trace or contact preparation; ordered contact findings and candidate counts are retained.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_packed_polyhedral_source_identity",
      "periodic_coordinate_source_expression",
      "periodic_rational_enclosure_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Packed polyhedral routes match mesh vertices, and all supported source laws produce rational expressions and certified star pieces."
      }
    ],
    "conclusion": "Unsupported source identity, expression, or enclosure conditions refuse embedding certification.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_bernstein_node_budget",
      "periodic_coordinate_bound_range",
      "periodic_image_budget",
      "periodic_candidate_pair_budget",
      "periodic_source_expression_resource_budget"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Node, image, candidate, expression-work, and scratch-storage counts stay within the request limits; binary64 enclosure endpoints are finite."
      }
    ],
    "conclusion": "An exceeded limit or unrepresentable enclosure reports unresolved resource evidence at the phase that incurred it.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_mapped_trace_mismatch"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Identified source traces have equal rational coordinate expressions after applying their group actions."
      }
    ],
    "conclusion": "Unequal identified traces violate quotient trace continuity.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "owner": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
    "checks": [
      "periodic_mapped_cell_overlap"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The classified source-map pair intersects outside its authenticated incidence traces."
      }
    ],
    "conclusion": "A nonincident image overlap violates quotient embedding and retains its piece, owner, and action observations.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._periodic_embedding.certify_periodic_mapped_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import combinations, product

import numpy as np
from jax import Array

import phydrax._fingerprint as fingerprint_owner
import phydrax._geometry_predicates as geometry_predicates_owner
import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.discretization._exact_power_geometry as exact_power_geometry_owner
import phydrax.discretization._periodic_topology as periodic_topology_owner
import phydrax.geometry._affine_mapped_contact as affine_mapped_contact_owner
import phydrax.geometry._embedding_cells as embedding_cells_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._exact_polyhedral_geometry as exact_polyhedral_geometry_owner
import phydrax.geometry._periodic_affine_degree as periodic_affine_degree_owner
import phydrax.geometry._periodic_power_preparation as periodic_power_preparation_owner
import phydrax.geometry._periodic_restriction as periodic_restriction_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.geometry._supermesh as supermesh_owner
import phydrax.linalg._small_batched as small_batched_owner
from phydrax._bvh import (
    _host_boxes_overlap,
    bvh_overlap_pair_blocks,
    PackedBVH,
    prepare_bvh,
)
from phydrax._strict import StrictModule
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization._cell_geometry_validity import CellValidityCertificate
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.discretization._periodic_exact import (
    exact_isometry_power,
    exact_periodic_generators,
    ExactPeriodicActionLattice,
    PeriodicIsometryGroup,
    PeriodicIsometryIdentityError,
)
from phydrax.discretization._reference_cell_types import (
    CellBlock,
    CellGeometryElement,
    PolyhedralBlock,
)
from phydrax.geometry._embedding_cells import (
    bounds,
    Cell,
    domain,
    vertices as embedding_vertices,
)
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    MeshCertificateEntityKind,
    MeshCertificateLimits,
)
from phydrax.geometry._periodic_image_frame import transformed_source_box
from phydrax.linalg._exact import solve_exact
from phydrax.typing import checked

from ..discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    CellVertexGeometryElement,
    coordinate_lagrange_element,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    RationalEnclosureError,
)
from ..discretization._coordinate_enclosure import (
    constant,
    coordinate_corner_images,
    coordinate_expressions,
    coordinate_reference_chain,
    CoordinateSourceBank,
    expression_add,
    expression_bernstein_coefficients,
    expression_evaluate,
    expression_node_count,
    expression_scale,
    expression_sequence_equal,
    expression_sum,
    ExpressionComposition,
    prepared_coordinate_source_bank,
    restrict_chart_expressions,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_interval,
    RadicalElement,
    reserve_algebraic_work,
)
from ..discretization._reference_cell import reference_cell_topology
from ._embedding_state import ContactDecision
from ._mapped_embedding import (
    cell_traces_equal,
    common_atlas_extension,
    injective,
    pair_decision,
    separating_plane,
    tensor_gluing,
)
from ._periodic_mapped_restriction import certify_periodic_mapped_restriction
from ._periodic_restriction import PeriodicRestrictionPremise
from ._rational import fraction_endpoint, rational_vector, translation_ranges


def _transform(
    cell: Cell,
    matrix: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...],
    vertices: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> Cell:
    dimension = len(cell.coordinates)
    budget.reserve(dimension * (dimension + 1))
    if all(
        weight == (axis == component)
        for component, row in enumerate(matrix[:-1])
        for axis, weight in enumerate(row[:-1])
    ):
        # A translation leaves every nonconstant coefficient unchanged. Share
        # the source expression for zero shifts; no polynomial is mutated.
        coordinates = tuple(
            value
            if not row[-1]
            else expression_add(
                value, constant(row[-1], cell.dimension, budget=budget),
                budget=budget,
            )
            for value, row in zip(cell.coordinates, matrix[:-1], strict=True)
        )
        return replace(
            cell, coordinates=coordinates, vertices=vertices, corner_images=None
        )
    coordinates = tuple(
        expression_sum(
            (
                *tuple(
                    expression_scale(value, weight, budget=budget)
                    for value, weight in zip(cell.coordinates, row[:-1], strict=True)
                ),
                constant(row[-1], cell.dimension, budget=budget),
            ),
            budget=budget,
        )
        for row in matrix[:-1]
    )
    return replace(cell, coordinates=coordinates, vertices=vertices, corner_images=None)


@checked
def affine_reference_contact(
    first: Cell,
    second: Cell,
    maximum_work: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision | None:
    """Decide affine tetra/prism contact through the prism's tetrahedral partition."""
    polytope = affine_mapped_contact_owner.contact_decision(
        first, second, maximum_work, budget=budget, method="affine_polytope"
    )
    if polytope is not None:
        return polytope

    tetrahedron = coordinate_lagrange_element("tetrahedron", 1)
    prism_tetrahedra = ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3))

    def pieces(cell: Cell) -> tuple[Cell, ...] | None:
        if cell.kind == "tetrahedron":
            return (cell,)
        if cell.kind != "prism":
            return None
        corners = tuple(
            tuple(
                expression_evaluate(value, point, budget=budget)
                for value in cell.coordinates
            )
            for point in cell.reference_vertices
        )
        result = []
        for indices in prism_tetrahedra:
            coordinates = coordinate_expressions(
                tetrahedron, tuple(corners[index] for index in indices), budget=budget
            )
            if coordinates is None:
                return None
            result.append(
                Cell(
                    coordinates,
                    "simplex",
                    "tetrahedron",
                    cell.dimension,
                    tuple(cell.vertices[index] for index in indices),
                    embedding_vertices("tetrahedron"),
                )
            )
        return tuple(result)

    left, right = pieces(first), pieces(second)
    if left is None or right is None:
        return None
    parent_traces = affine_mapped_contact_owner.shared_affine_parent_traces(first, second)
    work = 0
    for a in left:
        traces: dict[tuple[int, ...], None] = {}
        for parent_trace in parent_traces:
            parent_support = set(parent_trace)
            support = tuple(
                identifier for identifier in a.vertices if identifier in parent_support
            )
            if support and len(set(support)) <= a.dimension:
                traces[support] = None
        for b in right:
            decision = affine_mapped_contact_owner.contact_decision(
                a,
                b,
                maximum_work - work,
                authenticated_first_traces=tuple(traces),
                budget=budget,
                method="affine_simplex",
            )
            if decision is None:
                raise ValueError(
                    "affine_reference_contact.source must provide affine P1 tetrahedra; got unsupported expressions"
                )
            status = decision.status
            used = decision.work
            work += used
            if status != "separated":
                return ContactDecision(status, work, check=decision.check)
    return ContactDecision("separated", work)


def _layer_fiber_reference_cell(
    element: LayerColumnCellGeometryElement,
    source_local: CoordinateSourceBank,
    chart_coordinates: tuple[coordinate_enclosure_owner.Expression, ...] | None,
    cell: Cell,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> Cell | None:
    """Pull one root or child chart back to its affine fiber carrier."""
    affine_coordinates = element.fiber_reference_expressions(source_local, budget=budget)
    if affine_coordinates is None:
        return None
    if chart_coordinates is None:
        reference_coordinates = affine_coordinates
    else:
        composition = ExpressionComposition(chart_coordinates, budget=budget)
        reference_coordinates = (
            replace(cell, coordinates=affine_coordinates, corner_images=None)
            .composed(composition, budget=budget)
            .coordinates
        )
    corner_values = tuple(
        tuple(
            expression_evaluate(value, point, budget=budget)
            for value in reference_coordinates
        )
        for point in cell.reference_vertices
    )
    canonical_reference = coordinate_expressions(
        coordinate_lagrange_element(cell.kind, 1), corner_values, budget=budget
    )
    if canonical_reference is None:
        raise ValueError(
            "periodic_embedding.layer_fiber_child must provide an affine chart; got unsupported reference expressions"
        )
    return replace(cell, coordinates=canonical_reference, corner_images=None)


def source_contact_quantities(
    first: Cell,
    second: Cell,
    matrix: tuple[tuple[Fraction, ...], ...],
    exponents: tuple[int, ...],
    first_piece: int,
    second_piece: int,
    first_owner: int,
    second_owner: int,
    /,
) -> tuple[tuple[str, int], ...]:
    """Encode the contact action, source incidence, and offending piece identities."""
    quantities = [
        ("first_piece", first_piece),
        ("second_piece", second_piece),
        ("first_owner", first_owner),
        ("second_owner", second_owner),
        ("shared_vertex_count", len(set(first.vertices).intersection(second.vertices))),
    ]
    for axis, exponent in enumerate(exponents):
        quantities.extend(
            (
                (f"image_exponent_{axis}_positive", max(0, exponent)),
                (f"image_exponent_{axis}_negative", max(0, -exponent)),
            )
        )
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            quantities.extend(
                (
                    (
                        f"action_{row}_{column}_numerator_positive",
                        max(0, value.numerator),
                    ),
                    (
                        f"action_{row}_{column}_numerator_negative",
                        max(0, -value.numerator),
                    ),
                    (f"action_{row}_{column}_denominator", value.denominator),
                )
            )
    for label, cell in (("first", first), ("second", second)):
        quantities.extend(
            (f"{label}_vertex_{corner}", identifier)
            for corner, identifier in enumerate(cell.vertices)
        )
    encoded = []
    for name, value in quantities:
        if value <= (1 << 53) - 1:
            encoded.append((name, value))
            continue
        # Binary64 observations retain larger integers as little-endian 32-bit limbs.
        limb = 0
        while value:
            encoded.append((f"{name}_limb_{limb}", value & ((1 << 32) - 1)))
            value >>= 32
            limb += 1
        encoded.append((f"{name}_limb_count", limb))
    return tuple(encoded)


def _has_full_coefficient_source(element: CellGeometryElement) -> bool:
    """Return whether the source chain ends at a full barycentric coefficient law."""
    match element:
        case (
            RestrictedCellGeometryElement()
            | PolynomialComposedCellGeometryElement()
            | RationalComposedCellGeometryElement()
        ):
            return _has_full_coefficient_source(element.source_element)
        case BarycentricCellGeometryElement():
            return True
        case _:
            return False


def _closed_polygon_point(
    source: CoordinateSourceBank,
    point: tuple[coordinate_enclosure_owner.ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    winding = 0
    for a, b in zip(source, (*source[1:], source[0]), strict=True):
        side = planar_primitives_owner.turn(a, b, point, budget=budget)
        if not side and all(
            min(x, y) <= p <= max(x, y) for x, y, p in zip(a, b, point, strict=True)
        ):
            return True
        if a[1] <= point[1] < b[1] and side > 0:
            winding += 1
        elif b[1] <= point[1] < a[1] and side < 0:
            winding -= 1
    return bool(winding)


def _source_edge_parameters(
    source: CoordinateSourceBank,
    a: tuple[coordinate_enclosure_owner.ExactScalar, ...],
    b: tuple[coordinate_enclosure_owner.ExactScalar, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[coordinate_enclosure_owner.ExactScalar, ...]:
    parameters: set[coordinate_enclosure_owner.ExactScalar] = {Fraction(0), Fraction(1)}
    for c, d in zip(source, (*source[1:], source[0]), strict=True):
        planar_primitives_owner.reserve_fraction_work(
            (a, b, c, d), 48, 4, 16, budget=budget
        )
        r = tuple(y - x for x, y in zip(a, b, strict=True))
        s = tuple(y - x for x, y in zip(c, d, strict=True))
        q = tuple(y - x for x, y in zip(a, c, strict=True))
        divisor = r[0] * s[1] - r[1] * s[0]
        if divisor:
            t = (q[0] * s[1] - q[1] * s[0]) / divisor
            u = (q[0] * r[1] - q[1] * r[0]) / divisor
            if 0 <= t <= 1 and 0 <= u <= 1:
                parameters.add(t)
        elif not planar_primitives_owner.turn(a, b, c, budget=budget):
            axis = next(index for index, value in enumerate(r) if value)
            for point in (c, d):
                t = (point[axis] - a[axis]) / r[axis]
                if 0 <= t <= 1:
                    parameters.add(t)
    return tuple(sorted(parameters))


def _simple_source_polygon_is_convex(
    source: CoordinateSourceBank, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Use the caller's simple-loop proof and exact convex-hull area equality."""
    cached = budget.simple_source_polygon_convexity_cache.get(source)
    if cached is not None:
        return cached
    with budget.temporary_scope():
        hull = planar_primitives_owner.convex_hull(source, budget=budget)
        hull_area = abs(planar_primitives_owner.signed_measure(hull, budget=budget))
        source_area = abs(planar_primitives_owner.signed_measure(source, budget=budget))
        convex = bool(hull_area) and source_area == hull_area
        del hull
        budget.retain_basis((source,))
        budget.simple_source_polygon_convexity_cache[source] = convex
    return convex


@checked
def closed_source_polygon_contains(
    source: CoordinateSourceBank,
    subject: CoordinateSourceBank,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove a subject loop lies in a simple source polygon, including its boundary."""
    if (
        not source
        or not subject
        or not all(
            _closed_polygon_point(source, point, budget=budget) for point in subject
        )
    ):
        return False
    if _simple_source_polygon_is_convex(source, budget=budget):
        # A convex source contains the convex hull of the admitted vertices,
        # hence every subject segment and its interior without edge crossings.
        return True
    for a, b in zip(subject, (*subject[1:], subject[0]), strict=True):
        if a == b:
            continue
        ordered = _source_edge_parameters(source, a, b, budget=budget)
        for low, high in zip(ordered, ordered[1:]):
            planar_primitives_owner.reserve_fraction_work(
                (a, b, (low, high)), 8, 3, 8, budget=budget
            )
            middle = (low + high) / 2
            point = tuple(x + middle * (y - x) for x, y in zip(a, b, strict=True))
            if not _closed_polygon_point(source, point, budget=budget):
                return False
    # A simple polygon's connected exterior cannot cross a contained boundary.
    return True


def source_polygon_plane(
    points: CoordinateSourceBank, *, budget: CoordinateEnclosureBudget
) -> (
    tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], tuple[int, ...], int, int]
    | None
):

    for a in range(1, len(points)):
        for b in range(a + 1, len(points)):
            plane = planar_primitives_owner.plane_key(
                (points[0], points[a], points[b]), budget=budget
            )
            if plane is not None:
                key = plane[0]
                for point in points:
                    planar_primitives_owner.reserve_fraction_work(
                        (key, point), 2 * len(point), 1, 4, budget=budget
                    )
                    if sum(
                        (a * b for a, b in zip(key[:-1], point, strict=True)), key[-1]
                    ):
                        return None
                return plane
    return None


def _original_power_facet_authority(
    geometry: CellGeometrySpec,
    group: PeriodicIsometryGroup,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[
    tuple[
        int,
        tuple[coordinate_enclosure_owner.ExactScalar, ...],
        tuple[int, ...],
        CoordinateSourceBank,
    ],
    ...,
]:

    source = geometry.exact_source
    while isinstance(
        source,
        (
            exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource,
            exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource,
        ),
    ):
        source = source.parent
    if not isinstance(source, exact_power_geometry_owner.ExactPowerCellGeometrySource):
        return ()
    power_domain, constraints = source.domain_source, source.periodic_constraints
    if power_domain is None or not constraints:
        return ()
    if source.authority_binding != exact_power_geometry_owner.power_domain_binding(
        power_domain, constraints
    ):
        raise ValueError(
            "periodic_embedding.source.authority_binding must match its domain and controls; got mismatched binding"
        )
    preparation = source.periodic_preparation
    if (
        not isinstance(
            preparation, periodic_power_preparation_owner.PeriodicPowerPreparation
        )
        or not isinstance(preparation.periodic_group, PeriodicIsometryGroup)
        or preparation.periodic_group.group_id != group.group_id
    ):
        raise ValueError(
            "periodic_embedding.source.periodic_preparation must bind the requested group; got mismatched group"
        )
    if (
        source.periodic_source_binding
        != exact_power_geometry_owner.power_preparation_binding(preparation)
    ):
        raise ValueError(
            "periodic_embedding.source.periodic_source_binding must match its preparation; got mismatched binding"
        )

    if fingerprint_owner.array_tree_fingerprint(
        preparation.periodic_group
    ) != fingerprint_owner.array_tree_fingerprint(group):
        raise ValueError(
            "periodic_embedding.source.periodic_group must match every group leaf; got mismatched group payload"
        )

    budget.reserve(
        power_domain.facet_regions.shape[0], 64 + 64 * power_domain.facet_regions.shape[0]
    )
    roots = list(range(power_domain.facet_regions.shape[0]))

    def root(value: int) -> int:
        while roots[value] != value:
            budget.reserve(1)
            roots[value] = roots[roots[value]]
            value = roots[value]
        return value

    for constraint in source.periodic_constraints:
        if (
            constraint.source_scope.entity_dimension != 2
            or constraint.target_scope.entity_dimension != 2
        ):
            continue
        targets = np.asarray(constraint.target_scope.entity_ids)
        if constraint.source_entity_ids is None:
            sources = np.asarray(constraint.source_scope.entity_ids)
            # A singleton scope pair declares a facet correspondence.
            # Larger scopes use the source trace proof.
            pairs = (
                ((int(sources[0]), int(targets[0])),)
                if len(sources) == len(targets) == 1
                else ()
            )
        else:
            pairs = zip(np.asarray(constraint.source_entity_ids), targets, strict=True)
        for a, b in pairs:
            budget.reserve(1, 32)
            a, b = int(a), int(b)
            if not 0 <= a < len(roots) or not 0 <= b < len(roots):
                raise ValueError(
                    "periodic_embedding.source.periodic_constraints must index declared PLC facets; got out-of-range facet"
                )
            ra, rb = root(a), root(b)
            roots[max(ra, rb)] = min(ra, rb)
    points = planar_primitives_owner.rational_points(
        np.asarray(power_domain.vertices), budget=budget
    )

    budget.reserve(0, 256 + 24 * len(points))
    point_array = np.asarray(points, dtype=object)
    result = []
    offsets, indices = (
        np.asarray(power_domain.polygon_offsets),
        np.asarray(power_domain.polygon_vertices),
    )
    for polygon, facet in enumerate(np.asarray(power_domain.polygon_facets)):
        row = tuple(map(int, indices[offsets[polygon] : offsets[polygon + 1]]))
        work, storage = exact_polyhedral_geometry_owner.triangulation_charge(
            point_array, row, budget=budget
        )
        # triangulate_loop charges each performed operation; the bound is a priori.
        budget.admit_work_bound(work, storage_upper=storage)
        budget.reserve(0, storage)
        exact_polyhedral_geometry_owner.triangulate_loop(point_array, row, budget=budget)
        loop = tuple(points[index] for index in row)
        plane = source_polygon_plane(loop, budget=budget)
        if plane is None:
            raise ValueError(
                "periodic_embedding.source.polygon must have a supporting plane; got nonplanar or degenerate loop"
            )
        key, axes_, _, _ = plane
        budget.reserve(1, 256)
        result.append(
            (
                root(int(facet)),
                key,
                axes_,
                planar_primitives_owner.project(loop, axes_, budget=budget),
            )
        )
    return tuple(result)


def _planar_polyhedral_pieces(
    mesh: CellMesh,
    first_cell: int,
    count: int,
    vertices: np.ndarray,
    valid: np.ndarray,
    source: CoordinateSourceBank,
) -> tuple[np.ndarray, np.ndarray]:

    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            triangles, triangle_valid = supermesh_owner.polyhedral_face_triangles(
                connectivity, first_cell, count
            )
        case connectivity:
            raise TypeError(
                "periodic_embedding.mesh.connectivity must provide polyhedral face incidence; "
                f"got {type(connectivity).__name__}"
            )
    vertices = np.asarray(vertices, dtype=np.int64)
    ordered = np.sort(np.where(valid, vertices, np.iinfo(np.int64).max), axis=1)
    candidate_valid = ordered != np.iinfo(np.int64).max
    apex, _, invalid, uncertain = supermesh_owner.cone_decomposition(
        np.asarray(mesh.coordinates, dtype=np.float64),
        triangles,
        triangle_valid,
        np.where(candidate_valid, ordered, 0),
        candidate_valid,
        geometry_predicates_owner.PredicateMode.EXACT,
    )
    if np.any(invalid) or np.any(uncertain):
        raise RationalEnclosureError(
            "periodic_embedding.polyhedral_star must have certified carrier cones; got invalid or uncertain decomposition"
        )

    owners = []
    pieces = []
    for cell in range(count):
        signs = set()
        cell_pieces = []
        anchor = int(apex[cell])
        for triangle in triangles[cell][triangle_valid[cell]]:
            # The carrier proposes a star vertex; source coefficients determine
            # every cone sign, including rational power vertices.
            vectors = tuple(
                tuple(
                    value - origin
                    for value, origin in zip(
                        source[int(vertex)], source[anchor], strict=True
                    )
                )
                for vertex in triangle
            )
            determinant = small_batched_owner.cofactor_determinant_3(
                *tuple(np.asarray(vector, dtype=object) for vector in vectors)
            )
            if determinant:
                signs.add(1 if determinant > 0 else -1)
                cell_pieces.append((anchor, *triangle.tolist()))
        if len(signs) != 1:
            raise RationalEnclosureError(
                "periodic_embedding.polyhedral_star must have consistent rational source cone signs; got mixed or degenerate cones"
            )
        owners.extend([cell] * len(cell_pieces))
        pieces.extend(cell_pieces)
    return np.asarray(owners, dtype=np.int64), np.asarray(pieces, dtype=np.int64)


def _tensor_pair(
    first: Cell,
    second: Cell,
    state: EmbeddingState,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    shared = np.asarray(
        sorted(set(first.vertices).intersection(second.vertices)), dtype=np.int64
    )
    gluing = tensor_gluing(first, second, shared)
    if gluing is None:
        return False
    matrix, offset = gluing
    reference = np.asarray(
        reference_cell_topology(second.kind).vertices, dtype=np.float64
    )
    combined = np.concatenate(
        (
            np.asarray(reference_cell_topology(first.kind).vertices, dtype=np.float64),
            reference @ matrix.T + offset,
        )
    )
    lower = np.min(combined, axis=0)
    spans = np.max(combined, axis=0) - lower
    return common_atlas_extension(
        [first, second],
        {
            0: (
                np.eye(first.dimension, dtype=np.float64),
                np.zeros(first.dimension, dtype=np.float64),
            ),
            1: (matrix, offset),
        },
        lower,
        spans,
        state,
        limits,
        budget=budget,
    )


def _root_action_pair(
    root: Cell,
    first: Cell,
    second: Cell,
    physical_action: tuple[tuple[Fraction, ...], ...],
    root_corners: CoordinateSourceBank,
    state: EmbeddingState,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove the physical group action through its source reference chart.

    An affine corner chart proposes the action. Equality of root expressions
    establishes the map relation over the reference cell.
    """

    dimension = root.dimension
    reference = tuple(
        rational_vector(row) for row in reference_cell_topology(root.kind).vertices
    )
    basis = next(
        (
            indices
            for indices in combinations(range(1, len(reference)), dimension)
            if np.linalg.matrix_rank(
                np.asarray(
                    [
                        [float(reference[j][i] - reference[0][i]) for j in indices]
                        for i in range(dimension)
                    ],
                    dtype=np.float64,
                )
            )
            == dimension
        ),
        None,
    )
    if basis is None:
        return False
    reference_columns = [
        [reference[j][i] - reference[0][i] for i in range(dimension)] for j in basis
    ]
    physical_columns = [
        [root_corners[j][i] - root_corners[0][i] for i in range(dimension)] for j in basis
    ]
    coefficients = solve_exact(reference_columns, physical_columns)
    affine = [[coefficients[j][i] for j in range(dimension)] for i in range(dimension)]
    if (
        np.linalg.matrix_rank(
            np.asarray(
                [[float(value) for value in row] for row in affine], dtype=np.float64
            )
        )
        != dimension
    ):
        return False
    center = tuple(
        root_corners[0][i]
        - sum((affine[i][j] * reference[0][j] for j in range(dimension)), Fraction(0))
        for i in range(dimension)
    )
    rotated = [
        [
            sum(
                (physical_action[i][k] * affine[k][j] for k in range(dimension)),
                Fraction(0),
            )
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    displacement = [
        [
            sum(
                (physical_action[i][k] * center[k] for k in range(dimension)), Fraction(0)
            )
            + physical_action[i][-1]
            - center[i]
        ]
        for i in range(dimension)
    ]
    matrix_exact = solve_exact(affine, rotated)
    offset_exact = tuple(row[0] for row in solve_exact(affine, displacement))
    matrix = np.asarray(
        [[float(value) for value in row] for row in matrix_exact], dtype=np.float64
    )
    offset = np.asarray([float(value) for value in offset_exact], dtype=np.float64)
    if any(
        Fraction(float(value)) != exact
        for row, exact_row in zip(matrix, matrix_exact, strict=True)
        for value, exact in zip(row, exact_row, strict=True)
    ) or any(
        Fraction(float(value)) != exact
        for value, exact in zip(offset, offset_exact, strict=True)
    ):
        return False
    action = tuple(
        tuple((*row, start))
        for row, start in zip(matrix_exact, offset_exact, strict=True)
    ) + ((Fraction(0),) * dimension + (Fraction(1),),)
    image_chart = _transform(second, action, second.vertices, budget=budget)
    box_kind = "quadrilateral" if dimension == 2 else "hexahedron"
    identity = np.eye(dimension, dtype=np.float64)
    zero = np.zeros(dimension, dtype=np.float64)
    root_on_box = restrict_chart_expressions(
        root.coordinates, root.kind, box_kind, zero, identity, budget=budget
    )
    shifted = restrict_chart_expressions(
        root.coordinates, root.kind, box_kind, offset, matrix, budget=budget
    )
    transformed = _transform(
        Cell(root_on_box, "box", box_kind, dimension, (), ()),
        physical_action,
        (),
        budget=budget,
    ).coordinates
    if shifted != transformed:
        return False
    boxes = (
        bounds(first, zero, identity, budget=budget),
        bounds(image_chart, zero, identity, budget=budget),
    )
    lower = np.minimum(boxes[0][0], boxes[1][0])
    upper = np.maximum(boxes[0][1], boxes[1][1])
    spans = np.nextafter(upper - lower, np.inf)
    extension = Cell(
        restrict_chart_expressions(
            root.coordinates, root.kind, box_kind, lower, np.diag(spans), budget=budget
        ),
        "box",
        box_kind,
        dimension,
        (),
        (),
    )
    if any(
        expression_node_count(value, "box", dimension) > limits.maximum_bernstein_nodes
        for value in extension.coordinates
    ):
        return False
    if injective(extension, limits, state, budget=budget) is not None:
        return False
    contact_decision_1 = pair_decision(
        first,
        image_chart,
        limits,
        limits.maximum_subdivision_pieces - state.subdivision_pieces,
        budget=budget,
    )
    decision = contact_decision_1.status
    work = contact_decision_1.work
    state.subdivision_pieces += work
    return decision == "separated"


def _inverse_equivalent_pair_key(
    first: int,
    second: int,
    exponents: tuple[int, ...],
    orders: tuple[int, ...],
    /,
) -> tuple[int, int, tuple[int, ...]]:
    """Identify the same contact after applying the inverse group action."""

    inverse = tuple(
        (-exponent) % order if order else -exponent
        for exponent, order in zip(exponents, orders, strict=True)
    )
    forward = (first, second, exponents)
    reverse = (second, first, inverse)
    return min(forward, reverse)


@dataclass
class _PeriodicProgress:
    image_count: int = 0
    resource_kind: MeshCertificateEntityKind = "mesh"
    resource_ids: tuple[int, ...] = ()
    facet_trace_certificate_id: str | None = None


class _PeriodicRequest(StrictModule):
    state: EmbeddingState
    mesh: CellMesh
    geometry: CellGeometrySpec
    topology: periodic_topology_owner.PeriodicMeshTopology
    cell_ids: np.ndarray
    limits: MeshCertificateLimits
    budget: CoordinateEnclosureBudget
    progress: _PeriodicProgress
    validity: CellValidityCertificate | None


class _PeriodicGroup(StrictModule):
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...]
    orders: tuple[int, ...]
    periods: tuple[int, ...]
    translations: tuple[tuple[Fraction, ...], ...]
    finite_count: int
    finite: tuple[tuple[tuple[int, ...], tuple[tuple[Fraction, ...], ...]], ...]


class _LiftedVertices(StrictModule):
    roots: np.ndarray
    shifts: np.ndarray
    fixed: np.ndarray
    orders: tuple[int, ...]
    keys: dict[tuple[int, ...], int]

    def identifiers(
        self, vertices: np.ndarray, image: tuple[int, ...]
    ) -> tuple[int, ...]:
        roots, shifts, fixed, orders, keys = (
            self.roots,
            self.shifts,
            self.fixed,
            self.orders,
            self.keys,
        )
        result = []
        for vertex in vertices.tolist():
            exponents = tuple(
                0
                if is_fixed
                else ((int(value) + shift) % order if order else int(value) + shift)
                for value, shift, order, is_fixed in zip(
                    shifts[vertex], image, orders, fixed[vertex], strict=True
                )
            )
            key = (int(roots[vertex]), *exponents)
            result.append(keys.setdefault(key, len(keys)))
        return tuple(result)


class _SourceRoot(StrictModule):
    parent: int
    root: Cell
    chart: Cell
    corners: CoordinateSourceBank
    element_id: str
    parent_vertices: tuple[int, ...]


class _PeriodicCells(StrictModule):
    cells: list[Cell]
    owners: list[int]
    piece_vertices: list[np.ndarray]
    planar_polyhedral_owners: set[int]
    source_roots: dict[int, _SourceRoot]
    layer_reference_cells: dict[int, Cell]
    layer_column_owners: set[int]
    source_bank: CoordinateSourceBank
    lifted: _LiftedVertices
    affine_pieces: set[int]
    affine_source_ids: dict[int, str]
    # Rational corners of pieces whose whole coordinate map is proven affine.
    affine_corners: dict[int, CoordinateSourceBank]


class _ParentFace(StrictModule):
    dimension: int
    orbit: int
    source_vertices: tuple[int, ...]
    lifted_vertices: tuple[int, ...]


class _PeriodicFacets(StrictModule):
    facets: embedding_state_owner.Facets
    entity_ids: np.ndarray
    facet_rows: dict[int, int]
    orbits: np.ndarray
    anchors: np.ndarray
    parent_faces: dict[int, list[_ParentFace]]
    by_orbit: dict[
        int, dict[tuple[int, int], list[tuple[tuple[int, ...], tuple[int, ...]]]]
    ]


class _PeriodicSourceFaces(StrictModule):
    original_facets: tuple[
        tuple[
            int,
            tuple[coordinate_enclosure_owner.ExactScalar, ...],
            tuple[int, ...],
            CoordinateSourceBank,
        ],
        ...,
    ]
    source_faces: dict[
        tuple[int, tuple[int, ...]],
        tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], set[int]],
    ]


class _PeriodicImages(StrictModule):
    ranges: tuple[range, ...]
    base_boxes: list[tuple[np.ndarray, np.ndarray]]
    cell_ids: np.ndarray


_PERIODIC_CONTACT_CHUNK_SIZE = 16


@dataclass
class _PeriodicContactCache:
    decisions: dict[tuple[int, int, tuple[int, ...]], str]
    recorded_reuse: bool = False
    trace_permissions: dict[
        tuple[int, int, tuple[int, ...]], tuple[tuple[int, ...], ...]
    ] = field(default_factory=dict)
    source_supports: dict[
        tuple[tuple[int, ...], tuple[int, ...], int], tuple[int, ...] | None
    ] = field(default_factory=dict)
    transformed_face_planes: dict[
        tuple[
            tuple[coordinate_enclosure_owner.ExactScalar, ...],
            tuple[tuple[Fraction, ...], ...],
        ],
        tuple[coordinate_enclosure_owner.ExactScalar, ...],
    ] = field(default_factory=dict)
    transformed_source_points: dict[
        tuple[
            tuple[coordinate_enclosure_owner.ExactScalar, ...],
            tuple[tuple[Fraction, ...], ...],
        ],
        tuple[coordinate_enclosure_owner.ExactScalar, ...],
    ] = field(default_factory=dict)


def _prepare_periodic_group(request: _PeriodicRequest) -> _PeriodicGroup:
    state, limits, budget = request.state, request.limits, request.budget
    topology = request.topology
    maximum_images = limits.maximum_periodic_images
    matrices, orders = exact_periodic_generators(topology.cell, budget=budget)
    if isinstance(topology.cell, PeriodicIsometryGroup):
        linear_orders = topology.cell.linear_orders
        periods = topology.cell.translation_periods
    else:
        linear_orders = tuple(1 for _ in orders)
        periods = tuple(1 if not order else 0 for order in orders)
    translations = tuple(
        tuple(row[-1] for row in period_matrix[:-1])
        for matrix, period in zip(matrices, periods, strict=True)
        if period
        for period_matrix in (exact_isometry_power(matrix, period, budget=budget),)
    )
    finite_count = 1
    for order in linear_orders:
        finite_count *= order
    if finite_count > maximum_images:
        state.add("periodic_image_budget", "unresolved", "mesh")
        return _PeriodicGroup(matrices, orders, periods, translations, finite_count, ())
    exponents = tuple(product(*(range(order) for order in linear_orders)))
    transforms = ExactPeriodicActionLattice(matrices, orders).prepare(
        exponents, budget=budget
    )
    finite = tuple(zip(exponents, transforms, strict=True))
    return _PeriodicGroup(
        matrices, orders, periods, translations, finite_count, tuple(finite)
    )


def _periodic_source_rows(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    block: CellBlock | PolyhedralBlock,
    element: CellGeometryElement,
    route: Array,
    points: np.ndarray,
    first_cell: int,
) -> list[tuple[int, str, CellGeometryElement, np.ndarray, np.ndarray]] | None:
    state, mesh, cell_ids, budget = (
        request.state,
        request.mesh,
        request.cell_ids,
        request.budget,
    )
    planar_polyhedral_owners, source_bank = (
        prepared.planar_polyhedral_owners,
        prepared.source_bank,
    )
    source_rows: list[tuple[int, str, CellGeometryElement, np.ndarray, np.ndarray]]
    if isinstance(element, CellVertexGeometryElement) and block.cell_kind == "polyhedron":
        if not np.array_equal(
            np.asarray(route), np.asarray(block.vertices)
        ) or not np.array_equal(points, np.asarray(mesh.coordinates)):
            state.add(
                "periodic_packed_polyhedral_source_identity",
                "unresolved",
                "cell",
                cell_ids[first_cell : first_cell + block.cell_count],
            )
            return None
        planar_polyhedral_owners.update(range(first_cell, first_cell + block.cell_count))
        budget.reserve(block.vertices.size, block.vertices.size * 64)
        local_owners, rows = _planar_polyhedral_pieces(
            mesh,
            first_cell,
            block.cell_count,
            np.asarray(block.vertices),
            np.asarray(block.vertex_valid),
            source_bank,
        )
        source_rows = [
            (
                int(owner),
                "tetrahedron",
                coordinate_lagrange_element("tetrahedron", 1),
                vertices,
                vertices,
            )
            for owner, vertices in zip(local_owners, rows, strict=True)
        ]
    else:
        source_rows = [
            (owner, block.cell_kind, element, row, vertices)
            for owner, (row, vertices) in enumerate(
                zip(np.asarray(route), np.asarray(block.vertices), strict=True)
            )
        ]
    return source_rows


class _PeriodicRootPreparation(StrictModule):
    domain_certified: bool
    layer_column: bool


def _prepare_periodic_root(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    cell: Cell,
    source_element: CellGeometryElement,
    source_local: CoordinateSourceBank,
    block: CellBlock | PolyhedralBlock,
    local_owner: int,
    owner: int,
    layer_column_root: bool,
) -> _PeriodicRootPreparation:
    mesh, geometry, budget = request.mesh, request.geometry, request.budget
    source_roots, layer_column_owners, layer_reference_cells = (
        prepared.source_roots,
        prepared.layer_column_owners,
        prepared.layer_reference_cells,
    )
    record = geometry.restriction_source
    if record is None:
        return _PeriodicRootPreparation(False, layer_column_root)
    root_domain_certified = False
    root_element, chart_coordinates = coordinate_reference_chain(
        source_element, budget=budget
    )
    layer_column_root |= (
        isinstance(root_element, LayerColumnCellGeometryElement)
        and root_element.fiber_graph
    )
    root_coordinates = coordinate_expressions(root_element, source_local, budget=budget)
    root_corners = coordinate_corner_images(root_element, source_local, budget=budget)
    if (
        isinstance(root_element, LayerColumnCellGeometryElement)
        and root_element.fiber_graph
    ):
        layer_column_owners.add(owner)
        reference_cell = _layer_fiber_reference_cell(
            root_element,
            source_local,
            chart_coordinates,
            cell,
            budget=budget,
        )
        if reference_cell is None:
            raise ValueError(
                "periodic_embedding.layer_fiber must provide an affine carrier; got unsupported source"
            )
        layer_reference_cells[owner] = reference_cell
    if root_coordinates is not None and root_corners is not None:
        parent = int(np.asarray(record.block_parent_cell_ids[block.name])[local_owner])
        root = Cell(
            root_coordinates,
            domain(root_element.cell_kind),
            root_element.cell_kind,
            mesh.topological_dimension,
            (),
            embedding_vertices(root_element.cell_kind),
        )
        chart = replace(cell, coordinates=chart_coordinates, corner_images=None)
        parent_vertices = tuple(
            int(value)
            for value in np.asarray(record.block_parent_vertex_ids[block.name])[
                local_owner
            ]
            if value >= 0
        )
        source_roots[owner] = _SourceRoot(
            parent,
            root,
            chart,
            root_corners,
            root_element.element_id,
            parent_vertices,
        )

        root_domain_certified = all(
            min(
                expression_bernstein_coefficients(
                    value, cell.domain, cell.dimension, budget=budget
                )
            )
            >= 0
            for value in embedding_cells_owner.reference_constraints(
                chart_coordinates,
                root.kind,
                cell.dimension,
                budget=budget,
            )
        )
    return _PeriodicRootPreparation(root_domain_certified, layer_column_root)


def _prepare_periodic_piece(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    block: CellBlock | PolyhedralBlock,
    source_element: CellGeometryElement,
    row: np.ndarray,
    vertices: np.ndarray,
    kind: str,
    local_owner: int,
    owner: int,
) -> Cell | None:
    state, mesh, geometry, cell_ids, limits, budget = (
        request.state,
        request.mesh,
        request.geometry,
        request.cell_ids,
        request.limits,
        request.budget,
    )
    identifiers = prepared.lifted.identifiers
    source_bank, source_roots = prepared.source_bank, prepared.source_roots
    layer_column_owners, layer_reference_cells = (
        prepared.layer_column_owners,
        prepared.layer_reference_cells,
    )
    orders, record = group.orders, geometry.restriction_source
    source_local = tuple(source_bank[int(index)] for index in row)
    expressions = coordinate_expressions(source_element, source_local, budget=budget)
    if expressions is None:
        state.add(
            "periodic_coordinate_source_expression",
            "unresolved",
            "cell",
            cell_ids[owner : owner + 1],
        )
        return None
    if any(
        expression_node_count(value, domain(kind), mesh.topological_dimension)
        > limits.maximum_bernstein_nodes
        for value in expressions
    ):
        state.add(
            "periodic_bernstein_node_budget",
            "unresolved",
            "cell",
            cell_ids[owner : owner + 1],
        )
        return None
    cell = Cell(
        expressions,
        domain(kind),
        kind,
        mesh.topological_dimension,
        identifiers(vertices, (0,) * len(orders)),
        embedding_vertices(kind),
    )
    root_domain_certified = False
    layer_column_root = (
        isinstance(source_element, LayerColumnCellGeometryElement)
        and source_element.fiber_graph
    )
    if layer_column_root:
        layer_column_owners.add(owner)
        reference_cell = _layer_fiber_reference_cell(
            source_element, source_local, None, cell, budget=budget
        )
        if reference_cell is None:
            raise ValueError(
                "periodic_embedding.layer_fiber must provide an affine carrier; got unsupported source"
            )
        layer_reference_cells[owner] = reference_cell
    if (
        record is not None
        and not isinstance(source_element, CellVertexGeometryElement)
        and not _has_full_coefficient_source(source_element)
    ):
        root_preparation = _prepare_periodic_root(
            request,
            prepared,
            cell,
            source_element,
            source_local,
            block,
            local_owner,
            owner,
            layer_column_root,
        )
        root_domain_certified, layer_column_root = (
            root_preparation.domain_certified,
            root_preparation.layer_column,
        )
    root_record = source_roots.get(owner)
    # An affine simplex piece is the interpolant of these rational corners; its
    # restriction to any shared sub-simplex is therefore determined by them.
    corners = (
        coordinate_enclosure_owner.affine_simplex_corner_images(
            source_element, source_local, budget=budget
        )
        if kind in ("interval", "triangle", "tetrahedron")
        and source_element.cell_kind == kind
        else None
    )
    if corners is not None:
        prepared.affine_corners[len(prepared.cells)] = corners
        if affine_mapped_contact_owner.affine_expressions(cell.coordinates):
            # These corners just came from the canonical full-source affine
            # producer. Bind them now, before any later consumer can change a
            # coefficient; nested image scopes retain this original bank.
            affine_mapped_contact_owner._retain_affine_cell_images(
                cell, corners, budget=budget
            )
    # The proof returns only a reason; its derivative/SPD tensors are not cells.
    with budget.temporary_scope():
        affine_validity = (
            request.validity is not None
            and request.validity.all_certified
            and state.closed_cell_orientation
            and block.cell_kind == kind == source_element.cell_kind
            and corners is not None
        )
        if affine_validity:
            prepared.affine_pieces.add(len(prepared.cells))
            prepared.affine_source_ids[len(prepared.cells)] = source_element.element_id
            if state.subdivision_pieces >= limits.maximum_subdivision_pieces:
                reason = "local_injectivity_piece_budget"
            else:
                state.subdivision_pieces += 1
                reason = None
                if "periodic_affine_validity_reuse" not in state.checks:
                    state.checks.append("periodic_affine_validity_reuse")
        elif layer_column_root and "cell_validity" in state.checks:
            # The Bernstein validity certificate proved column injectivity.
            # Quotient traces, images, and inter-cell contacts are checked below.
            reason = None
            if "periodic_layer_column_validity_reuse" not in state.checks:
                state.checks.append("periodic_layer_column_validity_reuse")
        elif root_record is None or not root_domain_certified:
            reason = injective(cell, limits, state, budget=budget)
        else:
            reason = injective(
                root_record.root, limits, state, budget=budget
            ) or injective(root_record.chart, limits, state, budget=budget)
        if reason is not None:
            state.add(reason, "unresolved", "cell", cell_ids[owner : owner + 1])
    return cell


def _prepare_periodic_cells(
    request: _PeriodicRequest, group: _PeriodicGroup
) -> _PeriodicCells | None:
    state, mesh, geometry, cell_ids, budget = (
        request.state,
        request.mesh,
        request.geometry,
        request.cell_ids,
        request.budget,
    )
    topology = request.topology
    elements, routes, coordinates = geometry.resolve(mesh, budget=budget)
    points = np.asarray(coordinates, dtype=np.float64)
    source_bank = prepared_coordinate_source_bank(geometry, budget=budget)
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    fixed = np.asarray(topology.vertex_fixed_generators, dtype=np.bool_)
    lifted = _LiftedVertices(roots, shifts, fixed, group.orders, {})
    budget.reserve(0, 128 + 192 * cell_ids.size)
    prepared = _PeriodicCells(
        [], [], [], set(), {}, {}, set(), source_bank, lifted, set(), {}, {}
    )
    first_cell = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        request.progress.resource_kind = "cell"
        request.progress.resource_ids = tuple(
            int(value) for value in cell_ids[first_cell : first_cell + block.cell_count]
        )
        source_rows = _periodic_source_rows(
            request, prepared, block, element, route, points, first_cell
        )
        if source_rows is None:
            return None
        for local_owner, kind, source_element, row, vertices in source_rows:
            owner = first_cell + local_owner
            request.progress.resource_ids = (int(cell_ids[owner]),)
            cell = _prepare_periodic_piece(
                request,
                prepared,
                group,
                block,
                source_element,
                row,
                vertices,
                kind,
                local_owner,
                owner,
            )
            if cell is None:
                return None
            prepared.cells.append(cell)
            prepared.owners.append(owner)
            prepared.piece_vertices.append(vertices)
        first_cell += block.cell_count
    layer_column_owners, layer_reference_cells = (
        prepared.layer_column_owners,
        prepared.layer_reference_cells,
    )
    if layer_column_owners:
        if layer_column_owners != set(layer_reference_cells):
            raise ValueError(
                "periodic_embedding.layer_reference_cells must cover every fiber child; got missing child charts"
            )
        state.checks.extend(
            (
                "periodic_layer_column_graph_source",
                "periodic_layer_column_affine_reference_contact",
            )
        )
    request.progress.resource_kind, request.progress.resource_ids = "mesh", ()
    return prepared


def _prepare_periodic_facets(
    request: _PeriodicRequest, prepared: _PeriodicCells, group: _PeriodicGroup
) -> _PeriodicFacets:
    mesh, topology, budget = request.mesh, request.topology, request.budget
    identifiers = prepared.lifted.identifiers
    orders, planar_polyhedral_owners = group.orders, prepared.planar_polyhedral_owners
    facets = embedding_state_owner.mesh_facets(mesh, budget=budget)
    entity_ids = embedding_state_owner.facet_entity_ids(mesh, facets.rows)
    facet_rows = {
        int(value): row
        for row, value in enumerate(
            np.asarray(mesh.entity_set(mesh.topological_dimension - 1).entity_ids)
        )
    }
    orbits, _, anchors = (
        np.asarray(value) for value in topology.orbits(mesh.topological_dimension - 1)
    )

    edge_orbits = np.asarray(topology.orbits(1)[0])
    edge_rows = {}
    for rows_, corners_ in periodic_topology_owner.lifted_loops(mesh, 1):
        edge_rows.update(
            (frozenset(int(vertex) for vertex in corners), int(row_))
            for row_, corners in zip(rows_, corners_, strict=True)
        )
    vertex_orbits = np.asarray(topology.orbits(0)[0])
    parent_faces: dict[int, list[_ParentFace]] = {}
    for occurrence, entity in enumerate(entity_ids.tolist()):
        owner = int(facets.cells[occurrence])
        if owner not in planar_polyhedral_owners:
            continue
        row_vertices = facets.rows[occurrence]
        row_vertices = row_vertices[row_vertices >= 0]
        budget.reserve(len(row_vertices), 128 + 64 * len(row_vertices))
        face_keys = identifiers(row_vertices, (0,) * len(orders))
        parent_faces.setdefault(owner, []).append(
            _ParentFace(
                mesh.topological_dimension - 1,
                int(orbits[facet_rows[entity]]),
                tuple(int(vertex) for vertex in row_vertices),
                face_keys,
            )
        )
        # Close each source facet through its declared edge and vertex orbits.
        budget.reserve(2 * len(row_vertices), 512 * len(row_vertices))
        for vertex in row_vertices:
            vertex_ = int(vertex)
            parent_faces[owner].append(
                _ParentFace(
                    0,
                    int(vertex_orbits[vertex_]),
                    (vertex_,),
                    identifiers(
                        np.asarray((vertex_,), dtype=np.int64), (0,) * len(orders)
                    ),
                )
            )
        for position, vertex in enumerate(row_vertices):
            edge = int(vertex), int(row_vertices[(position + 1) % len(row_vertices)])
            edge_row = edge_rows[frozenset(edge)]
            parent_faces[owner].append(
                _ParentFace(
                    1,
                    int(edge_orbits[edge_row]),
                    edge,
                    identifiers(np.asarray(edge, dtype=np.int64), (0,) * len(orders)),
                )
            )
    by_orbit: dict[
        int, dict[tuple[int, int], list[tuple[tuple[int, ...], tuple[int, ...]]]]
    ] = {}
    for owner, entities in parent_faces.items():
        unique = {
            (face.dimension, frozenset(face.source_vertices)): face for face in entities
        }
        parent_faces[owner] = list(unique.values())
        lookup: dict[tuple[int, int], list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        for face in parent_faces[owner]:
            lookup.setdefault((face.dimension, face.orbit), []).append(
                (face.source_vertices, face.lifted_vertices)
            )
        by_orbit[owner] = lookup
    return _PeriodicFacets(
        facets, entity_ids, facet_rows, orbits, anchors, parent_faces, by_orbit
    )


def _prepare_periodic_source_faces(
    request: _PeriodicRequest, prepared: _PeriodicCells, facets: _PeriodicFacets
) -> _PeriodicSourceFaces:
    mesh, topology, geometry, budget = (
        request.mesh,
        request.topology,
        request.geometry,
        request.budget,
    )
    parent_faces, source_bank = facets.parent_faces, prepared.source_bank
    original_facets = (
        _original_power_facet_authority(geometry, topology.cell, budget=budget)
        if isinstance(topology.cell, PeriodicIsometryGroup)
        else ()
    )
    source_faces = {}
    for owner, entities in parent_faces.items() if original_facets else ():
        for face in entities:
            degree, vertices_ = face.dimension, face.source_vertices
            if degree != mesh.topological_dimension - 1:
                continue
            points_ = tuple(source_bank[vertex] for vertex in vertices_)
            plane = source_polygon_plane(points_, budget=budget)
            if plane is None:
                continue
            key = plane[0]
            budget.reserve(1, 256)
            authorities = set()
            for facet_root, source_plane, axes_, polygon in original_facets:
                budget.reserve(1)
                if source_plane == key and closed_source_polygon_contains(
                    polygon,
                    planar_primitives_owner.project(points_, axes_, budget=budget),
                    budget=budget,
                ):
                    if facet_root not in authorities:
                        budget.reserve(1, 128)
                    authorities.add(facet_root)
            if authorities:
                budget.reserve(len(authorities), 128 + 32 * len(authorities))
                source_faces[(owner, vertices_)] = (key, authorities)
    return _PeriodicSourceFaces(original_facets, source_faces)


def _trace_already_permitted(
    support: tuple[int, ...], permitted: dict[tuple[int, ...], None], /
) -> bool:
    """Reuse one authenticated convex simplex trace, never a union of traces."""
    return any(all(vertex in trace for vertex in support) for trace in permitted)


def _transport_source_point(
    point: tuple[coordinate_enclosure_owner.ExactScalar, ...],
    action: tuple[tuple[Fraction, ...], ...],
    cache: _PeriodicContactCache,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[coordinate_enclosure_owner.ExactScalar, ...]:
    """Reuse only exact numeric point transport, never trace authorization."""
    point_key = point, action
    transformed = cache.transformed_source_points.get(point_key)
    if transformed is None:
        budget.reserve(len(point) * (len(point) + 1))
        transformed = tuple(
            row[-1]
            + sum(
                (weight * value for weight, value in zip(row[:-1], point, strict=True)),
                Fraction(0),
            )
            for row in action[:-1]
        )
        budget.retain_basis((point_key, transformed))
        cache.transformed_source_points[point_key] = transformed
    return transformed


def _permit_orbit_traces(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    facets: _PeriodicFacets,
    first: Cell,
    owner_a: int,
    owner_b: int,
    action: tuple[tuple[Fraction, ...], ...],
    image: tuple[int, ...],
    permitted: dict[tuple[int, ...], None],
    cache: _PeriodicContactCache,
) -> None:
    budget, source_bank, identifiers = (
        request.budget,
        prepared.source_bank,
        prepared.lifted.identifiers,
    )
    parent_faces, by_orbit = facets.parent_faces, facets.by_orbit
    for face in parent_faces[owner_a]:
        degree_a, orbit_a, face_a, keys_a = (
            face.dimension,
            face.orbit,
            face.source_vertices,
            face.lifted_vertices,
        )
        source_a = dict(
            zip(
                keys_a,
                (source_bank[vertex] for vertex in face_a),
                strict=True,
            )
        )
        support = tuple(vertex for vertex in first.vertices if vertex in source_a)
        if not support or len(support) > first.dimension or _trace_already_permitted(support, permitted):
            continue
        for face_b, _ in by_orbit[owner_b].get((degree_a, orbit_a), ()):
            budget.reserve(len(face_b))
            keys_b = identifiers(np.asarray(face_b, dtype=np.int64), image)
            if set(keys_a) != set(keys_b):
                continue
            equal = True
            for key, vertex in zip(keys_b, face_b, strict=True):
                point = source_bank[vertex]
                transformed = _transport_source_point(point, action, cache, budget=budget)
                if transformed != source_a[key]:
                    equal = False
                    break
            if equal:
                budget.reserve(len(support), 128 + 64 * len(support))
                permitted[support] = None
                break


def _transport_source_plane(
    plane: tuple[coordinate_enclosure_owner.ExactScalar, ...],
    action: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[coordinate_enclosure_owner.ExactScalar, ...]:
    """Transport an earned plane through an exactly admitted proper isometry."""
    dimension = len(plane) - 1
    planar_primitives_owner.reserve_fraction_work(
        (*action, plane),
        2 * dimension**2 + 2 * dimension + 1 + dimension + 1,
        3 * (dimension + 1),
        (4 * dimension + 5) * dimension,
        budget=budget,
    )
    # x' = R x + t and R^-1 = R^T give n' = R n, d' = d - n'·t.
    normal = tuple(
        sum((a * b for a, b in zip(row[:-1], plane[:-1], strict=True)), Fraction(0))
        for row in action[:-1]
    )
    offset = plane[-1] - sum(
        (value * row[-1] for value, row in zip(normal, action[:-1], strict=True)),
        Fraction(0),
    )
    divisor = next((value for value in normal if value), None)
    if divisor is None:
        raise ValueError(
            "periodic_trace.action must preserve the nonzero source-plane normal; got a singular action"
        )
    return tuple(value / divisor for value in (*normal, offset))


def _permit_source_plane_traces(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    first: Cell,
    owner_b: int,
    action: tuple[tuple[Fraction, ...], ...],
    first_source_vertices: np.ndarray,
    permitted: dict[tuple[int, ...], None],
    cache: _PeriodicContactCache,
) -> None:
    mesh, budget, source_bank = request.mesh, request.budget, prepared.source_bank
    parent_faces, original_facets, source_faces = (
        facets.parent_faces,
        sources.original_facets,
        sources.source_faces,
    )
    source_vertices = tuple(int(vertex) for vertex in first_source_vertices)
    for source_facet, (facet_root, source_plane, axes_, polygon) in enumerate(original_facets):
        # A periodic authority root may contain several different source planes.
        support_key = first.vertices, source_vertices, source_facet
        if support_key in cache.source_supports:
            support = cache.source_supports[support_key]
        else:
            budget.reserve(0, 128 + 64 * len(first_source_vertices))
            support_points = []
            support_ids = []
            for identifier, vertex in zip(first.vertices, source_vertices, strict=True):
                point = source_bank[vertex]
                budget.reserve(2 * len(point))
                if sum(
                    (a * b for a, b in zip(source_plane[:-1], point, strict=True)),
                    source_plane[-1],
                ):
                    continue
                if closed_source_polygon_contains(
                    polygon,
                    planar_primitives_owner.project((point,), axes_, budget=budget),
                    budget=budget,
                ):
                    support_points.append(point)
                    support_ids.append(identifier)
            support = tuple(support_ids)
            if support and _trace_already_permitted(support, permitted):
                continue
            if not support or len(support) > first.dimension:
                support = None
            elif not closed_source_polygon_contains(
                polygon,
                planar_primitives_owner.project(tuple(support_points), axes_, budget=budget),
                budget=budget,
            ):
                support = None
            budget.retain_basis((support_key, support))
            cache.source_supports[support_key] = support
        if support is None or _trace_already_permitted(support, permitted):
            continue
        for face in parent_faces[owner_b]:
            degree_b, face_b = face.dimension, face.source_vertices
            if degree_b != mesh.topological_dimension - 1:
                continue
            authority_b = source_faces.get((owner_b, face_b))
            if authority_b is None or facet_root not in authority_b[1]:
                continue
            plane_key = authority_b[0], action
            if plane_key not in cache.transformed_face_planes:
                plane_b = _transport_source_plane(authority_b[0], action, budget=budget)
                budget.retain_basis((plane_key, plane_b))
                cache.transformed_face_planes[plane_key] = plane_b
            plane_b = cache.transformed_face_planes[plane_key]
            if plane_b == source_plane:
                budget.reserve(len(support), 128 + 64 * len(support))
                permitted[support] = None
                break


def _authored_periodic_traces(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    first: Cell,
    owner_a: int,
    owner_b: int,
    action: tuple[tuple[Fraction, ...], ...],
    image: tuple[int, ...],
    first_source_vertices: np.ndarray,
    cache: _PeriodicContactCache,
) -> tuple[tuple[int, ...], ...]:
    if owner_a not in facets.parent_faces or owner_b not in facets.parent_faces:
        return ()
    if (
        prepared.lifted.identifiers(first_source_vertices, (0,) * len(group.orders))
        != first.vertices
    ):
        raise ValueError(
            "periodic_trace.first_source_vertices must identify every first-piece lifted corner; got mismatched source bank"
        )
    permitted: dict[tuple[int, ...], None] = {}
    _permit_source_plane_traces(
        request,
        prepared,
        facets,
        sources,
        first,
        owner_b,
        action,
        first_source_vertices,
        permitted,
        cache,
    )
    _permit_orbit_traces(
        request, prepared, facets, first, owner_a, owner_b, action, image, permitted, cache
    )
    return tuple(permitted)


class _PieceTrace:
    """One piece in a facet's normalized lift, compared on its shared entities.

    A piece whose whole coordinate map is a proven affine simplex keeps only
    its isometry-transformed rational corners: two affine simplices restrict to
    the same map on every shared sub-simplex exactly when their shared corner
    images agree. Any other pair is decided by the full expression restriction.
    """

    __slots__ = ("_action", "_cell", "_source", "corners", "vertices")

    def __init__(
        self,
        source: Cell,
        action: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...],
        vertices: tuple[int, ...],
        corners: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...] | None,
    ) -> None:
        self._source, self._action, self._cell = source, action, None
        self.vertices, self.corners = vertices, corners

    @classmethod
    def prepare(
        cls,
        source: Cell,
        corners: CoordinateSourceBank | None,
        action: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...],
        vertices: tuple[int, ...],
        *,
        budget: CoordinateEnclosureBudget,
    ) -> _PieceTrace:
        if corners is None:
            return cls(source, action, vertices, None)
        budget.reserve(
            len(corners) * (len(action) - 1) * len(action),
            128 + 64 * len(corners) * (len(action) - 1),
        )
        images = tuple(
            tuple(
                sum(
                    (weight * value for weight, value in zip(row[:-1], point, strict=True)),
                    row[-1],
                )
                for row in action[:-1]
            )
            for point in corners
        )
        return cls(source, action, vertices, images)

    def cell(self, *, budget: CoordinateEnclosureBudget) -> Cell:
        if self._cell is None:
            self._cell = _transform(self._source, self._action, self.vertices, budget=budget)
        return self._cell

    def equal(self, other: _PieceTrace, *, budget: CoordinateEnclosureBudget) -> bool:
        if self.corners is None or other.corners is None:
            return cell_traces_equal(
                self.cell(budget=budget), other.cell(budget=budget), budget=budget
            )
        shared = set(self.vertices).intersection(other.vertices)
        budget.reserve(len(shared) * (1 + len(self.corners[0])))
        return all(
            self.corners[self.vertices.index(key)]
            == other.corners[other.vertices.index(key)]
            for key in shared
        )



def _certify_periodic_traces(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    prepared_facets: _PeriodicFacets,
) -> None:
    state, mesh, topology, budget = (
        request.state,
        request.mesh,
        request.topology,
        request.budget,
    )
    cells, owners, piece_vertices, identifiers = (
        prepared.cells,
        prepared.owners,
        prepared.piece_vertices,
        prepared.lifted.identifiers,
    )
    matrices, orders = group.matrices, group.orders
    facets, entity_ids, facet_rows, orbits, anchors = (
        prepared_facets.facets,
        prepared_facets.entity_ids,
        prepared_facets.facet_rows,
        prepared_facets.orbits,
        prepared_facets.anchors,
    )
    initial_findings = len(state.findings)
    traces: dict[int, tuple[_PieceTrace, ...]] = {}
    owner_pieces: dict[int, list[int]] = {}
    for piece, owner in enumerate(owners):
        owner_pieces.setdefault(owner, []).append(piece)
    lattice = ExactPeriodicActionLattice(matrices, orders)
    for occurrence, entity in enumerate(entity_ids.tolist()):
        request.progress.resource_kind, request.progress.resource_ids = (
            "facet",
            (int(entity),),
        )
        row = facet_rows[entity]
        image = tuple(-int(value) for value in anchors[row])
        cell_row = int(facets.cells[occurrence])
        action = lattice.prepare((image,), budget=budget)[0]
        normalized = tuple(
            _PieceTrace.prepare(
                cells[piece],
                prepared.affine_corners.get(piece),
                action,
                identifiers(piece_vertices[piece], image),
                budget=budget,
            )
            for piece in owner_pieces[cell_row]
        )
        orbit = int(orbits[row])
        previous = traces.get(orbit)
        if previous is not None and any(
            not a.equal(b, budget=budget) for a in previous for b in normalized
        ):
            state.add("periodic_mapped_trace_mismatch", "violated", "facet", (entity,))
        traces[orbit] = normalized
    state.boundary_facets = int(
        np.count_nonzero(
            np.asarray(
                topology.quotient.entities(mesh.topological_dimension - 1)
                .subset("boundary")
                .mask
            )
        )
    )
    if request.validity is not None and len(state.findings) == initial_findings:
        request.progress.facet_trace_certificate_id = (
            fingerprint_owner.canonical_fingerprint(
                {
                    "kind": "periodic-full-source-facet-traces",
                    "mesh": mesh.mesh_id,
                    "geometry": request.validity.geometry_id,
                    "periodic_topology": topology.periodic_topology_id,
                    "facets": tuple(int(value) for value in entity_ids),
                }
            )
        )
    request.progress.resource_kind, request.progress.resource_ids = "mesh", ()


def _prepare_periodic_affine_quotient(
    request: _PeriodicRequest, prepared: _PeriodicCells, facets: _PeriodicFacets
) -> None:
    """Bind the three earned premises of the closed affine quotient theorem."""
    validity = request.validity
    trace_id = request.progress.facet_trace_certificate_id
    mesh, state = request.mesh, request.state
    if (
        validity is None
        or not validity.all_certified
        or not state.closed_cell_orientation
        or state.findings
        or state.boundary_facets
        or trace_id is None
        or facets.parent_faces
        or mesh.topological_dimension != mesh.ambient_dimension
        or mesh.topological_dimension not in (2, 3)
        or len(prepared.affine_pieces) != len(prepared.cells)
        or not prepared.cells
    ):
        return
    request.budget.reserve(len(prepared.cells), 1024 + 128 * len(prepared.cells))
    affinity_id = fingerprint_owner.canonical_fingerprint(
        {
            "kind": "periodic-full-source-affine-simplex-bank",
            "mesh": mesh.mesh_id,
            "geometry": validity.geometry_id,
            "pieces": tuple(
                (
                    int(request.cell_ids[owner]),
                    prepared.affine_source_ids[piece],
                )
                for piece, owner in enumerate(prepared.owners)
            ),
        }
    )
    state.periodic_affine = embedding_state_owner.PeriodicAffineQuotientEvidence(
        mesh.mesh_id,
        validity.geometry_id,
        request.topology.periodic_topology_id,
        validity.certificate_id,
        affinity_id,
        trace_id,
    )
    state.checks.append("periodic_affine_quotient_contact")


def _periodic_piece_bounds(
    cell: Cell, affine: bool, *, budget: CoordinateEnclosureBudget
) -> tuple[np.ndarray, np.ndarray]:
    """Bound a proved affine simplex at its vertices; otherwise use Bernstein."""
    if not affine:
        return bounds(
            cell,
            np.zeros(cell.dimension, dtype=np.float64),
            np.eye(cell.dimension, dtype=np.float64),
            budget=budget,
        )
    budget.reserve(len(cell.coordinates) * len(cell.reference_vertices))
    intervals = []
    for coordinate in cell.coordinates:
        values = tuple(
            expression_evaluate(coordinate, point, budget=budget)
            for point in cell.reference_vertices
        )
        if any(isinstance(value, RadicalElement) for value in values):
            reserve_algebraic_work(
                values, 4 * len(values), 2 * len(values), 1, budget=budget
            )
        lower = coefficient_interval(min(values), budget=budget)[0]
        upper = coefficient_interval(max(values), budget=budget)[1]
        intervals.append(
            (
                fraction_endpoint(lower, upper=False, saturate=True),
                fraction_endpoint(upper, upper=True, saturate=True),
            )
        )
    result = np.asarray(intervals, dtype=np.float64)
    return result[:, 0], result[:, 1]


def _prepare_periodic_images(
    request: _PeriodicRequest, prepared: _PeriodicCells, group: _PeriodicGroup
) -> _PeriodicImages | None:
    state, mesh, cell_ids, budget = (
        request.state,
        request.mesh,
        request.cell_ids,
        request.budget,
    )
    cells, owners, finite, finite_count, translations = (
        prepared.cells,
        prepared.owners,
        group.finite,
        group.finite_count,
        group.translations,
    )
    maximum_images = request.limits.maximum_periodic_images
    cell_ids = cell_ids[np.asarray(owners, dtype=np.int64)]
    budget.reserve(
        finite_count * len(cells),
        finite_count * len(cells) * mesh.ambient_dimension * 16,
    )
    base_boxes = [
        _periodic_piece_bounds(cell, index in prepared.affine_pieces, budget=budget)
        for index, cell in enumerate(cells)
    ]
    boxes = [
        base_boxes[index]
        if not any(exponents)
        else _periodic_piece_bounds(
            _transform(cell, matrix, cell.vertices, budget=budget),
            index in prepared.affine_pieces,
            budget=budget,
        )
        for exponents, matrix in finite
        for index, cell in enumerate(cells)
    ]
    if any(not np.all(np.isfinite(bound)) for box in boxes for bound in box):
        state.add("periodic_coordinate_bound_range", "unresolved", "mesh")
        return None
    ranges = translation_ranges(boxes, translations)
    image_count = finite_count
    for interval in ranges:
        image_count *= len(interval)
    request.progress.image_count = image_count
    if image_count > maximum_images:
        state.add("periodic_image_budget", "unresolved", "mesh")
        return None
    request.progress.image_count = image_count
    return _PeriodicImages(ranges, base_boxes, cell_ids)


def _periodic_root_contact(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    first_piece: int,
    second_piece: int,
    image: Cell,
    transform_: tuple[tuple[Fraction, ...], ...],
) -> bool:
    state, limits, budget = request.state, request.limits, request.budget
    source_roots, owners = prepared.source_roots, prepared.owners
    a, b = first_piece, second_piece
    root_a, root_b = (
        source_roots.get(owners[a]),
        source_roots.get(owners[b]),
    )
    if (
        root_a is not None
        and root_b is not None
        and root_a.parent == root_b.parent
        and (root_a.element_id, root_a.parent_vertices)
        == (root_b.element_id, root_b.parent_vertices)
        and expression_sequence_equal(
            root_a.root.coordinates,
            root_b.root.coordinates,
            budget=budget,
        )
    ):
        chart_b = replace(root_b.chart, vertices=image.vertices, corner_images=None)
        if _root_action_pair(
            root_a.root,
            root_a.chart,
            chart_b,
            transform_,
            root_a.corners,
            state,
            limits,
            budget=budget,
        ):
            return True
    return False


def _layer_periodic_contact(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    a: int,
    b: int,
    image: Cell,
    transform_: tuple[tuple[Fraction, ...], ...],
) -> ContactDecision | None:
    state, limits, budget = request.state, request.limits, request.budget
    reference_first = prepared.layer_reference_cells.get(prepared.owners[a])
    reference_second = prepared.layer_reference_cells.get(prepared.owners[b])
    if reference_first is None or reference_second is None:
        return None
    reference_image = _transform(
        reference_second, transform_, image.vertices, budget=budget
    )
    if separating_plane(reference_first, reference_image, budget=budget):
        return ContactDecision("separated", 0)
    result = affine_reference_contact(
        reference_first,
        reference_image,
        limits.maximum_subdivision_pieces - state.subdivision_pieces,
        budget=budget,
    )
    if result is None:
        result = pair_decision(
            reference_first,
            reference_image,
            limits,
            limits.maximum_subdivision_pieces - state.subdivision_pieces,
            budget=budget,
        )
    state.subdivision_pieces += result.work
    return result


def _record_periodic_contact(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    images: _PeriodicImages,
    first: Cell,
    image: Cell,
    transform_: tuple[tuple[Fraction, ...], ...],
    exponents: tuple[int, ...],
    a: int,
    b: int,
    decision: str,
) -> None:
    state, cell_ids, owners = request.state, images.cell_ids, prepared.owners
    if decision == "overlap":
        state.findings.append(
            embedding_state_owner.MeshCertificateFinding(
                "periodic_mapped_cell_overlap",
                "violated",
                "cell",
                tuple(int(value) for value in cell_ids[[a, b]]),
                observations=source_contact_quantities(
                    first,
                    image,
                    transform_,
                    exponents,
                    a,
                    b,
                    owners[a],
                    owners[b],
                ),
            )
        )
    elif decision != "separated":
        state.add(
            decision,
            "unresolved",
            "cell",
            cell_ids[[a, b]],
        )


def _certify_periodic_pair(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    images: _PeriodicImages,
    cache: _PeriodicContactCache,
    a: int,
    b: int,
    image: Cell,
    transform_: tuple[tuple[Fraction, ...], ...],
    exponents: tuple[int, ...],
) -> None:
    state, limits, budget = request.state, request.limits, request.budget
    first, cell_ids, owners = prepared.cells[a], images.cell_ids, prepared.owners
    inverse_key = _inverse_equivalent_pair_key(a, b, exponents, group.orders)
    key = None if facets.parent_faces else inverse_key
    inverse_decision = cache.decisions.get(inverse_key)
    if inverse_decision is not None:
        if not cache.recorded_reuse:
            state.checks.append("periodic_inverse_pair_proof_reuse")
            cache.recorded_reuse = True
        _record_periodic_contact(
            request,
            prepared,
            images,
            first,
            image,
            transform_,
            exponents,
            a,
            b,
            inverse_decision,
        )
        return
    if not cell_traces_equal(first, image, budget=budget):
        state.add("periodic_mapped_trace_mismatch", "violated", "cell", cell_ids[[a, b]])
        return
    if state.periodic_affine is not None:
        overlap = affine_mapped_contact_owner.affine_simplex_interiors_overlap(
            first, image, budget=budget
        )
        if overlap is not None:
            decision = "overlap" if overlap else "separated"
            if key is not None:
                cache.decisions[key] = decision
            _record_periodic_contact(
                request,
                prepared,
                images,
                first,
                image,
                transform_,
                exponents,
                a,
                b,
                decision,
            )
            return
    if _tensor_pair(first, image, state, limits, budget=budget):
        return
    if _periodic_root_contact(request, prepared, a, b, image, transform_):
        return
    algebraic = any(
        isinstance(coefficient, RadicalElement)
        for cell in (first, image)
        for expression in cell.coordinates
        if isinstance(expression, dict)
        for coefficient in expression.values()
    )
    trace_authority_consumed = False

    def prepare_traces() -> tuple[tuple[int, ...], ...]:
        nonlocal trace_authority_consumed
        trace_authority_consumed = True
        # Authority depends on the first piece and the other parent boundary,
        # not on which tetrahedron of that parent is being compared.
        trace_key = a, owners[b], exponents
        allowed_traces = cache.trace_permissions.get(trace_key)
        if allowed_traces is None:
            allowed_traces = _authored_periodic_traces(
                request,
                prepared,
                group,
                facets,
                sources,
                first,
                owners[a],
                owners[b],
                transform_,
                exponents,
                prepared.piece_vertices[a],
                cache,
            )
            budget.retain_basis((trace_key, allowed_traces))
            cache.trace_permissions[trace_key] = allowed_traces
        return allowed_traces

    reference = _layer_periodic_contact(request, prepared, a, b, image, transform_)
    if reference is not None and reference.status == "separated":
        return
    decision = (
        (
            reference.unresolved_check()
            if reference.status == "unresolved"
            else reference.status
        )
        if reference is not None
        else None
    )
    if decision is None:
        affine_contact = affine_mapped_contact_owner.affine_simplex_contact(
            first,
            image,
            limits.maximum_work_units
            if algebraic
            else limits.maximum_subdivision_pieces - state.subdivision_pieces,
            trace_preparation=prepare_traces,
            budget=budget,
        )
        if affine_contact is not None:
            decision = (
                affine_contact.unresolved_check()
                if affine_contact.status == "unresolved"
                else affine_contact.status
            )
            work = affine_contact.work
            if decision == "separated" and not trace_authority_consumed:
                # Shared-simplex support is symmetric under the inverse exact
                # isometry, even when other pairs need authored parent traces.
                budget.retain_basis((inverse_key,))
                cache.decisions[inverse_key] = decision
        elif separating_plane(first, image, budget=budget):
            decision, work = "separated", 0
        else:
            contact = pair_decision(
                first,
                image,
                limits,
                limits.maximum_subdivision_pieces - state.subdivision_pieces,
                authenticated_first_traces=prepare_traces(),
                budget=budget,
            )
            decision = (
                contact.unresolved_check()
                if contact.status == "unresolved"
                else contact.status
            )
            work = contact.work
        state.subdivision_pieces += work
        if key is not None and decision in ("overlap", "separated"):
            cache.decisions[key] = decision
    _record_periodic_contact(
        request, prepared, images, first, image, transform_, exponents, a, b, decision
    )


def _periodic_contact_tree(
    lower: np.ndarray,
    upper: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> PackedBVH:
    def storage(amount: int) -> None:
        budget.reserve(0, amount)

    tree = prepare_bvh(
        lower,
        upper,
        dtype=np.float64,
        _charge_work=budget.reserve,
        _reserve_storage=storage,
    )
    # Host traversal retains a second bank: node bounds/links, leaf payload,
    # and item bounds. Admit it before the tree owner materializes it.
    budget.reserve(
        0,
        1024
        + 2 * lower.shape[0] * (16 * lower.shape[1] + 24)
        + (lower.shape[0] + tree.leaf_size) * 8
        + lower.nbytes
        + upper.nbytes,
    )
    return tree


def _certify_periodic_image_candidates(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    images: _PeriodicImages,
    cache: _PeriodicContactCache,
    transform_: tuple[tuple[Fraction, ...], ...],
    exponents: tuple[int, ...],
    b: int,
    candidates: np.ndarray,
) -> bool:
    state, budget, limits = request.state, request.budget, request.limits
    request.progress.resource_kind, request.progress.resource_ids = (
        "cell",
        (int(images.cell_ids[b]),),
    )
    with budget.polynomial_preparation_scope():
        original = prepared.cells[b]
        identity_image = not any(exponents)
        image = original if identity_image else _transform(
            original,
            transform_,
            prepared.lifted.identifiers(prepared.piece_vertices[b], exponents),
            budget=budget,
        )
        if not identity_image and affine_mapped_contact_owner.affine_expressions(original.coordinates):
            # Transform already-issued exact original corners directly, rather
            # than evaluating the newly transformed polynomial at them again.
            # The image key is issued immediately by this actual constructor.
            corners = affine_mapped_contact_owner._cell_images(original, budget=budget)
            budget.reserve(0, 128 + 64 * len(corners))
            image_corners = tuple(
                _transport_source_point(point, transform_, cache, budget=budget)
                for point in corners
            )
            affine_mapped_contact_owner._retain_affine_cell_images(
                image, image_corners, budget=budget
            )
        for index in candidates:
            a = int(index)
            with budget.temporary_scope():
                if not any(exponents) and (
                    a >= b or prepared.owners[a] == prepared.owners[b]
                ):
                    continue
                # The BVH already used a sound outward enclosure of the full
                # source image under this exact action. Its candidates go
                # directly to exact contact proof, not a duplicate box filter.
                if state.candidate_pairs >= limits.maximum_candidate_pairs:
                    state.add(
                        "periodic_candidate_pair_budget",
                        "unresolved",
                        "cell",
                        images.cell_ids[[a, b]],
                    )
                    return False
                state.candidate_pairs += 1
                request.progress.resource_ids = (
                    int(images.cell_ids[a]),
                    int(images.cell_ids[b]),
                )
                _certify_periodic_pair(
                    request,
                    prepared,
                    group,
                    facets,
                    sources,
                    images,
                    cache,
                    a,
                    b,
                    image,
                    transform_,
                    exponents,
                )
    return True


def _ordered_image_candidate_rows(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    count: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, ...]:
    """Bucket query IDs before sorting, preserving the full lexicographic bank.

    The second index is already an integer in ``range(count)``. Sorting all
    pairs compares unrelated query rows; integer bucketing visits each pair
    once, and only the first indices within each query need comparisons.
    Duplicates are retained, as in the previous full-bank ordering.
    """
    pairs = sum(first.size for first, _ in blocks)
    budget.reserve(count + 2 * pairs, 256 + 96 * count + 64 * pairs)
    rows: list[list[int]] = [[] for _ in range(count)]
    for first, second in blocks:
        for a, b in zip(first, second, strict=True):
            rows[int(b)].append(int(a))
    result: list[np.ndarray] = []
    for row in rows:
        size = len(row)
        budget.reserve(size * max(1, size.bit_length()))
        result.append(np.asarray(sorted(row), dtype=np.int64))
    return tuple(result)


def _certify_periodic_contact_chunk(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    images: _PeriodicImages,
    cache: _PeriodicContactCache,
    tree: PackedBVH,
    transform_: tuple[tuple[Fraction, ...], ...],
    exponents: tuple[int, ...],
    start: int,
    stop: int,
    candidate_rows: list[tuple[int, ...]] | None,
    source_box: tuple[np.ndarray, np.ndarray],
) -> bool:
    budget = request.budget
    dimension = request.mesh.ambient_dimension
    count = stop - start
    # Affine images of the full source enclosures bound every curved or
    # rational map. Directed field-valued row extrema retain touching contacts.
    with budget.temporary_scope():
        budget.reserve(2 * count * dimension, 512 * count + 16 * count * dimension)
        lower = np.empty((count, dimension), dtype=np.float64)
        upper = np.empty_like(lower)
        for local, b in enumerate(range(start, stop)):
            request.progress.resource_kind, request.progress.resource_ids = (
                "cell",
                (int(images.cell_ids[b]),),
            )
            if any(exponents):
                lower[local], upper[local] = transformed_source_box(
                    *images.base_boxes[b], transform_, budget=budget
                )
            else:
                # The zero group action leaves the already outward source
                # enclosure unchanged; do not reconstruct it with arithmetic.
                lower[local], upper[local] = images.base_boxes[b]
        # The earned whole-source enclosure contains every source item. Reject
        # image items outside it before rebuilding/traversing a query hierarchy;
        # the identical touching predicate still admits every possible pair.
        budget.reserve(
            count * (2 * dimension + 4),
            512 + 128 * count * dimension,
        )
        overlaps = _host_boxes_overlap(
            lower, upper, source_box[0], source_box[1], True, 0.0, 0.0
        )
        budget.reserve(count)
        active_count = int(np.count_nonzero(overlaps))
        if active_count == 0:
            return True
        query_indices = None
        if active_count != count:
            budget.reserve(
                count + 2 * active_count * dimension,
                256 + 8 * active_count + 16 * active_count * dimension,
            )
            query_indices = np.flatnonzero(overlaps)
            query_lower, query_upper = lower[query_indices], upper[query_indices]
        else:
            query_lower, query_upper = lower, upper
        query = _periodic_contact_tree(query_lower, query_upper, budget)
        maximum_block_pairs = 256
        # Pair banks, concatenation, lexicographic ordering and traversal stacks
        # are admitted even when the broad phase rejects every item.
        budget.reserve(
            0,
            64 * len(prepared.cells) * active_count
            + maximum_block_pairs
            * (512 + 32 * dimension)
            * (tree.max_depth + query.max_depth + 2),
        )

        def visit(count: int) -> None:
            budget.reserve(count * (2 * dimension + 4))

        owners: list[object] = []
        budget.reserve(
            sum(
                2 * value.bbox_min.size
                + 3 * value.left.size
                + value.leaf_items.size
                + 2 * value.item_bbox_min.size
                for value in (tree, query)
            )
        )
        blocks = list(
            bvh_overlap_pair_blocks(
                tree,
                query,
                include_touching=True,
                visit=visit,
                maximum_block_pairs=maximum_block_pairs,
                retain_owner=owners.append,
            )
        )
        if not blocks:
            return True
        rows = _ordered_image_candidate_rows(blocks, active_count, budget=budget)
        for local, selected in enumerate(rows):
            b = start + (local if query_indices is None else int(query_indices[local]))
            if candidate_rows is not None:
                budget.reserve(0, 128 + 48 * selected.size)
                row = tuple(int(index) for index in selected)
                budget.retain_polynomial_preparation((row,))
                candidate_rows[b] = row
            if selected.size and not _certify_periodic_image_candidates(
                request,
                prepared,
                group,
                facets,
                sources,
                images,
                cache,
                transform_,
                exponents,
                b,
                selected,
            ):
                return False
    return True


def _certify_periodic_contacts(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    images: _PeriodicImages,
) -> None:
    budget = request.budget
    cache = _PeriodicContactCache({})
    if not prepared.cells:
        return
    # Source whole-map outward bounds and tree are shared by every image chunk.
    with budget.temporary_scope(), budget.polynomial_preparation_scope():
        # Canonical affine sources already issued their original corners. Seed
        # only one fallback chunk; transformed expression-key aliases remain
        # image-consumer-local, while exact point/action values are request-owned.
        for cell in prepared.cells[:_PERIODIC_CONTACT_CHUNK_SIZE]:
            if affine_mapped_contact_owner.affine_expressions(cell.coordinates):
                affine_mapped_contact_owner._cell_images(cell, budget=budget)
        budget.reserve(
            2 * len(prepared.cells) * request.mesh.ambient_dimension,
            16 * len(prepared.cells) * request.mesh.ambient_dimension,
        )
        lower = np.asarray([box[0] for box in images.base_boxes], dtype=np.float64)
        upper = np.asarray([box[1] for box in images.base_boxes], dtype=np.float64)
        tree = _periodic_contact_tree(lower, upper, budget)
        budget.reserve(
            2 * len(prepared.cells) * request.mesh.ambient_dimension,
            16 * request.mesh.ambient_dimension,
        )
        source_box = np.min(lower, axis=0), np.max(upper, axis=0)
        _certify_periodic_contact_images(
            request, prepared, group, facets, sources, images, cache, tree, source_box
        )


def _transpose_image_candidates(
    rows: tuple[tuple[int, ...], ...], /, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[int, ...], ...]:
    """Transpose a complete conservative relation without another geometric query."""
    budget.reserve(0, 512 + 192 * len(rows) + 64 * sum(map(len, rows)))
    transposed: list[list[int]] = [[] for _ in rows]
    for second, firsts in enumerate(rows):
        for first in firsts:
            transposed[first].append(second)
    return tuple(tuple(row) for row in transposed)


def _certify_periodic_contact_images(
    request: _PeriodicRequest,
    prepared: _PeriodicCells,
    group: _PeriodicGroup,
    facets: _PeriodicFacets,
    sources: _PeriodicSourceFaces,
    images: _PeriodicImages,
    cache: _PeriodicContactCache,
    tree: PackedBVH,
    source_box: tuple[np.ndarray, np.ndarray],
) -> None:
    lattice = ExactPeriodicActionLattice(group.matrices, group.orders)
    budget = request.budget
    budget.reserve(0, 512)
    candidate_banks: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
    for finite_exponents, _ in group.finite:
        for translation_exponents in product(*images.ranges):
            iterator = iter(translation_exponents)
            # Winding remains G^(r + L*k) on each declared generator axis.
            exponents = tuple(
                value + period * next(iterator) if period else value
                for value, period in zip(finite_exponents, group.periods, strict=True)
            )
            transform_ = lattice.prepare((exponents,), budget=request.budget)[0]
            action_key = lattice.key(exponents)
            inverse_key = tuple(
                -value % order if order else -value
                for value, order in zip(action_key, group.orders, strict=True)
            )
            inverse_bank = candidate_banks.get(inverse_key)
            if inverse_bank is not None:
                # A ∩ G(B) maps bijectively to G^-1(A) ∩ B. Transposing a
                # complete conservative bank is therefore conservative too;
                # current contact/authority predicates still run for every row.
                with budget.temporary_scope():
                    rows = _transpose_image_candidates(inverse_bank, budget=budget)
                    for b, firsts in enumerate(rows):
                        if firsts and not _certify_periodic_image_candidates(
                            request, prepared, group, facets, sources, images, cache,
                            transform_, exponents, b, np.asarray(firsts, dtype=np.int64),
                        ):
                            return
                continue
            budget.reserve(0, 128 + 32 * len(prepared.cells))
            candidate_rows: list[tuple[int, ...]] | None = (
                [()] * len(prepared.cells) if inverse_key != action_key else None
            )
            for start in range(0, len(prepared.cells), _PERIODIC_CONTACT_CHUNK_SIZE):
                if not _certify_periodic_contact_chunk(
                    request,
                    prepared,
                    group,
                    facets,
                    sources,
                    images,
                    cache,
                    tree,
                    transform_,
                    exponents,
                    start,
                    min(start + _PERIODIC_CONTACT_CHUNK_SIZE, len(prepared.cells)),
                    candidate_rows,
                    source_box,
                ):
                    return
            if candidate_rows is not None:
                # Publish only after every chunk of this image was visited.
                bank = tuple(candidate_rows)
                budget.retain_polynomial_preparation((action_key, bank))
                candidate_banks[action_key] = bank


def _certify_periodic_restriction(
    request: _PeriodicRequest, restriction: PeriodicRestrictionPremise
) -> None:
    """Prove target-only premises, then transfer the source embedding or fall back.

    Group identity, cell preparation, facet traces and the exact parent and
    boundary-subdivision partition are current target proofs. Without admitted
    partition evidence the bounded image-pair proof below decides the whole
    embedding.
    """
    state, limits, progress = request.state, request.limits, request.progress
    state.checks.extend(
        ("periodic_authored_group_identity", "periodic_mapped_trace_equivariance")
    )
    group = _prepare_periodic_group(request)
    if group.finite_count > limits.maximum_periodic_images:
        state.checks.extend(_PERIODIC_IMAGE_CHECKS)
        progress.image_count = group.finite_count
        return
    prepared = _prepare_periodic_cells(request, group)
    if prepared is None:
        state.checks.extend(_PERIODIC_IMAGE_CHECKS)
        return
    facets = _prepare_periodic_facets(request, prepared, group)
    _certify_periodic_traces(request, prepared, group, facets)
    request.budget.reserve(facets.entity_ids.size, 256 + 16 * facets.entity_ids.size)
    occurrence_orbits = np.asarray(
        [int(facets.orbits[facets.facet_rows[int(entity)]]) for entity in facets.entity_ids],
        dtype=np.int64,
    )
    evidence = periodic_restriction_owner.certify_periodic_restriction_partition(
        state,
        request.mesh,
        request.geometry,
        request.validity,
        restriction,
        progress.facet_trace_certificate_id,
        occurrence_orbits,
        prepared.source_bank,
        budget=request.budget,
    )
    if evidence is not None:
        state.periodic_restriction = evidence
        return
    if not state.findings:
        mapped = certify_periodic_mapped_restriction(
            restriction.mapped_predecessor,
            restriction.mapped_transition,
            request.validity,
            restriction.source_regions,
            restriction.target_regions,
            restriction.source_domain,
            restriction.region_namespace_id,
            progress.facet_trace_certificate_id,
            budget=request.budget,
        )
        if mapped is not None:
            state.periodic_mapped_restriction = mapped
            state.checks.extend(
                (
                    "periodic_mapped_restriction_source_embedding",
                    "periodic_mapped_restriction_parent_partition",
                    "periodic_mapped_restriction_common_subcomplex",
                    "periodic_mapped_restriction_derived_region_assignment",
                )
            )
            return
    state.checks.extend(_PERIODIC_IMAGE_CHECKS)
    sources = _prepare_periodic_source_faces(request, prepared, facets)
    _prepare_periodic_affine_quotient(request, prepared, facets)
    images = _prepare_periodic_images(request, prepared, group)
    if images is not None:
        _certify_periodic_contacts(request, prepared, group, facets, sources, images)


_PERIODIC_IMAGE_CHECKS = (
    "periodic_full_source_hulls",
    "periodic_image_sufficiency",
    "periodic_mapped_self_image_embedding",
)


@checked
def certify_periodic_mapped_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    validity: CellValidityCertificate | None = None,
    restriction: PeriodicRestrictionPremise | None = None,
) -> int:
    """Certify source-map contacts in the quotient and return the image count.

    An authenticated ``restriction`` premise first attempts the exact parent
    partition theorem; outside its admission the bounded image-pair proof runs.
    """
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError(
            "certify_periodic_mapped_embedding.mesh.periodic_topology must declare quotient topology; got None"
        )
    if validity is not None:
        validity.require_bound(geometry, mesh=mesh)
    progress = _PeriodicProgress()
    request = _PeriodicRequest(
        state, mesh, geometry, topology, cell_ids, limits, budget, progress, validity
    )
    try:
        with (
            budget.measurement() as phase_measurement,
            budget.activate(),
            budget.bound_stage(limits.maximum_work_units, limits.maximum_scratch_bytes),
        ):
            degree = periodic_affine_degree_owner.certify_affine_torus_degree(
                state, mesh, geometry, cell_ids, budget=budget
            )
            if degree is not False:
                if isinstance(degree, embedding_state_owner.PeriodicAffineTorusDegreeEvidence):
                    state.periodic_affine_degree = degree
                # The degree proof quantifies all quotient images without
                # enumerating any; do not invent a visited-image count.
                return progress.image_count
            if restriction is not None:
                _certify_periodic_restriction(request, restriction)
                return progress.image_count
            state.checks.extend(
                (
                    "periodic_authored_group_identity",
                    "periodic_full_source_hulls",
                    "periodic_image_sufficiency",
                    "periodic_mapped_trace_equivariance",
                    "periodic_mapped_self_image_embedding",
                )
            )
            group = _prepare_periodic_group(request)
            if group.finite_count > limits.maximum_periodic_images:
                progress.image_count = group.finite_count
                return progress.image_count
            prepared = _prepare_periodic_cells(request, group)
            if prepared is None:
                return progress.image_count
            facets = _prepare_periodic_facets(request, prepared, group)
            sources = _prepare_periodic_source_faces(request, prepared, facets)
            _certify_periodic_traces(request, prepared, group, facets)
            _prepare_periodic_affine_quotient(request, prepared, facets)
            images = _prepare_periodic_images(request, prepared, group)
            if images is None:
                return progress.image_count
            _certify_periodic_contacts(request, prepared, group, facets, sources, images)
    except PeriodicIsometryIdentityError:
        state.add("periodic_authored_group_identity", "unresolved", "mesh")
    except RationalEnclosureError:
        state.add("periodic_rational_enclosure_premise", "unresolved", "mesh")
    except CoordinateEnclosureResourceError as error:
        state.add(
            "periodic_source_expression_resource_budget",
            "unresolved",
            progress.resource_kind,
            progress.resource_ids,
            resource_error=error,
            expression_budget=budget,
        )
    finally:
        state.source_expression_work_units = phase_measurement.work_units
        state.source_expression_peak_bytes = phase_measurement.peak_bytes_upper
    return progress.image_count
