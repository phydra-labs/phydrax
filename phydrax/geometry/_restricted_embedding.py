#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify embeddings of declared restrictions of injective source maps.

Parent identity includes source coefficients, routes, and reference charts.
Cyclic PLC triangle frames normalize coefficients and child reference maps
together; pyramid children retain the physical parent apex.

Root-union findings retain their prefixed check names. Evaluated prefixed labels
outside the closed vocabulary describe delegated proof dependencies.

Mesh certificate checks
-----------------------
[
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "mapped_restriction_source",
      "mapped_restriction_reference_domain",
      "mapped_restriction_embedding"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each child declares parent identity, source coefficients, vertex identities, and reference chart."
      },
      {
        "tag": "X",
        "statement": "Parent injectivity and root-union embedding are independently checked."
      }
    ],
    "conclusion": "Admission requires each child reference image to lie in its parent and every child contact to agree with declared incidence.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "restriction_source_corner_identity",
      "restriction_source_enclosure"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A source row supplies the reference cell corner count and a supported rational coordinate law."
      }
    ],
    "conclusion": "Missing corner identities or unsupported source expressions leave the child embedding unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "restriction_source_identity_mismatch",
      "restriction_source_corner_trace_mismatch",
      "restriction_root_trace_mismatch"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Repeated parent declarations have identical element identity, corner identity, and source coefficients after permitted cyclic frame normalization."
      },
      {
        "tag": "X",
        "statement": "Shared root traces and shared source corners have equal coordinate images."
      }
    ],
    "conclusion": "Conflicting parent declarations or identified traces violate source consistency.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "restriction_reference_domain"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Affine child corners satisfy every parent reference inequality, or the composed chart satisfies those inequalities throughout its domain."
      }
    ],
    "conclusion": "An unproved parent-domain inclusion leaves child embedding unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "restriction_bernstein_node_budget",
      "restriction_candidate_pair_budget",
      "restriction_root_candidate_budget",
      "restriction_root_corner_range"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Control-net, child-pair, and root-pair counts fit their limits, and root corner views are finite binary64 values."
      }
    ],
    "conclusion": "Exceeded capacity or an unrepresentable root corner view leaves embedding unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "restriction_reference_overlap"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A pair of child reference images overlaps outside their declared vertex, edge, or face incidence."
      },
      {
        "tag": "X",
        "statement": "Their common parent map is injective."
      }
    ],
    "conclusion": "Nonincident child reference overlap violates the retained-child embedding.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "violated"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
    "checks": [
      "restriction_root_embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Independent parent roots must form an embedded union before root injectivity can transfer all child contacts."
      }
    ],
    "conclusion": "A root overlap leaves the child proof unresolved because the overlap can lie outside the retained children.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding.certify_restricted_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding._straight_multi_affine_root_embedding",
    "checks": [
      "mapped_straight_boundary_degree"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Root maps are locally injective multi-affine boxes with matching facet traces and straight planar boundary images."
      },
      {
        "tag": "X",
        "statement": "The shared mapped-embedding degree proof certifies the boundary carrier."
      }
    ],
    "conclusion": "The root union delegates its boundary-degree obligation to the global mapped-embedding certifier.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding._straight_multi_affine_root_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "owner": "phydrax.geometry._restricted_embedding._affine_tetrahedral_root_embedding",
    "checks": [
      "restriction_affine_root:boundary_contact",
      "restriction_affine_root:boundary_contact_capacity",
      "restriction_affine_root:boundary_contact_predicates",
      "restriction_affine_root:curved_boundary_facet",
      "restriction_affine_root:degenerate_shell",
      "restriction_affine_root:exterior_degree_capacity",
      "restriction_affine_root:exterior_degree_predicates",
      "restriction_affine_root:exterior_degree_premise",
      "restriction_affine_root:inconsistent_facet_orientation",
      "restriction_affine_root:nested_component",
      "restriction_affine_root:nonmanifold_boundary_ridge",
      "restriction_affine_root:nonmanifold_facet",
      "restriction_affine_root:undecided_boundary_facet"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The shared global certifier checks root boundary incidence, orientation, contact classification, and component enclosure."
      },
      {
        "tag": "X",
        "statement": "Root boundary contact and exterior-degree work stay inside their request limits."
      },
      {
        "tag": "X",
        "statement": "A refused root-union check does not identify an overlap inside the retained children."
      }
    ],
    "conclusion": "Every listed root-union failure is transferred as unresolved source-cell evidence while retaining its underlying diagnostic check.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding._affine_tetrahedral_root_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding._straight_multi_affine_root_embedding",
    "checks": [
      "restriction_multi_affine_root:boundary_contact",
      "restriction_multi_affine_root:boundary_contact_capacity",
      "restriction_multi_affine_root:boundary_contact_predicates",
      "restriction_multi_affine_root:curved_boundary_facet",
      "restriction_multi_affine_root:degenerate_shell",
      "restriction_multi_affine_root:exterior_degree_capacity",
      "restriction_multi_affine_root:exterior_degree_predicates",
      "restriction_multi_affine_root:exterior_degree_premise",
      "restriction_multi_affine_root:inconsistent_facet_orientation",
      "restriction_multi_affine_root:nested_component",
      "restriction_multi_affine_root:nonmanifold_boundary_ridge",
      "restriction_multi_affine_root:nonmanifold_facet",
      "restriction_multi_affine_root:undecided_boundary_facet"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The shared global certifier checks root boundary incidence, orientation, contact classification, and component enclosure."
      },
      {
        "tag": "X",
        "statement": "Root boundary contact and exterior-degree work stay inside their request limits."
      },
      {
        "tag": "X",
        "statement": "A refused root-union check does not identify an overlap inside the retained children."
      }
    ],
    "conclusion": "Every listed root-union failure is transferred as unresolved source-cell evidence while retaining its underlying diagnostic check.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._restricted_embedding._straight_multi_affine_root_embedding",
      "locator": "admission and failure conditions",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "owner": "phydrax.geometry._restricted_embedding._reuse_restriction_root_embedding",
    "checks": ["mapped_restriction_root_embedding_reuse"],
    "premises": [
      {"tag": "X", "statement": "The predecessor certificate is regenerated from its bound source mesh, complete geometry, positive validity, and unchanged certificate limits."},
      {"tag": "X", "statement": "Each declared root matches its certified source cell by global ID, ordered corner IDs, complete coefficient action, and exact full source expressions."},
      {"tag": "N", "statement": "Child reference-domain inclusion and every within-parent child contact are independently checked on this request; authentication and comparison work is admitted before execution."}
    ],
    "conclusion": "The certified embedded source union supplies the identical root-union premise without repeating root contact subdivision. Missing or unequal source authority retains the full root-union proof.",
    "citation": {"kind": "definition", "reference": "Restriction of an embedding to a subset", "locator": "authenticated root-union implication", "url": ""},
    "failure_statuses": []
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import cast

import numpy as np

import phydrax.discretization._cell_geometry_predecessor as predecessor_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner
import phydrax.discretization._exact_plc_geometry as exact_plc_geometry_owner
import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._global_embedding as global_embedding_owner
import phydrax.geometry._mapped_coverage as mapped_coverage_owner
from phydrax._strict import StrictModule
from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._embedding_cells import bounds, Cell, domain, vertices
from phydrax.geometry._embedding_state import EmbeddingState, MeshCertificateLimits
from phydrax.typing import checked

from ..discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
)
from ..discretization._cell_geometry_predecessor import CellGeometryPredecessor
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import (
    affine_arguments,
    constant,
    coordinate_expressions,
    coordinate_reference_chain,
    coordinate_source_signature,
    CoordinateSourceBank,
    evaluate,
    ExactScalar,
    Expression,
    expression_add,
    expression_node_count,
    expression_reference_evaluate,
    expression_scale,
    expression_sequence_equal,
    expression_sum,
    multi_affine_coordinates,
    prepared_coordinate_source_bank,
    RationalPolynomial,
    restrict_chart_expressions,
)
from ..discretization._exact_algebraic_geometry import (
    coefficient_endpoint,
    RadicalElement,
    rational_coefficient,
    reserve_algebraic_work,
)
from ..discretization._reference_cell import reference_cell_topology
from ._embedding_cells import inside_reference, reference_constraints
from ._mapped_embedding import (
    injective,
    pair_decision,
    straight_boundary_images,
    tensor_component_injectivity,
    trace_continuity,
)
from ._mesh_certificates import MeshCertificateBinding
from ._rational import rational_vector


def _root_vertex_image(
    cell: Cell, identifier: int, *, budget: CoordinateEnclosureBudget
) -> tuple[ExactScalar, ...]:
    topology = reference_cell_topology(cell.kind)
    point = rational_vector(topology.vertices[cell.vertices.index(identifier)])
    if cell.kind == "pyramid" and point[2] == 1:
        point = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
    return tuple(
        expression_reference_evaluate(value, point, cell.domain, budget=budget)
        for value in cell.coordinates
    )


def _root_trace_restriction(
    cell: Cell,
    identifiers: tuple[int, ...],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...] | None:
    topology = reference_cell_topology(cell.kind)
    corners = np.asarray(topology.vertices, dtype=np.float64)[
        [cell.vertices.index(identifier) for identifier in identifiers]
    ]
    matrix = (
        (corners[[1]] - corners[0]).T
        if dimension == 1
        else (corners[[1, 2 if len(identifiers) == 3 else 3]] - corners[0]).T
    )
    target_kind = (
        "interval"
        if dimension == 1
        else "triangle"
        if len(identifiers) == 3
        else "quadrilateral"
    )
    return restrict_chart_expressions(
        cell.coordinates, cell.kind, target_kind, corners[0], matrix, budget=budget
    )


def _root_traces_equal(
    first: Cell,
    second: Cell,
    *,
    prepared_points: tuple[_Triangle2, _Triangle2] | None = None,
    budget: CoordinateEnclosureBudget,
) -> bool:
    shared = set(first.vertices).intersection(second.vertices)
    if not shared:
        return True
    if prepared_points is not None:
        first_points, second_points = prepared_points
        return all(
            first_points[first.vertices.index(identifier)]
            == second_points[second.vertices.index(identifier)]
            for identifier in shared
        )
    topologies = reference_cell_topology(first.kind), reference_cell_topology(second.kind)
    for identifier in shared:
        if _root_vertex_image(first, identifier, budget=budget) != _root_vertex_image(
            second, identifier, budget=budget
        ):
            return False
    for dimension in range(1, first.dimension):
        for entity in topologies[0].entities[dimension]:
            identifiers = tuple(first.vertices[index] for index in entity)
            if not set(identifiers) <= shared:
                continue
            matches = [
                candidate
                for candidate in topologies[1].entities[dimension]
                if {second.vertices[index] for index in candidate} == set(identifiers)
            ]
            if len(matches) != 1:
                return False
            restrictions = tuple(
                _root_trace_restriction(cell, identifiers, dimension, budget=budget)
                for cell in (first, second)
            )
            if (
                any(value is None for value in restrictions)
                or restrictions[0] != restrictions[1]
            ):
                return False
    return True


type _Point2 = tuple[ExactScalar, ExactScalar]
type _Triangle2 = tuple[_Point2, _Point2, _Point2]


def _orient2(first: _Point2, second: _Point2, third: _Point2, /) -> ExactScalar:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _between(point: _Point2, first: _Point2, second: _Point2, /) -> bool:
    return all(
        min(a, b) <= value <= max(a, b)
        for value, a, b in zip(point, first, second, strict=True)
    )


def _affine_triangle_chart_points(
    cell: Cell, /, *, budget: CoordinateEnclosureBudget
) -> _Triangle2 | None:
    if (
        cell.kind != "triangle"
        or cell.dimension != 2
        or len(cell.coordinates) != 2
        or any(
            isinstance(value, RationalPolynomial)
            or any(sum(index) > 1 for index in value)
            for value in cell.coordinates
        )
    ):
        return None
    return cast(
        _Triangle2,
        tuple(
            tuple(
                expression_reference_evaluate(
                    value, reference, cell.domain, budget=budget
                )
                for value in cell.coordinates
            )
            for reference in cell.reference_vertices
        ),
    )


class _TriangleContact(StrictModule):
    cells: tuple[Cell, Cell]
    points: tuple[_Triangle2, _Triangle2]
    orientations: tuple[ExactScalar, ...]
    edge_signs: tuple[tuple[tuple[ExactScalar, ...], ...], ...]


def _triangle_edge_overlap(
    contact: _TriangleContact, first_edge: int, second_edge: int
) -> bool:
    points, cells, edge_signs = contact.points, contact.cells, contact.edge_signs
    edge_slots = ((0, 1), (1, 2), (2, 0))
    first_slots, second_slots = edge_slots[first_edge], edge_slots[second_edge]
    a, b = (points[0][slot] for slot in first_slots)
    first_ids = {cells[0].vertices[slot] for slot in first_slots}
    c, d = (points[1][slot] for slot in second_slots)
    second_ids = {cells[1].vertices[slot] for slot in second_slots}
    ab_c, ab_d = (edge_signs[0][first_edge][slot] for slot in second_slots)
    cd_a, cd_b = (edge_signs[1][second_edge][slot] for slot in first_slots)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return True
    if ab_c == ab_d == cd_a == cd_b == 0:
        axis = int(abs(b[1] - a[1]) > abs(b[0] - a[0]))
        lower = max(min(a[axis], b[axis]), min(c[axis], d[axis]))
        upper = min(max(a[axis], b[axis]), max(c[axis], d[axis]))
        if lower < upper and first_ids != second_ids:
            return True
    for point, sign, point_id, start, end in (
        (c, ab_c, cells[1].vertices[second_slots[0]], a, b),
        (d, ab_d, cells[1].vertices[second_slots[1]], a, b),
        (a, cd_a, cells[0].vertices[first_slots[0]], c, d),
        (b, cd_b, cells[0].vertices[first_slots[1]], c, d),
    ):
        if sign != 0 or not _between(point, start, end):
            continue
        if point_id not in (first_ids & second_ids) or point not in (start, end):
            return True
    return False


def _triangle_vertex_overlap(contact: _TriangleContact) -> bool:
    points, cells, edge_signs, triangle_orientations = (
        contact.points,
        contact.cells,
        contact.edge_signs,
        contact.orientations,
    )
    for owner, other in ((0, 1), (1, 0)):
        orientation = triangle_orientations[other]
        for point_slot, (identifier, point) in enumerate(
            zip(cells[owner].vertices, points[owner], strict=True)
        ):
            signs = tuple(edge_signs[other][edge][point_slot] for edge in range(3))
            inside = (
                all(value > 0 for value in signs)
                if orientation > 0
                else all(value < 0 for value in signs)
            )
            if inside:
                return True
            enclosed = (
                all(value >= 0 for value in signs)
                if orientation > 0
                else all(value <= 0 for value in signs)
            )
            if enclosed and identifier not in cells[other].vertices:
                return True
    return False


def _affine_triangle_chart_decision(
    first: Cell,
    second: Cell,
    /,
    *,
    prepared_points: tuple[_Triangle2, _Triangle2] | None = None,
    budget: CoordinateEnclosureBudget,
) -> str | None:
    """Classify intersections of affine triangle maps using rational edge signs."""
    if prepared_points is None:
        first_points = _affine_triangle_chart_points(first, budget=budget)
        second_points = _affine_triangle_chart_points(second, budget=budget)
        if first_points is None or second_points is None:
            return None
        points = first_points, second_points
    else:
        points = prepared_points

    if any(
        isinstance(value, RadicalElement)
        for triangle in points
        for point in triangle
        for value in point
    ):
        reserve_algebraic_work((*points[0], *points[1]), 220, 220, 2, budget=budget)
    budget.reserve(220, 16384)
    triangle_orientations = tuple(_orient2(*triangle) for triangle in points)
    if any(value == 0 for value in triangle_orientations):
        return None
    edge_slots = ((0, 1), (1, 2), (2, 0))
    edge_signs = tuple(
        tuple(
            tuple(
                _orient2(
                    points[owner][edge[0]],
                    points[owner][edge[1]],
                    point,
                )
                for point in points[1 - owner]
            )
            for edge in edge_slots
        )
        for owner in range(2)
    )
    contact = _TriangleContact((first, second), points, triangle_orientations, edge_signs)
    for first_edge in range(3):
        for second_edge in range(3):
            if _triangle_edge_overlap(contact, first_edge, second_edge):
                return "overlap"
    return "overlap" if _triangle_vertex_overlap(contact) else "separated"


def _cyclic_triangle_reference_coordinates(
    coordinates: tuple[Expression, ...],
    rotation: int,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Expression, ...]:
    """Express a cyclic P1 triangle chart in its parent's first declared frame."""
    barycentric = (
        expression_add(
            constant(1, dimension, budget=budget),
            expression_scale(
                expression_sum(coordinates, budget=budget), -1, budget=budget
            ),
            budget=budget,
        ),
        *coordinates,
    )
    return barycentric[(1 - rotation) % 3], barycentric[(2 - rotation) % 3]


class _RestrictedRequest(StrictModule):
    state: EmbeddingState
    mesh: CellMesh
    geometry: CellGeometrySpec
    cell_ids: np.ndarray
    limits: MeshCertificateLimits
    budget: CoordinateEnclosureBudget
    predecessor: CellGeometryPredecessor | None = None


class _RootSource(StrictModule):
    cell: Cell
    local: CoordinateSourceBank
    element_id: str
    cyclic: bool


class _RestrictionCells(StrictModule):
    parents: dict[int, _RootSource]
    children: dict[int, list[tuple[int, Cell]]]


class _ParentBinding(StrictModule):
    rotation: int
    valid: bool


def _bind_restriction_parent(
    request: _RestrictedRequest,
    parents: dict[int, _RootSource],
    source: reference_cell_types_owner.CellGeometryElement,
    parent_id: int,
    parent_vertices: tuple[int, ...],
    local: CoordinateSourceBank,
    cyclic_triangle: bool,
    cursor: int,
) -> _ParentBinding:
    state, limits, budget = request.state, request.limits, request.budget
    cell_ids = request.cell_ids
    ledger = budget
    source_kind = source.cell_kind
    source_dimension = reference_cell_topology(source_kind).dimension
    rotation = 0
    if parent_id in parents:
        previous_record = parents[parent_id]
        previous, old_local, element_id, old_cyclic = (
            previous_record.cell,
            previous_record.local,
            previous_record.element_id,
            previous_record.cyclic,
        )
        # PLC P1 cardinal coefficients admit this orientation-preserving action.
        # Parent identities and coefficients are checked after reordering.
        if cyclic_triangle and old_cyclic and previous.vertices != parent_vertices:
            for shift in (1, 2):
                if (
                    parent_vertices
                    == previous.vertices[shift:] + previous.vertices[:shift]
                ):
                    rotation = shift
                    parent_vertices = previous.vertices
                    local = tuple(local[(index - shift) % 3] for index in range(3))
                    break
        if (
            source.element_id != element_id
            or previous.vertices != parent_vertices
            or old_local != local
        ):
            state.add(
                "restriction_source_identity_mismatch",
                "violated",
                "cell",
                cell_ids[cursor : cursor + 1],
            )
            return _ParentBinding(rotation, False)
    else:
        # Each root is proved once; its single-cell workspace is released
        # and only the source records used by the root cell are retained.
        with ledger.temporary_scope():
            prepared = multi_affine_coordinates(source, local, budget=budget)
            polynomials: tuple[Expression, ...] | None
            if prepared is None:
                polynomials, corner_images = (
                    coordinate_expressions(source, local, budget=budget),
                    None,
                )
            else:
                polynomials, corner_images = prepared
            if polynomials is None:
                state.add(
                    "restriction_source_enclosure",
                    "unresolved",
                    "cell",
                    cell_ids[cursor : cursor + 1],
                )
                return _ParentBinding(rotation, False)
            root = Cell(
                polynomials,
                domain(source_kind),
                source_kind,
                source_dimension,
                parent_vertices,
                vertices(source_kind),
                corner_images,
            )
            reason = injective(root, limits, state, budget=budget)
        ledger.retain_basis(root.coordinates)
        ledger.retain_basis(root.vertices)
        ledger.retain_basis(local)
        if root.corner_images is not None:
            ledger.retain_basis(root.corner_images)
        parents[parent_id] = _RootSource(root, local, source.element_id, cyclic_triangle)
        if reason is not None:
            state.add(reason, "unresolved", "cell", cell_ids[cursor : cursor + 1])
    return _ParentBinding(rotation, True)


def _prove_restriction_constraints(
    request: _RestrictedRequest,
    source_kind: str,
    source_dimension: int,
    target_kind: str,
    chart_coordinates: tuple[Expression, ...],
    cursor: int,
) -> bool:
    state, limits, budget = request.state, request.limits, request.budget
    cell_ids = request.cell_ids
    constraints = reference_constraints(
        chart_coordinates,
        source_kind,
        source_dimension,
        budget=budget,
    )
    for constraint in constraints:
        if (
            expression_node_count(constraint, domain(target_kind), source_dimension)
            > limits.maximum_bernstein_nodes
        ):
            state.add(
                "restriction_bernstein_node_budget",
                "unresolved",
                "cell",
                cell_ids[cursor : cursor + 1],
            )
            return False
        work = mapped_coverage_owner.SubdivisionLedger()
        contained = mapped_coverage_owner.nonnegative(
            constraint,
            domain(target_kind),
            source_dimension,
            limits.maximum_bernstein_nodes,
            limits.maximum_subdivision_depth,
            limits.maximum_subdivision_pieces - state.subdivision_pieces,
            work,
            budget=budget,
        )
        state.subdivision_pieces += work.pieces
        if not contained:
            state.add(
                "restriction_reference_domain",
                "unresolved",
                "cell",
                cell_ids[cursor : cursor + 1],
            )
            return False
    return True


def _prove_affine_restriction_domain(
    request: _RestrictedRequest,
    source_kind: str,
    target_kind: str,
    offset: np.ndarray,
    matrix: np.ndarray,
    cursor: int,
) -> bool:
    state, budget, cell_ids = request.state, request.budget, request.cell_ids
    arguments = affine_arguments(offset, matrix, budget=budget)
    for vertex in reference_cell_topology(target_kind).vertices:
        point = rational_vector(vertex)
        image = tuple(
            rational_coefficient(
                evaluate(value, point, budget=budget),
                role="restricted_embedding.reference_image",
            )
            for value in arguments
        )
        if not inside_reference(image, source_kind):
            state.add(
                "restriction_reference_domain",
                "unresolved",
                "cell",
                cell_ids[cursor : cursor + 1],
            )
            return False
    return True


def _prove_restriction_domain(
    request: _RestrictedRequest,
    element: reference_cell_types_owner.CellGeometryElement,
    source: reference_cell_types_owner.CellGeometryElement,
    block: reference_cell_types_owner.CellBlock
    | reference_cell_types_owner.PolyhedralBlock,
    chart_coordinates: tuple[Expression, ...],
    cursor: int,
) -> bool:
    source_kind = source.cell_kind
    source_dimension = reference_cell_topology(source_kind).dimension
    with request.budget.temporary_scope():
        match element:
            case (
                PolynomialComposedCellGeometryElement()
                | RationalComposedCellGeometryElement()
            ):
                return _prove_restriction_constraints(
                    request,
                    source_kind,
                    source_dimension,
                    block.cell_kind,
                    chart_coordinates,
                    cursor,
                )
            case RestrictedCellGeometryElement() if (
                element.source_element.element_id != source.element_id
            ):
                return _prove_restriction_constraints(
                    request,
                    source_kind,
                    source_dimension,
                    block.cell_kind,
                    chart_coordinates,
                    cursor,
                )
            case RestrictedCellGeometryElement():
                matrix = np.asarray(element.matrix, dtype=np.float64)
                offset = np.asarray(element.offset, dtype=np.float64)
            case _:
                matrix = np.eye(source_dimension, dtype=np.float64)
                offset = np.zeros((source_dimension,), dtype=np.float64)
        return _prove_affine_restriction_domain(
            request, source_kind, block.cell_kind, offset, matrix, cursor
        )


class _ReferenceFrame(StrictModule):
    coordinates: tuple[Expression, ...]
    corners: tuple[tuple[Fraction, ...], ...] | None


class _RestrictionBlock(StrictModule):
    source: reference_cell_types_owner.CellGeometryElement
    reference_coordinates: tuple[Expression, ...]
    chart_corners: tuple[tuple[Fraction, ...], ...] | None
    cyclic: bool
    parent_ids: np.ndarray
    vertex_ids: np.ndarray
    frames: dict[int, _ReferenceFrame]
    domain_proved: set[int]


def _prepare_restriction_block(
    request: _RestrictedRequest,
    record: CellGeometryRestrictionSource,
    block: reference_cell_types_owner.CellBlock
    | reference_cell_types_owner.PolyhedralBlock,
    element: reference_cell_types_owner.CellGeometryElement,
    cyclic_source_signature: dict[str, object] | None,
) -> _RestrictionBlock:
    budget = request.budget
    ledger = budget
    source, reference_coordinates = coordinate_reference_chain(element, budget=budget)
    ledger.retain_basis(reference_coordinates)
    cyclic_triangle = (
        cyclic_source_signature is not None
        and coordinate_source_signature(source) == cyclic_source_signature
    )
    parent_ids = np.asarray(record.block_parent_cell_ids[block.name], dtype=np.int64)
    vertex_ids = np.asarray(record.block_parent_vertex_ids[block.name], dtype=np.int64)
    # A multi-affine box chart is the interpolant of its rational reference
    # corner images, shared by every row of the block.
    chart_corners = None
    if block.cell_kind in ("quadrilateral", "hexahedron") and all(
        isinstance(value, dict)
        and all(exponent <= 1 for index in value for exponent in index)
        for value in reference_coordinates
    ):
        chart_corners = tuple(
            tuple(
                rational_coefficient(
                    expression_reference_evaluate(
                        value, point, domain(block.cell_kind), budget=budget
                    ),
                    role="restricted_embedding.reference_corner",
                )
                for value in reference_coordinates
            )
            for point in vertices(block.cell_kind)
        )
        ledger.retain_basis(chart_corners)
    frames = {0: _ReferenceFrame(reference_coordinates, chart_corners)}
    return _RestrictionBlock(
        source,
        reference_coordinates,
        chart_corners,
        cyclic_triangle,
        parent_ids,
        vertex_ids,
        frames,
        set(),
    )


def _restriction_frame(
    request: _RestrictedRequest,
    prepared: _RestrictionBlock,
    block: reference_cell_types_owner.CellBlock
    | reference_cell_types_owner.PolyhedralBlock,
    rotation: int,
) -> _ReferenceFrame:
    ledger, budget, mesh = request.budget, request.budget, request.mesh
    reference_coordinates, chart_corners, reference_frames = (
        prepared.reference_coordinates,
        prepared.chart_corners,
        prepared.frames,
    )
    if rotation not in reference_frames:
        with ledger.temporary_scope():
            chart_coordinates = _cyclic_triangle_reference_coordinates(
                reference_coordinates,
                rotation,
                mesh.topological_dimension,
                budget=budget,
            )
            frame_corners = (
                None
                if chart_corners is None
                else tuple(
                    tuple(
                        rational_coefficient(
                            expression_reference_evaluate(
                                value, point, domain(block.cell_kind), budget=budget
                            ),
                            role="restricted_embedding.rotated_reference_corner",
                        )
                        for value in chart_coordinates
                    )
                    for point in vertices(block.cell_kind)
                )
            )
        ledger.retain_basis(chart_coordinates)
        if frame_corners is not None:
            ledger.retain_basis(frame_corners)
        reference_frames[rotation] = _ReferenceFrame(chart_coordinates, frame_corners)
    return reference_frames[rotation]


def _prepare_restricted_children(
    request: _RestrictedRequest, record: CellGeometryRestrictionSource
) -> _RestrictionCells | None:
    state, mesh, geometry, cell_ids, budget = (
        request.state,
        request.mesh,
        request.geometry,
        request.cell_ids,
        request.budget,
    )
    ledger = budget
    elements, routes, _ = geometry.resolve(mesh)
    values = geometry.source_coordinates(budget=budget)
    match geometry.exact_source:
        case exact_plc_geometry_owner.ExactPlcCellGeometrySource():
            cyclic_source_signature = coordinate_source_signature(
                coordinate_lagrange_element("triangle", 1)
            )
        case _:
            cyclic_source_signature = None
    parents: dict[int, _RootSource] = {}
    children: dict[int, list[tuple[int, Cell]]] = {}
    cursor = 0
    state.checks.extend(
        (
            "mapped_restriction_source",
            "mapped_restriction_reference_domain",
            "mapped_restriction_embedding",
        )
    )
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        prepared_block = _prepare_restriction_block(
            request, record, block, element, cyclic_source_signature
        )
        source = prepared_block.source
        source_kind = source.cell_kind
        parent_ids, vertex_ids, cyclic_triangle, domain_proved = (
            prepared_block.parent_ids,
            prepared_block.vertex_ids,
            prepared_block.cyclic,
            prepared_block.domain_proved,
        )
        for row, target_vertices, parent_id, source_vertices in zip(
            np.asarray(route),
            np.asarray(block.vertices),
            parent_ids.tolist(),
            vertex_ids,
            strict=True,
        ):
            source_vertices = np.asarray(source_vertices, dtype=np.int64)
            local = tuple(values[index] for index in row)
            parent_vertices = tuple(source_vertices[source_vertices >= 0].tolist())
            if len(parent_vertices) != len(reference_cell_topology(source_kind).vertices):
                state.add(
                    "restriction_source_corner_identity",
                    "unresolved",
                    "cell",
                    cell_ids[cursor : cursor + 1],
                )
                return None
            binding = _bind_restriction_parent(
                request,
                parents,
                source,
                parent_id,
                parent_vertices,
                local,
                cyclic_triangle,
                cursor,
            )
            if not binding.valid:
                return None
            rotation = binding.rotation
            frame = _restriction_frame(request, prepared_block, block, rotation)
            chart_coordinates, frame_corners = frame.coordinates, frame.corners
            if rotation not in domain_proved:
                if not _prove_restriction_domain(
                    request, element, source, block, chart_coordinates, cursor
                ):
                    return None
                domain_proved.add(rotation)
            chart = Cell(
                chart_coordinates,
                domain(block.cell_kind),
                block.cell_kind,
                mesh.topological_dimension,
                tuple(target_vertices.tolist()),
                vertices(block.cell_kind),
                frame_corners,
            )
            ledger.retain_basis(chart.vertices)
            children.setdefault(parent_id, []).append((cursor, chart))
            cursor += 1
    return _RestrictionCells(parents, children)


class _RestrictionChartCache(StrictModule):
    bounds: dict[int, tuple[np.ndarray, np.ndarray]]
    points: dict[int, _Triangle2]
    separated: set[tuple[int, int, str, str, tuple[tuple[int, int], ...]]]


def _prepare_restriction_chart_cache(
    request: _RestrictedRequest,
    members: list[tuple[int, Cell]],
    cache: _RestrictionChartCache,
) -> None:
    budget, ledger = request.budget, request.budget
    chart_bounds, chart_points = cache.bounds, cache.points
    for _, chart in members:
        chart_key = id(chart.coordinates)
        if chart_key not in chart_bounds:
            with ledger.temporary_scope():
                chart_bounds[chart_key] = bounds(
                    chart,
                    np.zeros((chart.dimension,), dtype=np.float64),
                    np.eye(chart.dimension, dtype=np.float64),
                    budget=budget,
                )
        if chart_key not in chart_points:
            with ledger.temporary_scope():
                prepared_points = _affine_triangle_chart_points(chart, budget=budget)
            if prepared_points is not None:
                ledger.retain_basis(prepared_points)
                chart_points[chart_key] = prepared_points


def _certify_restriction_children(
    request: _RestrictedRequest, prepared: _RestrictionCells
) -> None:
    state, mesh, cell_ids, limits, budget = (
        request.state,
        request.mesh,
        request.cell_ids,
        request.limits,
        request.budget,
    )
    ledger = budget
    children = prepared.children
    # Injective root maps transfer reference contacts to their physical images,
    # including pyramid apex identifications.
    all_charts = [chart for members in children.values() for _, chart in members]
    ordered_charts: list[Cell | None] = [None] * len(all_charts)
    groups = np.full((len(all_charts),), -1, dtype=np.int64)
    for parent, members in children.items():
        for target, chart in members:
            ordered_charts[target] = chart
            groups[target] = parent
    if any(chart is None for chart in ordered_charts):
        raise ValueError(
            "certify_restricted_embedding.geometry.restriction_source must cover target cells; got missing source rows"
        )
    trace_continuity(
        state,
        mesh,
        [chart for chart in ordered_charts if chart is not None],
        groups,
        shared_charts=True,
        budget=budget,
    )
    # A block shares its chart expressions. Bounds and separation with a shared
    # corner-incidence pattern can be reused across its repeated parent maps.
    cache = _RestrictionChartCache({}, {}, set())
    chart_bounds, chart_points, separated = cache.bounds, cache.points, cache.separated
    for members in children.values():
        if len(members) <= 1:
            continue
        _prepare_restriction_chart_cache(request, members, cache)
        boxes = [chart_bounds[id(chart.coordinates)] for _, chart in members]
        lower = np.stack([value[0] for value in boxes])
        upper = np.stack([value[1] for value in boxes])
        a_rows, b_rows, exceeded = embedding_state_owner.candidate_pairs(
            lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
        )
        state.candidate_pairs += a_rows.size
        if exceeded:
            state.add("restriction_candidate_pair_budget", "unresolved", "mesh")
        for a, b in zip(a_rows.tolist(), b_rows.tolist(), strict=True):
            target_a, first = members[a]
            target_b, second = members[b]
            pattern = tuple(
                sorted(
                    (first.vertices.index(vertex), second.vertices.index(vertex))
                    for vertex in set(first.vertices).intersection(second.vertices)
                )
            )
            key = (
                id(first.coordinates),
                id(second.coordinates),
                first.kind,
                second.kind,
                pattern,
            )
            if key in separated:
                continue
            with ledger.temporary_scope():
                first_points = chart_points.get(id(first.coordinates))
                second_points = chart_points.get(id(second.coordinates))
                decision = _affine_triangle_chart_decision(
                    first,
                    second,
                    prepared_points=(
                        (first_points, second_points)
                        if first_points is not None and second_points is not None
                        else None
                    ),
                    budget=budget,
                )
                if decision is None:
                    contact_decision_1 = pair_decision(
                        first,
                        second,
                        limits,
                        limits.maximum_subdivision_pieces - state.subdivision_pieces,
                        checks=state.checks,
                        budget=budget,
                    )
                    decision = (
                        contact_decision_1.unresolved_check()
                        if contact_decision_1.status == "unresolved"
                        else contact_decision_1.status
                    )
                    work = contact_decision_1.work
                else:
                    work = 0
            state.subdivision_pieces += work
            if decision == "separated":
                separated.add(key)
            elif decision == "overlap":
                state.add(
                    "restriction_reference_overlap",
                    "violated",
                    "cell",
                    cell_ids[[target_a, target_b]],
                )
            else:
                state.add(decision, "unresolved", "cell", cell_ids[[target_a, target_b]])


class _RestrictionRootPairs(StrictModule):
    points: dict[int, _Triangle2]
    atlas_labels: np.ndarray
    first_rows: np.ndarray
    second_rows: np.ndarray


def _prepare_restriction_root_pairs(
    request: _RestrictedRequest,
    roots: list[int],
    root_cells: list[Cell],
    source_geometry_id: str,
) -> _RestrictionRootPairs:
    state, limits, budget, ledger = (
        request.state,
        request.limits,
        request.budget,
        request.budget,
    )
    affine_root_points: dict[int, _Triangle2] = {}
    for index, root in enumerate(root_cells):
        with ledger.temporary_scope():
            points = _affine_triangle_chart_points(root, budget=budget)
        if points is not None:
            ledger.retain_basis(points)
            affine_root_points[index] = points
    with ledger.temporary_scope():
        atlas_labels = _root_atlas_labels(
            state, root_cells, roots, source_geometry_id, limits, budget=budget
        )
        root_boxes = [
            bounds(
                root,
                np.zeros((root.dimension,), dtype=np.float64),
                np.eye(root.dimension, dtype=np.float64),
                budget=budget,
            )
            for root in root_cells
        ]
    a_rows, b_rows, exceeded = embedding_state_owner.candidate_pairs(
        np.stack([box[0] for box in root_boxes]),
        np.stack([box[1] for box in root_boxes]),
        limits.maximum_candidate_pairs - state.candidate_pairs,
    )
    state.candidate_pairs += a_rows.size
    if exceeded:
        state.add("restriction_root_candidate_budget", "unresolved", "mesh")
    return _RestrictionRootPairs(affine_root_points, atlas_labels, a_rows, b_rows)


def _authenticated_restriction_predecessor(
    request: _RestrictedRequest,
) -> (
    tuple[CellGeometryPredecessor, global_embedding_owner.GlobalEmbeddingCertificate]
    | None
):
    """Authenticate the request-bound predecessor before importing its root theorem."""
    predecessor = request.predecessor
    if predecessor is None or predecessor.source_embedding is None:
        return None
    source = predecessor.source_embedding
    if (
        source.status != "certified"
        or not predecessor.source_validity.all_certified
        or source.binding.limits_id != request.limits.limits_id
        or source.binding.junction_vertices
        or predecessor.source_mesh.periodic_topology is not None
        or request.mesh.periodic_topology is not None
    ):
        return None
    meshes = predecessor.source_mesh, request.mesh
    geometries = predecessor.source_geometry, request.geometry
    request.budget.reserve(
        sum(mesh.coordinates.size for mesh in meshes)
        + sum(
            block.vertices.size + block.global_ids.size
            for mesh in meshes
            for block in mesh.blocks
        )
        + sum(geometry.coordinates.size for geometry in geometries)
        + sum(route.size for geometry in geometries for route in geometry.geometry_dofs)
        + len(source.evaluated_checks)
        + len(source.findings),
        2048,
    )
    if predecessor.content_id != predecessor_owner._predecessor_content_id(
        predecessor.source_mesh,
        predecessor.source_geometry,
        predecessor.source_validity,
        source,
        request.mesh,
        request.geometry,
        predecessor.source_rows,
    ):
        return None
    binding = MeshCertificateBinding(
        predecessor.source_mesh,
        predecessor.source_geometry,
        source.binding.coordinate_scope,
        request.limits,
        source_id=source.binding.source_id,
        source_revision=source.binding.source_revision,
        junction_vertices=source.binding.junction_vertices,
    )
    if (
        binding.binding_id != source.binding.binding_id
        or binding.geometry_id != predecessor.source_validity.geometry_id
        or source.validity_certificate_id != predecessor.source_validity.certificate_id
    ):
        return None
    rebuilt = global_embedding_owner.GlobalEmbeddingCertificate(
        binding,
        predecessor.source_validity.certificate_id,
        source.findings,
        source.evaluated_checks,
        counts=global_embedding_owner.EmbeddingCounts(
            source.cell_count,
            source.boundary_facet_count,
            source.shell_count,
            source.candidate_pair_count,
            source.ray_test_count,
            source.subdivision_piece_count,
            source.periodic_image_count,
        ),
        boundary_degree=source.boundary_degree,
        periodic_affine=source.periodic_affine,
        restriction_root_embedding=source.restriction_root_embedding,
    )
    if rebuilt.certificate_id != source.certificate_id:
        return None
    source.binding.require(predecessor.source_mesh, predecessor.source_geometry)
    rebuilt = global_embedding_owner.GlobalEmbeddingCertificate.from_stored_fields(
        rebuilt
    )
    predecessor.source_validity.require_bound(
        predecessor.source_geometry, mesh=predecessor.source_mesh
    )
    return predecessor, rebuilt


def _reuse_restriction_root_embedding(
    request: _RestrictedRequest,
    prepared: _RestrictionCells,
    source_geometry_id: str,
) -> bool:
    """Import a certified root union after comparing source maps coefficientwise."""
    authenticated = _authenticated_restriction_predecessor(request)
    if authenticated is None:
        return False
    predecessor, certificate = authenticated
    if certificate.binding.geometry_id != source_geometry_id:
        return False
    mesh, geometry, budget = (
        predecessor.source_mesh,
        predecessor.source_geometry,
        request.budget,
    )
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    bank = prepared_coordinate_source_bank(geometry, budget=budget)
    budget.reserve(
        sum(block.cell_count for block in mesh.blocks),
        128 + 96 * sum(block.cell_count for block in mesh.blocks),
    )
    owners = {
        int(identifier): (block_index, row)
        for block_index, block in enumerate(mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids, dtype=np.int64))
    }
    host_routes = tuple(np.asarray(route, dtype=np.int64) for route in routes)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    corners = tuple(np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks)
    for identifier, root in prepared.parents.items():
        owner = owners.get(identifier)
        if owner is None:
            return False
        block, row = owner
        element = elements[block]
        budget.reserve(len(root.cell.vertices) + sum(len(point) for point in root.local))
        local = tuple(bank[int(index)] for index in host_routes[block][row])
        if (
            root.element_id != element.element_id
            or root.cell.vertices
            != tuple(int(value) for value in vertex_ids[corners[block][row]])
            or root.local != local
        ):
            return False
        coordinates = coordinate_expressions(element, local, budget=budget)
        if coordinates is None or not expression_sequence_equal(
            root.cell.coordinates, coordinates, budget=budget
        ):
            return False
    request.state.imported_root_embedding_certificate_id = certificate.certificate_id
    request.state.checks.append("mapped_restriction_root_embedding_reuse")
    return True


def _certify_restriction_roots(
    request: _RestrictedRequest, prepared: _RestrictionCells, source_geometry_id: str
) -> None:
    state, cell_ids, limits, budget = (
        request.state,
        request.cell_ids,
        request.limits,
        request.budget,
    )
    ledger = budget
    parents, children = prepared.parents, prepared.children
    roots = list(parents)
    root_cells = [parents[parent_id].cell for parent_id in roots]
    if len(root_cells) <= 1:
        return
    if state.clean:
        with ledger.temporary_scope():
            if _reuse_restriction_root_embedding(request, prepared, source_geometry_id):
                return
    if _affine_tetrahedral_root_embedding(
        state, root_cells, roots, source_geometry_id, limits, budget=budget
    ):
        return
    with ledger.temporary_scope():
        straight = _straight_multi_affine_root_embedding(
            state, root_cells, roots, source_geometry_id, limits, budget=budget
        )
    if straight:
        return
    root_pairs = _prepare_restriction_root_pairs(
        request, roots, root_cells, source_geometry_id
    )
    affine_root_points, atlas_labels, a_rows, b_rows = (
        root_pairs.points,
        root_pairs.atlas_labels,
        root_pairs.first_rows,
        root_pairs.second_rows,
    )
    for a, b in zip(a_rows.tolist(), b_rows.tolist(), strict=True):
        if atlas_labels[a] >= 0 and atlas_labels[a] == atlas_labels[b]:
            continue
        with ledger.temporary_scope():
            root_pair_points = (
                (affine_root_points[a], affine_root_points[b])
                if a in affine_root_points and b in affine_root_points
                else None
            )
            traced = _root_traces_equal(
                root_cells[a],
                root_cells[b],
                prepared_points=root_pair_points,
                budget=budget,
            )
            if not traced:
                decision, work = "restriction_root_trace_mismatch", 0
            else:
                decision = (
                    _affine_triangle_chart_decision(
                        root_cells[a],
                        root_cells[b],
                        prepared_points=root_pair_points,
                        budget=budget,
                    )
                    if root_pair_points is not None
                    else None
                )
                if decision is None:
                    contact_decision_2 = pair_decision(
                        root_cells[a],
                        root_cells[b],
                        limits,
                        limits.maximum_subdivision_pieces - state.subdivision_pieces,
                        checks=state.checks,
                        budget=budget,
                    )
                    decision = (
                        contact_decision_2.unresolved_check()
                        if contact_decision_2.status == "unresolved"
                        else contact_decision_2.status
                    )
                    work = contact_decision_2.work
                else:
                    work = 0
        if not traced:
            affected = [
                target for root in (roots[a], roots[b]) for target, _ in children[root]
            ]
            state.add(
                "restriction_root_trace_mismatch", "violated", "cell", cell_ids[affected]
            )
            continue
        state.subdivision_pieces += work
        if decision != "separated":
            affected = [
                target for root in (roots[a], roots[b]) for target, _ in children[root]
            ]
            # A root overlap may lie outside the retained children, so it reports
            # an unresolved child embedding obligation.
            state.add(
                "restriction_root_embedding_premise"
                if decision == "overlap"
                else decision,
                "unresolved",
                "cell",
                cell_ids[affected],
            )


@checked
def certify_restricted_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None = None,
) -> None:
    """Certify child reference contacts and the union of their declared source maps."""
    record = geometry.restriction_source
    if record is None:
        raise ValueError(
            "certify_restricted_embedding.geometry.restriction_source must declare source identities; got None"
        )
    request = _RestrictedRequest(
        state, mesh, geometry, cell_ids, limits, budget, predecessor
    )
    prepared = _prepare_restricted_children(request, record)
    if prepared is None:
        return
    _certify_restriction_children(request, prepared)
    _certify_restriction_roots(request, prepared, record.source_geometry_id)


def _affine_tetrahedral_root_embedding(
    state: embedding_state_owner.EmbeddingState,
    roots: list[Cell],
    identifiers: list[int],
    source_geometry_id: str,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Certify the root union through its affine tetrahedral corner image.

    Affine expressions and binary64 corner images identify the tetrahedra.
    Shared vertex identities determine their affine edge and face traces.
    """

    if any(
        root.kind != "tetrahedron"
        or root.dimension != 3
        or len(root.coordinates) != 3
        or any(
            not isinstance(expression, dict)
            or any(len(index) != 3 or sum(index) > 1 for index in expression)
            for expression in root.coordinates
        )
        for root in roots
    ):
        return False
    positions: dict[int, tuple[ExactScalar, ...]] = {}
    owners: dict[int, int] = {}
    rounded: dict[int, tuple[float, ...]] = {}
    for root_id, root in zip(identifiers, roots, strict=True):
        for vertex_id, corner in zip(root.vertices, root.reference_vertices, strict=True):
            image = tuple(
                expression_reference_evaluate(value, corner, root.domain, budget=budget)
                for value in root.coordinates
            )
            values = tuple(
                coefficient_endpoint(value, upper=True, saturate=True) for value in image
            )
            if any(
                not np.isfinite(value) or Fraction.from_float(value) != exact
                for value, exact in zip(values, image, strict=True)
            ):
                return False
            known = positions.get(vertex_id)
            if known is not None and known != image:
                state.add(
                    "restriction_source_corner_trace_mismatch",
                    "violated",
                    "source_cell",
                    tuple(sorted({owners[vertex_id], root_id})),
                )
                return True
            positions[vertex_id], rounded[vertex_id], owners[vertex_id] = (
                image,
                values,
                root_id,
            )
    vertex_ids = np.asarray(sorted(positions), dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    coordinates = np.asarray(
        [rounded[int(identifier)] for identifier in vertex_ids], dtype=np.float64
    )
    cells = np.asarray(
        [tuple(rows[identifier] for identifier in root.vertices) for root in roots],
        dtype=np.int64,
    )
    corners = coordinates[cells]
    orientation, certain = global_embedding_owner.mesh_orient3d(
        corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3], budget=budget
    )
    if not np.all(certain & (orientation > 0)):
        return False
    view = cell_mesh_owner.CellMesh(
        coordinates,
        (
            reference_cell_types_owner.CellBlock(
                "declared_affine_source_roots",
                "tetrahedron",
                cells,
                global_ids=np.asarray(identifiers, dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_geometry_id,
    )
    facets = embedding_state_owner.mesh_facets(view, budget=budget)
    premise = embedding_state_owner.EmbeddingState(
        [],
        [],
        candidate_pairs=state.candidate_pairs,
        ray_tests=state.ray_tests,
        subdivision_pieces=state.subdivision_pieces,
    )
    global_embedding_owner.pairing_findings(
        premise, view, facets, np.empty((0,), dtype=np.int64)
    )
    global_embedding_owner.volume_embedding(
        premise, view, coordinates, facets, limits, budget=budget
    )
    state.candidate_pairs = premise.candidate_pairs
    state.ray_tests = premise.ray_tests
    state.subdivision_pieces = premise.subdivision_pieces
    state.subdivision_depth = max(state.subdivision_depth, premise.subdivision_depth)
    state.checks.extend(f"restriction_affine_root:{check}" for check in premise.checks)
    facet_ids = embedding_state_owner.facet_entity_ids(view, facets.rows)

    root_ids = np.asarray(identifiers, dtype=np.int64)
    for finding in premise.findings:
        if finding.entity_kind == "cell":
            affected = np.asarray(finding.entity_ids, dtype=np.int64)
        elif finding.entity_kind == "facet":
            selected = np.isin(facet_ids, np.asarray(finding.entity_ids, dtype=np.int64))
            affected = np.unique(root_ids[facets.cells[selected]])
        else:
            affected = root_ids
        # A refused root union leaves retained-child embedding unresolved.
        state.add(
            f"restriction_affine_root:{finding.check}",
            "unresolved",
            "source_cell",
            affected,
        )
    return True


def _straight_multi_affine_root_embedding(
    state: embedding_state_owner.EmbeddingState,
    roots: list[Cell],
    identifiers: list[int],
    source_geometry_id: str,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Reuse the straight-boundary degree theorem of direct mapped meshes on roots.

    Every root is a multi-affine quadrilateral or hexahedron whose local
    injectivity is already proved. When the rational root corner images are binary64
    points, shared facets have identical traces and every boundary facet image is
    its straight or planar carrier, the root union is embedded exactly when the
    polygonal view of those points is. Any other case returns ``False`` and keeps
    the generic root atlas and pair proofs.
    """

    if not roots or not state.clean:
        return False
    kind, dimension = roots[0].kind, roots[0].dimension
    if (
        kind not in ("quadrilateral", "hexahedron")
        or dimension not in (2, 3)
        or any(
            root.corner_images is None
            or root.kind != kind
            or root.dimension != dimension
            or len(root.coordinates) != dimension
            for root in roots
        )
    ):
        return False
    positions: dict[int, tuple[ExactScalar, ...]] = {}
    for root in roots:
        if root.corner_images is None:
            return False
        for vertex_id, image in zip(root.vertices, root.corner_images, strict=True):
            known = positions.get(vertex_id)
            if known is not None and known != image:
                return False
            positions[vertex_id] = image
    vertex_ids = np.asarray(sorted(positions), dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    rounded = []
    for identifier in vertex_ids.tolist():
        values = tuple(
            coefficient_endpoint(value, upper=True, saturate=True)
            for value in positions[identifier]
        )
        if any(
            not np.isfinite(value) or Fraction.from_float(value) != exact
            for value, exact in zip(values, positions[identifier], strict=True)
        ):
            return False
        rounded.append(values)
    coordinates = np.asarray(rounded, dtype=np.float64)
    vertices = np.asarray(
        [tuple(rows[identifier] for identifier in root.vertices) for root in roots],
        dtype=np.int64,
    )
    view = cell_mesh_owner.CellMesh(
        coordinates,
        (
            reference_cell_types_owner.CellBlock(
                "declared_multi_affine_source_roots",
                kind,
                vertices,
                global_ids=np.asarray(identifiers, dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_geometry_id,
    )
    cells = [
        replace(root, vertices=tuple(vertices[row].tolist()))
        for row, root in enumerate(roots)
    ]
    facets = embedding_state_owner.mesh_facets(view, budget=budget)
    premise = embedding_state_owner.EmbeddingState(
        [],
        [],
        candidate_pairs=state.candidate_pairs,
        ray_tests=state.ray_tests,
        subdivision_pieces=state.subdivision_pieces,
    )
    global_embedding_owner.pairing_findings(
        premise, view, facets, np.empty((0,), dtype=np.int64)
    )
    trace_continuity(premise, view, cells, budget=budget)
    if not premise.clean or not straight_boundary_images(view, cells, budget=budget):
        return False
    premise.checks.append("mapped_straight_boundary_degree")
    global_embedding_owner.volume_embedding(
        premise, view, coordinates, facets, limits, budget=budget
    )
    state.candidate_pairs = premise.candidate_pairs
    state.ray_tests = premise.ray_tests
    state.subdivision_pieces = premise.subdivision_pieces
    state.subdivision_depth = max(state.subdivision_depth, premise.subdivision_depth)
    state.checks.extend(
        f"restriction_multi_affine_root:{check}" for check in premise.checks
    )
    facet_ids = embedding_state_owner.facet_entity_ids(view, facets.rows)
    root_ids = np.asarray(identifiers, dtype=np.int64)
    for finding in premise.findings:
        if finding.entity_kind == "cell":
            affected = np.asarray(finding.entity_ids, dtype=np.int64)
        elif finding.entity_kind == "facet":
            selected = np.isin(facet_ids, np.asarray(finding.entity_ids, dtype=np.int64))
            affected = np.unique(root_ids[facets.cells[selected]])
        else:
            affected = root_ids
        # A refused root union leaves retained-child embedding unresolved.
        state.add(
            f"restriction_multi_affine_root:{finding.check}",
            "unresolved",
            "source_cell",
            affected,
        )
    return True


def _root_atlas_labels(
    state: EmbeddingState,
    roots: list[Cell],
    identifiers: list[int],
    source_geometry_id: str,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Certify injectivity on the declared root topology.

    The corner view supplies incidence. Root expressions supply the atlas maps
    and their global extension.
    """

    absent = np.full((len(roots),), -1, dtype=np.int64)
    if (
        not roots
        or not state.clean
        or roots[0].kind not in ("quadrilateral", "hexahedron")
    ):
        return absent
    kind, dimension = roots[0].kind, roots[0].dimension
    if any(root.kind != kind or root.dimension != dimension for root in roots):
        return absent
    positions: dict[int, tuple[ExactScalar, ...]] = {}
    owners: dict[int, int] = {}
    for root_id, root in zip(identifiers, roots, strict=True):
        for identifier, corner in zip(
            root.vertices, root.reference_vertices, strict=True
        ):
            image = tuple(
                expression_reference_evaluate(value, corner, root.domain, budget=budget)
                for value in root.coordinates
            )
            previous = positions.get(identifier)
            if previous is not None and previous != image:
                state.add(
                    "restriction_source_corner_trace_mismatch",
                    "violated",
                    "source_cell",
                    tuple(sorted({owners[identifier], root_id})),
                )
                return absent
            positions[identifier] = image
            owners[identifier] = root_id
    vertex_ids = np.asarray(sorted(positions), dtype=np.int64)
    rows = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    coordinates = np.asarray(
        [
            tuple(float(value) for value in positions[int(identifier)])
            for identifier in vertex_ids
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(coordinates)):
        state.add("restriction_root_corner_range", "unresolved", "mesh")
        return absent
    vertices = np.asarray(
        [tuple(rows[identifier] for identifier in root.vertices) for root in roots],
        dtype=np.int64,
    )
    view = cell_mesh_owner.CellMesh(
        coordinates,
        (
            reference_cell_types_owner.CellBlock(
                "declared_source_roots",
                kind,
                vertices,
                global_ids=np.asarray(identifiers, dtype=np.int64),
            ),
        ),
        vertex_global_ids=vertex_ids,
        numeric_version=source_geometry_id,
    )
    cells = [
        replace(root, vertices=tuple(vertices[row].tolist()))
        for row, root in enumerate(roots)
    ]
    trace_continuity(state, view, cells, budget=budget)
    if not state.clean:
        return absent
    return tensor_component_injectivity(state, view, cells, limits, budget=budget)
