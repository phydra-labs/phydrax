#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous source-map embedding decisions with bounded interval subdivision.

Local injectivity uses strict monotonicity of a fixed projected physical Jacobian.
For standard reference planar cells, positive determinant together with a certified simple
full boundary supplies the Jordan-degree theorem when monotonicity is
inconclusive. Adjacent planar cells also use rational shared-edge equality and
oriented non-shared boundary separation. Volume meshes use boundary degree when
its premises are proved; other pair routes retain whole-image bounds,
tangent-cone contacts, and strict separating-plane proofs.

Mesh certificate checks
-----------------------
[
{"checks":["mapped_planar_jordan_injectivity"],"owner":"phydrax.geometry._mapped_embedding._planar_jordan_injective","premises":[{"tag":"N","statement":"The determinant is strictly positive on the closed cell and every boundary edge has a certified injective interval map."},{"tag":"X","statement":"Boundary pair contacts are exactly the endpoints shared in the reference disk."},{"tag":"T","statement":"The boundary is a simple positively oriented Jordan curve; the sum of positive local degrees is its winding number, zero outside and one inside."}],"conclusion":"The closed planar source map is injective.","citation":{"kind":"mathematical","reference":"Hatcher, Algebraic Topology","locator":"Section 2.2, Proposition 2.30 (sum of local degrees)","url":"https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"},"failure_statuses":[]},
{"checks":["mapped_planar_jordan_boundary_contact"],"owner":"phydrax.geometry._mapped_embedding._adjacent_planar_jordan_contact","premises":[{"tag":"N","statement":"Both closed-cell degree proofs succeed independently of the caller locally_injective hint."},{"tag":"X","statement":"The common edge traces agree exactly in opposite orientations, and every other boundary arc pair meets only at shared endpoints."},{"tag":"T","statement":"A Jordan-domain crosscut separates its two components; the three disjoint arcs of the theta graph place the two oriented cell interiors on opposite sides of the common arc."}],"conclusion":"The two closed source images intersect only on their shared edge.","citation":{"kind":"mathematical","reference":"Pommerenke, Boundary Behaviour of Conformal Maps (1992)","locator":"Proposition 2.12 (crosscut separation); theta-graph consequence","url":"https://link.springer.com/book/10.1007/978-3-662-02770-7"},"failure_statuses":[]},
{"checks":["mapped_local_injectivity"],"owner":"phydrax.geometry._mapped_embedding.injective","premises":[{"tag":"N","statement":"A fixed physical projection has a uniformly positive-definite symmetric Jacobian over the convex source cell, certified from shared Bernstein controls or diagonal-dominance bounds."}],"conclusion":"A successful local proof makes the closed source cell map injective; its work is counted in subdivision_pieces.","citation":{"kind":"mathematical","reference":"Gale and Nikaido, The Jacobian matrix and global univalence of mappings (1965)","locator":"Positive quasi-definite Jacobian criterion, pp. 81-93","url":"https://doi.org/10.1007/BF01360282"},"failure_statuses":[]},
{"checks":["mapped_convex_reference_atlas"],"owner":"phydrax.geometry._mapped_embedding.tensor_component_injectivity","premises":[{"tag":"X","statement":"Signed-permutation gluing is cycle consistent, vertex identity is bijective with lattice position, and every lattice cell of the rectangular union is occupied once."},{"tag":"N","statement":"Every piece has a strictly monotone Jacobian in one fixed projection and the shared traces are identical."}],"conclusion":"All cells in the certified rectangular atlas form one injective piecewise map.","citation":{"kind":"mathematical","reference":"Gale and Nikaido, The Jacobian matrix and global univalence of mappings (1965)","locator":"Positive quasi-definite Jacobian criterion, pp. 81-93","url":"https://doi.org/10.1007/BF01360282"},"failure_statuses":[]},
{"checks":["mapped_injective_atlas_extension"],"owner":"phydrax.geometry._mapped_embedding.common_atlas_extension","premises":[{"tag":"X","statement":"Each occupied chart is exactly the restriction of one polynomial extension to a common box."},{"tag":"N","statement":"The extension passes the full closed-cell injectivity proof."}],"conclusion":"Every occupied atlas cell is a restriction of the same injective extension.","citation":{"kind":"mathematical","reference":"Gale and Nikaido, The Jacobian matrix and global univalence of mappings (1965)","locator":"Positive quasi-definite Jacobian criterion, pp. 81-93","url":"https://doi.org/10.1007/BF01360282"},"failure_statuses":[]},
{"checks":["mapped_trace_continuity"],"owner":"phydrax.geometry._mapped_embedding.trace_continuity","premises":[{"tag":"X","statement":"Every interior facet with two incidences is compared in one shared reference vertex order using polynomial identity or multi-affine corner identity."}],"conclusion":"All interior-facet trace comparisons are evaluated; failures are reported separately as mapped_trace_mismatch or mapped_trace_denominator.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.trace_continuity","url":""},"failure_statuses":[]},
{"checks":["mapped_trace_mismatch"],"owner":"phydrax.geometry._mapped_embedding.cell_traces_equal","premises":[{"tag":"X","statement":"Two cells give different exact restricted expressions or images on a combinatorially shared entity."}],"conclusion":"The mapped coordinate action is not single-valued on the shared mesh entity.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.cell_traces_equal","url":""},"failure_statuses":["violated"]},
{"checks":["mapped_trace_denominator"],"owner":"phydrax.geometry._mapped_embedding.trace_continuity","premises":[{"tag":"N","statement":"A complete facet restriction cannot be admitted by the rational expression enclosure."}],"conclusion":"Facet trace equality remains unproved.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.trace_continuity","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_affine_source_maps"],"owner":"phydrax.geometry._mapped_embedding._certify_affine_source_embedding","premises":[{"tag":"X","statement":"Every complete source action is affine, all exact source corners agree across shared vertex IDs, and every numerical carrier is their round-to-nearest-even value."},{"tag":"N","statement":"Each affine cell has strictly positive signed measure or nonzero surface immersion."}],"conclusion":"The mapped mesh can use the exact affine-source contact proof without changing its source binding.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._certify_affine_source_embedding","url":""},"failure_statuses":[]},
{"checks":["mapped_exact_shared_corner_incidence"],"owner":"phydrax.geometry._mapped_embedding._collect_affine_images","premises":[{"tag":"X","statement":"All incident full source maps assign the same exact image to each shared vertex ID and their published corner carrier is the round-to-nearest-even view."}],"conclusion":"Affine contact predicates use the authenticated shared corner incidence.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._collect_affine_images","url":""},"failure_statuses":[]},
{"checks":["exact_source_binary64_value_equivalence"],"owner":"phydrax.geometry._mapped_embedding._certify_affine_source_embedding","premises":[{"tag":"X","statement":"Every exact predicate-source coordinate is equal to its binary64 execution value."}],"conclusion":"Binary64 predicate execution represents the exact source values without changing the mapped binding.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._certify_affine_source_embedding","url":""},"failure_statuses":[]},
{"checks":["mapped_exact_affine_surface_contact"],"owner":"phydrax.geometry._mapped_embedding._certify_affine_source_embedding","premises":[{"tag":"X","statement":"Complete affine triangle source maps have authenticated exact corner images and shared incidence."},{"tag":"N","statement":"The exact-source predicate view is admitted before triangle contact tests."}],"conclusion":"Surface contacts are evaluated on the complete affine source maps.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._certify_affine_source_embedding","url":""},"failure_statuses":[]},
{"checks":["mapped_affine_parent_partition"],"owner":"phydrax.geometry._mapped_embedding._affine_parent_partition","premises":[{"tag":"X","statement":"All child reference triangles have positive orientation, lie in the parent, have disjoint interiors, and their exact total area equals the parent area."}],"conclusion":"The children cover the parent triangle exactly once up to shared boundaries.","citation":{"kind":"mathematical","reference":"Lipman, Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing","locator":"Theorems 1 and 4; Section 3, cycle degree","url":"https://arxiv.org/abs/1310.0955"},"failure_statuses":[]},
{"checks":["mapped_cell_contact"],"owner":"phydrax.geometry._mapped_embedding.pair_decision","premises":[{"tag":"X","statement":"Shared source traces agree and any accepted shared-contact support is one authenticated convex reference entity."},{"tag":"N","statement":"An exact affine intersection, strict source-plane separation, closed-atlas injectivity, tangent-cone separation, or certified bisection proof decides each candidate."}],"conclusion":"The evaluated contact obligation either separates a candidate modulo shared entities or produces a distinct overlap/refusal finding.","citation":{"kind":"mathematical","reference":"Rockafellar, Convex Analysis","locator":"Section 11, separation theorems","url":"https://doi.org/10.1515/9781400873173"},"failure_statuses":[]},
{"checks":["mapped_affine_polytope_contact"],"owner":"phydrax.geometry._mapped_embedding.pair_decision","premises":[{"tag":"X","statement":"Both complete coordinate expressions are affine and their shared traces agree exactly; no corner interpolation replaces the source law."},{"tag":"N","statement":"The canonical exact convex-polytope contact owner checks separating axes of the affine simplex or prism images, admitting contact only on authenticated shared reference entities."}],"conclusion":"An applicable exact affine-polytope decision replaces generic mapped contact subdivision; genuinely curved source expressions retain the generic proof.","citation":{"kind":"mathematical","reference":"Rockafellar, Convex Analysis","locator":"Section 11, separation theorems","url":"https://doi.org/10.1515/9781400873173"},"failure_statuses":[]},
{"checks":["mapped_local_orientation"],"owner":"phydrax.geometry._mapped_embedding._affine_corner_measure","premises":[{"tag":"X","statement":"The full source action is affine and its exact corner determinant is nonpositive, or its surface edge cross product is zero."}],"conclusion":"The affine source cell fails positive volume orientation or surface immersion.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._affine_corner_measure","url":""},"failure_statuses":["violated"]},
{"checks":["mapped_cell_overlap"],"owner":"phydrax.geometry._mapped_embedding._crossing_witness","premises":[{"tag":"N","statement":"An interior overlap inclusion or a strict Krawczyk inclusion proves an interior contact of two full source maps."},{"tag":"X","statement":"The source interiors belong to distinct mesh cells."}],"conclusion":"The closed mapped cells have a contact beyond their shared reference entities.","citation":{"kind":"mathematical","reference":"R. E. Moore, A Test for Existence of Solutions to Nonlinear Systems (1977)","locator":"Krawczyk existence test, pp. 611-615","url":"https://doi.org/10.1137/0714040"},"failure_statuses":["violated"]},
{"checks":["mapped_self_contact"],"owner":"phydrax.geometry._mapped_embedding._self_contact_diagnostics","premises":[{"tag":"N","statement":"A strict contact inclusion is proved on two canonical child charts of one source cell."},{"tag":"X","statement":"The child chart interiors are disjoint in the source cell."}],"conclusion":"The source cell map has distinct interior preimages of the same physical point.","citation":{"kind":"mathematical","reference":"R. E. Moore, A Test for Existence of Solutions to Nonlinear Systems (1977)","locator":"Krawczyk existence test, pp. 611-615","url":"https://doi.org/10.1137/0714040"},"failure_statuses":["violated"]},
{"checks":["mapped_straight_boundary_degree"],"owner":"phydrax.geometry._mapped_embedding.straight_boundary_images","premises":[{"tag":"X","statement":"Every exact corner equals its binary64 carrier and each full boundary restriction is affine or has an identically zero bilinear plane residual."},{"tag":"N","statement":"The local closed-cell injectivity and orientation hypotheses have been proved."},{"tag":"T","statement":"Boundary cycle degree counts positive cell preimages and is preserved under the proved boundary image identity."}],"conclusion":"The straight carrier boundary-degree proof applies to the mapped source.","citation":{"kind":"mathematical","reference":"Lipman, Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing","locator":"Theorems 1 and 4; Section 3, cycle degree","url":"https://arxiv.org/abs/1310.0955"},"failure_statuses":[]},
{"checks":["local_injectivity_piece_budget"],"owner":"phydrax.geometry._mapped_embedding.injective","premises":[{"tag":"X","statement":"The local subdivision proof would exceed maximum_subdivision_pieces."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.injective","url":""},"failure_statuses":["unresolved"]},
{"checks":["local_injectivity_subdivision_depth"],"owner":"phydrax.geometry._mapped_embedding.injective","premises":[{"tag":"X","statement":"The current chart depth equals maximum_subdivision_depth."},{"tag":"N","statement":"A source piece reaches maximum_subdivision_depth without a strictly positive projected Jacobian bound."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.injective","url":""},"failure_statuses":["unresolved"]},
{"checks":["local_injectivity_premise"],"owner":"phydrax.geometry._mapped_embedding.injective","premises":[{"tag":"X","statement":"A full physical Jacobian or full-rank fixed rounded projection is unavailable."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.injective","url":""},"failure_statuses":["unresolved"]},
{"checks":["rational_denominator_premise"],"owner":"phydrax.geometry._mapped_embedding.injective","premises":[{"tag":"N","statement":"A projected rational Jacobian has no admitted denominator enclosure."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding.injective","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_intersection_piece_budget"],"owner":"phydrax.geometry._mapped_embedding._bisect_cell_contact","premises":[{"tag":"X","statement":"The next visited or queued chart pair would exceed the remaining subdivision-piece allowance."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._bisect_cell_contact","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_intersection_subdivision_depth"],"owner":"phydrax.geometry._mapped_embedding._bisect_cell_contact","premises":[{"tag":"X","statement":"The current chart depth equals maximum_subdivision_depth."},{"tag":"N","statement":"A nonseparated chart pair reaches maximum_subdivision_depth without a certified interior overlap."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._bisect_cell_contact","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_candidate_pair_budget"],"owner":"phydrax.geometry._mapped_embedding._certify_mapped_embedding","premises":[{"tag":"X","statement":"Closed image-box candidate generation exceeds the remaining maximum_candidate_pairs allowance."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._certify_mapped_embedding","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_self_contact_candidate_budget"],"owner":"phydrax.geometry._mapped_embedding._self_contact_diagnostics","premises":[{"tag":"X","statement":"The next canonical child-pair diagnostic exceeds maximum_candidate_pairs."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._self_contact_diagnostics","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_bernstein_node_budget"],"owner":"phydrax.geometry._mapped_embedding._prepare_embedding_cell","premises":[{"tag":"X","statement":"The maximum complete source-expression Bernstein node count exceeds maximum_bernstein_nodes."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._prepare_embedding_cell","url":""},"failure_statuses":["unresolved"]},
{"checks":["mapped_coordinate_bound_range"],"owner":"phydrax.geometry._mapped_embedding._prepare_embedding_cell","premises":[{"tag":"N","statement":"A complete root image enclosure has a nonfinite outward binary64 endpoint."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._prepare_embedding_cell","url":""},"failure_statuses":["unresolved"]},
{"checks":["coordinate_source_enclosure"],"owner":"phydrax.geometry._mapped_embedding._prepare_embedding_cell","premises":[{"tag":"X","statement":"The complete source coordinate expressions cannot be admitted for the declared element."}],"conclusion":"The named mapped proof obligation remains unresolved; no successful embedding conclusion is published from that failed obligation.","citation":{"kind":"definition","reference":"Mapped embedding proof and resource contract","locator":"phydrax.geometry._mapped_embedding._prepare_embedding_cell","url":""},"failure_statuses":["unresolved"]}
]
End mesh certificate checks
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import closing
from fractions import Fraction
from itertools import chain, combinations
from typing import assert_never, cast, final, NamedTuple, TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import Array

import phydrax.discretization._coordinate_enclosure as coordinate_enclosure_owner
import phydrax.discretization._coordinate_identity as coordinate_identity_owner
import phydrax.geometry._affine_mapped_contact as affine_mapped_contact_owner
import phydrax.geometry._boundary_degree_embedding as boundary_degree_embedding_owner
import phydrax.geometry._embedding_state as embedding_state_owner
import phydrax.geometry._global_embedding as mesh_certificates_owner
import phydrax.geometry._planar_primitives as planar_primitives_owner
import phydrax.geometry._prism_embedding as prism_embedding_owner
from phydrax.discretization._coordinate_budget import reserve_polynomial
from phydrax.discretization._reference_cell_types import CellGeometryElement

if TYPE_CHECKING:
    from ._domain_chart_chain import _CarrierAdmission

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_predecessor import CellGeometryPredecessor
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
)
from ..discretization._coordinate_enclosure import (
    affine_arguments,
    BernsteinDomain,
    box_corner_lattice,
    constant,
    coordinate_expressions,
    ExactScalar,
    Expression,
    expression_add as add,
    expression_bernstein_coefficients as bernstein_coefficients,
    expression_bounds as polynomial_bounds,
    expression_compose as compose,
    expression_control_net,
    expression_derivative as derivative,
    expression_evaluate as evaluate,
    expression_linear_combinations,
    expression_node_count as bernstein_node_count,
    expression_physical_jacobian as physical_jacobian,
    expression_reference_evaluate,
    expression_scale as scale,
    expression_sequence_equal,
    expression_sum as sum_polynomials,
    ExpressionComposition as ArgumentComposition,
    multi_affine_coordinates,
    Polynomial,
    prepared_coordinate_source_bank,
    RationalPolynomial,
    restrict_chart_expressions,
)
from ..discretization._exact_algebraic_geometry import (
    RadicalElement,
    rational_coefficient,
    reserve_algebraic_work,
)
from ..discretization._reference_cell import reference_cell_topology
from ..linalg._small_batched import cofactor_determinant
from ..typing import checked
from ._embedding_cells import (
    BisectionWorklist,
    bounds,
    Cell,
    children as embedding_children,
    convex_triangle_union,
    domain as embedding_domain,
    identity_reference_map,
    IntervalBoundaryCell,
    IntervalBoundaryHomotopy,
    IntervalSurfaceMap,
    vertices as embedding_vertices,
)
from ._embedding_state import ContactDecision, EmbeddingState, MeshCertificateLimits
from ._mesh_check_names import MeshCertificateCheck
from ._rational import rational_cross, rational_vector
from ._interval_enclosure import (
    interval_add,
    interval_dot,
    interval_multiply,
    interval_subtract,
)
from ._rational import fraction_endpoint
from ._planar_primitives import reserve_fraction_work


def _planar_boundary_edges(
    cell: Cell, /, *, budget: CoordinateEnclosureBudget
) -> tuple[Cell, ...] | None:
    """Restrict a standard reference planar cell to its full oriented boundary."""
    if (
        cell.dimension != 2
        or len(cell.coordinates) != 2
        or cell.kind not in ("triangle", "quadrilateral")
        or cell.reference_vertices != embedding_vertices(cell.kind)
    ):
        return None
    count = len(cell.vertices)
    edges = []
    for first in range(count):
        last = (first + 1) % count
        origin = np.asarray(
            [float(value) for value in cell.reference_vertices[first]], dtype=np.float64
        )
        endpoint = np.asarray(
            [float(value) for value in cell.reference_vertices[last]], dtype=np.float64
        )
        arguments = affine_arguments(origin, (endpoint - origin)[:, None], budget=budget)
        edges.append(
            Cell(
                tuple(
                    compose(value, arguments, budget=budget) for value in cell.coordinates
                ),
                "box",
                "interval",
                1,
                (cell.vertices[first], cell.vertices[last]),
                ((Fraction(0),), (Fraction(1),)),
            )
        )
    return tuple(edges)


def _planar_positive_jacobian(
    cell: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    """Prove strict positive orientation over a full planar source map."""

    physical = physical_jacobian(
        cell.coordinates, cell.kind, cell.dimension, budget=budget
    )
    if cell.dimension != 2 or len(cell.coordinates) != 2 or physical is None:
        return False, 0
    determinant = coordinate_enclosure_owner.expression_determinant(
        physical, variable_dimension=2, budget=budget
    )
    active = [(np.zeros((2,), dtype=np.float64), np.eye(2, dtype=np.float64), 0)]
    work = 0
    while active:
        if work >= maximum_work:
            return False, work
        origin, matrix, depth = active.pop()
        work += 1
        piece = compose(
            determinant, affine_arguments(origin, matrix, budget=budget), budget=budget
        )
        lower, _ = polynomial_bounds(piece, cell.domain, 2, budget=budget)
        if lower > 0.0:
            continue
        if depth >= limits.maximum_subdivision_depth:
            return False, work
        children = embedding_children(cell, origin, matrix)
        if work + len(active) + len(children) > maximum_work:
            return False, work
        active.extend((start, child, depth + 1) for start, child in children)
    return True, work


def _corner_lattice(cell: Cell) -> list[tuple[ExactScalar, ...]]:
    """Multi-affine corner images indexed by their reference-axis bitmask."""
    if cell.corner_images is None:
        raise ValueError(
            "mapped_embedding.corner_images must be available for a multi-affine cell; got None"
        )
    masks, _ = box_corner_lattice(cell.kind)
    corners: list[tuple[ExactScalar, ...]] = [()] * len(masks)
    for mask, image in zip(masks, cell.corner_images, strict=True):
        corners[mask] = image
    return corners


def _rounded_matrix(values: tuple[tuple[ExactScalar, ...], ...]) -> np.ndarray | None:
    maximum = Fraction(float(np.finfo(np.float64).max))
    if any(abs(value) > maximum for row in values for value in row):
        return None
    return np.asarray(
        [[float(value) for value in row] for row in values], dtype=np.float64
    )


def _midpoint_projection(
    midpoint: np.ndarray | None, dimension: int
) -> np.ndarray | None:
    """Fixed binary64 pseudo-inverse of a full-rank midpoint Jacobian."""
    if midpoint is None or np.linalg.matrix_rank(midpoint) != dimension:
        return None
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    return projection if rank == dimension else None


@checked
def jacobian_midpoint(
    cell: Cell,
    jacobian: tuple[tuple[Expression, ...], ...] | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray | None:
    if jacobian is None:
        jacobian = physical_jacobian(
            cell.coordinates, cell.kind, cell.dimension, budget=budget
        )
    if jacobian is None:
        return None
    point = (
        tuple(Fraction(1, cell.dimension + 1) for _ in range(cell.dimension))
        if cell.domain == "simplex"
        else (Fraction(1, 3), Fraction(1, 3), Fraction(1, 2))
        if cell.domain == "prism"
        else (Fraction(1, 2),) * cell.dimension
    )
    values = tuple(
        tuple(evaluate(entry, point, budget=budget) for entry in row) for row in jacobian
    )
    return _rounded_matrix(values)


@checked
def projected_jacobian(
    cell: Cell,
    projection: np.ndarray,
    jacobian: tuple[tuple[Expression, ...], ...] | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[Expression, ...], ...] | None:
    if jacobian is None:
        jacobian = physical_jacobian(
            cell.coordinates, cell.kind, cell.dimension, budget=budget
        )
    if jacobian is None:
        return None
    weights = tuple(rational_vector(row) for row in projection)
    columns = tuple(
        expression_linear_combinations(
            tuple(row[j] for row in jacobian), weights, budget=budget
        )
        for j in range(cell.dimension)
    )
    return tuple(tuple(column[i] for column in columns) for i in range(cell.dimension))


def _symmetric_net_polynomials(
    jacobian: tuple[tuple[Expression, ...], ...],
    dimension: int,
    composition: ArgumentComposition | None,
    symmetric_parts: dict[tuple[int, int], Expression],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[Polynomial, ...] | None:
    """Prepare the shared symmetric matrix entries before degree elevation."""
    entries: list[Expression] = [jacobian[i][i] for i in range(dimension)]
    for i in range(dimension):
        for j in range(i + 1, dimension):
            symmetric = symmetric_parts.get((i, j))
            if symmetric is None:
                symmetric = expression_linear_combinations(
                    (jacobian[i][j], jacobian[j][i]),
                    ((Fraction(1, 2), Fraction(1, 2)),),
                    budget=budget,
                )[0]
                symmetric_parts[i, j] = symmetric
            entries.append(symmetric)
    polynomials: list[Polynomial] = []
    for entry in entries:
        value = entry if composition is None else composition(entry, budget=budget)
        if isinstance(value, RationalPolynomial):
            return None
        polynomials.append(value)

    return tuple(polynomials)


def _common_bernstein_degrees(
    polynomials: tuple[Polynomial, ...], domain: BernsteinDomain, dimension: int
) -> tuple[int, ...]:
    """Return degrees that give every symmetric entry the same convex weights."""
    match domain:
        case "box":
            degrees = tuple(
                max((index[axis] for value in polynomials for index in value), default=0)
                for axis in range(dimension)
            )
        case "simplex":
            degrees = (
                max((sum(index) for value in polynomials for index in value), default=0),
            )
        case "prism":
            degrees = (
                max(
                    (sum(index[:2]) for value in polynomials for index in value),
                    default=0,
                ),
                max((index[2] for value in polynomials for index in value), default=0),
            )
        case _:
            raise ValueError(
                "mapped_embedding.domain must be box, simplex, or prism; got an unsupported domain"
            )
    return degrees


def _correlated_bernstein_spd(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    composition: ArgumentComposition | None,
    symmetric_parts: dict[tuple[int, int], Expression],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove strict monotonicity by convexity of symmetric positive-definite nets."""
    if dimension not in (2, 3):
        return False
    if any(isinstance(entry, RationalPolynomial) for row in jacobian for entry in row):
        # Different rational denominators do not share Bernstein convex weights.
        return False
    polynomials = _symmetric_net_polynomials(
        jacobian, dimension, composition, symmetric_parts, budget=budget
    )
    if polynomials is None:
        return False
    budget.reserve(dimension * sum(len(value) for value in polynomials))
    degrees = _common_bernstein_degrees(polynomials, domain, dimension)
    if not any(degrees):
        # A constant net has one rational control, independent of the domain.
        # Preserve the same principal-minor proof without preparing identity
        # monomial weights separately for every matrix entry.
        budget.reserve(len(polynomials))
        controls = tuple(
            (value.get((0,) * dimension, Fraction(0)),) for value in polynomials
        )
    else:
        controls = tuple(
            expression_control_net(value, domain, degrees, dimension, budget=budget)
            for value in polynomials
        )
    budget.reserve(sum(len(net) for net in controls))
    bits = max(
        (
            numerator_bits + denominator.bit_length()
            for net in controls
            for value in net
            for numerator_bits, denominator in (
                coordinate_enclosure_owner._scalar_profile(value),
            )
        ),
        default=1,
    )
    reserve_polynomial(0, 32, 0, 24 * bits + 4, budget=budget)
    for values in zip(*controls, strict=True):
        extra = None if dimension == 2 else (values[2], values[4], values[5])
        if not _spd_minors(values[0], values[1], values[dimension], extra, budget=budget):
            return False
    return True


def _spd_minors(
    a: ExactScalar,
    d: ExactScalar,
    b: ExactScalar,
    extra: tuple[ExactScalar, ExactScalar, ExactScalar] | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Positive leading principal minors of one rational symmetric control matrix.

    ``a``, ``d`` are the first two diagonal entries and ``b`` their coupling;
    ``extra`` carries ``(f, c, e)``: the third diagonal entry and its couplings
    to the first and second rows.
    """

    if (
        isinstance(a, RadicalElement)
        or isinstance(d, RadicalElement)
        or isinstance(b, RadicalElement)
        or (
            extra is not None
            and any(isinstance(value, RadicalElement) for value in extra)
        )
    ):
        reserve_algebraic_work((a, d, b, *(extra or ())), 32, 32, 4, budget=budget)
    budget.reserve(1)
    if a <= 0:
        return False
    budget.reserve(4)
    minor = a * d - b * b
    if minor <= 0:
        return False
    if extra is not None:
        f, c, e = extra
        budget.reserve(12)
        if f * minor - a * e * e - d * c * c + 2 * b * c * e <= 0:
            return False
    return True


@final
class _MultiAffineEdges(StrictModule, NonTrainableState):
    physical: tuple[dict[int, dict[int, ExactScalar]], ...]
    midpoint: np.ndarray | None
    image_numerator: int = eqx.field(static=True)
    image_common: int = eqx.field(static=True)


@final
class _SymmetricControls(StrictModule, NonTrainableState):
    values: list[tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]]]


def _multi_affine_edges(
    corners: list[tuple[ExactScalar, ...]],
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MultiAffineEdges:
    """Prepare rational physical edge columns and their midpoint projection input."""
    count, edges, ambient = len(corners), len(corners) // 2, len(corners[0])
    image_numerator, image_common = 1, 1
    algebraic = False
    budget.reserve(count * ambient)
    for image in corners:
        for value in image:
            numerator_bits, denominator = coordinate_enclosure_owner._scalar_profile(
                value
            )
            image_numerator = max(image_numerator, numerator_bits)
            image_common = math.lcm(image_common, denominator)
            algebraic = algebraic or isinstance(value, RadicalElement)
    if algebraic:
        reserve_algebraic_work(
            corners,
            ambient * dimension * (2 * edges + 1),
            edges * dimension * ambient,
            2,
            budget=budget,
        )
    reserve_polynomial(
        ambient * dimension * (2 * edges + 1),
        edges * dimension * ambient,
        dimension,
        image_numerator + image_common.bit_length() + 2,
        budget=budget,
    )
    physical = tuple(
        {
            mask: {
                row: value
                for row in range(ambient)
                if (value := corners[mask | (1 << axis)][row] - corners[mask][row])
            }
            for mask in range(count)
            if not (mask >> axis) & 1
        }
        for axis in range(dimension)
    )
    zero = Fraction(0)
    midpoint = _rounded_matrix(
        tuple(
            tuple(
                sum((edge.get(row, zero) for edge in physical[axis].values()), zero)
                / edges
                for axis in range(dimension)
            )
            for row in range(ambient)
        )
    )
    return _MultiAffineEdges(physical, midpoint, image_numerator, image_common)


@final
class _ProjectionWeights(StrictModule, NonTrainableState):
    rows: list[tuple[tuple[int, Fraction], ...]]
    numerator: int = eqx.field(static=True)
    common: int = eqx.field(static=True)


def _sparse_projection_weights(projection: np.ndarray) -> _ProjectionWeights:
    """Retain the rational lifts of nonzero rounded projection weights and size bounds."""
    weights = tuple(rational_vector(row) for row in projection)
    weight_numerator, weight_common = 1, 1
    sparse_weights = []
    for row in weights:
        sparse = []
        for axis, value in enumerate(row):
            weight_numerator = max(weight_numerator, abs(value.numerator).bit_length())
            weight_common = math.lcm(weight_common, value.denominator)
            if value:
                sparse.append((axis, value))
        sparse_weights.append(tuple(sparse))
    return _ProjectionWeights(sparse_weights, weight_numerator, weight_common)


def _symmetric_corner_controls(
    differences: tuple[dict[int, tuple[ExactScalar, ...]], ...],
    count: int,
    dimension: int,
    pairs: tuple[tuple[int, int], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _SymmetricControls:
    """Assemble symmetric corner matrices with shared convex Bernstein weights."""
    budget.reserve(2 * count * len(pairs))
    controls = []
    for mask in range(count):
        column = tuple(
            differences[axis][mask & ~(1 << axis)] for axis in range(dimension)
        )
        controls.append(
            (
                tuple(column[i][i] for i in range(dimension)),
                tuple((column[j][i] + column[i][j]) / 2 for i, j in pairs),
            )
        )
    budget.reserve(count * (dimension + len(pairs)))
    return _SymmetricControls(controls)


def _multi_affine_controls(
    prepared: _MultiAffineEdges,
    projection: np.ndarray,
    count: int,
    dimension: int,
    ambient: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _SymmetricControls:
    """Project rational edge columns and assemble symmetric corner matrices."""
    physical = prepared.physical
    image_numerator, image_common = prepared.image_numerator, prepared.image_common
    edges = count // 2
    zero = Fraction(0)
    budget.reserve(dimension * ambient)
    prepared_weights = _sparse_projection_weights(projection)
    sparse_weights, weight_numerator, weight_common = (
        prepared_weights.rows,
        prepared_weights.numerator,
        prepared_weights.common,
    )
    pairs = tuple((i, j) for i in range(dimension) for j in range(i + 1, dimension))
    products = sum(
        sum(axis in edge for axis, _ in row)
        for columns in physical
        for edge in columns.values()
        for row in sparse_weights
    )
    if any(
        isinstance(value, RadicalElement)
        for columns in physical
        for edge in columns.values()
        for value in edge.values()
    ):
        reserve_algebraic_work(
            tuple(
                value
                for columns in physical
                for edge in columns.values()
                for value in edge.values()
            ),
            products,
            edges * dimension * dimension + count * len(pairs),
            2,
            budget=budget,
        )
    reserve_polynomial(
        products,
        edges * dimension * dimension + count * len(pairs),
        dimension,
        image_numerator
        + image_common.bit_length()
        + weight_numerator
        + weight_common.bit_length()
        + ambient.bit_length()
        + 2,
        budget=budget,
    )
    differences = tuple(
        {
            mask: tuple(
                sum((weight * edge[axis] for axis, weight in row if axis in edge), zero)
                for row in sparse_weights
            )
            for mask, edge in columns.items()
        }
        for columns in physical
    )
    return _symmetric_corner_controls(differences, count, dimension, pairs, budget=budget)


def _multi_affine_monotone(cell: Cell, *, budget: CoordinateEnclosureBudget) -> bool:
    """Prove the full multi-affine Jacobian net strictly monotone.

    Physical edge columns are prepared once and used both for the midpoint
    projection and for every rational corner control. Sparse projection omits
    only proved zero coefficients. Every symmetric control matrix retains its
    full rational principal-minor checks without speculative cache-key visits.
    """
    dimension = cell.dimension
    if cell.corner_images is None or dimension not in (2, 3):
        return False
    corners = _corner_lattice(cell)
    count, ambient = len(corners), len(corners[0])

    prepared = _multi_affine_edges(corners, dimension, budget=budget)
    projection = _midpoint_projection(prepared.midpoint, dimension)
    if projection is None:
        return False
    controls = _multi_affine_controls(
        prepared, projection, count, dimension, ambient, budget=budget
    ).values
    bits = max(
        numerator_bits + denominator.bit_length()
        for diagonal, coupling in controls
        for value in (*diagonal, *coupling)
        for numerator_bits, denominator in (
            coordinate_enclosure_owner._scalar_profile(value),
        )
    )
    reserve_polynomial(0, 32, 0, 24 * bits + 4, budget=budget)
    for diagonal, coupling in controls:
        extra = None if dimension == 2 else (diagonal[2], coupling[1], coupling[2])
        if not _spd_minors(diagonal[0], diagonal[1], coupling[0], extra, budget=budget):
            return False
    return True


@checked
def strict_spd(
    jacobian: tuple[tuple[Expression, ...], ...],
    domain: BernsteinDomain,
    dimension: int,
    origin: np.ndarray,
    matrix: np.ndarray,
    symmetric_parts: dict[tuple[int, int], Expression],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    composition = (
        None
        if identity_reference_map(origin, matrix)
        else ArgumentComposition(
            affine_arguments(origin, matrix, budget=budget), budget=budget
        )
    )
    if _correlated_bernstein_spd(
        jacobian, domain, dimension, composition, symmetric_parts, budget=budget
    ):
        return True
    # The symmetric part is shared by rows i and j; enclose it once per piece.
    magnitudes: dict[tuple[int, int], float] = {}
    for i in range(dimension):
        entry = jacobian[i][i]
        diagonal = polynomial_bounds(
            entry if composition is None else composition(entry, budget=budget),
            domain,
            dimension,
            budget=budget,
        )[0]
        radius = 0.0
        for j in range(dimension):
            if i != j:
                pair = (min(i, j), max(i, j))
                magnitude = magnitudes.get(pair)
                if magnitude is None:
                    symmetric = symmetric_parts.get(pair)
                    if symmetric is None:
                        symmetric = expression_linear_combinations(
                            (jacobian[pair[0]][pair[1]], jacobian[pair[1]][pair[0]]),
                            ((Fraction(1, 2), Fraction(1, 2)),),
                            budget=budget,
                        )[0]
                        symmetric_parts[pair] = symmetric
                    interval = polynomial_bounds(
                        symmetric
                        if composition is None
                        else composition(symmetric, budget=budget),
                        domain,
                        dimension,
                        budget=budget,
                    )
                    magnitude = magnitudes[pair] = max(abs(value) for value in interval)
                radius = float(np.nextafter(radius + magnitude, math.inf))
        if not diagonal > radius:
            return False
    return True


@final
class _BoundarySimplicity(StrictModule, NonTrainableState):
    proved: bool = eqx.field(static=True)
    work: int = eqx.field(static=True)


def _planar_boundary_simple(
    edges: tuple[Cell, ...],
    limits: MeshCertificateLimits,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _BoundarySimplicity:
    """Prove every closed boundary edge injective and every edge pair endpoint-only."""
    boundary_state = embedding_state_owner.EmbeddingState([], [])
    for edge in edges:
        reason = injective(
            edge,
            limits,
            boundary_state,
            maximum_work=maximum_work - boundary_state.subdivision_pieces,
            budget=budget,
        )
        if reason is not None:
            return _BoundarySimplicity(False, boundary_state.subdivision_pieces)
    work = boundary_state.subdivision_pieces
    for first in range(len(edges)):
        for second in range(first + 1, len(edges)):
            contact_decision_3 = pair_decision(
                edges[first],
                edges[second],
                limits,
                maximum_work - work,
                budget=budget,
            )
            decision = contact_decision_3.status
            used = contact_decision_3.work
            work += used
            if decision != "separated":
                return _BoundarySimplicity(False, work)
    return _BoundarySimplicity(True, work)


def _planar_jordan_injective(
    cell: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    """Prove injectivity from positive orientation and a simple oriented boundary.

    N: the determinant is strictly positive on the closed reference disk and
    each boundary edge is injective. X: edge contacts are precisely the shared
    reference endpoints. T: the degree/winding-number lemma counts preimages
    with their Jacobian signs; a positively oriented simple boundary has winding
    number zero outside and one inside. Thus every interior image point has
    exactly one preimage. See Hatcher, Algebraic Topology, Section 2.2, Proposition 2.30
    (local-degree sum), and Lipman, Theorems 4 and 1 (mesh preimage counting). Failure to certify
    a bound or boundary contact returns False with consumed subdivision work.
    """
    if (
        cell.dimension != 2
        or len(cell.coordinates) != 2
        or maximum_work <= 0
        or len(set(cell.vertices)) != len(cell.vertices)
    ):
        return False, 0

    with budget.temporary_scope():
        budget.reserve(0, 2048 + 1024 * len(cell.vertices))
        edges = _planar_boundary_edges(cell, budget=budget)
        if edges is None:
            return False, 0
        oriented, work = _planar_positive_jacobian(
            cell, limits, maximum_work, budget=budget
        )
        if not oriented:
            return False, work
        boundary = _planar_boundary_simple(
            edges, limits, maximum_work - work, budget=budget
        )
        return boundary.proved, work + boundary.work


def _local_planar_jordan_fallback(
    cell: Cell,
    limits: MeshCertificateLimits,
    state: EmbeddingState,
    reason: MeshCertificateCheck,
    maximum_pieces: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MeshCertificateCheck | None:
    """Discharge a projection refusal when the closed-planar degree proof succeeds."""
    if cell.dimension != 2 or len(cell.coordinates) != 2:
        return reason
    proved, work = _planar_jordan_injective(
        cell, limits, maximum_pieces - state.subdivision_pieces, budget=budget
    )
    state.subdivision_pieces += work
    if not proved:
        return reason
    check: MeshCertificateCheck = "mapped_planar_jordan_injectivity"
    if check not in state.checks:
        state.checks.append(check)
    return None


def _constant_frame_nonsingular(
    projection: np.ndarray,
    dimension: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove the rounded constant frame nonsingular using rational arithmetic."""
    if projection.shape != (dimension, dimension) or not np.all(np.isfinite(projection)):
        return False
    budget.reserve(projection.size)
    frame = tuple(
        tuple(constant(Fraction(float(value)), 0, budget=budget) for value in row)
        for row in projection
    )
    determinant = coordinate_enclosure_owner.expression_determinant(
        frame, variable_dimension=0, budget=budget
    )
    return evaluate(determinant, (), budget=budget) != 0


def _box_p_matrix_univalent(
    cell: Cell,
    projection: np.ndarray,
    jacobian: tuple[tuple[Expression, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove closed-box univalence from all seven projected principal minors.

    The source is a polynomial map on the closed reference cube, hence C1 on
    a neighborhood of that rectangular domain. The constant binary64 frame is
    lifted to rational coefficients and its determinant is proved nonzero.
    Every projected principal minor has a strictly positive rational Bernstein
    lower bound over the whole cube. Gale--Nikaido rectangular univalence then
    makes the projected map injective, and the nonsingular frame gives the same
    conclusion for the source map. Other reference domains do not enter this
    proof. A nonpositive bound returns False; resource refusals propagate.

    Determinant expansion and Bernstein conversion use the coordinate
    arithmetic owner, which reserves from operand term counts, degrees and
    coefficient sizes before constructing the expanded polynomials or controls.

    Mesh certificate checks
    -----------------------
    [
    {"checks":["mapped_box_p_matrix_univalence","mapped_local_injectivity"],"owner":"phydrax.geometry._mapped_embedding._box_p_matrix_univalent","premises":[{"tag":"N","statement":"All seven principal minors of the fixed projected Jacobian have strictly positive rational Bernstein lower bounds over the full closed cube: three order-one minors, three order-two minors and the order-three determinant."},{"tag":"X","statement":"The source coordinates are polynomials on the closed axis-aligned rectangular reference cube, so the map is C1 on a neighborhood; the constant rounded projection is square and its rational determinant is nonzero."},{"tag":"T","statement":"Gale--Nikaido rectangular global univalence applies to a C1 map whose Jacobian is a P-matrix throughout the rectangle."}],"conclusion":"The projected map and, through its nonsingular constant frame, the source cell map are injective on the closed cube. Unsupported domains and nonpositive certified minor bounds do not satisfy this proof.","citation":{"kind":"mathematical","reference":"Gale and Nikaido, The Jacobian matrix and global univalence of mappings (1965)","locator":"Theorem 4, p. 86 (closed rectangular P-matrix global univalence)","url":"https://doi.org/10.1007/BF01360282"},"failure_statuses":[]}
    ]
    End mesh certificate checks
    """
    if (
        cell.domain != "box"
        or cell.kind != "hexahedron"
        or cell.dimension != 3
        or len(cell.coordinates) != 3
    ):
        return False
    for value in (
        *cell.coordinates,
        *(entry for row in jacobian for entry in row),
    ):
        match value:
            case RationalPolynomial():
                return False
            case dict():
                continue
            case _:
                assert_never(value)
    with budget.temporary_scope():
        if not _constant_frame_nonsingular(projection, cell.dimension, budget=budget):
            return False
        for order in (1, 2, 3):
            for indices in combinations(range(cell.dimension), order):
                minor = (
                    jacobian[indices[0]][indices[0]]
                    if order == 1
                    else coordinate_enclosure_owner.expression_determinant(
                        tuple(tuple(jacobian[i][j] for j in indices) for i in indices),
                        variable_dimension=cell.dimension,
                        budget=budget,
                    )
                )
                coefficients = bernstein_coefficients(
                    minor, cell.domain, cell.dimension, budget=budget
                )
                budget.reserve(len(coefficients))
                if min(coefficients) <= 0:
                    return False
    return True


def root_injective(
    cell: Cell,
    element: CellGeometryElement,
    local: coordinate_enclosure_owner.CoordinateSourceBank,
    limits: MeshCertificateLimits,
    state: EmbeddingState,
    *,
    budget: CoordinateEnclosureBudget,
) -> MeshCertificateCheck | None:
    """Reuse an earned root theorem, not its discarded Bernstein workspace.

    Both callers construct the complete root map from this exact element and
    coefficient bank. The reference family, ordered incidence, and every proof
    limit remain bound to the result. Logical subdivision coverage still counts
    against the receiving certificate, although its arithmetic is not repeated.
    """
    _, source_key = coordinate_enclosure_owner.coordinate_preparation_key(
        element, local, budget=budget
    )
    budget.reserve(1 + len(cell.vertices) + sum(map(len, cell.reference_vertices)))
    key = (
        source_key, cell.kind, cell.domain, cell.dimension, limits.limits_id,
        cell.vertices, cell.reference_vertices,
    )
    proof = budget.root_injectivity_cache.get(key)
    remaining = limits.maximum_subdivision_pieces - state.subdivision_pieces
    if proof is None:
        proved = embedding_state_owner.EmbeddingState([], [])
        with budget.polynomial_preparation_scope():
            reason = injective(
                cell, limits, proved, maximum_work=remaining, budget=budget
            )
        proof = proved.subdivision_pieces, cast(
            tuple[MeshCertificateCheck, ...], tuple(proved.checks)
        )
        if reason is None:
            budget.reserve(1, 128)
            budget.retain_basis((key, proof))
            budget.root_injectivity_cache[key] = proof
    else:
        reason = (
            "local_injectivity_piece_budget" if proof[0] > remaining else None
        )
        if reason is not None:
            return reason
    state.subdivision_pieces += proof[0]
    for check in proof[1]:
        if check not in state.checks:
            state.checks.append(check)
    return reason


@checked
def injective(
    cell: Cell,
    limits: MeshCertificateLimits,
    state: EmbeddingState,
    *,
    maximum_work: int | None = None,
    budget: CoordinateEnclosureBudget,
) -> MeshCertificateCheck | None:
    maximum_pieces = limits.maximum_subdivision_pieces
    if maximum_work is not None:
        maximum_pieces = min(maximum_pieces, state.subdivision_pieces + maximum_work)
    if cell.corner_images is not None:
        if state.subdivision_pieces >= maximum_pieces:
            return "local_injectivity_piece_budget"
        if _multi_affine_monotone(cell, budget=budget):
            state.subdivision_pieces += 1
            return None
    physical = physical_jacobian(
        cell.coordinates, cell.kind, cell.dimension, budget=budget
    )
    if physical is None:
        return "local_injectivity_premise"

    affine_key: tuple[tuple[ExactScalar, ...], ...] | None = None
    rows = []
    constant = True
    for row in physical:
        values = []
        for entry in row:
            if isinstance(entry, RationalPolynomial):
                constant = False
                break
            budget.reserve(len(entry))
            if any(any(index) for index in entry):
                constant = False
                break
            values.append(entry.get((0,) * cell.dimension, Fraction(0)))
        if not constant:
            break
        rows.append(tuple(values))
    if constant:
        affine_key = tuple(rows)
        budget.reserve(1)
        if affine_key in budget.affine_injectivity_cache:
            if state.subdivision_pieces >= maximum_pieces:
                return "local_injectivity_piece_budget"
            state.subdivision_pieces += 1
            return None
        if (
            cell.dimension in (2, 3)
            and len(affine_key) == cell.dimension
            and all(isinstance(value, (int, Fraction)) for row in affine_key for value in row)
        ):
            if state.subdivision_pieces >= maximum_pieces:
                return "local_injectivity_piece_budget"
            # A constant square Jacobian is globally injective precisely
            # when its exact matrix has full rank. No interval/SPD net is
            # needed to re-prove this linear-algebra identity.
            reserve_fraction_work(
                affine_key,
                3 if cell.dimension == 2 else 14,
                1,
                4 if cell.dimension == 2 else 18,
                budget=budget,
            )
            budget.reserve(0, 256 + 8 * cell.dimension * cell.dimension)
            determinant = cofactor_determinant(np.asarray(affine_key, dtype=object))
            if determinant != 0:
                budget.retain_basis((affine_key,))
                budget.affine_injectivity_cache.add(affine_key)
                state.subdivision_pieces += 1
                return None
    projection = _midpoint_projection(
        jacobian_midpoint(cell, physical, budget=budget), cell.dimension
    )
    if projection is None:
        return _local_planar_jordan_fallback(
            cell,
            limits,
            state,
            "local_injectivity_premise",
            maximum_pieces,
            budget=budget,
        )
    jacobian = projected_jacobian(cell, projection, physical, budget=budget)
    if jacobian is None:
        return _local_planar_jordan_fallback(
            cell,
            limits,
            state,
            "rational_denominator_premise",
            maximum_pieces,
            budget=budget,
        )
    active = [
        (
            np.zeros((cell.dimension,), dtype=np.float64),
            np.eye(cell.dimension, dtype=np.float64),
            0,
        )
    ]
    symmetric_parts: dict[tuple[int, int], Expression] = {}
    while active:
        origin, matrix, depth = active.pop()
        if state.subdivision_pieces >= maximum_pieces:
            return "local_injectivity_piece_budget"
        state.subdivision_pieces += 1
        if strict_spd(
            jacobian,
            cell.domain,
            cell.dimension,
            origin,
            matrix,
            symmetric_parts,
            budget=budget,
        ):
            continue
        if depth == 0 and _box_p_matrix_univalent(
            cell, projection, jacobian, budget=budget
        ):
            check: MeshCertificateCheck = "mapped_box_p_matrix_univalence"
            if check not in state.checks:
                state.checks.append(check)
            return None
        if depth == limits.maximum_subdivision_depth:
            return _local_planar_jordan_fallback(
                cell,
                limits,
                state,
                "local_injectivity_subdivision_depth",
                maximum_pieces,
                budget=budget,
            )
        active.extend(
            (start, child, depth + 1)
            for start, child in embedding_children(cell, origin, matrix)
        )
    if affine_key is not None:
        # The projected-Jacobian/SPD proof above established this injectivity result.
        # Translation and reference-domain subdivision cannot change a
        # constant physical Jacobian, so identical matrices reuse that proof.
        budget.retain_basis((affine_key,))
        budget.affine_injectivity_cache.add(affine_key)
    return None


def _signed_image(
    cell: Cell,
    normal: np.ndarray,
    offset: ExactScalar,
    *,
    budget: CoordinateEnclosureBudget,
) -> Expression:
    return add(
        sum_polynomials(
            tuple(
                scale(
                    value,
                    weight if isinstance(weight, RadicalElement) else Fraction(weight),
                    budget=budget,
                )
                for value, weight in zip(cell.coordinates, normal, strict=True)
            ),
            budget=budget,
        ),
        constant(offset, cell.dimension, budget=budget),
        budget=budget,
    )


def _strict_side(
    cell: Cell,
    polynomial: Expression,
    shared: set[int],
    *,
    require_shared_support: bool = True,
    budget: CoordinateEnclosureBudget,
) -> bool:
    coefficients = bernstein_coefficients(
        polynomial, cell.domain, cell.dimension, budget=budget
    )
    if min(coefficients) < 0 or max(coefficients) <= 0:
        return False
    vertices = (
        cell.vertices
        if cell.kind != "pyramid"
        else (*cell.vertices[:4], *([cell.vertices[4]] * 4))
    )
    for vertex, point in zip(vertices, cell.reference_vertices, strict=True):
        value = expression_reference_evaluate(
            polynomial, point, cell.domain, budget=budget
        )
        if vertex in shared:
            if value:
                return False
        elif require_shared_support and value <= 0:
            return False
    return True


class _PlaneControls(NamedTuple):
    """Rational same-degree coordinate nets retained for one separating operation."""

    coordinates: tuple[tuple[int, ...], ...]
    denominator: int
    degree_support: tuple[tuple[tuple[int, ...], ...], ...]
    coefficient_bits: int


def _prepare_plane_controls(
    cell: Cell, *, budget: CoordinateEnclosureBudget
) -> _PlaneControls | None:
    # Rational controls are ratios, not linear coordinate nets. The coordinate
    # enclosure numerator/denominator reduction remains on the expression route.
    if any(isinstance(value, RationalPolynomial) for value in cell.coordinates):
        return None
    source_polynomials = tuple(
        value for value in cell.coordinates if not isinstance(value, RationalPolynomial)
    )
    if any(
        not isinstance(coefficient, Fraction)
        for value in source_polynomials
        for coefficient in value.values()
    ):
        return None
    # Integer plane controls apply only to the exact rational subfield.
    polynomials = cast(tuple[dict[tuple[int, ...], Fraction], ...], source_polynomials)

    def degree(index: tuple[int, ...]) -> tuple[int, ...]:
        return (
            (sum(index),)
            if cell.domain == "simplex"
            else (sum(index[:2]), index[2])
            if cell.domain == "prism"
            else index
        )

    degrees = tuple(
        tuple(
            max((degree(index)[axis] for index in value), default=0)
            for axis in range(len(degree((0,) * cell.dimension)))
        )
        for value in polynomials
    )
    if not degrees or len(set(degrees)) != 1 or max(degrees[0]) <= 1:
        return None
    indices = tuple(dict.fromkeys(index for value in polynomials for index in value))

    source_denominator = math.lcm(
        *(
            coefficient.denominator
            for value in polynomials
            for coefficient in value.values()
        )
    )
    source_bits = (
        max(
            (
                abs(coefficient.numerator).bit_length()
                for value in polynomials
                for coefficient in value.values()
            ),
            default=1,
        )
        + source_denominator.bit_length()
    )
    support_terms = sum(
        sum(degree(index)[axis] == maximum for index in indices)
        for axis, maximum in enumerate(degrees[0])
    ) * len(polynomials)
    coordinate_enclosure_owner.reserve_polynomial(
        support_terms, support_terms, cell.dimension, source_bits, budget=budget
    )
    support = tuple(
        tuple(
            tuple(
                0
                if index not in value
                else value[index].numerator
                * (source_denominator // value[index].denominator)
                for value in polynomials
            )
            for index in indices
            if degree(index)[axis] == maximum
        )
        for axis, maximum in enumerate(degrees[0])
    )
    controls = tuple(
        tuple(
            rational_coefficient(coefficient, role="mapped_embedding.plane_controls")
            for coefficient in bernstein_coefficients(
                value, cell.domain, cell.dimension, budget=budget
            )
        )
        for value in source_polynomials
    )
    denominator = math.lcm(
        *(value.denominator for column in controls for value in column)
    )
    bits = (
        max(
            (
                abs(value.numerator).bit_length()
                for column in controls
                for value in column
            ),
            default=1,
        )
        + denominator.bit_length()
    )
    count = sum(len(column) for column in controls)
    coordinate_enclosure_owner.reserve_polynomial(
        count, count, cell.dimension, bits, budget=budget
    )
    integers = tuple(
        tuple(value.numerator * (denominator // value.denominator) for value in column)
        for column in controls
    )
    return _PlaneControls(integers, denominator, support, bits)


def _plane_preserves_degree(
    prepared: _PlaneControls, normal: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> bool:

    budget.reserve(sum(len(group) for group in prepared.degree_support) * len(normal))
    # Coefficient cancellation can lower a coordinate's scalar Bernstein degree. Keep
    # that route unchanged rather than replacing it with an elevated net.
    denominator = math.lcm(*(Fraction(value).denominator for value in normal))
    integers = tuple(
        Fraction(value).numerator * (denominator // Fraction(value).denominator)
        for value in normal
    )
    return all(
        any(
            sum(weight * value for weight, value in zip(integers, row, strict=True))
            for row in group
        )
        for group in prepared.degree_support
    )


def _prepared_plane_side(
    cell: Cell,
    prepared: _PlaneControls,
    normal: np.ndarray,
    offset: Fraction,
    vertex_values: np.ndarray,
    shared: set[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, bool]:

    weights = tuple(Fraction(value) for value in normal)
    denominator = math.lcm(offset.denominator, *(value.denominator for value in weights))
    bits = (
        prepared.coefficient_bits
        + max(
            abs(offset.numerator).bit_length(),
            *(abs(value.numerator).bit_length() for value in weights),
        )
        + denominator.bit_length()
        + (len(weights) + 1).bit_length()
    )
    nodes = len(prepared.coordinates[0])
    coordinate_enclosure_owner.reserve_polynomial(
        nodes * (2 * len(weights) + 1), nodes, cell.dimension, bits, budget=budget
    )
    integers = tuple(
        value.numerator * (denominator // value.denominator) for value in weights
    )
    constant = (
        offset.numerator * (denominator // offset.denominator) * prepared.denominator
    )
    # Every control has the same strictly positive denominator. Integer signs
    # are precisely the signs of the original normalized Fraction controls.
    coefficients = tuple(
        constant
        + sum(weight * value for weight, value in zip(integers, row, strict=True))
        for row in zip(*prepared.coordinates, strict=True)
    )
    if min(coefficients) < 0 or max(coefficients) <= 0:
        return False, False
    vertices = (
        cell.vertices
        if cell.kind != "pyramid"
        else (*cell.vertices[:4], *([cell.vertices[4]] * 4))
    )
    minimal = True
    for vertex, value in zip(vertices, vertex_values, strict=True):
        if vertex in shared:
            if value:
                return False, False
        elif value <= 0:
            minimal = False
    return minimal, True


def _box_entity_diagonal(
    cell: Cell, reference: tuple[tuple[Fraction, ...], ...], keys: list[int], degree: int
) -> tuple[int, ...] | None:
    """Return the opposite corner after checking the entity's free-axis count."""
    corners = [reference[cell.vertices.index(key)] for key in keys]
    free = [
        axis
        for axis in range(cell.dimension)
        if len({corner[axis] for corner in corners}) > 1
    ]
    if len(free) != degree:
        return None
    return tuple(
        key
        for key, corner in zip(keys, corners, strict=True)
        if all(corner[axis] != corners[0][axis] for axis in free)
    )


type _EntityRow = tuple[int, int, tuple[int, ...]]
_PROPER_ENTITY_TABLES: dict[
    str,
    tuple[
        tuple[_EntityRow, ...],
        tuple[int, ...],
        dict[frozenset[int], _EntityRow],
    ],
] = {}


def _proper_entity_table(
    kind: str, /
) -> tuple[
    tuple[_EntityRow, ...], tuple[int, ...], dict[frozenset[int], _EntityRow]
]:
    """Static ``(degree, position, local vertices)`` rows of a kind's proper faces."""
    table = _PROPER_ENTITY_TABLES.get(kind)
    if table is None:
        topology = reference_cell_topology(kind)
        rows = tuple(
            (degree, position, tuple(int(vertex) for vertex in entity))
            for degree in range(1, topology.dimension)
            for position, entity in enumerate(topology.entities[degree])
        )
        table = (
            rows,
            tuple(sorted({len(entity) for _, _, entity in rows})),
            {frozenset(entity): row for row in rows for entity in (row[2],)},
        )
        _PROPER_ENTITY_TABLES[kind] = table
    return table


def _multi_affine_trace_equal(
    first: Cell, second: Cell, shared: list[int], *, budget: CoordinateEnclosureBudget
) -> bool | None:
    """Compare multi-affine box-cell traces on every shared entity, or return None.

    A multi-affine map restricted to a box edge or square face is the interpolant
    of its corner images. When every common entity of ``first`` is an entity of
    ``second`` whose square face pairs the same diagonal vertices, the full
    restrictions agree exactly when the shared corner images agree. Any other
    common entity is left to the generic restriction proof.
    """
    if (
        first.corner_images is None
        or second.corner_images is None
        or first.kind != second.kind
        or first.dimension != second.dimension
    ):
        return None

    first_reference, second_reference = (
        first.reference_vertices,
        second.reference_vertices,
    )
    entities, sizes, lookup = _proper_entity_table(first.kind)
    # Only entities whose every vertex is shared can need a trace check. Probe
    # those vertex subsets directly when that is cheaper than scanning every
    # entity; both visit the same entities in degree and topology order.
    local = sorted(first.vertices.index(key) for key in shared)
    probes = sum(math.comb(len(local), size) for size in sizes)
    if probes < len(entities):
        budget.reserve(
            len(shared) * (len(first.coordinates) + 2 * first.dimension)
            + sum(math.comb(len(local), size) * size for size in sizes)
        )
        entities = tuple(
            sorted(
                lookup[subset]
                for size in sizes
                for subset in map(frozenset, combinations(local, size))
                if subset in lookup
            )
        )
    else:
        budget.reserve(
            len(shared) * (len(first.coordinates) + 2 * first.dimension)
            + sum(len(entity) for _, _, entity in entities)
        )
    for degree, _, entity in entities:
        keys = sorted(first.vertices[i] for i in entity)
        if not set(keys).issubset(shared):
            continue
        partners = []
        for cell, reference in ((first, first_reference), (second, second_reference)):
            partner = _box_entity_diagonal(cell, reference, keys, degree)
            if partner is None:
                return None
            partners.append(partner)
        if degree == 2 and (partners[0] != partners[1] or len(partners[0]) != 1):
            return None
    for key in shared:
        if (
            first.corner_images[first.vertices.index(key)]
            != second.corner_images[second.vertices.index(key)]
        ):
            return False
    return True


def _multi_affine_box(cell: Cell, /) -> bool:
    """Whether ``cell`` carries the rational corner images of a multi-affine box map."""
    return (
        cell.corner_images is not None
        and cell.domain == "box"
        and cell.kind != "pyramid"
    )


def _corner_plane_separates(
    first: Cell,
    second: Cell,
    shared: set[int],
    anchor: int | None,
    left: list[ExactScalar],
    right: list[ExactScalar],
) -> bool:
    """Accept one signed plane on the corner values of two multi-affine box cells.

    ``left``/``right`` are the signed normal projections of the rational corner
    images. The plane passes through a shared vertex, or the midpoint gap when
    none is shared, and both sides need the generic shared-support conditions.
    """
    offset = (
        left[first.vertices.index(anchor)]
        if anchor is not None
        else (max(left) + min(right)) / 2
    )
    below = [offset - value for value in left]
    above = [value - offset for value in right]
    if any(value < 0 for value in below) or any(value < 0 for value in above):
        return False

    def side(values: list[ExactScalar], vertices: tuple[int, ...], strict: bool) -> bool:
        if not any(values):
            return False
        for value, vertex in zip(values, vertices, strict=True):
            if vertex in shared:
                if value:
                    return False
            elif strict and value <= 0:
                return False
        return True

    return (
        side(below, first.vertices, True) and side(above, second.vertices, False)
    ) or (side(below, first.vertices, False) and side(above, second.vertices, True))


def _multi_affine_candidate_separation(
    first: Cell,
    second: Cell,
    shared: set[int],
    candidates: _PlaneCandidates,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Decide the ordered candidate normals of multi-affine box cells on corners.

    A signed plane value of a multi-affine map is multi-affine, so its Bernstein
    controls are exactly its corner values: accepting a candidate on rational
    corner images is the source-control acceptance of
    ``_candidate_plane_separates``. Base normals precede their pairwise sums and
    differences, as in the integer route.
    """
    if first.corner_images is None or second.corner_images is None:
        raise ValueError(
            "mapped_embedding.multi_affine_candidates must have rational corner images; got None"
        )
    a = np.asarray(first.corner_images, dtype=object)
    b = np.asarray(second.corner_images, dtype=object)
    ambient = a.shape[1]
    anchor = next(iter(shared), None)
    base = candidates.normals
    tested: set[tuple[Fraction, ...]] = set()

    def separates(normals: Iterable[np.ndarray]) -> bool:
        for normal in normals:
            if not np.any(normal):
                continue
            with budget.temporary_scope():
                key: tuple[Fraction, ...] | None = None
                if all(isinstance(value, Fraction) for value in normal):
                    # Match the canonical plane owner's unit-first-component
                    # convention. Fraction division includes its exact gcd
                    # reduction; reserve its operand-derived work and bit bound.
                    values = tuple(cast(Fraction, value) for value in normal)
                    planar_primitives_owner.reserve_fraction_work(
                        (values,), 3 * ambient, ambient, 4, budget=budget
                    )
                    pivot = next(value for value in values if value)
                    key = tuple(value / pivot for value in values)
                    if key in tested:
                        continue
                budget.reserve((ambient + 4) * (len(a) + len(b)) + ambient)
                projected_a, projected_b = list(a @ normal), list(b @ normal)
                for sign in (1, -1):
                    if _corner_plane_separates(
                        first,
                        second,
                        shared,
                        anchor,
                        [sign * value for value in projected_a],
                        [sign * value for value in projected_b],
                    ):
                        return True
                if key is not None:
                    # Only a completely tested plane is reusable. Positive
                    # scaling preserves every support sign and zero; negative
                    # scaling exchanges the two orientations just tested.
                    storage.set_bound(
                        storage.bound + budget.live_object_storage_upper(key) + 128
                    )
                    tested.add(key)
        return False

    with budget.live_storage() as storage:
        storage.set_bound(256)
        # Keep the original base vectors: scaling a base vector before sums
        # or differences would change the family of proposed planes.
        return separates(chain(
            base,
            (x + y for index, x in enumerate(base) for y in base[index + 1 :]),
            (x - y for index, x in enumerate(base) for y in base[index + 1 :]),
        ))


def _same_entity_trace(
    first: Cell,
    second: Cell,
    ordered: list[int],
    degree: int,
    a: np.ndarray,
    b: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Compare one shared entity in an independent common reference chart."""
    if degree == 0:
        left = tuple(
            expression_reference_evaluate(
                value,
                rational_vector(a[0]),
                first.domain,
                budget=budget,
            )
            for value in first.coordinates
        )
        right = tuple(
            expression_reference_evaluate(
                value,
                rational_vector(b[0]),
                second.domain,
                budget=budget,
            )
            for value in second.coordinates
        )
        if left != right:
            return False
    else:
        if len(ordered) not in (2, 3, 4):
            return True
        kind = (
            "interval"
            if degree == 1
            else "triangle"
            if len(ordered) == 3
            else "quadrilateral"
        )
        # Select an independent corner basis; the sorted vertex order is shared
        # by both restrictions, never selected by physical proximity.
        columns = (
            (1,)
            if degree == 1
            else next(
                (
                    pair
                    for pair in ((1, 2), (1, 3), (2, 3))
                    if max(pair) < len(ordered)
                    and np.linalg.matrix_rank((a[list(pair)] - a[0]).T) == degree
                    and np.linalg.matrix_rank((b[list(pair)] - b[0]).T) == degree
                ),
                (),
            )
        )
        if not columns:
            return False
        left_expressions = restrict_chart_expressions(
            first.coordinates,
            first.kind,
            kind,
            a[0],
            (a[list(columns)] - a[0]).T,
            budget=budget,
        )
        right_expressions = restrict_chart_expressions(
            second.coordinates,
            second.kind,
            kind,
            b[0],
            (b[list(columns)] - b[0]).T,
            budget=budget,
        )
        if not expression_sequence_equal(
            left_expressions, right_expressions, budget=budget
        ):
            return False
    return True


@checked
def cell_traces_equal(
    first: Cell, second: Cell, *, budget: CoordinateEnclosureBudget
) -> bool:
    shared = sorted(set(first.vertices).intersection(second.vertices))
    if not shared:
        return True
    decided = _multi_affine_trace_equal(first, second, shared, budget=budget)
    if decided is not None:
        return decided
    first_reference = reference_cell_topology(first.kind)
    second_reference = reference_cell_topology(second.kind)
    # Every shared reference-topology subentity, including vertex-only contacts, must
    # have the same full source expression in the same oriented parameter chart.
    for degree in range(first.dimension):
        entities = first_reference.entities[degree]
        for entity in entities:
            keys = [first.vertices[i] for i in entity]
            if not set(keys).issubset(shared):
                continue
            ordered = sorted(keys)
            a = np.asarray(first_reference.vertices, dtype=np.float64)[
                [first.vertices.index(key) for key in ordered]
            ]
            b = np.asarray(second_reference.vertices, dtype=np.float64)[
                [second.vertices.index(key) for key in ordered]
            ]
            if not _same_entity_trace(
                first, second, ordered, degree, a, b, budget=budget
            ):
                return False
    return True


@final
class _PlaneCandidates(StrictModule, NonTrainableState):
    first: np.ndarray
    second: np.ndarray
    normals: list[np.ndarray]


@final
class _IntegerPlaneCandidates(StrictModule, NonTrainableState):
    first: np.ndarray
    second: np.ndarray
    normals: list[np.ndarray]
    coordinate_denominator: int = eqx.field(static=True)
    bits: int = eqx.field(static=True)


def _plane_corner_images(
    cell: Cell, /, *, budget: CoordinateEnclosureBudget
) -> np.ndarray:
    """Evaluate rational reference corners in reference-vertex and coordinate order.

    Multi-affine box cells carry these exact images already.
    """
    if cell.corner_images is not None:
        budget.reserve(len(cell.corner_images) * len(cell.coordinates))
        return np.asarray(cell.corner_images, dtype=object)
    return np.asarray(
        [
            [
                expression_reference_evaluate(value, point, cell.domain, budget=budget)
                for value in cell.coordinates
            ]
            for point in cell.reference_vertices
        ],
        dtype=object,
    )


def _plane_candidates(
    first: Cell, second: Cell, shared: set[int], *, budget: CoordinateEnclosureBudget
) -> _PlaneCandidates:
    """Prepare rational corner images and the deterministic candidate normal order."""
    a = _plane_corner_images(first, budget=budget)
    b = _plane_corner_images(second, budget=budget)
    if any(isinstance(value, RadicalElement) for array in (a, b) for value in array.flat):
        reserve_algebraic_work((*a.flat, *b.flat), 4096, 2048, 4, budget=budget)
    ambient = a.shape[1]
    normals = [
        np.sum(b, axis=0) / len(b) - np.sum(a, axis=0) / len(a),
        *np.asarray(
            [[Fraction(i == j) for j in range(ambient)] for i in range(ambient)],
            dtype=object,
        ),
    ]

    def cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ),
            dtype=object,
        )

    if ambient == 2:
        normals.extend(
            np.asarray((-(q - p)[1], (q - p)[0]))
            for corners in (a, b)
            for p, q in zip(corners, np.roll(corners, -1, axis=0), strict=True)
        )
    elif ambient == 3:
        for first_vertex, second_vertex in combinations(sorted(shared), 2):
            tangent = (
                a[first.vertices.index(second_vertex)]
                - a[first.vertices.index(first_vertex)]
            )
            normals.extend(
                cross(tangent, axis)
                for axis in np.asarray(
                    [[Fraction(i == j) for j in range(3)] for i in range(3)], dtype=object
                )
            )
        topology = reference_cell_topology(first.kind)
        normals.extend(
            cross(a[face[1]] - a[face[0]], a[face[2]] - a[face[0]])
            for face in topology.entities[2]
            if len(face) >= 3
        )
        topology = reference_cell_topology(second.kind)
        normals.extend(
            cross(b[face[1]] - b[face[0]], b[face[2]] - b[face[0]])
            for face in topology.entities[2]
            if len(face) >= 3
        )
        if first.kind == second.kind == "triangle":
            normal_a = cross(a[1] - a[0], a[2] - a[0])
            normal_b = cross(b[1] - b[0], b[2] - b[0])
            tangent_normal = (
                normal_a + normal_b if normal_a @ normal_b >= 0 else normal_a - normal_b
            )
            edge_normals = [
                cross(tangent_normal, q - p)
                for corners in (a, b)
                for p, q in zip(corners, np.roll(corners, -1, axis=0), strict=True)
            ]
            normals.extend(edge_normals)
            normals.extend(x + y for x in edge_normals for y in edge_normals)
            normals.extend(x - y for x in edge_normals for y in edge_normals)
    return _PlaneCandidates(a, b, normals)


def _integer_plane_candidates(
    first: Cell, candidates: _PlaneCandidates, *, budget: CoordinateEnclosureBudget
) -> _IntegerPlaneCandidates:
    """Lift base candidates by common positive scales; cone sums stay pending."""
    a, b, normals = candidates.first, candidates.second, candidates.normals
    ambient = a.shape[1]
    budget.reserve(3 * len(normals) * ambient + 2 * (a.size + b.size))
    normal_denominator = math.lcm(
        *(Fraction(value).denominator for normal in normals for value in normal)
    )
    coordinate_denominator = math.lcm(
        *(value.denominator for corners in (a, b) for row in corners for value in row)
    )
    normal_bits = (
        max(
            abs(Fraction(value).numerator).bit_length()
            for normal in normals
            for value in normal
        )
        + normal_denominator.bit_length()
        + 2
    )
    coordinate_bits = (
        max(
            abs(value.numerator).bit_length()
            for corners in (a, b)
            for row in corners
            for value in row
        )
        + coordinate_denominator.bit_length()
    )
    bits = max(normal_bits, coordinate_bits)
    # The base normals and both corner banks; their pairwise cone sums and
    # differences are formed only after every base normal declined.
    count = len(normals) * ambient + a.size + b.size
    coordinate_enclosure_owner.reserve_polynomial(
        count, count, first.dimension, bits, budget=budget
    )
    # One common positive scale preserves every base vector AND all cone
    # sums/differences. Independently normalizing each vector would not.
    base_normals = tuple(
        np.asarray(
            [
                Fraction(value).numerator
                * (normal_denominator // Fraction(value).denominator)
                for value in normal
            ],
            dtype=object,
        )
        for normal in normals
    )
    normals = list(base_normals)
    a = np.asarray(
        [
            [
                value.numerator * (coordinate_denominator // value.denominator)
                for value in row
            ]
            for row in a
        ],
        dtype=object,
    )
    b = np.asarray(
        [
            [
                value.numerator * (coordinate_denominator // value.denominator)
                for value in row
            ]
            for row in b
        ],
        dtype=object,
    )
    return _IntegerPlaneCandidates(a, b, normals, coordinate_denominator, bits)


def _integer_plane_combinations(
    first: Cell, candidates: _IntegerPlaneCandidates, *, budget: CoordinateEnclosureBudget
) -> list[np.ndarray]:
    """Form the ordered cone sums, then differences, of the lifted base normals.

    Every pair shares the common positive scale of its base normals, so each
    sum and difference is the same direction as its rational counterpart.
    """
    base_normals = candidates.normals
    ambient = candidates.first.shape[1]
    count = (len(base_normals) ** 2 - len(base_normals)) * ambient
    coordinate_enclosure_owner.reserve_polynomial(
        count, count, first.dimension, candidates.bits, budget=budget
    )
    normals = [
        x + y for index, x in enumerate(base_normals) for y in base_normals[index + 1 :]
    ]
    normals.extend(
        x - y for index, x in enumerate(base_normals) for y in base_normals[index + 1 :]
    )
    return normals


def _source_plane_separates(
    first: Cell,
    second: Cell,
    shared: set[int],
    normal: np.ndarray,
    offset: ExactScalar,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove a separating plane on all source-control coefficients."""
    pa = scale(_signed_image(first, normal, offset, budget=budget), -1, budget=budget)
    pb = _signed_image(second, normal, offset, budget=budget)
    return (
        _strict_side(first, pa, shared, budget=budget)
        and _strict_side(second, pb, shared, require_shared_support=False, budget=budget)
    ) or (
        _strict_side(first, pa, shared, require_shared_support=False, budget=budget)
        and _strict_side(second, pb, shared, budget=budget)
    )


def _algebraic_candidate_family_separates(
    first: Cell,
    second: Cell,
    shared: set[int],
    candidates: _PlaneCandidates,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Test ordered-field planes without converting source coefficients to integers."""
    a, b = candidates.first, candidates.second
    for normal in candidates.normals:
        if not np.any(normal):
            continue
        reserve_algebraic_work(
            (*a.flat, *b.flat, *normal.flat), 1024, 512, 3, budget=budget
        )
        projected_a, projected_b = a @ normal, b @ normal
        for sign in (1, -1):
            n = sign * normal
            left, right = sign * projected_a, sign * projected_b
            if shared:
                anchor = a[first.vertices.index(next(iter(shared)))]
                offset = -sum(
                    weight * value for weight, value in zip(n, anchor, strict=True)
                )
            else:
                offset = -(max(left) + min(right)) / 2
            if any(value < 0 for value in -left - offset) or any(
                value < 0 for value in right + offset
            ):
                continue
            if _source_plane_separates(first, second, shared, n, offset, budget=budget):
                return True
    return False


def _candidate_plane_separates(
    first: Cell,
    second: Cell,
    shared: set[int],
    normal: np.ndarray,
    candidates: _IntegerPlaneCandidates,
    prepared_a: _PlaneControls | None,
    prepared_b: _PlaneControls | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Certify both signs of a candidate against full coordinate control nets."""
    a, b = candidates.first, candidates.second
    ambient = a.shape[1]
    coordinate_denominator = candidates.coordinate_denominator
    budget.reserve(2 * (a.size + b.size))
    projected_a, projected_b = a @ (2 * normal), b @ (2 * normal)
    for sign in (1, -1):
        n = sign * 2 * normal
        left, right = sign * projected_a, sign * projected_b
        budget.reserve(ambient + 2 * (len(left) + len(right)))
        if shared:
            vertex = next(iter(shared))
            anchor = a[first.vertices.index(vertex)]
            integer_offset = -sum(
                weight * value for weight, value in zip(n, anchor, strict=True)
            )
        else:
            # The factor two makes the rational midpoint an integer.
            integer_offset = -(max(left) + min(right)) // 2
        corners_a, corners_b = -left - integer_offset, right + integer_offset
        # This is a rational-coordinate rejection test. Every plane
        # that passes still needs its full source-control proof.
        if any(value < 0 for value in corners_a) or any(value < 0 for value in corners_b):
            continue
        offset = Fraction(integer_offset, coordinate_denominator)
        prepared = (
            prepared_a is not None
            and prepared_b is not None
            and _plane_preserves_degree(prepared_a, normal, budget=budget)
            and _plane_preserves_degree(prepared_b, normal, budget=budget)
        )
        if prepared and prepared_a is not None and prepared_b is not None:
            minimal_a, permissive_a = _prepared_plane_side(
                first,
                prepared_a,
                -n,
                -offset,
                corners_a,
                shared,
                budget=budget,
            )
            minimal_b, permissive_b = _prepared_plane_side(
                second,
                prepared_b,
                n,
                offset,
                corners_b,
                shared,
                budget=budget,
            )
            if (minimal_a and permissive_b) or (permissive_a and minimal_b):
                return True
            continue
        if _source_plane_separates(first, second, shared, n, offset, budget=budget):
            return True
    return False


def _separating_candidate_family(
    first: Cell,
    second: Cell,
    shared: set[int],
    candidates: _PlaneCandidates,
    prepared_controls: tuple[_PlaneControls | None, _PlaneControls | None] | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Authenticate candidate normals against the full source-control nets."""
    if any(
        isinstance(value, RadicalElement)
        for array in (candidates.first, candidates.second, *candidates.normals)
        for value in array.flat
    ):
        with budget.temporary_scope():
            return _algebraic_candidate_family_separates(
                first, second, shared, candidates, budget=budget
            )
    if _multi_affine_box(first) and _multi_affine_box(second):
        with budget.temporary_scope():
            return _multi_affine_candidate_separation(
                first, second, shared, candidates, budget=budget
            )
    with budget.temporary_scope():
        integers = _integer_plane_candidates(first, candidates, budget=budget)
        prepared_a, prepared_b = (
            (
                _prepare_plane_controls(first, budget=budget),
                _prepare_plane_controls(second, budget=budget),
            )
            if prepared_controls is None
            else prepared_controls
        )

        def separates(normals: list[np.ndarray]) -> bool:
            for normal in normals:
                with budget.temporary_scope():
                    if not np.any(normal):
                        continue
                    if _candidate_plane_separates(
                        first,
                        second,
                        shared,
                        normal,
                        integers,
                        prepared_a,
                        prepared_b,
                        budget=budget,
                    ):
                        return True
            return False

        # The deterministic order is every base normal, then their cone sums
        # and differences; later candidates are only formed when needed.
        if separates(integers.normals):
            return True
        return separates(_integer_plane_combinations(first, integers, budget=budget))


def _surface_edge_candidates(
    first: Cell,
    second: Cell,
    candidates: _PlaneCandidates,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PlaneCandidates | None:
    """Propose rational surface separating axes without assuming affine maps.

    Edge cross-products supply axes between different support planes; crossed
    tangent normals supply in-plane axes for coplanar patches. Corner hulls
    only propose normals: source controls and shared zero sets own acceptance.
    """
    if first.dimension != 2 or second.dimension != 2 or candidates.first.shape[1] != 3:
        return None
    first_edges = reference_cell_topology(first.kind).entities[1]
    second_edges = reference_cell_topology(second.kind).entities[1]
    budget.reserve(
        12
        * (
            len(first_edges) * len(second_edges)
            + len(first_edges)
            + len(second_edges)
            + 3
        )
    )
    normals = list(candidates.normals)
    a, b = candidates.first, candidates.second
    normal_a = np.asarray(
        rational_cross(tuple(a[1] - a[0]), tuple(a[2] - a[0])), dtype=object
    )
    normal_b = np.asarray(
        rational_cross(tuple(b[1] - b[0]), tuple(b[2] - b[0])), dtype=object
    )
    tangent_normal = (
        normal_a + normal_b if normal_a @ normal_b >= 0 else normal_a - normal_b
    )
    for corners, edges in ((a, first_edges), (b, second_edges)):
        for edge in edges:
            direction = corners[edge[1]] - corners[edge[0]]
            normals.append(
                np.asarray(
                    rational_cross(tuple(tangent_normal), tuple(direction)), dtype=object
                )
            )
    for first_edge in first_edges:
        a = candidates.first[first_edge[1]] - candidates.first[first_edge[0]]
        for second_edge in second_edges:
            b = candidates.second[second_edge[1]] - candidates.second[second_edge[0]]
            normals.append(np.asarray(rational_cross(tuple(a), tuple(b)), dtype=object))
    return _PlaneCandidates(candidates.first, candidates.second, normals)


@checked
def separating_plane(
    first: Cell,
    second: Cell,
    prepared_controls: tuple[_PlaneControls | None, _PlaneControls | None] | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove full mapped-image separation by a certified supporting plane."""
    shared = set(first.vertices).intersection(second.vertices)
    candidates = _plane_candidates(first, second, shared, budget=budget)
    if _separating_candidate_family(
        first, second, shared, candidates, prepared_controls, budget=budget
    ):
        return True
    surface_candidates = _surface_edge_candidates(
        first, second, candidates, budget=budget
    )
    return surface_candidates is not None and _separating_candidate_family(
        first, second, shared, surface_candidates, prepared_controls, budget=budget
    )


def _inside(point: np.ndarray, cell: Cell, margin: float = 0.0) -> bool:
    if np.any(point <= margin) or np.any(point >= 1.0 - margin):
        return False
    if cell.domain == "simplex":
        return bool(np.sum(point) < 1.0 - margin)
    if cell.domain == "prism":
        return bool(np.sum(point[:2]) < 1.0 - margin)
    return True


def _overlap_witness(
    first: Cell,
    second: Cell,
    source: np.ndarray | None = None,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    if (
        len(first.coordinates) != first.dimension
        or len(second.coordinates) != second.dimension
    ):
        return False
    ambient = second.dimension
    if source is None:
        source = np.full(
            (first.dimension,),
            1.0 / (first.dimension + 1) if first.domain == "simplex" else 0.25,
            dtype=np.float64,
        )
    if not _inside(source, first):
        return False
    target = tuple(
        evaluate(value, rational_vector(source), budget=budget)
        for value in first.coordinates
    )
    point = np.full(
        (ambient,),
        1.0 / (ambient + 1) if second.domain == "simplex" else 0.5,
        dtype=np.float64,
    )
    jacobian = tuple(
        tuple(derivative(value, axis, budget=budget) for axis in range(ambient))
        for value in second.coordinates
    )
    for _ in range(_CONTACT_NEWTON_STEPS):
        center = rational_vector(point)
        residual = np.asarray(
            [
                float(evaluate(value, center, budget=budget) - q)
                for value, q in zip(second.coordinates, target, strict=True)
            ],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [
                [float(evaluate(entry, center, budget=budget)) for entry in row]
                for row in jacobian
            ],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(matrix) != ambient:
            return False
        step = np.linalg.solve(matrix, residual)
        point -= step
        if np.max(np.abs(step)) < _CONTACT_CANDIDATE_TOLERANCE:
            break
    if not _inside(point, second):
        return False
    radius = (
        min(
            float(np.min(point)),
            float(np.min(1.0 - point)),
            (1.0 - float(np.sum(point))) / ambient
            if second.domain == "simplex"
            else (1.0 - float(np.sum(point[:2]))) / 2.0
            if second.domain == "prism"
            else 1.0,
        )
        * _CONTACT_INTERIOR_RADIUS_FRACTION
    )
    if radius <= 0:
        return False
    center = rational_vector(point)
    midpoint = np.asarray(
        [
            [float(evaluate(entry, center, budget=budget)) for entry in row]
            for row in jacobian
        ],
        dtype=np.float64,
    )
    preconditioner = np.linalg.solve(midpoint, np.eye(ambient, dtype=np.float64))
    arguments = ArgumentComposition(
        affine_arguments(
            point - radius,
            np.eye(ambient, dtype=np.float64) * (2.0 * radius),
            budget=budget,
        ),
        budget=budget,
    )
    residual_polynomials = tuple(
        add(value, constant(-q, ambient, budget=budget), budget=budget)
        for value, q in zip(second.coordinates, target, strict=True)
    )
    for i in range(ambient):
        transformed = sum_polynomials(
            tuple(
                scale(value, Fraction(float(preconditioner[i, k])), budget=budget)
                for k, value in enumerate(residual_polynomials)
            ),
            budget=budget,
        )
        displacement = abs(evaluate(transformed, center, budget=budget))
        gain = Fraction(0)
        for j in range(ambient):
            entry = sum_polynomials(
                tuple(
                    scale(
                        jacobian[k][j],
                        Fraction(float(preconditioner[i, k])),
                        budget=budget,
                    )
                    for k in range(ambient)
                ),
                budget=budget,
            )
            remainder = add(
                constant(int(i == j), ambient, budget=budget),
                scale(entry, -1, budget=budget),
                budget=budget,
            )
            coefficients = bernstein_coefficients(
                arguments(remainder, budget=budget), "box", ambient, budget=budget
            )
            gain += max(abs(value) for value in coefficients)
        if displacement + gain * Fraction(radius) >= Fraction(radius):
            return False
    return True


# Candidate iteration tolerances choose probes only; strict inclusion proves contact.
_CONTACT_NEWTON_STEPS = 20
_CONTACT_CANDIDATE_TOLERANCE = 1.0e-12
# One eighth of the nearest source-boundary margin leaves a strict interior cube.
_CONTACT_INTERIOR_RADIUS_FRACTION = 0.125
_CONTACT_INTERIOR_PROBES = (0.1, 0.2, 0.3, 0.4)
# Outward image bounds retain two interval vectors and coordinate-expression workspace.
_CONTACT_BOUND_BASE_BYTES = 2048
_CONTACT_BOUND_COORDINATE_BYTES = 128


@final
class _CrossingCandidate(StrictModule, NonTrainableState):
    first: np.ndarray
    second: np.ndarray
    jacobian: np.ndarray


@final
class _CrossingSystem(StrictModule, NonTrainableState):
    difference: tuple[Expression, ...]
    radius: float = eqx.field(static=True)


def _crossing_candidate(
    first: Cell, second: Cell, ambient: int, *, budget: CoordinateEnclosureBudget
) -> _CrossingCandidate | None:
    """Locate a rounded contact candidate; certification follows in exact arithmetic."""
    point_a = np.full(
        (first.dimension,),
        1.0 / (first.dimension + 1) if first.domain == "simplex" else 0.5,
        dtype=np.float64,
    )
    point_b = np.full(
        (second.dimension,),
        1.0 / (second.dimension + 1) if second.domain == "simplex" else 0.5,
        dtype=np.float64,
    )
    derivatives_a = tuple(
        tuple(derivative(value, axis, budget=budget) for axis in range(first.dimension))
        for value in first.coordinates
    )
    derivatives_b = tuple(
        tuple(derivative(value, axis, budget=budget) for axis in range(second.dimension))
        for value in second.coordinates
    )
    for _ in range(_CONTACT_NEWTON_STEPS):
        a = rational_vector(point_a)
        b = rational_vector(point_b)
        residual = np.asarray(
            [
                float(evaluate(x, a, budget=budget) - evaluate(y, b, budget=budget))
                for x, y in zip(first.coordinates, second.coordinates, strict=True)
            ],
            dtype=np.float64,
        )
        matrix = np.asarray(
            [
                [float(evaluate(entry, a, budget=budget)) for entry in row_a]
                + [-float(evaluate(entry, b, budget=budget)) for entry in row_b]
                for row_a, row_b in zip(derivatives_a, derivatives_b, strict=True)
            ],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(matrix) != ambient:
            return None
        update, _, rank, _ = np.linalg.lstsq(matrix, residual, rcond=None)
        if rank != ambient:
            return None
        point_a -= update[: first.dimension]
        point_b -= update[first.dimension :]
        if np.max(np.abs(update)) < _CONTACT_CANDIDATE_TOLERANCE:
            break
    return _CrossingCandidate(point_a, point_b, matrix)


def _crossing_system(
    first: Cell,
    second: Cell,
    candidate: _CrossingCandidate,
    ambient: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _CrossingSystem | None:
    """Restrict independent parameters to a cube strictly inside both source cells."""
    point_a, point_b, matrix = candidate.first, candidate.second, candidate.jacobian
    if not _inside(point_a, first) or not _inside(point_b, second):
        return None
    selected = next(
        (
            columns
            for columns in combinations(range(matrix.shape[1]), ambient)
            if np.linalg.matrix_rank(matrix[:, columns]) == ambient
        ),
        None,
    )
    if selected is None:
        return None
    margins = [
        float(np.min(point_a)),
        float(np.min(1.0 - point_a)),
        float(np.min(point_b)),
        float(np.min(1.0 - point_b)),
    ]
    for point, cell in ((point_a, first), (point_b, second)):
        if cell.domain == "simplex":
            margins.append((1.0 - float(np.sum(point))) / cell.dimension)
        elif cell.domain == "prism":
            margins.append((1.0 - float(np.sum(point[:2]))) / 2.0)
    radius = min(margins) * _CONTACT_INTERIOR_RADIUS_FRACTION
    variables = coordinate_enclosure_owner.axes(ambient, budget=budget)
    parameters = np.concatenate((point_a, point_b))

    arguments = tuple(
        coordinate_enclosure_owner.add(
            constant(Fraction(float(value)), ambient, budget=budget),
            variables[selected.index(i)] if i in selected else {},
            budget=budget,
        )
        for i, value in enumerate(parameters)
    )
    difference = tuple(
        add(
            compose(a, arguments[: first.dimension], budget=budget),
            scale(
                compose(b, arguments[first.dimension :], budget=budget), -1, budget=budget
            ),
            budget=budget,
        )
        for a, b in zip(first.coordinates, second.coordinates, strict=True)
    )
    return _CrossingSystem(difference, radius)


def _crossing_inclusion(
    system: _CrossingSystem, ambient: int, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Prove a strict Krawczyk inclusion for the square contact system."""
    difference, radius = system.difference, system.radius
    origin = (Fraction(0),) * ambient
    jacobian = tuple(
        tuple(derivative(value, axis, budget=budget) for axis in range(ambient))
        for value in difference
    )
    midpoint = np.asarray(
        [
            [float(evaluate(entry, origin, budget=budget)) for entry in row]
            for row in jacobian
        ],
        dtype=np.float64,
    )
    if np.linalg.matrix_rank(midpoint) != ambient:
        return False
    preconditioner = np.linalg.solve(midpoint, np.eye(ambient, dtype=np.float64))
    cube = ArgumentComposition(
        affine_arguments(
            np.full((ambient,), -radius, dtype=np.float64),
            np.eye(ambient, dtype=np.float64) * (2.0 * radius),
            budget=budget,
        ),
        budget=budget,
    )
    for i in range(ambient):
        residual = sum_polynomials(
            tuple(
                scale(value, Fraction(float(preconditioner[i, k])), budget=budget)
                for k, value in enumerate(difference)
            ),
            budget=budget,
        )
        displacement = abs(evaluate(residual, origin, budget=budget))
        gain = Fraction(0)
        for j in range(ambient):
            entry = sum_polynomials(
                tuple(
                    scale(
                        jacobian[k][j],
                        Fraction(float(preconditioner[i, k])),
                        budget=budget,
                    )
                    for k in range(ambient)
                ),
                budget=budget,
            )
            remainder = add(
                constant(int(i == j), ambient, budget=budget),
                scale(entry, -1, budget=budget),
                budget=budget,
            )
            coefficients = bernstein_coefficients(
                cube(remainder, budget=budget), "box", ambient, budget=budget
            )
            gain += max(abs(value) for value in coefficients)
        if displacement + gain * Fraction(radius) >= Fraction(radius):
            return False
    return True


def _crossing_witness(
    first: Cell, second: Cell, *, budget: CoordinateEnclosureBudget
) -> bool:
    """An interval-Newton inclusion witnesses an interior transverse contact."""

    ambient = len(first.coordinates)
    if ambient > first.dimension + second.dimension or ambient != len(second.coordinates):
        return False
    candidate = _crossing_candidate(first, second, ambient, budget=budget)
    if candidate is None:
        return False
    system = _crossing_system(first, second, candidate, ambient, budget=budget)
    return system is not None and _crossing_inclusion(system, ambient, budget=budget)


@final
class _TriangleAtlas(StrictModule, NonTrainableState):
    projection: np.ndarray
    transforms: list[
        tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]]
    ]


def _projected_corner_images(
    cell: Cell, projection: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> tuple[tuple[ExactScalar, ...], ...]:
    """Project rational source corner images using fixed rounded projection weights."""
    corners = [
        tuple(
            expression_reference_evaluate(value, point, cell.domain, budget=budget)
            for value in cell.coordinates
        )
        for point in cell.reference_vertices
    ]
    if any(isinstance(value, RadicalElement) for point in corners for value in point):
        work = len(corners) * projection.shape[0] * projection.shape[1]
        reserve_algebraic_work(corners, work, work, 2, budget=budget)
    return tuple(
        tuple(
            sum(
                (
                    Fraction(float(weight)) * value
                    for weight, value in zip(row, point, strict=True)
                ),
                Fraction(0),
            )
            for row in projection
        )
        for point in corners
    )


def _triangle_atlas(
    first: Cell, second: Cell, shared: set[int], *, budget: CoordinateEnclosureBudget
) -> _TriangleAtlas | None:
    """Prepare two compatible triangle charts whose union is convex."""
    midpoint = jacobian_midpoint(first, budget=budget)
    if midpoint is None:
        return None
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    if rank != 2:
        return None
    projected = [
        _projected_corner_images(cell, projection, budget=budget)
        for cell in (first, second)
    ]
    if any(
        isinstance(value, RadicalElement)
        for corners in projected
        for point in corners
        for value in point
    ):
        reserve_algebraic_work((*projected[0], *projected[1]), 64, 64, 4, budget=budget)
    identifiers = sorted(shared)
    a, b = (projected[0][first.vertices.index(identifier)] for identifier in identifiers)
    c = projected[0][
        next(i for i, identifier in enumerate(first.vertices) if identifier not in shared)
    ]
    d = projected[1][
        next(
            i for i, identifier in enumerate(second.vertices) if identifier not in shared
        )
    ]
    for identifier in identifiers:
        if (
            projected[0][first.vertices.index(identifier)]
            != projected[1][second.vertices.index(identifier)]
        ):
            return None

    if not convex_triangle_union(a, b, c, d, budget=budget):
        return None
    transforms = []
    for corners in projected:
        matrix = tuple(
            tuple(corners[j + 1][i] - corners[0][i] for j in range(2)) for i in range(2)
        )
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if not determinant:
            return None
        transforms.append(
            (
                (matrix[1][1] / determinant, -matrix[0][1] / determinant),
                (-matrix[1][0] / determinant, matrix[0][0] / determinant),
            )
        )
    return _TriangleAtlas(projection, transforms)


def _adjacent_triangle_injectivity(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    """Prove physical projection injectivity on a checked convex two-cell chart."""
    if first.kind != "triangle" or second.kind != "triangle":
        return False, 0
    shared = set(first.vertices).intersection(second.vertices)
    if len(shared) != 2:
        return False, 0
    atlas = _triangle_atlas(first, second, shared, budget=budget)
    if atlas is None:
        return False, 0
    projection, transforms = atlas.projection, atlas.transforms
    work = 0
    for cell, transform in zip((first, second), transforms, strict=True):
        local = projected_jacobian(cell, projection, budget=budget)
        if local is None:
            return False, work
        jacobian = tuple(
            tuple(
                sum_polynomials(
                    tuple(
                        scale(local[i][k], transform[k][j], budget=budget)
                        for k in range(2)
                    ),
                    budget=budget,
                )
                for j in range(2)
            )
            for i in range(2)
        )
        active = [(np.zeros((2,), dtype=np.float64), np.eye(2, dtype=np.float64), 0)]
        symmetric_parts: dict[tuple[int, int], Expression] = {}
        while active:
            origin, piece, depth = active.pop()
            if work >= maximum_work:
                return False, work
            work += 1
            if strict_spd(
                jacobian, "simplex", 2, origin, piece, symmetric_parts, budget=budget
            ):
                continue
            if depth == limits.maximum_subdivision_depth:
                return False, work
            children = embedding_children(cell, origin, piece)
            if len(active) + len(children) > maximum_work - work:
                return False, work
            active.extend((start, child, depth + 1) for start, child in children)
    return True, work


def _adjacent_tensor_injectivity(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[bool, int]:
    """Continuous piecewise monotonicity on a convex two-cell tensor atlas."""
    shared = np.asarray(
        sorted(set(first.vertices).intersection(second.vertices)), dtype=np.int64
    )
    gluing = tensor_gluing(first, second, shared)
    if gluing is None:
        return False, 0
    matrix, offset = gluing
    dimension = first.dimension
    reference = np.asarray(reference_cell_topology(first.kind).vertices, dtype=np.float64)
    face = reference[[first.vertices.index(int(value)) for value in shared]]
    fixed = np.flatnonzero(np.ptp(face, axis=0) == 0)
    if fixed.size != 1:
        return False, 0
    axis = int(fixed[0])
    origin = np.zeros(dimension, dtype=np.float64)
    origin[axis] = face[0, axis]
    tangent = np.eye(dimension, dtype=np.float64)[
        :, [i for i in range(dimension) if i != axis]
    ]
    trace_kind = "interval" if dimension == 2 else "quadrilateral"
    left = restrict_chart_expressions(
        first.coordinates, first.kind, trace_kind, origin, tangent, budget=budget
    )
    right = restrict_chart_expressions(
        second.coordinates,
        second.kind,
        trace_kind,
        matrix.T @ (origin - offset),
        matrix.T @ tangent,
        budget=budget,
    )
    if left != right:
        return False, 0
    midpoint = jacobian_midpoint(first, budget=budget)
    if midpoint is None:
        return False, 0
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    if rank != dimension:
        return False, 0
    work = 0
    for cell, transform in (
        (first, np.eye(dimension, dtype=np.float64)),
        (second, matrix),
    ):
        local = projected_jacobian(cell, projection, budget=budget)
        if local is None:
            return False, work
        jacobian = tuple(
            tuple(
                sum_polynomials(
                    tuple(
                        scale(
                            local[i][k], Fraction(float(transform[j, k])), budget=budget
                        )
                        for k in range(dimension)
                    ),
                    budget=budget,
                )
                for j in range(dimension)
            )
            for i in range(dimension)
        )
        active = [
            (
                np.zeros(dimension, dtype=np.float64),
                np.eye(dimension, dtype=np.float64),
                0,
            )
        ]
        symmetric_parts: dict[tuple[int, int], Expression] = {}
        while active:
            start, piece, depth = active.pop()
            if work >= maximum_work:
                return False, work
            work += 1
            if strict_spd(
                jacobian,
                cell.domain,
                dimension,
                start,
                piece,
                symmetric_parts,
                budget=budget,
            ):
                continue
            if depth == limits.maximum_subdivision_depth:
                return False, work
            children = embedding_children(cell, start, piece)
            if len(active) + len(children) > maximum_work - work:
                return False, work
            active.extend(
                (child_start, child, depth + 1) for child_start, child in children
            )
    return True, work


@final
class _JordanBoundaryPair(StrictModule, NonTrainableState):
    first_edges: tuple[Cell, ...]
    second_edges: tuple[Cell, ...]
    shared_first: Cell
    shared_second: Cell


def _prepare_jordan_boundary_pair(
    first: Cell, second: Cell, *, budget: CoordinateEnclosureBudget
) -> _JordanBoundaryPair | ContactDecision:
    """Prepare the oriented common arc and prove coefficient trace identity."""
    if (
        first.dimension != 2
        or second.dimension != 2
        or len(first.coordinates) != 2
        or len(second.coordinates) != 2
        or first.kind not in ("triangle", "quadrilateral")
        or second.kind not in ("triangle", "quadrilateral")
    ):
        return ContactDecision("unresolved", 0)
    shared = set(first.vertices).intersection(second.vertices)
    if len(shared) != 2:
        return ContactDecision("unresolved", 0)

    budget.reserve(0, 2048 + 512 * (len(first.vertices) + len(second.vertices)))
    first_edges, second_edges = (
        _planar_boundary_edges(first, budget=budget),
        _planar_boundary_edges(second, budget=budget),
    )
    if first_edges is None or second_edges is None:
        return ContactDecision("unresolved", 0)
    shared_first = next(
        (edge for edge in first_edges if set(edge.vertices) == shared), None
    )
    shared_second = next(
        (edge for edge in second_edges if set(edge.vertices) == shared), None
    )
    if shared_first is None or shared_second is None:
        return ContactDecision("unresolved", 0)
    if shared_first.vertices != tuple(reversed(shared_second.vertices)):
        return ContactDecision("unresolved", 0)
    reversal = affine_arguments(
        np.ones((1,), dtype=np.float64), -np.eye(1, dtype=np.float64), budget=budget
    )
    if not expression_sequence_equal(
        shared_first.coordinates,
        tuple(
            compose(value, reversal, budget=budget) for value in shared_second.coordinates
        ),
        budget=budget,
    ):
        return ContactDecision("unresolved", 0, check="mapped_trace_mismatch")
    return _JordanBoundaryPair(first_edges, second_edges, shared_first, shared_second)


def _jordan_other_boundary_contacts(
    boundary_pair: _JordanBoundaryPair,
    limits: MeshCertificateLimits,
    maximum_work: int,
    work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision:
    """Separate every noncommon boundary-arc pair for the crosscut implication."""
    first_edges, second_edges = boundary_pair.first_edges, boundary_pair.second_edges
    shared_first, shared_second = boundary_pair.shared_first, boundary_pair.shared_second
    for edge_a in first_edges:
        for edge_b in second_edges:
            if edge_a is shared_first and edge_b is shared_second:
                continue
            if work >= maximum_work:
                return ContactDecision("unresolved", work)
            with budget.temporary_scope():
                # Dimension-one contacts allow only the shared corner images;
                # they cannot enter this dimension-two Jordan route.
                contact_decision_4 = pair_decision(
                    edge_a, edge_b, limits, maximum_work - work, budget=budget
                )
                extra = contact_decision_4.work
            work += extra
            if contact_decision_4.check == "mapped_trace_mismatch":
                return ContactDecision("unresolved", work, check="mapped_trace_mismatch")
            if contact_decision_4.status != "separated":
                return ContactDecision("unresolved", work)
    return ContactDecision("separated", work)


@checked
def _adjacent_planar_jordan_contact(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    *,
    locally_injective: bool = False,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision:
    """Prove two oriented Jordan cells meet only on their common edge.

    N: both full cell maps have positive Jacobian and injective boundaries.
    X: the shared edge restrictions agree exactly with opposite orientations;
    all other boundary arcs meet only at their shared endpoints. T: the crosscut
    (theta-graph) lemma for Jordan disks puts the two positively oriented disk
    interiors on opposite sides of the common arc when the remaining arcs are
    disjoint. This proves contact only on that arc. See Pommerenke, Boundary Behaviour of Conformal Maps (1992),
    Proposition 2.12 (a crosscut separates a simply connected domain into two
    components), applied to the three boundary arcs of this theta graph. ``locally_injective``
    is a chart-reuse hint: it never skips the closed-boundary hypothesis checks.
    """

    boundary_pair = _prepare_jordan_boundary_pair(first, second, budget=budget)
    match boundary_pair:
        case ContactDecision():
            return boundary_pair
        case _JordanBoundaryPair():
            first_edges, second_edges = (
                boundary_pair.first_edges,
                boundary_pair.second_edges,
            )
        case _:
            assert_never(boundary_pair)
    work = 0
    for cell, edges in ((first, first_edges), (second, second_edges)):
        with budget.temporary_scope():
            if locally_injective:
                proved, extra = _planar_positive_jacobian(
                    cell, limits, maximum_work - work, budget=budget
                )
                if proved:
                    boundary = _planar_boundary_simple(
                        edges, limits, maximum_work - work - extra, budget=budget
                    )
                    proved, extra = boundary.proved, extra + boundary.work
            else:
                proved, extra = _planar_jordan_injective(
                    cell, limits, maximum_work - work, budget=budget
                )
        work += extra
        if not proved:
            return ContactDecision("unresolved", work)
    return _jordan_other_boundary_contacts(
        boundary_pair, limits, maximum_work, work, budget=budget
    )


def pair_decision(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    prepared_controls: tuple[_PlaneControls | None, _PlaneControls | None] | None = None,
    *,
    authenticated_first_traces: tuple[tuple[int, ...], ...] = (),
    tangent_cone: bool = True,
    checks: list[str] | None = None,
    locally_injective: bool = False,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision:
    if not cell_traces_equal(first, second, budget=budget):
        return ContactDecision("unresolved", 0, check="mapped_trace_mismatch")
    # Multi-affine box pairs decide the whole ordered candidate family,
    # coordinate axes included, on rational corner images before any
    # other contact route; a refusal there is final for that family.
    multi_affine = _multi_affine_box(first) and _multi_affine_box(second)
    if multi_affine and separating_plane(first, second, prepared_controls, budget=budget):
        return ContactDecision("separated", 0)

    affine = affine_mapped_contact_owner.contact_decision(
        first,
        second,
        maximum_work,
        method="affine_simplex",
        authenticated_first_traces=authenticated_first_traces,
        budget=budget,
    )
    if affine is not None:
        return affine
    polytope = affine_mapped_contact_owner.contact_decision(
        first,
        second,
        maximum_work,
        method="affine_polytope",
        budget=budget,
    )
    if polytope is not None:
        if checks is not None and "mapped_affine_polytope_contact" not in checks:
            checks.append("mapped_affine_polytope_contact")
        return polytope
    cone_work = 0
    if tangent_cone:
        contact = affine_mapped_contact_owner.contact_decision(
            first,
            second,
            maximum_work,
            method="tangent_cone",
            limits=limits,
            budget=budget,
        )
        if contact is not None:
            proved = contact.status == "separated"
            cone_work = contact.work
            if proved:
                return ContactDecision("separated", cone_work)
    if not multi_affine and separating_plane(
        first, second, prepared_controls, budget=budget
    ):
        return ContactDecision("separated", cone_work)
    adjacent, adjacency_work = _adjacent_triangle_injectivity(
        first, second, limits, maximum_work - cone_work, budget=budget
    )
    adjacency_work += cone_work
    if adjacent:
        return ContactDecision("separated", adjacency_work)
    tensor_adjacent, tensor_work = _adjacent_tensor_injectivity(
        first, second, limits, maximum_work - adjacency_work, budget=budget
    )
    adjacency_work += tensor_work
    if tensor_adjacent:
        return ContactDecision("separated", adjacency_work)

    prism_adjacent, prism_work = prism_embedding_owner.adjacent_prism_injectivity(
        first, second, limits, maximum_work - adjacency_work, budget=budget
    )
    adjacency_work += prism_work
    if prism_adjacent:
        return ContactDecision("separated", adjacency_work)
    if checks is not None and first.dimension == second.dimension == 2:
        contact_decision_1 = _adjacent_planar_jordan_contact(
            first,
            second,
            limits,
            maximum_work - adjacency_work,
            locally_injective=locally_injective,
            budget=budget,
        )
        jordan_work = contact_decision_1.work
        adjacency_work += jordan_work
        if contact_decision_1.status == "separated":
            key: MeshCertificateCheck = "mapped_planar_jordan_boundary_contact"
            if key not in checks:
                checks.append(key)
            return ContactDecision(
                contact_decision_1.status, adjacency_work, check=contact_decision_1.check
            )
        if contact_decision_1.check == "mapped_trace_mismatch":
            return ContactDecision(
                contact_decision_1.status, adjacency_work, check=contact_decision_1.check
            )
    if (
        _overlap_witness(first, second, budget=budget)
        or _overlap_witness(second, first, budget=budget)
        or _crossing_witness(first, second, budget=budget)
    ):
        return ContactDecision("overlap", adjacency_work)
    for parameter in _CONTACT_INTERIOR_PROBES:
        source_a = np.full((first.dimension,), parameter, dtype=np.float64)
        source_b = np.full((second.dimension,), parameter, dtype=np.float64)
        if _overlap_witness(first, second, source_a, budget=budget) or _overlap_witness(
            second, first, source_b, budget=budget
        ):
            return ContactDecision("overlap", adjacency_work)

    return _bisect_cell_contact(
        first, second, limits, maximum_work, adjacency_work, budget=budget
    )


def _bisect_cell_contact(
    first: Cell,
    second: Cell,
    limits: MeshCertificateLimits,
    maximum_work: int,
    adjacency_work: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> ContactDecision:
    """Decide the remaining chart pairs under one bounded FIFO storage owner."""
    with budget.live_storage() as live:
        active = BisectionWorklist(first.dimension, second.dimension, live)
        work = adjacency_work
        undecided: MeshCertificateCheck | None = None
        while active:
            with budget.temporary_scope():
                a, b, depth = active.pop()
                work += 1
                if work > maximum_work:
                    return ContactDecision(
                        "unresolved", work - 1, check="mapped_intersection_piece_budget"
                    )
                budget.reserve(
                    0,
                    _CONTACT_BOUND_BASE_BYTES
                    + _CONTACT_BOUND_COORDINATE_BYTES
                    * max(len(first.coordinates), len(second.coordinates)),
                )
                lo_a, hi_a = bounds(first, *a, budget=budget)
                lo_b, hi_b = bounds(second, *b, budget=budget)
                if np.any(hi_a < lo_b) or np.any(hi_b < lo_a):
                    del a, b, lo_a, hi_a, lo_b, hi_b
                    active.release_piece()
                    continue
                if depth > 0:
                    composition_a, composition_b = (
                        ArgumentComposition(
                            affine_arguments(*a, budget=budget), budget=budget
                        ),
                        ArgumentComposition(
                            affine_arguments(*b, budget=budget), budget=budget
                        ),
                    )
                    piece_a = first.composed(composition_a, budget=budget)
                    piece_b = second.composed(composition_b, budget=budget)
                    if (
                        _overlap_witness(piece_a, piece_b, budget=budget)
                        or _overlap_witness(piece_b, piece_a, budget=budget)
                        or _crossing_witness(piece_a, piece_b, budget=budget)
                    ):
                        return ContactDecision("overlap", work)
                    del piece_a, piece_b, composition_a, composition_b
                if depth == limits.maximum_subdivision_depth:
                    undecided = "mapped_intersection_subdivision_depth"
                    del a, b, lo_a, hi_a, lo_b, hi_b
                    active.release_piece()
                    continue
                active.reserve_split(2 ** max(first.dimension, second.dimension))
                if np.max(hi_a - lo_a) >= np.max(hi_b - lo_b):
                    children = tuple(
                        (child, b, depth + 1) for child in embedding_children(first, *a)
                    )
                else:
                    children = tuple(
                        (a, child, depth + 1) for child in embedding_children(second, *b)
                    )
                if len(active) + len(children) > maximum_work - work:
                    return ContactDecision(
                        "unresolved", work, check="mapped_intersection_piece_budget"
                    )
                active.extend(children)
                del children
            del a, b, lo_a, hi_a, lo_b, hi_b
            active.release_piece()
        return (
            ContactDecision("separated", work)
            if undecided is None
            else ContactDecision("unresolved", work, check=undecided)
        )


def _interval_trace_identity(
    first: Cell,
    second: Cell,
    vertices: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool | None:
    """Compare polynomial-cell edge trace coefficients, or return None for another owner.

    Composing a total-degree-``p`` polynomial with an affine edge map gives a
    univariate polynomial of degree at most ``p``. Two such restrictions agree
    identically exactly when they agree at ``p + 1`` distinct rational parameters.
    """
    polynomials: list[tuple[Polynomial, ...]] = []
    for cell in (first, second):
        if cell.kind == "pyramid":
            return None
        coordinates: list[Polynomial] = []
        for value in cell.coordinates:
            if isinstance(value, RationalPolynomial):
                return None
            coordinates.append(value)
        polynomials.append(tuple(coordinates))
    # Maximum total degree of the composed source polynomials, never a
    # nominal element degree; affine edge restriction cannot exceed it.
    degree = max(
        (
            sum(index)
            for coordinates in polynomials
            for value in coordinates
            for index in value
        ),
        default=0,
    )
    # Both edge parameters run from the facet's first nominal vertex to its
    # second, so opposite local cell orientations share one rational chart.
    ends = []
    for cell in (first, second):
        reference = reference_cell_topology(cell.kind).vertices
        ends.append(
            tuple(
                rational_vector(reference[cell.vertices.index(int(vertex))])
                for vertex in vertices
            )
        )
    for step in range(degree + 1):
        parameter = Fraction(step, max(degree, 1))
        values = []
        for restricted, (start, end) in zip(polynomials, ends, strict=True):
            point = tuple(
                a + parameter * (b - a) for a, b in zip(start, end, strict=True)
            )
            values.append(
                tuple(evaluate(value, point, budget=budget) for value in restricted)
            )
        if values[0] != values[1]:
            return False
    return True


def _multi_affine_trace_identity(
    first: Cell,
    second: Cell,
    vertices: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool | None:
    """Compare multi-affine box-cell facet traces, or return None for another owner.

    A multi-affine map restricted to an edge or square face of its box is
    multi-affine on that facet and equals the interpolant of its corner values.
    When both cells order the facet vertices around the same square, the two
    restrictions agree identically exactly when the rational corner images of the
    shared vertices agree.
    """
    if (
        first.corner_images is None
        or second.corner_images is None
        or vertices.size not in (2, 4)
    ):
        return None

    images = []
    for cell, corner_images in (
        (first, first.corner_images),
        (second, second.corner_images),
    ):
        rows = [cell.vertices.index(int(vertex)) for vertex in vertices]
        if vertices.size == 4:
            budget.reserve(2 * cell.dimension)
            # Opposite corners of the ordered square share their midpoint.
            reference = cell.reference_vertices
            if any(
                reference[rows[0]][axis] + reference[rows[2]][axis]
                != reference[rows[1]][axis] + reference[rows[3]][axis]
                for axis in range(cell.dimension)
            ):
                return None
        images.append(tuple(corner_images[row] for row in rows))
    budget.reserve(vertices.size * len(first.coordinates))
    return images[0] == images[1]


def _facet_trace(
    cell: Cell, vertices: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> tuple[ExactScalar, ...] | tuple[Expression, ...]:
    """Restrict a facet in the shared vertex order, including the pyramid apex."""
    reference = np.asarray(reference_cell_topology(cell.kind).vertices, dtype=np.float64)
    corners = reference[[cell.vertices.index(int(vertex)) for vertex in vertices]]
    if vertices.size == 1:
        source = rational_vector(corners[0])
        if cell.kind == "pyramid" and source[2] == 1:
            source = (Fraction(1, 2), Fraction(1, 2), Fraction(1))
        return tuple(evaluate(value, source, budget=budget) for value in cell.coordinates)
    matrix = (
        (corners[[1]] - corners[0]).T
        if vertices.size == 2
        else (corners[[1, 2 if vertices.size == 3 else 3]] - corners[0]).T
    )
    return restrict_chart_expressions(
        cell.coordinates,
        cell.kind,
        "interval"
        if vertices.size == 2
        else "triangle"
        if vertices.size == 3
        else "quadrilateral",
        corners[0],
        matrix,
        budget=budget,
    )


@checked
def trace_continuity(
    state: EmbeddingState,
    mesh: CellMesh,
    cells: list[Cell],
    cell_groups: np.ndarray | None = None,
    *,
    shared_charts: bool = False,
    geometry: CellGeometrySpec | None = None,
    budget: CoordinateEnclosureBudget,
    retain_facet_traces: bool = False,
) -> boundary_degree_embedding_owner._FacetTraceWitness | None:
    """Prove coefficient identity of both traces on every interior facet.

    With ``shared_charts`` many cells share one rational corner-image record, as the
    block charts of restricted children do; a multi-affine identity decided for
    two records at the same local facet corners is then reused, not re-proved.
    That corner identity precedes polynomial edge evaluation; records outside
    its hypotheses retain the full expression proof.
    """

    with budget.live_storage() as storage:
        connections = _trace_continuity(
            state, mesh, cells, cell_groups,
            shared_charts=shared_charts, geometry=geometry, budget=budget,
            storage=storage,
            retain_connections=retain_facet_traces and cell_groups is None,
        )
        if retain_facet_traces and cell_groups is None and state.clean:
            return boundary_degree_embedding_owner._issue_facet_trace_witness(
                state, mesh, cells, connections, geometry=geometry, budget=budget,
            )
    return None


def _trace_continuity(
    state: EmbeddingState,
    mesh: CellMesh,
    cells: list[Cell],
    cell_groups: np.ndarray | None,
    *,
    shared_charts: bool,
    geometry: CellGeometrySpec | None,
    budget: CoordinateEnclosureBudget,
    storage: CoordinateLiveStorage,
    retain_connections: bool,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    facets = embedding_state_owner.mesh_facets(mesh, geometry=geometry, budget=budget)
    state.checks.append("mapped_trace_continuity")
    # These internally prepared coordinate expressions are strongly owned by
    # cells and never mutated during this callback-free proof phase. Only the
    # ordered local facet, not global vertex numbering, enters a restriction.
    traces: dict[
        tuple[int, str, str, int, tuple[int, ...]],
        tuple[ExactScalar, ...] | tuple[Expression, ...],
    ] = {}
    storage.set_bound(1024)
    decided: dict[tuple[int, ...], bool | None] = {}
    connections = []
    for group in np.flatnonzero(facets.counts == 2):
        occurrences = np.flatnonzero(facets.group == group)
        if (
            cell_groups is not None
            and cell_groups[int(facets.cells[occurrences[0]])]
            != cell_groups[int(facets.cells[occurrences[1]])]
        ):
            continue
        # Restrictions are needed only for this facet; the source cell
        # expressions stay retained by the enclosing proof's source bank.
        with budget.temporary_scope():
            first = cells[int(facets.cells[occurrences[0]])]
            second = cells[int(facets.cells[occurrences[1]])]
            vertices = facets.rows[occurrences[0]]
            vertices = vertices[vertices >= 0]
            if retain_connections:
                storage.set_bound(storage.bound + 512, work=1)
                connections.append((
                    int(facets.cells[occurrences[0]]),
                    int(facets.cells[occurrences[1]]),
                    tuple(int(vertex) for vertex in vertices),
                ))
            entities = np.asarray((group,), dtype=np.int64)
            shared_records = (
                shared_charts
                and first.corner_images is not None
                and second.corner_images is not None
            )
            identity: bool | None = None
            if shared_records:
                key = (
                    id(first.corner_images),
                    id(second.corner_images),
                    *(first.vertices.index(int(vertex)) for vertex in vertices),
                    -1,
                    *(second.vertices.index(int(vertex)) for vertex in vertices),
                )
                if key not in decided:
                    decided[key] = _multi_affine_trace_identity(
                        first, second, vertices, budget=budget
                    )
                identity = decided[key]
            if identity is None and vertices.size == 2:
                identity = _interval_trace_identity(
                    first, second, vertices, budget=budget
                )
            if identity is None and not shared_records:
                identity = _multi_affine_trace_identity(
                    first, second, vertices, budget=budget
                )
            if identity is not None:
                if not identity:
                    state.add("mapped_trace_mismatch", "violated", "facet", entities)
                continue
            restrictions = []
            for cell in (first, second):
                local_facet = tuple(cell.vertices.index(int(vertex)) for vertex in vertices)
                key = (
                    id(cell.coordinates), cell.kind, cell.domain, cell.dimension, local_facet
                )
                budget.reserve(1 + len(local_facet))
                trace = traces.get(key)
                if trace is None:
                    trace = _facet_trace(cell, vertices, budget=budget)
                    storage.set_bound(
                        storage.bound + 512 + budget.live_object_storage_upper(trace)
                    )
                    traces[key] = trace
                restrictions.append(trace)
            if any(value is None for value in restrictions):
                state.add("mapped_trace_denominator", "unresolved", "facet", entities)
            elif restrictions[0] != restrictions[1]:
                state.add("mapped_trace_mismatch", "violated", "facet", entities)
    return tuple(connections)


@checked
def tensor_gluing(
    first: Cell, second: Cell, shared: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    if first.kind not in ("quadrilateral", "hexahedron") or second.kind != first.kind:
        return None
    dimension = first.dimension
    if shared.size != 2 ** (dimension - 1):
        return None
    a = np.asarray(reference_cell_topology(first.kind).vertices, dtype=np.float64)[
        [first.vertices.index(int(vertex)) for vertex in shared]
    ]
    b = np.asarray(reference_cell_topology(second.kind).vertices, dtype=np.float64)[
        [second.vertices.index(int(vertex)) for vertex in shared]
    ]
    fixed_a = np.flatnonzero(np.ptp(a, axis=0) == 0)
    fixed_b = np.flatnonzero(np.ptp(b, axis=0) == 0)
    if fixed_a.size != 1 or fixed_b.size != 1:
        return None
    ia, ib = int(fixed_a[0]), int(fixed_b[0])
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    matrix[ia, ib] = (1.0 if a[0, ia] == 1.0 else -1.0) * (
        1.0 if b[0, ib] == 0.0 else -1.0
    )
    for axis in range(dimension):
        if axis == ib:
            continue
        delta = b - b[0]
        candidates = np.flatnonzero(
            (np.abs(delta[:, axis]) == 1.0) & (np.count_nonzero(delta, axis=1) == 1)
        )
        if candidates.size != 1:
            return None
        row = int(candidates[0])
        matrix[:, axis] = (a[row] - a[0]) / delta[row, axis]
    offset = a[0] - matrix @ b[0]
    if not np.array_equal(b @ matrix.T + offset, a):
        return None
    return matrix, offset


@checked
def common_atlas_extension(
    cells: list[Cell],
    charts: dict[int, tuple[np.ndarray, np.ndarray]],
    lower: np.ndarray,
    spans: np.ndarray,
    state: EmbeddingState,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove a single polynomial map is injective beyond an occupied atlas.

    This extension hypothesis permits cavities and nonconvex reference occupancy;
    merely matching nodal values, or positive Jacobians on occupied cells, does
    not establish it.
    """
    common: tuple[Expression, ...] | None = None
    with budget.temporary_scope(), budget.live_storage() as common_storage:
        for cell_id, (matrix, offset) in charts.items():
            common_bound = None
            with budget.temporary_scope():
                before_expression = budget.temporary_bytes_upper
                arguments = affine_arguments(-matrix.T @ offset, matrix.T, budget=budget)
                expression = tuple(
                    compose(value, arguments, budget=budget)
                    for value in cells[cell_id].coordinates
                )
                if common is None:
                    common = expression
                    common_bound = budget.temporary_bytes_upper - before_expression
                elif not expression_sequence_equal(common, expression, budget=budget):
                    return False
            # The first polynomial bank survives comparison; later banks do not.
            if common_bound is not None:
                common_storage.set_bound(common_bound)
        if common is None:
            return False
        dimension = lower.size
        chart = affine_arguments(lower, np.diag(spans), budget=budget)
        extension = Cell(
            tuple(compose(value, chart, budget=budget) for value in common),
            "box",
            cells[next(iter(charts))].kind,
            dimension,
            (),
            (),
        )
        if any(
            bernstein_node_count(value, "box", dimension) > limits.maximum_bernstein_nodes
            for value in extension.coordinates
        ):
            return False
        return injective(extension, limits, state, budget=budget) is None


@final
class _TensorCharts(StrictModule, NonTrainableState):
    charts: dict[int, tuple[np.ndarray, np.ndarray]]
    consistent: bool = eqx.field(static=True)


@final
class _TensorPositions(StrictModule, NonTrainableState):
    origins: np.ndarray
    spans: np.ndarray


def _tensor_component_charts(
    root: int,
    root_cell: Cell,
    cells: list[Cell],
    adjacency: list[list[tuple[int, np.ndarray]]],
) -> _TensorCharts:
    """Propagate signed-permutation charts and detect incompatible gluing cycles."""
    dimension = root_cell.dimension
    charts = {
        root: (
            np.eye(dimension, dtype=np.float64),
            np.zeros((dimension,), dtype=np.float64),
        )
    }
    active = [root]
    consistent = True
    while active:
        a = active.pop()
        outer, shift = charts[a]
        for b, shared in adjacency[a]:
            gluing = tensor_gluing(cells[a], cells[b], shared)
            if gluing is None:
                consistent = False
                continue
            matrix, offset = gluing
            chart = (outer @ matrix, shift + outer @ offset)
            if b in charts:
                old = charts[b]
                consistent &= bool(
                    np.array_equal(old[0], chart[0]) and np.array_equal(old[1], chart[1])
                )
            else:
                charts[b] = chart
                active.append(b)
    return _TensorCharts(charts, consistent)


def _tensor_component_positions(
    cells: list[Cell], charts: dict[int, tuple[np.ndarray, np.ndarray]]
) -> _TensorPositions | None:
    """Require one vertex per chart position and one occupied cell per lattice origin."""
    consistent = True
    positions = []
    vertices_by_position: dict[tuple[float, ...], int] = {}
    positions_by_vertex: dict[int, tuple[float, ...]] = {}
    for cell_id, (matrix, offset) in charts.items():
        reference = np.asarray(
            reference_cell_topology(cells[cell_id].kind).vertices, dtype=np.float64
        )
        points = reference @ matrix.T + offset
        positions.append(tuple(np.min(points, axis=0).tolist()))
        for vertex, point in zip(cells[cell_id].vertices, points, strict=True):
            position = tuple(point.tolist())
            if (
                position in vertices_by_position
                and vertices_by_position[position] != vertex
            ):
                consistent = False
            if vertex in positions_by_vertex and positions_by_vertex[vertex] != position:
                consistent = False
            vertices_by_position[position] = vertex
            positions_by_vertex[vertex] = position
    if not consistent or len(set(positions)) != len(positions):
        return None
    origins = np.asarray(positions, dtype=np.float64)
    spans = np.max(origins, axis=0) - np.min(origins, axis=0) + 1.0
    return _TensorPositions(origins, spans)


def _tensor_atlas_monotone(
    root_cell: Cell,
    cells: list[Cell],
    charts: dict[int, tuple[np.ndarray, np.ndarray]],
    state: EmbeddingState,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Certify the piecewise projected Jacobian on a rectangular chart union."""
    dimension = root_cell.dimension
    consistent = True
    midpoint = jacobian_midpoint(root_cell, budget=budget)
    if midpoint is None or np.linalg.matrix_rank(midpoint) != dimension:
        return False
    projection, _, rank, _ = np.linalg.lstsq(
        midpoint, np.eye(midpoint.shape[0], dtype=np.float64), rcond=None
    )
    if rank != dimension:
        return False
    for cell_id, (matrix, _) in charts.items():
        with budget.temporary_scope():
            local = projected_jacobian(cells[cell_id], projection, budget=budget)
            if local is None:
                consistent = False
                break
            # A signed permutation is orthogonal exactly.
            jacobian = tuple(
                tuple(
                    sum_polynomials(
                        tuple(
                            scale(
                                local[i][k],
                                Fraction(float(matrix[j, k])),
                                budget=budget,
                            )
                            for k in range(dimension)
                        ),
                        budget=budget,
                    )
                    for j in range(dimension)
                )
                for i in range(dimension)
            )
            active_pieces = [
                (
                    np.zeros((dimension,), dtype=np.float64),
                    np.eye(dimension, dtype=np.float64),
                    0,
                )
            ]
            symmetric_parts: dict[tuple[int, int], Expression] = {}
            while active_pieces:
                origin, transform, depth = active_pieces.pop()
                if (
                    state.subdivision_pieces >= limits.maximum_subdivision_pieces
                    or depth > limits.maximum_subdivision_depth
                ):
                    consistent = False
                    break
                state.subdivision_pieces += 1
                if strict_spd(
                    jacobian,
                    cells[cell_id].domain,
                    dimension,
                    origin,
                    transform,
                    symmetric_parts,
                    budget=budget,
                ):
                    continue
                if depth == limits.maximum_subdivision_depth:
                    consistent = False
                    break
                active_pieces.extend(
                    (start, child, depth + 1)
                    for start, child in embedding_children(
                        cells[cell_id], origin, transform
                    )
                )
            if not consistent:
                break
    return consistent


@checked
def tensor_component_injectivity(
    state: EmbeddingState,
    mesh: CellMesh,
    cells: list[Cell],
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """A full rectangular reference atlas has a convex domain.

    Strict monotonicity of its continuous piecewise map proves injectivity of
    the *whole* component, including diagonally adjacent curved faces.
    """

    labels = np.full((len(cells),), -1, dtype=np.int64)
    facets = embedding_state_owner.mesh_facets(mesh, budget=budget)
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in cells]
    for group in np.flatnonzero(facets.counts == 2):
        occurrences = np.flatnonzero(facets.group == group)
        a, b = (int(value) for value in facets.cells[occurrences])
        shared = facets.rows[occurrences[0]]
        shared = shared[shared >= 0]
        adjacency[a].append((b, shared))
        adjacency[b].append((a, shared))
    visited: set[int] = set()
    for root, root_cell in enumerate(cells):
        if root in visited or root_cell.kind not in ("quadrilateral", "hexahedron"):
            continue
        component = _tensor_component_charts(root, root_cell, cells, adjacency)
        charts = component.charts
        visited.update(charts)
        if not component.consistent:
            continue
        positions = _tensor_component_positions(cells, charts)
        if positions is None:
            continue
        origins, spans = positions.origins, positions.spans
        if not np.all(origins == np.round(origins)):
            continue
        if np.prod(spans) != len(charts):
            if common_atlas_extension(
                cells,
                charts,
                np.min(origins, axis=0),
                spans,
                state,
                limits,
                budget=budget,
            ):
                labels[list(charts)] = root
                state.checks.append("mapped_injective_atlas_extension")
            continue
        consistent = _tensor_atlas_monotone(
            root_cell, cells, charts, state, limits, budget=budget
        )
        if consistent:
            labels[list(charts)] = root
            state.checks.append("mapped_convex_reference_atlas")
    return labels


def _carrier_corners_agree(
    cell: Cell, points: np.ndarray, *, budget: CoordinateEnclosureBudget
) -> bool:
    """Authenticate the binary carrier against every rational source corner image."""
    if cell.corner_images is not None:
        budget.reserve(len(cell.corner_images) * len(cell.coordinates))
        if any(
            value != Fraction(float(coordinate))
            for image, vertex in zip(cell.corner_images, cell.vertices, strict=True)
            for value, coordinate in zip(image, points[vertex], strict=True)
        ):
            return False
        return True
    reference = cell.reference_vertices
    vertices = (
        cell.vertices
        if cell.kind != "pyramid"
        else (*cell.vertices[:4], *([cell.vertices[4]] * 4))
    )
    for vertex, point in zip(vertices, reference, strict=True):
        if any(
            evaluate(value, point, budget=budget) != Fraction(float(coordinate))
            for value, coordinate in zip(cell.coordinates, points[vertex], strict=True)
        ):
            return False
    return True


def _straight_multi_affine_facet(
    cell: Cell,
    vertices: np.ndarray,
    points: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Use rational corner coplanarity for a degree-one box facet."""
    if vertices.size == 4:
        budget.reserve(5 * len(cell.coordinates) + 8)
        p = [rational_vector(point) for point in points[vertices]]
        u = tuple(b - a for a, b in zip(p[0], p[1], strict=True))
        v = tuple(b - a for a, b in zip(p[0], p[3], strict=True))
        w = tuple(b - a for a, b in zip(p[0], p[2], strict=True))
        normal = (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )
        if not any(normal) or sum(
            (a * b for a, b in zip(normal, w, strict=True)), Fraction(0)
        ):
            return False
    return True


def _bilinear_facet_in_plane(
    mapped: tuple[Polynomial, ...],
    points: np.ndarray,
    vertices: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove the full bilinear plane residual is zero, not only its corners."""
    # A planar, injective bilinear face has image equal to its corner quadrilateral:
    # all four boundary restrictions are straight and the Jordan degree is 1.
    if any(
        any(exponent > 1 for exponent in index)
        for polynomial in mapped
        for index in polynomial
    ):
        return False
    p = [rational_vector(point) for point in points[vertices]]
    u = tuple(b - a for a, b in zip(p[0], p[1], strict=True))
    v = tuple(b - a for a, b in zip(p[0], p[3], strict=True))
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    if not any(normal):
        return False
    plane = add(
        sum_polynomials(
            tuple(
                scale(value, weight, budget=budget)
                for value, weight in zip(mapped, normal, strict=True)
            ),
            budget=budget,
        ),
        constant(
            -sum((a * b for a, b in zip(normal, p[0], strict=True)), Fraction(0)),
            2,
            budget=budget,
        ),
        budget=budget,
    )
    if plane:
        return False
    return True


def _straight_polynomial_facet(
    cell: Cell,
    vertices: np.ndarray,
    points: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Restrict a boundary facet and prove its affine or planar bilinear image."""
    reference = np.asarray(reference_cell_topology(cell.kind).vertices, dtype=np.float64)
    corners = reference[[cell.vertices.index(int(vertex)) for vertex in vertices]]
    matrix = (
        (corners[[1]] - corners[0]).T
        if vertices.size == 2
        else (corners[[1, 2 if vertices.size == 3 else 3]] - corners[0]).T
    )
    source_polynomials = tuple(
        value for value in cell.coordinates if not isinstance(value, RationalPolynomial)
    )
    if len(source_polynomials) != len(cell.coordinates):
        return False
    mapped = coordinate_enclosure_owner.restrict_coordinates(
        source_polynomials, cell.kind, corners[0], matrix, budget=budget
    )
    if mapped is None:
        return False
    if vertices.size <= 3:
        if any(sum(index) > 1 for polynomial in mapped for index in polynomial):
            return False
        return True
    return _bilinear_facet_in_plane(mapped, points, vertices, budget=budget)


@checked
def straight_boundary_images(
    mesh: CellMesh, cells: list[Cell], *, budget: CoordinateEnclosureBudget
) -> bool:
    """Prove that the mapped boundary equals its authenticated polygonal carrier."""

    if any(
        isinstance(value, RationalPolynomial)
        for cell in cells
        for value in cell.coordinates
    ):
        return False

    points = np.asarray(mesh.coordinates, dtype=np.float64)
    for cell in cells:
        if not _carrier_corners_agree(cell, points, budget=budget):
            return False
    facets = embedding_state_owner.mesh_facets(mesh, budget=budget)
    for occurrence in np.flatnonzero(facets.boundary):
        cell = cells[int(facets.cells[occurrence])]
        vertices = facets.rows[occurrence]
        vertices = vertices[vertices >= 0]
        if cell.corner_images is not None and vertices.size in (2, 4):
            if not _straight_multi_affine_facet(cell, vertices, points, budget=budget):
                return False
        elif not _straight_polynomial_facet(cell, vertices, points, budget=budget):
            return False
    return True


def _plane_controls_storage_upper(
    controls: _PlaneControls, ledger: CoordinateEnclosureBudget, /
) -> int:
    """Meter the immutable local control net before it escapes a node scope."""
    size = sys.getsizeof(controls) + 128
    for bank in controls:
        ledger.reserve(1)
        size += sys.getsizeof(bank) + 128
    for net in controls.coordinates:
        ledger.reserve(1 + len(net))
        size += sys.getsizeof(net) + 128
        size += sum(sys.getsizeof(value) + 128 for value in net)
    for support in controls.degree_support:
        ledger.reserve(1)
        size += sys.getsizeof(support) + 128
        for indices in support:
            ledger.reserve(1 + len(indices))
            size += sys.getsizeof(indices) + 128
            size += sum(sys.getsizeof(value) + 128 for value in indices)
    return size


@final
class _ParentReferenceAction(StrictModule, NonTrainableState):
    root: CellGeometryElement
    vertices: tuple[tuple[Fraction, ...], ...]


@final
class _ChildImages(StrictModule, NonTrainableState):
    values: list[tuple[ExactScalar, ...]]


def _parent_reference_action(
    element: CellGeometryElement,
    reference: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _ParentReferenceAction | None:
    """Admit a full P1-unity source chart and evaluate its affine child corners."""
    chain = coordinate_enclosure_owner.try_partition_unity_reference_chain(
        element, budget=budget
    )
    if chain is None:
        return None
    root, arguments = chain
    if any(
        isinstance(value, coordinate_enclosure_owner.RationalPolynomial)
        or any(sum(index) > 1 for index in value)
        for value in arguments
    ):
        return None
    polynomial_arguments = cast(tuple[Polynomial, ...], arguments)
    exact_reference = tuple(
        tuple(
            rational_coefficient(
                coordinate_enclosure_owner.evaluate(value, point, budget=budget),
                role="mapped_embedding.parent_reference",
            )
            for value in polynomial_arguments
        )
        for point in reference
    )
    return _ParentReferenceAction(root, exact_reference)


def _parent_child_images(
    exact_reference: tuple[tuple[Fraction, ...], ...],
    exact_parent: tuple[tuple[ExactScalar, ...], ...],
    ambient: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _ChildImages:
    """Interpolate child corners with rational weights and cardinal-corner reuse."""
    if any(
        isinstance(value, RadicalElement) for point in exact_parent for value in point
    ):
        work = len(exact_reference) * len(exact_parent) * ambient
        reserve_algebraic_work(exact_parent, work, work, 2, budget=budget)
    physical = []
    interpolation_work = 0
    for point in exact_reference:
        weights = (1 - sum(point, Fraction(0)), *point)
        nonzero = tuple(index for index, weight in enumerate(weights) if weight)
        if len(nonzero) == 1 and weights[nonzero[0]] == 1:
            physical.append(exact_parent[nonzero[0]])
            continue
        interpolation_work += len(nonzero) * ambient
        physical.append(
            tuple(
                sum(
                    (weights[index] * exact_parent[index][axis] for index in nonzero),
                    Fraction(0),
                )
                for axis in range(ambient)
            )
        )

    budget.reserve(interpolation_work)
    return _ChildImages(physical)


def _partition_pair_separated(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Separate two rational reference triangles by one directed supporting edge."""
    separated = False
    for subject, other in (
        (first, second),
        (second, first),
    ):
        for edge in range(3):
            a, b = subject[edge], subject[(edge + 1) % 3]

            budget.reserve(3)
            sides = tuple(
                (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                for point in other
            )
            if max(sides) <= 0:
                separated = True
                break
        if separated:
            break
    return separated


def _affine_parent_partition(
    partitions: list[tuple[tuple[Fraction, ...], ...]],
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Require positive in-domain children, unit area and pairwise disjoint interiors."""
    total = Fraction(0)
    for triangle in partitions:
        a, b, c = triangle
        twice = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if twice <= 0 or any(
            value < 0 or sum(point, Fraction(0)) > 1
            for point in triangle
            for value in point
        ):
            return False
        total += twice
    if total != 1:
        return False
    for first in range(len(partitions)):
        for second in range(first + 1, len(partitions)):
            separated = _partition_pair_separated(
                partitions[first], partitions[second], budget=budget
            )
            if not separated:
                return False
    return True


@final
class _ParentPartition(StrictModule, NonTrainableState):
    vertices: dict[int, tuple[int, ...]]
    corners: dict[int, tuple[tuple[ExactScalar, ...], ...]]
    children: dict[int, list[tuple[tuple[Fraction, ...], ...]]]


def _prepare_parent_partition(
    mesh: CellMesh, geometry: CellGeometrySpec, *, budget: CoordinateEnclosureBudget
) -> _ParentPartition | None:
    """Authenticate child carriers and retain each scientific parent map once."""
    origin = geometry.restriction_source
    if origin is None:
        return None
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    bank = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    reference = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    parent_vertices: dict[int, tuple[int, ...]] = {}
    parent_corners: dict[int, tuple[tuple[ExactScalar, ...], ...]] = {}
    children: dict[int, list[tuple[tuple[Fraction, ...], ...]]] = {}
    target_images: list[tuple[ExactScalar, ...] | None] = [None] * mesh.coordinates.shape[
        0
    ]
    numeric = np.asarray(mesh.coordinates, dtype=np.float64)
    for block, element, routes_ in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind != "triangle":
            return None
        parent_ids = np.asarray(origin.block_parent_cell_ids[block.name], dtype=np.int64)
        parent_rows = np.asarray(
            origin.block_parent_vertex_ids[block.name], dtype=np.int64
        )
        target_rows = np.asarray(block.vertices, dtype=np.int64)
        prepared: _ParentReferenceAction | None = None
        parent_images: tuple[tuple[ExactScalar, ...] | None, ...] | None = None
        for row, (parent, vertices, target_vertices, route) in enumerate(
            zip(
                parent_ids.tolist(),
                parent_rows.tolist(),
                target_rows.tolist(),
                np.asarray(routes_).tolist(),
                strict=True,
            )
        ):
            if prepared is None:
                # Earn the reference law for this actual block owner, never
                # reuse an action just because its asserted element ID agrees.
                prepared = _parent_reference_action(element, reference, budget=budget)
                if prepared is None:
                    return None
                root = prepared.root
                if root.cell_kind == "triangle":
                    # These are independent temporary operand rows, not source
                    # vertex identities. The existing batch authenticates the
                    # complete root, current bank and every routed coefficient
                    # once, while the checks below still bind every real child
                    # and scientific parent. Admit the table and output slots
                    # before constructing either.
                    count = parent_ids.size
                    budget.reserve(0, 512 + 96 * count)
                    corner_rows = np.arange(3 * count, dtype=np.int64).reshape((-1, 3))
                    parent_images = coordinate_enclosure_owner.coordinate_vertex_images(
                        ((root, np.asarray(routes_), corner_rows),),
                        bank,
                        3 * count,
                        mismatch="mapped_embedding.parent_corner_images",
                        budget=budget,
                    )
                    if parent_images is not None:
                        budget.retain_basis(parent_images)
            root, exact_reference = prepared.root, prepared.vertices
            if parent_images is None:
                # Nontriangular roots and unsupported corner-weight laws keep
                # their original full per-cell coordinate evaluation.
                local = tuple(bank[int(index)] for index in route)
                exact_parent = coordinate_enclosure_owner.coordinate_corner_images(
                    root, local, budget=budget
                )
                if exact_parent is None or len(exact_parent) != 3:
                    return None
            else:
                first, second, third = parent_images[3 * row : 3 * row + 3]
                if first is None or second is None or third is None:
                    return None
                exact_parent = (first, second, third)
            physical = _parent_child_images(
                exact_reference, exact_parent, mesh.ambient_dimension, budget=budget
            ).values
            for vertex, image in zip(target_vertices, physical, strict=True):
                if not np.array_equal(
                    np.asarray(
                        tuple(float(value) for value in image), dtype=np.float64
                    ).view(np.uint64),
                    numeric[int(vertex)].view(np.uint64),
                ):
                    raise ValueError(
                        "mapped_embedding.child_carrier must be the RNE of its rational source map; got unequal coordinates"
                    )
                previous = target_images[int(vertex)]
                if previous is not None and previous != image:
                    raise ValueError(
                        "mapped_embedding.child_vertices must agree on rational shared images; got conflicting images"
                    )
                target_images[int(vertex)] = image
            identifier = int(parent)
            vertex_ids = tuple(int(value) for value in vertices)
            known_vertices = parent_vertices.setdefault(identifier, vertex_ids)
            known_corners = parent_corners.setdefault(identifier, exact_parent)
            if known_vertices != vertex_ids or known_corners != exact_parent:
                raise ValueError(
                    "mapped_embedding.parent_map must agree across its children; got conflicting vertex rows or corner images"
                )
            children.setdefault(identifier, []).append(exact_reference)
    if any(value is None for value in target_images):
        return None
    return _ParentPartition(parent_vertices, parent_corners, children)


def _certify_affine_parent_surface_contact(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Certify contacts on the parent atlas of a nested affine partition."""

    partition = _prepare_parent_partition(mesh, geometry, budget=budget)
    if partition is None:
        return False
    parent_vertices, parent_corners, children = (
        partition.vertices,
        partition.corners,
        partition.children,
    )
    compact: dict[int, int] = {}
    points: list[tuple[ExactScalar, ...]] = []
    triangles: list[tuple[int, int, int]] = []
    identifiers: list[int] = []
    for parent in sorted(children):
        partitions = children[parent]
        if not _affine_parent_partition(partitions, budget=budget):
            return False
        row = []
        for vertex, point in zip(
            parent_vertices[parent], parent_corners[parent], strict=True
        ):
            position = compact.get(vertex)
            if position is None:
                position = len(points)
                compact[vertex] = position
                points.append(point)
            elif points[position] != point:
                raise ValueError(
                    "mapped_embedding.parent_vertices must agree on rational shared images; got conflicting images"
                )
            row.append(position)
        triangles.append((row[0], row[1], row[2]))
        identifiers.append(parent)
    predicate_points, binary = mesh_certificates_owner.exact_source_predicate_view(
        np.asarray(points, dtype=object), budget=budget
    )
    if binary:
        state.checks.append("exact_source_binary64_value_equivalence")
    mesh_certificates_owner.pairwise_contacts(
        state,
        predicate_points,
        np.asarray(triangles, dtype=np.int64),
        np.asarray(identifiers, dtype=np.int64),
        "cell",
        "cell_contact",
        limits,
        budget=budget,
    )
    state.checks.extend(
        (
            "mapped_affine_source_maps",
            "mapped_exact_shared_corner_incidence",
            "mapped_cell_contact",
            "mapped_exact_affine_surface_contact",
            "mapped_affine_parent_partition",
        )
    )
    return True


@final
class _AffineSourceImages(StrictModule, NonTrainableState):
    values: tuple[tuple[ExactScalar, ...], ...]
    violated: bool = eqx.field(static=True)


def _affine_corner_measure(
    corners: tuple[tuple[ExactScalar, ...], ...],
    dimension: int,
    ambient: int,
    surface: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> ExactScalar | int:
    """Measure rational affine orientation or surface immersion after its reserve."""
    if any(isinstance(value, RadicalElement) for point in corners for value in point):
        reserve_algebraic_work(corners, 64, 64, 3, budget=budget)
    columns = tuple(
        tuple(corner[axis] - corners[0][axis] for axis in range(ambient))
        for corner in corners[1:]
    )
    budget.reserve(dimension * ambient + (9 if surface else 3 if dimension == 2 else 14))
    if surface:
        a, b = columns
        cross = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        determinant = int(any(value != 0 for value in cross))
    elif dimension == 2:
        determinant = columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]
    else:
        a, b, c = columns
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
    return determinant


def _collect_affine_images(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    dimension: int,
    surface: bool,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineSourceImages | None:
    """Authenticate corner carriers, shared incidence and affine cell orientation."""
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    bank = prepared_coordinate_source_bank(geometry, budget=budget)
    ledger = budget
    ledger.reserve(0, 128 + 8 * mesh.coordinates.shape[0])
    images: list[tuple[ExactScalar, ...] | None] = [None] * mesh.coordinates.shape[0]
    numeric = np.asarray(mesh.coordinates, dtype=np.float64)
    cursor = 0

    for block, element, routes_ in zip(mesh.blocks, elements, routes, strict=True):
        expressions = coordinate_enclosure_owner.source_expressions(
            element, budget=budget
        )
        if expressions is None:
            return None
        # A higher-degree element is admitted cell by cell when its full
        # coordinate action is exactly affine (straight Lagrange nodes).
        element_affine = affine_mapped_contact_owner.affine_expressions(expressions)
        if element_affine:
            polynomial_expressions = cast(tuple[Polynomial, ...], expressions)
            ledger.reserve(sum(len(value) for value in polynomial_expressions))
        for vertices, route in zip(
            np.asarray(block.vertices), np.asarray(routes_), strict=True
        ):
            local = tuple(bank[int(index)] for index in route)
            if not element_affine:
                action = coordinate_enclosure_owner.coordinate_expressions(
                    element, local, budget=budget
                )
                if action is None or not affine_mapped_contact_owner.affine_expressions(
                    action
                ):
                    return None
            corners_ = coordinate_enclosure_owner.coordinate_corner_images(
                element, local, budget=budget
            )
            if corners_ is None:
                return None
            corners = corners_
            for vertex, image in zip(vertices, corners, strict=True):
                ledger.reserve(len(image))
                if not np.array_equal(
                    np.asarray(
                        tuple(float(value) for value in image), dtype=np.float64
                    ).view(np.uint64),
                    numeric[int(vertex)].view(np.uint64),
                ):
                    raise ValueError(
                        "mapped_embedding.corner_carrier must be the RNE of its full source map; got unequal coordinates"
                    )
                previous = images[int(vertex)]
                if previous is not None and previous != image:
                    state.add(
                        "mapped_trace_mismatch",
                        "violated",
                        "cell",
                        cell_ids[cursor : cursor + 1],
                    )
                    return _AffineSourceImages((), True)
                images[int(vertex)] = image
            determinant = _affine_corner_measure(
                corners, dimension, mesh.ambient_dimension, surface, budget=budget
            )
            if determinant <= 0:
                state.add(
                    "mapped_local_orientation",
                    "violated",
                    "cell",
                    cell_ids[cursor : cursor + 1],
                )
                return _AffineSourceImages((), True)
            cursor += 1
    if any(image is None for image in images):
        # Unused carrier rows have no authored cell map and must not be promoted
        # to source authority by copying numerical coordinates.
        return None
    exact = tuple(image for image in images if image is not None)
    return _AffineSourceImages(exact, False)


def _certify_affine_source_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> bool:
    """Prove affine simplex maps with authenticated shared corner images.

    A mapped coefficient bank may have nonbinary corner images. Its correctly
    rounded mesh is not the source: affine incidence/contact predicates consume
    the authenticated rational images, without changing the public mapped binding.
    """

    dimension = mesh.topological_dimension
    surface = dimension == 2 and mesh.ambient_dimension == 3
    if (dimension != mesh.ambient_dimension and not surface) or dimension not in (2, 3):
        return False
    kind = "triangle" if dimension == 2 else "tetrahedron"
    if any(block.cell_kind != kind for block in mesh.blocks):
        return False
    if geometry.restriction_source is not None:
        if surface and _certify_affine_parent_surface_contact(
            state, mesh, geometry, limits, budget=budget
        ):
            return True
        # Restricted children do not independently define the source map. The
        # restricted route certifies the full declared roots and
        # their area-and-contact partition before promoting child embedding.
        return False
    collected = _collect_affine_images(
        state, mesh, geometry, cell_ids, dimension, surface, budget=budget
    )
    if collected is None:
        return False
    if collected.violated:
        return True
    exact = collected.values
    ledger = budget
    ledger.retain_basis(exact)
    state.checks.extend(
        (
            "mapped_affine_source_maps",
            "mapped_exact_shared_corner_incidence",
            "mapped_cell_contact",
        )
    )
    if surface:
        state.checks.append("mapped_exact_affine_surface_contact")
        predicate_points, binary = mesh_certificates_owner.exact_source_predicate_view(
            np.asarray(exact, dtype=object), budget=budget
        )
        if binary:
            state.checks.append("exact_source_binary64_value_equivalence")
        mesh_certificates_owner.surface_embedding(
            state, mesh, predicate_points, cell_ids, limits, budget=budget
        )
    else:
        predicate_points, binary = mesh_certificates_owner.exact_source_predicate_view(
            np.asarray(exact, dtype=object), budget=budget
        )
        if binary:
            state.checks.append("exact_source_binary64_value_equivalence")
        mesh_certificates_owner.volume_embedding(
            state,
            mesh,
            predicate_points,
            embedding_state_owner.mesh_facets(mesh, budget=budget),
            limits,
            budget=budget,
        )
    return True


@checked
def certify_mapped_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None = None,
) -> None:
    """Keep root cells, bounds and local control caches live for this proof."""

    ledger = budget
    with ledger.temporary_scope():
        if _certify_affine_source_embedding(
            state, mesh, geometry, cell_ids, limits, budget=budget
        ):
            return
    with ledger.temporary_scope():
        with ledger.live_storage() as live:
            _certify_mapped_embedding(
                state,
                mesh,
                geometry,
                cell_ids,
                limits,
                live,
                budget=budget,
                predecessor=predecessor,
            )


def _self_contact_diagnostics(
    cell: Cell,
    state: EmbeddingState,
    entities: np.ndarray,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Probe standard reference child crossings after a failed local-injectivity proof."""
    children = embedding_children(
        cell,
        np.zeros((cell.dimension,), dtype=np.float64),
        np.eye(cell.dimension, dtype=np.float64),
    )
    pieces = []
    for start, matrix in children:
        composition = ArgumentComposition(
            affine_arguments(start, matrix, budget=budget), budget=budget
        )
        pieces.append(cell.composed(composition, budget=budget))
    for a in range(len(pieces)):
        for b in range(a + 1, len(pieces)):
            if state.candidate_pairs >= limits.maximum_candidate_pairs:
                state.add(
                    "mapped_self_contact_candidate_budget",
                    "unresolved",
                    "cell",
                    entities,
                )
                break
            state.candidate_pairs += 1
            if _crossing_witness(pieces[a], pieces[b], budget=budget):
                state.add(
                    "mapped_self_contact",
                    "violated",
                    "cell",
                    entities,
                )


@final
class _PreparedEmbeddingCell(StrictModule, NonTrainableState):
    cell: Cell
    lower: np.ndarray
    upper: np.ndarray


def _prepare_embedding_cell(
    element: CellGeometryElement,
    local: tuple[tuple[Fraction, ...], ...],
    vertices: np.ndarray,
    kind: str,
    domain: BernsteinDomain,
    dimension: int,
    state: EmbeddingState,
    entities: np.ndarray,
    limits: MeshCertificateLimits,
    *,
    budget: CoordinateEnclosureBudget,
) -> _PreparedEmbeddingCell | None:
    """Prepare one full source map, certify local injectivity, and enclose its image."""
    prepared = multi_affine_coordinates(element, local, budget=budget)
    polynomials: tuple[Expression, ...] | None
    if prepared is None:
        polynomials, corner_images = (
            coordinate_expressions(element, local, budget=budget),
            None,
        )
    else:
        polynomials, corner_images = prepared
    if polynomials is None:
        state.add(
            "coordinate_source_enclosure",
            "unresolved",
            "cell",
            entities,
        )
        return None
    count = max(bernstein_node_count(value, domain, dimension) for value in polynomials)
    if count > limits.maximum_bernstein_nodes:
        state.add(
            "mapped_bernstein_node_budget",
            "unresolved",
            "cell",
            entities,
        )
        return None
    cell = Cell(
        polynomials,
        domain,
        kind,
        dimension,
        tuple(vertices.tolist()),
        embedding_vertices(kind),
        corner_images,
    )
    root_lower, root_upper = bounds(
        cell,
        np.zeros((cell.dimension,), dtype=np.float64),
        np.eye(cell.dimension, dtype=np.float64),
        budget=budget,
    )
    if not (np.all(np.isfinite(root_lower)) and np.all(np.isfinite(root_upper))):
        state.add(
            "mapped_coordinate_bound_range",
            "unresolved",
            "cell",
            entities,
        )
        return None
    reason = root_injective(cell, element, local, limits, state, budget=budget)
    if reason is not None:
        state.add(reason, "unresolved", "cell", entities)
        _self_contact_diagnostics(cell, state, entities, limits, budget=budget)
    return _PreparedEmbeddingCell(cell, root_lower, root_upper)


@final
class _EmbeddingCarrier(StrictModule, NonTrainableState):
    cells: list[Cell]
    boxes: list[tuple[np.ndarray, np.ndarray]]


def _prepare_embedding_carrier(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    points: tuple[tuple[ExactScalar, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _EmbeddingCarrier | None:
    """Retain all proved root maps and commit their unchanged coordinate cache payload."""
    ledger = budget
    cells: list[Cell] = []
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    descriptors = []
    prepared_coordinates = []
    prepared_ids = []
    offset = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        kind = block.cell_kind
        domain = embedding_domain(kind)
        for row, vertices in zip(
            np.asarray(route), np.asarray(block.vertices), strict=True
        ):
            with ledger.polynomial_preparation_scope():
                local = tuple(points[index] for index in row)
                prepared_cell = _prepare_embedding_cell(
                    element,
                    local,
                    vertices,
                    kind,
                    domain,
                    mesh.topological_dimension,
                    state,
                    cell_ids[offset : offset + 1],
                    limits,
                    budget=budget,
                )
                if prepared_cell is None:
                    return None
                cell, root_lower, root_upper = (
                    prepared_cell.cell,
                    prepared_cell.lower,
                    prepared_cell.upper,
                )
            ledger.retain_basis(cell.coordinates)
            ledger.retain_basis(cell.reference_vertices)
            ledger.retain_basis(cell.vertices)
            if cell.corner_images is not None:
                ledger.retain_basis(cell.corner_images)
            descriptors.append((kind, element, local, cell.vertices))
            prepared_coordinates.append(cell.coordinates)
            prepared_ids.append(int(cell_ids[offset]))
            cells.append(cell)
            boxes.append((root_lower, root_upper))
            offset += 1
    key = coordinate_identity_owner.coordinate_scope_key(mesh, geometry)
    prepared = coordinate_enclosure_owner.PreparedCoordinateCells(
        tuple(prepared_ids),
        tuple(descriptors),
        tuple(prepared_coordinates),
    )
    ledger.retain_basis(
        (key, prepared, prepared.cell_ids, prepared.descriptors, prepared.coordinates)
    )
    ledger.prepared_cell_cache[key] = prepared
    return _EmbeddingCarrier(cells, boxes)


def _certify_embedding_pairs(
    state: EmbeddingState,
    cells: list[Cell],
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    first: np.ndarray,
    second: np.ndarray,
    atlas_labels: np.ndarray,
    live: CoordinateLiveStorage,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Certify candidate pairs in deterministic order and publish contact findings."""
    ledger = budget
    prepared_controls: dict[int, _PlaneControls | None] = {}
    for a, b in zip(first.tolist(), second.tolist(), strict=True):
        if atlas_labels[a] >= 0 and atlas_labels[a] == atlas_labels[b]:
            continue
        with ledger.polynomial_preparation_scope():
            trace_equal = cell_traces_equal(cells[a], cells[b], budget=budget)
        if not trace_equal:
            state.add("mapped_trace_mismatch", "violated", "cell", cell_ids[[a, b]])
            continue
        for index in (a, b):
            if index in prepared_controls:
                continue
            with ledger.polynomial_preparation_scope():
                prepared = _prepare_plane_controls(cells[index], budget=budget)
                if prepared is not None:
                    # This cache is local to the proof, not a retained history
                    # bank. Its immutable payload escapes this scope.
                    live.set_bound(
                        live.bound + _plane_controls_storage_upper(prepared, ledger)
                    )
            prepared_controls[index] = prepared
        with ledger.polynomial_preparation_scope():
            contact_decision_2 = pair_decision(
                cells[a],
                cells[b],
                limits,
                limits.maximum_subdivision_pieces - state.subdivision_pieces,
                (prepared_controls[a], prepared_controls[b]),
                checks=state.checks,
                locally_injective=state.clean,
                budget=budget,
            )
        state.subdivision_pieces += contact_decision_2.work
        match contact_decision_2.status:
            case "separated":
                continue
            case "overlap":
                state.add("mapped_cell_overlap", "violated", "cell", cell_ids[[a, b]])
            case "unresolved":
                check = contact_decision_2.unresolved_check()
                state.add(
                    check,
                    "violated" if check == "mapped_trace_mismatch" else "unresolved",
                    "cell",
                    cell_ids[[a, b]],
                )
            case _:
                assert_never(contact_decision_2.status)


def _certify_mapped_embedding(
    state: EmbeddingState,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    cell_ids: np.ndarray,
    limits: MeshCertificateLimits,
    live: CoordinateLiveStorage,
    *,
    budget: CoordinateEnclosureBudget,
    predecessor: CellGeometryPredecessor | None = None,
) -> None:

    ledger = budget

    state.checks.extend(("mapped_local_injectivity", "mapped_cell_contact"))
    elements, routes, _ = geometry.resolve(mesh, budget=budget)

    points = prepared_coordinate_source_bank(geometry, budget=budget)
    if geometry.restriction_source is not None and all(
        coordinate_enclosure_owner.has_cartesian_reference_chain(element)
        for element in elements
    ):
        # Restricted proofs compose mapped proof kernels; defer this orchestration edge to the request.
        from ._restricted_embedding import certify_restricted_embedding

        certify_restricted_embedding(
            state,
            mesh,
            geometry,
            cell_ids,
            limits,
            budget=budget,
            predecessor=predecessor,
        )
        return
    total = cell_ids.size
    # Cell dictionaries, box owners/views, stacked bounds, and the
    # local lists/dictionaries survive all per-cell temporary scopes.
    live.set_bound(1024 + total * (1536 + 64 * mesh.ambient_dimension), work=total)
    carrier = _prepare_embedding_carrier(
        state, mesh, geometry, cell_ids, limits, elements, routes, points, budget=budget
    )
    if carrier is None:
        return
    cells, boxes = carrier.cells, carrier.boxes
    with ledger.polynomial_preparation_scope():
        trace_continuity(state, mesh, cells, geometry=geometry, budget=budget)
        straight_boundary = (
            state.clean
            and mesh.topological_dimension == mesh.ambient_dimension
            and mesh.topological_dimension in (2, 3)
            and straight_boundary_images(mesh, cells, budget=budget)
        )
    if straight_boundary:
        state.checks.append("mapped_straight_boundary_degree")
        mesh_certificates_owner.volume_embedding(
            state,
            mesh,
            np.asarray(mesh.coordinates, dtype=np.float64),
            embedding_state_owner.mesh_facets(mesh, budget=budget),
            limits,
            budget=budget,
        )
        return

    with ledger.polynomial_preparation_scope():
        decided = boundary_degree_embedding_owner.certify_boundary_degree_embedding(
            state, mesh, cells, cell_ids, limits, budget=budget
        )
    if decided:
        return
    with ledger.polynomial_preparation_scope():
        atlas_labels = tensor_component_injectivity(
            state, mesh, cells, limits, budget=budget
        )
    lower = np.stack([value[0] for value in boxes])
    upper = np.stack([value[1] for value in boxes])
    unbounded = ~(np.all(np.isfinite(lower), axis=1) & np.all(np.isfinite(upper), axis=1))
    if np.any(unbounded):
        state.add(
            "mapped_coordinate_bound_range", "unresolved", "cell", cell_ids[unbounded]
        )
        return
    first, second, exceeded = embedding_state_owner.candidate_pairs(
        lower, upper, limits.maximum_candidate_pairs - state.candidate_pairs
    )
    state.candidate_pairs += first.size
    if exceeded:
        state.add("mapped_candidate_pair_budget", "unresolved", "mesh")
    _certify_embedding_pairs(
        state, cells, cell_ids, limits, first, second, atlas_labels, live, budget=budget
    )


# The source side of this proof is deliberately not a coordinate Expression.
# Native interval jets and exact polynomial target restrictions have different
# authorities; only their outward enclosures meet in the following arithmetic.
type _HomotopyInterval = tuple[np.ndarray, np.ndarray]
_HOMOTOPY_REFERENCE = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))


class _IntervalHomotopyRefusal(ValueError):
    """An unresolved sufficient inequality, optionally naming its facets."""

    def __init__(self, reason: str, entities: tuple[int, ...] | None = None) -> None:
        super().__init__(reason)
        self.entities = entities


_HOMOTOPY_MEMO_ENTRIES = 1 << 14
# Key tuple, reference bytes, map identity and up to four (3, 2) host
# intervals: a completed jet and its raw/completed orientation hull.
_HOMOTOPY_MEMO_ENTRY_BYTES = 2048
# One prefetched batch inserts at most a quarter of the memo, so its own LRU
# entries survive until every predicate of that batch has read them.
_HOMOTOPY_BATCH = 1024
# One retained pair/task record or prepared vertex cone: indices, exact target
# rays, rounded proposal rays, projections and transforms.
_HOMOTOPY_RECORD_BYTES = 1024

# (cell, reference triangle, jet): a jet row also yields its orientation hull.
type _SourceRequest = tuple[IntervalBoundaryCell, np.ndarray, bool]


class _HomotopyVertex(NamedTuple):
    """Reference rays from one shared corner and the exact target rays there."""

    directions: np.ndarray
    point: np.ndarray
    target_rays: tuple[tuple[ExactScalar, ...], ...]
    rounded_rays: np.ndarray
    # Outward binary bounds of rational target rays; None for radical rays.
    ray_bounds: _HomotopyInterval | None


class _IntervalHomotopyWork:
    """One piece/query/storage allowance for every obligation of this request.

    Source and exact-target enclosures are memoized per map and dyadic
    reference triangle: deviation, monotonicity and pair obligations revisit
    the same subdivision pieces. A memo hit is the identical enclosure of the
    identical request, so it consumes no new source query. Missing source rows
    are enclosed in per-map batches, still one source query per row. The
    bounded LRU memo is charged to live storage.
    Authenticated native source rows additionally share the original request's
    bounded, byte-accounted source LRU across proofs. Target Bernstein bounds
    always remain proof-local because a new candidate changes their meaning.
    """

    def __init__(
        self,
        limits: MeshCertificateLimits,
        budget: CoordinateEnclosureBudget,
        live: CoordinateLiveStorage,
        base_bytes: int,
    ) -> None:
        self.limits = limits
        self.budget = budget
        self.live = live
        self.base_bytes = base_bytes
        self.pieces = 0
        self.distances = 0
        self.pairs = 0
        self.source_queries = 0
        self.ray_tests = 0
        self.pending = 0
        self.records = 0
        self.memo: dict[tuple[object, ...], tuple[_HomotopyInterval, ...]] = {}
        self.vertices: dict[tuple[int, int], _HomotopyVertex] = {}
        self._source_tokens: dict[int, tuple[object, str | tuple[str, int]]] = {}
        self._source_carriers: dict[int, tuple[object, str]] = {}
        self._source_admissions: dict[int, _CarrierAdmission] = {}
        self.storage(0)

    def storage(self, pending: int, *, additional_source_admissions: int = 0) -> None:
        # Each entry owns at most two (3,2) references, time bounds, depths,
        # interval vectors/matrices and the worklist tuple. The live bound is
        # this request's high-water mark: it is raised, never lowered early.
        self.pending = pending
        size = (
            self.base_bytes
            + 4096 * (pending + self.limits.maximum_subdivision_depth + 3)
            + _HOMOTOPY_MEMO_ENTRY_BYTES * len(self.memo)
            + _HOMOTOPY_RECORD_BYTES * (
                self.records + len(self.vertices)
                + len(self._source_tokens) + len(self._source_carriers)
            )
            + 512 * (len(self._source_admissions) + additional_source_admissions)
        )
        if size > self.live.bound:
            self.live.set_bound(size)

    def _retain_source_admission(self) -> None:
        """Admit each persistent proof-local entry before creating it."""
        self.storage(self.pending, additional_source_admissions=1)

    def retain(self, records: int) -> None:
        self.records += records
        self.storage(self.pending)

    def piece(self) -> None:
        if self.pieces >= self.limits.maximum_subdivision_pieces:
            raise _IntervalHomotopyRefusal("interval_homotopy_piece_limit")
        self.pieces += 1
        self.budget.reserve(1)

    def poll(self) -> None:
        self.budget.reserve(0)

    def distance(self) -> None:
        if self.distances >= self.limits.maximum_distance_evaluations:
            raise _IntervalHomotopyRefusal("interval_homotopy_distance_limit")
        self.distances += 1

    def _lookup(self, key: tuple[object, ...]) -> tuple[_HomotopyInterval, ...] | None:
        value = self.memo.pop(key, None)
        if value is not None:
            self.memo[key] = value
        return value

    def _remember(self, key: tuple[object, ...], value: tuple[_HomotopyInterval, ...]) -> None:
        if len(self.memo) >= _HOMOTOPY_MEMO_ENTRIES:
            del self.memo[next(iter(self.memo))]
        self.memo[key] = value

    def _source_token(self, cell: IntervalBoundaryCell) -> str | tuple[str, int]:
        source = cell.source
        from ._domain_chart_chain import SourceChartMap, SourceChartMapError, source_chart_enclosure_identity

        previous = self._source_tokens.get(id(source))
        immutable_native = self._source_carriers.get(id(source))
        if previous is not None and previous[0] is source and (
            type(source) is not SourceChartMap
            or (immutable_native is not None and immutable_native[0] is source)
        ):
            # This work object owns one pure proof. Only a fully authenticated
            # immutable native source enters its root memo; new proofs and
            # actual native evaluations still authenticate their own inputs.
            return previous[1]

        identity = (
            source_chart_enclosure_identity(
                source, self.budget, self._source_carriers,
                admissions=self._source_admissions,
                retain_admission=self._retain_source_admission,
            )
            if type(source) is SourceChartMap else None
        )
        if type(source) is SourceChartMap and identity is None:
            raise SourceChartMapError("source_chart_requires_genuine_reprepare")
        # Other protocol implementations have only proof-local object
        # identities; an asserted map_id never earns cross-proof reuse.
        token = identity if identity is not None else (source.map_id, id(source))
        self._source_tokens[id(source)] = (source, token)
        self.storage(self.pending)
        return token

    def _source_lookup(self, cell, kind: str, raw: bytes):
        token = self._source_token(cell)
        key = (kind, token, raw)
        value = self._lookup(key)
        if value is None and isinstance(token, str):
            cache = self.budget.source_homotopy_enclosure_cache
            value = cache.pop((kind, token, raw), None)
            if value is not None:
                cache[kind, token, raw] = value
                self._remember(key, value)
                self.storage(self.pending)
        return value

    def _enclose(
        self, requests: list[tuple[IntervalBoundaryCell, list[np.ndarray]]], jet: bool
    ) -> None:
        """Enclose source rows of several maps; each row is one source query."""
        count = sum(len(references) for _, references in requests)
        if self.source_queries + count > self.limits.maximum_source_samples:
            raise _IntervalHomotopyRefusal("interval_homotopy_source_query_limit")
        self.source_queries += count
        kinds: dict[
            type[IntervalSurfaceMap], list[tuple[IntervalBoundaryCell, list[np.ndarray]]]
        ] = {}
        for cell, references in requests:
            kinds.setdefault(type(cell.source), []).append((cell, references))
        kind = "jet" if jet else "image"
        for owner, group in kinds.items():
            results = owner.enclosures(
                [(cell.source, np.stack(references)) for cell, references in group],
                image=not jet, jacobian=jet, budget=self.budget,
            )
            if len(results) != len(group):
                raise _IntervalHomotopyRefusal("interval_homotopy_invalid_enclosure")
            for (cell, references), result in zip(group, results, strict=True):
                parts = (result.jacobian, result.orientation) if jet else (result.image,)
                shape = (len(references), 3, 2) if jet else (len(references), 3)
                if any(part is None for part in parts):
                    raise _IntervalHomotopyRefusal("interval_homotopy_invalid_enclosure")
                checked = [
                    _homotopy_checked_interval(cast(_HomotopyInterval, part), shape)
                    for part in parts
                ]
                for index, reference in enumerate(references):
                    raw = reference.tobytes()
                    token = self._source_token(cell)
                    rows = tuple((lower[index], upper[index]) for lower, upper in checked)
                    if isinstance(token, str):
                        stored = self.budget.source_homotopy_enclosure(
                            (kind, token, raw), rows,
                            maximum_entries=_HOMOTOPY_MEMO_ENTRIES,
                        )
                    else:
                        stored = tuple((lower.copy(), upper.copy()) for lower, upper in rows)
                    self._remember((kind, token, raw), stored)
        self.storage(self.pending)

    def prefetch(self, requests: Iterable[_SourceRequest]) -> None:
        """Enclose every missing source row of these requests in batches of maps."""
        groups: dict[tuple[object, bool], tuple[IntervalBoundaryCell, dict[bytes, np.ndarray]]] = {}
        for cell, reference, jet in requests:
            reference = np.asarray(reference, dtype=np.float64)
            raw = reference.tobytes()
            kind = "jet" if jet else "image"
            if self._source_lookup(cell, kind, raw) is not None:
                continue
            groups.setdefault((self._source_token(cell), jet), (cell, {}))[1].setdefault(raw, reference)
        for wanted in (False, True):
            batch: list[tuple[IntervalBoundaryCell, list[np.ndarray]]] = []
            rows = 0
            for (_, jet), (cell, references) in groups.items():
                if jet != wanted:
                    continue
                pending = list(references.values())
                while pending:
                    taken = pending[: _HOMOTOPY_BATCH - rows]
                    pending = pending[len(taken) :]
                    batch.append((cell, taken))
                    rows += len(taken)
                    if rows == _HOMOTOPY_BATCH:
                        self._enclose(batch, wanted)
                        batch, rows = [], 0
            if batch:
                self._enclose(batch, wanted)

    def source_interval(
        self, cell: IntervalBoundaryCell, reference: np.ndarray, *,
        jet: bool, orientation: bool = False,
    ) -> _HomotopyInterval:
        reference = np.asarray(reference, dtype=np.float64)
        kind, raw = "jet" if jet else "image", reference.tobytes()
        value = self._source_lookup(cell, kind, raw)
        if value is None:
            self._enclose([(cell, [reference])], jet)
            value = self.memo[kind, self._source_token(cell), raw]
        return value[1] if orientation else value[0]

    def target_interval(
        self,
        cell: IntervalBoundaryCell,
        kind: str,
        expressions: tuple[Expression, ...],
        reference: np.ndarray,
    ) -> _HomotopyInterval:
        """Exact-target hull of one cell's coordinates or Jacobian, memoized."""
        return self.target_enclosure(cell, kind, expressions, reference)[0]

    def target_enclosure(
        self,
        cell: IntervalBoundaryCell,
        kind: str,
        expressions: tuple[Expression, ...],
        reference: np.ndarray,
    ) -> tuple[_HomotopyInterval, _HomotopyInterval | None]:
        """Memoized exact-target hull and outward Bernstein control net."""
        key = (kind, cell.entity, np.asarray(reference, dtype=np.float64).tobytes())
        cached = self._lookup(key)
        if cached is not None:
            return cached[0], cached[1] if len(cached) > 1 else None
        box, net = _homotopy_target_enclosure(expressions, reference, budget=self.budget)
        self._remember(key, (box,) if net is None else (box, net))
        self.storage(self.pending)
        return box, net


def _homotopy_checked_interval(
    value: _HomotopyInterval, shape: tuple[int, ...]
) -> _HomotopyInterval:
    lower, upper = (np.asarray(part, dtype=np.float64) for part in value)
    if (
        lower.shape != shape
        or upper.shape != shape
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise _IntervalHomotopyRefusal("interval_homotopy_invalid_enclosure")
    return lower, upper


def _homotopy_quadratic_controls(
    expressions: tuple[Expression, ...], reference: np.ndarray
) -> list[tuple[list[int], int]] | None:
    """Exact degree-two Bernstein nets of rational quadratics on a subtriangle.

    On the subtriangle with corners r_i the restriction has control values
    P(r_i) and 2 P((r_i + r_j) / 2) - (P(r_i) + P(r_j)) / 2. These are exactly
    the degree-two simplex Bernstein coefficients of the composed polynomial.
    Binary corners are exact integers on one dyadic grid that also holds their
    midpoints, so each expression's six controls are integers over one common
    denominator, returned as (controls, denominator). Radical or higher-degree
    coefficients return ``None``.
    """
    ratios = [
        float(value).as_integer_ratio()
        for value in np.asarray(reference, dtype=np.float64).reshape(-1)
    ]
    scale = 2 * max(denominator for _, denominator in ratios)
    grid = [numerator * (scale // denominator) for numerator, denominator in ratios]
    corners = [(grid[2 * k], grid[2 * k + 1]) for k in range(3)]
    points = [
        *corners,
        *(
            ((corners[i][0] + corners[j][0]) // 2, (corners[i][1] + corners[j][1]) // 2)
            for i, j in ((0, 1), (1, 2), (0, 2))
        ),
    ]
    hulls = []
    for value in expressions:
        if not isinstance(value, dict):
            return None
        terms = []
        common = 1
        for exponent, coefficient in value.items():
            if (
                not isinstance(coefficient, (int, Fraction))
                or isinstance(coefficient, bool)
                or len(exponent) != 2
                or min(exponent) < 0
                or sum(exponent) > 2
            ):
                return None
            coefficient = Fraction(coefficient)
            terms.append((exponent, coefficient))
            common = math.lcm(common, coefficient.denominator)
        # value * scale**2 * common is an integer polynomial in grid points.
        integer_terms = [
            (
                exponent[0],
                exponent[1],
                coefficient.numerator
                * (common // coefficient.denominator)
                * scale ** (2 - exponent[0] - exponent[1]),
            )
            for exponent, coefficient in terms
        ]
        values = [
            sum(term * x**first * y**second for first, second, term in integer_terms)
            for x, y in points
        ]
        controls = [
            *(2 * corner for corner in values[:3]),
            *(
                4 * values[3 + index] - values[i] - values[j]
                for index, (i, j) in enumerate(((0, 1), (1, 2), (0, 2)))
            ),
        ]
        hulls.append((controls, 2 * scale * scale * common))
    return hulls


def _homotopy_target_enclosure(
    expressions: tuple[Expression, ...],
    reference: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_HomotopyInterval, _HomotopyInterval | None]:
    """Exact-hull box and outward (6,k) Bernstein control net, when rational.

    The degree-two control net of any scalar combination n.P on the
    subtriangle is n applied to this vector net, so it bounds every projection.
    """
    budget.reserve(6, 1024)
    nets = _homotopy_quadratic_controls(expressions, reference)
    if nets is not None:
        budget.reserve(len(nets))
        box = (
            np.asarray(
                [fraction_endpoint(Fraction(min(net), scale), upper=False) for net, scale in nets]
            ),
            np.asarray(
                [fraction_endpoint(Fraction(max(net), scale), upper=True) for net, scale in nets]
            ),
        )
        # Integer true division is correctly rounded; one ulp is outward.
        rounded = np.asarray([[value / scale for value in net] for net, scale in nets]).T
        return box, (np.nextafter(rounded, -np.inf), np.nextafter(rounded, np.inf))
    exact = np.asarray([rational_vector(point) for point in reference], dtype=object)
    composition = ArgumentComposition(
        affine_arguments(
            exact[0], (exact[1:] - exact[0]).T, budget=budget
        ),
        budget=budget,
    )
    result = np.asarray(
        [
            polynomial_bounds(
                composition(value, budget=budget), "simplex", 2, budget=budget
            )
            for value in expressions
        ],
        dtype=np.float64,
    )
    return (result[:, 0], result[:, 1]), None


def _homotopy_target_bounds(
    expressions: tuple[Expression, ...],
    reference: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> _HomotopyInterval:
    return _homotopy_target_enclosure(expressions, reference, budget=budget)[0]


def _homotopy_jets(
    cell: IntervalBoundaryCell,
    jacobian: tuple[Expression, ...],
    reference: np.ndarray,
    *,
    work: _IntervalHomotopyWork,
    orientation: bool = False,
) -> tuple[_HomotopyInterval, _HomotopyInterval]:
    source = work.source_interval(cell, reference, jet=True, orientation=orientation)
    lower, upper = work.target_interval(cell, "jacobian", jacobian, reference)
    return source, (lower.reshape(3, 2), upper.reshape(3, 2))


def _homotopy_linear(
    left: _HomotopyInterval, right: _HomotopyInterval
) -> _HomotopyInterval:
    """Small (batched) matrix product with directed rounding on every accumulation.

    Operands are (..., a, b) and (..., b, c). Elementwise outward products are
    accumulated left to right from zero, the same rounding sequence as one
    ``interval_dot`` per entry; leading batch axes never mix entries.
    """
    left_lower, left_upper = np.asarray(left[0]), np.asarray(left[1])
    products = interval_multiply(
        (left_lower[..., :, :, None], left_upper[..., :, :, None]),
        (np.asarray(right[0])[..., None, :, :], np.asarray(right[1])[..., None, :, :]),
    )
    shape = np.asarray(products[0]).shape
    lower, upper = np.zeros(shape[:-2] + shape[-1:]), np.zeros(shape[:-2] + shape[-1:])
    for index in range(left_lower.shape[-1]):
        lower, upper = interval_add(
            (lower, upper),
            (np.asarray(products[0])[..., index, :], np.asarray(products[1])[..., index, :]),
        )
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _homotopy_spd(
    jacobian: _HomotopyInterval,
    projection: np.ndarray,
    transform: _HomotopyInterval,
) -> bool:
    projected = _homotopy_linear(
        _homotopy_linear((projection, projection), jacobian), transform
    )
    low, high = projected
    off = interval_multiply(
        interval_add((low[0, 1], high[0, 1]), (low[1, 0], high[1, 0])),
        (np.asarray(0.5), np.asarray(0.5)),
    )
    square = interval_multiply(off, off)
    determinant = interval_subtract(
        interval_multiply(
            (low[0, 0], high[0, 0]), (low[1, 1], high[1, 1])
        ),
        square,
    )
    return bool(low[0, 0] > 0 and determinant[0] > 0)


def _homotopy_children(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # All reference coordinates start dyadic. Refuse stagnation rather than
    # treating a rounded midpoint as an exact partition.
    edges = ((0, 1), (1, 2), (2, 0))
    a, b = max(edges, key=lambda edge: np.linalg.norm(reference[edge[0]] - reference[edge[1]]))
    midpoint = (reference[a] + reference[b]) * 0.5
    if np.array_equal(midpoint, reference[a]) or np.array_equal(midpoint, reference[b]):
        raise _IntervalHomotopyRefusal("interval_homotopy_reference_resolution")
    if any(
        Fraction(float(midpoint[k]))
        != (Fraction(float(reference[a, k])) + Fraction(float(reference[b, k]))) / 2
        for k in range(2)
    ):
        raise _IntervalHomotopyRefusal("interval_homotopy_reference_resolution")
    first, second = reference.copy(), reference.copy()
    first[b], second[a] = midpoint, midpoint
    return first, second


# (refusal facets, predicate on a reference subtriangle, its source rows).
type _CoverTask = tuple[
    tuple[int, ...],
    Callable[[np.ndarray], bool],
    Callable[[np.ndarray], tuple[_SourceRequest, ...]],
]


def _homotopy_covers(
    tasks: list[_CoverTask],
    work: _IntervalHomotopyWork,
    reason: str,
    *,
    depth_offset: int = 0,
    batch: int = _HOMOTOPY_BATCH,
) -> None:
    """Prove every task predicate on a bisection cover of the reference triangle.

    The worklist is popped in last-in batches of at most ``batch`` pieces whose
    missing source rows are enclosed together before any predicate reads them.
    Every piece is charged and depth-limited exactly as in independent covers;
    a two-piece batch keeps the depth-first order of a failing sibling pair.
    """
    active = [(task, _HOMOTOPY_REFERENCE, depth_offset) for task in reversed(range(len(tasks)))]
    while active:
        work.storage(len(active))
        chunk = active[-batch:]
        del active[-batch:]
        with work.budget.temporary_scope():
            work.prefetch(
                request for task, reference, _ in chunk for request in tasks[task][2](reference)
            )
        for task, reference, depth in reversed(chunk):
            work.piece()
            with work.budget.temporary_scope():
                proved = tasks[task][1](reference)
            work.poll()
            if proved:
                continue
            if depth >= work.limits.maximum_subdivision_depth:
                raise _IntervalHomotopyRefusal(reason, tasks[task][0])
            active.extend((task, child, depth + 1) for child in _homotopy_children(reference))
            work.storage(len(active) + len(chunk))


def _homotopy_exact_matrix(matrix: tuple) -> _HomotopyInterval:
    return (
        np.asarray(
            [[fraction_endpoint(value, upper=False) for value in row] for row in matrix]
        ),
        np.asarray(
            [[fraction_endpoint(value, upper=True) for value in row] for row in matrix]
        ),
    )


def _homotopy_common_atlas(
    first: Cell,
    second: Cell,
    shared: set[int],
    *,
    budget: CoordinateEnclosureBudget,
) -> _TriangleAtlas | None:
    atlas = _triangle_atlas(first, second, shared, budget=budget)
    if atlas is not None:
        return atlas
    # A first-face tangent projection can collapse a perpendicular second face.
    # Propose a bisector projection instead, with a convex ABSTRACT diamond.
    # Its straight common edge lives in reference space, never in physical space.
    a, b = sorted(shared)
    corners = [
        np.asarray(_plane_corner_images(cell, budget=budget), dtype=np.float64)
        for cell in (first, second)
    ]
    origin = corners[0][first.vertices.index(a)]
    tangent = corners[0][first.vertices.index(b)] - origin
    length_squared = float(tangent @ tangent)
    if not math.isfinite(length_squared) or length_squared <= 0:
        return None
    edge_row = tangent / length_squared
    outer = [
        points[next(i for i, key in enumerate(cell.vertices) if key not in shared)] - origin
        for cell, points in zip((first, second), corners, strict=True)
    ]
    transverse = [value - float(value @ edge_row) * tangent for value in outer]
    lengths = [float(np.linalg.norm(value)) for value in transverse]
    if any(not math.isfinite(value) or value <= 0 for value in lengths):
        return None
    normal_row = transverse[0] / lengths[0] - transverse[1] / lengths[1]
    if not np.any(normal_row):
        return None
    projection = np.stack((edge_row, normal_row))
    projected_outer = [projection @ value for value in outer]
    if projected_outer[0][1] <= 0 or projected_outer[1][1] >= 0:
        return None
    height = max(
        1.0,
        *(4 * (value[0] - 0.5) ** 2 / abs(value[1]) for value in projected_outer),
    )
    if not math.isfinite(height):
        return None
    height_exact = Fraction(height)
    start, end = (Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))
    outer_points = (
        (Fraction(1, 2), height_exact),
        (Fraction(1, 2), -height_exact),
    )
    if not convex_triangle_union(start, end, *outer_points, budget=budget):
        return None
    transforms: list[
        tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]]
    ] = []
    for cell, point in zip((first, second), outer_points, strict=True):
        chart = [start if key == a else end if key == b else point for key in cell.vertices]
        matrix = tuple(
            tuple(chart[j + 1][i] - chart[0][i] for j in range(2)) for i in range(2)
        )
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        budget.reserve(1, 1024)
        transforms.append((
            (matrix[1][1] / determinant, -matrix[0][1] / determinant),
            (-matrix[1][0] / determinant, matrix[0][0] / determinant),
        ))
    # The rounded candidate is not evidence. Both source and target Jacobians
    # must subsequently pass the outward SPD proof in this fixed exact atlas.
    return _TriangleAtlas(projection, transforms)


def _homotopy_candidate_pairs(
    lower: np.ndarray, upper: np.ndarray, work: _IntervalHomotopyWork
) -> Generator[tuple[int, int], None, None]:
    with work.budget.temporary_scope(), work.budget.live_storage() as live:
        bvh = prepare_bvh(
            lower,
            upper,
            dtype=np.float64,
            _charge_work=work.budget.reserve,
            _reserve_storage=lambda size: work.budget.reserve(0, size),
        )
        block = 1024
        # Median-tree depth bounds pending DFS blocks. Include both host banks
        # and all simultaneous indexed box, mask and child arrays.
        live.set_bound(
            8192 + 1024 * len(lower)
            + block * (512 + 64 * (len(lower).bit_length() + 1))
        )
        for first, second in bvh_overlap_pair_blocks(
            bvh, bvh, include_touching=True,
            maximum_block_pairs=block, visit=work.budget.reserve,
        ):
            for a, b in zip(first, second, strict=True):
                if a >= b:
                    continue
                if work.pairs >= work.limits.maximum_candidate_pairs:
                    raise _IntervalHomotopyRefusal("interval_homotopy_candidate_pair_limit")
                work.pairs += 1
                yield int(a), int(b)


def _homotopy_monotone(
    items: Sequence[
        tuple[
            IntervalBoundaryCell, tuple[Expression, ...], np.ndarray, _HomotopyInterval,
            tuple[int, ...],
        ]
    ],
    work: _IntervalHomotopyWork,
) -> None:
    """Prove each (cell, jacobian, projection, transform, facets) item monotone."""
    tasks: list[_CoverTask] = []
    for cell, jacobian, projection, transform, entities in items:

        def predicate(
            reference: np.ndarray,
            cell: IntervalBoundaryCell = cell,
            jacobian: tuple[Expression, ...] = jacobian,
            projection: np.ndarray = projection,
            transform: _HomotopyInterval = transform,
        ) -> bool:
            return all(
                _homotopy_spd(jet, projection, transform)
                for jet in _homotopy_jets(
                    cell, jacobian, reference, work=work, orientation=True
                )
            )

        def requests(
            reference: np.ndarray, cell: IntervalBoundaryCell = cell
        ) -> tuple[_SourceRequest, ...]:
            return ((cell, reference, True),)

        tasks.append((entities, predicate, requests))
    # A fixed projection/atlas is essential: changing it at children would prove
    # only local rank, not global injectivity. SPD is convex in homotopy time.
    # Integrating the a.e. directional inequality along absolutely continuous
    # chart segments also covers continuous piecewise-C1 native chart joins.
    # The source hull includes RAW authored and completed jets: convexity also
    # proves orientation along the authored-face-to-representative correction.
    _homotopy_covers(tasks, work, "interval_homotopy_monotonicity_unresolved")


def _homotopy_norm_upper(value: _HomotopyInterval) -> float:
    absolute = np.maximum(np.abs(value[0]), np.abs(value[1]))
    square = interval_dot((absolute, absolute), (absolute, absolute))[1]
    return float(np.nextafter(np.sqrt(square), np.inf))


def _homotopy_deviation(
    items: list[tuple[IntervalBoundaryCell, tuple[Expression, ...], float]],
    work: _IntervalHomotopyWork,
) -> list[float]:
    """Bound each representative-to-target deviation within its allowance."""
    maxima = [0.0] * len(items)
    tasks: list[_CoverTask] = []
    for index, (cell, jacobian, tolerance) in enumerate(items):

        def predicate(
            reference: np.ndarray,
            index: int = index,
            cell: IntervalBoundaryCell = cell,
            jacobian: tuple[Expression, ...] = jacobian,
            tolerance: float = tolerance,
        ) -> bool:
            work.distance()
            point = np.broadcast_to(reference[0], (3, 2))
            source_point = work.source_interval(cell, point, jet=False)
            target_point = work.target_interval(cell, "target", cell.target, point)
            difference = interval_subtract(source_point, target_point)
            source_jet, target_jet = _homotopy_jets(
                cell, jacobian, reference, work=work
            )
            difference_lower, difference_upper = interval_subtract(source_jet, target_jet)
            displacement_lower, displacement_upper = interval_subtract(
                (np.min(reference, axis=0), np.max(reference, axis=0)),
                (reference[0], reference[0]),
            )
            increment = _homotopy_linear(
                (np.asarray(difference_lower), np.asarray(difference_upper)),
                (
                    np.asarray(displacement_lower)[:, None],
                    np.asarray(displacement_upper)[:, None],
                ),
            )
            residual_lower, residual_upper = interval_add(
                difference, (increment[0][:, 0], increment[1][:, 0])
            )
            bound = _homotopy_norm_upper(
                (np.asarray(residual_lower), np.asarray(residual_upper))
            )
            if bound > tolerance or not math.isfinite(bound):
                return False
            maxima[index] = max(maxima[index], bound)
            return True

        def requests(
            reference: np.ndarray, cell: IntervalBoundaryCell = cell
        ) -> tuple[_SourceRequest, ...]:
            return (
                (cell, np.broadcast_to(reference[0], (3, 2)), False),
                (cell, reference, True),
            )

        tasks.append(((cell.entity,), predicate, requests))
    _homotopy_covers(tasks, work, "interval_homotopy_deviation_unresolved")
    return maxima


def _homotopy_blend(
    source: _HomotopyInterval, target: _HomotopyInterval, time: tuple[float, float]
) -> _HomotopyInterval:
    # H is affine in t: hull of its endpoint enclosures avoids the dependency
    # inflation of independently multiplying (1-t) and t over the whole slab.
    # H(.,0)=S and H(.,1)=P exactly, so those endpoints need no arithmetic.
    endpoints = []
    for value in time:
        if value == 0.0:
            endpoints.append(source)
            continue
        if value == 1.0:
            endpoints.append(target)
            continue
        t = np.asarray(value)
        endpoints.append(
            interval_add(
                interval_multiply(
                    interval_subtract((np.asarray(1.0), np.asarray(1.0)), (t, t)),
                    source,
                ),
                interval_multiply((t, t), target),
            )
        )
    return (
        np.minimum(endpoints[0][0], endpoints[1][0]),
        np.maximum(endpoints[0][1], endpoints[1][1]),
    )


def _homotopy_image(
    cell: IntervalBoundaryCell,
    reference: np.ndarray,
    time: tuple[float, float],
    *,
    work: _IntervalHomotopyWork,
) -> _HomotopyInterval:
    source = work.source_interval(cell, reference, jet=False)
    target = work.target_interval(cell, "target", cell.target, reference)
    return _homotopy_checked_interval(_homotopy_blend(source, target, time), (3,))


def _homotopy_target_vertex_rays(
    target: Cell,
    jacobian: tuple[Expression, ...],
    vertex: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[ExactScalar, ...], ...]:
    corner = target.vertices.index(vertex)
    point = target.reference_vertices[corner]
    jet = tuple(
        expression_reference_evaluate(value, point, "simplex", budget=budget)
        for value in jacobian
    )
    reserve_algebraic_work(jet, 12, 6, 2, budget=budget)
    return tuple(
        tuple(
            sum(
                (
                    jet[2 * axis + j] * (other[j] - point[j])
                    for j in range(2)
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        for local, other in enumerate(target.reference_vertices) if local != corner
    )


def _homotopy_target_cone_signs(
    normal: np.ndarray,
    rays: tuple[tuple[ExactScalar, ...], ...],
    work: _IntervalHomotopyWork,
    bounds: _HomotopyInterval | None = None,
) -> tuple[bool, bool]:
    """Exact signs of n.r for the first two rays and the remaining rays.

    Outward rounded (k,3) ray ``bounds`` decide every sign whose interval dot
    product excludes zero; otherwise all signs are evaluated exactly.
    """
    if work.ray_tests + len(rays) > work.limits.maximum_ray_tests:
        raise _IntervalHomotopyRefusal("interval_homotopy_ray_test_limit")
    if bounds is not None:
        lower, upper = interval_multiply((normal[None, :], normal[None, :]), bounds)
        products = np.asarray(lower), np.asarray(upper)
        dot: _HomotopyInterval = (np.zeros(len(rays)), np.zeros(len(rays)))
        for axis in range(3):
            lower, upper = interval_add(dot, (products[0][:, axis], products[1][:, axis]))
            dot = np.asarray(lower), np.asarray(upper)
        positive, negative = dot[0] > 0, dot[1] < 0
        if np.all(positive | negative):
            work.ray_tests += len(rays)
            work.budget.reserve(3 * len(rays))
            return (
                bool(np.all(positive[:2]) and np.all(negative[2:])),
                bool(np.all(negative[:2]) and np.all(positive[2:])),
            )
    dots = []
    weights = rational_vector(normal)
    for ray in rays:
        work.ray_tests += 1
        reserve_algebraic_work((*weights, *ray), 3, 3, 2, budget=work.budget)
        dots.append(sum(
            (weight * value for weight, value in zip(weights, ray, strict=True)),
            Fraction(0),
        ))
    return (
        all(value > 0 for value in dots[:2]) and all(value < 0 for value in dots[2:]),
        all(value < 0 for value in dots[:2]) and all(value > 0 for value in dots[2:]),
    )


def _homotopy_vertex_prepared(
    index: int,
    cell: IntervalBoundaryCell,
    target: Cell,
    jacobian: tuple[Expression, ...],
    vertex: int,
    work: _IntervalHomotopyWork,
) -> _HomotopyVertex:
    """Reference rays and exact target rays of one cell corner, once per request."""
    key = (index, vertex)
    prepared = work.vertices.get(key)
    if prepared is None:
        corner = cell.vertices.index(vertex)
        rays = _homotopy_target_vertex_rays(target, jacobian, vertex, budget=work.budget)
        ray_bounds = None
        if all(isinstance(value, Fraction) for ray in rays for value in ray):
            ray_bounds = (
                np.asarray([
                    [
                        fraction_endpoint(
                            rational_coefficient(value, role="homotopy.vertex_ray"), upper=False
                        )
                        for value in ray
                    ]
                    for ray in rays
                ]),
                np.asarray([
                    [
                        fraction_endpoint(
                            rational_coefficient(value, role="homotopy.vertex_ray"), upper=True
                        )
                        for value in ray
                    ]
                    for ray in rays
                ]),
            )
        prepared = _HomotopyVertex(
            (
                _HOMOTOPY_REFERENCE[[i for i in range(3) if i != corner]]
                - _HOMOTOPY_REFERENCE[corner]
            ).T,
            np.repeat(_HOMOTOPY_REFERENCE[corner][None, :], 3, axis=0),
            rays,
            np.asarray([[float(value) for value in ray] for ray in rays], dtype=np.float64),
            ray_bounds,
        )
        work.vertices[key] = prepared
        work.storage(work.pending)
    return prepared


def _homotopy_subsets(count: int, size: int) -> np.ndarray:
    return np.asarray(list(combinations(range(count), size)), dtype=np.int64).reshape(-1, size)


def _homotopy_max_margins(rays: np.ndarray) -> np.ndarray:
    """Rounded unit n maximizing min n.u over each batch's unit rows u.

    ``rays`` is (P,R,3); rows of length zero are ignored and batches without a
    positive margin return NaN. The minimum-norm point of conv(U) is the
    max-margin direction and lies on a point, segment or triangle face of U, so
    those faces are enumerated. A proposal is never evidence; interval covers
    decide every separator.
    """
    lengths = np.linalg.norm(rays, axis=2)
    usable = np.isfinite(lengths) & (lengths > 0)
    units = np.where(
        usable[..., None], rays / np.where(usable, lengths, 1.0)[..., None], np.nan
    )
    points = [units]
    for size in (2, 3):
        subsets = _homotopy_subsets(units.shape[1], size)
        if not len(subsets):
            break
        base = units[:, subsets[:, 0]]
        edges = units[:, subsets[:, 1:]] - base[:, :, None, :]
        gram = np.einsum("psij,pskj->psik", edges, edges)
        with np.errstate(invalid="ignore"):
            regular = np.nan_to_num(np.linalg.det(gram), nan=0.0) > 1e-12
        safe = np.where(regular[..., None, None], gram, np.eye(size - 1))
        rhs = np.where(regular[..., None], -np.einsum("psij,psj->psi", edges, base), 0.0)
        weights = np.linalg.solve(safe, rhs[..., None])[..., 0]
        inside = regular & np.all(weights >= 0, axis=2) & (np.sum(weights, axis=2) <= 1)
        points.append(np.where(
            inside[..., None], base + np.einsum("psi,psij->psj", weights, edges), np.nan
        ))
    candidates = np.concatenate(points, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = candidates / np.linalg.norm(candidates, axis=2)[..., None]
        margins = np.where(
            usable[:, None, :], np.einsum("pcj,prj->pcr", normals, np.nan_to_num(units)), np.inf
        ).min(axis=2)
    margins = np.where(np.isfinite(margins), margins, -np.inf)
    best = np.argmax(margins, axis=1)
    chosen = normals[np.arange(len(best)), best]
    return np.where((margins[np.arange(len(best)), best] > 0)[:, None], chosen, np.nan)


def _homotopy_cone_rays(
    cells: tuple[IntervalBoundaryCell, IntervalBoundaryCell],
    prepared: tuple[_HomotopyVertex, _HomotopyVertex],
    jacobians: tuple[tuple[Expression, ...], tuple[Expression, ...]],
    work: _IntervalHomotopyWork,
) -> tuple[np.ndarray, np.ndarray]:
    """Sampled cone rays oriented positive on the first and negative on the second.

    Returns (16,3) rays: shared-vertex source jet midpoints and exact target
    rays, then whole-triangle source/target jet midpoints; the first eight per
    cell pair are the vertex-only rays.
    """
    near: list[np.ndarray] = []
    wide: list[np.ndarray] = []
    for cell, data, jacobian, side in zip(cells, prepared, jacobians, (1.0, -1.0), strict=True):
        vertex = work.source_interval(cell, data.point, jet=True)
        root = work.source_interval(cell, _HOMOTOPY_REFERENCE, jet=True)
        target = work.target_interval(cell, "jacobian", jacobian, _HOMOTOPY_REFERENCE)
        near.extend((
            side * ((0.5 * (vertex[0] + vertex[1])) @ data.directions).T,
            side * data.rounded_rays,
        ))
        wide.extend((
            side * ((0.5 * (root[0] + root[1])) @ data.directions).T,
            side * ((0.5 * (target[0] + target[1])).reshape(3, 2) @ data.directions).T,
        ))
    return np.concatenate(near + wide), np.concatenate(near)


def _homotopy_ray_proposals(
    cells: tuple[IntervalBoundaryCell, IntervalBoundaryCell],
    prepared: tuple[_HomotopyVertex, _HomotopyVertex],
    jacobians: tuple[tuple[Expression, ...], tuple[Expression, ...]],
    work: _IntervalHomotopyWork,
) -> Generator[np.ndarray, None, None]:
    """Rounded separators of sampled cone rays, positive on the first cell.

    The whole-triangle proposal comes first; the vertex-only separator is
    formed only if the first one is not accepted.
    """
    wide, near = _homotopy_cone_rays(cells, prepared, jacobians, work)
    first = _homotopy_max_margins(wide[None])[0]
    if np.all(np.isfinite(first)):
        yield first
    second = _homotopy_max_margins(near[None])[0]
    if np.all(np.isfinite(second)) and not np.array_equal(first, second):
        yield second


def _homotopy_vertex_cone(
    indices: tuple[int, int],
    cells: tuple[IntervalBoundaryCell, IntervalBoundaryCell],
    targets: tuple[Cell, Cell],
    jacobians: tuple[tuple[Expression, ...], tuple[Expression, ...]],
    vertex: int,
    work: _IntervalHomotopyWork,
) -> None:
    """Separate two vertex-adjacent cones: n>0 on the first, n<0 on the second."""
    prepared = (
        _homotopy_vertex_prepared(indices[0], cells[0], targets[0], jacobians[0], vertex, work),
        _homotopy_vertex_prepared(indices[1], cells[1], targets[1], jacobians[1], vertex, work),
    )
    target_rays = prepared[0].target_rays + prepared[1].target_rays
    first_bounds, second_bounds = prepared[0].ray_bounds, prepared[1].ray_bounds
    target_bounds = (
        None
        if first_bounds is None or second_bounds is None
        else (
            np.concatenate((first_bounds[0], second_bounds[0])),
            np.concatenate((first_bounds[1], second_bounds[1])),
        )
    )
    # The degenerate reference triangle at the shared vertex encloses the
    # actual source jet there (one-sided hull at native chart joins).
    vertex_jets = [
        work.source_interval(cell, data.point, jet=True)
        for cell, data in zip(cells, prepared, strict=True)
    ]

    # Necessary conditions only ORDER and skip strict separators: a candidate
    # whose ray upper bound at the vertex itself is nonpositive at either
    # homotopy end can never satisfy the strict cone inequality on a slab
    # containing that end. Every accepted separator is still proved below.
    def screened(normal: np.ndarray) -> tuple[float, bool]:
        lowest = math.inf
        length = float(np.linalg.norm(normal))
        for jet, data, side in zip(vertex_jets, prepared, (1.0, -1.0), strict=True):
            oriented = (side * normal)[None, :]
            rays = _homotopy_linear(
                _homotopy_linear((oriented, oriented), jet),
                (data.directions, data.directions),
            )
            if not np.all(rays[1] > 0):
                return -math.inf, False
            lowest = min(lowest, float(np.min(rays[0])) / length)
        return lowest, True

    proposals: list[tuple[np.ndarray, bool, bool]] = []
    proposed = _homotopy_ray_proposals(cells, prepared, jacobians, work)
    planar: list[tuple[np.ndarray, bool, bool]] | None = None

    def attempts() -> Generator[tuple[np.ndarray, bool, bool], None, None]:
        # Cone-ray proposals are formed lazily; the authored target plane
        # family is ranked only if none of them proves a slab.
        nonlocal planar
        yield from list(proposals)
        for normal in proposed:
            with work.budget.temporary_scope():
                target_possible = _homotopy_target_cone_signs(
                    normal, target_rays, work, target_bounds
                )[0]
            proposals.append((normal, screened(normal)[1], target_possible))
            yield proposals[-1]
        if planar is None:
            candidates = _plane_candidates(targets[0], targets[1], {vertex}, budget=work.budget)
            normals = [
                np.asarray(normal, dtype=np.float64)
                for normal in candidates.normals
                if any(normal)
            ]
            ranked = []
            for candidate, normal in enumerate(normals):
                if not np.all(np.isfinite(normal)):
                    continue
                with work.budget.temporary_scope():
                    target_signs = _homotopy_target_cone_signs(
                        normal, target_rays, work, target_bounds
                    )
                for sign_index, sign in enumerate((1.0, -1.0)):
                    oriented = sign * normal
                    lowest, source_possible = screened(oriented)
                    score = lowest if source_possible and target_signs[sign_index] else -math.inf
                    ranked.append((
                        (-score, candidate, sign_index),
                        (oriented, source_possible, target_signs[sign_index]),
                    ))
            ranked.sort(key=lambda item: item[0])
            planar = [attempt for _, attempt in ranked]
        yield from planar

    entities = (cells[0].entity, cells[1].entity)
    active = [((0.0, 1.0), 0)]
    while active:
        work.storage(len(active))
        time, depth = active.pop()
        proved = False
        for normal, source_possible, target_possible in attempts():
            # A nonpositive exact target ray at the actual vertex excludes
            # this STRICT separator, not the embedding or the contact.
            if (time[0] == 0.0 and not source_possible) or (
                time[1] == 1.0 and not target_possible
            ):
                continue
            try:
                for cell, jacobian, data, side in zip(
                    cells, jacobians, prepared, (1.0, -1.0), strict=True
                ):
                    oriented = (side * normal)[None, :]

                    def predicate(
                        reference: np.ndarray,
                        cell: IntervalBoundaryCell = cell,
                        jacobian: tuple[Expression, ...] = jacobian,
                        oriented: np.ndarray = oriented,
                        directions: np.ndarray = data.directions,
                    ) -> bool:
                        source, target = _homotopy_jets(
                            cell, jacobian, reference, work=work
                        )
                        jet = _homotopy_blend(source, target, time)
                        rays = _homotopy_linear(
                            _homotopy_linear((oriented, oriented), jet),
                            (directions, directions),
                        )
                        return bool(np.all(rays[0] > 0))

                    def requests(
                        reference: np.ndarray, cell: IntervalBoundaryCell = cell
                    ) -> tuple[_SourceRequest, ...]:
                        return ((cell, reference, True),)

                    _homotopy_covers(
                        [(entities, predicate, requests)],
                        work,
                        "interval_homotopy_vertex_cone_unresolved",
                        depth_offset=depth,
                        batch=2,
                    )
            except _IntervalHomotopyRefusal as error:
                if str(error) != "interval_homotopy_vertex_cone_unresolved":
                    raise
            else:
                proved = True
                break
        if proved:
            continue
        if depth >= work.limits.maximum_subdivision_depth:
            raise _IntervalHomotopyRefusal("interval_homotopy_vertex_cone_unresolved", entities)
        middle = (time[0] + time[1]) * 0.5
        if middle in time:
            raise _IntervalHomotopyRefusal("interval_homotopy_time_resolution", entities)
        work.storage(len(active) + 2)
        active.extend((((time[0], middle), depth + 1), ((middle, time[1]), depth + 1)))
    # For x=v+a*d0+b*d1, a,b>=0, absolute continuity along [v,x] and
    # the a.e. one-sided-jet inequality give n.(H(x,t)-H(v,t)) the strict
    # prescribed sign unless a=b=0; a derivative at every join is unnecessary.
    # The authenticated vertex trace makes H(v,t) the SAME moving vertex.


def _homotopy_tube_separated(
    first: IntervalBoundaryCell,
    second: IntervalBoundaryCell,
    targets: tuple[tuple[Polynomial, ...], tuple[Polynomial, ...]],
    deviations: tuple[float, float],
    directions: list[np.ndarray],
    work: _IntervalHomotopyWork,
) -> bool:
    """Separate the two moving images inside their certified target tubes.

    Each cell's deviation proof bounds sup_x |S(x)-P(x)| by D. Since
    H(x,t)-P(x)=(1-t)(S(x)-P(x)), every image H(.,t) stays in the closed
    D-tube of its exact target for EVERY t, so tube separation needs no time
    refinement. Separation is tested by outward axis boxes and by outward
    projections of the exact Bernstein nets on the given rounded directions.
    Failure is only inconclusive; the caller then refines the synchronized
    swept images.
    """
    projections: list[tuple[np.ndarray, list[float]]] = []
    for direction in directions:
        if not np.all(np.isfinite(direction)) or not np.any(direction):
            continue
        direction = direction / float(np.max(np.abs(direction)))
        norm = _homotopy_norm_upper((direction, direction))
        projections.append((
            direction,
            [float(np.nextafter(norm * radius, np.inf)) for radius in deviations],
        ))
    active = [(_HOMOTOPY_REFERENCE, _HOMOTOPY_REFERENCE, 0)]
    while active:
        work.storage(len(active))
        a, b, depth = active.pop()
        work.piece()
        with work.budget.temporary_scope():
            enclosures = [
                work.target_enclosure(cell, "target", target, reference)
                for cell, target, reference in zip(
                    (first, second), targets, (a, b), strict=True
                )
            ]
            tubes = [
                (np.nextafter(box[0] - radius, -np.inf), np.nextafter(box[1] + radius, np.inf))
                for (box, _), radius in zip(enclosures, deviations, strict=True)
            ]
            separated = bool(
                np.any(tubes[0][1] < tubes[1][0]) or np.any(tubes[1][1] < tubes[0][0])
            )
            nets = [net for _, net in enclosures]
            for direction, projected_radii in projections:
                if separated or nets[0] is None or nets[1] is None:
                    break
                ranges = [_homotopy_net_range(direction, cast(_HomotopyInterval, net)) for net in nets]
                low = [
                    float(np.nextafter(value[0] - radius, -np.inf))
                    for value, radius in zip(ranges, projected_radii, strict=True)
                ]
                high = [
                    float(np.nextafter(value[1] + radius, np.inf))
                    for value, radius in zip(ranges, projected_radii, strict=True)
                ]
                separated = high[0] < low[1] or high[1] < low[0]
        work.poll()
        if separated:
            continue
        if depth >= work.limits.maximum_subdivision_depth:
            return False
        work.storage(len(active) + 2)
        widths = [float(np.max(box[1] - box[0])) for box, _ in enclosures]
        if widths[0] >= widths[1]:
            active.extend((child, b, depth + 1) for child in _homotopy_children(a))
        else:
            active.extend((a, child, depth + 1) for child in _homotopy_children(b))
    return True


def _homotopy_net_range(direction: np.ndarray, net: _HomotopyInterval) -> tuple[float, float]:
    """Outward range of the scalar net n.b_k, which bounds n.P on its subtriangle."""
    lower, upper = interval_multiply((direction[None, :], direction[None, :]), net)
    products = np.asarray(lower), np.asarray(upper)
    total: _HomotopyInterval = (np.zeros(len(net[0])), np.zeros(len(net[0])))
    for axis in range(net[0].shape[1]):
        lower, upper = interval_add(total, (products[0][:, axis], products[1][:, axis]))
        total = np.asarray(lower), np.asarray(upper)
    return float(np.min(total[0])), float(np.max(total[1]))


_HOMOTOPY_SAMPLE_LATTICE = np.asarray(
    [(i / 4, j / 4) for i in range(5) for j in range(5 - i)], dtype=np.float64
)


def _homotopy_target_samples(target: tuple[Polynomial, ...]) -> np.ndarray:
    """Rounded target points on an order-four reference lattice; never evidence."""
    x, y = _HOMOTOPY_SAMPLE_LATTICE[:, 0], _HOMOTOPY_SAMPLE_LATTICE[:, 1]
    return np.stack(
        [
            sum(
                (float(coefficient) * x ** exponent[0] * y ** exponent[1]
                 for exponent, coefficient in value.items()),
                np.zeros_like(x),
            )
            for value in target
        ],
        axis=1,
    )


def _homotopy_disjoint(
    items: list[
        tuple[
            IntervalBoundaryCell,
            IntervalBoundaryCell,
            tuple[tuple[Polynomial, ...], tuple[Polynomial, ...]],
            tuple[float, float],
        ]
    ],
    work: _IntervalHomotopyWork,
) -> None:
    """Separate every nonadjacent pair on synchronized homotopy slabs.

    Rounded target lattice samples only propose projection directions: the
    closest sampled points' difference and the anchor-centroid difference. The
    certified target-tube test is skipped when sampled target points already
    lie closer than the two tube radii: closed tubes then intersect and that
    sufficient test cannot succeed. Remaining pairs share one worklist, popped
    in batches whose source images are enclosed together. Each entry keeps its
    own pair, pieces, slab and fair depth limit; images separate by an axis or
    along the closest-sample direction.
    """
    samples: dict[int, np.ndarray] = {}
    for first, second, targets, _ in items:
        for cell, target in zip((first, second), targets, strict=True):
            if cell.entity not in samples:
                samples[cell.entity] = _homotopy_target_samples(target)
    pending = []
    directions: dict[int, np.ndarray | None] = {}
    for index, (first, second, targets, deviations) in enumerate(items):
        gaps = samples[second.entity][None, :, :] - samples[first.entity][:, None, :]
        lengths = np.linalg.norm(gaps, axis=2)
        nearest = np.unravel_index(int(np.argmin(lengths)), lengths.shape)
        closest = gaps[nearest]
        centroid = np.mean(second.anchor, axis=0) - np.mean(first.anchor, axis=0)
        if float(lengths[nearest]) >= sum(deviations) and _homotopy_tube_separated(
            first, second, targets, deviations, [closest, centroid], work
        ):
            continue
        pending.append(index)
        scale = float(np.max(np.abs(closest)))
        directions[index] = (
            closest / scale if math.isfinite(scale) and scale > 0 else None
        )
    active = [
        (index, _HOMOTOPY_REFERENCE, _HOMOTOPY_REFERENCE, (0.0, 1.0), (0, 0, 0))
        for index in reversed(pending)
    ]
    while active:
        work.storage(len(active))
        chunk = active[-_HOMOTOPY_BATCH:]
        del active[-_HOMOTOPY_BATCH:]
        with work.budget.temporary_scope():
            work.prefetch(
                request
                for index, a, b, _, _ in chunk
                for request in ((items[index][0], a, False), (items[index][1], b, False))
            )
        for index, a, b, time, depths in reversed(chunk):
            first, second = items[index][0], items[index][1]
            work.piece()
            work.distance()
            with work.budget.temporary_scope():
                first_image = _homotopy_image(first, a, time, work=work)
                second_image = _homotopy_image(second, b, time, work=work)
            work.poll()
            if np.any(first_image[1] < second_image[0]) or np.any(
                second_image[1] < first_image[0]
            ):
                continue
            direction = directions[index]
            if direction is not None:
                row = (direction[None, :], direction[None, :])
                projected = [
                    _homotopy_linear(row, (image[0][:, None], image[1][:, None]))
                    for image in (first_image, second_image)
                ]
                if (
                    projected[0][1][0, 0] < projected[1][0][0, 0]
                    or projected[1][1][0, 0] < projected[0][0][0, 0]
                ):
                    continue
            entities = (first.entity, second.entity)
            if sum(depths) >= work.limits.maximum_subdivision_depth:
                raise _IntervalHomotopyRefusal(
                    "interval_homotopy_nonadjacent_contact_unresolved", entities
                )
            # Fair reference/reference/time refinement. Both images ALWAYS use
            # the same slab; intersections of independent swept tubes are not
            # witnesses.
            axis = min(range(3), key=lambda value: depths[value])
            child_depths = (
                depths[0] + int(axis == 0),
                depths[1] + int(axis == 1),
                depths[2] + int(axis == 2),
            )
            if axis == 0:
                active.extend(
                    (index, child, b, time, child_depths) for child in _homotopy_children(a)
                )
            elif axis == 1:
                active.extend(
                    (index, a, child, time, child_depths) for child in _homotopy_children(b)
                )
            else:
                middle = (time[0] + time[1]) * 0.5
                if middle in time:
                    raise _IntervalHomotopyRefusal(
                        "interval_homotopy_time_resolution", entities
                    )
                active.extend(
                    (index, a, b, child, child_depths)
                    for child in ((time[0], middle), (middle, time[1]))
                )
            work.storage(len(active) + len(chunk))


def _homotopy_p2_coordinates(
    coordinates: tuple[Expression, ...],
) -> tuple[Polynomial, ...]:
    polynomials: list[Polynomial] = []
    for value in coordinates:
        if not isinstance(value, dict):
            raise _IntervalHomotopyRefusal("interval_homotopy_target_not_p2")
        polynomial = cast(Polynomial, value)
        if any(
            len(exponent) != 2 or min(exponent) < 0 or sum(exponent) > 2
            for exponent in polynomial
        ):
            raise _IntervalHomotopyRefusal("interval_homotopy_target_not_p2")
        polynomials.append(polynomial)
    return tuple(polynomials)


def _homotopy_target_identity(target: tuple[Polynomial, ...]) -> tuple:
    def scalar(value: ExactScalar) -> tuple:
        terms = value.terms if isinstance(value, RadicalElement) else ((1, value),)
        return tuple((radicand, str(coefficient.numerator), str(coefficient.denominator)) for radicand, coefficient in terms)

    return tuple(
        tuple((exponent, scalar(coefficient)) for exponent, coefficient in sorted(value.items()))
        for value in target
    )


def certify_interval_boundary_homotopy(
    cells: tuple[IntervalBoundaryCell, ...],
    limits: MeshCertificateLimits,
    /,
    *,
    tolerance: float,
    budget: CoordinateEnclosureBudget,
) -> IntervalBoundaryHomotopy:
    """Prove representative-to-P2 isotopy and original authored-source deviation.

    The caller authenticates source-authority coverage and target boundary
    completeness. Here closure, manifold links, all shared traces, one-sided rank,
    self/pair injectivity, synchronized-time separation and deviation are
    independent proof obligations.

    The raw-face-to-representative correction is separately bounded in distance
    and orientation. Its outward distance bound consumes the SAME tolerance;
    contact is proved for the closed representative, not for unglued raw faces.
    A failed sufficient interval inequality is
    unresolved, never an intersection witness. Resource exceptions from the
    shared coordinate ledger deliberately propagate unchanged.
    """
    entities: tuple[int, ...] = ()
    work = None
    try:
        if not cells or not math.isfinite(tolerance) or tolerance < 0:
            raise _IntervalHomotopyRefusal("interval_homotopy_invalid_request")
        with (
            budget.temporary_scope(),
            budget.bound_stage(limits.maximum_work_units, limits.maximum_scratch_bytes),
            budget.live_storage() as live,
        ):
            work = _IntervalHomotopyWork(limits, budget, live, 8192 + 4096 * len(cells))
            targets: list[Cell] = []
            target_polynomials: list[tuple[Polynomial, ...]] = []
            jacobians: list[tuple[Expression, ...]] = []
            authored_bounds: list[float] = []
            edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
            vertex_uses: dict[int, list[int]] = {}
            facets: set[tuple[int, ...]] = set()
            entity_ids: set[int] = set()
            identities = []
            for index, cell in enumerate(cells):
                entities = (cell.entity,)
                work.piece()
                if (
                    len(cell.vertices) != 3
                    or len(set(cell.vertices)) != 3
                    or cell.orientation not in (-1, 1)
                    or len(cell.target) != 3
                    or cell.anchor.shape != (3, 3)
                    or not np.all(np.isfinite(cell.anchor))
                    or not cell.source.map_id
                ):
                    raise _IntervalHomotopyRefusal("interval_homotopy_invalid_cell")
                authored_bound = float(cell.source.authored_deviation_bound)
                if not math.isfinite(authored_bound) or authored_bound < 0:
                    raise _IntervalHomotopyRefusal("interval_homotopy_invalid_authored_bound")
                if authored_bound > tolerance:
                    raise _IntervalHomotopyRefusal("interval_homotopy_authored_deviation_exceeds_tolerance")
                authored_bounds.append(authored_bound)
                key = tuple(sorted(cell.vertices))
                if key in facets or cell.entity in entity_ids:
                    raise _IntervalHomotopyRefusal("interval_homotopy_duplicate_facet")
                facets.add(key)
                entity_ids.add(cell.entity)
                target_coordinates = _homotopy_p2_coordinates(cell.target)
                target_polynomials.append(target_coordinates)
                if any(
                    bernstein_node_count(value, "simplex", 2) > limits.maximum_bernstein_nodes
                    for value in target_coordinates
                ):
                    raise _IntervalHomotopyRefusal("interval_homotopy_bernstein_limit")
                target = Cell(
                    target_coordinates, "simplex", "triangle", 2, cell.vertices,
                    embedding_vertices("triangle"),
                )
                if any(
                    tuple(expression_reference_evaluate(value, reference, "simplex", budget=budget) for value in target_coordinates)
                    != rational_vector(cell.anchor[corner])
                    for corner, reference in enumerate(target.reference_vertices)
                ):
                    raise _IntervalHomotopyRefusal("interval_homotopy_anchor_mismatch")
                targets.append(target)
                jacobians.append(
                    tuple(derivative(value, axis, budget=budget) for value in target_coordinates for axis in range(2))
                )
                traces = []
                for local, vertex in enumerate(cell.vertices):
                    identity = cell.source.trace_identity((local,))
                    if not identity:
                        raise _IntervalHomotopyRefusal("interval_homotopy_source_vertex_unsupported")
                    traces.append(identity)
                    vertex_uses.setdefault(vertex, []).append(index)
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    start, end = cell.vertices[a], cell.vertices[b]
                    edge = (start, end) if start < end else (end, start)
                    ordered = (a, b) if start < end else (b, a)
                    identity = cell.source.trace_identity(ordered)
                    if not identity:
                        raise _IntervalHomotopyRefusal("interval_homotopy_source_edge_unsupported")
                    traces.append(identity)
                    edges.setdefault(edge, []).append((index, cell.orientation * (1 if start < end else -1)))
                identities.append((
                    cell.source.map_id, cell.entity, cell.vertices, cell.orientation,
                    cell.anchor.copy(), _homotopy_target_identity(target_coordinates), tuple(traces),
                    authored_bound,
                ))
            for edge, uses in edges.items():
                if len(uses) != 2 or uses[0][1] + uses[1][1] != 0:
                    entities = tuple(cells[index].entity for index, _ in uses)
                    raise _IntervalHomotopyRefusal("interval_homotopy_boundary_not_closed_oriented")
            for vertex, uses in vertex_uses.items():
                # Edge cancellation alone admits pinched vertices. The link
                # must be one connected degree-two cycle.
                link: dict[int, set[int]] = {index: set() for index in uses}
                for index in uses:
                    for other in cells[index].vertices:
                        if other != vertex:
                            edge_key = (vertex, other) if vertex < other else (other, vertex)
                            for neighbor, _ in edges[edge_key]:
                                if neighbor != index:
                                    link[index].add(neighbor)
                reached, pending = set(), [uses[0]]
                while pending:
                    index = pending.pop()
                    if index not in reached:
                        reached.add(index)
                        pending.extend(link[index] - reached)
                if len(reached) != len(uses) or any(len(neighbors) != 2 for neighbors in link.values()):
                    entities = tuple(cells[index].entity for index in uses)
                    raise _IntervalHomotopyRefusal("interval_homotopy_nonmanifold_vertex")
                first = uses[0]
                identity = cells[first].source.trace_identity((cells[first].vertices.index(vertex),))
                for index in uses[1:]:
                    entities = (cells[first].entity, cells[index].entity)
                    if (
                        cells[index].source.trace_identity((cells[index].vertices.index(vertex),)) != identity
                        or not _same_entity_trace(
                            targets[first], targets[index], [vertex], 0,
                            _HOMOTOPY_REFERENCE[[cells[first].vertices.index(vertex)]],
                            _HOMOTOPY_REFERENCE[[cells[index].vertices.index(vertex)]],
                            budget=budget,
                        )
                    ):
                        raise _IntervalHomotopyRefusal("interval_homotopy_vertex_trace_mismatch")
            for edge, uses in edges.items():
                first, second = (use[0] for use in uses)
                entities = (cells[first].entity, cells[second].entity)
                trace_ids = [
                    cells[index].source.trace_identity(tuple(cells[index].vertices.index(vertex) for vertex in edge))
                    for index in (first, second)
                ]
                if trace_ids[0] != trace_ids[1] or not cell_traces_equal(targets[first], targets[second], budget=budget):
                    raise _IntervalHomotopyRefusal("interval_homotopy_edge_trace_mismatch")
            distance_upper = 0.0
            identity_transform = (np.eye(2), np.eye(2))
            projections = []
            for cell, target in zip(cells, targets, strict=True):
                entities = (cell.entity,)
                projection = _midpoint_projection(jacobian_midpoint(target, budget=budget), 2)
                if projection is None:
                    raise _IntervalHomotopyRefusal("interval_homotopy_singular_target_chart")
                projections.append(projection)
            entities = ()
            _homotopy_monotone(
                [
                    (cell, jacobian, projection, identity_transform, (cell.entity,))
                    for cell, jacobian, projection in zip(cells, jacobians, projections, strict=True)
                ],
                work,
            )
            allowances = [
                float(interval_subtract(
                    (np.asarray(tolerance), np.asarray(tolerance)),
                    (np.asarray(authored_bound), np.asarray(authored_bound)),
                )[0])
                for authored_bound in authored_bounds
            ]
            deviations = _homotopy_deviation(
                list(zip(cells, jacobians, allowances, strict=True)), work
            )
            for cell, completed_distance, authored_bound in zip(
                cells, deviations, authored_bounds, strict=True
            ):
                entities = (cell.entity,)
                authored_distance = float(interval_add(
                    (np.asarray(completed_distance), np.asarray(completed_distance)),
                    (np.asarray(authored_bound), np.asarray(authored_bound)),
                )[1])
                if not math.isfinite(authored_distance) or authored_distance > tolerance:
                    raise _IntervalHomotopyRefusal("interval_homotopy_deviation_unresolved")
                distance_upper = max(distance_upper, authored_distance)
            lower, upper = np.empty((len(cells), 3)), np.empty((len(cells), 3))
            for start in range(0, len(cells), _HOMOTOPY_BATCH):
                with budget.temporary_scope():
                    work.prefetch(
                        (cell, _HOMOTOPY_REFERENCE, False)
                        for cell in cells[start : start + _HOMOTOPY_BATCH]
                    )
                for index in range(start, min(start + _HOMOTOPY_BATCH, len(cells))):
                    entities = (cells[index].entity,)
                    work.distance()
                    lower[index], upper[index] = _homotopy_image(
                        cells[index], _HOMOTOPY_REFERENCE, (0.0, 1.0), work=work
                    )
            # Swept boxes only discard pairs with a rigorous disjoint enclosure.
            # Overlapping boxes are NOT contact witnesses: refine common time.
            # Pairs are grouped by contact type; every pair keeps its obligation.
            edge_pairs: list[tuple[int, int, set[int]]] = []
            vertex_pairs: list[tuple[int, int, int]] = []
            separate_pairs: list[tuple[int, int]] = []
            with closing(_homotopy_candidate_pairs(lower, upper, work)) as pairs:
                for first, second in pairs:
                    entities = (cells[first].entity, cells[second].entity)
                    work.piece()
                    work.retain(1)
                    shared = set(cells[first].vertices).intersection(cells[second].vertices)
                    if len(shared) == 2:
                        edge_pairs.append((first, second, shared))
                    elif len(shared) == 1:
                        vertex_pairs.append((first, second, next(iter(shared))))
                    else:
                        separate_pairs.append((first, second))
            items = []
            for first, second, shared in edge_pairs:
                entities = (cells[first].entity, cells[second].entity)
                atlas = _homotopy_common_atlas(
                    targets[first], targets[second], shared, budget=budget
                )
                if atlas is None:
                    raise _IntervalHomotopyRefusal(
                        "interval_homotopy_common_convex_atlas_unresolved"
                    )
                items.extend(
                    (
                        cells[index], jacobians[index], atlas.projection,
                        _homotopy_exact_matrix(transform), entities,
                    )
                    for index, transform in zip((first, second), atlas.transforms, strict=True)
                )
                work.retain(2)
            entities = ()
            _homotopy_monotone(items, work)
            work.retain(-len(items))
            del items
            for start in range(0, len(vertex_pairs), _HOMOTOPY_BATCH // 4):
                chunk = vertex_pairs[start : start + _HOMOTOPY_BATCH // 4]
                with budget.temporary_scope():
                    work.prefetch(
                        (cells[index], reference, True)
                        for first, second, vertex in chunk
                        for index in (first, second)
                        for reference in (
                            _HOMOTOPY_REFERENCE,
                            np.repeat(
                                _HOMOTOPY_REFERENCE[cells[index].vertices.index(vertex)][None, :],
                                3,
                                axis=0,
                            ),
                        )
                    )
                for first, second, vertex in chunk:
                    entities = (cells[first].entity, cells[second].entity)
                    _homotopy_vertex_cone(
                        (first, second), (cells[first], cells[second]),
                        (targets[first], targets[second]),
                        (jacobians[first], jacobians[second]), vertex, work,
                    )
            entities = ()
            _homotopy_disjoint(
                [
                    (
                        cells[first], cells[second],
                        (target_polynomials[first], target_polynomials[second]),
                        (deviations[first], deviations[second]),
                    )
                    for first, second in separate_pairs
                ],
                work,
            )
            for cell, identity in zip(cells, identities, strict=True):
                if (
                    cell.source.map_id != identity[0]
                    or not np.array_equal(cell.anchor, identity[4])
                    or _homotopy_target_identity(_homotopy_p2_coordinates(cell.target)) != identity[5]
                    or float(cell.source.authored_deviation_bound) != identity[7]
                ):
                    entities = (cell.entity,)
                    raise _IntervalHomotopyRefusal("interval_homotopy_source_state_changed")
            work.poll()
            proof_id = canonical_fingerprint({
                "kind": "continuous-native-interval-boundary-homotopy",
                "cells": tuple(identities),
                "limits": limits.limits_id,
                "tolerance": tolerance,
                "distance_upper": distance_upper,
            })
            return IntervalBoundaryHomotopy(True, distance_upper, "", (), work.pieces, proof_id)
    except _IntervalHomotopyRefusal as error:
        return IntervalBoundaryHomotopy(
            False,
            math.inf,
            str(error),
            entities if error.entities is None else error.entities,
            0 if work is None else work.pieces,
            None,
        )
