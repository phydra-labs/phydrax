#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry transitions of cell coordinate maps across topology and motion epochs.

A `CellGeometryTransition` builds the successor `CellGeometrySpec` of one topology
or coordinate epoch from the source coordinate map and records how: which source
cell and reference coordinates every target cell was evaluated from, which target
cell owns every target geometry node, how far shared nodes evaluated from
different cells disagree, the Bernstein bound of any approximation, and the mapped
measure before and after.

Nested refinement composes the source Lagrange map of degree ``p`` with the affine
child reference map of each target simplex. ``P_p`` is closed under affine
reparametrization, so interpolating the composition at the target element's nodes
reproduces the source map exactly on every target cell (up to rounding of the
evaluation). Coarsening interpolates the piecewise fine map at the coarse element
nodes; the coarse map is generally not the fine map, so it is accepted only under
an explicit approximation policy whose sup-norm bound is the maximum Bernstein
coefficient of the difference on every fine cell (convex-hull property). Vertex
displacement moves every coordinate node by the affine interpolation of its cell's
corner displacements, which keeps curved faces curved and is algebraic for that
declared motion extension.
"""

from __future__ import annotations

import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import (
    assert_never,
    Literal,
    NamedTuple,
)
from weakref import ref, ReferenceType

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

import phydrax.discretization.fem._reference_rules as reference_rules_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.discretization._coordinate_budget import reserve_polynomial
from phydrax.discretization._reference_cell_types import CellGeometryElement
from phydrax.geometry._rational import fraction_endpoint

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._validation import nonnegative_finite_float
from ..ein import contract
from ..geometry import (
    _meshing_domain as imported_geometry__meshing_domain,
    _surface_source_support as imported_geometry__surface_source_support,
)
from ..geometry._rational import rational_vector
from ..geometry.brep import _patches as imported_geometry_brep__patches
from ..linalg import (
    determinant_small_linear,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..linalg._hermitian_spectral import fraction_sqrt_interval
from ..linalg._small_batched import cofactor_determinant_3
from ..typing import checked
from . import (
    _cell_geometry as imported_cell_geometry,
    _cell_geometry_transition_budget as transition_budget_owner,
    _cell_geometry_transition_types as transition_types,
    _chart_reconstruction as chart_reconstruction_owner,
    _coordinate_enclosure as imported_coordinate_enclosure,
    _mapped_geometry_measure as mapped_measure_owner,
    _sphere_chart_records as sphere_chart_owner,
    _surface_chart_records as surface_chart_owner,
)
from ._cell_geometry import (
    BarycentricCellGeometryElement,
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    exact_barycentric_action,
    LayerColumnCellGeometryElement,
    owned_coordinate_preparation,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    require_scalar_coordinate_element,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget
from ._coordinate_enclosure import (
    add,
    affine_arguments,
    compose,
    coordinate_corner_images,
    coordinate_expressions,
    coordinate_polynomials,
    Expression,
    expression_add,
    expression_scale,
    polynomial_bounds,
    rounded_point,
    scale,
    source_basis,
)
from ._coordinate_identity import coordinate_scope_key
from ._coordinate_operands import (
    _coordinate_operand_request,
    coordinate_input_owner_fields,
    OwnerFields,
    require_coordinate_input_containers,
    require_coordinate_input_owners,
)
from ._exact_algebraic_geometry import (
    coefficient_bits,
    coefficient_endpoint,
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
    rational_coefficient,
    reserve_algebraic_work,
)
from ._exact_plc_geometry import (
    ExactPlcBudget,
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_rational_geometry import ExactRationalCellGeometrySource
from ._reference_cell import reference_cell_topology


_SIMPLEX_DIMENSIONS = {"interval": 1, "triangle": 2, "tetrahedron": 3}
_EPSILON = float(np.finfo(np.float64).eps)


# Host-only immutable preparation: coordinate maps are evaluated once per epoch
# transition from NumPy topology tables; the accepted successor geometry is the
# device-facing product.


class SimplexGeometryBlock(NamedTuple):
    element: reference_types_owner.FiniteElementSpec
    routes: NDArray[np.int64]
    cell_ids: NDArray[np.int64]
    dimension: int
    degree: int


@checked
def simplex_blocks(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    role: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[SimplexGeometryBlock, ...]:
    """Per-block full simplex Lagrange coordinate elements of ``geometry``."""

    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    blocks = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        dimension = _SIMPLEX_DIMENSIONS.get(block.cell_kind)
        match element, dimension:
            case reference_types_owner.FiniteElementSpec(), int():
                pass
            case _:
                raise ValueError(
                    f"geometry_transition.{role}.elements must be full simplex Lagrange maps; got {block.cell_kind} block {block.name!r}"
                )
        complete = imported_cell_geometry.coordinate_lagrange_element(
            block.cell_kind, element.degree
        )
        if element.element_id != complete.element_id:
            raise ValueError(
                f"geometry_transition.{role}.elements must span the full Lagrange space; got incompatible element in block {block.name!r}"
            )
        blocks.append(
            SimplexGeometryBlock(
                element,
                np.asarray(route, dtype=np.int64),
                np.asarray(block.global_ids, dtype=np.int64),
                dimension,
                element.degree,
            )
        )
    return tuple(blocks)


def _barycentric(points: np.ndarray, /) -> np.ndarray:
    return np.concatenate((1.0 - np.sum(points, axis=-1, keepdims=True), points), axis=-1)


# Polynomial simplex coordinate maps whose mapped hat measures are algebraic.
type _PolynomialSimplexElement = (
    reference_types_owner.FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
)


def _tabulate(
    element: _PolynomialSimplexElement, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    flat = points.reshape((-1, points.shape[-1]))
    values, gradients = element.tabulate(flat)
    count = element.local_dof_count
    return (
        np.asarray(values, dtype=np.float64).reshape(points.shape[:-1] + (count,)),
        np.asarray(gradients, dtype=np.float64).reshape(
            points.shape[:-1] + (count, points.shape[-1])
        ),
    )


def _evaluate(
    element: reference_types_owner.FiniteElementSpec,
    nodes: np.ndarray,
    points: np.ndarray,
    /,
) -> np.ndarray:
    """Coordinate map of cells with nodes ``(C, m, D)`` at points ``(C, q, d)``."""

    values, _ = _tabulate(element, points)
    return np.asarray(contract("cqm,cmD->cqD", values, nodes))


class _CellIndex(NamedTuple):
    """Cells sorted by global ID with their block and block row."""

    ids: NDArray[np.int64]
    blocks: NDArray[np.int64]
    rows: NDArray[np.int64]


def _cell_index(blocks: tuple[SimplexGeometryBlock, ...], /) -> _CellIndex:
    ids = np.concatenate([block.cell_ids for block in blocks])
    owners = np.concatenate(
        [
            np.full(block.cell_ids.shape, index, dtype=np.int64)
            for index, block in enumerate(blocks)
        ]
    )
    rows = np.concatenate(
        [np.arange(block.cell_ids.size, dtype=np.int64) for block in blocks]
    )
    order = np.argsort(ids, kind="stable")
    return _CellIndex(ids[order], owners[order], rows[order])


def _locate(index: _CellIndex, identifiers: np.ndarray, name: str, /) -> np.ndarray:
    position = np.minimum(np.searchsorted(index.ids, identifiers), index.ids.size - 1)
    if np.any(index.ids[position] != identifiers):
        raise ValueError(
            f"geometry_transition.{name} must name present mesh cells; got absent cell ID"
        )
    return position


def _extent(coordinates: np.ndarray, /) -> float:
    span = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
    return max(float(np.max(np.abs(span))), float(np.max(np.abs(coordinates))), 1.0)


def _rule(block: SimplexGeometryBlock, extra: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Reference rule algebraic for ``extra`` plus the full-dimensional density degree."""

    points, weights = reference_rules_owner.host_degree_aware_reference_rule(
        block.element.cell_kind, extra + block.dimension * max(block.degree - 1, 0)
    )
    return np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _densities(
    block: SimplexGeometryBlock, coordinates: np.ndarray, points: np.ndarray, /
) -> np.ndarray:
    """Jacobian measure density ``sqrt(det(J^T J))`` of every cell at every point."""

    return _element_densities(
        block.element, block.routes, block.dimension, coordinates, points
    )


def _element_densities(
    element: _PolynomialSimplexElement,
    routes: NDArray[np.int64],
    dimension: int,
    coordinates: np.ndarray,
    points: np.ndarray,
    /,
) -> np.ndarray:
    _, gradients = _tabulate(element, points)
    jacobian = np.asarray(contract("cmD,qmd->cqDd", coordinates[routes], gradients))
    gram = np.swapaxes(jacobian, -1, -2) @ jacobian
    determinant = np.asarray(
        determinant_small_linear(SmallLinearSolvePlan(dimension), gram)
    )
    return np.sqrt(np.maximum(determinant, 0.0))


def _measure(
    blocks: tuple[SimplexGeometryBlock, ...], coordinates: np.ndarray, /
) -> tuple[float, bool]:
    """Mapped measure of every cell; algebraic for full-dimensional polynomial maps."""

    total = 0.0
    for block in blocks:
        points, weights = _rule(block, 0)
        total += float(np.sum(_densities(block, coordinates, points) @ weights))
    exact = all(coordinates.shape[1] == block.dimension for block in blocks)
    return total, exact


def _budget(count: int, policy: transition_types.CellGeometryTransitionPolicy, /) -> int:
    if count > policy.maximum_evaluations:
        raise transition_types.CellGeometryTransitionError(
            "resource_limit",
            "geometry_transition.evaluations must fit the evaluation budget; got excess basis evaluations",
            measured=count,
            limit=policy.maximum_evaluations,
        )
    return count


class PlacedGeometryNodes(NamedTuple):
    coordinates: NDArray[np.float64]
    owner_cells: NDArray[np.int64]
    owner_locals: NDArray[np.int32]
    continuity: float


def place(
    blocks: tuple[SimplexGeometryBlock, ...], values: list[np.ndarray], node_count: int, /
) -> PlacedGeometryNodes:
    """Scatter per-cell node values into the shared layout, first owner by cell ID.

    Every other evaluation of a shared node contributes its disagreement with the
    owner value to the continuity residual.
    """

    ambient = values[0].shape[-1]
    routes = np.concatenate([block.routes.reshape((-1,)) for block in blocks])
    cells = np.concatenate(
        [np.repeat(block.cell_ids, block.routes.shape[1]) for block in blocks]
    )
    slots = np.concatenate(
        [
            np.tile(np.arange(block.routes.shape[1]), block.cell_ids.size)
            for block in blocks
        ]
    )
    flat = np.concatenate([value.reshape((-1, ambient)) for value in values])
    order = np.lexsort((slots, cells))
    nodes, first = np.unique(routes[order], return_index=True)
    if not np.array_equal(nodes, np.arange(node_count)):
        raise ValueError(
            "geometry_transition.target_nodes must be referenced by target cells; got unreferenced coordinate node"
        )
    owner = order[first]
    coordinates = flat[owner]
    residual = float(np.max(np.abs(flat - coordinates[routes]), initial=0.0))
    return PlacedGeometryNodes(
        coordinates, cells[owner], slots[owner].astype(np.int32), residual
    )


@checked
def vertex_rows(
    mesh: CellMesh, blocks: tuple[SimplexGeometryBlock, ...], coordinates: np.ndarray, /
) -> np.ndarray:
    """Target mesh vertex rows at their coordinate vertex nodes."""

    vertices = np.full((mesh.coordinates.shape[0], coordinates.shape[1]), np.nan)
    for mesh_block, block in zip(mesh.blocks, blocks, strict=True):
        vertex_dofs = [entity[0] for entity in block.element.entity_dofs[0]]
        rows = np.asarray(mesh_block.vertices, dtype=np.int64)
        vertices[rows] = coordinates[block.routes[:, vertex_dofs]]
    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            "geometry_transition.target_vertices must be coordinate vertex nodes; got missing vertex node"
        )
    return vertices


def _witness(
    witnesses: transition_types.NestedReferenceWitnesses | None,
    dimension: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> transition_types.NestedReferenceWitnesses:
    if witnesses is None:
        empty = np.zeros((0,), dtype=np.int64)
        return transition_types.NestedReferenceWitnesses(
            empty, empty, np.zeros((0, dimension + 1, dimension), dtype=np.float64)
        )
    fine = np.asarray(witnesses.fine_cell_ids, dtype=np.int64)
    coarse = np.asarray(witnesses.coarse_cell_ids, dtype=np.int64)
    vertices = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    if (
        fine.ndim != 1
        or coarse.shape != fine.shape
        or vertices.shape != (fine.size, dimension + 1, dimension)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError(
            "geometry_transition.witnesses must be finite aligned cell and reference-corner arrays; got incompatible arrays"
        )
    bank = transition_types.require_reference_homogeneous(
        vertices,
        witnesses.fine_reference_homogeneous,
        "geometry_transition.witnesses",
        budget=budget,
    )
    return transition_types.NestedReferenceWitnesses(fine, coarse, vertices, bank)


def _admit_transition_banks(
    *banks: NDArray[np.float64] | None, budget: CoordinateEnclosureBudget
) -> None:
    """Charge `CellGeometryTransition`'s rational bank checks before construction."""
    for bank in banks:
        if bank is None:
            continue
        rows = int(np.count_nonzero(np.any(np.asarray(bank) != 0.0, axis=-1)))
        budget.reserve(
            *transition_types.reference_homogeneous_preparation_bounds(
                rows, int(np.asarray(bank).shape[-1])
            )
        )


def _excursion(vertices: np.ndarray, /) -> float:
    barycentric = _barycentric(vertices)
    return float(
        max(
            0.0,
            np.max(-barycentric, initial=0.0),
            np.max(barycentric - 1.0, initial=0.0),
        )
    )


def _witness_excursion(witnesses: transition_types.NestedReferenceWitnesses, /) -> float:
    """Barycentric excursion, exact zero for source-owned homogeneous corners.

    Nonnegative homogeneous weights lie in the closed reference simplex exactly;
    only their rounded execution corners may leave it.
    """
    vertices = witnesses.fine_reference_vertices
    if vertices.shape[0] == 0:
        return 0.0
    barycentric = _barycentric(vertices)
    defects = np.maximum(
        np.max(-barycentric, axis=-1, initial=0.0),
        np.max(barycentric - 1.0, axis=-1, initial=0.0),
    )
    bank = witnesses.fine_reference_homogeneous
    if bank is not None:
        defects = np.where(np.any(bank != 0.0, axis=-1), 0.0, defects)
    return float(np.max(defects, initial=0.0))


def _exact_witness_corners(
    witnesses: transition_types.NestedReferenceWitnesses, row: int, /
) -> tuple[tuple[Fraction, ...], ...]:
    bank = witnesses.fine_reference_homogeneous
    return transition_types.exact_reference_corners(
        witnesses.fine_reference_vertices[row], None if bank is None else bank[row]
    )


def _witness_affine_arguments(
    witnesses: transition_types.NestedReferenceWitnesses,
    row: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[imported_coordinate_enclosure.Polynomial, ...]:
    """Child-to-parent reference map of one certificate in rational arithmetic."""
    return _exact_corner_affine_arguments(
        _exact_witness_corners(witnesses, row), budget=budget
    )


def _exact_corner_affine_arguments(
    corners: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[imported_coordinate_enclosure.Polynomial, ...]:
    """Simplex reference map ``corner_0 + sum_j xi_j (corner_j - corner_0)``."""
    dimension = len(corners[0])
    origin = np.empty(dimension, dtype=object)
    matrix = np.empty((dimension, len(corners) - 1), dtype=object)
    for axis in range(dimension):
        origin[axis] = corners[0][axis]
        for column, corner in enumerate(corners[1:]):
            matrix[axis, column] = corner[axis] - corners[0][axis]
    return affine_arguments(origin, matrix, budget=budget)


def _empty_values(
    blocks: tuple[SimplexGeometryBlock, ...], ambient: int, /
) -> list[np.ndarray]:
    return [
        np.full(
            (block.cell_ids.size, block.routes.shape[1], ambient),
            np.nan,
            dtype=np.float64,
        )
        for block in blocks
    ]


class _Restricted(NamedTuple):
    parents: NDArray[np.int64]
    reference: NDArray[np.float64]
    evaluations: int
    homogeneous: NDArray[np.float64] | None


def _restrict(
    source_blocks: tuple[SimplexGeometryBlock, ...],
    source_coordinates: np.ndarray,
    target_blocks: tuple[SimplexGeometryBlock, ...],
    refinement: transition_types.NestedReferenceWitnesses,
    values: list[np.ndarray],
    policy: transition_types.CellGeometryTransitionPolicy,
    /,
) -> _Restricted:
    """Write the source maps composed with child reference maps at target nodes.

    ``parents``/``reference`` follow target cells in ascending global-ID order.
    """

    source_index = _cell_index(source_blocks)
    target_index = _cell_index(target_blocks)
    dimension = target_blocks[0].dimension
    parents = np.full((target_index.ids.size,), -1, dtype=np.int64)
    reference = np.zeros((target_index.ids.size, dimension + 1, dimension))
    homogeneous = (
        None
        if refinement.fine_reference_homogeneous is None
        else np.zeros((target_index.ids.size, dimension + 1, dimension + 1))
    )
    targets = _locate(target_index, refinement.fine_cell_ids, "Refinement")
    sources = _locate(source_index, refinement.coarse_cell_ids, "Refinement")
    parents[targets] = refinement.coarse_cell_ids
    reference[targets] = refinement.fine_reference_vertices
    if homogeneous is not None:
        homogeneous[targets] = refinement.fine_reference_homogeneous
    widest = max(block.element.local_dof_count for block in source_blocks)
    work = _budget(
        sum(
            int(np.sum(target_index.blocks[targets] == index))
            * block.element.local_dof_count
            * widest
            for index, block in enumerate(target_blocks)
        ),
        policy,
    )
    for target_block in np.unique(target_index.blocks[targets]):
        block = target_blocks[int(target_block)]
        nodes = _barycentric(np.asarray(block.element.reference_nodes, np.float64))
        for source_block in np.unique(source_index.blocks[sources]):
            group = np.flatnonzero(
                (target_index.blocks[targets] == target_block)
                & (source_index.blocks[sources] == source_block)
            )
            if group.size == 0:
                continue
            parent = source_blocks[int(source_block)]
            points = np.asarray(
                contract("nk,ckd->cnd", nodes, refinement.fine_reference_vertices[group])
            )
            parent_nodes = source_coordinates[
                parent.routes[source_index.rows[sources[group]]]
            ]
            values[int(target_block)][target_index.rows[targets[group]]] = _evaluate(
                parent.element, parent_nodes, points
            )
    return _Restricted(parents, reference, work, homogeneous)


def _fine_choices(
    coarse_block: SimplexGeometryBlock,
    corners: np.ndarray,
    coarse: np.ndarray,
    policy: transition_types.CellGeometryTransitionPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine reference record row, coarse node slot, and fine reference point of every node.

    Each coarse node is evaluated in the fine simplex containing it with the
    largest minimum barycentric coordinate; any containing fine cell gives the
    same value for a continuous source, so the choice is not a tolerance decision.
    """

    nodes = np.asarray(coarse_block.element.reference_nodes, dtype=np.float64)
    width, dimension = nodes.shape
    # Solve W lambda' = eta - w_0 on each fine reference frame W.
    frames = np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
    solved = solve_small_linear(
        SmallLinearSolvePlan(dimension),
        np.broadcast_to(frames[:, None], (corners.shape[0], width, dimension, dimension)),
        nodes[None] - corners[:, None, 0],
    )
    if not bool(np.all(np.asarray(solved.successful))):
        raise transition_types.CellGeometryTransitionError(
            "coverage",
            "geometry_transition.coarsening must have nonsingular fine reference simplices; got degenerate reference record",
            measured=0.0,
            limit=1.0,
        )
    local = np.asarray(solved.value, dtype=np.float64)
    score = np.min(_barycentric(local), axis=-1).reshape((-1,))
    groups = np.repeat(coarse, width)
    slots = np.tile(np.arange(width), corners.shape[0])
    order = np.lexsort((-score, slots, groups))
    _, first = np.unique(groups[order] * width + slots[order], return_index=True)
    chosen = order[first]
    if chosen.size != np.unique(coarse).size * width:
        raise ValueError(
            "geometry_transition.coarsening must cover every coarsened target cell; got missing fine-cell witnesses"
        )
    if np.min(score[chosen]) < -policy.continuity_tolerance:
        raise transition_types.CellGeometryTransitionError(
            "coverage",
            "geometry_transition.coarsening must cover every coarse reference node; got uncovered node",
            measured=-float(np.min(score[chosen])),
            limit=policy.continuity_tolerance,
        )
    fine, slot = np.divmod(chosen, width)
    return fine, slot, local[fine, slot]


class _Coarsened(NamedTuple):
    bound: float
    evaluations: int


def _coarsen(
    source_blocks: tuple[SimplexGeometryBlock, ...],
    source_coordinates: np.ndarray,
    target_blocks: tuple[SimplexGeometryBlock, ...],
    coarsening: transition_types.NestedReferenceWitnesses,
    values: list[np.ndarray],
    policy: transition_types.CellGeometryTransitionPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _Coarsened:
    """Interpolate fine maps at coarse nodes and bound the coarse-map deviation.

    The deviation on each fine cell is a polynomial of the fine degree whose
    Bernstein coefficients bound its sup norm (convex-hull property); the computed
    conversion is enlarged by its certified residual.
    """

    if coarsening.fine_cell_ids.size == 0:
        return _Coarsened(0.0, 0)
    source_index = _cell_index(source_blocks)
    target_index = _cell_index(target_blocks)
    coarse = _locate(target_index, coarsening.coarse_cell_ids, "Coarsening")
    fine = _locate(source_index, coarsening.fine_cell_ids, "Coarsening")
    dimension = target_blocks[0].dimension
    widest = max(block.element.local_dof_count for block in source_blocks)
    work = _budget(fine.size * widest * (widest + 3 * widest), policy)
    corners = coarsening.fine_reference_vertices
    bound = 0.0
    for coarse_block in np.unique(target_index.blocks[coarse]):
        block = target_blocks[int(coarse_block)]
        members = np.flatnonzero(target_index.blocks[coarse] == coarse_block)
        chosen, slots, points = _fine_choices(
            block, corners[members], coarse[members], policy
        )
        owners = fine[members[chosen]]
        for fine_block in np.unique(source_index.blocks[owners]):
            group = np.flatnonzero(source_index.blocks[owners] == fine_block)
            child = source_blocks[int(fine_block)]
            nodes = source_coordinates[child.routes[source_index.rows[owners[group]]]]
            values[int(coarse_block)][
                target_index.rows[coarse[members[chosen[group]]]], slots[group]
            ] = _evaluate(child.element, nodes, points[group][:, None, :])[:, 0]
        for fine_block in np.unique(source_index.blocks[fine[members]]):
            group = members[source_index.blocks[fine[members]] == fine_block]
            child = source_blocks[int(fine_block)]
            for member in group:
                coarse_nodes = values[int(coarse_block)][
                    target_index.rows[coarse[member]]
                ]
                fine_nodes = source_coordinates[
                    child.routes[source_index.rows[fine[member]]]
                ]
                parent_polynomials = coordinate_polynomials(
                    block.element, coarse_nodes, budget=budget
                )
                child_polynomials = coordinate_polynomials(
                    child.element, fine_nodes, budget=budget
                )
                if parent_polynomials is None or child_polynomials is None:
                    raise ValueError(
                        "geometry_transition.coarsening must provide source-polynomial bounds; got missing source basis"
                    )
                arguments = _witness_affine_arguments(
                    coarsening, int(member), budget=budget
                )
                for parent_map, child_map in zip(
                    parent_polynomials, child_polynomials, strict=True
                ):
                    difference = add(
                        compose(parent_map, arguments, budget=budget),
                        scale(child_map, -1, budget=budget),
                        budget=budget,
                    )
                    lower, upper = polynomial_bounds(
                        difference, "simplex", dimension, budget=budget
                    )
                    bound = max(bound, abs(lower), abs(upper))
    return _Coarsened(bound * (1.0 + 8.0 * _EPSILON), work)


def _transition(
    kind: transition_types.CellGeometryTransitionKind,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    geometry: CellGeometrySpec,
    placed: PlacedGeometryNodes,
    restricted: _Restricted,
    coarsening: transition_types.NestedReferenceWitnesses,
    evidence: transition_types.CellGeometryTransitionEvidence,
    policy: transition_types.CellGeometryTransitionPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> transition_types.CellGeometryTransition:
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    order = np.argsort(target_ids, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    parent_ids, reference = restricted.parents, restricted.reference
    parent_bank = None if restricted.homogeneous is None else restricted.homogeneous[rank]
    coarse_bank = coarsening.fine_reference_homogeneous
    vertex_coordinates = vertex_rows(
        target_mesh,
        simplex_blocks(target_mesh, geometry, "target", budget=budget),
        np.asarray(geometry.coordinates, dtype=np.float64),
    )
    source_id = cell_geometry_id(source_geometry)
    target_id = cell_geometry_id(geometry)
    _admit_transition_banks(parent_bank, coarse_bank, budget=budget)
    stored_targets = jnp.asarray(target_ids)
    stored_parents = jnp.asarray(parent_ids[rank])
    stored_reference = jnp.asarray(reference[rank])
    stored_coarsened = jnp.asarray(coarsening.fine_cell_ids)
    stored_into = jnp.asarray(coarsening.coarse_cell_ids)
    stored_coarse_reference = jnp.asarray(coarsening.fine_reference_vertices)
    parent_homogeneous = None if parent_bank is None else jnp.asarray(parent_bank)
    coarsened_homogeneous = None if coarse_bank is None else jnp.asarray(coarse_bank)
    transition_id = (
        canonical_fingerprint(
            {
                "kind": "cell-geometry-transition",
                "source_topology": source_mesh.topology_id,
                "target_topology": target_mesh.topology_id,
                "source_geometry": source_id,
                "target_geometry": target_id,
                "parents": array_tree_fingerprint(parent_ids[rank]),
                "reference": array_tree_fingerprint(reference[rank]),
                "coarsened": array_tree_fingerprint(
                    np.concatenate(
                        (
                            coarsening.fine_cell_ids[:, None],
                            coarsening.coarse_cell_ids[:, None],
                        ),
                        axis=1,
                    )
                ),
                "evidence": evidence.evidence_id,
                "policy": policy.policy_id,
            }
        )
        if parent_homogeneous is None and coarsened_homogeneous is None
        # Unrounded source-owned corners use the deterministic replay-checked
        # identity over the stored arrays themselves.
        else transition_types.exact_root_transition_id(
            source_topology_id=source_mesh.topology_id,
            target_topology_id=target_mesh.topology_id,
            source_geometry_id=source_id,
            target_geometry_id=target_id,
            target_cell_ids=stored_targets,
            parent_cell_ids=stored_parents,
            parent_reference_vertices=stored_reference,
            parent_reference_homogeneous=parent_homogeneous,
            coarsened_cell_ids=stored_coarsened,
            coarsened_into_ids=stored_into,
            coarsened_reference_vertices=stored_coarse_reference,
            coarsened_reference_homogeneous=coarsened_homogeneous,
            evidence_id=evidence.evidence_id,
            policy_id=policy.policy_id,
        )
    )
    return transition_types.CellGeometryTransition(
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        source_geometry_id=source_id,
        target_geometry_id=target_id,
        geometry=geometry,
        vertex_coordinates=jnp.asarray(vertex_coordinates),
        target_cell_ids=stored_targets,
        parent_cell_ids=stored_parents,
        parent_reference_vertices=stored_reference,
        coarsened_cell_ids=stored_coarsened,
        coarsened_into_ids=stored_into,
        coarsened_reference_vertices=stored_coarse_reference,
        node_owner_cells=jnp.asarray(placed.owner_cells),
        node_owner_locals=jnp.asarray(placed.owner_locals),
        evidence=evidence,
        policy_id=policy.policy_id,
        parent_reference_homogeneous=parent_homogeneous,
        coarsened_reference_homogeneous=coarsened_homogeneous,
        transition_id=transition_id,
    )


def _plc_parent_polynomials(
    parent: int,
    element: CellGeometryElement,
    bank: imported_coordinate_enclosure.CoordinateSourceBank,
    route: np.ndarray,
    memo: dict[int, tuple[imported_coordinate_enclosure.Polynomial, ...]],
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[imported_coordinate_enclosure.Polynomial, ...]:
    """Reuse the source-cell polynomial map for every child of one parent."""
    cached = memo.get(parent)
    if cached is not None:
        return cached
    prepared = coordinate_polynomials(
        element, tuple(bank[int(index)] for index in route), budget=budget
    )
    if prepared is None:
        raise ValueError(
            "plc_transition.source must provide an affine coordinate source; got absent affine source"
        )
    budget.reserve(0, 96)
    memo[parent] = prepared
    return prepared


def _rational_plc_point(
    point: tuple[imported_coordinate_enclosure.ExactScalar, ...],
    /,
) -> tuple[Fraction, ...]:
    """Validate the rational coefficient capability of a PLC coordinate map."""
    return tuple(
        rational_coefficient(value, role="plc_transition.source_coordinates")
        for value in point
    )


def _plc_reference_node_weights(
    element: reference_types_owner.FiniteElementSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[int, Fraction], ...], ...]:
    """Compile the existing binary64 barycentric node rule once per element."""
    dimension = _SIMPLEX_DIMENSIONS[element.cell_kind]
    budget.reserve(0, 128 + element.local_dof_count * (64 + (dimension + 1) * 448))
    return tuple(
        tuple(
            (corner, Fraction(float(weight)))
            for corner, weight in enumerate(weights)
            if weight
        )
        for weights in _barycentric(np.asarray(element.reference_nodes, dtype=np.float64))
    )


def _plc_node_reference_point(
    weights: tuple[tuple[int, Fraction], ...],
    corners: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[Fraction, ...]:
    """Select unit rows exactly; retain rational arithmetic for all other rows."""
    if len(weights) == 1 and weights[0][1] == 1:
        return corners[weights[0][0]]
    return tuple(
        sum(
            (weight * corners[corner][axis] for corner, weight in weights),
            Fraction(0),
        )
        for axis in range(len(corners[0]))
    )


def _plc_affine_point(
    corners: tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...],
    tail: tuple[Fraction, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[imported_coordinate_enclosure.ExactScalar, ...]:
    """Evaluate the exact affine corner law used by PLC correspondence proofs."""
    with budget.temporary_scope():
        # A common denominator divides the product of the input denominators.
        # Include construction of 1 - sum(tail) and all weighted partial sums
        # before allocating any new rational weights or coordinate values.
        bits = (
            sum(coefficient_bits(value) for point in corners for value in point)
            + 2 * sum(coefficient_bits(value) for value in tail)
            + 16
        )
        reserve_polynomial(0, 12, 0, bits, budget=budget)
        weights = (Fraction(1) - sum(tail, Fraction(0)), *tail)
        budget.reserve(6 + 3 * sum(bool(weight) for weight in weights))
        # Cardinal reference points select an existing source authority. Keep
        # the same admission for the inspected affine row, but do not rebuild
        # its rational coordinates by multiplying by one and adding zeros.
        for index, weight in enumerate(weights):
            if weight == 1 and all(
                value == 0 for other, value in enumerate(weights) if other != index
            ):
                return corners[index]
        if any(
            isinstance(value, RadicalElement)
            for point in corners
            for value in point
        ):
            reserve_algebraic_work(corners, 12, 12, 1, budget=budget)
        return tuple(
            sum(
                (
                    weight * point[axis]
                    for weight, point in zip(weights, corners, strict=True)
                    if weight
                ),
                Fraction(0),
            )
            for axis in range(3)
        )


def _plc_nested_coordinates(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    source_blocks: tuple[SimplexGeometryBlock, ...],
    target_blocks: tuple[SimplexGeometryBlock, ...],
    refined: transition_types.NestedReferenceWitnesses,
    source: ExactPlcCellGeometrySource,
    policy: transition_types.CellGeometryTransitionPolicy,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[PlacedGeometryNodes, _Restricted, ExactPlcCellGeometrySource]:

    bank = source.prepare(source_geometry.coordinates, budget=budget).vertices
    count = target_mesh.coordinates.shape[0]
    budget.reserve(0, count * (64 + 3 * (128 + 2 * ((source.maximum_bits + 7) // 8))))
    authorities: list[tuple[Fraction, ...] | None] = [None] * count
    coordinates = np.full((count, 3), np.nan, dtype=np.float64)
    strata = np.zeros(count, dtype=np.int8)
    rows = np.full(count, -1, dtype=np.int64)
    parameters = np.zeros((count, 2), dtype=np.float64)
    exact_ratios: list[tuple[tuple[int, int], tuple[int, int]]] | None = (
        [((0, 1), (0, 1))] * count if source.vertex_parameter_ratios is not None else None
    )
    original = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(source_mesh.vertex_global_ids))
    }
    carrier = np.asarray(source_geometry.coordinates)
    original_strata = np.asarray(source.vertex_strata)
    original_rows = np.asarray(source.vertex_rows)
    original_parameters = np.asarray(source.vertex_parameters)
    for vertex, identifier in enumerate(np.asarray(target_mesh.vertex_global_ids)):
        slot = original.get(int(identifier))
        if slot is not None:
            coordinates[vertex] = carrier[slot]
            strata[vertex] = original_strata[slot]
            rows[vertex] = original_rows[slot]
            parameters[vertex] = original_parameters[slot]
            if exact_ratios is not None and source.vertex_parameter_ratios is not None:
                exact_ratios[vertex] = source.vertex_parameter_ratios[slot]
            authorities[vertex] = tuple(bank[slot])
    source_index, target_index = _cell_index(source_blocks), _cell_index(target_blocks)
    targets = _locate(target_index, refined.fine_cell_ids, "Refinement")
    sources = _locate(source_index, refined.coarse_cell_ids, "Refinement")
    parents = np.full(target_index.ids.size, -1, dtype=np.int64)
    reference = np.zeros((target_index.ids.size, 4, 3), dtype=np.float64)
    parents[targets], reference[targets] = (
        refined.coarse_cell_ids,
        refined.fine_reference_vertices,
    )
    homogeneous = None
    if refined.fine_reference_homogeneous is not None:
        homogeneous = np.zeros((target_index.ids.size, 4, 4), dtype=np.float64)
        homogeneous[targets] = refined.fine_reference_homogeneous
    exact_work = ExactPlcBudget(
        min(source.maximum_work, policy.maximum_evaluations),
        source.maximum_bits,
        budget=budget,
    )
    locator = None
    parent_polynomials: dict[
        int, tuple[imported_coordinate_enclosure.Polynomial, ...]
    ] = {}
    budget.reserve(0, 64)
    point_witnesses: dict[
        tuple[Fraction, ...],
        tuple[
            int,
            int,
            tuple[float, ...],
            tuple[Fraction, ...],
            tuple[tuple[int, int], tuple[int, int]] | None,
        ],
    ] = {}
    # Children reuse their parent's exact noncardinal point evaluations.
    # Cardinal roots already share the source-owned bank and need no extra memo.
    # Only completed evaluations are charged; each retained memo row is admitted.
    parent_points: dict[tuple[int, tuple[Fraction, ...]], tuple[Fraction, ...]] = {}
    memo_storage = 128 + 6 * (128 + 2 * ((source.maximum_bits + 7) // 8))
    node_weights: dict[str, tuple[tuple[tuple[int, Fraction], ...], ...]] = {}
    for block in target_blocks:
        key = block.element.element_id
        if key not in node_weights:
            node_weights[key] = _plc_reference_node_weights(
                block.element, budget=budget
            )
    source_corner_nodes: dict[str, dict[tuple[Fraction, ...], int]] = {}
    affine_corner_nodes: dict[str, tuple[int, ...] | None] = {}
    for block in source_blocks:
        key = block.element.element_id
        if key in source_corner_nodes:
            continue
        nodes = imported_cell_geometry._cardinal_corner_nodes(
            block.element, budget=budget
        )
        budget.reserve(0, 256 + (0 if nodes is None else len(nodes) * 1024))
        source_corner_nodes[key] = (
            {}
            if nodes is None
            else {
                tuple(Fraction(float(value)) for value in corner): node
                for corner, node in zip(
                    reference_cell_topology(block.element.cell_kind).vertices,
                    nodes,
                    strict=True,
                )
            }
        )
        affine_corner_nodes[key] = (
            nodes
            if nodes is not None
            and imported_coordinate_enclosure.affine_simplex_source(
                block.element, budget=budget
            )
            else None
        )
    order = np.argsort(refined.fine_cell_ids, kind="stable")
    for row, child, parent in zip(
        order.tolist(),
        targets[order],
        sources[order],
        strict=True,
    ):
        target = target_blocks[int(target_index.blocks[child])]
        coarse = source_blocks[int(source_index.blocks[parent])]
        route = coarse.routes[int(source_index.rows[parent])]
        corner_nodes = source_corner_nodes[coarse.element.element_id]
        affine_nodes = affine_corner_nodes[coarse.element.element_id]
        # Exact source-owned child corners, never their rounded execution view.
        corners = _exact_witness_corners(refined, row)
        nodes = node_weights[target.element.element_id]
        target_route = target.routes[int(target_index.rows[child])]
        for vertex, weights in zip(target_route, nodes, strict=True):
            reference_point = _plc_node_reference_point(weights, corners)
            source_node = corner_nodes.get(reference_point)
            if source_node is not None:
                # Preparation already authenticated these rational root rows.
                # Reuse that authority, not a per-parent copy/memo of it.
                point = bank[int(route[source_node])]
            else:
                point_key = (int(parent), reference_point)
                point = parent_points.get(point_key)
                if point is None:
                    budget.reserve(0, memo_storage)
                    if affine_nodes is not None:
                        point = _rational_plc_point(
                            _plc_affine_point(
                                tuple(bank[int(route[node])] for node in affine_nodes),
                                reference_point,
                                budget=budget,
                            )
                        )
                    else:
                        polynomials = _plc_parent_polynomials(
                            int(parent),
                            coarse.element,
                            bank,
                            route,
                            parent_polynomials,
                            budget=budget,
                        )
                        point = _rational_plc_point(
                            tuple(
                                imported_coordinate_enclosure.evaluate(
                                    polynomial,
                                    reference_point,
                                    budget=budget,
                                )
                                for polynomial in polynomials
                            )
                        )
                    parent_points[point_key] = point
            if len(point) != 3:
                raise ValueError(
                    "plc_transition.source must have three coordinate dimensions; got incompatible source dimension"
                )
            previous = authorities[vertex]
            # Only algebraic source authority equality can bypass reference record location;
            # equal rounded carrier coordinates do not establish a shared seam.
            if previous is not None and point == previous:
                continue
            located = point_witnesses.get(point)
            if located is None:
                if locator is None:
                    locator = source.prepare_witness_locator(exact_work, budget=budget)
                stratum, row, witness, authority, ratios = source.locate_witness(
                    (point[0], point[1], point[2]),
                    exact_work,
                    locator=locator,
                    budget=budget,
                )
                budget.reserve(
                    0,
                    128 + 6 * (128 + 2 * ((source.maximum_bits + 7) // 8)),
                )
                located = (stratum, row, tuple(witness), tuple(authority), ratios)
                point_witnesses[point] = located
            stratum, row, witness, authority, ratios = located
            if previous is not None:
                if authority != previous:
                    raise ValueError(
                        "plc_transition.continuity must agree at shared target vertices; got unequal rational coordinates"
                    )
                continue
            authorities[vertex] = tuple(authority)
            strata[vertex], rows[vertex], parameters[vertex] = stratum, row, witness
            if exact_ratios is not None:
                if ratios is None:
                    raise RuntimeError(
                        "plc_transition.source must retain rational source coordinates; got absent rational source"
                    )
                exact_ratios[vertex] = ratios
            coordinates[vertex] = tuple(float(value) for value in authority)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "plc_transition.target_vertices must bind every target vertex; got unbound vertex"
        )
    successor = source.with_witnesses(
        strata,
        rows,
        parameters,
        vertex_parameter_ratios=None if exact_ratios is None else tuple(exact_ratios),
        budget=budget,
    )
    successor.prepare(coordinates, budget=budget)
    placed = place(
        target_blocks, [coordinates[block.routes] for block in target_blocks], count
    )
    return (
        placed,
        _Restricted(parents, reference, exact_work.work, homogeneous),
        successor,
    )


@checked
def plc_nested_approximation(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    refined: transition_types.NestedReferenceWitnesses,
    coarsened: transition_types.NestedReferenceWitnesses,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:

    ledger = budget
    # Unrounded source-owned corners are authenticated before any algebraic bound.
    refined = _witness(refined, 3, budget=budget)
    coarsened = _witness(coarsened, 3, budget=budget)

    cells = []
    for mesh, geometry in (
        (source_mesh, source_geometry),
        (target_mesh, target_geometry),
    ):
        bank = imported_coordinate_enclosure.prepared_coordinate_source_bank(
            geometry, budget=budget
        )
        blocks = simplex_blocks(mesh, geometry, "PLC correspondence", budget=budget)
        if any(block.dimension != 3 or block.degree != 1 for block in blocks):
            raise ValueError(
                "plc_transition.correspondence must use affine tetrahedral maps; got unsupported coordinate element"
            )
        cells.append(
            {
                int(identifier): tuple(
                    bank[int(np.asarray(route)[entity[0]])]
                    for entity in block.element.entity_dofs[0]
                )
                for block in blocks
                for identifier, route in zip(block.cell_ids, block.routes, strict=True)
            }
        )
    source_cells, target_cells = cells
    bounds = []
    for witnesses, fine_cells, coarse_cells in (
        (refined, target_cells, source_cells),
        (coarsened, source_cells, target_cells),
    ):
        bound = _plc_nested_bound(witnesses, fine_cells, coarse_cells, budget=ledger)
        bounds.append(coefficient_endpoint(bound, upper=True, saturate=True))
    return bounds[0], bounds[1]


def _plc_nested_bound(
    witnesses: transition_types.NestedReferenceWitnesses,
    fine_cells: dict[
        int, tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...]
    ],
    coarse_cells: dict[
        int, tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...]
    ],
    *,
    budget: CoordinateEnclosureBudget,
) -> imported_coordinate_enclosure.ExactScalar:
    """Reuse identical source-corner evaluations while checking every child corner."""
    bound: imported_coordinate_enclosure.ExactScalar = Fraction(0)
    with budget.temporary_scope():
        expected_points: dict[
            tuple[int, tuple[Fraction, ...]],
            tuple[imported_coordinate_enclosure.ExactScalar, ...],
        ] = {}
        budget.reserve(0, 64)
        for row, (fine_id, coarse_id) in enumerate(
            zip(witnesses.fine_cell_ids, witnesses.coarse_cell_ids, strict=True)
        ):
            fine, coarse = fine_cells[int(fine_id)], coarse_cells[int(coarse_id)]
            for actual, tail in zip(
                fine, _exact_witness_corners(witnesses, row), strict=True
            ):
                key = (int(coarse_id), tail)
                expected = expected_points.get(key)
                if expected is None:
                    expected = _plc_affine_point(coarse, tail, budget=budget)
                    budget.reserve(
                        0,
                        256
                        + sum(
                            128
                            + 4
                            * ((coefficient_bits(value) + 7) // 8)
                            * (
                                len(value.terms) + 1
                                if isinstance(value, RadicalElement)
                                else 1
                            )
                            for value in (*tail, *expected)
                        ),
                    )
                    expected_points[key] = expected
                # Every seam comparison remains, including shared reference
                # corners. Equal exact authorities have zero defect without
                # allocating three rational differences for every incidence.
                if actual != expected:
                    bound = max(
                        bound,
                        *(
                            abs(first - second)
                            for first, second in zip(actual, expected, strict=True)
                        ),
                    )
    return bound


def _exact_tetra_reference_weight(
    points: tuple[tuple[Fraction, ...], ...], /, *, budget: CoordinateEnclosureBudget
) -> Fraction:
    """Rational determinant of one unrounded child-to-parent tetrahedron map."""

    budget.reserve(3 * len(points))
    bits = max(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        for point in points
        for value in point
    )
    reserve_polynomial(26, 15, 0, 3 * (2 * bits + 1) + 3, budget=budget)
    columns = tuple(
        tuple(points[column + 1][axis] - points[0][axis] for column in range(3))
        for axis in range(3)
    )
    matrix = np.asarray(columns, dtype=object)
    determinant = np.asarray(
        cofactor_determinant_3(matrix[0], matrix[1], matrix[2]), dtype=object
    ).item()
    if not isinstance(determinant, Fraction):
        raise TypeError(
            "plc_transition.reference_determinant must be a Fraction; got nonrational determinant"
        )
    if determinant <= 0:
        raise ValueError(
            "plc_transition.reference_measure must be positive; got nonpositive rational measure"
        )
    return determinant


def _require_exact_refinement_partition(
    refined: transition_types.NestedReferenceWitnesses,
    source_cell_ids: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove repeated child patterns partition each source cell."""
    rows = refined.fine_reference_vertices
    bank = refined.fine_reference_homogeneous
    with budget.temporary_scope():
        budget.reserve(
            2 * rows.shape[0],
            512 + rows.shape[0] * (256 + rows.shape[1] * rows.shape[2] * 8),
        )
        weights: dict[tuple[tuple[Fraction, ...], ...], Fraction] = {}
        counts: dict[int, dict[tuple[tuple[Fraction, ...], ...], int]] = {}
        for row, parent in enumerate(refined.coarse_cell_ids):
            key = _exact_witness_corners(refined, row)
            if bank is not None and np.any(bank[row] != 0.0):
                # Exact source-owned keys exceed one binary64 word per coordinate.
                budget.reserve(
                    0,
                    sum(
                        128
                        + (abs(value.numerator).bit_length() + 7) // 8
                        + (value.denominator.bit_length() + 7) // 8
                        for corner in key
                        for value in corner
                    ),
                )
            if key not in weights:
                weights[key] = _exact_tetra_reference_weight(key, budget=budget)
            parent_counts = counts.setdefault(int(parent), {})
            parent_counts[key] = parent_counts.get(key, 0) + 1
        expected = {int(identifier) for identifier in source_cell_ids.tolist()}
        if set(counts) != expected:
            raise ValueError(
                "plc_transition.partition must cover every source cell; got uncovered source cell"
            )
        for parent_counts in counts.values():
            budget.reserve(2 * len(parent_counts))
            total = sum(
                (count * weights[key] for key, count in parent_counts.items()),
                Fraction(0),
            )
            if total != 1:
                raise ValueError(
                    "plc_transition.partition must tile each source reference cell; got unequal rational measure"
                )


def _transition_exact_plc_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    refinement: transition_types.NestedReferenceWitnesses | None,
    coarsening: transition_types.NestedReferenceWitnesses | None,
    policy: transition_types.CellGeometryTransitionPolicy,
    *,
    budget: CoordinateEnclosureBudget,
) -> transition_types.CellGeometryTransition:

    ledger = budget

    with ledger.bound_stage(policy.maximum_evaluations, ledger.maximum_memory_bytes):
        return _execute_exact_plc_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            target_layout,
            refinement,
            coarsening,
            policy,
            budget=ledger,
        )


def _require_plc_successor_corners(
    blocks: tuple[SimplexGeometryBlock, ...],
    bank: imported_coordinate_enclosure.CoordinateSourceBank,
    coordinates: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Check every exact incidence, rounding each distinct authority only once."""
    with budget.temporary_scope():
        budget.reserve(0, 1024 + coordinates.shape[0] * 33)
        carrier_bits = np.ascontiguousarray(coordinates, dtype=np.float64).view(
            np.uint64
        )
        used = np.zeros(coordinates.shape[0], dtype=np.bool_)
        for block in blocks:
            nodes = imported_cell_geometry._cardinal_corner_nodes(
                block.element, budget=budget
            )
            budget.reserve(block.routes.size)
            if nodes == tuple(range(block.element.local_dof_count)):
                # The shared cardinal proof makes each image the very same
                # bank[vertex] authority, not merely an equal rounded carrier.
                used[block.routes] = True
                continue
            for route in block.routes:
                images = (
                    tuple(bank[int(route[node])] for node in nodes)
                    if nodes is not None
                    else imported_coordinate_enclosure.coordinate_corner_images(
                        block.element,
                        tuple(bank[int(vertex)] for vertex in route),
                        budget=budget,
                    )
                )
                if images is None:
                    raise ValueError(
                        "plc_transition.successor must retain the affine corner map; got absent corner images"
                    )
                for vertex, image in zip(route, images, strict=True):
                    if image != bank[int(vertex)]:
                        raise ValueError(
                            "plc_transition.continuity must agree at shared successor corners; got unequal rational coordinates"
                        )
                    used[vertex] = True
        for vertex in np.flatnonzero(used):
            if not np.array_equal(
                imported_coordinate_enclosure.rounded_point(bank[int(vertex)]).view(
                    np.uint64
                ),
                carrier_bits[vertex],
            ):
                raise ValueError(
                    "plc_transition.carrier_corners must be correctly rounded map images; got different binary64 corner coordinates"
                )


def _execute_exact_plc_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    refinement: transition_types.NestedReferenceWitnesses | None,
    coarsening: transition_types.NestedReferenceWitnesses | None,
    policy: transition_types.CellGeometryTransitionPolicy,
    *,
    budget: CoordinateEnclosureBudget,
) -> transition_types.CellGeometryTransition:
    starting_work = budget.work_units
    source = source_geometry.exact_source
    match source:
        case ExactPlcCellGeometrySource():
            pass
        case _:
            raise TypeError(
                "plc_transition.source must be a PLC cell geometry source; got incompatible source owner"
            )
    source_blocks = simplex_blocks(source_mesh, source_geometry, "source", budget=budget)
    target_blocks = simplex_blocks(target_mesh, target_layout, "target", budget=budget)
    if any(
        block.dimension != 3 or block.degree != 1
        for block in (*source_blocks, *target_blocks)
    ):
        raise ValueError(
            "plc_transition.layout must preserve affine tetrahedral layouts; got incompatible coordinate layout"
        )
    refined = _witness(refinement, 3, budget=budget)
    coarsened = _witness(coarsening, 3, budget=budget)
    target_ids = np.concatenate([block.cell_ids for block in target_blocks])
    covered = np.concatenate(
        (refined.fine_cell_ids, np.unique(coarsened.coarse_cell_ids))
    )
    if covered.size != target_ids.size or not np.array_equal(
        np.sort(covered), np.sort(target_ids)
    ):
        raise ValueError(
            "plc_transition.witnesses must give each target cell one nested disposition; got missing or repeated reference record"
        )
    with budget.temporary_scope():
        placed, restricted, successor = _plc_nested_coordinates(
            source_mesh,
            source_geometry,
            target_mesh,
            source_blocks,
            target_blocks,
            refined,
            source,
            policy,
            budget=budget,
        )
    candidate = target_mesh.with_coordinates(
        placed.coordinates, numeric_version=target_mesh.numeric_version
    )
    geometry = CellGeometrySpec.plc(candidate, successor)
    # The successor source, not a rounded carrier-only surrogate, owns all
    # incident affine maps. Validate its algebraic corner images before any
    # association transfer or candidate publication can consume the seam.

    successor_bank = geometry.source_coordinates(budget=budget)
    _require_plc_successor_corners(
        target_blocks, successor_bank, np.asarray(candidate.coordinates), budget=budget
    )
    refinement_bound, coarsening_bound = plc_nested_approximation(
        source_mesh,
        source_geometry,
        candidate,
        geometry,
        refined,
        coarsened,
        budget=budget,
    )
    slack = policy.continuity_tolerance * _extent(np.asarray(source_geometry.coordinates))
    match policy.coarsening:
        case "exact_only":
            tolerance = slack
        case "bounded_interpolation":
            tolerance = policy.coarsening_tolerance + slack
        case invalid:
            assert_never(invalid)
    if refinement_bound > slack or coarsening_bound > tolerance:
        raise transition_types.CellGeometryTransitionError(
            "approximation_bound",
            "plc_transition.approximation_bound must fit the declared correspondence bound; got excess source deviation",
            measured=max(refinement_bound, coarsening_bound),
            limit=max(slack, tolerance),
        )
    source_values, source_errors, source_exact = (
        mapped_measure_owner.certified_cell_measures(
            source_mesh,
            source_geometry,
            maximum_work=policy.maximum_evaluations,
            budget=budget,
        )
    )
    source_measure = math.fsum(source_values)
    source_error = math.fsum(source_errors) + abs(float(np.spacing(source_measure)))
    if not coarsened.fine_cell_ids.size:
        source_ids = np.concatenate([block.cell_ids for block in source_blocks])
        _require_exact_refinement_partition(refined, source_ids, budget=budget)
        target_measure = source_measure
        target_error = source_error
        target_exact = source_exact
    else:
        target_values, target_errors, target_exact = (
            mapped_measure_owner.certified_cell_measures(
                candidate,
                geometry,
                maximum_work=policy.maximum_evaluations,
                budget=budget,
            )
        )
        target_measure = math.fsum(target_values)
        target_error = math.fsum(target_errors) + abs(float(np.spacing(target_measure)))
    containment = max(
        _witness_excursion(refined),
        _witness_excursion(coarsened),
    )
    evidence = transition_types.CellGeometryTransitionEvidence(
        "coarsening_interpolation"
        if coarsened.fine_cell_ids.size
        else "nested_restriction",
        exact=containment == 0.0 and max(refinement_bound, coarsening_bound) <= slack,
        node_count=placed.coordinates.shape[0],
        evaluation_count=budget.work_units - starting_work,
        containment_defect=containment,
        continuity_residual=placed.continuity,
        approximation_bound=max(refinement_bound, coarsening_bound),
        approximation_tolerance=tolerance,
        rounding_slack=slack,
        source_measure=source_measure,
        target_measure=target_measure,
        measure_exact=source_exact and target_exact,
        source_measure_error_bound=source_error,
        target_measure_error_bound=target_error,
        coverage_defect=max(
            abs(target_measure - source_measure) - source_error - target_error, 0.0
        )
        / max(abs(source_measure), np.finfo(np.float64).tiny),
    )
    return _transition(
        evidence.kind,
        source_mesh,
        source_geometry,
        candidate,
        geometry,
        placed,
        restricted,
        coarsened,
        evidence,
        policy,
        budget=budget,
    )


def _nested_producer(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
) -> Literal["exact_plc", "restricted", "interpolated"]:
    """Select the nested coordinate producer for one source and target layout."""
    simplex_layout = (
        all(
            block.cell_kind in _SIMPLEX_DIMENSIONS
            for block in (*source_mesh.blocks, *target_mesh.blocks)
        )
        and not any(
            isinstance(
                element,
                (BarycentricCellGeometryElement, RestrictedCellGeometryElement),
            )
            for element in source_geometry.elements
        )
        and {element.element_id for element in source_geometry.elements}
        == {element.element_id for element in target_layout.elements}
    )
    if (
        isinstance(source_geometry.exact_source, ExactPlcCellGeometrySource)
        and simplex_layout
        and source_geometry.restriction_source is None
        and all(block.cell_kind == "tetrahedron" for block in source_mesh.blocks)
    ):
        return "exact_plc"
    if not simplex_layout or source_geometry.exact_source is not None:
        return "restricted"
    return "interpolated"


@checked
def nested_edge_midpoints_admitted(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    edges: NDArray[np.int64],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> NDArray[np.bool_]:
    """Whether nested splits publish each edge's exact chart midpoint as authority.

    ``edges`` are ``(E, 2)`` source vertex rows split by a kind-preserving
    nested refinement into the affine target layout. Restricted producers keep
    the exact source map, whose split node is the chart image itself.
    The exact PLC simplex producer rebinds each split node to a source row and
    admits only carriers inside that row's original bound; the same location,
    rounding and bound decide each exact midpoint here. Interpolating simplex
    producers publish binary64 nodes and admit none.
    """
    rows = np.asarray(edges, dtype=np.int64).reshape((-1, 2))
    layout = CellGeometrySpec.affine(source_mesh)
    match _nested_producer(source_mesh, source_geometry, source_mesh, layout):
        case "restricted":
            return np.ones(rows.shape[0], dtype=np.bool_)
        case "interpolated":
            return np.zeros(rows.shape[0], dtype=np.bool_)
        case "exact_plc":
            pass
        case producer:
            assert_never(producer)
    source = source_geometry.exact_source
    if (
        not isinstance(source, ExactPlcCellGeometrySource)
        or source_geometry.geometry_layout_id != layout.geometry_layout_id
    ):
        # Degree-one tetrahedral PLC banks are vertex rows only in this layout.
        return np.zeros(rows.shape[0], dtype=np.bool_)
    bank = source.prepare(source_geometry.coordinates, budget=budget).vertices
    exact_work = ExactPlcBudget(source.maximum_work, source.maximum_bits, budget=budget)
    locator = source.prepare_witness_locator(exact_work, budget=budget)
    admitted = np.zeros(rows.shape[0], dtype=np.bool_)
    for index, (first, second) in enumerate(rows.tolist()):
        # The affine corner law at the reference edge midpoint.
        point = tuple(
            (left + right) / 2
            for left, right in zip(
                _rational_plc_point(bank[first]),
                _rational_plc_point(bank[second]),
                strict=True,
            )
        )
        admitted[index] = source.admits_located_witness(
            point, exact_work, locator=locator, budget=budget
        )
    return admitted


@checked
def transition_nested_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    refinement: transition_types.NestedReferenceWitnesses | None,
    coarsening: transition_types.NestedReferenceWitnesses | None = None,
    policy: transition_types.CellGeometryTransitionPolicy | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> transition_types.CellGeometryTransition:
    """Carry a coordinate map through one nested topology edit.

    ``target_layout`` fixes the target coordinate elements and node routes (its
    coordinate values are ignored). Simplex nodal maps retain their coordinate
    element; mixed maps retain their source coefficients and reference actions.
    Every target cell is either a refinement fine cell (preserved cells
    are their own parents with the identity reference record) or the coarse cell of
    full coarsening witnesses. Refinement nodes are the source map composed
    with the child reference map; coarsened cells follow ``policy.coarsening``.
    Refusals raise `CellGeometryTransitionError` and publish nothing.
    """

    policy_ = (
        transition_types.CellGeometryTransitionPolicy() if policy is None else policy
    )
    if isinstance(source_geometry.exact_source, ExactPlcCellGeometryConvexSource):
        raise ValueError(
            "geometry_transition.source must provide a simplex source bank; got PLC convex target bank"
        )
    with (
        (
            nullcontext(budget)
            if budget is not None
            else owned_coordinate_preparation(
                policy_.maximum_evaluations,
                sys.maxsize,
                maximum_geometry_queries=policy_.maximum_evaluations,
            )
        ) as budget,
        transition_budget_owner.charged_geometry_transition(policy_, budget=budget),
    ):
        producer = _nested_producer(
            source_mesh, source_geometry, target_mesh, target_layout
        )
        if producer == "exact_plc":
            return _transition_exact_plc_geometry(
                source_mesh,
                source_geometry,
                target_mesh,
                target_layout,
                refinement,
                coarsening,
                policy_,
                budget=budget,
            )
        if producer == "restricted":
            return _transition_mixed_nested_geometry(
                source_mesh,
                source_geometry,
                target_mesh,
                refinement=refinement,
                coarsening=coarsening,
                policy=policy_,
                budget=budget,
            )
        source_blocks = simplex_blocks(
            source_mesh, source_geometry, "source", budget=budget
        )
        target_blocks = simplex_blocks(
            target_mesh, target_layout, "target", budget=budget
        )
        dimension = source_mesh.topological_dimension
        source_elements = {block.element.element_id for block in source_blocks}
        target_elements = {block.element.element_id for block in target_blocks}
        if target_mesh.topological_dimension != dimension or target_elements != (
            source_elements
        ):
            raise ValueError(
                "geometry_transition.elements must retain the source coordinate elements; got different target dimension or elements"
            )
        refined = _witness(refinement, dimension, budget=budget)
        coarsened = _witness(coarsening, dimension, budget=budget)
        target_ids = np.concatenate([block.cell_ids for block in target_blocks])
        covered = np.concatenate(
            (refined.fine_cell_ids, np.unique(coarsened.coarse_cell_ids))
        )
        if covered.size != target_ids.size or not np.array_equal(
            np.sort(covered), np.sort(target_ids)
        ):
            raise ValueError(
                "geometry_transition.witnesses must give each target cell one nested disposition; got missing or repeated reference record"
            )
        source_coordinates = np.asarray(source_geometry.coordinates, dtype=np.float64)
        values = _empty_values(target_blocks, source_coordinates.shape[1])
        restricted = _restrict(
            source_blocks, source_coordinates, target_blocks, refined, values, policy_
        )
        coarse = _coarsen(
            source_blocks,
            source_coordinates,
            target_blocks,
            coarsened,
            values,
            policy_,
            budget=budget,
        )
        _budget(restricted.evaluations + coarse.evaluations, policy_)
        placed = place(
            target_blocks, values, np.asarray(target_layout.coordinates).shape[0]
        )
        slack = policy_.continuity_tolerance * _extent(source_coordinates)
        if placed.continuity > slack:
            raise transition_types.CellGeometryTransitionError(
                "discontinuous_source",
                "geometry_transition.continuity must agree at shared target nodes; got excess incident-cell discrepancy",
                measured=placed.continuity,
                limit=slack,
            )
        match policy_.coarsening:
            case "exact_only":
                tolerance = slack
            case "bounded_interpolation":
                tolerance = policy_.coarsening_tolerance + slack
            case approximation:
                assert_never(approximation)
        if coarse.bound > tolerance:
            raise transition_types.CellGeometryTransitionError(
                "approximation_bound",
                f"geometry_transition.approximation_bound must fit the {policy_.coarsening} policy; got excess coarse-map deviation",
                measured=coarse.bound,
                limit=tolerance,
            )
        elements, routes, _ = target_layout.resolve(target_mesh, budget=budget)
        names = tuple(block.name for block in target_mesh.blocks)
        geometry = CellGeometrySpec(
            dict(zip(names, elements, strict=True)),
            dict(zip(names, routes, strict=True)),
            placed.coordinates,
            budget=budget,
        )
        source_measure, measure_exact = _measure(source_blocks, source_coordinates)
        target_measure, _ = _measure(target_blocks, placed.coordinates)
        containment = max(
            _witness_excursion(refined),
            _witness_excursion(coarsened),
        )
        evidence = transition_types.CellGeometryTransitionEvidence(
            "coarsening_interpolation"
            if coarsened.fine_cell_ids.size
            else "nested_restriction",
            exact=containment <= 64.0 * _EPSILON and coarse.bound <= slack,
            node_count=placed.coordinates.shape[0],
            evaluation_count=restricted.evaluations + coarse.evaluations,
            containment_defect=containment,
            continuity_residual=placed.continuity,
            approximation_bound=coarse.bound,
            approximation_tolerance=tolerance,
            rounding_slack=slack,
            source_measure=source_measure,
            target_measure=target_measure,
            measure_exact=measure_exact,
            coverage_defect=abs(target_measure - source_measure)
            / max(abs(source_measure), np.finfo(np.float64).tiny),
        )
        return _transition(
            evidence.kind,
            source_mesh,
            source_geometry,
            target_mesh,
            geometry,
            placed,
            restricted,
            coarsened,
            evidence,
            policy_,
            budget=budget,
        )


type _DisplacedMeasures = tuple[float, float, float | None, float | None, bool]


@dataclass(frozen=True, slots=True)
class _DisplacedMeasureIssue:
    """Numerics earned by one actual transition; no reconstructed proof authority."""

    source_mesh: CellMesh
    source_geometry: CellGeometrySpec
    target_mesh: CellMesh
    target_geometry_id: str
    names: tuple[str, ...]
    values: tuple[object, ...]
    owners: OwnerFields
    containers: tuple[tuple[list[object], tuple[object, ...]], ...]
    immutable_arrays: tuple[Array, ...]
    mutable_arrays: tuple[tuple[np.ndarray, str, tuple[int, ...], bytes], ...]
    measures: _DisplacedMeasures
    metadata_bytes: int


# Identity keys avoid Equinox value equality. Neither the seal nor its callback
# owns the transition: copies, decoded records and collected owners have no issue.
_ISSUED_DISPLACED_MEASURES: dict[
    int, tuple[ReferenceType[transition_types.CellGeometryTransition], _DisplacedMeasureIssue]
] = {}


def _issue_displaced_measures(
    transition: transition_types.CellGeometryTransition,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    measures: _DisplacedMeasures,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    if not budget.capture_coordinate_operands or not budget.capture_measure_operands:
        return
    from .._meshcore import NativeBatchTicket, NativeScopeTicket

    if not isinstance(budget.ticket, (NativeBatchTicket, NativeScopeTicket)):
        return
    _coordinate_operand_request(budget)
    names = tuple(member.name for member in fields(transition))
    budget.reserve(len(names), 128 + 32 * len(names))
    values = tuple(getattr(transition, name) for name in names)
    arrays: list[Array | np.ndarray] = []
    containers: list[tuple[list[object], tuple[object, ...]]] = []
    owners = coordinate_input_owner_fields(
        (source_mesh, source_geometry, target_mesh, *values),
        budget,
        arrays=arrays,
        containers=containers,
    )
    if owners is None:
        return
    immutable_arrays = tuple(array for array in arrays if isinstance(array, Array))
    if any(array.is_deleted() for array in immutable_arrays):
        return
    mutable_arrays = []
    for array in arrays:
        if type(array) is np.ndarray:
            budget.reserve(array.size + 1, array.nbytes + 256)
            mutable_arrays.append(
                (array, array.dtype.str, array.shape, array.tobytes(order="C"))
            )
    metadata = (
        1024 + 64 * len(names) + 16 * len(immutable_arrays)
        + sum(256 + 64 * len(names_) for _, _, names_, _ in owners)
        + sum(128 + 16 * len(children) for _, children in containers)
        + sum(256 + len(payload) for _, _, _, payload in mutable_arrays)
    )
    budget.retain_operand_metadata(metadata)
    # This exact numeric tuple becomes a kernel operand profile in the actual
    # earning budget. Only a paid, privately issued PreparedCoordinateOperands
    # containing that profile can authorize its later use.
    budget.retain_basis(measures, operand=True, _kernel_operands=True)
    issue = _DisplacedMeasureIssue(
        source_mesh, source_geometry, target_mesh, transition.target_geometry_id,
        names, values, owners, tuple(containers), immutable_arrays,
        tuple(mutable_arrays), measures, metadata,
    )
    identity = id(transition)
    _ISSUED_DISPLACED_MEASURES[identity] = (
        ref(transition, lambda _: _ISSUED_DISPLACED_MEASURES.pop(identity, None)),
        issue,
    )


def _displaced_measures_for(
    transition: transition_types.CellGeometryTransition | None,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    geometry: CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> _DisplacedMeasures | None:
    if type(transition) is not transition_types.CellGeometryTransition:
        return None
    budget.reserve(1)
    entry = _ISSUED_DISPLACED_MEASURES.get(id(transition))
    if entry is None or entry[0]() is not transition:
        return None
    issue = entry[1]
    prepared = budget._prepared_coordinate_operands
    if (
        issue.source_mesh is not source_mesh
        or issue.source_geometry is not source_geometry
        or issue.target_mesh is not target_mesh
        or prepared is None
        or prepared.mesh is not target_mesh
    ):
        return None
    budget.reserve(len(prepared._profiles) + len(issue.names))
    if (
        not any(profile.basis is issue.measures for profile in prepared._profiles)
        or any(
            getattr(transition, name) is not value
            for name, value in zip(issue.names, issue.values, strict=True)
        )
    ):
        return None
    with budget.temporary_scope():
        budget.reserve(0, issue.metadata_bytes)
        if (
            not require_coordinate_input_owners(issue.owners, budget)
            or not require_coordinate_input_containers(issue.containers, budget)
        ):
            return None
        budget.reserve(len(issue.immutable_arrays))
        if any(array.is_deleted() for array in issue.immutable_arrays):
            return None
        for array, dtype, shape, payload in issue.mutable_arrays:
            with budget.temporary_scope():
                budget.reserve(array.size + 1, array.nbytes + 256)
                if (
                    array.dtype.str != dtype
                    or array.shape != shape
                    or array.dtype.hasobject
                    or array.tobytes(order="C") != payload
                ):
                    return None
        if cell_geometry_id(geometry) != issue.target_geometry_id:
            return None
        prepared.require(
            target_mesh, prepared.geometry, prepared.earning_receipt, budget,
            coordinate_contract=prepared._coordinate_contract,
        )
        return issue.measures


@checked
def transition_displaced_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    /,
    *,
    policy: transition_types.CellGeometryTransitionPolicy | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    _issued_transition: transition_types.CellGeometryTransition | None = None,
) -> transition_types.CellGeometryTransition:
    """Move a coordinate map with its mesh vertices on an unchanged topology.

    Every coordinate node moves by the affine interpolation of its cell's corner
    displacements, ``x'(xi) = x(xi) + sum_i lambda_i(xi) (v'_i - v_i)``: vertex
    nodes land on the moved vertices and the curved part of each map is kept.
    A shared node moves identically from every incident cell because the
    displacement on a shared entity depends only on that entity's corners.
    Exact PLC ancestry admits only unstratified interior-node displacement.
    Every stratified source row stays fixed and the target retains the same
    exact source graph. A live issued transition may supply only its paid numeric
    measure tuple after current owner and operand admission; geometry, witnesses,
    evidence and identities are always reconstructed.
    """

    policy_ = (
        transition_types.CellGeometryTransitionPolicy() if policy is None else policy
    )
    if target_mesh.topology_id != source_mesh.topology_id:
        raise ValueError(
            "geometry_transition.target_topology must equal the source topology; got different topology ID"
        )
    plc = isinstance(source_geometry.exact_source, ExactPlcCellGeometrySource)
    if isinstance(source_geometry.exact_source, ExactPlcCellGeometryConvexSource):
        raise ValueError(
            "geometry_transition.source must support coefficient displacement; got convex PLC source ancestry"
        )
    with (
        (
            nullcontext(budget)
            if budget is not None
            else owned_coordinate_preparation(
                policy_.maximum_evaluations,
                sys.maxsize,
                maximum_geometry_queries=policy_.maximum_evaluations,
            )
        ) as budget,
        transition_budget_owner.charged_geometry_transition(policy_, budget=budget),
    ):
        blocks = simplex_blocks(source_mesh, source_geometry, "source", budget=budget)
        _budget(
            sum(block.routes.size * (block.dimension + 1) for block in blocks), policy_
        )
        source_coordinates = np.asarray(source_geometry.coordinates, dtype=np.float64)
        if plc:
            source = source_geometry.exact_source
            assert isinstance(source, ExactPlcCellGeometrySource)
            target_coordinates = np.asarray(target_mesh.coordinates, dtype=np.float64)
            if (
                source_coordinates.shape != target_coordinates.shape
                or not np.array_equal(source_coordinates, source_mesh.coordinates)
                or any(block.degree != 1 for block in blocks)
            ):
                raise ValueError(
                    "geometry_transition.PLC displacement requires the unchanged affine vertex-node layout"
                )
            stratified = np.asarray(source.vertex_strata) != 0
            if not np.array_equal(
                target_coordinates[stratified], source_coordinates[stratified]
            ):
                raise ValueError(
                    "geometry_transition.PLC displacement must retain every stratified source row, including embedded features and interfaces"
                )
            # These are exact free-node binary64 authorities, not a rounded
            # source + (target - source) arithmetic reconstruction.
            values = [target_coordinates[block.routes] for block in blocks]
        else:
            displacement = np.asarray(target_mesh.coordinates, dtype=np.float64) - np.asarray(
                source_mesh.coordinates, dtype=np.float64
            )
            values = []
            for mesh_block, block in zip(source_mesh.blocks, blocks, strict=True):
                nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
                corners = displacement[np.asarray(mesh_block.vertices, dtype=np.int64)]
                values.append(
                    source_coordinates[block.routes]
                    + np.asarray(contract("nk,ckD->cnD", _barycentric(nodes), corners))
                )
        placed = place(blocks, values, source_coordinates.shape[0])
        slack = policy_.continuity_tolerance * _extent(source_coordinates)
        if placed.continuity > slack:
            raise transition_types.CellGeometryTransitionError(
                "discontinuous_source",
                "geometry_transition.continuity must agree at shared displaced nodes; got excess incident-cell discrepancy",
                measured=placed.continuity,
                limit=slack,
            )
        elements, routes, _ = source_geometry.resolve(source_mesh, budget=budget)
        names = tuple(block.name for block in source_mesh.blocks)
        geometry = (
            source_geometry.with_coordinates(placed.coordinates, budget=budget)
            if plc
            else CellGeometrySpec(
                dict(zip(names, elements, strict=True)),
                dict(zip(names, routes, strict=True)),
                placed.coordinates,
            )
        )
        measures = (
            _displaced_measures_for(
                _issued_transition, source_mesh, source_geometry, target_mesh,
                geometry, budget,
            )
            if plc else None
        )
        source_error = target_error = None
        if measures is not None:
            source_measure, target_measure, source_error, target_error, measure_exact = measures
        elif plc:
            source_values, source_errors, source_exact = (
                mapped_measure_owner.certified_cell_measures(
                    source_mesh, source_geometry, budget=budget
                )
            )
            target_values, target_errors, target_exact = (
                mapped_measure_owner.certified_cell_measures(
                    target_mesh, geometry, budget=budget
                )
            )
            source_measure, target_measure = math.fsum(source_values), math.fsum(target_values)
            source_error, target_error = math.fsum(source_errors), math.fsum(target_errors)
            measure_exact = source_exact and target_exact
        else:
            source_measure, measure_exact = _measure(blocks, source_coordinates)
            target_measure, _ = _measure(blocks, placed.coordinates)
        # Witnesses follow ascending cell IDs: every cell is its own identity parent.
        identity = np.sort(np.concatenate([block.cell_ids for block in blocks]))
        dimension = source_mesh.topological_dimension
        reference = np.broadcast_to(
            np.concatenate((np.zeros((1, dimension)), np.eye(dimension)), axis=0),
            (identity.size, dimension + 1, dimension),
        ).copy()
        evidence = transition_types.CellGeometryTransitionEvidence(
            "vertex_displacement",
            exact=True,
            node_count=placed.coordinates.shape[0],
            evaluation_count=sum(
                block.routes.size * (block.dimension + 1) for block in blocks
            ),
            containment_defect=0.0,
            continuity_residual=placed.continuity,
            approximation_bound=0.0,
            approximation_tolerance=slack,
            rounding_slack=slack,
            source_measure=source_measure,
            target_measure=target_measure,
            measure_exact=measure_exact,
            coverage_defect=None,
            source_measure_error_bound=source_error,
            target_measure_error_bound=target_error,
        )
        transition = _transition(
            "vertex_displacement",
            source_mesh,
            source_geometry,
            target_mesh,
            geometry,
            placed,
            _Restricted(identity, reference, 0, None),
            _witness(None, dimension, budget=budget),
            evidence,
            policy_,
            budget=budget,
        )
        if plc and measures is None:
            _issue_displaced_measures(
                transition, source_mesh, source_geometry, target_mesh,
                (source_measure, target_measure, source_error, target_error, measure_exact),
                budget,
            )
        return transition


@checked
def is_affine_cell_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> bool:
    """Prove a full affine source map bound to its approved numerical carrier."""
    if budget is not None:
        key = coordinate_scope_key(mesh, geometry)
        if key in budget.affine_source_carrier_cache:
            return True
        admitted = _affine_source_carrier(mesh, geometry, budget=budget)
        if admitted:
            budget.retain_basis((key,))
            budget.affine_source_carrier_cache.add(key)
        return admitted
    work_upper, bytes_upper = cell_geometry_preparation_bounds(geometry)
    # Convex source layout authentication and the coefficient bank may each
    # prepare the source; all other source routes use no more repetitions.
    with owned_coordinate_preparation(
        2 * work_upper,
        2 * bytes_upper,
        maximum_geometry_queries=2 * cell_geometry_preparation_query_upper(geometry),
    ) as budget:
        return _affine_source_carrier(mesh, geometry, budget=budget)


def _affine_source_carrier(
    mesh: CellMesh, geometry: CellGeometrySpec, /, *, budget: CoordinateEnclosureBudget
) -> bool:

    bank = imported_coordinate_enclosure.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    elements, routes, _ = geometry._resolve(
        mesh, exact_source_prepared=True, budget=budget
    )
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    structural = tuple(
        not isinstance(element, imported_cell_geometry.CellVertexGeometryElement)
        and imported_coordinate_enclosure.affine_simplex_source(element, budget=budget)
        for element in elements
    )
    # Authenticate all incident structural simplex charts together. This evaluates
    # each distinct routed corner operation once, including shared-vertex equality,
    # rather than rebuilding a complete local coefficient key for every cell.
    if all(structural):
        images = imported_coordinate_enclosure.prepared_coordinate_vertex_images(
            mesh, geometry,
            mismatch="Affine source charts must agree on every shared vertex; got unequal exact images.",
            budget=budget,
        )
        if images is not None:
            selected = [index for index, image in enumerate(images) if image is not None]
            if not selected:
                return True
            rounded = np.asarray(
                [[float(value) for value in image] for image in images if image is not None],
                dtype=np.float64,
            )
            actual = carrier[selected]
            if np.array_equal(rounded.view(np.uint64), actual.view(np.uint64)):
                return True
            if isinstance(geometry.exact_source, ExactPlcCellGeometrySource):
                approved = np.asarray(geometry.coordinates, dtype=np.float64)[selected]
                return (
                    all(images[index] == bank[index] for index in selected)
                    and np.array_equal(approved.view(np.uint64), actual.view(np.uint64))
                )
            return False
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        vertices = np.asarray(block.vertices)
        rows = np.asarray(route)
        if isinstance(element, imported_cell_geometry.CellVertexGeometryElement):
            # Variable-topology elements are the vertex carrier, not a
            # tensor Q1 chart. Its source law was renewed above.
            if not np.array_equal(rows, vertices):
                return False
            selected = np.unique(vertices[vertices >= 0])
            rounded = np.asarray(
                [[float(value) for value in bank[int(index)]] for index in selected],
                dtype=np.float64,
            )
            if not np.array_equal(
                rounded.view(np.uint64), carrier[selected].view(np.uint64)
            ):
                return False
            continue
        # A degree-one simplex root proves every restricted/barycentric row
        # affine once; its rows then need only their exact corner images.
        structural = imported_coordinate_enclosure.affine_simplex_source(
            element, budget=budget
        )
        for row, corners in zip(rows, vertices, strict=True):
            local = tuple(bank[int(index)] for index in row if index >= 0)
            images = (
                imported_coordinate_enclosure.coordinate_corner_images(
                    element, local, budget=budget
                )
                if structural
                else None
            )
            if images is None:
                polynomials = imported_coordinate_enclosure.coordinate_polynomials(
                    element, local, budget=budget
                )
                if polynomials is None or any(
                    sum(exponent) > 1
                    for polynomial in polynomials
                    for exponent, coefficient in polynomial.items()
                    if coefficient
                ):
                    return False
                images = imported_coordinate_enclosure.corner_images(
                    polynomials, block.cell_kind, budget=budget
                )
            rounded = np.asarray(
                [[float(value) for value in image] for image in images],
                dtype=np.float64,
            )
            actual = carrier[corners[corners >= 0]]
            if rounded.shape != actual.shape or not np.array_equal(
                rounded.view(np.uint64),
                actual.view(np.uint64),
            ):
                match geometry.exact_source:
                    case ExactPlcCellGeometrySource():
                        selected = corners[corners >= 0]
                        approved = np.asarray(geometry.coordinates, dtype=np.float64)[
                            selected
                        ]
                        if images != tuple(
                            bank[int(index)] for index in selected
                        ) or not np.array_equal(
                            approved.view(np.uint64), actual.view(np.uint64)
                        ):
                            return False
                    case _:
                        return False
    return True


@checked
def affine_vertex_positions(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...]:
    """Exact positions of every scientific vertex under an affine source map.

    Row ``i`` is the vertex ``mesh.vertex_global_ids[i]``. Every incident cell's
    affine source law (any admitted family: P1, coefficient actions, restricted
    or composed charts, unrounded PLC/rational/algebraic sources) is evaluated at its
    reference corners through the coordinate-source owner, keeping its
    `ExactScalar` field values, and all incident cells must agree exactly on the
    same vertex. The binary64 carrier must be the round-to-nearest view of each
    position (the same rounding `is_affine_cell_geometry` admits), except where
    an unrounded PLC source approves its own certified carrier.
    """
    bank = imported_coordinate_enclosure.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    elements, routes, _ = geometry._resolve(
        mesh, exact_source_prepared=True, budget=budget
    )
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    witness_carrier = isinstance(geometry.exact_source, ExactPlcCellGeometrySource)
    count = int(carrier.shape[0])
    budget.reserve(count, 64 + 8 * count)
    positions: list[tuple[imported_coordinate_enclosure.ExactScalar, ...] | None] = [
        None
    ] * count
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        if block.cell_kind not in _SIMPLEX_DIMENSIONS:
            raise ValueError(
                "affine_vertex_positions.mesh must contain simplex cells; got non-simplex block"
            )
        for row, corners in zip(
            np.asarray(route), np.asarray(block.vertices), strict=True
        ):
            local = tuple(bank[int(index)] for index in row if index >= 0)
            polynomials = imported_coordinate_enclosure.coordinate_polynomials(
                element, local, budget=budget
            )
            if polynomials is None or any(
                sum(exponent) > 1
                for polynomial in polynomials
                for exponent, coefficient in polynomial.items()
                if coefficient
            ):
                raise ValueError(
                    "affine_vertex_positions.geometry must be an affine source map; got non-affine cell"
                )
            images = imported_coordinate_enclosure.corner_images(
                polynomials, block.cell_kind, budget=budget
            )
            for vertex, image in zip(corners.tolist(), images, strict=True):
                point = tuple(image)
                previous = positions[int(vertex)]
                if previous is not None and previous != point:
                    raise ValueError(
                        "affine_vertex_positions.geometry must agree at shared vertices; got unequal unrounded positions"
                    )
                if (
                    previous is None
                    and not witness_carrier
                    and (
                        tuple(_rounded_scalar(value, budget=budget)[0] for value in point)
                        != tuple(carrier[int(vertex), :].tolist())
                    )
                ):
                    raise ValueError(
                        "affine_vertex_positions.mesh.coordinates must be the round-to-nearest view of the exact source vertices; got different carrier"
                    )
                positions[int(vertex)] = point
    result: list[tuple[imported_coordinate_enclosure.ExactScalar, ...]] = []
    for point in positions:
        if point is None:
            raise ValueError(
                "affine_vertex_positions.mesh must reference every vertex row; got unreferenced vertex"
            )
        result.append(point)
    return tuple(result)


def _rounded_scalar(
    value: imported_coordinate_enclosure.ExactScalar,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, Fraction]:
    """Round-to-nearest binary64 view and a rational bound of its error.

    Rational values (including rational-subfield field elements) have their
    rational error, zero iff representable; irrational values use the field
    owner's certified budgeted rounding.
    """
    if isinstance(value, RadicalElement):
        if not value.terms or (len(value.terms) == 1 and value.terms[0][0] == 1):
            value = rational_coefficient(value, role="exact_root.coordinate")
        else:
            return value.rounded(budget=budget)
    rounded = float(value)
    return rounded, abs(Fraction(rounded) - value)


@checked
def vertex_positions_are_carrier(
    positions: tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...],
    mesh: CellMesh,
    /,
) -> bool:
    """Whether exact vertex positions are exactly the mesh's binary64 carrier."""
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    return len(positions) == carrier.shape[0] and all(
        len(point) == carrier.shape[1]
        and all(
            value == Fraction(float(rounded))
            for value, rounded in zip(point, carrier[index, :].tolist(), strict=True)
        )
        for index, point in enumerate(positions)
    )


@checked
def affine_geometry_is_carrier(mesh: CellMesh, geometry: CellGeometrySpec, /) -> bool:
    """Source-owner proof that the geometry is the mesh's own binary64 P1 carrier.

    True only for the standard affine layout with no unrounded, periodic,
    restriction, or storage source and bitwise-equal coordinates, so the unrounded
    vertex positions are the carrier itself. False means "not proved", and
    callers must then use `affine_vertex_positions`.
    """
    return (
        geometry.exact_source is None
        and geometry.periodic_source is None
        and geometry.restriction_source is None
        and geometry.storage is None
        and geometry.geometry_layout_id
        == imported_cell_geometry.CellGeometrySpec.affine(mesh).geometry_layout_id
        and np.array_equal(
            np.asarray(geometry.coordinates, dtype=np.float64).view(np.uint64),
            np.asarray(mesh.coordinates, dtype=np.float64).view(np.uint64),
        )
    )


@checked
def exact_root_points(
    positions: tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...],
    sources: NDArray[np.int64] | NDArray[np.int32],
    homogeneous: NDArray[np.float64],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Round-to-nearest carriers and certified deviations of exact P1 roots.

    Root ``r`` is ``S = (h[r,0] P[s0] + h[r,1] P[s1]) / (h[r,0] + h[r,1])`` over
    exact source positions ``P`` from `affine_vertex_positions` (rational or
    multiquadratic field values); its carrier is ``RNE(S)`` and its deviation an
    upward-rounded Euclidean ``|carrier - S|`` (zero iff equal), the norm of
    `LevelSetEvidence.root_deviations`.
    """
    rows = np.asarray(sources)
    weights = np.asarray(homogeneous, dtype=np.float64)
    if (
        rows.ndim != 2
        or rows.shape[1] != 2
        or weights.shape != rows.shape
        or not np.all(np.isfinite(weights))
        or np.any(np.signbit(weights))
        or not np.all(np.any(weights != 0.0, axis=1))
    ):
        raise ValueError(
            "exact_root_points.inputs must be aligned (roots, 2) sources and positive-signed homogeneous rows; got incompatible arrays"
        )
    if any(
        not isinstance(value, (Fraction, RadicalElement))
        or len(point) != len(positions[0])
        for point in positions
        for value in point
    ):
        raise TypeError(
            "exact_root_points.positions must be aligned unrounded coordinate rows; got incompatible positions"
        )
    if np.any(rows < 0) or np.any(rows >= len(positions)):
        raise ValueError(
            "exact_root_points.sources must index mesh vertex rows; got out-of-range source"
        )
    ambient = len(positions[0]) if positions else 0
    count = rows.shape[0]
    used = {int(index) for index in rows.reshape(-1).tolist()}
    profile = tuple(positions[index] for index in sorted(used))
    bits = max(
        (coefficient_bits(value) for point in profile for value in point), default=1
    )
    # Two products, one sum and one quotient per component; carriers and
    # deviations are the only escaping storage.
    budget.reserve(
        6 * count * max(ambient, 1),
        count * max(ambient, 1) * (256 + 2 * ((bits + 2 * 1076 + 64 + 7) // 8)),
    )
    if any(isinstance(value, RadicalElement) for point in profile for value in point):
        # Field products, quotients and certified rounding of each component.
        reserve_algebraic_work(
            profile,
            6 * count * max(ambient, 1),
            count * max(ambient, 1),
            2,
            budget=budget,
        )
    carriers = np.empty((count, ambient), dtype=np.float64)
    deviations = np.zeros(count, dtype=np.float64)
    for root in range(count):
        first, second = (Fraction(float(value)) for value in weights[root, :].tolist())
        total = first + second
        start, end = positions[int(rows[root, 0])], positions[int(rows[root, 1])]
        squared = Fraction(0)
        for axis in range(ambient):
            exact = (first * start[axis] + second * end[axis]) / total
            rounded, error = _rounded_scalar(exact, budget=budget)
            carriers[root, axis] = rounded
            squared += error**2
        _, upper = fraction_sqrt_interval(squared, budget=budget)
        deviations[root] = fraction_endpoint(upper, upper=True)
    return carriers, deviations


@checked
def nested_geometry_degree(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> int:
    """Layout preparation degree of an admitted nested coordinate map.

    Simplex nodal restriction requires one common specified coordinate degree.
    Mixed coefficient composition retains every block's own source degree; the
    returned maximum is only the preparation extent, not a geometry downgrade.
    """

    if budget is None:
        work_upper, bytes_upper = cell_geometry_preparation_bounds(geometry)
        request = owned_coordinate_preparation(
            2 * work_upper,
            2 * bytes_upper,
            maximum_geometry_queries=2 * cell_geometry_preparation_query_upper(geometry),
        )
    else:
        request = nullcontext(budget)

    with request as budget:
        elements, _, _ = geometry.resolve(mesh, budget=budget)
        if any(
            block.cell_kind not in _SIMPLEX_DIMENSIONS for block in mesh.blocks
        ) or any(
            not isinstance(element, reference_types_owner.FiniteElementSpec)
            for element in elements
        ):
            if any(
                not isinstance(
                    element,
                    (
                        reference_types_owner.FiniteElementSpec,
                        BarycentricCellGeometryElement,
                        RestrictedCellGeometryElement,
                        PolynomialComposedCellGeometryElement,
                        RationalComposedCellGeometryElement,
                        SplineCellGeometryElement,
                        LayerColumnCellGeometryElement,
                    ),
                )
                for element in elements
            ):
                raise ValueError(
                    "nested_geometry.elements must provide scalar reference coordinate maps; got unsupported element"
                )
            return max(
                require_scalar_coordinate_element(element, "Nested mixed geometry").degree
                for element in elements
            )
        degrees = {
            block.degree
            for block in simplex_blocks(mesh, geometry, "source", budget=budget)
        }
        if len(degrees) != 1:
            raise ValueError(
                "nested_geometry.degree must be common across simplex blocks; got different coordinate degrees"
            )
        return degrees.pop()


@checked
def cell_geometry_vertex_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> np.ndarray:
    """Integral of every mesh-vertex P1 hat function over the mapped cells.

    Full simplex Lagrange maps and their polynomial restrictions
    (barycentric actions, restricted and composed charts, as published by
    nested refinement) are measured. The rule is algebraic for full-dimensional
    polynomial coordinate maps (hat degree one plus the Jacobian determinant
    degree).
    """

    if budget is None:
        work_upper, bytes_upper = cell_geometry_preparation_bounds(geometry)
        request = owned_coordinate_preparation(
            work_upper,
            bytes_upper,
            maximum_geometry_queries=cell_geometry_preparation_query_upper(geometry),
        )
    else:
        request = nullcontext(budget)
    with request as budget:
        elements, routes, _ = geometry.resolve(mesh, budget=budget)
        coordinates = np.asarray(geometry.coordinates, dtype=np.float64)
        measures = np.zeros((mesh.coordinates.shape[0],), dtype=np.float64)
        for mesh_block, element, route in zip(mesh.blocks, elements, routes, strict=True):
            measured: _PolynomialSimplexElement
            cell_dimension: int
            match element, _SIMPLEX_DIMENSIONS.get(mesh_block.cell_kind):
                case reference_types_owner.FiniteElementSpec(), int() as dimension:
                    if (
                        element.element_id
                        != imported_cell_geometry.coordinate_lagrange_element(
                            mesh_block.cell_kind, element.degree
                        ).element_id
                    ):
                        raise ValueError(
                            f"geometry_transition.measured.elements must span the full Lagrange space; got incompatible element in block {mesh_block.name!r}"
                        )
                    measured, cell_dimension = element, dimension
                case (
                    BarycentricCellGeometryElement()
                    | RestrictedCellGeometryElement()
                    | PolynomialComposedCellGeometryElement(),
                    int() as dimension,
                ) if element.cell_kind == mesh_block.cell_kind:
                    measured, cell_dimension = element, dimension
                case _:
                    raise ValueError(
                        f"geometry_transition.measured.elements must be full simplex Lagrange maps or their polynomial restrictions; got {mesh_block.cell_kind} block {mesh_block.name!r}"
                    )
            points, weights = reference_rules_owner.host_degree_aware_reference_rule(
                mesh_block.cell_kind, 1 + cell_dimension * max(measured.degree - 1, 0)
            )
            points = np.asarray(points, dtype=np.float64)
            local = np.asarray(
                contract(
                    "cq,q,qk->ck",
                    _element_densities(
                        measured,
                        np.asarray(route, dtype=np.int64),
                        cell_dimension,
                        coordinates,
                        points,
                    ),
                    np.asarray(weights, dtype=np.float64),
                    _barycentric(points),
                )
            )
            np.add.at(
                measures,
                np.asarray(mesh_block.vertices, dtype=np.int64).reshape((-1,)),
                local.reshape((-1,)),
            )
        return measures


def _reference_affine(kind: str, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    dimension = vertices.shape[1]
    frame_indices = (
        (1, 3)
        if kind == "quadrilateral"
        else (1, 2)
        if kind == "triangle"
        else (1,)
        if kind == "interval"
        else (1, 2, 3)
        if kind in ("tetrahedron", "prism")
        else (1, 3, 4)
    )
    frame = (vertices[list(frame_indices)] - vertices[0]).T
    image = (corners[list(frame_indices)] - corners[0]).T
    solved = solve_small_linear(SmallLinearSolvePlan(dimension), frame.T, image.T)
    if not bool(np.asarray(solved.successful)):
        raise ValueError(
            "nested_geometry.reference_frame must be nonsingular; got failed linear solve"
        )
    matrix = np.asarray(solved.value, dtype=np.float64).T
    offset = corners[0] - matrix @ vertices[0]
    if (
        np.max(
            np.abs(vertices @ matrix.T + offset - corners[: vertices.shape[0]]),
            initial=0.0,
        )
        > 64 * _EPSILON
    ):
        raise ValueError(
            "nested_geometry.reference_corners must equal an affine reference submap; got unequal reconstructed corners"
        )
    if (
        float(
            np.asarray(determinant_small_linear(SmallLinearSolvePlan(dimension), matrix))
        )
        <= 0
    ):
        raise ValueError(
            "nested_geometry.reference_jacobian must be positive; got nonpositive determinant"
        )
    return matrix, offset


def _mixed_map_measure(
    mesh: CellMesh, geometry: CellGeometrySpec, *, budget: CoordinateEnclosureBudget
) -> tuple[float, bool]:
    values, _, exact = mapped_measure_owner.certified_cell_measures(
        mesh, geometry, budget=budget
    )
    return math.fsum(values), exact


def _restricted_element(
    element: CellGeometryElement, kind: str, matrix: np.ndarray, offset: np.ndarray
) -> (
    reference_types_owner.FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement
):

    match element:
        case (
            reference_types_owner.FiniteElementSpec()
            | BarycentricCellGeometryElement()
            | RestrictedCellGeometryElement()
            | PolynomialComposedCellGeometryElement()
            | RationalComposedCellGeometryElement()
            | SplineCellGeometryElement()
            | LayerColumnCellGeometryElement()
        ):
            pass
        case _:
            raise ValueError(
                "nested_geometry.element must provide a scalar tabulated coordinate basis; got unsupported element"
            )
    element = require_scalar_coordinate_element(element, "Restricted coordinates")
    dimension = reference_cell_topology(kind).dimension
    if (
        element.cell_kind == kind
        and np.array_equal(matrix, np.eye(dimension, dtype=np.float64))
        and np.array_equal(offset, np.zeros(dimension, dtype=np.float64))
    ):
        return element
    return RestrictedCellGeometryElement(element, kind, matrix, offset)


def _nondyadic_homogeneous_rows(
    corners: NDArray[np.float64], bank: NDArray[np.float64] | None, /
) -> NDArray[np.bool_]:
    """Corners whose validated exact coordinate is not its binary64 view."""
    mask = np.zeros(corners.shape[0], dtype=np.bool_)
    if bank is None:
        return mask
    for index in range(min(bank.shape[0], corners.shape[0])):
        point = transition_types.homogeneous_reference_point(bank[index, :])
        mask[index] = point is not None and any(
            Fraction(float(value)) != value for value in point
        )
    return mask


def _homogeneous_child_element(
    element: CellGeometryElement,
    kind: str,
    source_kind: str,
    corners: tuple[tuple[Fraction, ...], ...],
    bank: NDArray[np.float64],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> BarycentricCellGeometryElement | PolynomialComposedCellGeometryElement:
    """Exact coordinate map of one source-owned nested simplex.

    Full P1 coefficient sources retain an exact barycentric action: row ``i``
    holds the parent barycentric coordinates of child corner ``i``, retained
    homogeneous rows authenticate their rounded execution rows, and zero rows
    must be exactly representable in binary64. Every other admitted parent
    (restricted, composed, or curved) is pulled back through the exact rational
    P1 chart of the child corners, so no rounded restriction is ever published.
    """
    if kind != source_kind or kind not in _SIMPLEX_DIMENSIONS:
        raise ValueError(
            "nested_geometry.reference_homogeneous must restrict a simplex to the same simplex family; got different cell kind"
        )
    match element:
        case (
            reference_types_owner.FiniteElementSpec()
            | BarycentricCellGeometryElement() as source
        ) if source.degree == 1 and source.local_dof_count == len(corners):
            rows = tuple(
                (Fraction(1) - sum(corner, Fraction(0)), *corner) for corner in corners
            )
            if any(
                not bank[index, :].any()
                and any(Fraction(float(value)) != value for value in row)
                for index, row in enumerate(rows)
            ):
                raise ValueError(
                    "nested_geometry.reference_homogeneous must retain every non-binary64 corner; got inexact binary corner action"
                )
            # The constructor re-proves every retained row exactly.
            budget.reserve(
                *imported_cell_geometry.barycentric_homogeneous_preparation_bounds(
                    len(rows)
                )
            )
            return BarycentricCellGeometryElement(
                source,
                np.asarray(
                    [[float(value) for value in row] for row in rows], dtype=np.float64
                ),
                bank,
            )
        case (
            reference_types_owner.FiniteElementSpec()
            | BarycentricCellGeometryElement()
            | RestrictedCellGeometryElement()
            | PolynomialComposedCellGeometryElement()
            | RationalComposedCellGeometryElement()
            | SplineCellGeometryElement()
            | LayerColumnCellGeometryElement() as source
        ):
            # Rational static (numerator, denominator) chart nodes; the composed
            # owner certifies the source expressions under them.
            budget.reserve(
                len(corners) * len(corners[0]),
                sum(
                    128
                    + (abs(value.numerator).bit_length() + 7) // 8
                    + (value.denominator.bit_length() + 7) // 8
                    for corner in corners
                    for value in corner
                ),
            )
            numerators = np.empty((len(corners), len(corners[0])), dtype=object)
            denominators = np.empty_like(numerators)
            for row, corner in enumerate(corners):
                for column, value in enumerate(corner):
                    numerators[row, column] = int(value.numerator)
                    denominators[row, column] = int(value.denominator)
            return PolynomialComposedCellGeometryElement(
                source,
                imported_cell_geometry.coordinate_lagrange_element(kind, 1),
                numerators,
                denominators,
                budget=budget,
            )
        case _:
            raise ValueError(
                "nested_geometry.reference_homogeneous must restrict a scalar coordinate source; got unsupported source element"
            )


def _reference_containment_defect(kind: str, points: NDArray[np.float64]) -> float:
    match kind:
        case "tetrahedron":
            defect = _excursion(points)
        case "prism":
            defect = max(
                _excursion(points[:, :2]),
                float(np.max(-points[:, 2], initial=0.0)),
                float(np.max(points[:, 2] - 1, initial=0.0)),
            )
        case "pyramid":
            z = points[:, 2]
            defect = float(
                max(
                    np.max(z / 2 - points[:, 0], initial=0.0),
                    np.max(z / 2 - points[:, 1], initial=0.0),
                    np.max(points[:, 0] - (1 - z / 2), initial=0.0),
                    np.max(points[:, 1] - (1 - z / 2), initial=0.0),
                    np.max(-z, initial=0.0),
                    np.max(z - 1, initial=0.0),
                )
            )
        case _:
            defect = float(
                max(np.max(-points, initial=0.0), np.max(points - 1, initial=0.0))
            )

    return defect


class _RestoredCoordinateCell(StrictModule):
    element: CellGeometryElement
    route: NDArray[np.int64]
    source_block: int
    ancestry_row: int


def _restore_coordinate_siblings(
    siblings: list[_MixedReference],
    target_kind: str,
    source_mesh: CellMesh,
    source_elements: tuple[CellGeometryElement, ...],
    source_routes: tuple[NDArray[np.int64], ...],
    source_index: dict[int, tuple[int, int]],
    source_coordinates: NDArray[np.float64],
    source_parent_cells: dict[str, NDArray[np.int64]],
    policy: transition_types.CellGeometryTransitionPolicy,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RestoredCoordinateCell:
    candidates = []
    for fine, corners, bank in siblings:
        index, fine_row = source_index[fine]
        restricted = source_elements[index]
        fine_kind = source_mesh.blocks[index].cell_kind
        arity = len(reference_cell_topology(fine_kind).vertices)
        if isinstance(restricted, BarycentricCellGeometryElement):
            # Full P1 coefficient actions, including source-owned roots, are
            # restored by exact rational action equality only.
            exact = transition_types.exact_reference_corners(
                corners[:arity], None if bank is None else bank[:arity]
            )
            expected = tuple(
                (Fraction(1) - sum(corner, Fraction(0)), *corner) for corner in exact
            )
            budget.reserve(
                *imported_cell_geometry.barycentric_homogeneous_preparation_bounds(
                    len(expected)
                )
            )
            if fine_kind != target_kind or (
                exact_barycentric_action(restricted) != expected
            ):
                raise transition_types.CellGeometryTransitionError(
                    "approximation_bound",
                    "nested_geometry.child_action must equal the recorded rational coefficient action; got different barycentric action",
                    measured=math.inf,
                    limit=policy.coarsening_tolerance,
                )
            candidates.append(
                (
                    restricted.source_element,
                    np.asarray(source_routes[index][fine_row], dtype=np.int64),
                )
            )
            continue
        nondyadic = _nondyadic_homogeneous_rows(corners[:arity], bank).any()
        if nondyadic and not isinstance(
            restricted,
            (PolynomialComposedCellGeometryElement, RationalComposedCellGeometryElement),
        ):
            raise transition_types.CellGeometryTransitionError(
                "approximation_bound",
                "nested_geometry.child_action must retain an exact coefficient action for source-owned roots; got rounded restriction",
                measured=math.inf,
                limit=policy.coarsening_tolerance,
            )
        matrix, offset = _reference_affine(fine_kind, corners)
        match restricted:
            case RestrictedCellGeometryElement():
                if not np.array_equal(
                    np.asarray(restricted.matrix), matrix
                ) or not np.array_equal(np.asarray(restricted.offset), offset):
                    raise transition_types.CellGeometryTransitionError(
                        "approximation_bound",
                        "nested_geometry.child_action must match the recorded affine restriction; got different matrix or offset",
                        measured=math.inf,
                        limit=policy.coarsening_tolerance,
                    )
                candidate = restricted.source_element
            case (
                PolynomialComposedCellGeometryElement()
                | RationalComposedCellGeometryElement()
            ):
                authored = imported_coordinate_enclosure.reference_composition_arguments(
                    restricted, budget=budget
                )
                expected = (
                    _exact_corner_affine_arguments(
                        transition_types.exact_reference_corners(
                            corners[:arity], None if bank is None else bank[:arity]
                        ),
                        budget=budget,
                    )
                    if nondyadic
                    else imported_coordinate_enclosure.affine_arguments(
                        offset, matrix, budget=budget
                    )
                )
                if authored != expected:
                    raise transition_types.CellGeometryTransitionError(
                        "approximation_bound",
                        "nested_geometry.child_action must equal the recorded affine reference action; got different authored polynomial action",
                        measured=math.inf,
                        limit=policy.coarsening_tolerance,
                    )
                candidate = restricted.source_element
            case _:
                raise transition_types.CellGeometryTransitionError(
                    "approximation_bound",
                    "nested_geometry.child_action must retain a recorded source restriction; got unsupported sibling element",
                    measured=math.inf,
                    limit=policy.coarsening_tolerance,
                )
        candidates.append(
            (
                candidate,
                np.asarray(source_routes[index][fine_row], dtype=np.int64),
            )
        )
    element, route = candidates[0]
    if any(
        candidate.element_id != element.element_id
        or not np.array_equal(source_coordinates[child_route], source_coordinates[route])
        for candidate, child_route in candidates[1:]
    ):
        raise transition_types.CellGeometryTransitionError(
            "approximation_bound",
            "nested_geometry.sibling_coefficients must agree on the parent map; got different parent elements or coefficients",
            measured=math.inf,
            limit=policy.coarsening_tolerance,
        )
    if element.cell_kind != target_kind:
        raise ValueError(
            "nested_geometry.parent_family must match the restored target family; got different cell kind"
        )
    index, ancestry_row = source_index[siblings[0][0]]
    ancestry_name = source_mesh.blocks[index].name
    ancestor = source_parent_cells[ancestry_name][ancestry_row]
    if any(
        source_parent_cells[source_mesh.blocks[source_index[fine][0]].name][
            source_index[fine][1]
        ]
        != ancestor
        for fine, *_ in siblings
    ):
        raise ValueError(
            "nested_geometry.source_parent must be common to all siblings; got different scientific parent IDs"
        )

    return _RestoredCoordinateCell(element, route, index, ancestry_row)


class _MixedParentIdentity(StrictModule):
    origin: CellGeometryRestrictionSource | None
    cells: dict[str, NDArray[np.int64]]
    vertices: dict[str, NDArray[np.int64]]


def _mixed_parent_identity(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_elements: tuple[CellGeometryElement, ...],
) -> _MixedParentIdentity:
    origin = source_geometry.restriction_source
    if origin is None and any(
        isinstance(element, RestrictedCellGeometryElement) for element in source_elements
    ):
        raise ValueError(
            "nested_geometry.source_parent must retain scientific parent geometry identities; got absent restriction source"
        )
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    if origin is None:
        source_parent_cells = {
            block.name: np.asarray(block.global_ids, dtype=np.int64)
            for block in source_mesh.blocks
        }
        source_parent_vertices = {
            block.name: source_vertex_ids[np.asarray(block.vertices)]
            for block in source_mesh.blocks
        }
    else:
        source_parent_cells = {
            name: np.asarray(value)
            for name, value in origin.block_parent_cell_ids.items()
        }
        source_parent_vertices = {
            name: np.asarray(value)
            for name, value in origin.block_parent_vertex_ids.items()
        }

    return _MixedParentIdentity(origin, source_parent_cells, source_parent_vertices)


class _MixedGeometryCells(StrictModule):
    source_coordinates: NDArray[np.float64]
    target_ids: NDArray[np.int64]
    parents: NDArray[np.int64]
    reference: NDArray[np.float64]
    elements: dict[str, CellGeometryElement]
    routes: dict[str, NDArray[np.int64]]
    origin_cells: dict[str, NDArray[np.int64]]
    origin_vertices: dict[str, NDArray[np.int64]]
    owner: dict[int, tuple[int, int]]
    vertex_values: dict[int, NDArray[np.float64]]
    continuity: float
    containment: float
    coarsening: transition_types.NestedReferenceWitnesses | None
    homogeneous: NDArray[np.float64] | None


def _require_mixed_refinement_partition(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    geometry: CellGeometrySpec,
    prepared: _MixedGeometryCells,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove total-measure equality by the canonical oriented reference tiling.

    The mixed coordinate builder has already established each exact source
    restriction. The independent nested-reference owner proves containment,
    positive orientation, interior-facet cancellation and full parent coverage;
    equality of child determinant sums alone would not suffice.
    """
    # Deferred: the reference owner also consumes geometry-transition records.
    from ._nested_reference import nested_reference_pairs

    source_count = sum(block.cell_count for block in source_mesh.blocks)
    target_count = prepared.target_ids.size
    budget.reserve(
        source_count + target_count,
        512
        + 256 * source_count
        + 512 * target_count
        + prepared.reference.nbytes
        + (0 if prepared.homogeneous is None else prepared.homogeneous.nbytes),
    )
    source_ids = np.concatenate([np.asarray(block.global_ids) for block in source_mesh.blocks])
    target_ids = np.concatenate([np.asarray(block.global_ids) for block in target_mesh.blocks])
    if set(source_ids.tolist()) != set(prepared.parents.tolist()):
        raise ValueError(
            "nested_geometry.partition must cover every source cell; got missing source parents"
        )
    source_rows = {int(identifier): row for row, identifier in enumerate(source_ids)}
    witness_rows = {
        int(identifier): row for row, identifier in enumerate(prepared.target_ids)
    }
    if len(witness_rows) != target_count or set(target_ids.tolist()) != set(witness_rows):
        raise ValueError(
            "nested_geometry.partition must bind every target cell once; got unequal witness IDs"
        )
    order = np.asarray([witness_rows[int(identifier)] for identifier in target_ids], dtype=np.int64)
    parents = np.asarray(
        [source_rows[int(prepared.parents[row])] for row in order], dtype=np.int64
    )
    nested_reference_pairs(
        source_mesh,
        source_geometry,
        target_mesh,
        geometry,
        parent_cells=parents,
        parent_reference_vertices=prepared.reference[order],
        parent_reference_homogeneous=None
        if prepared.homogeneous is None
        else prepared.homogeneous[order],
        budget=budget,
    )


def _mixed_transition_evidence(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    geometry: CellGeometrySpec,
    prepared: _MixedGeometryCells,
    policy: transition_types.CellGeometryTransitionPolicy,
    slack: float,
    node_count: int,
    *,
    starting_work_units: int,
    budget: CoordinateEnclosureBudget,
) -> transition_types.CellGeometryTransitionEvidence:
    ledger = budget
    continuity, containment = prepared.continuity, prepared.containment
    coarsened = (
        prepared.coarsening is not None and prepared.coarsening.fine_cell_ids.size > 0
    )
    with transition_budget_owner.charged_geometry_transition(policy, budget=ledger):
        source_values, source_errors, measure_exact = (
            mapped_measure_owner.certified_cell_measures(
                source_mesh,
                source_geometry,
                budget=budget,
            )
        )
        if coarsened:
            target_values, target_errors, target_measure_exact = (
                mapped_measure_owner.certified_cell_measures(
                    target_mesh,
                    geometry,
                    budget=budget,
                )
            )
        else:
            _require_mixed_refinement_partition(
                source_mesh, source_geometry, target_mesh, geometry, prepared, budget=budget
            )
            target_values, target_errors, target_measure_exact = (
                source_values, source_errors, measure_exact
            )
    work = ledger.work_units - starting_work_units
    source_measure, target_measure = math.fsum(source_values), math.fsum(target_values)
    source_error = math.fsum(source_errors) + abs(float(np.spacing(source_measure)))
    target_error = math.fsum(target_errors) + abs(float(np.spacing(target_measure)))
    return transition_types.CellGeometryTransitionEvidence(
        "parent_restoration" if coarsened else "nested_restriction",
        exact=True,
        node_count=node_count,
        evaluation_count=work,
        containment_defect=containment,
        continuity_residual=continuity,
        approximation_bound=0.0,
        approximation_tolerance=slack,
        rounding_slack=slack,
        source_measure=source_measure,
        target_measure=target_measure,
        measure_exact=measure_exact and target_measure_exact,
        source_measure_error_bound=source_error,
        target_measure_error_bound=target_error,
        coverage_defect=max(
            0.0, abs(target_measure - source_measure) - source_error - target_error
        )
        / max(abs(source_measure), np.finfo(np.float64).tiny),
    )


def _publish_mixed_nested_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    policy: transition_types.CellGeometryTransitionPolicy,
    prepared: _MixedGeometryCells,
    *,
    starting_work_units: int,
    budget: CoordinateEnclosureBudget,
) -> transition_types.CellGeometryTransition:
    source_coordinates = prepared.source_coordinates
    target_ids, parents, reference = (
        prepared.target_ids,
        prepared.parents,
        prepared.reference,
    )
    elements, routes = prepared.elements, prepared.routes
    origin_cells, origin_vertices = prepared.origin_cells, prepared.origin_vertices
    owner, vertex_values = prepared.owner, prepared.vertex_values
    continuity, containment = prepared.continuity, prepared.containment
    coarsening = prepared.coarsening
    origin = source_geometry.restriction_source
    dimension = target_mesh.topological_dimension
    slack = policy.continuity_tolerance * _extent(source_coordinates)
    if continuity > slack or containment > 64 * _EPSILON:
        raise transition_types.CellGeometryTransitionError(
            "discontinuous_source" if continuity > slack else "coverage",
            "nested_geometry.continuity_and_containment must fit shared-node and parent-domain bounds; got excess continuity or containment defect",
            measured=max(continuity, containment),
            limit=max(slack, 64 * _EPSILON),
        )
    used = np.asarray(sorted(owner), dtype=np.int64)
    remap = np.full(source_coordinates.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    restriction_source = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry)
        if origin is None
        else origin.source_geometry_id,
        source_mesh.topology_id if origin is None else origin.source_topology_id,
        origin_cells,
        origin_vertices,
    )
    periodic_source = (
        None
        if source_geometry.periodic_source is None
        else source_geometry.periodic_source.reindexed(used)
    )
    exact_source = source_geometry.exact_source
    if isinstance(exact_source, ExactPlcCellGeometrySource):
        ratios = exact_source.vertex_parameter_ratios
        exact_source = exact_source.with_witnesses(
            np.asarray(exact_source.vertex_strata)[used],
            np.asarray(exact_source.vertex_rows)[used],
            np.asarray(exact_source.vertex_parameters)[used],
            vertex_parameter_ratios=(
                None if ratios is None else tuple(ratios[int(index)] for index in used)
            ),
            budget=budget,
        )
    target_routes = {name: remap[route] for name, route in routes.items()}
    if isinstance(
        exact_source,
        (ExactRationalCellGeometrySource, ExactAlgebraicCellGeometrySource),
    ):
        exact_source = exact_source.reindexed(
            used, elements, target_routes, budget=budget
        )
    geometry = CellGeometrySpec(
        elements,
        target_routes,
        source_coordinates[used],
        restriction_source=restriction_source,
        exact_source=exact_source,
        periodic_source=periodic_source,
        budget=budget,
    )
    vertex_coordinates = jnp.asarray(
        np.stack(
            [vertex_values[index] for index in range(target_mesh.coordinates.shape[0])]
        )
    )
    target_mesh = target_mesh.with_coordinates(
        vertex_coordinates, numeric_version=target_mesh.numeric_version
    )
    evidence = _mixed_transition_evidence(
        source_mesh,
        source_geometry,
        target_mesh,
        geometry,
        prepared,
        policy,
        slack,
        used.size,
        starting_work_units=starting_work_units,
        budget=budget,
    )
    empty = np.zeros(0, dtype=np.int64)
    coarse_witness = (
        transition_types.NestedReferenceWitnesses(
            empty, empty, np.zeros((0, 8, dimension), dtype=np.float64)
        )
        if coarsening is None
        else coarsening
    )
    _admit_transition_banks(
        prepared.homogeneous,
        coarse_witness.fine_reference_homogeneous,
        budget=budget,
    )
    source_id, target_id = cell_geometry_id(source_geometry), cell_geometry_id(geometry)
    stored_targets = jnp.asarray(target_ids)
    stored_parents = jnp.asarray(parents)
    stored_reference = jnp.asarray(reference)
    stored_coarsened = jnp.asarray(coarse_witness.fine_cell_ids)
    stored_into = jnp.asarray(coarse_witness.coarse_cell_ids)
    stored_coarse_reference = jnp.asarray(coarse_witness.fine_reference_vertices)
    parent_homogeneous = (
        None if prepared.homogeneous is None else jnp.asarray(prepared.homogeneous)
    )
    coarsened_homogeneous = (
        None
        if coarse_witness.fine_reference_homogeneous is None
        else jnp.asarray(coarse_witness.fine_reference_homogeneous)
    )
    transition_id = (
        canonical_fingerprint(
            {
                "kind": "mixed-coordinate-restriction",
                "source": source_id,
                "target": target_id,
                "source_topology": source_mesh.topology_id,
                "target_topology": target_mesh.topology_id,
                "policy": policy.policy_id,
                "evidence": evidence.evidence_id,
            }
        )
        if parent_homogeneous is None and coarsened_homogeneous is None
        else transition_types.exact_root_transition_id(
            source_topology_id=source_mesh.topology_id,
            target_topology_id=target_mesh.topology_id,
            source_geometry_id=source_id,
            target_geometry_id=target_id,
            target_cell_ids=stored_targets,
            parent_cell_ids=stored_parents,
            parent_reference_vertices=stored_reference,
            parent_reference_homogeneous=parent_homogeneous,
            coarsened_cell_ids=stored_coarsened,
            coarsened_into_ids=stored_into,
            coarsened_reference_vertices=stored_coarse_reference,
            coarsened_reference_homogeneous=coarsened_homogeneous,
            evidence_id=evidence.evidence_id,
            policy_id=policy.policy_id,
        )
    )
    return transition_types.CellGeometryTransition(
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        source_geometry_id=source_id,
        target_geometry_id=target_id,
        geometry=geometry,
        vertex_coordinates=vertex_coordinates,
        target_cell_ids=stored_targets,
        parent_cell_ids=stored_parents,
        parent_reference_vertices=stored_reference,
        coarsened_cell_ids=stored_coarsened,
        coarsened_into_ids=stored_into,
        coarsened_reference_vertices=stored_coarse_reference,
        node_owner_cells=jnp.asarray(
            [owner[int(node)][0] for node in used], dtype=jnp.int64
        ),
        node_owner_locals=jnp.asarray(
            [owner[int(node)][1] for node in used], dtype=jnp.int32
        ),
        evidence=evidence,
        policy_id=policy.policy_id,
        parent_reference_homogeneous=parent_homogeneous,
        coarsened_reference_homogeneous=coarsened_homogeneous,
        transition_id=transition_id,
    )


type _MixedReference = tuple[int, np.ndarray, np.ndarray | None]


def _validated_mixed_witness(
    witnesses: transition_types.NestedReferenceWitnesses | None,
    name: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> transition_types.NestedReferenceWitnesses | None:
    if witnesses is None:
        return None
    vertices = np.asarray(witnesses.fine_reference_vertices, dtype=np.float64)
    return transition_types.NestedReferenceWitnesses(
        np.asarray(witnesses.fine_cell_ids, dtype=np.int64),
        np.asarray(witnesses.coarse_cell_ids, dtype=np.int64),
        vertices,
        transition_types.require_reference_homogeneous(
            vertices, witnesses.fine_reference_homogeneous, name, budget=budget
        ),
    )


def _mixed_nested_dispositions(
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    refinement: transition_types.NestedReferenceWitnesses | None,
    coarsening: transition_types.NestedReferenceWitnesses | None,
) -> tuple[
    dict[int, tuple[int, int]],
    dict[int, _MixedReference],
    dict[int, list[_MixedReference]],
    np.ndarray,
]:
    """Index source rows and require exactly one disposition per target cell.

    Each reference record keeps its validated unrounded homogeneous row, if any.
    """
    source_index = {
        int(identifier): (index, row)
        for index, block in enumerate(source_mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids))
    }
    refined: dict[int, _MixedReference] = {}
    if refinement is not None:
        bank = refinement.fine_reference_homogeneous
        for row, (fine, coarse, reference) in enumerate(
            zip(
                refinement.fine_cell_ids,
                refinement.coarse_cell_ids,
                refinement.fine_reference_vertices,
                strict=True,
            )
        ):
            refined[int(fine)] = (
                int(coarse),
                np.asarray(reference),
                None if bank is None else bank[row],
            )
    coarsened: dict[int, list[_MixedReference]] = {}
    if coarsening is not None:
        bank = coarsening.fine_reference_homogeneous
        for row, (fine, coarse, reference) in enumerate(
            zip(
                coarsening.fine_cell_ids,
                coarsening.coarse_cell_ids,
                coarsening.fine_reference_vertices,
                strict=True,
            )
        ):
            coarsened.setdefault(int(coarse), []).append(
                (int(fine), np.asarray(reference), None if bank is None else bank[row])
            )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    if set(refined) & set(coarsened) or set(refined) | set(coarsened) != set(target_ids):
        raise ValueError(
            "nested_geometry.witnesses must give each mixed target cell one nested disposition; got missing or overlapping disposition"
        )
    return source_index, refined, coarsened, target_ids


def _retain_mixed_corner_images(
    row: np.ndarray,
    images: tuple[tuple[imported_coordinate_enclosure.ExactScalar, ...], ...],
    exact_vertices: dict[int, tuple[imported_coordinate_enclosure.ExactScalar, ...]],
    vertex_values: dict[int, np.ndarray],
    continuity: float,
) -> float:
    """Keep first-owner rounded corners while bounding rational shared-image defects."""
    for vertex, image in zip(row.tolist(), images, strict=True):
        if vertex in exact_vertices:
            continuity = max(
                continuity,
                float(
                    max(
                        abs(first - second)
                        for first, second in zip(
                            exact_vertices[vertex], image, strict=True
                        )
                    )
                ),
            )
        else:
            exact_vertices[vertex] = image
            vertex_values[vertex] = rounded_point(image)
    return continuity


def _transition_mixed_nested_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    *,
    refinement: transition_types.NestedReferenceWitnesses | None,
    coarsening: transition_types.NestedReferenceWitnesses | None,
    policy: transition_types.CellGeometryTransitionPolicy,
    budget: CoordinateEnclosureBudget,
) -> transition_types.CellGeometryTransition:
    """Compose coefficient bases exactly, restoring full recorded siblings.

    Child nodal interpolation is not used. Each target block owns one affine
    reference action. Rational pyramid maps stay rational restricted maps; their
    validity must be accepted separately by the geometry certification owner.
    """

    starting_work_units = budget.work_units
    source_elements, source_routes, source_coordinates_ = source_geometry.resolve(
        source_mesh, budget=budget
    )
    source_routes = tuple(np.asarray(route, dtype=np.int64) for route in source_routes)
    if any(
        not isinstance(
            element,
            (
                reference_types_owner.FiniteElementSpec,
                BarycentricCellGeometryElement,
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
                SplineCellGeometryElement,
                LayerColumnCellGeometryElement,
            ),
        )
        for element in source_elements
    ):
        raise ValueError(
            "nested_geometry.elements must provide scalar coordinate maps; got unsupported source element"
        )
    source_coordinates = np.asarray(source_coordinates_, dtype=np.float64)
    refinement = _validated_mixed_witness(
        refinement, "nested_geometry.refinement", budget=budget
    )
    coarsening = _validated_mixed_witness(
        coarsening, "nested_geometry.coarsening", budget=budget
    )
    source_index, refined, coarsened, target_ids = _mixed_nested_dispositions(
        source_mesh, target_mesh, refinement, coarsening
    )
    ledger = budget
    with transition_budget_owner.charged_geometry_transition(policy, budget=ledger):
        source_coefficients = source_geometry.source_coordinates(budget=budget)
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    ancestry = _mixed_parent_identity(source_mesh, source_geometry, source_elements)
    source_parent_cells, source_parent_vertices = ancestry.cells, ancestry.vertices
    origin_cells: dict[str, NDArray[np.int64]] = {}
    origin_vertices: dict[str, NDArray[np.int64]] = {}
    parents = np.full(target_ids.size, -1, dtype=np.int64)
    dimension = target_mesh.topological_dimension
    reference = np.zeros((target_ids.size, 8, dimension), dtype=np.float64)
    homogeneous = (
        None
        if refinement is None or refinement.fine_reference_homogeneous is None
        else np.zeros((target_ids.size, 8, dimension + 1), dtype=np.float64)
    )
    # Published carrier corners are the correctly rounded binary64 images of the
    # algebraic restricted maps; the maps, not these floats, are the geometry.
    exact_vertices: dict[int, tuple[imported_coordinate_enclosure.ExactScalar, ...]] = {}
    vertex_values: dict[int, np.ndarray] = {}
    continuity = containment = 0.0
    row_index = 0
    owner: dict[int, tuple[int, int]] = {}
    reference_actions: dict[
        tuple[str, tuple[int, ...], bytes], tuple[np.ndarray, np.ndarray]
    ] = {}
    restricted_elements: dict[
        tuple[int, tuple[str, tuple[int, ...], bytes]], CellGeometryElement
    ] = {}
    for block in target_mesh.blocks:
        block_elements, block_routes = [], []
        block_origins, block_origin_vertices = [], []
        vertices = np.asarray(
            reference_cell_topology(block.cell_kind).vertices, dtype=np.float64
        )
        arity = vertices.shape[0]
        for identifier_, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            identifier = int(identifier_)
            if identifier in refined:
                parent, corners, bank = refined[identifier]
                index, parent_row = source_index[parent]
                source_kind = source_mesh.blocks[index].cell_kind
                exact_rows = _nondyadic_homogeneous_rows(corners[:arity], bank)
                if exact_rows.any():
                    # Source-owned roots: the rational P1 coefficient action owns
                    # the child map; rounded corners are execution views only.
                    assert bank is not None
                    element_key = (
                        index,
                        (block.cell_kind, bank.shape, bank[:arity].tobytes()),
                    )
                    element = restricted_elements.get(element_key)
                    if element is None:
                        element = _homogeneous_child_element(
                            source_elements[index],
                            block.cell_kind,
                            source_kind,
                            transition_types.exact_reference_corners(
                                corners[:arity], bank[:arity]
                            ),
                            bank[:arity],
                            budget=budget,
                        )
                        restricted_elements[element_key] = element
                else:
                    action_key = (block.cell_kind, corners.shape, corners.tobytes())
                    action = reference_actions.get(action_key)
                    if action is None:
                        action = _reference_affine(block.cell_kind, corners)
                        reference_actions[action_key] = action
                    matrix, offset = action
                    element_key = (index, action_key)
                    element = restricted_elements.get(element_key)
                    if element is None:
                        element = _restricted_element(
                            source_elements[index], block.cell_kind, matrix, offset
                        )
                        restricted_elements[element_key] = element
                route = source_routes[index][parent_row]
                ancestry_row = parent_row
                parents[row_index] = parent
                reference[row_index, : corners.shape[0]] = corners
                if homogeneous is not None and bank is not None:
                    homogeneous[row_index, : bank.shape[0]] = bank
                # Nonnegative homogeneous corners lie in the parent exactly.
                points = corners[:arity][~exact_rows]
                defect = _reference_containment_defect(source_kind, points)
                containment = max(containment, defect)
            else:
                restored = _restore_coordinate_siblings(
                    coarsened[identifier],
                    block.cell_kind,
                    source_mesh,
                    source_elements,
                    source_routes,
                    source_index,
                    source_coordinates,
                    source_parent_cells,
                    policy,
                    budget=budget,
                )
                element, route = restored.element, restored.route
                index, ancestry_row = restored.source_block, restored.ancestry_row
            block_elements.append(element)
            block_routes.append(route)
            ancestry_name = source_mesh.blocks[index].name
            block_origins.append(source_parent_cells[ancestry_name][ancestry_row])
            ordered_vertices = np.full(8, -1, dtype=np.int64)
            corners_ = source_parent_vertices[ancestry_name][ancestry_row]
            ordered_vertices[: corners_.size] = corners_
            block_origin_vertices.append(ordered_vertices)
            for slot, node in enumerate(route):
                owner.setdefault(int(node), (identifier, slot))
            with transition_budget_owner.charged_geometry_transition(
                policy, budget=ledger
            ):
                images = coordinate_corner_images(
                    element,
                    tuple(source_coefficients[index] for index in route),
                    budget=budget,
                )
                if images is None:
                    raise ValueError(
                        "nested_geometry.source_expressions must provide coordinate expressions; got absent source expressions"
                    )
            continuity = _retain_mixed_corner_images(
                row, images, exact_vertices, vertex_values, continuity
            )
            row_index += 1
        element = block_elements[0]
        if any(other.element_id != element.element_id for other in block_elements[1:]):
            raise ValueError(
                "nested_geometry.block_elements must agree within each target block; got different restricted element IDs"
            )
        elements[block.name] = element
        routes[block.name] = np.stack(block_routes)
        origin_cells[block.name] = np.asarray(block_origins, dtype=np.int64)
        origin_vertices[block.name] = np.stack(block_origin_vertices)
    prepared = _MixedGeometryCells(
        source_coordinates,
        target_ids,
        parents,
        reference,
        elements,
        routes,
        origin_cells,
        origin_vertices,
        owner,
        vertex_values,
        continuity,
        containment,
        coarsening,
        homogeneous,
    )
    return _publish_mixed_nested_geometry(
        source_mesh,
        source_geometry,
        target_mesh,
        policy,
        prepared,
        starting_work_units=starting_work_units,
        budget=budget,
    )


__all__ = [
    "transition_chart_deformed_cell_geometry",
    "coordinate_element_lebesgue_bound",
    "reconstruct_parametric_surface_cell_geometry",
    "cell_geometry_vertex_measures",
    "is_affine_cell_geometry",
    "nested_edge_midpoints_admitted",
    "nested_geometry_degree",
    "transition_displaced_cell_geometry",
    "transition_nested_cell_geometry",
]


@checked
def coordinate_element_lebesgue_bound(
    element: reference_types_owner.FiniteElementSpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Bound sum|phi_i| by common-Bernstein absolute column sums."""

    if element.cell_kind != "triangle" or element.conformity != "H1":
        raise ValueError(
            "surface_reconstruction.element must be a scalar H1 triangle; got incompatible reference element"
        )
    basis = source_basis(element, budget=budget)
    if basis is None or any(
        max((sum(index) for index in polynomial), default=0) != element.degree
        for polynomial in basis
    ):
        raise ValueError(
            "surface_reconstruction.basis must have one common Bernstein degree; got incompatible basis degrees"
        )
    coefficients = tuple(
        imported_coordinate_enclosure.bernstein_coefficients(
            polynomial, "simplex", 2, budget=budget
        )
        for polynomial in basis
    )
    bound = max(
        sum(abs(value) for value in column) for column in zip(*coefficients, strict=True)
    )
    return fraction_endpoint(bound, upper=True, saturate=True)


def _surface_affine_defect(
    element: reference_types_owner.FiniteElementSpec,
    corners: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """algebraic enclosure of L-I_pL for encoded basis and affine corner chord L."""

    basis = source_basis(element, budget=budget)
    if basis is None:
        raise ValueError(
            "surface_reconstruction.basis must provide a source-polynomial enclosure; got missing basis enclosure"
        )
    variables = imported_coordinate_enclosure.axes(2, budget=budget)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    bound = Fraction(0)
    for component in range(3):
        values = rational_vector(corners[:, component])
        affine = add(
            imported_coordinate_enclosure.constant(values[0], 2, budget=budget),
            add(
                scale(variables[0], values[1] - values[0], budget=budget),
                scale(variables[1], values[2] - values[0], budget=budget),
                budget=budget,
            ),
            budget=budget,
        )
        interpolated = imported_coordinate_enclosure.constant(0, 2, budget=budget)
        for polynomial, node in zip(basis, nodes, strict=True):
            node_ = np.asarray(node, dtype=np.float64)
            value = (
                values[0]
                + Fraction(float(node_[0])) * (values[1] - values[0])
                + Fraction(float(node_[1])) * (values[2] - values[0])
            )
            interpolated = add(
                interpolated, scale(polynomial, value, budget=budget), budget=budget
            )
        difference = add(affine, scale(interpolated, -1, budget=budget), budget=budget)
        bound += max(
            abs(value)
            for value in imported_coordinate_enclosure.bernstein_coefficients(
                difference, "simplex", 2, budget=budget
            )
        )
    return coefficient_endpoint(bound, upper=True, saturate=True)


@checked
def surface_node_enclosures(
    domain: imported_geometry__meshing_domain.MeshingDomain,
    patch: int,
    charts: np.ndarray,
    nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Host certificate queries enclosing algebraic affine-chart node parameters."""

    lower = np.empty((charts.shape[0], nodes.shape[0], 3), dtype=np.float64)
    upper = np.empty_like(lower)
    weights = tuple(
        (
            Fraction(1) - Fraction(float(node[0])) - Fraction(float(node[1])),
            Fraction(float(node[0])),
            Fraction(float(node[1])),
        )
        for node in nodes
    )
    surface = domain.patches[patch].surface
    for cell, corners in enumerate(charts):
        exact_corners = tuple(rational_vector(row) for row in corners)
        for slot, barycentric in enumerate(weights):
            parameters = tuple(
                sum(
                    (
                        weight * corner[axis]
                        for weight, corner in zip(barycentric, exact_corners, strict=True)
                    ),
                    Fraction(0),
                )
                for axis in range(2)
            )
            box = np.asarray(
                [
                    [
                        fraction_endpoint(value, upper=False, saturate=True)
                        for value in parameters
                    ],
                    [
                        fraction_endpoint(value, upper=True, saturate=True)
                        for value in parameters
                    ],
                ],
                dtype=np.float64,
            )
            enclosure = np.asarray(surface.bounding_box(box), dtype=np.float64)
            if (
                enclosure.shape != (2, 3)
                or not np.all(np.isfinite(enclosure))
                or np.any(enclosure[1] < enclosure[0])
            ):
                raise ValueError(
                    "surface_reconstruction.node_enclosure must be finite; got missing or nonfinite patch enclosure"
                )
            lower[cell, slot], upper[cell, slot] = enclosure
    return lower, upper


def _surface_reconstruction_bounds(
    blocks: tuple[SimplexGeometryBlock, ...],
    placed: PlacedGeometryNodes,
    enclosures: list[tuple[np.ndarray, np.ndarray]],
    chord_bounds: np.ndarray,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    total, lebesgue, affine, errors = [], [], [], []
    offset = 0
    for block, (lower, upper) in zip(blocks, enclosures, strict=True):
        constant_ = coordinate_element_lebesgue_bound(block.element, budget=budget)
        nodes = placed.coordinates[block.routes]
        vertex_dofs = [entity[0] for entity in block.element.entity_dofs[0]]
        for row in range(block.cell_ids.size):
            point_errors = tuple(
                sum(
                    (
                        max(
                            abs(Fraction(float(value)) - Fraction(float(low))),
                            abs(Fraction(float(value)) - Fraction(float(high))),
                        )
                        for value, low, high in zip(
                            point, low_point, high_point, strict=True
                        )
                    ),
                    Fraction(0),
                )
                for point, low_point, high_point in zip(
                    nodes[row], lower[row], upper[row], strict=True
                )
            )
            node_error = max(point_errors)
            corner_error = max(point_errors[index] for index in vertex_dofs)
            defect = _surface_affine_defect(
                block.element, nodes[row, vertex_dofs], budget=budget
            )
            bound = (
                (Fraction(1) + Fraction(constant_))
                * (Fraction(float(chord_bounds[offset + row])) + corner_error)
                + Fraction(defect)
                + Fraction(constant_) * node_error
            )
            total.append(fraction_endpoint(bound, upper=True, saturate=True))
            lebesgue.append(constant_)
            affine.append(defect)
            errors.append(fraction_endpoint(node_error, upper=True, saturate=True))
        offset += block.cell_ids.size
    return (
        np.asarray(total, dtype=np.float64),
        np.asarray(lebesgue, dtype=np.float64),
        np.asarray(affine, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def _require_affine_surface_chart_continuity(
    mesh: CellMesh,
    elements: tuple[CellGeometryElement, ...],
) -> bool:
    """Check linear triangle chart continuity when that representation applies."""
    charts: list[PolynomialComposedCellGeometryElement] = []
    for element in elements:
        match element:
            case PolynomialComposedCellGeometryElement() if (
                element.chart_element.cell_kind == "triangle"
                and element.chart_element.degree == 1
            ):
                charts.append(element)
            case _:
                return False
    if charts:
        vertex_charts: dict[int, tuple[Fraction, Fraction]] = {}
        vertex_ids = np.asarray(mesh.vertex_global_ids)
        for block, element in zip(mesh.blocks, charts, strict=True):
            corner_dofs = tuple(
                entity[0] for entity in element.chart_element.entity_dofs[0]
            )
            corners: tuple[tuple[Fraction, Fraction], ...] = tuple(
                (
                    Fraction(*element.chart_coefficients[dof][0]),
                    Fraction(*element.chart_coefficients[dof][1]),
                )
                for dof in corner_dofs
            )
            for vertices in np.asarray(block.vertices):
                for identifier, chart in zip(vertex_ids[vertices], corners, strict=True):
                    previous = vertex_charts.setdefault(int(identifier), chart)
                    if previous != chart:
                        raise transition_types.CellGeometryTransitionError(
                            "discontinuous_source",
                            "surface_reconstruction.chart_continuity must agree at shared chart vertices; got different rational chart coordinates",
                            measured=math.inf,
                            limit=0.0,
                        )
        return True
    return True


def _require_equal_surface_traces(
    previous: tuple[Expression, ...],
    trace: tuple[Expression, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    for left, right in zip(previous, trace, strict=True):
        numerator, _ = imported_coordinate_enclosure.expression_parts(
            expression_add(
                left,
                expression_scale(right, -1, budget=budget),
                budget=budget,
            ),
            1,
            budget=budget,
        )
        if numerator:
            raise transition_types.CellGeometryTransitionError(
                "discontinuous_source",
                "surface_reconstruction.trace_continuity must agree over shared edge traces; got nonzero expression difference",
                measured=math.inf,
                limit=0.0,
            )


def _require_retained_surface_continuity(
    mesh: CellMesh, geometry: CellGeometrySpec, *, budget: CoordinateEnclosureBudget
) -> None:
    """Authenticate whole shared edge traces, not only shared corner samples."""

    elements, routes, _ = geometry.resolve(mesh)
    if _require_affine_surface_chart_continuity(mesh, elements):
        return
    controls = geometry.source_coordinates(budget=budget)
    variable = imported_coordinate_enclosure.axes(1, budget=budget)[0]
    references = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    traces: dict[tuple[int, int], tuple[Expression, ...]] = {}
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for vertices, dofs in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            values = coordinate_expressions(
                element,
                tuple(controls[int(index)] for index in dofs),
                budget=budget,
            )
            if values is None:
                raise ValueError(
                    "surface_reconstruction.source_expressions must provide the retained coordinate law; got absent source expressions"
                )
            ids = np.asarray(mesh.vertex_global_ids)[vertices]
            for first, last in ((0, 1), (1, 2), (2, 0)):
                if ids[first] > ids[last]:
                    first, last = last, first
                key = int(ids[first]), int(ids[last])
                arguments = tuple(
                    add(
                        imported_coordinate_enclosure.constant(
                            references[first][axis], 1, budget=budget
                        ),
                        scale(
                            variable,
                            references[last][axis] - references[first][axis],
                            budget=budget,
                        ),
                        budget=budget,
                    )
                    for axis in range(2)
                )
                trace = tuple(
                    imported_coordinate_enclosure.expression_compose(
                        value, arguments, budget=budget
                    )
                    for value in values
                )
                previous = traces.get(key)
                if previous is not None:
                    _require_equal_surface_traces(previous, trace, budget=budget)
                else:
                    traces[key] = trace


def _reconstruct_retained_surface_source(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    atlas: imported_geometry__surface_source_support.SurfaceSourceRootAtlas,
    cell_ids: np.ndarray,
    cell_patches: np.ndarray,
    cell_charts: np.ndarray,
    entities: tuple[str, ...],
    occurrences: tuple[tuple[str, ...], ...],
    policy: transition_types.CellGeometryTransitionPolicy,
    *,
    budget: CoordinateEnclosureBudget,
) -> chart_reconstruction_owner.SurfaceGeometryReconstruction:
    """Continuous equality by source source-expression composition, not samples.

    The target map is S composed with the declared affine UV triangle. source
    knot/weight/control banks survive unchanged. Validity, embedding and the
    old-to-new continuous difference remain the chart-deformation owner's gates.
    """

    if atlas.domain.domain_id != domain.domain_id:
        raise ValueError(
            "surface_reconstruction.source_atlas must bind the supplied source domain; got different domain ID"
        )
    atlas.require_current()
    target_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in target_mesh.blocks]
    )
    ids, patches, charts = (
        np.asarray(cell_ids),
        np.asarray(cell_patches),
        np.asarray(cell_charts, dtype=np.float64),
    )
    if (
        ids.shape != target_ids.shape
        or not np.issubdtype(ids.dtype, np.integer)
        or np.unique(ids).size != ids.size
        or not np.array_equal(np.sort(ids), np.sort(target_ids))
    ):
        raise ValueError(
            "surface_reconstruction.cell_ids must name every target cell once; got missing or repeated cell ID"
        )
    if (
        patches.shape != ids.shape
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any((patches < 0) | (patches >= len(domain.patches)))
        or charts.shape != (ids.size, 3, 2)
        or not np.all(np.isfinite(charts))
    ):
        raise ValueError(
            "surface_reconstruction.charts must align finite UV triangles with present patches; got incompatible chart array"
        )
    if (
        len(entities) != ids.size
        or len(occurrences) != ids.size
        or any(
            entity != domain.entity_id(2, int(patch))
            or occurrence != domain.source_occurrences[2][int(patch)]
            for entity, occurrence, patch in zip(
                entities, occurrences, patches, strict=True
            )
        )
    ):
        raise ValueError(
            "surface_reconstruction.entities must retain source entities and occurrences; got changed entity or occurrence"
        )
    rows = {int(identifier): row for row, identifier in enumerate(ids)}
    order = np.asarray(
        [rows[int(identifier)] for identifier in target_ids], dtype=np.int64
    )
    patches, charts = patches[order], charts[order]

    ledger = budget

    starting_work = ledger.work_units
    with transition_budget_owner.charged_geometry_transition(policy, budget=ledger):
        ledger.reserve(12 * ids.size)
        mesh, geometry = (
            imported_geometry__surface_source_support.restrict_surface_source_atlas_geometry(
                atlas,
                target_mesh,
                patches,
                charts,
                maximum_support_queries=policy.maximum_evaluations,
                canonical_blocks=True,
                budget=budget,
            )
        )
        _require_retained_surface_continuity(mesh, geometry, budget=budget)
    work = ledger.work_units - starting_work

    zeros = jnp.zeros((ids.size,), dtype=jnp.float64)
    # There are no reconstructed nodal coefficients: these are the source
    # scientific control axes, and therefore no interpolation-node owner exists.
    identity = canonical_fingerprint(
        {
            "kind": "retained-rational-surface-reconstruction",
            "atlas": atlas.atlas_id,
            "source_geometry": cell_geometry_id(source_geometry),
            "source_topology": source_mesh.topology_id,
            "target_geometry": cell_geometry_id(geometry),
            "target_topology": mesh.topology_id,
            "cells": array_tree_fingerprint(target_ids),
            "patches": array_tree_fingerprint(patches),
            "charts": array_tree_fingerprint(charts),
            "policy": policy.policy_id,
        }
    )
    return chart_reconstruction_owner.SurfaceGeometryReconstruction(
        source_geometry_id=cell_geometry_id(source_geometry),
        source_topology_id=source_mesh.topology_id,
        target_topology_id=mesh.topology_id,
        domain_id=domain.domain_id,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        cell_geometry_entity_ids=tuple(entities[row] for row in order),
        cell_occurrence_paths=tuple(occurrences[row] for row in order),
        geometry=geometry,
        target_mesh=mesh,
        vertex_coordinates=mesh.coordinates,
        cell_global_ids=jnp.asarray(target_ids),
        cell_patches=jnp.asarray(patches, dtype=jnp.int32),
        cell_charts=jnp.asarray(charts),
        fidelity_bounds=zeros,
        lebesgue_bounds=zeros,
        affine_reproduction_bounds=zeros,
        node_enclosure_bounds=zeros,
        node_owner_cells=None,
        node_owner_locals=None,
        continuity_residual=0.0,
        evaluation_count=work,
        reconstruction_id=identity,
    )


class _SurfaceReconstructionLayout(StrictModule):
    blocks: tuple[SimplexGeometryBlock, ...]
    target_ids: NDArray[np.int64]
    order: NDArray[np.int64]
    patches: NDArray[np.int32]
    charts: NDArray[np.float64]


def _surface_reconstruction_layout(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    cell_ids: NDArray[np.int64],
    cell_patches: NDArray[np.int64],
    cell_charts: NDArray[np.float64],
    cell_geometry_entity_ids: tuple[str, ...],
    cell_occurrence_paths: tuple[tuple[str, ...], ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _SurfaceReconstructionLayout:
    source_elements, _, _ = source_geometry.resolve(source_mesh, budget=budget)
    degrees = {
        require_scalar_coordinate_element(element, "Surface reconstruction").degree
        for element in source_elements
    }
    if len(degrees) != 1:
        raise ValueError(
            "surface_reconstruction.degree must be common to source blocks; got different coordinate degrees"
        )
    blocks = simplex_blocks(target_mesh, target_layout, "target", budget=budget)
    if any(
        block.element.cell_kind != "triangle" or block.degree not in degrees
        for block in blocks
    ):
        raise ValueError(
            "surface_reconstruction.degree must retain the source triangle coordinate degree; got incompatible target element"
        )
    ids, patches = np.asarray(cell_ids), np.asarray(cell_patches)
    charts = np.asarray(cell_charts, dtype=np.float64)
    target_ids = np.concatenate([block.cell_ids for block in blocks])
    if (
        ids.shape != target_ids.shape
        or not np.issubdtype(ids.dtype, np.integer)
        or np.unique(ids).size != ids.size
        or not np.array_equal(np.sort(ids), np.sort(target_ids))
        or patches.shape != ids.shape
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any((patches < 0) | (patches >= len(domain.patches)))
        or charts.shape != (ids.size, 3, 2)
        or not np.all(np.isfinite(charts))
    ):
        raise ValueError(
            "surface_reconstruction.charts must align cell IDs, patches, and finite charts; got incompatible arrays"
        )
    if (
        len(cell_geometry_entity_ids) != ids.size
        or len(cell_occurrence_paths) != ids.size
        or any(
            entity != domain.entity_id(2, int(patch))
            or occurrence != domain.source_occurrences[2][int(patch)]
            for entity, occurrence, patch in zip(
                cell_geometry_entity_ids, cell_occurrence_paths, patches, strict=True
            )
        )
    ):
        raise ValueError(
            "surface_reconstruction.entities must retain geometry entities and occurrences; got different entity or occurrence"
        )
    indices = {int(identifier): row for row, identifier in enumerate(ids)}
    order = np.asarray(
        [indices[int(identifier)] for identifier in target_ids], dtype=np.int64
    )
    patches, charts = patches[order].astype(np.int32), charts[order]

    return _SurfaceReconstructionLayout(blocks, target_ids, order, patches, charts)


class _SurfaceReconstructionSamples(StrictModule):
    values: list[NDArray[np.float64]]
    enclosures: list[tuple[NDArray[np.float64], NDArray[np.float64]]]
    chord: NDArray[np.float64]


def _surface_reconstruction_samples(
    domain: imported_geometry__meshing_domain.MeshingDomain,
    layout: _SurfaceReconstructionLayout,
) -> _SurfaceReconstructionSamples:
    blocks, target_ids, patches, charts = (
        layout.blocks,
        layout.target_ids,
        layout.patches,
        layout.charts,
    )
    values, enclosures = [], []
    chord = np.zeros(target_ids.size, dtype=np.float64)
    first = 0
    for block in blocks:
        stop = first + block.cell_ids.size
        nodes = np.asarray(block.element.reference_nodes, dtype=np.float64)
        parameters = np.asarray(
            contract("nk,ckd->cnd", _barycentric(nodes), charts[first:stop])
        )
        values.append(
            domain.evaluate(
                np.repeat(patches[first:stop], nodes.shape[0]),
                parameters.reshape((-1, 2)),
            ).reshape((block.cell_ids.size, nodes.shape[0], 3))
        )
        low, high = np.empty_like(values[-1]), np.empty_like(values[-1])
        for patch in np.unique(patches[first:stop]):
            rows = np.flatnonzero(patches[first:stop] == patch)
            low[rows], high[rows] = surface_node_enclosures(
                domain, int(patch), charts[first:stop][rows], nodes
            )
            chord[first + rows] = domain.interpolation_bounds(
                int(patch), charts[first:stop][rows]
            )
        enclosures.append((low, high))
        first = stop
    if not np.all(np.isfinite(chord)):
        raise ValueError(
            "surface_reconstruction.interpolation_bound must be finite; got nonfinite source bound"
        )

    return _SurfaceReconstructionSamples(values, enclosures, chord)


@checked
def reconstruct_parametric_surface_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec | None,
    domain: imported_geometry__meshing_domain.MeshingDomain,
    /,
    *,
    domain_id: str,
    cell_ids: np.ndarray,
    cell_patches: np.ndarray,
    cell_charts: np.ndarray,
    cell_geometry_entity_ids: tuple[str, ...],
    cell_occurrence_paths: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: transition_types.CellGeometryTransitionPolicy | None = None,
    source_atlas: imported_geometry__surface_source_support.SurfaceSourceRootAtlas
    | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> chart_reconstruction_owner.SurfaceGeometryReconstruction:
    """Prepare bounded degree-preserving target coordinates without old coverage claims."""

    if domain_id != domain.domain_id:
        raise ValueError(
            "surface_reconstruction.domain_id must name the supplied domain; got different domain ID"
        )
    if (
        source_mesh.topological_dimension != 2
        or target_mesh.topological_dimension != 2
        or source_mesh.ambient_dimension != 3
        or target_mesh.ambient_dimension != 3
    ):
        raise ValueError(
            "surface_reconstruction.dimension must use two-dimensional meshes in three-dimensional space; got incompatible mesh dimensions"
        )
    limit = nonnegative_finite_float(maximum_fidelity, "maximum_fidelity")
    policy_ = (
        transition_types.CellGeometryTransitionPolicy() if policy is None else policy
    )
    with (
        (
            nullcontext(budget)
            if budget is not None
            else owned_coordinate_preparation(
                policy_.maximum_evaluations,
                sys.maxsize,
                maximum_geometry_queries=policy_.maximum_evaluations,
            )
        ) as budget,
        transition_budget_owner.charged_geometry_transition(policy_, budget=budget),
    ):
        if source_atlas is not None:
            return _reconstruct_retained_surface_source(
                source_mesh,
                source_geometry,
                target_mesh,
                domain,
                source_atlas,
                cell_ids,
                cell_patches,
                cell_charts,
                cell_geometry_entity_ids,
                cell_occurrence_paths,
                policy_,
                budget=budget,
            )
        if any(
            isinstance(
                domain.patches[int(patch)].surface,
                imported_geometry_brep__patches.BSplineSurfacePatch,
            )
            for patch in np.asarray(cell_patches).reshape(-1)
            if 0 <= int(patch) < len(domain.patches)
        ):
            raise ValueError(
                "surface_reconstruction.source_atlas must retain the spline source map; got missing spline source atlas"
            )
        if target_layout is None:
            raise TypeError(
                "surface_reconstruction.target_layout must provide a coordinate layout; got None"
            )
        layout = _surface_reconstruction_layout(
            source_mesh,
            source_geometry,
            target_mesh,
            target_layout,
            domain,
            cell_ids,
            cell_patches,
            cell_charts,
            cell_geometry_entity_ids,
            cell_occurrence_paths,
            budget=budget,
        )
        blocks, target_ids, order = layout.blocks, layout.target_ids, layout.order
        patches, charts = layout.patches, layout.charts
        work = _budget(
            sum(
                block.routes.size * (block.element.local_dof_count + 6)
                for block in blocks
            ),
            policy_,
        )
        samples = _surface_reconstruction_samples(domain, layout)
        values, enclosures, chord = samples.values, samples.enclosures, samples.chord
        placed = place(blocks, values, np.asarray(target_layout.coordinates).shape[0])
        slack = policy_.continuity_tolerance * _extent(placed.coordinates)
        if placed.continuity > slack:
            raise transition_types.CellGeometryTransitionError(
                "discontinuous_source",
                "surface_reconstruction.continuity must agree at shared nodes; got excess chart discrepancy",
                measured=placed.continuity,
                limit=slack,
            )
        bound, lebesgue, affine, errors = _surface_reconstruction_bounds(
            blocks,
            placed,
            enclosures,
            chord,
            budget=budget,
        )
        if np.max(bound, initial=0.0) > limit:
            raise transition_types.CellGeometryTransitionError(
                "approximation_bound",
                "surface_reconstruction.fidelity must fit the continuous source-fidelity bound; got excess reconstruction error",
                measured=float(np.max(bound)),
                limit=limit,
            )
        elements, routes, _ = target_layout.resolve(target_mesh, budget=budget)
        names = tuple(block.name for block in target_mesh.blocks)
        geometry = CellGeometrySpec(
            dict(zip(names, elements, strict=True)),
            dict(zip(names, routes, strict=True)),
            placed.coordinates,
            budget=budget,
        )
        if target_mesh.periodic_topology is not None:
            geometry = geometry.with_periodic_source(target_mesh)
        # Publish the reconstructed nodal map's carrier before endpoint
        # certification. The staged edit still carries the controller's source
        # evaluations, which need not have the same binary64 evaluation path.
        vertex_coordinates = jnp.asarray(
            vertex_rows(target_mesh, blocks, placed.coordinates)
        )
        target_mesh = target_mesh.with_coordinates(
            vertex_coordinates, numeric_version=target_mesh.numeric_version
        )
        reconstruction_id = canonical_fingerprint(
            {
                "kind": "parametric-surface-geometry-reconstruction",
                "source_geometry": cell_geometry_id(source_geometry),
                "source_topology": source_mesh.topology_id,
                "target_geometry": cell_geometry_id(geometry),
                "target_topology": target_mesh.topology_id,
                "domain": domain.domain_id,
                "cells": array_tree_fingerprint(target_ids),
                "patches": array_tree_fingerprint(patches),
                "charts": array_tree_fingerprint(charts),
                "bounds": array_tree_fingerprint(bound),
                "limit": limit,
                "policy": policy_.policy_id,
            }
        )
        return chart_reconstruction_owner.SurfaceGeometryReconstruction(
            source_geometry_id=cell_geometry_id(source_geometry),
            source_topology_id=source_mesh.topology_id,
            target_topology_id=target_mesh.topology_id,
            domain_id=domain.domain_id,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
            cell_geometry_entity_ids=tuple(
                cell_geometry_entity_ids[index] for index in order
            ),
            cell_occurrence_paths=tuple(cell_occurrence_paths[index] for index in order),
            geometry=geometry,
            target_mesh=target_mesh,
            vertex_coordinates=vertex_coordinates,
            cell_global_ids=jnp.asarray(target_ids),
            cell_patches=jnp.asarray(patches),
            cell_charts=jnp.asarray(charts),
            fidelity_bounds=jnp.asarray(bound),
            lebesgue_bounds=jnp.asarray(lebesgue),
            affine_reproduction_bounds=jnp.asarray(affine),
            node_enclosure_bounds=jnp.asarray(errors),
            node_owner_cells=jnp.asarray(placed.owner_cells),
            node_owner_locals=jnp.asarray(placed.owner_locals),
            continuity_residual=placed.continuity,
            evaluation_count=work,
            reconstruction_id=reconstruction_id,
        )


@checked
def transition_chart_deformed_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: chart_reconstruction_owner.SurfaceGeometryReconstruction
    | sphere_chart_owner.SphereGeometryReconstruction,
    deformation: surface_chart_owner.PreparedSurfaceChartDeformation
    | sphere_chart_owner.PreparedSphereChartDeformation,
    /,
    *,
    policy: transition_types.CellGeometryTransitionPolicy,
    budget: CoordinateEnclosureBudget | None = None,
) -> transition_types.CellGeometryTransition:
    """Accept a declared bounded material chart deformation with full evidence.

    The physical surface measure may change. full native material coverage is
    retained in ``chart_deformation`` and bound by ``chart_correspondence_id``;
    neither equal physical areas nor a nested parent stencil is implied.
    """
    if policy.reconstruction != "bounded_chart_deformation":
        raise ValueError(
            "material_transition.policy must select bounded_chart_deformation; got different reconstruction policy"
        )
    match reconstruction, deformation:
        case (
            sphere_chart_owner.SphereGeometryReconstruction(),
            sphere_chart_owner.PreparedSphereChartDeformation(),
        ):
            if deformation.target_atlas.atlas_id != reconstruction.target_atlas.atlas_id:
                raise ValueError(
                    "material_transition.target_atlas must agree between reconstruction and correspondence; got different atlas IDs"
                )
            reconstruction.target_validity.require_bound(
                reconstruction.geometry, mesh=target_mesh
            )
            reconstruction.target_embedding.binding.require(
                target_mesh, reconstruction.geometry
            )
            evaluations = reconstruction.evaluation_count
        case (
            chart_reconstruction_owner.SurfaceGeometryReconstruction(),
            surface_chart_owner.PreparedSurfaceChartDeformation(),
        ):
            evaluations = reconstruction.evaluation_count + deformation.evaluation_count
        case sphere_chart_owner.SphereGeometryReconstruction(), _:
            raise TypeError(
                "material_transition.deformation must certify the radial reconstruction; got incompatible chart certificate"
            )
        case chart_reconstruction_owner.SurfaceGeometryReconstruction(), _:
            raise TypeError(
                "material_transition.deformation must certify the surface reconstruction; got incompatible chart certificate"
            )
        case invalid, _:
            assert_never(invalid)
    if isinstance(deformation, sphere_chart_owner.PreparedSphereChartDeformation):
        # The owning request's ledger reuses its producer's proof.
        deformation.require_bound(
            source_mesh,
            source_geometry,
            target_mesh,
            reconstruction.geometry,
            budget=budget,
        )
    else:
        deformation.require_bound(
            source_mesh, source_geometry, target_mesh, reconstruction.geometry
        )
    if (
        reconstruction.source_geometry_id,
        reconstruction.source_topology_id,
        reconstruction.target_topology_id,
        reconstruction.domain_id,
    ) != (
        cell_geometry_id(source_geometry),
        source_mesh.topology_id,
        target_mesh.topology_id,
        deformation.domain_id,
    ):
        raise ValueError(
            "material_transition.reconstruction must bind the source and target; got stale geometry or topology IDs"
        )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    reconstruction_ids = np.asarray(reconstruction.cell_global_ids)
    if target_ids.size != reconstruction_ids.size or not np.array_equal(
        np.sort(target_ids), np.sort(reconstruction_ids)
    ):
        raise ValueError(
            "material_transition.cell_ids must name the target scientific cells; got different cell ordering"
        )
    if deformation.maximum_displacement_bound > policy.reconstruction_tolerance:
        raise transition_types.CellGeometryTransitionError(
            "approximation_bound",
            "material_transition.displacement_bound must fit the declared geometric budget; got excess chart deformation",
            measured=deformation.maximum_displacement_bound,
            limit=policy.reconstruction_tolerance,
        )
    work = _budget(evaluations, policy)
    source_measure = math.fsum(np.asarray(deformation.source_cell_measures))
    target_measure = math.fsum(np.asarray(deformation.target_cell_measures))
    source_error = math.fsum(np.asarray(deformation.source_measure_errors)) + abs(
        float(np.spacing(source_measure))
    )
    target_error = math.fsum(np.asarray(deformation.target_measure_errors)) + abs(
        float(np.spacing(target_measure))
    )
    evidence = transition_types.CellGeometryTransitionEvidence(
        "bounded_chart_deformation",
        exact=False,
        node_count=reconstruction.geometry.coordinates.shape[0],
        evaluation_count=work,
        containment_defect=0.0,
        continuity_residual=reconstruction.continuity_residual,
        approximation_bound=deformation.maximum_displacement_bound,
        approximation_tolerance=policy.reconstruction_tolerance,
        rounding_slack=policy.continuity_tolerance
        * _extent(np.asarray(reconstruction.geometry.coordinates)),
        source_measure=source_measure,
        target_measure=target_measure,
        measure_exact=False,
        source_measure_error_bound=source_error,
        target_measure_error_bound=target_error,
        coverage_defect=None,
    )
    empty = np.zeros(0, dtype=np.int64)
    return transition_types.CellGeometryTransition(
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        source_geometry_id=cell_geometry_id(source_geometry),
        target_geometry_id=cell_geometry_id(reconstruction.geometry),
        geometry=reconstruction.geometry,
        vertex_coordinates=reconstruction.vertex_coordinates,
        target_cell_ids=jnp.asarray(target_ids),
        parent_cell_ids=jnp.full(target_ids.shape, -1, dtype=jnp.int64),
        parent_reference_vertices=jnp.zeros((target_ids.size, 3, 2), dtype=jnp.float64),
        coarsened_cell_ids=jnp.asarray(empty),
        coarsened_into_ids=jnp.asarray(empty),
        coarsened_reference_vertices=jnp.zeros((0, 3, 2), dtype=jnp.float64),
        node_owner_cells=reconstruction.node_owner_cells,
        node_owner_locals=reconstruction.node_owner_locals,
        evidence=evidence,
        policy_id=policy.policy_id,
        chart_correspondence_id=deformation.deformation_id,
        chart_deformation=deformation,
        transition_id=canonical_fingerprint(
            {
                "kind": "chart-deformed-cell-geometry-transition",
                "source_geometry": cell_geometry_id(source_geometry),
                "target_geometry": cell_geometry_id(reconstruction.geometry),
                "source_topology": source_mesh.topology_id,
                "target_topology": target_mesh.topology_id,
                "correspondence": deformation.deformation_id,
                "reconstruction": reconstruction.reconstruction_id,
                "policy": policy.policy_id,
                "evidence": evidence.evidence_id,
            }
        ),
    )
