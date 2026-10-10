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
corner displacements, which keeps curved faces curved and is exact for that
declared motion extension.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import assert_never, Literal, NamedTuple, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..geometry._mesh_certificates import GlobalEmbeddingCertificate
from ..linalg import determinant_small_linear, SmallLinearSolvePlan, solve_small_linear
from ..linalg._hermitian_spectral import _fraction_sqrt_interval
from ..typing import Dim, Float64, Int32, Int64, parse
from ._cell_geometry import (
    _require_scalar_coordinate_element,
    BarycentricCellGeometryElement,
    CellGeometryElement,
    CellGeometryRestrictionSource,
    CellGeometrySpec,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    SplineCellGeometryElement,
)
from ._cell_geometry_validity import cell_geometry_id
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import (
    add,
    affine_arguments,
    compose,
    coordinate_corner_images,
    coordinate_expressions,
    coordinate_polynomials,
    CoordinateCoefficients,
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateSourceBank,
    Expression,
    expression_add,
    expression_bounds,
    expression_compose,
    expression_derivative,
    expression_determinant,
    expression_multiply,
    expression_scale,
    Polynomial,
    polynomial_bounds,
    RationalPolynomial,
    restrict_chart_expressions,
    rounded_point,
    scale,
    source_basis,
)
from ._exact_plc_geometry import (
    _ExactPlcBudget,
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)
from ._reference_cell import reference_cell_topology
from ._sphere_chart_deformation import (
    PreparedSphereChartDeformation,
    SphereGeometryReconstruction,
)
from ._surface_chart_deformation import PreparedSurfaceChartDeformation


if TYPE_CHECKING:
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry._surface_source_support import SurfaceSourceRootAtlas
    from ._nested_reference import _NestedReferencePair
    from .fem._reference import FiniteElementSpec

CellGeometryTransitionKind: TypeAlias = Literal[
    "nested_restriction",
    "coarsening_interpolation",
    "parent_restoration",
    "vertex_displacement",
    "bounded_chart_deformation",
    "source_realization",
]
CoarseningGeometryApproximation: TypeAlias = Literal[
    "exact_only", "bounded_interpolation"
]
SurfaceGeometryCorrespondence: TypeAlias = Literal[
    "refuse", "bounded_chart_deformation", "source_realization"
]
"""``exact_only`` accepts a coarse map only when it reproduces the fine map up to
rounding (for example the restored parent of a nested refinement);
``bounded_interpolation`` accepts the coarse Lagrange interpolant when its
certified sup-norm deviation stays within ``coarsening_tolerance``."""
CellGeometryTransitionRefusal: TypeAlias = Literal[
    "resource_limit", "discontinuous_source", "approximation_bound", "coverage"
]

_SIMPLEX_DIMENSIONS = {"interval": 1, "triangle": 2, "tetrahedron": 3}
_EPSILON = float(np.finfo(np.float64).eps)


class CellGeometryTransitionError(RuntimeError):
    """A geometry transition refused its declared contract; the source is kept."""

    def __init__(
        self,
        reason: CellGeometryTransitionRefusal,
        message: str,
        /,
        *,
        measured: float,
        limit: float,
    ) -> None:
        self.reason = parse(reason, CellGeometryTransitionRefusal, "reason")
        self.measured = float(measured)
        self.limit = float(limit)
        super().__init__(
            f"{message} (measured {self.measured:.6g}, limit {self.limit:.6g})"
        )


class NestedReferenceWitnesses(NamedTuple):
    """Reference-coordinate witnesses of nested simplices.

    Fine cell ``fine_cell_ids[c]`` lies in coarse cell ``coarse_cell_ids[c]``;
    ``fine_reference_vertices[c, i]`` is the coarse-cell reference coordinate of
    the fine cell's local vertex ``i`` (reference vertex ``0`` at the origin and
    vertex ``j`` at ``e_j``).
    """

    fine_cell_ids: np.ndarray
    coarse_cell_ids: np.ndarray
    fine_reference_vertices: np.ndarray


def _finite_non_negative(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return number


class CellGeometryTransitionPolicy(StrictModule, NonTrainableState):
    """Approximation, continuity, and resource controls of a geometry transition.

    ``coarsening`` selects the coarse-map approximation policy and
    ``coarsening_tolerance`` its absolute sup-norm bound in coordinate units.
    ``continuity_tolerance`` bounds the disagreement of a shared node evaluated
    from different cells and the rounding slack of exact claims, relative to the
    coordinate extent. ``maximum_evaluations`` bounds basis evaluations (points
    times local basis functions) before any is allocated, plus the exact
    coefficient-term visits of certified mapped measures, each reserved before
    the operation that performs it.
    """

    coarsening: CoarseningGeometryApproximation = eqx.field(static=True)
    coarsening_tolerance: float = eqx.field(static=True)
    reconstruction: SurfaceGeometryCorrespondence = eqx.field(static=True)
    reconstruction_tolerance: float = eqx.field(static=True)
    continuity_tolerance: float = eqx.field(static=True)
    maximum_evaluations: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coarsening: CoarseningGeometryApproximation = "exact_only",
        coarsening_tolerance: float = 0.0,
        reconstruction: SurfaceGeometryCorrespondence = "refuse",
        reconstruction_tolerance: float = 0.0,
        continuity_tolerance: float = 1e-12,
        maximum_evaluations: int = 1 << 26,
    ) -> None:
        approximation = parse(coarsening, CoarseningGeometryApproximation, "coarsening")
        tolerance = _finite_non_negative(coarsening_tolerance, "coarsening_tolerance")
        correspondence = parse(
            reconstruction, SurfaceGeometryCorrespondence, "reconstruction"
        )
        deformation_tolerance = _finite_non_negative(
            reconstruction_tolerance, "reconstruction_tolerance"
        )
        continuity = _finite_non_negative(continuity_tolerance, "continuity_tolerance")
        if isinstance(maximum_evaluations, bool) or not isinstance(
            maximum_evaluations, (int, np.integer)
        ):
            raise TypeError("maximum_evaluations must be an integer.")
        if maximum_evaluations < 1:
            raise ValueError("maximum_evaluations must be positive.")
        self.coarsening = approximation
        self.coarsening_tolerance = tolerance
        self.reconstruction = correspondence
        self.reconstruction_tolerance = deformation_tolerance
        self.continuity_tolerance = continuity
        self.maximum_evaluations = int(maximum_evaluations)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-transition-policy",
                "coarsening": approximation,
                "coarsening_tolerance": tolerance,
                "reconstruction": correspondence,
                "reconstruction_tolerance": deformation_tolerance,
                "continuity_tolerance": continuity,
                "maximum_evaluations": self.maximum_evaluations,
            }
        )


class CellGeometryTransitionEvidence(StrictModule, NonTrainableState):
    """What one geometry transition evaluated, bounded, and measured.

    ``exact`` claims that every target map equals the source map it restricts (or
    the declared displacement extension) up to ``rounding_slack``.
    ``containment_defect`` is the largest barycentric excursion of a nested
    witness outside its coarse reference simplex, ``continuity_residual`` the
    largest disagreement of one shared node evaluated from different cells, and
    ``approximation_bound`` the certified sup-norm deviation of coarsened maps from
    the fine maps they replace (zero without coarsening). ``measure_exact``
    describes exact integration before floating-point publication. Mixed mapped
    transitions additionally report absolute source/target integration and
    publication error bounds; ``None`` means no quantitative enclosure was
    prepared by the older simplex route. ``coverage_defect`` compares measures
    after subtracting those bounds and is absent for displacement.
    """

    kind: CellGeometryTransitionKind = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    containment_defect: float = eqx.field(static=True)
    continuity_residual: float = eqx.field(static=True)
    approximation_bound: float = eqx.field(static=True)
    approximation_tolerance: float = eqx.field(static=True)
    rounding_slack: float = eqx.field(static=True)
    source_measure: float = eqx.field(static=True)
    target_measure: float = eqx.field(static=True)
    measure_exact: bool = eqx.field(static=True)
    source_measure_error_bound: float | None = eqx.field(static=True)
    target_measure_error_bound: float | None = eqx.field(static=True)
    coverage_defect: float | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CellGeometryTransitionKind,
        /,
        *,
        exact: bool,
        node_count: int,
        evaluation_count: int,
        containment_defect: float,
        continuity_residual: float,
        approximation_bound: float,
        approximation_tolerance: float,
        rounding_slack: float,
        source_measure: float,
        target_measure: float,
        measure_exact: bool,
        coverage_defect: float | None,
        source_measure_error_bound: float | None = None,
        target_measure_error_bound: float | None = None,
    ) -> None:
        self.kind = parse(kind, CellGeometryTransitionKind, "kind")
        self.exact = bool(exact)
        self.node_count = int(node_count)
        self.evaluation_count = int(evaluation_count)
        self.containment_defect = float(containment_defect)
        self.continuity_residual = float(continuity_residual)
        self.approximation_bound = float(approximation_bound)
        self.approximation_tolerance = float(approximation_tolerance)
        self.rounding_slack = float(rounding_slack)
        self.source_measure = float(source_measure)
        self.target_measure = float(target_measure)
        self.measure_exact = bool(measure_exact)
        self.source_measure_error_bound = (
            None
            if source_measure_error_bound is None
            else _finite_non_negative(
                source_measure_error_bound, "source_measure_error_bound"
            )
        )
        self.target_measure_error_bound = (
            None
            if target_measure_error_bound is None
            else _finite_non_negative(
                target_measure_error_bound, "target_measure_error_bound"
            )
        )
        self.coverage_defect = None if coverage_defect is None else float(coverage_defect)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "cell-geometry-transition-evidence",
                "transition": self.kind,
                "exact": self.exact,
                "node_count": self.node_count,
                "evaluation_count": self.evaluation_count,
                "containment_defect": self.containment_defect,
                "continuity_residual": self.continuity_residual,
                "approximation_bound": self.approximation_bound,
                "approximation_tolerance": self.approximation_tolerance,
                "rounding_slack": self.rounding_slack,
                "source_measure": self.source_measure,
                "target_measure": self.target_measure,
                "measure_exact": self.measure_exact,
                "source_measure_error_bound": self.source_measure_error_bound,
                "target_measure_error_bound": self.target_measure_error_bound,
                "coverage_defect": self.coverage_defect,
            }
        )


class _TargetCellDim(Dim):
    """Target cells in concatenated block order."""


class _CornerDim(Dim, minimum=2):
    """Corners of one simplex."""


class _ParentCornerDim(Dim, minimum=2):
    """Target-cell reference-corner capacity of the parent witness bank."""


class _CoarsenedCornerDim(Dim, minimum=2):
    """Fine source-cell corner capacity, independent of the target bank."""


class _ReferenceDim(Dim, minimum=1):
    """Reference-cell dimension."""


class _FineCellDim(Dim):
    """Source cells coarsened into target cells."""


class _VertexDim(Dim, minimum=1):
    """Target mesh vertex rows."""


class _AmbientDim(Dim, minimum=1):
    """Ambient coordinate dimension."""


class _NodeDim(Dim, minimum=1):
    """Target coordinate nodes."""


class CellGeometryTransition(StrictModule, NonTrainableState):
    """Accepted successor coordinate map with its source/reference witnesses.

    ``geometry`` is the complete target `CellGeometrySpec` and
    ``vertex_coordinates`` the target mesh vertex rows it places (the corners the
    target `CellMesh` must carry). Target cell ``c`` (concatenated block order,
    ``target_cell_ids[c]``) was evaluated from source cell ``parent_cell_ids[c]`` at
    parent reference corners ``parent_reference_vertices[c]``; a coarsened target
    cell has parent ``-1`` and its source cells are the ``coarsened_*`` witnesses.
    Geometry node ``n`` was evaluated as local node ``node_owner_locals[n]`` of
    target cell ``node_owner_cells[n]``.
    A bounded chart deformation instead retains its actual
    ``chart_deformation`` certificate. Its material correspondence and physical
    area inventories cannot be recovered from nested parent corners or the
    ``chart_correspondence_id`` alone.
    """

    __strict_contract__ = True

    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    geometry: CellGeometrySpec
    vertex_coordinates: Float64[_VertexDim, _AmbientDim]
    target_cell_ids: Int64[_TargetCellDim]
    parent_cell_ids: Int64[_TargetCellDim]
    parent_reference_vertices: Float64[_TargetCellDim, _ParentCornerDim, _ReferenceDim]
    coarsened_cell_ids: Int64[_FineCellDim]
    coarsened_into_ids: Int64[_FineCellDim]
    coarsened_reference_vertices: Float64[
        _FineCellDim, _CoarsenedCornerDim, _ReferenceDim
    ]
    node_owner_cells: Int64[_NodeDim] | None
    node_owner_locals: Int32[_NodeDim] | None
    evidence: CellGeometryTransitionEvidence
    policy_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)
    chart_correspondence_id: str | None = eqx.field(static=True, default=None)
    chart_deformation: (
        PreparedSurfaceChartDeformation | PreparedSphereChartDeformation | None
    ) = None


# Host-only immutable preparation: coordinate maps are evaluated once per epoch
# transition from NumPy topology tables; the accepted successor geometry is the
# device-facing product.


class _Block(NamedTuple):
    element: FiniteElementSpec
    routes: np.ndarray
    cell_ids: np.ndarray
    dimension: int
    degree: int


def _simplex_blocks(
    mesh: CellMesh, geometry: CellGeometrySpec, role: str, /
) -> tuple[_Block, ...]:
    """Per-block complete simplex Lagrange coordinate elements of ``geometry``."""

    from ._cell_geometry import coordinate_lagrange_element
    from .fem._reference import FiniteElementSpec

    elements, routes, _ = geometry.resolve(mesh)
    blocks = []
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        dimension = _SIMPLEX_DIMENSIONS.get(block.cell_kind)
        if dimension is None or not isinstance(element, FiniteElementSpec):
            raise ValueError(
                "Geometry transitions require simplex Lagrange coordinate maps; "
                f"{role} block {block.name!r} has {block.cell_kind} cells."
            )
        complete = coordinate_lagrange_element(block.cell_kind, element.degree)
        if element.element_id != complete.element_id:
            raise ValueError(
                f"The {role} block {block.name!r} coordinate element is not the "
                "complete Lagrange element of its degree."
            )
        blocks.append(
            _Block(
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


def _tabulate(
    element: FiniteElementSpec, points: np.ndarray, /
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
    element: FiniteElementSpec, nodes: np.ndarray, points: np.ndarray, /
) -> np.ndarray:
    """Coordinate map of cells with nodes ``(C, m, D)`` at points ``(C, q, d)``."""

    values, _ = _tabulate(element, points)
    return np.asarray(contract("cqm,cmD->cqD", values, nodes))


class _CellIndex(NamedTuple):
    """Cells sorted by global ID with their block and block row."""

    ids: np.ndarray
    blocks: np.ndarray
    rows: np.ndarray


def _cell_index(blocks: tuple[_Block, ...], /) -> _CellIndex:
    ids = np.concatenate([block.cell_ids for block in blocks])
    owners = np.concatenate(
        [np.full(block.cell_ids.shape, index) for index, block in enumerate(blocks)]
    )
    rows = np.concatenate([np.arange(block.cell_ids.size) for block in blocks])
    order = np.argsort(ids, kind="stable")
    return _CellIndex(ids[order], owners[order], rows[order])


def _locate(index: _CellIndex, identifiers: np.ndarray, name: str, /) -> np.ndarray:
    position = np.minimum(np.searchsorted(index.ids, identifiers), index.ids.size - 1)
    if np.any(index.ids[position] != identifiers):
        raise ValueError(f"{name} references cells absent from its mesh.")
    return position


def _extent(coordinates: np.ndarray, /) -> float:
    span = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)
    return max(float(np.max(np.abs(span))), float(np.max(np.abs(coordinates))), 1.0)


def _rule(block: _Block, extra: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Reference rule exact for ``extra`` plus the full-dimensional density degree."""

    from .fem._generic import _degree_aware_reference_rule

    points, weights = _degree_aware_reference_rule(
        block.element.cell_kind, extra + block.dimension * max(block.degree - 1, 0)
    )
    return np.asarray(points, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def _densities(
    block: _Block, coordinates: np.ndarray, points: np.ndarray, /
) -> np.ndarray:
    """Jacobian measure density ``sqrt(det(J^T J))`` of every cell at every point."""

    _, gradients = _tabulate(block.element, points)
    jacobian = np.asarray(contract("cmD,qmd->cqDd", coordinates[block.routes], gradients))
    gram = np.swapaxes(jacobian, -1, -2) @ jacobian
    determinant = np.asarray(
        determinant_small_linear(SmallLinearSolvePlan(block.dimension), gram)
    )
    return np.sqrt(np.maximum(determinant, 0.0))


def _measure(
    blocks: tuple[_Block, ...], coordinates: np.ndarray, /
) -> tuple[float, bool]:
    """Mapped measure of every cell; exact for full-dimensional polynomial maps."""

    total = 0.0
    for block in blocks:
        points, weights = _rule(block, 0)
        total += float(np.sum(_densities(block, coordinates, points) @ weights))
    exact = all(coordinates.shape[1] == block.dimension for block in blocks)
    return total, exact


def _budget(count: int, policy: CellGeometryTransitionPolicy, /) -> int:
    if count > policy.maximum_evaluations:
        raise CellGeometryTransitionError(
            "resource_limit",
            "The geometry transition exceeds its basis-evaluation budget",
            measured=count,
            limit=policy.maximum_evaluations,
        )
    return count


class _Placed(NamedTuple):
    coordinates: np.ndarray
    owner_cells: np.ndarray
    owner_locals: np.ndarray
    continuity: float


def _place(
    blocks: tuple[_Block, ...], values: list[np.ndarray], node_count: int, /
) -> _Placed:
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
        raise ValueError("Target cells must reference every target coordinate node.")
    owner = order[first]
    coordinates = flat[owner]
    residual = float(np.max(np.abs(flat - coordinates[routes]), initial=0.0))
    return _Placed(coordinates, cells[owner], slots[owner].astype(np.int32), residual)


def _vertex_rows(
    mesh: CellMesh, blocks: tuple[_Block, ...], coordinates: np.ndarray, /
) -> np.ndarray:
    """Target mesh vertex rows at their coordinate vertex nodes."""

    vertices = np.full((mesh.coordinates.shape[0], coordinates.shape[1]), np.nan)
    for mesh_block, block in zip(mesh.blocks, blocks, strict=True):
        vertex_dofs = [entity[0] for entity in block.element.entity_dofs[0]]
        rows = np.asarray(mesh_block.vertices, dtype=np.int64)
        vertices[rows] = coordinates[block.routes[:, vertex_dofs]]
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Every target mesh vertex must be a coordinate vertex node.")
    return vertices


def _witness(
    witnesses: NestedReferenceWitnesses | None, dimension: int, /
) -> NestedReferenceWitnesses:
    if witnesses is None:
        empty = np.zeros((0,), dtype=np.int64)
        return NestedReferenceWitnesses(
            empty, empty, np.zeros((0, dimension + 1, dimension), dtype=np.float64)
        )
    if not isinstance(witnesses, NestedReferenceWitnesses):
        raise TypeError("Nested witnesses must be NestedReferenceWitnesses or None.")
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
            "Nested witnesses must be finite (C,), (C,), (C, d+1, d) arrays."
        )
    return NestedReferenceWitnesses(fine, coarse, vertices)


def _excursion(vertices: np.ndarray, /) -> float:
    barycentric = _barycentric(vertices)
    return float(
        max(
            0.0,
            np.max(-barycentric, initial=0.0),
            np.max(barycentric - 1.0, initial=0.0),
        )
    )


def _empty_values(blocks: tuple[_Block, ...], ambient: int, /) -> list[np.ndarray]:
    return [
        np.full((block.cell_ids.size, block.routes.shape[1], ambient), np.nan)
        for block in blocks
    ]


class _Restricted(NamedTuple):
    parents: np.ndarray
    reference: np.ndarray
    evaluations: int


def _restrict(
    source_blocks: tuple[_Block, ...],
    source_coordinates: np.ndarray,
    target_blocks: tuple[_Block, ...],
    refinement: NestedReferenceWitnesses,
    values: list[np.ndarray],
    policy: CellGeometryTransitionPolicy,
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
    targets = _locate(target_index, refinement.fine_cell_ids, "Refinement")
    sources = _locate(source_index, refinement.coarse_cell_ids, "Refinement")
    parents[targets] = refinement.coarse_cell_ids
    reference[targets] = refinement.fine_reference_vertices
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
    return _Restricted(parents, reference, work)


def _fine_choices(
    coarse_block: _Block,
    corners: np.ndarray,
    coarse: np.ndarray,
    policy: CellGeometryTransitionPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine witness row, coarse node slot, and fine reference point of every node.

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
        raise CellGeometryTransitionError(
            "coverage",
            "A coarsening witness has a degenerate fine reference simplex",
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
        raise ValueError("Every coarsened target cell needs fine-cell witnesses.")
    if np.min(score[chosen]) < -policy.continuity_tolerance:
        raise CellGeometryTransitionError(
            "coverage",
            "Fine witnesses do not cover every coarse reference node",
            measured=-float(np.min(score[chosen])),
            limit=policy.continuity_tolerance,
        )
    fine, slot = np.divmod(chosen, width)
    return fine, slot, local[fine, slot]


class _Coarsened(NamedTuple):
    bound: float
    evaluations: int


def _coarsen(
    source_blocks: tuple[_Block, ...],
    source_coordinates: np.ndarray,
    target_blocks: tuple[_Block, ...],
    coarsening: NestedReferenceWitnesses,
    values: list[np.ndarray],
    policy: CellGeometryTransitionPolicy,
    /,
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
                parent_polynomials = coordinate_polynomials(block.element, coarse_nodes)
                child_polynomials = coordinate_polynomials(child.element, fine_nodes)
                if parent_polynomials is None or child_polynomials is None:
                    raise ValueError(
                        "Coarsening requires source-level polynomial bounds."
                    )
                arguments = affine_arguments(
                    corners[member, 0], (corners[member, 1:] - corners[member, :1]).T
                )
                for parent_map, child_map in zip(
                    parent_polynomials, child_polynomials, strict=True
                ):
                    difference = add(compose(parent_map, arguments), scale(child_map, -1))
                    lower, upper = polynomial_bounds(difference, "simplex", dimension)
                    bound = max(bound, abs(lower), abs(upper))
    return _Coarsened(bound * (1.0 + 8.0 * _EPSILON), work)


def _transition(
    kind: CellGeometryTransitionKind,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    geometry: CellGeometrySpec,
    placed: _Placed,
    parents: tuple[np.ndarray, np.ndarray],
    coarsening: NestedReferenceWitnesses,
    evidence: CellGeometryTransitionEvidence,
    policy: CellGeometryTransitionPolicy,
    /,
) -> CellGeometryTransition:
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    order = np.argsort(target_ids, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    parent_ids, reference = parents
    vertex_coordinates = _vertex_rows(
        target_mesh,
        _simplex_blocks(target_mesh, geometry, "target"),
        np.asarray(geometry.coordinates, dtype=np.float64),
    )
    source_id = cell_geometry_id(source_geometry)
    target_id = cell_geometry_id(geometry)
    return CellGeometryTransition(
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        source_geometry_id=source_id,
        target_geometry_id=target_id,
        geometry=geometry,
        vertex_coordinates=jnp.asarray(vertex_coordinates),
        target_cell_ids=jnp.asarray(target_ids),
        parent_cell_ids=jnp.asarray(parent_ids[rank]),
        parent_reference_vertices=jnp.asarray(reference[rank]),
        coarsened_cell_ids=jnp.asarray(coarsening.fine_cell_ids),
        coarsened_into_ids=jnp.asarray(coarsening.coarse_cell_ids),
        coarsened_reference_vertices=jnp.asarray(coarsening.fine_reference_vertices),
        node_owner_cells=jnp.asarray(placed.owner_cells),
        node_owner_locals=jnp.asarray(placed.owner_locals),
        evidence=evidence,
        policy_id=policy.policy_id,
        transition_id=canonical_fingerprint(
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
        ),
    )


def _plc_nested_coordinates(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    source_blocks: tuple[_Block, ...],
    target_blocks: tuple[_Block, ...],
    refined: NestedReferenceWitnesses,
    source: ExactPlcCellGeometrySource,
    policy: CellGeometryTransitionPolicy,
    ledger: CoordinateEnclosureBudget,
) -> tuple[_Placed, _Restricted, ExactPlcCellGeometrySource]:
    from ._coordinate_enclosure import evaluate

    bank = source_geometry.source_coordinates()
    count = target_mesh.coordinates.shape[0]
    ledger.reserve(0, count * (64 + 3 * (128 + 2 * ((source.maximum_bits + 7) // 8))))
    authorities: list[tuple[Fraction, ...] | None] = [None] * count
    coordinates = np.full((count, 3), np.nan, dtype=np.float64)
    strata = np.zeros(count, dtype=np.int8)
    rows = np.full(count, -1, dtype=np.int64)
    parameters = np.zeros((count, 2), dtype=np.float64)
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
    budget = _ExactPlcBudget(
        min(source.maximum_work, policy.maximum_evaluations), source.maximum_bits
    )
    locator = source._prepare_witness_locator(budget)
    point_witnesses: dict[
        tuple[Fraction, ...],
        tuple[int, int, tuple[float, ...], tuple[Fraction, ...]],
    ] = {}
    order = np.argsort(refined.fine_cell_ids, kind="stable")
    for child, parent, corners in zip(
        targets[order],
        sources[order],
        refined.fine_reference_vertices[order],
        strict=True,
    ):
        target = target_blocks[int(target_index.blocks[child])]
        coarse = source_blocks[int(source_index.blocks[parent])]
        route = coarse.routes[int(source_index.rows[parent])]
        polynomials = coordinate_polynomials(
            coarse.element, tuple(bank[int(index)] for index in route)
        )
        if polynomials is None:
            raise ValueError("Exact PLC restriction lost its affine coordinate source.")
        local_points = (
            _barycentric(np.asarray(target.element.reference_nodes, dtype=np.float64))
            @ corners
        )
        target_route = target.routes[int(target_index.rows[child])]
        for vertex, local in zip(target_route, local_points, strict=True):
            point = tuple(
                evaluate(polynomial, tuple(Fraction(float(value)) for value in local))
                for polynomial in polynomials
            )
            if len(point) != 3:
                raise ValueError(
                    "Exact PLC restriction requires a three-dimensional coordinate source."
                )
            previous = authorities[vertex]
            # Only exact source authority equality can bypass witness location;
            # equal rounded carrier coordinates do not establish a shared seam.
            if previous is not None and point == previous:
                continue
            located = point_witnesses.get(point)
            if located is None:
                stratum, row, witness, authority = source._locate_witness(
                    (point[0], point[1], point[2]),
                    budget,
                    locator=locator,
                )
                ledger.reserve(
                    0,
                    128 + 6 * (128 + 2 * ((source.maximum_bits + 7) // 8)),
                )
                located = (stratum, row, tuple(witness), tuple(authority))
                point_witnesses[point] = located
            stratum, row, witness, authority = located
            if previous is not None:
                if authority != previous:
                    raise ValueError(
                        "Incident PLC source restrictions disagree on an exact shared target vertex."
                    )
                continue
            authorities[vertex] = tuple(authority)
            strata[vertex], rows[vertex], parameters[vertex] = stratum, row, witness
            coordinates[vertex] = tuple(float(value) for value in authority)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Nested PLC witnesses leave an unbound target vertex.")
    successor = source._with_witnesses(strata, rows, parameters)
    successor.prepare(coordinates)
    placed = _place(
        target_blocks, [coordinates[block.routes] for block in target_blocks], count
    )
    return placed, _Restricted(parents, reference, budget.work), successor


def _plc_nested_approximation(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    refined: NestedReferenceWitnesses,
    coarsened: NestedReferenceWitnesses,
) -> tuple[float, float]:
    from ._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        outward,
        prepared_coordinate_source_bank,
    )

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError("PLC correspondence lost its owning coefficient ledger.")
    cells = []
    for mesh, geometry in (
        (source_mesh, source_geometry),
        (target_mesh, target_geometry),
    ):
        bank = prepared_coordinate_source_bank(geometry)
        blocks = _simplex_blocks(mesh, geometry, "PLC correspondence")
        if any(block.dimension != 3 or block.degree != 1 for block in blocks):
            raise ValueError("PLC correspondence requires exact affine tetrahedral maps.")
        cells.append(
            {
                int(identifier): tuple(
                    bank[int(route[entity[0]])] for entity in block.element.entity_dofs[0]
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
        bound = Fraction(0)
        for fine_id, coarse_id, corners in zip(
            witnesses.fine_cell_ids,
            witnesses.coarse_cell_ids,
            witnesses.fine_reference_vertices,
            strict=True,
        ):
            fine, coarse = fine_cells[int(fine_id)], coarse_cells[int(coarse_id)]
            for actual, reference in zip(fine, corners, strict=True):
                tail = tuple(Fraction(float(value)) for value in reference)
                weights = (Fraction(1) - sum(tail, Fraction(0)), *tail)
                ledger.reserve(6 + 3 * sum(bool(weight) for weight in weights))
                expected = tuple(
                    sum(
                        (
                            weight * point[axis]
                            for weight, point in zip(weights, coarse, strict=True)
                            if weight
                        ),
                        Fraction(0),
                    )
                    for axis in range(3)
                )
                # An affine difference attains its component extrema at corners.
                bound = max(
                    bound,
                    *(
                        abs(first - second)
                        for first, second in zip(actual, expected, strict=True)
                    ),
                )
        bounds.append(outward(bound, math.inf))
    return bounds[0], bounds[1]


def _exact_tetra_reference_weight(
    corners: np.ndarray,
    ledger: CoordinateEnclosureBudget,
    /,
) -> Fraction:
    """Exact determinant of one binary64 child-to-parent tetrahedron map."""
    from ._coordinate_enclosure import _reserve_polynomial

    ledger.reserve(corners.size)
    points = tuple(tuple(Fraction(float(value)) for value in point) for point in corners)
    bits = max(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        for point in points
        for value in point
    )
    _reserve_polynomial(26, 15, 0, 3 * (2 * bits + 1) + 3)
    columns = tuple(
        tuple(points[column + 1][axis] - points[0][axis] for column in range(3))
        for axis in range(3)
    )
    determinant = (
        columns[0][0] * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
        - columns[0][1] * (columns[1][0] * columns[2][2] - columns[1][2] * columns[2][0])
        + columns[0][2] * (columns[1][0] * columns[2][1] - columns[1][1] * columns[2][0])
    )
    if determinant <= 0:
        raise ValueError(
            "Nested PLC refinement has a non-positive exact reference measure."
        )
    return determinant


def _require_exact_refinement_partition(
    refined: NestedReferenceWitnesses,
    source_cell_ids: np.ndarray,
    ledger: CoordinateEnclosureBudget,
    /,
) -> None:
    """Prove repeated child patterns partition each authoritative source cell."""
    rows = refined.fine_reference_vertices
    with ledger.temporary_scope():
        ledger.reserve(
            2 * rows.shape[0],
            512 + rows.shape[0] * (256 + rows.shape[1] * rows.shape[2] * 8),
        )
        weights: dict[bytes, Fraction] = {}
        counts: dict[int, dict[bytes, int]] = {}
        for parent, corners in zip(
            refined.coarse_cell_ids,
            rows,
            strict=True,
        ):
            contiguous = np.ascontiguousarray(corners, dtype=np.float64)
            key = contiguous.view(np.uint64).tobytes()
            if key not in weights:
                weights[key] = _exact_tetra_reference_weight(contiguous, ledger)
            parent_counts = counts.setdefault(int(parent), {})
            parent_counts[key] = parent_counts.get(key, 0) + 1
        expected = {int(identifier) for identifier in source_cell_ids.tolist()}
        if set(counts) != expected:
            raise ValueError("Nested PLC refinement does not cover every source cell.")
        for parent_counts in counts.values():
            ledger.reserve(2 * len(parent_counts))
            total = sum(
                (count * weights[key] for key, count in parent_counts.items()),
                Fraction(0),
            )
            if total != 1:
                raise ValueError(
                    "Nested PLC refinement is not an exact reference partition."
                )


def _transition_exact_plc_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    refinement: NestedReferenceWitnesses | None,
    coarsening: NestedReferenceWitnesses | None,
    policy: CellGeometryTransitionPolicy,
) -> CellGeometryTransition:
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(policy.maximum_evaluations, sys.maxsize)
        with ledger.activate():
            return _transition_exact_plc_geometry(
                source_mesh,
                source_geometry,
                target_mesh,
                target_layout,
                refinement,
                coarsening,
                policy,
            )
    with ledger.bound_stage(policy.maximum_evaluations, ledger.maximum_memory_bytes):
        return _execute_exact_plc_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            target_layout,
            refinement,
            coarsening,
            policy,
            ledger,
        )


def _execute_exact_plc_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    refinement: NestedReferenceWitnesses | None,
    coarsening: NestedReferenceWitnesses | None,
    policy: CellGeometryTransitionPolicy,
    ledger: CoordinateEnclosureBudget,
) -> CellGeometryTransition:
    starting_work = ledger.work_units
    source = source_geometry.exact_source
    if not isinstance(source, ExactPlcCellGeometrySource):
        raise TypeError("Nested PLC transitions require their canonical source owner.")
    source_blocks = _simplex_blocks(source_mesh, source_geometry, "source")
    target_blocks = _simplex_blocks(target_mesh, target_layout, "target")
    if any(
        block.dimension != 3 or block.degree != 1
        for block in (*source_blocks, *target_blocks)
    ):
        raise ValueError(
            "Exact PLC transitions preserve direct affine tetrahedral layouts."
        )
    refined, coarsened = _witness(refinement, 3), _witness(coarsening, 3)
    target_ids = np.concatenate([block.cell_ids for block in target_blocks])
    covered = np.concatenate(
        (refined.fine_cell_ids, np.unique(coarsened.coarse_cell_ids))
    )
    if covered.size != target_ids.size or not np.array_equal(
        np.sort(covered), np.sort(target_ids)
    ):
        raise ValueError(
            "Every target PLC cell requires one refinement or coarsening witness."
        )
    with ledger.temporary_scope():
        placed, restricted, successor = _plc_nested_coordinates(
            source_mesh,
            source_geometry,
            target_mesh,
            source_blocks,
            target_blocks,
            refined,
            source,
            policy,
            ledger,
        )
    candidate = target_mesh.with_coordinates(
        placed.coordinates, numeric_version=target_mesh.numeric_version
    )
    geometry = CellGeometrySpec.plc(candidate, successor)
    # The successor source, not a rounded carrier-only surrogate, owns all
    # incident affine maps. Validate its exact corner images before any
    # association transfer or candidate publication can consume the seam.
    from ._coordinate_enclosure import coordinate_corner_images, rounded_point

    successor_bank = geometry.source_coordinates()
    carrier_bits = np.ascontiguousarray(candidate.coordinates, dtype=np.float64).view(
        np.uint64
    )
    for block in target_blocks:
        for route in block.routes:
            ledger.reserve(route.size)
            images = coordinate_corner_images(
                block.element, tuple(successor_bank[int(vertex)] for vertex in route)
            )
            if images is None:
                raise ValueError(
                    "The PLC successor lost its complete exact affine corner map."
                )
            for vertex, image in zip(route, images, strict=True):
                if image != successor_bank[int(vertex)]:
                    raise ValueError(
                        "Incident PLC successor maps disagree on an exact shared corner."
                    )
                if not np.array_equal(
                    rounded_point(image).view(np.uint64), carrier_bits[vertex]
                ):
                    raise ValueError(
                        "PLC successor carrier corners are not the correctly rounded exact map images."
                    )
    refinement_bound, coarsening_bound = _plc_nested_approximation(
        source_mesh,
        source_geometry,
        candidate,
        geometry,
        refined,
        coarsened,
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
        raise CellGeometryTransitionError(
            "approximation_bound",
            "Nested PLC source correspondence exceeds its declared bound",
            measured=max(refinement_bound, coarsening_bound),
            limit=max(slack, tolerance),
        )
    source_values, source_errors, source_exact = _certified_cell_measures(
        source_mesh,
        source_geometry,
        maximum_work=policy.maximum_evaluations,
    )
    source_measure = math.fsum(source_values)
    source_error = math.fsum(source_errors) + abs(float(np.spacing(source_measure)))
    if not coarsened.fine_cell_ids.size:
        source_ids = np.concatenate([block.cell_ids for block in source_blocks])
        _require_exact_refinement_partition(refined, source_ids, ledger)
        target_measure = source_measure
        target_error = source_error
        target_exact = source_exact
    else:
        target_values, target_errors, target_exact = _certified_cell_measures(
            candidate,
            geometry,
            maximum_work=policy.maximum_evaluations,
        )
        target_measure = math.fsum(target_values)
        target_error = math.fsum(target_errors) + abs(float(np.spacing(target_measure)))
    containment = max(
        _excursion(refined.fine_reference_vertices),
        _excursion(coarsened.fine_reference_vertices),
    )
    evidence = CellGeometryTransitionEvidence(
        "coarsening_interpolation"
        if coarsened.fine_cell_ids.size
        else "nested_restriction",
        exact=containment == 0.0 and max(refinement_bound, coarsening_bound) <= slack,
        node_count=placed.coordinates.shape[0],
        evaluation_count=ledger.work_units - starting_work,
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
        (restricted.parents, restricted.reference),
        coarsened,
        evidence,
        policy,
    )


def transition_nested_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec,
    /,
    *,
    refinement: NestedReferenceWitnesses | None,
    coarsening: NestedReferenceWitnesses | None = None,
    policy: CellGeometryTransitionPolicy | None = None,
) -> CellGeometryTransition:
    """Carry a simplex Lagrange coordinate map through one nested topology edit.

    ``target_layout`` fixes the target coordinate elements and node routes (its
    coordinate values are ignored); each target block must use its source block's
    element. Every target cell is either a refinement fine cell (preserved cells
    are their own parents with the identity witness) or the coarse cell of
    complete coarsening witnesses. Refinement nodes are the source map composed
    with the child reference map; coarsened cells follow ``policy.coarsening``.
    Refusals raise `CellGeometryTransitionError` and publish nothing.
    """

    for value, name in (
        (source_mesh, "source_mesh"),
        (target_mesh, "target_mesh"),
    ):
        if not isinstance(value, CellMesh):
            raise TypeError(f"{name} must be CellMesh.")
    for value, name in (
        (source_geometry, "source_geometry"),
        (target_layout, "target_layout"),
    ):
        if not isinstance(value, CellGeometrySpec):
            raise TypeError(f"{name} must be CellGeometrySpec.")
    policy_ = CellGeometryTransitionPolicy() if policy is None else policy
    if not isinstance(policy_, CellGeometryTransitionPolicy):
        raise TypeError("policy must be CellGeometryTransitionPolicy or None.")
    if isinstance(source_geometry.exact_source, ExactPlcCellGeometryConvexSource):
        raise ValueError(
            "Nested transitions cannot republish an exact PLC convex target bank as a simplex source."
        )
    if isinstance(source_geometry.exact_source, ExactPlcCellGeometrySource):
        return _transition_exact_plc_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            target_layout,
            refinement,
            coarsening,
            policy_,
        )
    simplex_layout = (
        all(
            block.cell_kind in _SIMPLEX_DIMENSIONS
            for block in (*source_mesh.blocks, *target_mesh.blocks)
        )
        and not any(
            isinstance(
                element, (BarycentricCellGeometryElement, RestrictedCellGeometryElement)
            )
            for element in source_geometry.elements
        )
        and {element.element_id for element in source_geometry.elements}
        == {element.element_id for element in target_layout.elements}
    )
    if not simplex_layout:
        return _transition_mixed_nested_geometry(
            source_mesh,
            source_geometry,
            target_mesh,
            refinement=refinement,
            coarsening=coarsening,
            policy=policy_,
        )
    source_blocks = _simplex_blocks(source_mesh, source_geometry, "source")
    target_blocks = _simplex_blocks(target_mesh, target_layout, "target")
    dimension = source_mesh.topological_dimension
    source_elements = {block.element.element_id for block in source_blocks}
    target_elements = {block.element.element_id for block in target_blocks}
    if target_mesh.topological_dimension != dimension or target_elements != (
        source_elements
    ):
        raise ValueError("Nested transitions keep the source coordinate elements.")
    refined = _witness(refinement, dimension)
    coarsened = _witness(coarsening, dimension)
    target_ids = np.concatenate([block.cell_ids for block in target_blocks])
    covered = np.concatenate(
        (refined.fine_cell_ids, np.unique(coarsened.coarse_cell_ids))
    )
    if covered.size != target_ids.size or not np.array_equal(
        np.sort(covered), np.sort(target_ids)
    ):
        raise ValueError(
            "Every target cell needs exactly one refinement or coarsening witness."
        )
    source_coordinates = np.asarray(source_geometry.coordinates, dtype=np.float64)
    values = _empty_values(target_blocks, source_coordinates.shape[1])
    restricted = _restrict(
        source_blocks, source_coordinates, target_blocks, refined, values, policy_
    )
    coarse = _coarsen(
        source_blocks, source_coordinates, target_blocks, coarsened, values, policy_
    )
    _budget(restricted.evaluations + coarse.evaluations, policy_)
    placed = _place(target_blocks, values, np.asarray(target_layout.coordinates).shape[0])
    slack = policy_.continuity_tolerance * _extent(source_coordinates)
    if placed.continuity > slack:
        raise CellGeometryTransitionError(
            "discontinuous_source",
            "Shared target nodes evaluate differently from their incident cells",
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
        raise CellGeometryTransitionError(
            "approximation_bound",
            "The coarse coordinate map deviates from the fine maps beyond the "
            f"{policy_.coarsening} policy",
            measured=coarse.bound,
            limit=tolerance,
        )
    elements, routes, _ = target_layout.resolve(target_mesh)
    names = tuple(block.name for block in target_mesh.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        dict(zip(names, routes, strict=True)),
        placed.coordinates,
    )
    source_measure, measure_exact = _measure(source_blocks, source_coordinates)
    target_measure, _ = _measure(target_blocks, placed.coordinates)
    containment = max(
        _excursion(refined.fine_reference_vertices),
        _excursion(coarsened.fine_reference_vertices),
    )
    evidence = CellGeometryTransitionEvidence(
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
        (restricted.parents, restricted.reference),
        coarsened,
        evidence,
        policy_,
    )


def transition_displaced_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    /,
    *,
    policy: CellGeometryTransitionPolicy | None = None,
) -> CellGeometryTransition:
    """Move a coordinate map with its mesh vertices on an unchanged topology.

    Every coordinate node moves by the affine interpolation of its cell's corner
    displacements, ``x'(xi) = x(xi) + sum_i lambda_i(xi) (v'_i - v_i)``: vertex
    nodes land on the moved vertices and the curved part of each map is kept.
    A shared node moves identically from every incident cell because the
    displacement on a shared entity depends only on that entity's corners.
    """

    if not isinstance(source_mesh, CellMesh) or not isinstance(target_mesh, CellMesh):
        raise TypeError("source_mesh and target_mesh must be CellMesh.")
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec.")
    policy_ = CellGeometryTransitionPolicy() if policy is None else policy
    if not isinstance(policy_, CellGeometryTransitionPolicy):
        raise TypeError("policy must be CellGeometryTransitionPolicy or None.")
    if target_mesh.topology_id != source_mesh.topology_id:
        raise ValueError("A displaced geometry keeps the source topology.")
    if isinstance(
        source_geometry.exact_source,
        (ExactPlcCellGeometrySource, ExactPlcCellGeometryConvexSource),
    ):
        raise ValueError(
            "Coefficient transitions republish rounded carriers and cannot carry exact PLC source ancestry."
        )
    blocks = _simplex_blocks(source_mesh, source_geometry, "source")
    _budget(sum(block.routes.size * (block.dimension + 1) for block in blocks), policy_)
    source_coordinates = np.asarray(source_geometry.coordinates, dtype=np.float64)
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
    placed = _place(blocks, values, source_coordinates.shape[0])
    slack = policy_.continuity_tolerance * _extent(source_coordinates)
    if placed.continuity > slack:
        raise CellGeometryTransitionError(
            "discontinuous_source",
            "Shared coordinate nodes move differently from their incident cells",
            measured=placed.continuity,
            limit=slack,
        )
    elements, routes, _ = source_geometry.resolve(source_mesh)
    names = tuple(block.name for block in source_mesh.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        dict(zip(names, routes, strict=True)),
        placed.coordinates,
    )
    source_measure, measure_exact = _measure(blocks, source_coordinates)
    target_measure, _ = _measure(blocks, placed.coordinates)
    # Witnesses follow ascending cell IDs: every cell is its own identity parent.
    identity = np.sort(np.concatenate([block.cell_ids for block in blocks]))
    dimension = source_mesh.topological_dimension
    reference = np.broadcast_to(
        np.concatenate((np.zeros((1, dimension)), np.eye(dimension)), axis=0),
        (identity.size, dimension + 1, dimension),
    ).copy()
    evidence = CellGeometryTransitionEvidence(
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
    )
    return _transition(
        "vertex_displacement",
        source_mesh,
        source_geometry,
        target_mesh,
        geometry,
        placed,
        (identity, reference),
        _witness(None, dimension),
        evidence,
        policy_,
    )


def is_affine_cell_geometry(mesh: CellMesh, geometry: CellGeometrySpec, /) -> bool:
    """Prove the complete source map is affine and binds its RNE carrier."""

    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("is_affine_cell_geometry requires a CellMesh and its geometry.")
    from .._meshcore import current_native_execution_budget
    from ..geometry._mesh_certificates import MeshCertificateLimits
    from ._coordinate_enclosure import coordinate_enclosure_budget

    native = current_native_execution_budget()
    if native is None:
        limits = MeshCertificateLimits()
        work, storage = limits.maximum_work_units, limits.maximum_scratch_bytes
    else:
        remaining = native.remaining()
        work, storage = remaining.remaining_work_units, remaining.remaining_scratch_bytes
    ledger = coordinate_enclosure_budget(work, storage)
    with ledger.activate():
        try:
            return _affine_source_carrier(mesh, geometry)
        finally:
            ledger.charge_native_work(
                ledger.work_units - ledger.native_charged_work_units
            )


def _affine_source_carrier(mesh: CellMesh, geometry: CellGeometrySpec, /) -> bool:
    from ._cell_geometry import CellVertexGeometryElement
    from ._coordinate_enclosure import (
        coordinate_polynomials,
        corner_images,
        prepared_coordinate_source_bank,
    )

    bank = prepared_coordinate_source_bank(geometry)
    elements, routes, _ = geometry._resolve(mesh, exact_source_prepared=True)
    carrier = np.asarray(mesh.coordinates, dtype=np.float64)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        vertices = np.asarray(block.vertices)
        rows = np.asarray(route)
        if isinstance(element, CellVertexGeometryElement):
            # Variable-topology elements are the actual vertex carrier, not a
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
        for row, corners in zip(rows, vertices, strict=True):
            local = tuple(bank[int(index)] for index in row if index >= 0)
            polynomials = coordinate_polynomials(element, local)
            if polynomials is None or any(
                sum(exponent) > 1
                for polynomial in polynomials
                for exponent, coefficient in polynomial.items()
                if coefficient
            ):
                return False
            images = corner_images(polynomials, block.cell_kind)
            rounded = np.asarray(
                [[float(value) for value in image] for image in images],
                dtype=np.float64,
            )
            actual = carrier[corners[corners >= 0]]
            if rounded.shape != actual.shape or not np.array_equal(
                rounded.view(np.uint64),
                actual.view(np.uint64),
            ):
                return False
    return True


def nested_geometry_degree(mesh: CellMesh, geometry: CellGeometrySpec, /) -> int:
    """Layout preparation degree of an admitted nested coordinate map.

    Simplex nodal restriction requires one common canonical coordinate degree.
    Mixed coefficient composition retains every block's own source degree; the
    returned maximum is only the preparation extent, not a geometry downgrade.
    """

    if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
        raise TypeError("nested_geometry_degree requires a CellMesh and its geometry.")
    from .fem._reference import FiniteElementSpec

    elements, _, _ = geometry.resolve(mesh)
    if any(block.cell_kind not in _SIMPLEX_DIMENSIONS for block in mesh.blocks) or any(
        not isinstance(element, FiniteElementSpec) for element in elements
    ):
        if any(
            not isinstance(
                element,
                (
                    FiniteElementSpec,
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
                "Nested mixed geometry requires scalar reference coordinate maps."
            )
        return max(
            _require_scalar_coordinate_element(element, "Nested mixed geometry").degree
            for element in elements
        )
    degrees = {block.degree for block in _simplex_blocks(mesh, geometry, "source")}
    if len(degrees) != 1:
        raise ValueError("Nested geometry transitions require one coordinate degree.")
    return degrees.pop()


def cell_geometry_vertex_measures(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> np.ndarray:
    """Integral of every mesh-vertex P1 hat function over the mapped cells.

    The rule is exact for full-dimensional polynomial coordinate maps (hat degree
    one plus the Jacobian determinant degree).
    """

    blocks = _simplex_blocks(mesh, geometry, "measured")
    coordinates = np.asarray(geometry.coordinates, dtype=np.float64)
    measures = np.zeros((mesh.coordinates.shape[0],), dtype=np.float64)
    for mesh_block, block in zip(mesh.blocks, blocks, strict=True):
        points, weights = _rule(block, 1)
        local = np.asarray(
            contract(
                "cq,q,qk->ck",
                _densities(block, coordinates, points),
                weights,
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
        raise ValueError("A mixed reference template has a singular source frame.")
    matrix = np.asarray(solved.value, dtype=np.float64).T
    offset = corners[0] - matrix @ vertices[0]
    if (
        np.max(
            np.abs(vertices @ matrix.T + offset - corners[: vertices.shape[0]]),
            initial=0.0,
        )
        > 64 * _EPSILON
    ):
        raise ValueError("A mixed witness is not an affine reference submap.")
    if (
        float(
            np.asarray(determinant_small_linear(SmallLinearSolvePlan(dimension), matrix))
        )
        <= 0
    ):
        raise ValueError("A mixed reference submap must preserve orientation.")
    return matrix, offset


@contextmanager
def _charged(
    ledger: CoordinateEnclosureBudget, policy: CellGeometryTransitionPolicy, /
) -> Iterator[None]:
    """Charge exact coefficient-term work to the transition evaluation budget.

    Exact rational corner images and measures evaluate no quadrature points; the
    coordinate-enclosure ledger reserves each operation's term visits before it
    runs, so exhaustion refuses before the work is performed. The supplied
    ledger independently owns any producer-specific polynomial-storage limit.
    """
    try:
        with ledger.activate():
            yield
    except CoordinateEnclosureResourceError as error:
        raise CellGeometryTransitionError(
            "resource_limit",
            (
                "The geometry transition exceeds its basis-evaluation budget"
                if error.resource == "coefficient_work"
                else f"The geometry transition exceeds its {error.resource} byte budget"
            ),
            measured=error.requested,
            limit=error.limit,
        ) from error


def _mixed_map_measure(mesh: CellMesh, geometry: CellGeometrySpec) -> tuple[float, bool]:
    values, _, exact = _certified_cell_measures(mesh, geometry)
    return math.fsum(values), exact


def _mapped_expression_restriction(
    polynomials: tuple[Expression, ...],
    source_kind: str,
    target_kind: str,
    matrix: NDArray[np.float64],
    offset: NDArray[np.float64],
) -> tuple[Expression, ...]:
    """Restrict the owning exact expression, retaining nonremovable quotients."""
    return restrict_chart_expressions(
        polynomials, source_kind, target_kind, offset, matrix
    )


def _mapped_coordinate_expressions(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
) -> tuple[Expression, ...]:
    scalar = _require_scalar_coordinate_element(element, "Mapped coordinate expressions")
    coordinates = coordinate_expressions(scalar, local)
    if coordinates is None:
        raise ValueError(
            "Mapped integration requires an authoritative coordinate source expression."
        )
    return coordinates


def _mapped_density_expression(
    element: CellGeometryElement, local: CoordinateCoefficients
) -> Expression:
    from ._coordinate_enclosure import (
        _coefficient_profile,
        _COORDINATE_BUDGET,
        _coordinate_source_rows,
        _reserve_polynomial,
        axes,
        constant,
        determinant,
        evaluate,
        multiply,
        power,
        rational_expression,
    )

    ledger = _COORDINATE_BUDGET.get()
    coefficients = _coordinate_source_rows(local)
    element = _require_scalar_coordinate_element(element, "Mapped density expressions")
    if (
        ledger is not None
        and isinstance(element, RestrictedCellGeometryElement)
        and len(coefficients[0]) == element.topological_dimension
    ):
        parent = element.source_element
        parent_density = _mapped_density_expression(parent, coefficients)
        dimension = element.topological_dimension
        if parent.cell_kind == "pyramid":
            collapse = power(
                add(constant(1, dimension), scale(axes(dimension)[2], -1)), 2, dimension
            )
            if isinstance(parent_density, RationalPolynomial):
                parent_density = rational_expression(
                    parent_density.numerator,
                    multiply(parent_density.denominator, collapse),
                )
            else:
                parent_density = rational_expression(parent_density, collapse)
        matrix = np.asarray(element.matrix, dtype=np.float64)
        action = tuple(
            tuple(constant(Fraction(float(value)), dimension) for value in row)
            for row in matrix
        )
        action_measure = evaluate(determinant(action), (Fraction(0),) * dimension)
        if action_measure <= 0:
            raise ValueError(
                "Mapped integration requires a certified strictly positive coordinate Jacobian."
            )
        density = expression_scale(
            _mapped_expression_restriction(
                (parent_density,),
                parent.cell_kind,
                element.cell_kind,
                matrix,
                np.asarray(element.offset, dtype=np.float64),
            )[0],
            action_measure,
        )
        if element.cell_kind == "pyramid":
            density = expression_multiply(
                density,
                power(
                    add(constant(1, dimension), scale(axes(dimension)[2], -1)),
                    2,
                    dimension,
                ),
            )
        return density
    cache_key = (
        element.element_id,
        canonical_fingerprint(array_tree_fingerprint(element)),
        coefficients,
    )
    if ledger is not None and cache_key in ledger.density_cache:
        return ledger.density_cache[cache_key][0]
    coordinates = _mapped_coordinate_expressions(element, local)
    dimension = element.topological_dimension
    axis_plane = False
    if len(coordinates) != dimension:
        if (
            element.cell_kind != "quadrilateral"
            or dimension != 2
            or len(coordinates) != 3
        ):
            raise ValueError(
                "Certified mapped integration requires full-dimensional or exact axis-plane quad geometry."
            )
        constant_axes = tuple(
            axis
            for axis, polynomial in enumerate(coordinates)
            if all(
                not expression_derivative(polynomial, reference_axis)
                for reference_axis in range(dimension)
            )
        )
        if len(constant_axes) != 1:
            raise ValueError(
                "Embedded quad integration requires an exactly constant source-expression axis-plane coordinate."
            )
        coordinates = tuple(
            polynomial
            for axis, polynomial in enumerate(coordinates)
            if axis != constant_axes[0]
        )
        axis_plane = True
    polynomial_coordinates = tuple(
        value for value in coordinates if not isinstance(value, RationalPolynomial)
    )
    affine = len(polynomial_coordinates) == len(coordinates) and all(
        sum(index) <= 1 for value in polynomial_coordinates for index in value
    )
    if affine and element.cell_kind != "pyramid":
        units = tuple(
            tuple(int(axis == column) for axis in range(dimension))
            for column in range(dimension)
        )
        jacobian_values = tuple(
            tuple(value.get(unit, Fraction(0)) for unit in units)
            for value in polynomial_coordinates
        )
        numerator, denominator = _coefficient_profile(polynomial_coordinates)
        match dimension:
            case 1:
                arithmetic, terms = 0, 1
            case 2:
                arithmetic, terms = 3, 2
            case 3:
                arithmetic, terms = 17, 6
            case _:
                raise ValueError(
                    "Affine mapped integration supports dimensions one through three."
                )
        _reserve_polynomial(
            dimension * dimension + arithmetic,
            terms,
            0,
            dimension * (numerator + denominator) + terms.bit_length(),
        )
        if dimension == 1:
            determinant_value = jacobian_values[0][0]
        elif dimension == 2:
            determinant_value = (
                jacobian_values[0][0] * jacobian_values[1][1]
                - jacobian_values[0][1] * jacobian_values[1][0]
            )
        else:
            determinant_value = (
                jacobian_values[0][0]
                * (
                    jacobian_values[1][1] * jacobian_values[2][2]
                    - jacobian_values[1][2] * jacobian_values[2][1]
                )
                - jacobian_values[0][1]
                * (
                    jacobian_values[1][0] * jacobian_values[2][2]
                    - jacobian_values[1][2] * jacobian_values[2][0]
                )
                + jacobian_values[0][2]
                * (
                    jacobian_values[1][0] * jacobian_values[2][1]
                    - jacobian_values[1][1] * jacobian_values[2][0]
                )
            )
        if axis_plane:
            determinant_value = abs(determinant_value)
        if determinant_value <= 0:
            raise ValueError(
                "Mapped integration requires a certified strictly positive coordinate Jacobian."
            )
        density = constant(determinant_value, dimension)
        if ledger is not None:
            retained = (density,)
            ledger.retain_basis(retained)
            ledger.density_cache[cache_key] = retained
        return density
    if affine:
        # Retained affine coordinate coefficients already prove a constant
        # Jacobian. Do not differentiate and reserve each discarded constant term.
        units = tuple(
            tuple(int(axis == column) for axis in range(dimension))
            for column in range(dimension)
        )
        jacobian = tuple(
            tuple(constant(value.get(unit, Fraction(0)), dimension) for unit in units)
            for value in polynomial_coordinates
        )
    else:
        jacobian = tuple(
            tuple(expression_derivative(value, axis) for axis in range(dimension))
            for value in coordinates
        )
    density = expression_determinant(jacobian)
    domain = (
        "simplex"
        if element.cell_kind in ("interval", "triangle", "tetrahedron")
        else "prism"
        if element.cell_kind == "prism"
        else "box"
    )
    if element.cell_kind == "pyramid":
        # The collapsed chart determinant already includes (1-w)^2. Prove the
        # physical Jacobian positive before integrating that chart density.
        from ._coordinate_enclosure import rational_expression

        collapse = power(
            add(constant(1, dimension), scale(axes(dimension)[2], -1)), 2, dimension
        )
        if isinstance(density, RationalPolynomial):
            physical_density = rational_expression(
                density.numerator, multiply(density.denominator, collapse)
            )
        else:
            physical_density = rational_expression(density, collapse)
    else:
        physical_density = density
    lower, upper = expression_bounds(physical_density, domain, dimension)
    if axis_plane and upper < 0:
        density, lower = expression_scale(density, -1), -upper
    if lower <= 0:
        raise ValueError(
            "Mapped integration requires a certified strictly positive coordinate Jacobian."
        )
    if ledger is not None:
        retained = (density,)
        ledger.retain_basis(retained)
        ledger.density_cache[cache_key] = retained
    return density


def _mapped_polynomial_integral_fraction(polynomial: Polynomial, kind: str) -> Fraction:
    from fractions import Fraction

    total = Fraction(0)
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(len(polynomial))
    for index, coefficient in polynomial.items():
        if kind in ("interval", "triangle", "tetrahedron"):
            moment = Fraction(
                math.prod(math.factorial(i) for i in index),
                math.factorial(sum(index) + len(index)),
            )
        elif kind == "prism":
            moment = Fraction(
                math.factorial(index[0]) * math.factorial(index[1]),
                math.factorial(index[0] + index[1] + 2) * (index[2] + 1),
            )
        else:
            moment = Fraction(1, math.prod(i + 1 for i in index))
        total += coefficient * moment
    return total


def _integrate_mapped_polynomial(
    polynomial: Expression, kind: str
) -> tuple[float, float]:
    """Exact source-expression integral and outward binary64 publication error."""
    from ._coordinate_enclosure import outward

    if isinstance(polynomial, RationalPolynomial):
        value, error, _ = _integrate_mapped_rational(polynomial, kind)
        return value, error
    total = _mapped_polynomial_integral_fraction(polynomial, kind)
    value = float(total)
    error = max(value - outward(total, -math.inf), outward(total, math.inf) - value)
    return value, error


def _integrate_mapped_rational(
    value: RationalPolynomial,
    kind: str,
    *,
    absolute_tolerance: Fraction = Fraction(1, 10**12),
    relative_tolerance: float = 0.0,
    maximum_terms: int = 32,
) -> tuple[float, float, bool]:
    """Integrate the exact quotient with a certified geometric-series remainder.

    An apex singularity of a tetrahedron is first removed by its exact cone
    chart. This is a change of variables of the owning expression, not a
    replacement by the corner interpolant.
    """
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    if _COORDINATE_BUDGET.get() is None:
        ledger = CoordinateEnclosureBudget(100_000_000, sys.maxsize)
        try:
            with ledger.activate():
                return _integrate_mapped_rational(
                    value,
                    kind,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    maximum_terms=maximum_terms,
                )
        except CoordinateEnclosureResourceError as error:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Rational integration coefficient work exhausted its declared budget.",
                measured=error.requested,
                limit=ledger.maximum_work_units,
            ) from error
    from ._coordinate_enclosure import (
        axes,
        bernstein_coefficients,
        constant,
        determinant,
        evaluate,
        multiply,
        outward,
        power,
    )

    dimension = len(next(iter(value.denominator)))
    if kind == "tetrahedron":
        vertices = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
        singular = [
            index
            for index, vertex in enumerate(vertices)
            if not evaluate(value.denominator, tuple(Fraction(float(x)) for x in vertex))
        ]
        if len(singular) == 1:
            apex = vertices[singular[0]]
            base = vertices[[index for index in range(4) if index != singular[0]]]
            u, v, w = axes(3)
            collapse = add(constant(1, 3), scale(w, -1))
            base_chart = affine_arguments(base[0], (base[1:] - base[0]).T)
            triangle_chart = tuple(compose(term, (u, v)) for term in base_chart)
            arguments = tuple(
                add(multiply(term, collapse), scale(w, Fraction(float(x))))
                for term, x in zip(triangle_chart, apex, strict=True)
            )
            mapped = expression_compose(value, arguments)
            frame = np.column_stack(
                (base[1] - base[0], base[2] - base[0], apex - base[0])
            )
            factor = abs(
                evaluate(
                    determinant(
                        tuple(
                            tuple(constant(Fraction(float(entry)), 3) for entry in row)
                            for row in frame
                        )
                    ),
                    (Fraction(0),) * 3,
                )
            )
            mapped = expression_multiply(mapped, scale(power(collapse, 2, 3), factor))
            if isinstance(mapped, RationalPolynomial):
                return _integrate_mapped_rational(
                    mapped,
                    "prism",
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    maximum_terms=maximum_terms,
                )
            integral, error = _integrate_mapped_polynomial(mapped, "prism")
            return integral, error, True
    domain = (
        "simplex"
        if kind in ("interval", "triangle", "tetrahedron")
        else "prism"
        if kind == "prism"
        else "box"
    )
    controls = bernstein_coefficients(value.denominator, domain, dimension)
    lower, upper = min(controls), max(controls)
    if upper < 0:
        value = RationalPolynomial(
            scale(value.numerator, -1), scale(value.denominator, -1)
        )
        lower, upper = -upper, -lower
    if lower <= 0:
        raise ValueError(
            "Rational integration denominator has no strictly positive enclosure."
        )
    center = (lower + upper) / 2
    ratio = add(constant(1, dimension), scale(value.denominator, -1 / center))
    rho = (upper - lower) / (upper + lower)
    numerator_bound = max(
        abs(coefficient)
        for coefficient in bernstein_coefficients(value.numerator, domain, dimension)
    )
    volume = _mapped_polynomial_integral_fraction(constant(1, dimension), kind)
    term = constant(1, dimension)
    total = Fraction(0)
    for order in range(maximum_terms):
        total += (
            _mapped_polynomial_integral_fraction(multiply(value.numerator, term), kind)
            / center
        )
        remainder = volume * numerator_bound * rho ** (order + 1) / (center * (1 - rho))
        allowed = max(
            absolute_tolerance,
            Fraction(relative_tolerance) * max(abs(total) - remainder, Fraction(0)),
        )
        if remainder <= allowed:
            lower_result, upper_result = (
                outward(total - remainder, -math.inf),
                outward(total + remainder, math.inf),
            )
            published = float(total)
            return (
                published,
                max(published - lower_result, upper_result - published),
                remainder == 0,
            )
        term = multiply(term, ratio)
    raise CellGeometryTransitionError(
        "resource_limit",
        "Rational integration series exhausted its certified remainder budget.",
        measured=maximum_terms,
        limit=maximum_terms,
    )


def _mapped_geometry_cells(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
) -> tuple[
    tuple[
        FiniteElementSpec
        | BarycentricCellGeometryElement
        | RestrictedCellGeometryElement
        | PolynomialComposedCellGeometryElement
        | RationalComposedCellGeometryElement
        | SplineCellGeometryElement
        | LayerColumnCellGeometryElement,
        CoordinateSourceBank,
    ],
    ...,
]:
    from ._coordinate_enclosure import prepared_coordinate_source_bank

    elements, routes, _ = geometry.resolve(mesh)
    local = prepared_coordinate_source_bank(geometry)
    return tuple(
        (
            _require_scalar_coordinate_element(element, "Mapped geometry cells"),
            tuple(local[index] for index in row),
        )
        for element, route in zip(elements, routes, strict=True)
        for row in np.asarray(route, dtype=np.int64)
    )


def _dense_polynomial_interval(
    polynomial: Polynomial,
    *,
    monomial_bounds: dict[tuple[int, ...], Fraction] | None = None,
    reference_radius: Fraction | None = None,
) -> tuple[NDArray[np.float64], Fraction]:
    from fractions import Fraction

    shape = tuple(
        max((index[axis] for index in polynomial), default=0) + 1 for axis in range(2)
    )
    values = np.zeros(shape, dtype=np.float64)
    error = Fraction(0)
    for index, coefficient in polynomial.items():
        values[index] = float(coefficient)
        bound = (
            Fraction(1)
            if monomial_bounds is None
            else _monomial_uniform_bound(index, monomial_bounds, reference_radius)
        )
        error += abs(coefficient - Fraction(float(values[index]))) * bound
    return values, error


def _dense_polynomial_norm(values: NDArray[np.float64]) -> Fraction:
    from fractions import Fraction

    return sum(
        (abs(Fraction(float(value))) for value in values.flat if value), Fraction(0)
    )


def _monomial_uniform_bound(
    index: tuple[int, ...],
    cache: dict[tuple[int, ...], Fraction],
    radius: Fraction | None = None,
) -> Fraction:
    value = cache.get(index)
    if value is not None:
        return value
    a, b = index
    from ._coordinate_enclosure import _reserve_polynomial

    _reserve_polynomial(1, 1, 2, (a + b) * (a + b + 1).bit_length() + 2)
    # Centred coordinates lie in [-radius, radius]; original simplex coordinates
    # use their constrained monomial maximizer. Both norms are submultiplicative.
    value = (
        radius ** (a + b)
        if radius is not None
        else Fraction(a**a * b**b, (a + b) ** (a + b))
        if a and b
        else Fraction(1)
    )
    cache[index] = value
    return value


def _dense_domain_polynomial_norm(
    values: NDArray[np.float64],
    simplex_bounds: dict[tuple[int, ...], Fraction] | None,
    reference_radius: Fraction | None = None,
) -> Fraction:
    if simplex_bounds is None:
        return _dense_polynomial_norm(values)
    return sum(
        (
            abs(Fraction(float(value)))
            * _monomial_uniform_bound(index, simplex_bounds, reference_radius)
            for index, value in np.ndenumerate(values)
            if value
        ),
        Fraction(0),
    )


def _dense_polynomial_product(
    first: NDArray[np.float64],
    first_error: Fraction,
    second: NDArray[np.float64],
    second_error: Fraction,
    work: list[int],
    *,
    simplex_bounds: dict[tuple[int, ...], Fraction] | None = None,
    reference_radius: Fraction | None = None,
) -> tuple[NDArray[np.float64], Fraction]:
    """Dense coefficient product with a rigorous coefficient l1 error enclosure."""
    from fractions import Fraction

    entries = [(index, float(value)) for index, value in np.ndenumerate(second) if value]
    products = first.size * len(entries)
    if work[0] + products > work[1]:
        raise CellGeometryTransitionError(
            "resource_limit",
            "Embedded measure coefficient-product budget exhausted.",
            measured=work[0] + products,
            limit=work[1],
        )
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(products)
    work[0] += products
    result = np.zeros(
        tuple(a + b - 1 for a, b in zip(first.shape, second.shape, strict=True)),
        dtype=np.float64,
    )
    for (i, j), value in entries:
        result[i : i + first.shape[0], j : j + first.shape[1]] += value * first
    first_norm, second_norm = (
        _dense_domain_polynomial_norm(value, simplex_bounds, reference_radius)
        for value in (first, second)
    )
    rounding = Fraction(float(np.finfo(np.float64).eps)) * (2 * len(entries) + 1)
    if rounding >= 1:
        raise ValueError(
            "Embedded measure coefficient error bound exceeds arithmetic range."
        )
    error = (
        first_error * second_norm
        + second_error * first_norm
        + first_error * second_error
        + rounding / (1 - rounding) * first_norm * second_norm
        + products * Fraction(float(np.nextafter(0.0, math.inf)))
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("Embedded measure polynomial arithmetic overflowed.")
    return result, error


def _dense_polynomial_integral_fraction(
    values: NDArray[np.float64],
    kind: str,
    weight: Polynomial | None = None,
) -> Fraction:
    from fractions import Fraction

    from ._coordinate_enclosure import multiply

    polynomial: Polynomial = {
        index: Fraction(float(value)) for index, value in np.ndenumerate(values) if value
    }
    return _mapped_polynomial_integral_fraction(
        polynomial if weight is None else multiply(polynomial, weight), kind
    )


def _validate_embedded_denominators(
    coordinates: tuple[Expression, ...], cell_kind: str, /
) -> None:
    from ._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        evaluate,
        expression_reference_evaluate,
    )

    domain = "simplex" if cell_kind == "triangle" else "box"
    denominators: list[tuple[Polynomial, RationalPolynomial]] = []
    budget = _COORDINATE_BUDGET.get()
    for coordinate in coordinates:
        if not isinstance(coordinate, RationalPolynomial):
            continue
        for denominator, _ in denominators:
            if budget is not None:
                budget.reserve(len(coordinate.denominator) + len(denominator))
            if len(coordinate.denominator) != len(denominator):
                continue
            pivot = next(iter(denominator))
            if pivot not in coordinate.denominator:
                continue
            ratio = coordinate.denominator[pivot] / denominator[pivot]
            if all(
                coordinate.denominator.get(index) == ratio * value
                for index, value in denominator.items()
            ):
                break
        else:
            denominators.append((coordinate.denominator, coordinate))
    for denominator, coordinate in denominators:
        for vertex in reference_cell_topology(cell_kind).vertices:
            point = tuple(Fraction(float(value)) for value in vertex)
            if not evaluate(denominator, point):
                expression_reference_evaluate(coordinate, point, domain)


def _embedded_squared_density(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    *,
    prepared_coordinates: tuple[Expression, ...] | None = None,
    denominators_certified: bool = False,
) -> Expression:
    from ._coordinate_enclosure import expression_sum

    element = _require_scalar_coordinate_element(element, "Embedded squared density")
    coordinates = (
        _mapped_coordinate_expressions(element, local)
        if prepared_coordinates is None
        else prepared_coordinates
    )
    if (
        element.topological_dimension != 2
        or len(coordinates) != 3
        or element.cell_kind not in ("triangle", "quadrilateral")
    ):
        raise ValueError(
            "Enclosed embedded integration requires canonical triangle/quad maps into three dimensions."
        )
    if not denominators_certified:
        _validate_embedded_denominators(coordinates, element.cell_kind)
    first = tuple(expression_derivative(value, 0) for value in coordinates)
    second = tuple(expression_derivative(value, 1) for value in coordinates)
    cross = tuple(
        expression_add(
            expression_multiply(first[i], second[j]),
            expression_scale(expression_multiply(first[j], second[i]), -1),
        )
        for i, j in ((1, 2), (2, 0), (0, 1))
    )
    return expression_sum(tuple(expression_multiply(value, value) for value in cross))


def _compress_dense_interval(
    values: NDArray[np.float64],
    publication_error: Fraction,
    omitted_budget: Fraction,
    /,
    *,
    simplex_bounds: dict[tuple[int, ...], Fraction] | None = None,
    reference_radius: Fraction | None = None,
) -> tuple[NDArray[np.float64], Fraction]:
    """Discard coherent degree tails only under an exact uniform-error bound."""
    degree = sum(size - 1 for size in values.shape)
    if degree <= 4 or omitted_budget <= 0:
        return values, publication_error
    by_degree = [Fraction(0) for _ in range(degree + 1)]
    for index, value in np.ndenumerate(values):
        if value:
            bound = (
                Fraction(1)
                if simplex_bounds is None
                else _monomial_uniform_bound(index, simplex_bounds, reference_radius)
            )
            by_degree[sum(index)] += abs(Fraction(float(value))) * bound
    discarded, retained_degree = Fraction(0), degree
    for candidate in range(degree, 4, -1):
        if discarded + by_degree[candidate] > omitted_budget:
            break
        discarded += by_degree[candidate]
        retained_degree = candidate - 1
    if retained_degree == degree:
        return values, publication_error
    shape = tuple(min(size, retained_degree + 1) for size in values.shape)
    compressed = np.array(values[: shape[0], : shape[1]], dtype=np.float64, copy=True)
    for index in np.ndindex(compressed.shape):
        if sum(index) > retained_degree:
            compressed[index] = 0.0
    return compressed, publication_error + discarded


def _compressed_dense_polynomial_interval(
    polynomial: Polynomial,
    omitted_budget: Fraction,
    /,
    *,
    simplex_bounds: dict[tuple[int, ...], Fraction] | None = None,
    reference_radius: Fraction | None = None,
) -> tuple[NDArray[np.float64], Fraction]:
    values, error = _dense_polynomial_interval(
        polynomial, monomial_bounds=simplex_bounds, reference_radius=reference_radius
    )
    return _compress_dense_interval(
        values,
        error,
        omitted_budget,
        simplex_bounds=simplex_bounds,
        reference_radius=reference_radius,
    )


def _sqrt_polynomial_integral_piece(
    polynomial: Polynomial,
    kind: str,
    absolute_tolerance: Fraction | float,
    relative_tolerance: float,
    maximum_terms: int,
    work: list[int],
    *,
    weight: Polynomial | None = None,
) -> tuple[Fraction, Fraction, Fraction] | None:
    """Validated binomial approximation with enclosed coefficient compression."""
    from fractions import Fraction

    from ._coordinate_enclosure import bernstein_coefficients, constant

    domain = "simplex" if kind == "triangle" else "box"
    coefficients = bernstein_coefficients(polynomial, domain, 2)
    lower, upper = min(coefficients), max(coefficients)
    if upper <= 0:
        raise ValueError("Embedded coordinate source has no positive Gram measure.")
    if lower <= 0:
        return None
    center = (lower + upper) / 2
    ratio = add(scale(polynomial, 1 / center), constant(-1, 2))
    # Bernstein positivity gives rho < 1. High-degree conditioning is decided
    # by its propagated arithmetic enclosure, not a coefficient-norm heuristic.
    high_degree = max((sum(index) for index in ratio), default=0) > 4
    simplex_bounds: dict[tuple[int, ...], Fraction] | None = (
        {} if high_degree and kind == "triangle" else None
    )
    if not high_degree and sum(
        (abs(value) for value in ratio.values()), Fraction(0)
    ) > Fraction(1, 2):
        return None
    rho = (upper - lower) / (upper + lower)
    root_lower, root_upper = _fraction_sqrt_interval(center)
    root_midpoint, root_error = (
        (root_lower + root_upper) / 2,
        (root_upper - root_lower) / 2,
    )
    reference_volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    weight_coefficients = (
        (Fraction(1),) if weight is None else bernstein_coefficients(weight, domain, 2)
    )
    weight_sup = max(abs(value) for value in weight_coefficients)
    weight_integral = (
        reference_volume
        if weight is None
        else _mapped_polynomial_integral_fraction(weight, kind)
    )
    lower_integral = (
        _fraction_sqrt_interval(lower)[0] * abs(weight_integral)
        if min(weight_coefficients) >= 0 or max(weight_coefficients) <= 0
        else Fraction(0)
    )
    tolerance = max(
        Fraction(absolute_tolerance), Fraction(relative_tolerance) * lower_integral
    )
    compression_budget = (
        Fraction(0)
        if weight_sup == 0
        else min(
            tolerance / (16 * root_upper * reference_volume * weight_sup), (1 - rho) / 16
        )
    )
    reference_radius = Fraction(1, 2) if high_degree else None
    local_weight = weight
    if high_degree:
        centered = affine_arguments(np.asarray((0.5, 0.5)), np.eye(2))
        ratio = compose(ratio, centered)
        local_weight = None if weight is None else compose(weight, centered)
        simplex_bounds = {}
    ratio_values, ratio_error = _compressed_dense_polynomial_interval(
        ratio,
        compression_budget,
        simplex_bounds=simplex_bounds,
        reference_radius=reference_radius,
    )
    power_values, power_error = np.ones((1, 1), dtype=np.float64), Fraction(0)
    series_integral, integral_error = Fraction(0), Fraction(0)
    coefficient = Fraction(1)
    moment_cache: dict[tuple[int, int], Fraction] = {}

    def prepared_integral(values: NDArray[np.float64]) -> Fraction:
        from ._coordinate_enclosure import _COORDINATE_BUDGET

        ledger = _COORDINATE_BUDGET.get()
        terms = tuple(
            (index, float(value)) for index, value in np.ndenumerate(values) if value
        )
        if ledger is not None:
            ledger.reserve(len(terms))
        result = Fraction(0)
        weights = {(0, 0): Fraction(1)} if local_weight is None else local_weight
        for index, value in terms:
            moment = moment_cache.get(index)
            if moment is None:
                moment = Fraction(0)
                for power, coefficient_ in weights.items():
                    exponents = tuple(a + b for a, b in zip(index, power, strict=True))
                    # Exact moments of the original domain in coordinates
                    # (u - 1/2, v - 1/2), not a box surrogate for the triangle.
                    a, b = exponents

                    def interval_moment(order: int) -> Fraction:
                        return (
                            Fraction(0)
                            if order % 2
                            else Fraction(1, (order + 1) * 2**order)
                        )

                    functional = (
                        (
                            (-1) ** (b + 1) * interval_moment(a + b + 1)
                            - Fraction(-1, 2) ** (b + 1) * interval_moment(a)
                        )
                        / (b + 1)
                        if kind == "triangle"
                        else interval_moment(a) * interval_moment(b)
                    )
                    moment += coefficient_ * functional
                moment_cache[index] = moment
            result += Fraction(value) * moment
        return result

    for order in range(maximum_terms):
        integral = (
            prepared_integral(power_values)
            if high_degree
            else _dense_polynomial_integral_fraction(power_values, kind, weight)
        )
        series_integral += coefficient * integral
        integral_error += abs(coefficient) * power_error * reference_volume * weight_sup
        next_coefficient = coefficient * (Fraction(1, 2) - order) / (order + 1)
        remainder = (
            abs(next_coefficient)
            * rho ** (order + 1)
            / (1 - rho)
            * reference_volume
            * weight_sup
        )
        error = root_upper * (integral_error + remainder) + root_error * abs(
            series_integral
        )
        if error <= tolerance / 2:
            return root_midpoint * series_integral, error, tolerance
        if high_degree and root_upper * integral_error > tolerance / 2:
            return None
        if order + 1 < maximum_terms:
            power_values, power_error = _dense_polynomial_product(
                power_values,
                power_error,
                ratio_values,
                ratio_error,
                work,
                simplex_bounds=simplex_bounds,
                reference_radius=reference_radius,
            )
            if high_degree and compression_budget:
                power_values, power_error = _compress_dense_interval(
                    power_values,
                    power_error,
                    compression_budget / (maximum_terms * abs(next_coefficient)),
                    simplex_bounds=simplex_bounds,
                    reference_radius=reference_radius,
                )
            coefficient = next_coefficient
    return None


def _embedded_measure_submaps(
    kind: str,
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], ...]:
    half = 0.5 * np.eye(2)
    if kind == "quadrilateral":
        return tuple((half, np.asarray([x, y])) for x in (0.0, 0.5) for y in (0.0, 0.5))
    return (
        (half, np.asarray([0.0, 0.0])),
        (half, np.asarray([0.5, 0.0])),
        (half, np.asarray([0.0, 0.5])),
        (np.asarray([[0.0, -0.5], [0.5, 0.5]]), np.asarray([0.5, 0.0])),
    )


def _sqrt_weighted_integral_pass(
    gram: Polynomial,
    weight: Polynomial,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    subcells: list[int],
) -> tuple[float, float]:
    from contextlib import nullcontext
    from fractions import Fraction

    from ._coordinate_enclosure import _COORDINATE_BUDGET, outward

    ledger = _COORDINATE_BUDGET.get()
    pending = [(gram, weight, Fraction(1))]
    value, error = Fraction(0), Fraction(0)
    while pending:
        piece, local_weight, fraction = pending.pop()
        subcells[0] += 1
        if subcells[0] > subcells[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Embedded measure reference-subdivision budget exhausted.",
                measured=subcells[0],
                limit=subcells[1],
            )
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            integral = _sqrt_polynomial_integral_piece(
                piece, kind, tolerance, 0.0, maximum_terms, work, weight=local_weight
            )
        if integral is None:
            for matrix, offset in _embedded_measure_submaps(kind):
                arguments = affine_arguments(offset, matrix)
                with ledger.temporary_scope() if ledger is not None else nullcontext():
                    child_gram, child_weight = (
                        compose(piece, arguments),
                        compose(local_weight, arguments),
                    )
                if ledger is not None:
                    _certified_sqrt_retain_expressions((child_gram, child_weight))
                pending.append((child_gram, child_weight, fraction / 4))
        else:
            central, bound, _ = integral
            value += fraction * central
            error += fraction * bound
    published = float(value)
    error += abs(Fraction(published) - value)
    return published, (outward(error, math.inf) if error else 0.0)


def _certified_sqrt_exact_polynomial_root(
    value: Polynomial, work: list[int]
) -> Polynomial | None:
    """Extract an exact polynomial square by leading-term division, or decline."""
    from ._coordinate_enclosure import multiply

    if not value:
        return {}
    leading = max(value)
    coefficient = value[leading]
    if coefficient < 0 or any(index % 2 for index in leading):
        return None
    a, b = math.isqrt(coefficient.numerator), math.isqrt(coefficient.denominator)
    if a * a != coefficient.numerator or b * b != coefficient.denominator:
        return None
    root_index = tuple(index // 2 for index in leading)
    root_coefficient = Fraction(a, b)
    root = {root_index: root_coefficient}
    while True:
        proposed = work[0] + len(root) ** 2
        if proposed > work[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Rational Gram square extraction exhausted its work budget.",
                measured=proposed,
                limit=work[1],
            )
        work[0] = proposed
        residual = add(value, scale(multiply(root, root), -1))
        if not residual:
            return root
        residual_index = max(residual)
        next_index = tuple(i - j for i, j in zip(residual_index, root_index, strict=True))
        if any(index < 0 for index in next_index) or next_index >= min(root):
            return None
        root[next_index] = residual[residual_index] / (2 * root_coefficient)


def _certified_sqrt_dense_sum(
    first: NDArray[np.float64],
    first_error: Fraction,
    second: NDArray[np.float64],
    second_error: Fraction,
    work: list[int],
) -> tuple[NDArray[np.float64], Fraction]:
    """Enclose coefficient addition exactly, including cancellation and rounding."""
    shape = tuple(max(a, b) for a, b in zip(first.shape, second.shape, strict=True))
    entries = math.prod(shape)
    proposed = work[0] + entries
    if proposed > work[1]:
        raise CellGeometryTransitionError(
            "resource_limit",
            "Rational Gram coefficient-sum budget exhausted.",
            measured=proposed,
            limit=work[1],
        )
    work[0] = proposed
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        ledger.reserve(entries, 3 * entries * np.dtype(np.float64).itemsize)
    result = np.zeros(shape, dtype=np.float64)
    result[: first.shape[0], : first.shape[1]] = first
    result[: second.shape[0], : second.shape[1]] += second
    if not np.all(np.isfinite(result)):
        raise ValueError("Rational Gram reciprocal coefficient addition overflowed.")
    rounding = Fraction(0)
    for index, value in np.ndenumerate(result):
        a = (
            first[index]
            if all(i < n for i, n in zip(index, first.shape, strict=True))
            else 0.0
        )
        b = (
            second[index]
            if all(i < n for i, n in zip(index, second.shape, strict=True))
            else 0.0
        )
        rounding += abs(Fraction(float(value)) - Fraction(float(a)) - Fraction(float(b)))
    return result, first_error + second_error + rounding


def _certified_sqrt_dense_product(
    first: NDArray[np.float64],
    first_error: Fraction,
    second: NDArray[np.float64],
    second_error: Fraction,
    work: list[int],
) -> tuple[NDArray[np.float64], Fraction]:
    """Reserve dense scratch and coefficient-entry storage before convolution."""
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        shape = tuple(a + b - 1 for a, b in zip(first.shape, second.shape, strict=True))
        ledger.reserve(
            0, math.prod(shape) * np.dtype(np.float64).itemsize + second.size * 256
        )
    return _dense_polynomial_product(first, first_error, second, second_error, work)


def _certified_sqrt_dense_weight_piece(
    polynomial: Polynomial,
    weight: NDArray[np.float64],
    weight_error: Fraction,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
) -> tuple[Fraction, Fraction] | None:
    """Polynomial moments with enclosed dense products, without a Fraction expansion."""
    from ._coordinate_enclosure import bernstein_coefficients, constant

    domain = "simplex" if kind == "triangle" else "box"
    controls = bernstein_coefficients(polynomial, domain, 2)
    lower, upper = min(controls), max(controls)
    if upper <= 0:
        raise ValueError("Embedded coordinate source has no positive Gram measure.")
    if lower <= 0:
        return None
    center = (lower + upper) / 2
    ratio = add(scale(polynomial, 1 / center), constant(-1, 2))
    if sum((abs(value) for value in ratio.values()), Fraction(0)) > Fraction(1, 8):
        return None
    rho = (upper - lower) / (upper + lower)
    root_lower, root_upper = _fraction_sqrt_interval(center)
    root_midpoint, root_error = (
        (root_lower + root_upper) / 2,
        (root_upper - root_lower) / 2,
    )
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    weight_sup = _dense_polynomial_norm(weight) + weight_error
    ratio_values, ratio_error = _dense_polynomial_interval(ratio)
    power_values, power_error = np.ones((1, 1), dtype=np.float64), Fraction(0)
    series_values, series_error = np.zeros((1, 1), dtype=np.float64), Fraction(0)
    coefficient = Fraction(1)
    for order in range(maximum_terms):
        coefficient_values, coefficient_error = _dense_polynomial_interval(
            constant(coefficient, 2)
        )
        term_values, term_error = _certified_sqrt_dense_product(
            power_values, power_error, coefficient_values, coefficient_error, work
        )
        series_values, series_error = _certified_sqrt_dense_sum(
            series_values, series_error, term_values, term_error, work
        )
        next_coefficient = coefficient * (Fraction(1, 2) - order) / (order + 1)
        remainder = abs(next_coefficient) * rho ** (order + 1) / (1 - rho)
        approximation_error = (
            root_upper * (series_error + remainder) * volume * weight_sup
        )
        if approximation_error <= tolerance / 8:
            break
        if order + 1 < maximum_terms:
            power_values, power_error = _certified_sqrt_dense_product(
                power_values, power_error, ratio_values, ratio_error, work
            )
            coefficient = next_coefficient
    else:
        return None
    factor_values, factor_error = _dense_polynomial_interval(constant(root_midpoint, 2))
    density_values, density_error = _certified_sqrt_dense_product(
        series_values,
        series_error + remainder,
        factor_values,
        factor_error + root_error,
        work,
    )
    weighted_values, weighted_error = _certified_sqrt_dense_product(
        density_values, density_error, weight, weight_error, work
    )
    error = weighted_error * volume
    if error > tolerance / 2:
        return None
    return _dense_polynomial_integral_fraction(weighted_values, kind), error


def _certified_sqrt_rational_piece(
    gram: Expression,
    weight: Expression,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
) -> tuple[Fraction, Fraction] | None:
    """Reduce sqrt(N/D)*A/B to sqrt(N*D)*A/(D*B), with an enclosed reciprocal."""
    from ._coordinate_enclosure import (
        bernstein_coefficients,
        constant,
        expression_parts,
        multiply,
    )

    domain = "simplex" if kind == "triangle" else "box"
    numerator, denominator = expression_parts(gram, 2)
    weighted_numerator, weighted_denominator = expression_parts(weight, 2)

    def product(first: Polynomial, second: Polynomial) -> Polynomial:
        proposed = work[0] + len(first) * len(second)
        if proposed > work[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Rational Gram coefficient-product budget exhausted.",
                measured=proposed,
                limit=work[1],
            )
        work[0] = proposed
        return multiply(first, second)

    for divisor in (denominator, weighted_denominator):
        if min(bernstein_coefficients(divisor, domain, 2)) <= 0:
            raise ValueError(
                "Rational Gram integration requires a strictly positive denominator enclosure."
            )
    denominator_root = _certified_sqrt_exact_polynomial_root(denominator, work)
    if denominator_root is not None:
        root_controls = bernstein_coefficients(denominator_root, domain, 2)
        if max(root_controls) < 0:
            denominator_root = scale(denominator_root, -1)
        elif min(root_controls) <= 0:
            denominator_root = None
    polynomial = (
        numerator if denominator_root is not None else product(numerator, denominator)
    )
    divisor = product(
        denominator_root if denominator_root is not None else denominator,
        weighted_denominator,
    )
    if set(polynomial) == {(0, 0)} and set(divisor) == {(0, 0)}:
        root_lower, root_upper = _fraction_sqrt_interval(polynomial[(0, 0)])
        moment = (
            _mapped_polynomial_integral_fraction(weighted_numerator, kind)
            / divisor[(0, 0)]
        )
        return (root_lower + root_upper) * moment / 2, (root_upper - root_lower) * abs(
            moment
        ) / 2
    exponent = 1
    while len(divisor) > 1:
        base = _certified_sqrt_exact_polynomial_root(divisor, work)
        if base is None:
            break
        base_controls = bernstein_coefficients(base, domain, 2)
        if max(base_controls) < 0:
            base = scale(base, -1)
        elif min(base_controls) <= 0:
            break
        divisor, exponent = base, 2 * exponent
    controls = bernstein_coefficients(divisor, domain, 2)
    lower, upper = min(controls), max(controls)
    if lower <= 0:
        return None
    center = (lower + upper) / 2
    ratio = add(constant(1, 2), scale(divisor, -1 / center))
    rho = (upper - lower) / (upper + lower)
    if sum((abs(value) for value in ratio.values()), Fraction(0)) > Fraction(1, 8):
        return None
    density_controls = bernstein_coefficients(polynomial, domain, 2)
    density_lower, density_upper = min(density_controls), max(density_controls)
    if density_upper <= 0:
        raise ValueError("Embedded coordinate source has no positive Gram measure.")
    if density_lower <= 0:
        return None
    root_ratio = add(
        scale(polynomial, 2 / (density_lower + density_upper)), constant(-1, 2)
    )
    if sum((abs(value) for value in root_ratio.values()), Fraction(0)) > Fraction(1, 8):
        return None
    weight_upper = max(
        abs(value) for value in bernstein_coefficients(weighted_numerator, domain, 2)
    )
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    ratio_values, ratio_error = _dense_polynomial_interval(ratio)
    series_values, series_error = np.zeros((1, 1), dtype=np.float64), Fraction(0)
    power_values, power_error = np.ones((1, 1), dtype=np.float64), Fraction(0)
    remainder = Fraction(0)
    coefficient = Fraction(1)
    for order in range(maximum_terms):
        coefficient_values, coefficient_error = _dense_polynomial_interval(
            constant(coefficient, 2)
        )
        term_values, term_error = _certified_sqrt_dense_product(
            power_values, power_error, coefficient_values, coefficient_error, work
        )
        series_values, series_error = _certified_sqrt_dense_sum(
            series_values, series_error, term_values, term_error, work
        )
        next_coefficient = coefficient * Fraction(exponent + order, order + 1)
        tail_ratio = rho * Fraction(exponent + order + 1, order + 2)
        if tail_ratio < 1:
            remainder = (
                _fraction_sqrt_interval(density_upper)[1]
                * weight_upper
                * volume
                * next_coefficient
                * rho ** (order + 1)
                / (center**exponent * (1 - tail_ratio))
            )
            if remainder <= tolerance / 4:
                break
        if order + 1 < maximum_terms:
            power_values, power_error = _certified_sqrt_dense_product(
                power_values, power_error, ratio_values, ratio_error, work
            )
            coefficient = next_coefficient
    else:
        return None
    numerator_values, numerator_error = _dense_polynomial_interval(
        scale(weighted_numerator, 1 / center**exponent)
    )
    weight_values, weight_error = _certified_sqrt_dense_product(
        numerator_values, numerator_error, series_values, series_error, work
    )
    integral = _certified_sqrt_dense_weight_piece(
        polynomial, weight_values, weight_error, kind, tolerance / 2, maximum_terms, work
    )
    if integral is None:
        return None
    value, error = integral
    return value, error + remainder


def _certified_sqrt_retain_expressions(values: tuple[Expression, ...]) -> None:
    """Account for live coefficient data, not released expansion intermediates."""
    from ._coordinate_enclosure import _coefficient_profile, _reserve_polynomial

    parts: tuple[Polynomial, ...] = tuple(
        polynomial
        for value in values
        for polynomial in (
            (value.numerator, value.denominator)
            if isinstance(value, RationalPolynomial)
            else (value,)
        )
    )
    numerator_bits, denominator_bits = _coefficient_profile(parts)
    _reserve_polynomial(
        0, sum(len(part) for part in parts), 2, numerator_bits + denominator_bits
    )


def _certified_sqrt_expression_child(
    gram: Expression,
    weight: Expression,
    arguments: tuple[Polynomial, ...],
    ledger: CoordinateEnclosureBudget,
) -> tuple[Expression, Expression]:
    """Release expansion scratch while retaining the actual queued source data."""
    with ledger.temporary_scope():
        child_gram = expression_compose(gram, arguments)
        child_weight = expression_compose(weight, arguments)
    _certified_sqrt_retain_expressions((child_gram, child_weight))
    return child_gram, child_weight


def _certified_sqrt_expression_pass(
    gram: Expression,
    weight: Expression,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
    subcells: list[int],
) -> tuple[float, float]:
    from ._coordinate_enclosure import _COORDINATE_BUDGET, outward

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError(
            "Certified rational Gram integration requires its active coefficient ledger."
        )
    submaps = (
        _embedded_measure_submaps(kind)
        if kind != "interval"
        else tuple(
            (
                np.asarray(((0.5, 0.0), (0.0, 1.0)), dtype=np.float64),
                np.asarray((offset, 0.0), dtype=np.float64),
            )
            for offset in (0.0, 0.5)
        )
    )
    pending = [(gram, weight, Fraction(1))]
    value, error = Fraction(0), Fraction(0)
    while pending:
        piece, local_weight, fraction = pending.pop()
        subcells[0] += 1
        if subcells[0] > subcells[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Embedded measure reference-subdivision budget exhausted.",
                measured=subcells[0],
                limit=subcells[1],
            )
        with ledger.temporary_scope():
            integral = _certified_sqrt_rational_piece(
                piece,
                local_weight,
                "quadrilateral" if kind == "interval" else kind,
                tolerance,
                maximum_terms,
                work,
            )
        if integral is None:
            for matrix, offset in submaps:
                arguments = affine_arguments(offset, matrix)
                child_gram, child_weight = _certified_sqrt_expression_child(
                    piece, local_weight, arguments, ledger
                )
                pending.append(
                    (
                        child_gram,
                        child_weight,
                        fraction / (2 if kind == "interval" else 4),
                    )
                )
        else:
            central, bound = integral
            value += fraction * central
            error += fraction * bound
    published = float(value)
    error += abs(Fraction(published) - value)
    return published, outward(error, math.inf) if error else 0.0


def _certified_sqrt_cone_transform(
    gram: Expression,
    weight: Expression,
    kind: str,
) -> tuple[Expression, Expression, str]:
    """Integrate a bounded rational apex through its exact measure-preserving cone.

    Coordinate corner values belong to the source-limit owner. Tangent Gram
    may have a bounded directional limit; no value is assigned at that null set.
    """
    from ._coordinate_enclosure import (
        axes,
        constant,
        evaluate,
        expression_bernstein_coefficients,
        multiply,
    )

    if kind != "triangle":
        return gram, weight, kind
    vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    singular = {
        vertex
        for value in (gram, weight)
        if isinstance(value, RationalPolynomial)
        for vertex in vertices
        if not evaluate(value.denominator, vertex)
    }
    if not singular:
        return gram, weight, kind
    if len(singular) != 1:
        raise ValueError(
            "Rational Gram cone integration requires one removable source corner."
        )
    apex = next(iter(singular))
    for value in (gram, weight):
        expression_bernstein_coefficients(value, "simplex", 2)
    base = tuple(vertex for vertex in vertices if vertex != apex)
    u, collapse = axes(2)
    arguments = tuple(
        add(constant(a, 2), multiply(collapse, add(constant(b - a, 2), scale(u, c - b))))
        for a, b, c in zip(apex, base[0], base[1], strict=True)
    )
    return (
        expression_compose(gram, arguments),
        expression_multiply(expression_compose(weight, arguments), collapse),
        "quadrilateral",
    )


def _certified_sqrt_polynomial_integral_state(
    gram: Expression,
    weight: Expression,
    kind: str,
    absolute_tolerance: Fraction | float,
    relative_tolerance: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
) -> tuple[float, float]:
    from fractions import Fraction

    from ._coordinate_enclosure import (
        bernstein_coefficients,
        expression_bernstein_coefficients,
    )

    if (
        kind == "interval"
        or isinstance(gram, RationalPolynomial)
        or isinstance(weight, RationalPolynomial)
    ):
        from ._coordinate_enclosure import _COORDINATE_BUDGET

        if _COORDINATE_BUDGET.get() is None:
            ledger = CoordinateEnclosureBudget(work[1] - work[0], sys.maxsize)
            try:
                with ledger.activate():
                    return _certified_sqrt_polynomial_integral_state(
                        gram,
                        weight,
                        kind,
                        absolute_tolerance,
                        relative_tolerance,
                        maximum_subcells,
                        maximum_terms,
                        work,
                    )
            except CoordinateEnclosureResourceError as error:
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "Rational Gram source coefficient budget exhausted.",
                    measured=error.requested,
                    limit=ledger.maximum_work_units,
                ) from error
        gram, weight, kind = _certified_sqrt_cone_transform(gram, weight, kind)
        domain = "simplex" if kind == "triangle" else "box"
        coefficients = expression_bernstein_coefficients(gram, domain, 2)
        weight_coefficients = expression_bernstein_coefficients(weight, domain, 2)
        upper = max(coefficients)
        if upper <= 0:
            raise ValueError("Embedded coordinate source has no positive Gram measure.")
        volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
        characteristic = (
            _fraction_sqrt_interval(upper)[1]
            * max(abs(x) for x in weight_coefficients)
            * volume
        )
        absolute, relative = Fraction(absolute_tolerance), Fraction(relative_tolerance)
        tolerance = (
            max(absolute, characteristic / 8)
            if relative * characteristic > absolute
            else absolute
        )
        subcells = [0, int(maximum_subcells)]
        while True:
            value, error = _certified_sqrt_expression_pass(
                gram, weight, kind, tolerance, maximum_terms, work, subcells
            )
            lower_magnitude = max(abs(Fraction(value)) - Fraction(error), Fraction(0))
            allowed = max(absolute, relative * lower_magnitude)
            if Fraction(error) <= allowed:
                return value, error
            tolerance = (
                max(absolute, allowed)
                if lower_magnitude
                else max(absolute, tolerance / 4)
            )
    domain = "simplex" if kind == "triangle" else "box"
    gram_coefficients = bernstein_coefficients(gram, domain, 2)
    weight_coefficients = bernstein_coefficients(weight, domain, 2)
    upper = max(gram_coefficients)
    if upper <= 0:
        raise ValueError("Embedded coordinate source has no positive Gram measure.")
    weight_sup = max(abs(value) for value in weight_coefficients)
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    characteristic = _fraction_sqrt_interval(upper)[1] * weight_sup * volume
    lower_integral = Fraction(0)
    lower = min(gram_coefficients)
    if lower > 0 and (min(weight_coefficients) >= 0 or max(weight_coefficients) <= 0):
        lower_integral = _fraction_sqrt_interval(lower)[0] * abs(
            _mapped_polynomial_integral_fraction(weight, kind)
        )
    absolute, relative = Fraction(absolute_tolerance), Fraction(relative_tolerance)
    if relative * characteristic <= absolute:
        tolerance = absolute
    elif lower_integral:
        tolerance = max(absolute, relative * lower_integral)
    else:
        tolerance = max(absolute, characteristic / 8)
    subcells = [0, int(maximum_subcells)]
    while True:
        value, error = _sqrt_weighted_integral_pass(
            gram, weight, kind, tolerance, maximum_terms, work, subcells
        )
        lower_magnitude = max(abs(Fraction(value)) - Fraction(error), Fraction(0))
        allowed = max(absolute, relative * lower_magnitude)
        if Fraction(error) <= allowed:
            return value, error
        tolerance = (
            max(absolute, relative * lower_magnitude)
            if lower_magnitude
            else max(absolute, tolerance / 4)
        )


def _certified_sqrt_polynomial_integral(
    gram: Expression,
    weight: Expression,
    kind: str,
    /,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
) -> tuple[float, float]:
    """Enclose a signed exact source weight times sqrt(Gram).

    Polynomial reduction order is unchanged. Rational expressions use a positive
    Bernstein denominator proof and polynomial reciprocal/binomial moments with
    absolute remainder bounds. Relative acceptance uses the proved magnitude of
    the whole integral, including cancellation; all passes share resource limits.
    """
    from fractions import Fraction

    absolute = _finite_non_negative(absolute_tolerance, "absolute_tolerance")
    relative = _finite_non_negative(relative_tolerance, "relative_tolerance")
    if kind not in ("interval", "triangle", "quadrilateral") or absolute == relative == 0:
        raise ValueError(
            "Weighted Gram integration requires interval/triangle/quad support and a positive error budget."
        )
    if min(maximum_work, maximum_subcells, maximum_binomial_terms) <= 0:
        raise ValueError("Weighted Gram integration requires positive work budgets.")
    if (
        kind != "interval"
        and not isinstance(gram, RationalPolynomial)
        and not isinstance(weight, RationalPolynomial)
    ):
        if any(len(index) != 2 for polynomial in (gram, weight) for index in polynomial):
            raise ValueError(
                "Weighted Gram polynomials require two reference dimensions."
            )
        gram = {index: Fraction(value) for index, value in gram.items() if value}
        weight = {index: Fraction(value) for index, value in weight.items() if value}
        return _certified_sqrt_polynomial_integral_state(
            gram,
            weight,
            kind,
            absolute,
            relative,
            int(maximum_subcells),
            int(maximum_binomial_terms),
            [0, int(maximum_work)],
        )
    from ._coordinate_enclosure import expression_parts, rational_expression

    dimension = 1 if kind == "interval" else 2
    expressions: list[Expression] = []
    for value in (gram, weight):
        numerator, denominator = expression_parts(value, dimension)
        if any(
            len(index) != dimension
            for polynomial in (numerator, denominator)
            for index in polynomial
        ):
            raise ValueError(
                "Weighted Gram expressions have incompatible reference dimensions."
            )
        numerator = {
            index: Fraction(coefficient)
            for index, coefficient in numerator.items()
            if coefficient
        }
        denominator = {
            index: Fraction(coefficient)
            for index, coefficient in denominator.items()
            if coefficient
        }
        if kind == "interval":
            lifted_numerator: Polynomial = {
                (index[0], 0): coefficient for index, coefficient in numerator.items()
            }
            lifted_denominator: Polynomial = {
                (index[0], 0): coefficient for index, coefficient in denominator.items()
            }
            numerator, denominator = lifted_numerator, lifted_denominator
        expressions.append(
            rational_expression(numerator, denominator)
            if isinstance(value, RationalPolynomial)
            else numerator
        )
    gram, weight = expressions
    return _certified_sqrt_polynomial_integral_state(
        gram,
        weight,
        kind,
        absolute,
        relative,
        int(maximum_subcells),
        int(maximum_binomial_terms),
        [0, int(maximum_work)],
    )


class MappedEdgeArcLengthEnclosure(NamedTuple):
    value: float
    error: float
    lower: float
    upper: float


class PreparedMappedEdgeArcLength(NamedTuple):
    gram: Expression

    def integrate(
        self,
        /,
        *,
        absolute_tolerance: float = 1e-10,
        relative_tolerance: float = 1e-10,
        maximum_work: int = 100_000_000,
        maximum_subcells: int = 10000,
        maximum_binomial_terms: int = 32,
    ) -> MappedEdgeArcLengthEnclosure:
        from ._coordinate_enclosure import constant, outward

        value, error = _certified_sqrt_polynomial_integral(
            self.gram,
            constant(1, 1),
            "interval",
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            maximum_work=maximum_work,
            maximum_subcells=maximum_subcells,
            maximum_binomial_terms=maximum_binomial_terms,
        )
        lower = (
            value if error == 0 else outward(Fraction(value) - Fraction(error), -math.inf)
        )
        upper = (
            value if error == 0 else outward(Fraction(value) + Fraction(error), math.inf)
        )
        if lower <= 0:
            raise ValueError(
                "Mapped edge arc length has no strictly positive certified enclosure."
            )
        return MappedEdgeArcLengthEnclosure(value, error, lower, upper)


def _prepare_mapped_edge_arc_length(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    start: tuple[Fraction, ...],
    end: tuple[Fraction, ...],
    /,
) -> PreparedMappedEdgeArcLength:
    """Restrict the authoritative coordinate expressions to the complete edge."""
    from ._coordinate_enclosure import axes, constant, expression_sum

    element = _require_scalar_coordinate_element(element, "Mapped edge arc length")
    dimension = element.topological_dimension
    if len(start) != dimension or len(end) != dimension or start == end:
        raise ValueError(
            "Mapped edge endpoints must be distinct reference points of the source dimension."
        )
    variable = axes(1)[0]
    arguments = tuple(
        add(constant(Fraction(a), 1), scale(variable, Fraction(b) - Fraction(a)))
        for a, b in zip(start, end, strict=True)
    )
    edge = tuple(
        expression_compose(value, arguments)
        for value in _mapped_coordinate_expressions(element, local)
    )
    velocity = tuple(expression_derivative(value, 0) for value in edge)
    gram = expression_sum(tuple(expression_multiply(value, value) for value in velocity))
    return PreparedMappedEdgeArcLength(gram)


def _certified_sqrt_dyadic_interval(
    value: Fraction, work: list[int]
) -> tuple[Fraction, Fraction]:
    """Prove a 128-bit dyadic radical enclosure with integer arithmetic."""
    if value <= 0:
        raise ValueError("A prepared Gram radical requires a positive squared scale.")
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    proposed = work[0] + 1
    if proposed > work[1]:
        raise CellGeometryTransitionError(
            "resource_limit",
            "Prepared Gram radical work exhausted.",
            measured=proposed,
            limit=work[1],
        )
    ledger = _COORDINATE_BUDGET.get()
    if ledger is not None:
        bits = value.numerator.bit_length() + value.denominator.bit_length() + 512
        ledger.reserve(1, 256 + bits)
    work[0] = proposed
    denominator = 1 << 128
    scaled = value.numerator << 256
    numerator = math.isqrt(scaled // value.denominator)
    lower = Fraction(numerator, denominator)
    return (
        lower,
        lower
        if numerator * numerator * value.denominator == scaled
        else Fraction(numerator + 1, denominator),
    )


def _certified_sqrt_exact_product(
    first: Polynomial, second: Polynomial, work: list[int]
) -> Polynomial:
    from ._coordinate_enclosure import multiply

    proposed = work[0] + len(first) * len(second)
    if proposed > work[1]:
        raise CellGeometryTransitionError(
            "resource_limit",
            "Prepared Gram polynomial product work exhausted.",
            measured=proposed,
            limit=work[1],
        )
    work[0] = proposed
    return multiply(first, second)


def _certified_sqrt_denominator_power(
    divisor: Polynomial,
    domain: str,
    work: list[int],
) -> tuple[Polynomial, int]:
    from ._coordinate_enclosure import bernstein_coefficients

    exponent = 1
    while len(divisor) > 1:
        base = _certified_sqrt_exact_polynomial_root(divisor, work)
        if base is None:
            break
        controls = bernstein_coefficients(base, domain, 2)
        if max(controls) < 0:
            base = scale(base, -1)
        elif min(controls) <= 0:
            break
        divisor, exponent = base, 2 * exponent
    return divisor, exponent


def _certified_sqrt_exact_reciprocal(
    base: Polynomial,
    exponent: int,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
) -> tuple[Polynomial, Fraction] | None:
    from ._coordinate_enclosure import bernstein_coefficients, constant

    if exponent == 0:
        return constant(1, 2), Fraction(0)
    domain = "simplex" if kind == "triangle" else "box"
    controls = bernstein_coefficients(base, domain, 2)
    lower, upper = min(controls), max(controls)
    if lower <= 0:
        raise ValueError("Prepared Gram reciprocal requires a positive denominator.")
    center = (lower + upper) / 2
    rho = (upper - lower) / (upper + lower)
    if rho > Fraction(1, 8):
        return None
    ratio = add(constant(1, 2), scale(base, -1 / center))
    series: Polynomial = {}
    term = constant(1, 2)
    coefficient = Fraction(1)
    for order in range(maximum_terms):
        series = add(series, scale(term, coefficient / center**exponent))
        next_coefficient = coefficient * Fraction(exponent + order, order + 1)
        tail_ratio = rho * Fraction(exponent + order + 1, order + 2)
        if tail_ratio < 1:
            error = (
                next_coefficient
                * rho ** (order + 1)
                / (center**exponent * (1 - tail_ratio))
            )
            if error <= tolerance:
                return series, error
        if order + 1 < maximum_terms:
            term = _certified_sqrt_exact_product(term, ratio, work)
            coefficient = next_coefficient
    return None


def _certified_sqrt_exact_density_piece(
    gram: Expression,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
) -> tuple[Polynomial, Fraction] | None:
    """Exact polynomial approximation of a positive rational radical, with a uniform tail."""
    from ._coordinate_enclosure import bernstein_coefficients, constant, expression_parts

    domain = "simplex" if kind == "triangle" else "box"
    numerator, divisor = expression_parts(gram, 2)
    if min(bernstein_coefficients(divisor, domain, 2)) <= 0:
        raise ValueError("Prepared Gram approximation requires a positive denominator.")
    root = _certified_sqrt_exact_polynomial_root(divisor, work)
    if root is None:
        numerator = _certified_sqrt_exact_product(numerator, divisor, work)
    else:
        root_controls = bernstein_coefficients(root, domain, 2)
        if max(root_controls) < 0:
            root = scale(root, -1)
        elif min(root_controls) <= 0:
            return None
        divisor = root
    base, exponent = _certified_sqrt_denominator_power(divisor, domain, work)
    controls = bernstein_coefficients(base, domain, 2)
    base_lower, base_upper = min(controls), max(controls)
    if base_lower <= 0 or (base_upper - base_lower) / (
        base_upper + base_lower
    ) > Fraction(1, 8):
        return None
    square = _certified_sqrt_exact_product(base, base, work)
    leading = max(numerator)
    factor = (
        numerator[leading] / square[leading] if leading == max(square) else Fraction(0)
    )
    residual = add(numerator, scale(square, -factor))
    rho = (
        max(abs(value) for value in bernstein_coefficients(residual, domain, 2))
        / (factor * base_lower**2)
        if factor > 0
        else Fraction(1)
    )
    if rho <= Fraction(1, 8):
        ratio = scale(residual, 1 / factor)
        initial_exponent, exponent_step = exponent - 1, 2
    else:
        values = bernstein_coefficients(numerator, domain, 2)
        lower, upper = min(values), max(values)
        if lower <= 0:
            return None
        factor = (lower + upper) / 2
        rho = (upper - lower) / (upper + lower)
        if rho > Fraction(1, 8):
            return None
        ratio = add(scale(numerator, 1 / factor), constant(-1, 2))
        initial_exponent, exponent_step = exponent, 0
    root_lower, root_upper = _certified_sqrt_dyadic_interval(factor, work)
    root_midpoint, root_error = (
        (root_lower + root_upper) / 2,
        (root_upper - root_lower) / 2,
    )
    series: Polynomial = {}
    power = constant(1, 2)
    coefficient, reciprocal_error = Fraction(1), Fraction(0)
    for order in range(maximum_terms):
        power_bound = max(
            abs(value) for value in bernstein_coefficients(power, domain, 2)
        )
        reciprocal_goal = (
            tolerance / (8 * maximum_terms * root_upper * abs(coefficient) * power_bound)
            if power_bound
            else tolerance
        )
        reciprocal = _certified_sqrt_exact_reciprocal(
            base,
            initial_exponent + exponent_step * order,
            kind,
            reciprocal_goal,
            maximum_terms,
            work,
        )
        if reciprocal is None:
            return None
        polynomial, error = reciprocal
        series = add(
            series,
            scale(_certified_sqrt_exact_product(power, polynomial, work), coefficient),
        )
        reciprocal_error += abs(coefficient) * power_bound * error
        next_coefficient = coefficient * (Fraction(1, 2) - order) / (order + 1)
        tail = (
            abs(next_coefficient)
            * rho ** (order + 1)
            / ((1 - rho) * base_lower**initial_exponent)
        )
        series_bound = (
            1 + rho / (2 * (1 - rho))
        ) / base_lower**initial_exponent + reciprocal_error
        error = root_upper * (reciprocal_error + tail) + root_error * series_bound
        if error <= tolerance / 2:
            return scale(series, root_midpoint), error
        if order + 1 < maximum_terms:
            power = _certified_sqrt_exact_product(power, ratio, work)
            coefficient = next_coefficient
    return None


class _PreparedSqrtIntegralPiece(NamedTuple):
    arguments: tuple[Polynomial, ...]
    polynomial: Polynomial
    error: Fraction
    fraction: Fraction
    kind: str
    moments: dict[tuple[int, ...], Fraction]

    def moment(self, index: tuple[int, ...], work: list[int]) -> Fraction:
        if index in self.moments:
            return self.moments[index]
        from ._coordinate_enclosure import _coefficient_profile, _reserve_polynomial

        proposed = work[0] + len(self.polynomial)
        if proposed > work[1]:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Prepared Gram moment work exhausted.",
                measured=proposed,
                limit=work[1],
            )
        numerator_bits, denominator_bits = _coefficient_profile((self.polynomial,))
        degree = (
            sum(
                max((term[axis] for term in self.polynomial), default=0) + index[axis]
                for axis in range(2)
            )
            + 2
        )
        _reserve_polynomial(
            len(self.polynomial),
            1,
            2,
            numerator_bits
            + denominator_bits
            + degree * max(degree, 1).bit_length()
            + len(self.polynomial).bit_length(),
        )
        work[0] = proposed
        total = Fraction(0)
        for exponent, coefficient in self.polynomial.items():
            first, second = exponent[0] + index[0], exponent[1] + index[1]
            moment = (
                Fraction(
                    math.factorial(first) * math.factorial(second),
                    math.factorial(first + second + 2),
                )
                if self.kind == "triangle"
                else Fraction(1, (first + 1) * (second + 1))
            )
            total += coefficient * moment
        self.moments[index] = total
        return total


class _PreparedMappedSqrtIntegral(NamedTuple):
    pieces: tuple[_PreparedSqrtIntegralPiece, ...]
    work: list[int]
    ledger: CoordinateEnclosureBudget

    def integral(self, weight: Polynomial) -> tuple[float, float]:
        from ._coordinate_enclosure import bernstein_coefficients, outward

        value, error = Fraction(0), Fraction(0)
        with self.ledger.activate():
            for piece in self.pieces:
                local_weight = compose(weight, piece.arguments)
                coefficients = bernstein_coefficients(
                    local_weight, "simplex" if piece.kind == "triangle" else "box", 2
                )
                central = sum(
                    (
                        coefficient * piece.moment(index, self.work)
                        for index, coefficient in local_weight.items()
                    ),
                    Fraction(0),
                )
                volume = Fraction(1, 2) if piece.kind == "triangle" else Fraction(1)
                value += piece.fraction * central
                error += (
                    piece.fraction
                    * volume
                    * piece.error
                    * max(abs(coefficient) for coefficient in coefficients)
                )
        published = float(value)
        error += abs(Fraction(published) - value)
        return published, outward(error, math.inf) if error else 0.0


def _certified_sqrt_exact_weighted_density_piece(
    gram: Expression,
    denominator: Polynomial,
    exponent: int,
    kind: str,
    tolerance: Fraction,
    maximum_terms: int,
    work: list[int],
) -> tuple[Polynomial, Fraction] | None:
    """Keep a proved positive reciprocal factor independent of the Gram radical."""
    if exponent == 0:
        return _certified_sqrt_exact_density_piece(
            gram, kind, tolerance, maximum_terms, work
        )
    from ._coordinate_enclosure import bernstein_coefficients

    domain = "simplex" if kind == "triangle" else "box"
    controls = bernstein_coefficients(denominator, domain, 2)
    lower, upper = min(controls), max(controls)
    if lower <= 0:
        raise ValueError("Prepared weighted density requires a positive denominator.")
    if (upper - lower) / (upper + lower) > Fraction(1, 8):
        return None
    density = _certified_sqrt_exact_density_piece(
        gram, kind, tolerance * lower**exponent / 4, maximum_terms, work
    )
    if density is None:
        return None
    polynomial, density_error = density
    magnitude = max(abs(value) for value in bernstein_coefficients(polynomial, domain, 2))
    reciprocal = _certified_sqrt_exact_reciprocal(
        denominator,
        exponent,
        kind,
        tolerance / (4 * max(magnitude, Fraction(1))),
        maximum_terms,
        work,
    )
    if reciprocal is None:
        return None
    factor, reciprocal_error = reciprocal
    result = _certified_sqrt_exact_product(polynomial, factor, work)
    return result, density_error / lower**exponent + magnitude * reciprocal_error


def _certified_sqrt_prepare_expression_integral(
    gram: Expression,
    kind: str,
    uniform_tolerance: Fraction,
    work: list[int],
    maximum_subcells: int = 10000,
    maximum_terms: int = 32,
    *,
    jacobian: Polynomial | None = None,
    arguments: tuple[Polynomial, ...] | None = None,
    denominator: Polynomial | None = None,
    denominator_exponent: int = 0,
) -> _PreparedMappedSqrtIntegral:
    """Prepare exact moment functionals for an already-owned reference density."""
    from ._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        axes,
        bernstein_coefficients,
        constant,
    )

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(work[1] - work[0], sys.maxsize)
    if uniform_tolerance <= 0 or maximum_subcells <= 0 or maximum_terms <= 0:
        raise ValueError(
            "Prepared density requires positive precision and resource limits."
        )
    pieces: list[_PreparedSqrtIntegralPiece] = []
    with ledger.activate():
        pending = [
            (
                gram,
                constant(1, 2) if jacobian is None else jacobian,
                axes(2) if arguments is None else arguments,
                Fraction(1),
                constant(1, 2) if denominator is None else denominator,
            )
        ]
        visited = 0
        while pending:
            local_gram, local_jacobian, local_arguments, fraction, local_denominator = (
                pending.pop()
            )
            visited += 1
            if visited > maximum_subcells:
                raise CellGeometryTransitionError(
                    "resource_limit",
                    "Prepared Gram reference subdivision exhausted.",
                    measured=visited,
                    limit=maximum_subcells,
                )
            domain = "simplex" if kind == "triangle" else "box"
            jacobian_sup = max(
                abs(value) for value in bernstein_coefficients(local_jacobian, domain, 2)
            )
            with ledger.temporary_scope():
                approximation = _certified_sqrt_exact_weighted_density_piece(
                    local_gram,
                    local_denominator,
                    denominator_exponent,
                    kind,
                    uniform_tolerance / max(jacobian_sup, Fraction(1)),
                    maximum_terms,
                    work,
                )
                if approximation is not None:
                    polynomial, error = approximation
                    polynomial = _certified_sqrt_exact_product(
                        polynomial, local_jacobian, work
                    )
            if approximation is not None:
                _certified_sqrt_retain_expressions((polynomial,))
                pieces.append(
                    _PreparedSqrtIntegralPiece(
                        local_arguments,
                        polynomial,
                        error * jacobian_sup,
                        fraction,
                        kind,
                        {},
                    )
                )
            else:
                for matrix, offset in _embedded_measure_submaps(kind):
                    child_arguments = affine_arguments(offset, matrix)
                    child_gram, child_jacobian = _certified_sqrt_expression_child(
                        local_gram, local_jacobian, child_arguments, ledger
                    )
                    if isinstance(child_jacobian, RationalPolynomial):
                        raise ValueError(
                            "Prepared reference Jacobian lost its exact polynomial expression."
                        )
                    mapped_arguments = tuple(
                        compose(value, child_arguments) for value in local_arguments
                    )
                    child_denominator = (
                        compose(local_denominator, child_arguments)
                        if denominator_exponent
                        else local_denominator
                    )
                    pending.append(
                        (
                            child_gram,
                            child_jacobian,
                            mapped_arguments,
                            fraction / 4,
                            child_denominator,
                        )
                    )
    return _PreparedMappedSqrtIntegral(tuple(pieces), work, ledger)


def _certified_sqrt_prepare_mapped_integral(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    uniform_tolerance: Fraction,
    work: list[int],
    maximum_subcells: int = 10000,
    maximum_terms: int = 32,
) -> _PreparedMappedSqrtIntegral:
    """Prepare one actual density for many exact weighted physical moment functionals."""
    from ._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        axes,
        bernstein_coefficients,
        constant,
        coordinate_reference_chain,
        derivative,
        determinant,
        multiply,
    )

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(work[1] - work[0], sys.maxsize)
    with ledger.activate():
        gram = _embedded_squared_density(element, local)
        jacobian = constant(1, 2)
        if isinstance(gram, RationalPolynomial) and isinstance(
            element,
            (
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
            ),
        ):
            root, arguments = coordinate_reference_chain(element)
            # Rational reference pullbacks remain in the actual Gram; extracting
            # their Jacobian into the polynomial moment path would lose its denominator.
            polynomial_arguments = tuple(
                value for value in arguments if not isinstance(value, RationalPolynomial)
            )
            if root.topological_dimension == 2 and len(polynomial_arguments) == len(
                arguments
            ):
                jacobian = determinant(
                    tuple(
                        tuple(derivative(value, axis) for axis in range(2))
                        for value in polynomial_arguments
                    )
                )
                controls = bernstein_coefficients(
                    jacobian, "simplex" if element.cell_kind == "triangle" else "box", 2
                )
                if max(controls) < 0:
                    jacobian = scale(jacobian, -1)
                elif min(controls) <= 0:
                    raise ValueError(
                        "Prepared embedded density requires an oriented exact reference map."
                    )
                gram = expression_compose(
                    _embedded_squared_density(root, local), arguments
                )
        kind = element.cell_kind
        arguments = axes(2)
        if kind == "triangle" and any(
            isinstance(value, RationalPolynomial) for value in (gram, jacobian)
        ):
            from ._coordinate_enclosure import evaluate

            vertices = (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            singular = {
                vertex
                for value in (gram, jacobian)
                if isinstance(value, RationalPolynomial)
                for vertex in vertices
                if not evaluate(value.denominator, vertex)
            }
            if singular:
                apex = next(iter(singular))
                base = tuple(vertex for vertex in vertices if vertex != apex)
                u, collapse = axes(2)
                arguments = tuple(
                    add(
                        constant(a, 2),
                        multiply(collapse, add(constant(b - a, 2), scale(u, c - b))),
                    )
                    for a, b, c in zip(apex, base[0], base[1], strict=True)
                )
                gram, jacobian, kind = _certified_sqrt_cone_transform(
                    gram, jacobian, kind
                )
                if isinstance(jacobian, RationalPolynomial):
                    raise ValueError(
                        "Prepared embedded reference measure must be polynomial after its exact cone chart."
                    )
        return _certified_sqrt_prepare_expression_integral(
            gram,
            kind,
            uniform_tolerance,
            work,
            maximum_subcells,
            maximum_terms,
            jacobian=jacobian,
            arguments=arguments,
        )


def _certified_sqrt_mapped_integral_state(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    weight: Expression,
    absolute_tolerance: Fraction | float,
    relative_tolerance: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
) -> tuple[float, float]:
    """Keep exact pullback determinants outside the root of a rational Gram."""
    from ._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        bernstein_coefficients,
        coordinate_reference_chain,
        derivative,
        determinant,
        expression_compose,
        expression_multiply,
    )

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        ledger = CoordinateEnclosureBudget(work[1] - work[0], sys.maxsize)
        try:
            with ledger.activate():
                return _certified_sqrt_mapped_integral_state(
                    element,
                    local,
                    weight,
                    absolute_tolerance,
                    relative_tolerance,
                    maximum_subcells,
                    maximum_terms,
                    work,
                )
        except CoordinateEnclosureResourceError as error:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Mapped Gram source preparation exhausted its coefficient budget.",
                measured=error.requested,
                limit=ledger.maximum_work_units,
            ) from error
    with ledger.temporary_scope():
        gram = _embedded_squared_density(element, local)
        if isinstance(gram, RationalPolynomial) and isinstance(
            element,
            (
                RestrictedCellGeometryElement,
                PolynomialComposedCellGeometryElement,
                RationalComposedCellGeometryElement,
            ),
        ):
            root, arguments = coordinate_reference_chain(element)
            # Keep a rational reference Jacobian inside the actual Gram expression.
            polynomial_arguments = tuple(
                value for value in arguments if not isinstance(value, RationalPolynomial)
            )
            if root.topological_dimension == 2 and len(polynomial_arguments) == len(
                arguments
            ):
                jacobian = determinant(
                    tuple(
                        tuple(derivative(value, axis) for axis in range(2))
                        for value in polynomial_arguments
                    )
                )
                domain = "simplex" if element.cell_kind == "triangle" else "box"
                controls = bernstein_coefficients(jacobian, domain, 2)
                if max(controls) < 0:
                    jacobian = scale(jacobian, -1)
                elif min(controls) <= 0:
                    raise ValueError(
                        "Embedded rational pullback requires a sign-definite source-reference Jacobian."
                    )
                gram = expression_compose(
                    _embedded_squared_density(root, local), arguments
                )
                weight = expression_multiply(weight, jacobian)
        return _certified_sqrt_polynomial_integral_state(
            gram,
            weight,
            element.cell_kind,
            absolute_tolerance,
            relative_tolerance,
            maximum_subcells,
            maximum_terms,
            work,
        )


def _certified_embedded_cell_measure(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    absolute_tolerance: Fraction,
    relative_tolerance: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
) -> tuple[float, float]:
    from ._coordinate_enclosure import constant

    element = _require_scalar_coordinate_element(element, "Certified embedded measure")
    return _certified_sqrt_mapped_integral_state(
        element,
        local,
        constant(1, 2),
        absolute_tolerance,
        relative_tolerance,
        maximum_subcells,
        maximum_terms,
        work,
    )


def _reuse_embedded_polynomial_area(
    gram: Polynomial,
    kind: str,
    references: list[tuple[Polynomial, Fraction, float, float]],
    absolute: Fraction,
    relative: float,
    maximum_subcells: int,
    maximum_terms: int,
    work: list[int],
) -> tuple[tuple[float, float], tuple[Polynomial, Fraction, float, float] | None]:
    """Reuse an area enclosure only after proving the actual Gram difference."""
    from ._coordinate_enclosure import bernstein_coefficients, constant, outward

    if all(not any(index) for index in gram):
        # Constant Grams have their own direct certified square-root integral.
        # Comparing distinct scalar areas against every earlier cell is quadratic
        # work and cannot improve that proof.
        return _certified_sqrt_polynomial_integral_state(
            gram,
            constant(1, 2),
            kind,
            absolute,
            relative,
            maximum_subcells,
            maximum_terms,
            work,
        ), None
    domain = "simplex" if kind == "triangle" else "box"
    volume = Fraction(1, 2) if kind == "triangle" else Fraction(1)
    for reference, lower, value, error in reversed(references):
        if gram == reference:
            allowed = max(
                absolute,
                Fraction(relative) * max(Fraction(value) - Fraction(error), Fraction(0)),
            )
            if Fraction(error) <= allowed:
                return (value, error), None
        difference = add(gram, scale(reference, -1))
        controls = bernstein_coefficients(difference, domain, 2)
        delta = max(abs(coefficient) for coefficient in controls)
        if delta >= lower:
            continue
        denominator = (
            _fraction_sqrt_interval(lower)[0] + _fraction_sqrt_interval(lower - delta)[0]
        )
        if denominator <= 0:
            continue
        # Nonnegative Bernstein bases have equal reference integrals. Their
        # absolute-control envelope encloses integral(abs(actual Gram delta)).
        delta_integral = (
            volume
            * sum((abs(coefficient) for coefficient in controls), Fraction(0))
            / len(controls)
        )
        bound = Fraction(error) + delta_integral / denominator
        published = outward(bound, math.inf) if bound else 0.0
        allowed = max(
            absolute,
            Fraction(relative) * max(Fraction(value) - Fraction(published), Fraction(0)),
        )
        if Fraction(published) <= allowed:
            return (value, published), None
    value, error = _certified_sqrt_polynomial_integral_state(
        gram,
        constant(1, 2),
        kind,
        absolute,
        relative,
        maximum_subcells,
        maximum_terms,
        work,
    )
    lower = min(bernstein_coefficients(gram, domain, 2))
    return (value, error), ((gram, lower, value, error) if lower > 0 else None)


def _exact_affine_simplex_measure_from_corners(
    corners: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[float, float, bool]:
    """Exact affine simplex corners to a bounded binary64 measure."""
    from ._coordinate_enclosure import _COORDINATE_BUDGET, _reserve_polynomial, outward

    dimension = len(corners) - 1
    ambient = len(corners[0])
    embedded_triangle = dimension == 2 and ambient == 3
    if dimension not in (1, 2, 3) or any(len(point) != ambient for point in corners):
        raise ValueError("Exact affine simplex measure requires aligned corners.")
    if ambient != dimension and not embedded_triangle:
        raise ValueError(
            "Exact affine simplex measure supports full-dimensional cells and embedded triangles."
        )
    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError("Exact affine measure lost its coefficient ledger.")
    ledger.reserve(sum(len(point) for point in corners))
    bits = max(
        abs(value.numerator).bit_length() + value.denominator.bit_length()
        for point in corners
        for value in point
    )
    if embedded_triangle:
        _reserve_polynomial(20, 12, 0, 4 * bits + 4)
        first = tuple(corners[1][axis] - corners[0][axis] for axis in range(ambient))
        second = tuple(corners[2][axis] - corners[0][axis] for axis in range(ambient))
        cross = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        squared = sum((value * value for value in cross), Fraction(0))
        if squared <= 0:
            raise ValueError(
                "Mapped integration requires a certified positive embedded Jacobian."
            )
        lower_root, upper_root = _fraction_sqrt_interval(
            squared, coordinate_budget=ledger
        )
        lower_measure, upper_measure = lower_root / 2, upper_root / 2
        value = math.sqrt(float(squared)) / 2
        return (
            value,
            max(
                value - outward(lower_measure, -math.inf),
                outward(upper_measure, math.inf) - value,
            ),
            lower_root == upper_root,
        )
    match dimension:
        case 1:
            arithmetic, terms = 1, 1
        case 2:
            arithmetic, terms = 7, 2
        case 3:
            arithmetic, terms = 26, 6
        case _:
            raise ValueError(
                "Exact affine simplex measure supports dimensions one through three."
            )
    _reserve_polynomial(
        arithmetic,
        dimension * dimension + terms,
        0,
        dimension * (2 * bits + 1) + terms.bit_length(),
    )
    columns = tuple(
        tuple(corners[column + 1][axis] - corners[0][axis] for column in range(dimension))
        for axis in range(dimension)
    )
    if dimension == 1:
        determinant_value = columns[0][0]
    elif dimension == 2:
        determinant_value = columns[0][0] * columns[1][1] - columns[0][1] * columns[1][0]
    else:
        determinant_value = (
            columns[0][0]
            * (columns[1][1] * columns[2][2] - columns[1][2] * columns[2][1])
            - columns[0][1]
            * (columns[1][0] * columns[2][2] - columns[1][2] * columns[2][0])
            + columns[0][2]
            * (columns[1][0] * columns[2][1] - columns[1][1] * columns[2][0])
        )
    if determinant_value <= 0:
        raise ValueError(
            "Mapped integration requires a certified strictly positive coordinate Jacobian."
        )
    measure = determinant_value / math.factorial(dimension)
    value = float(measure)
    error = max(value - outward(measure, -math.inf), outward(measure, math.inf) - value)
    return value, error, True


def _exact_affine_simplex_measure(
    element: CellGeometryElement,
    local: CoordinateCoefficients,
    /,
) -> tuple[float, float, bool] | None:
    """Exact degree-one simplex measure without materializing its polynomials."""
    scalar = _require_scalar_coordinate_element(element, "Exact affine measure")
    dimension = scalar.topological_dimension
    ambient = len(local[0])
    if (
        scalar.cell_kind not in ("interval", "triangle", "tetrahedron")
        or scalar.degree != 1
        or (
            ambient != dimension and not (scalar.cell_kind == "triangle" and ambient == 3)
        )
    ):
        return None
    corners = coordinate_corner_images(scalar, local)
    if corners is None or len(corners) != dimension + 1:
        return None
    return _exact_affine_simplex_measure_from_corners(corners)


def _certified_cell_measures(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    /,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-10,
    maximum_work: int = 100_000_000,
    maximum_subcells: int = 10000,
    maximum_binomial_terms: int = 32,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Physical cell measures with quantitative absolute integration errors.

    Volume/axis-plane densities retain exact rational integration. General
    embedded triangle/quad sqrt-Gram densities use Bernstein-positive dyadic
    subdivisions and validated binomial polynomial integration: remainder,
    source conversion, coefficient arithmetic and publication errors are enclosed.
    No quadrature samples certify this measure. ``exact`` is false on that route.

    Absolute tolerance is allocated across cells; relative budgets use certified
    lower areas, so summed errors are at most absolute plus relative total area.
    ``maximum_work`` bounds dense coefficient products across all cells;
    ``maximum_subcells`` bounds visited reference pieces per cell. Exhaustion
    raises structured ``CellGeometryTransitionError`` without partial acceptance.
    """
    from fractions import Fraction

    absolute = _finite_non_negative(absolute_tolerance, "absolute_tolerance")
    relative = _finite_non_negative(relative_tolerance, "relative_tolerance")
    if (
        absolute == relative == 0
        or min(maximum_work, maximum_subcells, maximum_binomial_terms) <= 0
    ):
        raise ValueError("Embedded integration requires positive error and work budgets.")
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    if _COORDINATE_BUDGET.get() is None:
        ledger = CoordinateEnclosureBudget(maximum_work, sys.maxsize)
        try:
            with ledger.activate():
                return _certified_cell_measures(
                    mesh,
                    geometry,
                    absolute_tolerance=absolute,
                    relative_tolerance=relative,
                    maximum_work=maximum_work,
                    maximum_subcells=maximum_subcells,
                    maximum_binomial_terms=maximum_binomial_terms,
                )
        except CoordinateEnclosureResourceError as error:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Mapped measure coefficient work exhausted its declared budget.",
                measured=error.requested,
                limit=maximum_work,
            ) from error
    match geometry.exact_source:
        case (
            ExactPowerCellGeometryLinearActionSource()
            | ExactPowerCellGeometryRestrictionSource()
        ) if all(block.cell_kind == "hexahedron" for block in mesh.blocks):
            # The retained exact vertex bank defines the actual Q1 polynomials.
            pass
        case (
            ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            from ._exact_power_consumers import exact_power_cell_measures

            return exact_power_cell_measures(mesh, geometry, maximum_work=maximum_work)
        case None:
            pass
        case ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
            pass
        case invalid:
            assert_never(invalid)
    from ._coordinate_enclosure import _coefficient_profile, _reserve_polynomial

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError(
            "The physical measure operation lost its owning coefficient ledger."
        )
    cells = _mapped_geometry_cells(mesh, geometry)
    integrals: list[tuple[float, float]] = []
    exact, work = True, [0, int(maximum_work)]
    absolute_per_cell = Fraction(absolute) / len(cells)
    with ledger.temporary_scope():
        references: dict[str, list[tuple[Polynomial, Fraction, float, float]]] = {}
        for element, local in cells:
            reference = None
            with ledger.temporary_scope():
                affine = _exact_affine_simplex_measure(element, local)
                if affine is not None:
                    affine_integral = affine[:2]
                    exact = exact and affine[2]
                    allowed = max(
                        absolute_per_cell,
                        Fraction(relative)
                        * max(
                            Fraction(affine_integral[0]) - Fraction(affine_integral[1]),
                            Fraction(0),
                        ),
                    )
                    if Fraction(affine_integral[1]) > allowed:
                        raise CellGeometryTransitionError(
                            "resource_limit",
                            "Exact measure publication exceeds its error budget.",
                            measured=affine_integral[1],
                            limit=float(allowed),
                        )
                    integrals.append(affine_integral)
                    continue
                embedded = element.topological_dimension == 2 and len(local[0]) == 3
                coordinates: tuple[Expression, ...] = ()
                constant_axes = []
                if embedded:
                    coordinates = _mapped_coordinate_expressions(element, local)
                    if element.cell_kind == "quadrilateral":
                        from ._coordinate_enclosure import (
                            expression_derivative as derivative,
                        )

                        constant_axes = [
                            axis
                            for axis, polynomial in enumerate(coordinates)
                            if all(not derivative(polynomial, i) for i in range(2))
                        ]
                if not embedded or (
                    element.cell_kind == "quadrilateral" and len(constant_axes) == 1
                ):
                    density = _mapped_density_expression(element, local)
                    if isinstance(density, RationalPolynomial):
                        value, error, rational_exact = _integrate_mapped_rational(
                            density,
                            element.cell_kind,
                            absolute_tolerance=absolute_per_cell,
                            relative_tolerance=relative,
                            maximum_terms=maximum_binomial_terms,
                        )
                        integral = value, error
                        exact = exact and rational_exact
                    else:
                        integral = _integrate_mapped_polynomial(
                            density, element.cell_kind
                        )
                    allowed = max(
                        absolute_per_cell,
                        Fraction(relative)
                        * max(Fraction(integral[0]) - Fraction(integral[1]), Fraction(0)),
                    )
                    if Fraction(integral[1]) > allowed:
                        raise CellGeometryTransitionError(
                            "resource_limit",
                            "Exact measure publication exceeds its error budget.",
                            measured=integral[1],
                            limit=float(allowed),
                        )
                elif any(isinstance(value, RationalPolynomial) for value in coordinates):
                    exact = False
                    integral = _certified_embedded_cell_measure(
                        element,
                        local,
                        absolute_per_cell,
                        relative,
                        int(maximum_subcells),
                        int(maximum_binomial_terms),
                        work,
                    )
                else:
                    exact = False
                    gram = _embedded_squared_density(
                        element, local, prepared_coordinates=coordinates
                    )
                    if isinstance(gram, RationalPolynomial):
                        raise TypeError(
                            "Polynomial coordinate maps produced a rational Gram expression."
                        )
                    integral, reference = _reuse_embedded_polynomial_area(
                        gram,
                        element.cell_kind,
                        references.get(element.cell_kind, []),
                        absolute_per_cell,
                        relative,
                        int(maximum_subcells),
                        int(maximum_binomial_terms),
                        work,
                    )
                integrals.append(integral)
            if reference is not None:
                gram = reference[0]
                numerator, denominator = _coefficient_profile((gram,))
                _reserve_polynomial(1, len(gram), 2, numerator + denominator)
                references.setdefault(element.cell_kind, []).append(reference)
    return (
        np.asarray([value for value, _ in integrals], dtype=np.float64),
        np.asarray([error for _, error in integrals], dtype=np.float64),
        exact,
    )


def _mapped_root_identities(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
) -> tuple[tuple[str, str, int, tuple[int, ...]], ...]:
    origin = geometry.restriction_source
    result: list[tuple[str, str, int, tuple[int, ...]]] = []
    for block in mesh.blocks:
        if origin is None:
            cells = np.asarray(block.global_ids)
            corners = np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)]
            geometry_id, topology_id = cell_geometry_id(geometry), mesh.topology_id
        else:
            cells = np.asarray(origin.block_parent_cell_ids[block.name])
            corners = np.asarray(origin.block_parent_vertex_ids[block.name])
            geometry_id, topology_id = (
                origin.source_geometry_id,
                origin.source_topology_id,
            )
        result.extend(
            (
                geometry_id,
                topology_id,
                int(cell),
                tuple(int(v) for v in vertices if v >= 0),
            )
            for cell, vertices in zip(cells, corners, strict=True)
        )
    return tuple(result)


def _certify_nested_geometry_pairs(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    pairs: tuple[_NestedReferencePair, ...],
) -> float:
    """Prove paired coordinate equality from canonical exact source expressions."""
    source = _mapped_geometry_cells(source_mesh, source_geometry)
    target = _mapped_geometry_cells(target_mesh, target_geometry)
    if (
        source_geometry.restriction_source is not None
        or target_geometry.restriction_source is not None
    ):
        source_roots = _mapped_root_identities(source_mesh, source_geometry)
        target_roots = _mapped_root_identities(target_mesh, target_geometry)
        if any(
            source_roots[pair.source_cell] != target_roots[pair.target_cell]
            for pair in pairs
        ):
            raise ValueError(
                "Nested geometry restriction has stale or mismatched scientific root identities."
            )
    bound = 0.0
    for pair in pairs:
        fine, coarse = (
            (source[pair.source_cell], target[pair.target_cell])
            if pair.fine_is_source
            else (target[pair.target_cell], source[pair.source_cell])
        )
        fine_element, fine_local = fine
        coarse_element, coarse_local = coarse
        fine_coordinates = _mapped_coordinate_expressions(fine_element, fine_local)
        coarse_coordinates = _mapped_coordinate_expressions(coarse_element, coarse_local)
        restricted = _mapped_expression_restriction(
            coarse_coordinates,
            coarse_element.cell_kind,
            fine_element.cell_kind,
            pair.matrix,
            pair.offset,
        )
        domain = (
            "simplex"
            if fine_element.cell_kind in ("interval", "triangle", "tetrahedron")
            else "prism"
            if fine_element.cell_kind == "prism"
            else "box"
        )
        for actual, expected in zip(fine_coordinates, restricted, strict=True):
            difference = expression_add(actual, expression_scale(expected, -1))
            if difference:
                lower, upper = expression_bounds(
                    difference, domain, fine_element.topological_dimension
                )
                bound = max(bound, abs(lower), abs(upper))
    scale_ = max(
        float(np.max(np.abs(np.asarray(source_geometry.coordinates)))),
        float(np.max(np.abs(np.asarray(target_geometry.coordinates)))),
        1.0,
    )
    if bound > 64 * _EPSILON * scale_:
        raise ValueError(
            "Nested geometry source expressions do not represent the same physical map."
        )
    return bound


def _restricted_element(
    element: CellGeometryElement, kind: str, matrix: np.ndarray, offset: np.ndarray
) -> (
    FiniteElementSpec
    | BarycentricCellGeometryElement
    | RestrictedCellGeometryElement
    | PolynomialComposedCellGeometryElement
    | RationalComposedCellGeometryElement
    | SplineCellGeometryElement
    | LayerColumnCellGeometryElement
):
    from .fem._reference import FiniteElementSpec

    if not isinstance(
        element,
        (
            FiniteElementSpec,
            BarycentricCellGeometryElement,
            RestrictedCellGeometryElement,
            PolynomialComposedCellGeometryElement,
            RationalComposedCellGeometryElement,
            SplineCellGeometryElement,
            LayerColumnCellGeometryElement,
        ),
    ):
        raise ValueError("A restricted coordinate basis must be scalar and tabulated.")
    element = _require_scalar_coordinate_element(element, "Restricted coordinates")
    dimension = reference_cell_topology(kind).dimension
    if (
        element.cell_kind == kind
        and np.array_equal(matrix, np.eye(dimension, dtype=np.float64))
        and np.array_equal(offset, np.zeros(dimension, dtype=np.float64))
    ):
        return element
    return RestrictedCellGeometryElement(element, kind, matrix, offset)


def _transition_mixed_nested_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    *,
    refinement: NestedReferenceWitnesses | None,
    coarsening: NestedReferenceWitnesses | None,
    policy: CellGeometryTransitionPolicy,
) -> CellGeometryTransition:
    """Compose coefficient bases exactly, restoring complete recorded siblings.

    Child nodal interpolation is not used. Each target block owns one affine
    reference action. Rational pyramid maps stay rational restricted maps; their
    validity must be accepted separately by the geometry certification owner.
    """
    from .fem._reference import FiniteElementSpec

    source_elements, source_routes, source_coordinates_ = source_geometry.resolve(
        source_mesh
    )
    if any(
        not isinstance(
            element,
            (
                FiniteElementSpec,
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
        raise ValueError("Mixed restriction requires scalar coordinate maps.")
    source_coordinates = np.asarray(source_coordinates_, dtype=np.float64)
    source_index = {
        int(identifier): (index, row)
        for index, block in enumerate(source_mesh.blocks)
        for row, identifier in enumerate(np.asarray(block.global_ids))
    }
    refined = (
        {}
        if refinement is None
        else {
            int(fine): (int(coarse), reference)
            for fine, coarse, reference in zip(
                refinement.fine_cell_ids,
                refinement.coarse_cell_ids,
                refinement.fine_reference_vertices,
                strict=True,
            )
        }
    )
    coarsened: dict[int, list[tuple[int, np.ndarray]]] = {}
    if coarsening is not None:
        for fine, coarse, reference in zip(
            coarsening.fine_cell_ids,
            coarsening.coarse_cell_ids,
            coarsening.fine_reference_vertices,
            strict=True,
        ):
            coarsened.setdefault(int(coarse), []).append((int(fine), reference))
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    if set(refined) & set(coarsened) or set(refined) | set(coarsened) != set(target_ids):
        raise ValueError("Every mixed target cell needs exactly one nested disposition.")
    ledger = CoordinateEnclosureBudget(policy.maximum_evaluations, sys.maxsize)
    with _charged(ledger, policy):
        source_coefficients = source_geometry.source_coordinates()
    elements: dict[str, CellGeometryElement] = {}
    routes: dict[str, np.ndarray] = {}
    origin = source_geometry.restriction_source
    if origin is None and any(
        isinstance(element, RestrictedCellGeometryElement) for element in source_elements
    ):
        raise ValueError(
            "Restricted source maps require explicit scientific parent geometry identities."
        )
    origin_cells: dict[str, np.ndarray] = {}
    origin_vertices: dict[str, np.ndarray] = {}
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
    parents = np.full(target_ids.size, -1, dtype=np.int64)
    dimension = target_mesh.topological_dimension
    reference = np.zeros((target_ids.size, 8, dimension), dtype=np.float64)
    # Published carrier corners are the correctly rounded binary64 images of the
    # exact restricted maps; the maps, not these floats, are the geometry.
    exact_vertices: dict[int, tuple[Fraction, ...]] = {}
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
        for identifier_, row in zip(
            np.asarray(block.global_ids), np.asarray(block.vertices), strict=True
        ):
            identifier = int(identifier_)
            if identifier in refined:
                parent, corners = refined[identifier]
                index, parent_row = source_index[parent]
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
                route = np.asarray(source_routes[index][parent_row], dtype=np.int64)
                ancestry_row = parent_row
                parents[row_index] = parent
                reference[row_index, : corners.shape[0]] = corners
                points = corners[: vertices.shape[0]]
                kind = source_mesh.blocks[index].cell_kind
                if kind == "tetrahedron":
                    defect = _excursion(points)
                elif kind == "prism":
                    defect = max(
                        _excursion(points[:, :2]),
                        float(np.max(-points[:, 2], initial=0.0)),
                        float(np.max(points[:, 2] - 1, initial=0.0)),
                    )
                elif kind == "pyramid":
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
                else:
                    defect = float(
                        max(np.max(-points, initial=0.0), np.max(points - 1, initial=0.0))
                    )
                containment = max(containment, defect)
            else:
                candidates = []
                for fine, corners in coarsened[identifier]:
                    index, fine_row = source_index[fine]
                    restricted = source_elements[index]
                    matrix, offset = _reference_affine(
                        source_mesh.blocks[index].cell_kind, corners
                    )
                    if isinstance(restricted, RestrictedCellGeometryElement):
                        if not np.array_equal(
                            np.asarray(restricted.matrix), matrix
                        ) or not np.array_equal(np.asarray(restricted.offset), offset):
                            raise CellGeometryTransitionError(
                                "approximation_bound",
                                "Sibling maps do not match their exact recorded child actions",
                                measured=math.inf,
                                limit=policy.coarsening_tolerance,
                            )
                        candidate = restricted.source_element
                    elif isinstance(
                        restricted,
                        (
                            PolynomialComposedCellGeometryElement,
                            RationalComposedCellGeometryElement,
                        ),
                    ):
                        from ._coordinate_enclosure import (
                            affine_arguments,
                            reference_composition_arguments,
                        )

                        authored = reference_composition_arguments(restricted)
                        expected = affine_arguments(offset, matrix)
                        if authored != expected:
                            raise CellGeometryTransitionError(
                                "approximation_bound",
                                "Sibling source-reference charts do not equal their complete recorded affine actions",
                                measured=math.inf,
                                limit=policy.coarsening_tolerance,
                            )
                        candidate = restricted.source_element
                    else:
                        raise CellGeometryTransitionError(
                            "approximation_bound",
                            "Sibling maps lack an exact recorded restriction",
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
                    or not np.array_equal(
                        source_coordinates[child_route], source_coordinates[route]
                    )
                    for candidate, child_route in candidates[1:]
                ):
                    raise CellGeometryTransitionError(
                        "approximation_bound",
                        "Sibling parent coefficient maps disagree",
                        measured=math.inf,
                        limit=policy.coarsening_tolerance,
                    )
                if element.cell_kind != block.cell_kind:
                    raise ValueError(
                        "Sibling restoration must recover the parent family."
                    )
                index, ancestry_row = source_index[coarsened[identifier][0][0]]
                ancestry_name = source_mesh.blocks[index].name
                ancestor = source_parent_cells[ancestry_name][ancestry_row]
                if any(
                    source_parent_cells[source_mesh.blocks[source_index[fine][0]].name][
                        source_index[fine][1]
                    ]
                    != ancestor
                    for fine, _ in coarsened[identifier]
                ):
                    raise ValueError(
                        "Sibling coordinate maps must have one scientific source parent."
                    )
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
            with _charged(ledger, policy):
                images = coordinate_corner_images(
                    element, tuple(source_coefficients[index] for index in route)
                )
                if images is None:
                    raise ValueError(
                        "Mixed restriction requires an exact coordinate source expression."
                    )
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
            row_index += 1
        element = block_elements[0]
        if any(other.element_id != element.element_id for other in block_elements[1:]):
            raise ValueError("Mixed target blocks must group identical restricted maps.")
        elements[block.name] = element
        routes[block.name] = np.stack(block_routes)
        origin_cells[block.name] = np.asarray(block_origins, dtype=np.int64)
        origin_vertices[block.name] = np.stack(block_origin_vertices)
    slack = policy.continuity_tolerance * _extent(source_coordinates)
    if continuity > slack or containment > 64 * _EPSILON:
        raise CellGeometryTransitionError(
            "discontinuous_source" if continuity > slack else "coverage",
            "Mixed restrictions violate continuity or parent containment",
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
    geometry = CellGeometrySpec(
        elements,
        {name: remap[route] for name, route in routes.items()},
        source_coordinates[used],
        restriction_source=restriction_source,
        periodic_source=periodic_source,
    )
    with _charged(ledger, policy):
        source_values, source_errors, measure_exact = _certified_cell_measures(
            source_mesh, source_geometry
        )
        target_values, target_errors, target_measure_exact = _certified_cell_measures(
            target_mesh, geometry
        )
    work = ledger.work_units
    source_measure, target_measure = math.fsum(source_values), math.fsum(target_values)
    source_error = math.fsum(source_errors) + abs(float(np.spacing(source_measure)))
    target_error = math.fsum(target_errors) + abs(float(np.spacing(target_measure)))
    evidence = CellGeometryTransitionEvidence(
        "parent_restoration" if coarsened else "nested_restriction",
        exact=True,
        node_count=used.size,
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
    empty = np.zeros(0, dtype=np.int64)
    coarse_witness = (
        NestedReferenceWitnesses(
            empty, empty, np.zeros((0, 8, dimension), dtype=np.float64)
        )
        if coarsening is None
        else coarsening
    )
    return CellGeometryTransition(
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        source_geometry_id=cell_geometry_id(source_geometry),
        target_geometry_id=cell_geometry_id(geometry),
        geometry=geometry,
        vertex_coordinates=jnp.asarray(
            np.stack(
                [
                    vertex_values[index]
                    for index in range(target_mesh.coordinates.shape[0])
                ]
            )
        ),
        target_cell_ids=jnp.asarray(target_ids),
        parent_cell_ids=jnp.asarray(parents),
        parent_reference_vertices=jnp.asarray(reference),
        coarsened_cell_ids=jnp.asarray(coarse_witness.fine_cell_ids),
        coarsened_into_ids=jnp.asarray(coarse_witness.coarse_cell_ids),
        coarsened_reference_vertices=jnp.asarray(coarse_witness.fine_reference_vertices),
        node_owner_cells=jnp.asarray(
            [owner[int(node)][0] for node in used], dtype=jnp.int64
        ),
        node_owner_locals=jnp.asarray(
            [owner[int(node)][1] for node in used], dtype=jnp.int32
        ),
        evidence=evidence,
        policy_id=policy.policy_id,
        transition_id=canonical_fingerprint(
            {
                "kind": "mixed-coordinate-restriction",
                "source": cell_geometry_id(source_geometry),
                "target": cell_geometry_id(geometry),
                "source_topology": source_mesh.topology_id,
                "target_topology": target_mesh.topology_id,
                "policy": policy.policy_id,
                "evidence": evidence.evidence_id,
            }
        ),
    )


__all__ = [
    "CellGeometryTransition",
    "CellGeometryTransitionError",
    "CellGeometryTransitionEvidence",
    "CellGeometryTransitionKind",
    "CellGeometryTransitionPolicy",
    "CellGeometryTransitionRefusal",
    "CoarseningGeometryApproximation",
    "NestedReferenceWitnesses",
    "SourceGeometryRealization",
    "prepare_source_geometry_realization",
    "SurfaceGeometryReconstruction",
    "SurfaceGeometryCorrespondence",
    "transition_chart_deformed_cell_geometry",
    "coordinate_element_lebesgue_bound",
    "reconstruct_parametric_surface_cell_geometry",
    "cell_geometry_vertex_measures",
    "is_affine_cell_geometry",
    "nested_geometry_degree",
    "transition_displaced_cell_geometry",
    "transition_nested_cell_geometry",
]


class SurfaceGeometryReconstruction(StrictModule, NonTrainableState):
    """Bounded target map, not an accepted old-to-new transition.

    Polynomial reconstruction bounds encoded-basis reproduction, point
    enclosures and shared-node placement. Retained spline reconstruction uses
    original source-expression composition and has no interpolation defects.
    ``target_mesh`` retains the actual coefficient-element block layout.
    Old-map inverse/declared-chart coverage, trim coverage, validity, embedding
    and transfer remain separate gates; zero source fidelity is not an accepted
    zero-displacement material correspondence.
    """

    __strict_contract__ = True

    source_geometry_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    cell_geometry_entity_ids: tuple[str, ...] = eqx.field(static=True)
    cell_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    geometry: CellGeometrySpec
    target_mesh: CellMesh
    vertex_coordinates: Float64[_VertexDim, _AmbientDim]
    cell_global_ids: Int64[_TargetCellDim]
    cell_patches: Int32[_TargetCellDim]
    cell_charts: Float64[_TargetCellDim, _CornerDim, _ReferenceDim]
    fidelity_bounds: Float64[_TargetCellDim]
    lebesgue_bounds: Float64[_TargetCellDim]
    affine_reproduction_bounds: Float64[_TargetCellDim]
    node_enclosure_bounds: Float64[_TargetCellDim]
    node_owner_cells: Int64[_NodeDim] | None
    node_owner_locals: Int32[_NodeDim] | None
    continuity_residual: float = eqx.field(static=True)
    evaluation_count: int = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    @property
    def exact(self) -> bool:
        return False

    @property
    def source_inverse_coverage(self) -> None:
        return None


def coordinate_element_lebesgue_bound(element: FiniteElementSpec, /) -> float:
    """Bound sum|phi_i| by actual common-Bernstein absolute column sums."""
    from ._coordinate_enclosure import bernstein_coefficients, outward
    from .fem._reference import FiniteElementSpec

    if not isinstance(element, FiniteElementSpec):
        raise TypeError("element must be a finite-element coordinate specification.")
    if element.cell_kind != "triangle" or element.conformity != "H1":
        raise ValueError("Surface reconstruction requires scalar H1 triangles.")
    basis = source_basis(element)
    if basis is None or any(
        max((sum(index) for index in polynomial), default=0) != element.degree
        for polynomial in basis
    ):
        raise ValueError("The coordinate basis lacks a common exact Bernstein degree.")
    coefficients = tuple(
        bernstein_coefficients(polynomial, "simplex", 2) for polynomial in basis
    )
    bound = max(
        sum(abs(value) for value in column) for column in zip(*coefficients, strict=True)
    )
    return outward(bound, math.inf)


def _surface_affine_defect(element: FiniteElementSpec, corners: np.ndarray) -> float:
    """Exact enclosure of L-I_pL for encoded basis and affine corner chord L."""
    from ._coordinate_enclosure import axes, bernstein_coefficients, constant, outward

    basis = source_basis(element)
    if basis is None:
        raise ValueError("The coordinate basis has no source-level polynomial enclosure.")
    variables = axes(2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    bound = Fraction(0)
    for component in range(3):
        values = tuple(Fraction(float(value)) for value in corners[:, component])
        affine = add(
            constant(values[0], 2),
            add(
                scale(variables[0], values[1] - values[0]),
                scale(variables[1], values[2] - values[0]),
            ),
        )
        interpolated = constant(0, 2)
        for polynomial, node in zip(basis, nodes, strict=True):
            node_ = np.asarray(node, dtype=np.float64)
            value = (
                values[0]
                + Fraction(float(node_[0])) * (values[1] - values[0])
                + Fraction(float(node_[1])) * (values[2] - values[0])
            )
            interpolated = add(interpolated, scale(polynomial, value))
        difference = add(affine, scale(interpolated, -1))
        bound += max(
            abs(value) for value in bernstein_coefficients(difference, "simplex", 2)
        )
    return outward(bound, math.inf)


def _surface_node_enclosures(
    domain: MeshingDomain, patch: int, charts: np.ndarray, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Host certificate queries enclosing exact affine-chart node parameters."""
    from ._coordinate_enclosure import outward

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
        exact_corners = tuple(
            tuple(Fraction(float(value)) for value in row) for row in corners
        )
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
                    [outward(value, -math.inf) for value in parameters],
                    [outward(value, math.inf) for value in parameters],
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
                    "The authoritative patch lacks a finite point enclosure."
                )
            lower[cell, slot], upper[cell, slot] = enclosure
    return lower, upper


def _surface_reconstruction_bounds(
    blocks: tuple[_Block, ...],
    placed: _Placed,
    enclosures: list[tuple[np.ndarray, np.ndarray]],
    chord_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ._coordinate_enclosure import outward

    total, lebesgue, affine, errors = [], [], [], []
    offset = 0
    for block, (lower, upper) in zip(blocks, enclosures, strict=True):
        constant_ = coordinate_element_lebesgue_bound(block.element)
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
            defect = _surface_affine_defect(block.element, nodes[row, vertex_dofs])
            bound = (
                (Fraction(1) + Fraction(constant_))
                * (Fraction(float(chord_bounds[offset + row])) + corner_error)
                + Fraction(defect)
                + Fraction(constant_) * node_error
            )
            total.append(outward(bound, math.inf))
            lebesgue.append(constant_)
            affine.append(defect)
            errors.append(outward(node_error, math.inf))
        offset += block.cell_ids.size
    return (
        np.asarray(total, dtype=np.float64),
        np.asarray(lebesgue, dtype=np.float64),
        np.asarray(affine, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
    )


def _require_retained_surface_continuity(
    mesh: CellMesh, geometry: CellGeometrySpec
) -> None:
    """Authenticate whole shared edge traces, not only shared corner samples."""
    from ._coordinate_enclosure import (
        axes,
        constant,
        expression_compose,
        expression_parts,
    )

    elements, routes, _ = geometry.resolve(mesh)
    if all(
        isinstance(element, PolynomialComposedCellGeometryElement)
        and element.chart_element.cell_kind == "triangle"
        and element.chart_element.degree == 1
        for element in elements
    ):
        vertex_charts: dict[int, tuple[Fraction, Fraction]] = {}
        vertex_ids = np.asarray(mesh.vertex_global_ids)
        for block, element in zip(mesh.blocks, elements, strict=True):
            if not isinstance(element, PolynomialComposedCellGeometryElement):
                raise RuntimeError("Retained chart fast path lost its element type.")
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
                        raise CellGeometryTransitionError(
                            "discontinuous_source",
                            "Retained source patches disagree on an exact shared chart vertex.",
                            measured=math.inf,
                            limit=0.0,
                        )
        return
    controls = geometry.source_coordinates()
    variable = axes(1)[0]
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
                element, tuple(controls[int(index)] for index in dofs)
            )
            if values is None:
                raise ValueError("Retained surface lacks its actual source expressions.")
            ids = np.asarray(mesh.vertex_global_ids)[vertices]
            for first, last in ((0, 1), (1, 2), (2, 0)):
                if ids[first] > ids[last]:
                    first, last = last, first
                key = int(ids[first]), int(ids[last])
                arguments = tuple(
                    add(
                        constant(references[first][axis], 1),
                        scale(variable, references[last][axis] - references[first][axis]),
                    )
                    for axis in range(2)
                )
                trace = tuple(expression_compose(value, arguments) for value in values)
                previous = traces.get(key)
                if previous is not None:
                    for left, right in zip(previous, trace, strict=True):
                        numerator, _ = expression_parts(
                            expression_add(left, expression_scale(right, -1)), 1
                        )
                        if numerator:
                            raise CellGeometryTransitionError(
                                "discontinuous_source",
                                "Retained source patches disagree on a complete shared edge trace.",
                                measured=math.inf,
                                limit=0.0,
                            )
                else:
                    traces[key] = trace


def _reconstruct_retained_surface_source(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    domain: MeshingDomain,
    atlas: SurfaceSourceRootAtlas,
    cell_ids: np.ndarray,
    cell_patches: np.ndarray,
    cell_charts: np.ndarray,
    entities: tuple[str, ...],
    occurrences: tuple[tuple[str, ...], ...],
    policy: CellGeometryTransitionPolicy,
) -> SurfaceGeometryReconstruction:
    """Continuous equality by original source-expression composition, not samples.

    The target map is S composed with the declared affine UV triangle. Original
    knot/weight/control banks survive unchanged. Validity, embedding and the
    old-to-new continuous difference remain the chart-deformation owner's gates.
    """
    from ..geometry._surface_source_support import (
        restrict_surface_source_atlas_geometry,
        SurfaceSourceRootAtlas,
    )

    if (
        not isinstance(atlas, SurfaceSourceRootAtlas)
        or atlas.domain.domain_id != domain.domain_id
    ):
        raise ValueError(
            "Rational reconstruction requires the owning original source atlas."
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
            "Retained source charts must name every scientific target cell exactly once."
        )
    if (
        patches.shape != ids.shape
        or not np.issubdtype(patches.dtype, np.integer)
        or np.any((patches < 0) | (patches >= len(domain.patches)))
        or charts.shape != (ids.size, 3, 2)
        or not np.all(np.isfinite(charts))
    ):
        raise ValueError(
            "Retained source charts require actual patch indices and finite UV triangles."
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
            "Retained source charts change an original entity or occurrence."
        )
    rows = {int(identifier): row for row, identifier in enumerate(ids)}
    order = np.asarray(
        [rows[int(identifier)] for identifier in target_ids], dtype=np.int64
    )
    patches, charts = patches[order], charts[order]
    from .._meshcore import charge_native_geometry_queries
    from ._coordinate_enclosure import _COORDINATE_BUDGET
    from ._surface_chart_deformation import _original_coefficient_limits

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        maximum_work, maximum_memory = _original_coefficient_limits(
            policy.maximum_evaluations, sys.maxsize
        )
        if maximum_work <= 0 or maximum_memory <= 0:
            raise CellGeometryTransitionError(
                "resource_limit",
                "Retained source reconstruction has no remaining original coefficient resources.",
                measured=1,
                limit=min(maximum_work, maximum_memory),
            )
        ledger = CoordinateEnclosureBudget(maximum_work, maximum_memory)
    starting_work = ledger.work_units
    with _charged(ledger, policy):
        ledger.reserve(12 * ids.size)
        mesh, geometry = restrict_surface_source_atlas_geometry(
            atlas,
            target_mesh,
            patches,
            charts,
            maximum_support_queries=policy.maximum_evaluations,
            canonical_blocks=True,
        )
        _require_retained_surface_continuity(mesh, geometry)
    work = ledger.work_units - starting_work
    charge_native_geometry_queries(0, work_units=work)
    zeros = jnp.zeros((ids.size,), dtype=jnp.float64)
    # There are no reconstructed nodal coefficients: these are the original
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
    return SurfaceGeometryReconstruction(
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


def reconstruct_parametric_surface_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_layout: CellGeometrySpec | None,
    domain: MeshingDomain,
    /,
    *,
    domain_id: str,
    cell_ids: np.ndarray,
    cell_patches: np.ndarray,
    cell_charts: np.ndarray,
    cell_geometry_entity_ids: tuple[str, ...],
    cell_occurrence_paths: tuple[tuple[str, ...], ...],
    maximum_fidelity: float,
    policy: CellGeometryTransitionPolicy | None = None,
    source_atlas: SurfaceSourceRootAtlas | None = None,
) -> SurfaceGeometryReconstruction:
    """Prepare bounded degree-preserving target coordinates without old coverage claims."""
    from ..geometry._meshing_domain import MeshingDomain
    from ..geometry.brep._patches import BSplineSurfacePatch

    if not isinstance(domain, MeshingDomain) or domain_id != domain.domain_id:
        raise ValueError("Chart witnesses must name the supplied authoritative domain.")
    if (
        not isinstance(source_mesh, CellMesh)
        or not isinstance(target_mesh, CellMesh)
        or source_mesh.topological_dimension != 2
        or target_mesh.topological_dimension != 2
        or source_mesh.ambient_dimension != 3
        or target_mesh.ambient_dimension != 3
    ):
        raise ValueError("Surface reconstruction requires two-dimensional meshes in 3D.")
    if not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError(
            "Reconstruction requires a canonical source coordinate specification."
        )
    limit = _finite_non_negative(maximum_fidelity, "maximum_fidelity")
    policy_ = CellGeometryTransitionPolicy() if policy is None else policy
    if not isinstance(policy_, CellGeometryTransitionPolicy):
        raise TypeError("policy must be CellGeometryTransitionPolicy.")
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
        )
    if any(
        isinstance(domain.patches[int(patch)].surface, BSplineSurfacePatch)
        for patch in np.asarray(cell_patches).reshape(-1)
        if 0 <= int(patch) < len(domain.patches)
    ):
        raise ValueError(
            "Spline surface reconstruction requires its retained original source atlas; nodal interpolation cannot preserve the declared source map."
        )
    if not isinstance(target_layout, CellGeometrySpec):
        raise TypeError(
            "Polynomial reconstruction requires its canonical target coordinate layout."
        )
    source_elements, _, _ = source_geometry.resolve(source_mesh)
    degrees = {
        _require_scalar_coordinate_element(element, "Surface reconstruction").degree
        for element in source_elements
    }
    if len(degrees) != 1:
        raise ValueError(
            "Nonnested surface reconstruction requires one coordinate degree."
        )
    blocks = _simplex_blocks(target_mesh, target_layout, "target")
    if any(
        block.element.cell_kind != "triangle" or block.degree not in degrees
        for block in blocks
    ):
        raise ValueError(
            "Reconstruction must retain the source triangle coordinate degree."
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
        raise ValueError("Explicit cell IDs, patch identities and charts must align.")
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
            "Witnesses must retain canonical geometry entities and occurrences."
        )
    indices = {int(identifier): row for row, identifier in enumerate(ids)}
    order = np.asarray(
        [indices[int(identifier)] for identifier in target_ids], dtype=np.int64
    )
    patches, charts = patches[order].astype(np.int32), charts[order]
    work = _budget(
        sum(block.routes.size * (block.element.local_dof_count + 6) for block in blocks),
        policy_,
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
            low[rows], high[rows] = _surface_node_enclosures(
                domain, int(patch), charts[first:stop][rows], nodes
            )
            chord[first + rows] = domain.interpolation_bounds(
                int(patch), charts[first:stop][rows]
            )
        enclosures.append((low, high))
        first = stop
    if not np.all(np.isfinite(chord)):
        raise ValueError("The source lacks a continuous interpolation bound.")
    placed = _place(blocks, values, np.asarray(target_layout.coordinates).shape[0])
    slack = policy_.continuity_tolerance * _extent(placed.coordinates)
    if placed.continuity > slack:
        raise CellGeometryTransitionError(
            "discontinuous_source",
            "Target charts disagree at shared nodes",
            measured=placed.continuity,
            limit=slack,
        )
    bound, lebesgue, affine, errors = _surface_reconstruction_bounds(
        blocks, placed, enclosures, chord
    )
    if np.max(bound, initial=0.0) > limit:
        raise CellGeometryTransitionError(
            "approximation_bound",
            "Reconstruction exceeds continuous source fidelity",
            measured=float(np.max(bound)),
            limit=limit,
        )
    elements, routes, _ = target_layout.resolve(target_mesh)
    names = tuple(block.name for block in target_mesh.blocks)
    geometry = CellGeometrySpec(
        dict(zip(names, elements, strict=True)),
        dict(zip(names, routes, strict=True)),
        placed.coordinates,
    )
    if target_mesh.periodic_topology is not None:
        geometry = geometry.with_periodic_source(target_mesh)
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
    return SurfaceGeometryReconstruction(
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
        vertex_coordinates=jnp.asarray(
            _vertex_rows(target_mesh, blocks, placed.coordinates)
        ),
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


def transition_chart_deformed_cell_geometry(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    reconstruction: SurfaceGeometryReconstruction | SphereGeometryReconstruction,
    deformation: PreparedSurfaceChartDeformation | PreparedSphereChartDeformation,
    /,
    *,
    policy: CellGeometryTransitionPolicy,
) -> CellGeometryTransition:
    """Accept a declared bounded material chart deformation with complete evidence.

    The physical surface measure may change. Complete native material coverage is
    retained in ``chart_deformation`` and bound by ``chart_correspondence_id``;
    neither equal physical areas nor a nested parent stencil is implied.
    """
    if not isinstance(policy, CellGeometryTransitionPolicy):
        raise TypeError("policy must be CellGeometryTransitionPolicy.")
    if policy.reconstruction != "bounded_chart_deformation":
        raise ValueError(
            "A bounded chart deformation requires an explicit geometry policy selection."
        )
    if isinstance(reconstruction, SphereGeometryReconstruction):
        if not isinstance(deformation, PreparedSphereChartDeformation):
            raise TypeError(
                "A sphere reconstruction requires its genuine radial material correspondence."
            )
        if deformation.target_atlas.atlas_id != reconstruction.target_atlas.atlas_id:
            raise ValueError(
                "Sphere reconstruction and correspondence name different actual target material maps."
            )
        reconstruction.target_validity.require_bound(
            reconstruction.geometry, mesh=target_mesh
        )
        reconstruction.target_embedding.binding.require(
            target_mesh, reconstruction.geometry
        )
        evaluations = reconstruction.evaluation_count
    elif isinstance(reconstruction, SurfaceGeometryReconstruction):
        if not isinstance(deformation, PreparedSurfaceChartDeformation):
            raise TypeError(
                "A surface reconstruction requires its genuine authored chart correspondence."
            )
        evaluations = reconstruction.evaluation_count + deformation.evaluation_count
    else:
        raise TypeError(
            "An accepted material transition requires its actual owning reconstruction."
        )
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
            "Surface reconstruction is stale for the accepted source and target."
        )
    target_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    reconstruction_ids = np.asarray(reconstruction.cell_global_ids)
    if target_ids.size != reconstruction_ids.size or not np.array_equal(
        np.sort(target_ids), np.sort(reconstruction_ids)
    ):
        raise ValueError(
            "Surface reconstruction rows do not name the target scientific cells."
        )
    if deformation.maximum_displacement_bound > policy.reconstruction_tolerance:
        raise CellGeometryTransitionError(
            "approximation_bound",
            "The old-to-target chart deformation exceeds its declared geometric budget",
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
    evidence = CellGeometryTransitionEvidence(
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
    return CellGeometryTransition(
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


_SOURCE_GEOMETRY_REALIZATION_TOKEN = object()


class SourceGeometryRealization(StrictModule, NonTrainableState):
    """A declared material-reference correspondence between different domains.

    Both embeddings are certified independently. Measures and their absolute
    errors refer to physical cells, not to an inferred common physical domain.
    """

    __strict_contract__ = True

    transition: CellGeometryTransition
    source_embedding: GlobalEmbeddingCertificate
    target_embedding: GlobalEmbeddingCertificate
    source_cell_measures: Float64[_TargetCellDim]
    target_cell_measures: Float64[_TargetCellDim]
    source_measure_errors: Float64[_TargetCellDim]
    target_measure_errors: Float64[_TargetCellDim]
    peak_expression_bytes: int = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition: CellGeometryTransition,
        source_embedding: GlobalEmbeddingCertificate,
        target_embedding: GlobalEmbeddingCertificate,
        source_cell_measures: Float64[_TargetCellDim],
        target_cell_measures: Float64[_TargetCellDim],
        source_measure_errors: Float64[_TargetCellDim],
        target_measure_errors: Float64[_TargetCellDim],
        peak_expression_bytes: int,
        /,
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _SOURCE_GEOMETRY_REALIZATION_TOKEN:
            raise TypeError(
                "SourceGeometryRealization is constructed by prepare_source_geometry_realization."
            )
        self.transition = transition
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_cell_measures = source_cell_measures
        self.target_cell_measures = target_cell_measures
        self.source_measure_errors = source_measure_errors
        self.target_measure_errors = target_measure_errors
        self.peak_expression_bytes = peak_expression_bytes
        self.realization_id = canonical_fingerprint(
            {
                "kind": "source-geometry-realization-measures",
                "transition": transition.transition_id,
                "source_measures": array_tree_fingerprint(source_cell_measures),
                "target_measures": array_tree_fingerprint(target_cell_measures),
                "source_errors": array_tree_fingerprint(source_measure_errors),
                "target_errors": array_tree_fingerprint(target_measure_errors),
            }
        )
        self.require_current()

    def require_current(self, /) -> None:
        """Validate retained numerical measure evidence after reconstruction."""
        expected = canonical_fingerprint(
            {
                "kind": "source-geometry-realization-measures",
                "transition": self.transition.transition_id,
                "source_measures": array_tree_fingerprint(self.source_cell_measures),
                "target_measures": array_tree_fingerprint(self.target_cell_measures),
                "source_errors": array_tree_fingerprint(self.source_measure_errors),
                "target_errors": array_tree_fingerprint(self.target_measure_errors),
            }
        )
        if expected != self.realization_id:
            raise ValueError(
                "Source-realization physical measures differ from their owning preparation."
            )
        source, target = (
            np.asarray(self.source_cell_measures),
            np.asarray(self.target_cell_measures),
        )
        old_errors, new_errors = (
            np.asarray(self.source_measure_errors),
            np.asarray(self.target_measure_errors),
        )
        count = self.transition.target_cell_ids.shape[0]
        if any(
            value.shape != (count,) for value in (source, target, old_errors, new_errors)
        ) or (
            not all(
                np.all(np.isfinite(value))
                for value in (source, target, old_errors, new_errors)
            )
            or np.any(old_errors < 0)
            or np.any(new_errors < 0)
            or np.any(source <= old_errors)
            or np.any(target <= new_errors)
        ):
            raise ValueError(
                "Source-realization physical inventories lack positive quantitative enclosures."
            )


def _require_realization_reference_complex(source: CellMesh, target: CellMesh, /) -> None:
    if source.ambient_dimension != target.ambient_dimension:
        raise ValueError(
            "Source realization must retain its declared ambient coordinate space."
        )
    if source.topology_id != target.topology_id:
        raise ValueError(
            "Source realization requires the same declared reference complex."
        )
    if len(source.blocks) != len(target.blocks) or not np.array_equal(
        np.asarray(source.vertex_global_ids), np.asarray(target.vertex_global_ids)
    ):
        raise ValueError("Source realization changes the reference vertex identities.")
    for old, new in zip(source.blocks, target.blocks, strict=True):
        if (
            (old.name, old.cell_kind) != (new.name, new.cell_kind)
            or not np.array_equal(np.asarray(old.global_ids), np.asarray(new.global_ids))
            or not np.array_equal(np.asarray(old.vertices), np.asarray(new.vertices))
        ):
            raise ValueError(
                "Source realization changes oriented reference-cell identities."
            )


def _realization_displacement_bound(
    source: CellMesh,
    source_geometry: CellGeometrySpec,
    target: CellMesh,
    target_geometry: CellGeometrySpec,
    ledger: CoordinateEnclosureBudget,
    /,
) -> float:
    maximum = 0.0
    changed = False
    old_cells = _mapped_geometry_cells(source, source_geometry)
    new_cells = _mapped_geometry_cells(target, target_geometry)
    for (old_element, old_values), (new_element, new_values) in zip(
        old_cells, new_cells, strict=True
    ):
        with ledger.temporary_scope():
            old = _mapped_coordinate_expressions(old_element, old_values)
            new = _mapped_coordinate_expressions(new_element, new_values)
            domain = (
                "simplex"
                if old_element.cell_kind in _SIMPLEX_DIMENSIONS
                else "prism"
                if old_element.cell_kind == "prism"
                else "box"
            )
            component_bounds = []
            for first, second in zip(old, new, strict=True):
                difference = expression_add(second, expression_scale(first, -1))
                changed = changed or bool(difference)
                lower, upper = expression_bounds(
                    difference, domain, source.topological_dimension
                )
                component_bounds.append(max(abs(lower), abs(upper)))
            squared = sum(
                (Fraction(value) ** 2 for value in component_bounds), Fraction(0)
            )
            bound = float(_fraction_sqrt_interval(squared)[1])
            maximum = max(maximum, float(np.nextafter(bound, math.inf)))
    if not changed:
        raise ValueError("Coordinate interpolation is not a changed source realization.")
    return maximum


def prepare_source_geometry_realization(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    target_mesh: CellMesh,
    target_geometry: CellGeometrySpec,
    /,
    *,
    source_embedding: GlobalEmbeddingCertificate,
    target_embedding: GlobalEmbeddingCertificate,
    policy: CellGeometryTransitionPolicy,
    maximum_storage_bytes: int,
) -> SourceGeometryRealization:
    """Certify an explicit same-reference material map, not physical equality."""
    from ..geometry._mesh_certificates import GlobalEmbeddingCertificate

    if policy.reconstruction != "source_realization":
        raise ValueError(
            "Source realization requires explicit material-reference admission."
        )
    if not isinstance(source_embedding, GlobalEmbeddingCertificate) or not isinstance(
        target_embedding, GlobalEmbeddingCertificate
    ):
        raise TypeError(
            "Source realization requires actual global embedding certificates."
        )
    source_embedding.binding.require(source_mesh, source_geometry)
    target_embedding.binding.require(target_mesh, target_geometry)
    if source_embedding.status != "certified" or target_embedding.status != "certified":
        raise ValueError("Both physical domains must be globally embedded.")
    _require_realization_reference_complex(source_mesh, target_mesh)
    if cell_geometry_id(source_geometry) == cell_geometry_id(target_geometry):
        raise ValueError("A source realization must change the actual coordinate map.")
    if maximum_storage_bytes < 1 or policy.maximum_evaluations < 4:
        raise ValueError("Source realization requires positive storage and work budgets.")
    integration_limit = policy.maximum_evaluations
    ledger = CoordinateEnclosureBudget(policy.maximum_evaluations, maximum_storage_bytes)
    with _charged(ledger, policy):
        bound = _realization_displacement_bound(
            source_mesh, source_geometry, target_mesh, target_geometry, ledger
        )
        source_measures, source_errors, old_exact = _certified_cell_measures(
            source_mesh, source_geometry, maximum_work=integration_limit
        )
        target_measures, target_errors, new_exact = _certified_cell_measures(
            target_mesh, target_geometry, maximum_work=integration_limit
        )
    if bound > policy.reconstruction_tolerance:
        raise CellGeometryTransitionError(
            "approximation_bound",
            "Source realization exceeds its full-map displacement budget",
            measured=bound,
            limit=policy.reconstruction_tolerance,
        )
    source_total, target_total = math.fsum(source_measures), math.fsum(target_measures)
    evidence = CellGeometryTransitionEvidence(
        "source_realization",
        exact=False,
        node_count=target_geometry.coordinates.shape[0],
        evaluation_count=ledger.work_units,
        containment_defect=0.0,
        continuity_residual=0.0,
        approximation_bound=bound,
        approximation_tolerance=policy.reconstruction_tolerance,
        rounding_slack=0.0,
        source_measure=source_total,
        target_measure=target_total,
        measure_exact=old_exact and new_exact,
        coverage_defect=None,
        source_measure_error_bound=math.fsum(source_errors)
        + abs(float(np.spacing(source_total))),
        target_measure_error_bound=math.fsum(target_errors)
        + abs(float(np.spacing(target_total))),
    )
    ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in target_mesh.blocks]
    )
    node_cells = np.full(target_geometry.coordinates.shape[0], -1, dtype=np.int64)
    node_locals = np.full(node_cells.shape, -1, dtype=np.int32)
    offset = 0
    for routes in target_geometry.resolve(target_mesh)[1]:
        for row, route in enumerate(np.asarray(routes, dtype=np.int64)):
            route = np.array(route, dtype=np.int64, copy=False)
            if route.ndim != 1:
                raise ValueError(
                    "Realized coordinate-cell routes must remain rank-one rows."
                )
            fresh = node_cells[route] < 0
            node_cells[route[fresh]] = offset + row
            node_locals[route[fresh]] = np.flatnonzero(fresh)
        offset += routes.shape[0]
    if np.any(node_cells < 0):
        raise ValueError(
            "Source realization leaves coordinate coefficients without owning cells."
        )
    dimension = target_mesh.topological_dimension
    empty = jnp.zeros((0,), dtype=jnp.int64)
    transition_id = canonical_fingerprint(
        {
            "kind": "source-geometry-realization",
            "source": cell_geometry_id(source_geometry),
            "target": cell_geometry_id(target_geometry),
            "topology": target_mesh.topology_id,
            "source_embedding": source_embedding.certificate_id,
            "target_embedding": target_embedding.certificate_id,
            "evidence": evidence.evidence_id,
            "policy": policy.policy_id,
        }
    )
    transition = CellGeometryTransition(
        source_topology_id=source_mesh.topology_id,
        target_topology_id=target_mesh.topology_id,
        source_geometry_id=cell_geometry_id(source_geometry),
        target_geometry_id=cell_geometry_id(target_geometry),
        geometry=target_geometry,
        vertex_coordinates=target_mesh.coordinates,
        target_cell_ids=jnp.asarray(ids),
        parent_cell_ids=jnp.full(ids.shape, -1, dtype=jnp.int64),
        parent_reference_vertices=jnp.zeros(
            (ids.size, dimension + 1, dimension), dtype=jnp.float64
        ),
        coarsened_cell_ids=empty,
        coarsened_into_ids=empty,
        coarsened_reference_vertices=jnp.zeros(
            (0, dimension + 1, dimension), dtype=jnp.float64
        ),
        node_owner_cells=jnp.asarray(node_cells),
        node_owner_locals=jnp.asarray(node_locals),
        evidence=evidence,
        policy_id=policy.policy_id,
        transition_id=transition_id,
    )
    return SourceGeometryRealization(
        transition,
        source_embedding,
        target_embedding,
        jnp.asarray(source_measures),
        jnp.asarray(target_measures),
        jnp.asarray(source_errors),
        jnp.asarray(target_errors),
        ledger.peak_bytes_upper,
        _construction_token=_SOURCE_GEOMETRY_REALIZATION_TOKEN,
    )
