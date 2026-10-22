#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Transition records and material correspondence contracts below transfer executors."""

from __future__ import annotations

import numbers
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from decimal import Decimal
from fractions import Fraction
from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._host_storage import declared_integer_storage_bytes
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_finite_float
from ..typing import Dim, Float64, Int32, Int64, parse
from ._cell_geometry import CellGeometrySpec
from ._cell_mesh import CellMesh
from ._coordinate_budget import CoordinateEnclosureBudget


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

    ``fine_reference_homogeneous`` optionally retains the source-owned rational
    coordinates of simplex coarse cells: a nonzero row ``[c, i]`` holds finite
    nonnegative binary64 homogeneous weights of coarse reference corners
    ``0..d`` whose rational sum is positive, so the unrounded reference
    coordinate is ``h[1:] / sum(h)`` and ``fine_reference_vertices[c, i]`` must be
    its round-to-nearest execution view. An all-zero row (and an absent bank)
    declares the binary64 vertex row itself exactly representable, as for dyadic
    subdivision.
    """

    fine_cell_ids: NDArray[np.int64]
    coarse_cell_ids: NDArray[np.int64]
    fine_reference_vertices: NDArray[np.float64]
    fine_reference_homogeneous: NDArray[np.float64] | None = None


def homogeneous_reference_point(
    row: Sequence[float] | NDArray[np.float64], /
) -> tuple[Fraction, ...] | None:
    """Rational reference coordinate of one homogeneous corner row; ``None`` if zero."""
    weights = tuple(Fraction(float(value)) for value in row)
    total = sum(weights, Fraction(0))
    if not total:
        return None
    return tuple(weight / total for weight in weights[1:])


def reference_homogeneous_preparation_bounds(
    rows: int, homogeneous_width: int, /
) -> tuple[int, int]:
    """Shape-only ``(work, bytes)`` upper bound of validating ``rows`` exact rows.

    Each row builds ``w`` binary64 Fractions, one exact sum and ``w - 1``
    quotients whose integers stay below ``3 * 1075 + w.bit_length()`` bits, and
    compares ``w - 1`` round-to-nearest views.
    """
    if rows < 0 or homogeneous_width < 2:
        raise ValueError(
            f"reference_homogeneous.bounds must have nonnegative rows and width at least 2; got {(rows, homogeneous_width)!r}"
        )
    bits = 3 * 1075 + homogeneous_width.bit_length()
    fraction_bytes = 256 + 2 * declared_integer_storage_bytes(bits)
    return (
        rows * 4 * homogeneous_width,
        rows * 2 * homogeneous_width * fraction_bytes,
    )


_BINARY64_EXACT_INTEGER = 1 << 53


def exact_binary64_values(value: ArrayLike, /) -> NDArray[np.float64] | None:
    """The binary64 array holding exactly the values of ``value``, or ``None``.

    Nothing is rounded: integers beyond the binary64 significand, extended
    floats, complex numbers, strings and exact rationals without a binary64
    value have no such array. A float64 input is returned unchanged.
    """
    array = np.asarray(value)
    dtype = array.dtype
    if dtype == np.float64:
        return array
    match dtype.kind:
        case "b":
            return array.astype(np.float64)
        case "i" | "u":
            values = array.astype(np.float64)
            if dtype.itemsize <= 4:
                return values
            wide = array > _BINARY64_EXACT_INTEGER
            if dtype.kind == "i":
                wide |= array < -_BINARY64_EXACT_INTEGER
            if any(
                original != int(converted)
                for original, converted in zip(
                    array[wide].tolist(), values[wide].tolist(), strict=True
                )
            ):
                return None
            return values
        case "f":
            values = array.astype(np.float64)
            if np.can_cast(dtype, np.float64, "safe"):
                return values
            restored = values.astype(dtype)
            same = (restored == array) | (np.isnan(restored) & np.isnan(array))
            return values if bool(np.all(same)) else None
        case "O":
            try:
                values = array.astype(np.float64)
                if all(
                    _exact_object_number(original, converted)
                    for original, converted in zip(
                        array.flat, values.ravel().tolist(), strict=True
                    )
                ):
                    return values
            except (TypeError, ValueError, ArithmeticError):
                pass
            return None
        case _:
            return None


def _exact_object_number(original: object, converted: float, /) -> bool:
    """Whether one object element equals its binary64 conversion exactly."""
    if converted != converted:
        return original != original
    match original:
        case float() | int() | Fraction() | Decimal():
            return original == converted
        case numbers.Rational():
            return Fraction(original.numerator, original.denominator) == converted
        case np.floating():
            return bool(original == original.dtype.type(converted))
        case _:
            return False


def require_binary64_witness(value: ArrayLike, name: str, /) -> NDArray[np.float64]:
    """Read a real reference bank as binary64, refusing any value it would round."""
    values = exact_binary64_values(value)
    if values is None:
        raise ValueError(
            f"{name} must hold binary64 values; got {np.asarray(value).dtype} values that binary64 would round"
        )
    return values


def exact_int64_values(value: ArrayLike, /) -> NDArray[np.int64] | None:
    """The int64 array of an integer-dtype ``value``, or ``None``.

    Only signed or unsigned integer dtypes whose values all fit int64 qualify;
    floats, booleans and objects are never truncated or wrapped into identifiers.
    """
    array = np.asarray(value)
    if array.dtype.kind not in ("i", "u") or (
        not np.can_cast(array.dtype, np.int64)
        and array.size
        and int(array.max()) > np.iinfo(np.int64).max
    ):
        return None
    return array.astype(np.int64, copy=False)


def require_int64_witness(value: ArrayLike, name: str, /) -> NDArray[np.int64]:
    """Read cell identifiers as int64, refusing any value a cast would change."""
    values = exact_int64_values(value)
    if values is None:
        raise ValueError(
            f"{name} must hold integer identifiers within int64; got {np.asarray(value).dtype} values"
        )
    return values


def require_reference_homogeneous(
    vertices: NDArray[np.float64],
    homogeneous: ArrayLike | None,
    name: str,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> NDArray[np.float64] | None:
    """Validate an exact homogeneous bank against its rounded execution vertices.

    Returns a read-only host copy, or ``None`` when no exact bank is retained.
    A supplied ``budget`` is charged before the rational pass; ``None`` declares
    an unbounded standalone check.
    """
    if homogeneous is None:
        return None
    rows = require_binary64_witness(vertices, f"{name}.reference_vertices")
    bank = np.array(
        require_binary64_witness(homogeneous, f"{name}.reference_homogeneous"),
        copy=True,
    )
    if (
        rows.ndim != 3
        or bank.shape != (*rows.shape[:2], rows.shape[2] + 1)
        or not np.all(np.isfinite(bank))
        or np.any(np.signbit(bank))
    ):
        raise ValueError(
            f"{name}.reference_homogeneous must be finite (cells, corners, d+1) rows of positive-signed weights aligned with the reference vertices; got incompatible bank"
        )
    flat_bank = bank.reshape((-1, bank.shape[-1]))
    flat_rows = rows.reshape((-1, rows.shape[-1]))
    exact = np.any(flat_bank != 0.0, axis=-1)
    # Each distinct (homogeneous row, rounded vertex) pair is proved once.
    distinct = np.unique(
        np.concatenate((flat_bank[exact], flat_rows[exact]), axis=1), axis=0
    )
    width = bank.shape[-1]
    work, storage = reference_homogeneous_preparation_bounds(distinct.shape[0], width)
    with nullcontext() if budget is None else budget.temporary_scope():
        if budget is not None:
            budget.reserve(work, storage)
        for record in distinct:
            point = homogeneous_reference_point(record[:width])
            if point is None or tuple(float(value) for value in point) != tuple(
                float(value) for value in record[width:]
            ):
                raise ValueError(
                    f"{name}.reference_vertices must be the round-to-nearest view of their exact homogeneous coordinates; got unequal binary64 corner"
                )
    bank.setflags(write=False)
    return bank


def exact_reference_corners(
    vertices: NDArray[np.float64],
    homogeneous: NDArray[np.float64] | None,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    """Rational reference corners of one validated row ``(corners, d)``."""
    corners: list[tuple[Fraction, ...]] = []
    rows = np.asarray(vertices, dtype=np.float64)
    for index in range(rows.shape[0]):
        point = (
            None
            if homogeneous is None
            else homogeneous_reference_point(homogeneous[index, :])
        )
        corners.append(
            tuple(Fraction(float(value)) for value in rows[index, :].tolist())
            if point is None
            else point
        )
    return tuple(corners)


def has_exact_homogeneous_rows(homogeneous: NDArray[np.float64] | None, /) -> bool:
    """Whether a validated bank names any nondyadic source-owned corner."""
    return homogeneous is not None and bool(np.any(np.asarray(homogeneous) != 0.0))


class CellGeometryTransitionPolicy(StrictModule, NonTrainableState):
    """Approximation, continuity, and resource controls of a geometry transition.

    ``coarsening`` selects the coarse-map approximation policy and
    ``coarsening_tolerance`` its absolute sup-norm bound in coordinate units.
    ``continuity_tolerance`` bounds the disagreement of a shared node evaluated
    from different cells and the rounding slack of algebraic claims, relative to the
    coordinate extent. ``maximum_evaluations`` bounds basis evaluations (points
    times local basis functions) before any is allocated, plus the algebraic
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
        tolerance = nonnegative_finite_float(coarsening_tolerance, "coarsening_tolerance")
        correspondence = parse(
            reconstruction, SurfaceGeometryCorrespondence, "reconstruction"
        )
        deformation_tolerance = nonnegative_finite_float(
            reconstruction_tolerance, "reconstruction_tolerance"
        )
        continuity = nonnegative_finite_float(
            continuity_tolerance, "continuity_tolerance"
        )
        if isinstance(maximum_evaluations, bool) or not isinstance(
            maximum_evaluations, (int, np.integer)
        ):
            raise TypeError(
                "geometry_transition.maximum_evaluations must be an integer; got noninteger value"
            )
        if maximum_evaluations < 1:
            raise ValueError(
                "geometry_transition.maximum_evaluations must be positive; got nonpositive value"
            )
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

    The equality flag states that every target map equals the source map it restricts (or
    the declared displacement extension) up to ``rounding_slack``.
    ``containment_defect`` is the largest barycentric excursion of a nested
    reference record outside its coarse reference simplex, ``continuity_residual`` the
    largest disagreement of one shared node evaluated from different cells, and
    ``approximation_bound`` the certified sup-norm deviation of coarsened maps from
    the fine maps they replace (zero without coarsening). ``measure_exact``
    describes algebraic integration before floating-point publication. Mixed mapped
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
            else nonnegative_finite_float(
                source_measure_error_bound, "source_measure_error_bound"
            )
        )
        self.target_measure_error_bound = (
            None
            if target_measure_error_bound is None
            else nonnegative_finite_float(
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


class GeometryTargetCellDim(Dim):
    """Target cells in concatenated block order."""


class GeometryCornerDim(Dim, minimum=2):
    """Corners of one simplex."""


class GeometryParentCornerDim(Dim, minimum=2):
    """Target-cell reference-corner capacity of the parent reference record bank."""


class GeometryCoarsenedCornerDim(Dim, minimum=2):
    """Fine source-cell corner capacity, independent of the target bank."""


class GeometryReferenceDim(Dim, minimum=1):
    """Reference-cell dimension."""


class GeometryHomogeneousDim(Dim, minimum=2):
    """Homogeneous weights of the simplex reference corners (dimension plus one)."""


class GeometryFineCellDim(Dim):
    """Source cells coarsened into target cells."""


class GeometryVertexDim(Dim, minimum=1):
    """Target mesh vertex rows."""


class GeometryAmbientDim(Dim, minimum=1):
    """Ambient coordinate dimension."""


class GeometryNodeDim(Dim, minimum=1):
    """Target coordinate nodes."""


class MaterialChartDeformation(StrictModule):
    """Bind a material correspondence to its four coordinate/topology identities.

    Abstract identity attributes do not add stored fields. Concrete material
    chart owners validate coverage, displacement, and physical inventory.
    """

    __strict_abstract__ = True

    source_topology_id: eqx.AbstractVar[str]
    target_topology_id: eqx.AbstractVar[str]
    source_geometry_id: eqx.AbstractVar[str]
    target_geometry_id: eqx.AbstractVar[str]
    deformation_id: eqx.AbstractVar[str]

    @abstractmethod
    def require_bound(
        self,
        source_mesh: CellMesh,
        source_geometry: CellGeometrySpec,
        target_mesh: CellMesh,
        target_geometry: CellGeometrySpec,
        /,
    ) -> None:
        """Validate correspondence identity and scientific evidence for both maps."""
        raise NotImplementedError


class CellGeometryTransition(StrictModule, NonTrainableState):
    """Accepted successor coordinate map with its source/reference witnesses.

    ``geometry`` is the full target `CellGeometrySpec` and
    ``vertex_coordinates`` the target mesh vertex rows it places (the corners the
    target `CellMesh` must carry). Target cell ``c`` (concatenated block order,
    ``target_cell_ids[c]``) was evaluated from source cell ``parent_cell_ids[c]`` at
    parent reference corners ``parent_reference_vertices[c]``; a coarsened target
    cell has parent ``-1`` and its source cells are the ``coarsened_*`` witnesses.
    Geometry node ``n`` was evaluated as local node ``node_owner_locals[n]`` of
    target cell ``node_owner_cells[n]``.
    A bounded chart deformation instead retains its ``chart_deformation``
    certificate. Its material correspondence and physical area inventories
    cannot be recovered from nested parent corners or the
    ``chart_correspondence_id`` alone.

    ``parent_reference_homogeneous`` and ``coarsened_reference_homogeneous``
    retain the exact source-owned simplex reference corners of the matching
    vertex banks (see `NestedReferenceWitnesses`); the binary64 corners are then
    their round-to-nearest execution views, never the exact geometry.
    """

    __strict_contract__ = True

    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    geometry: CellGeometrySpec
    vertex_coordinates: Float64[GeometryVertexDim, GeometryAmbientDim]
    target_cell_ids: Int64[GeometryTargetCellDim]
    parent_cell_ids: Int64[GeometryTargetCellDim]
    parent_reference_vertices: Float64[
        GeometryTargetCellDim, GeometryParentCornerDim, GeometryReferenceDim
    ]
    coarsened_cell_ids: Int64[GeometryFineCellDim]
    coarsened_into_ids: Int64[GeometryFineCellDim]
    coarsened_reference_vertices: Float64[
        GeometryFineCellDim, GeometryCoarsenedCornerDim, GeometryReferenceDim
    ]
    node_owner_cells: Int64[GeometryNodeDim] | None
    node_owner_locals: Int32[GeometryNodeDim] | None
    evidence: CellGeometryTransitionEvidence
    policy_id: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)
    chart_correspondence_id: str | None = eqx.field(static=True, default=None)
    chart_deformation: MaterialChartDeformation | None = None
    parent_reference_homogeneous: (
        Float64[GeometryTargetCellDim, GeometryParentCornerDim, GeometryHomogeneousDim]
        | None
    ) = None
    coarsened_reference_homogeneous: (
        Float64[GeometryFineCellDim, GeometryCoarsenedCornerDim, GeometryHomogeneousDim]
        | None
    ) = None

    def __check_init__(self) -> None:
        require_reference_homogeneous(
            np.asarray(self.parent_reference_vertices),
            None
            if self.parent_reference_homogeneous is None
            else np.asarray(self.parent_reference_homogeneous),
            "CellGeometryTransition.parent",
        )
        require_reference_homogeneous(
            np.asarray(self.coarsened_reference_vertices),
            None
            if self.coarsened_reference_homogeneous is None
            else np.asarray(self.coarsened_reference_homogeneous),
            "CellGeometryTransition.coarsened",
        )
        if (
            self.parent_reference_homogeneous is not None
            or self.coarsened_reference_homogeneous is not None
        ) and self.transition_id != exact_root_transition_id(
            source_topology_id=self.source_topology_id,
            target_topology_id=self.target_topology_id,
            source_geometry_id=self.source_geometry_id,
            target_geometry_id=self.target_geometry_id,
            target_cell_ids=self.target_cell_ids,
            parent_cell_ids=self.parent_cell_ids,
            parent_reference_vertices=self.parent_reference_vertices,
            parent_reference_homogeneous=self.parent_reference_homogeneous,
            coarsened_cell_ids=self.coarsened_cell_ids,
            coarsened_into_ids=self.coarsened_into_ids,
            coarsened_reference_vertices=self.coarsened_reference_vertices,
            coarsened_reference_homogeneous=self.coarsened_reference_homogeneous,
            evidence_id=self.evidence.evidence_id,
            policy_id=self.policy_id,
        ):
            raise ValueError(
                "CellGeometryTransition.transition_id must bind the full rational reference banks; got stale or forged identity"
            )

    def parent_reference_witnesses(self) -> NestedReferenceWitnesses:
        """Nested parent rows with their rational bank, excluding coarsened targets."""
        parents = np.asarray(self.parent_cell_ids, dtype=np.int64)
        nested = parents >= 0
        bank = self.parent_reference_homogeneous
        return NestedReferenceWitnesses(
            np.asarray(self.target_cell_ids, dtype=np.int64)[nested],
            parents[nested],
            np.asarray(self.parent_reference_vertices, dtype=np.float64)[nested],
            None if bank is None else np.asarray(bank, dtype=np.float64)[nested],
        )

    def coarsening_witnesses(self) -> NestedReferenceWitnesses:
        """Fine-to-coarse rows with their rational bank."""
        bank = self.coarsened_reference_homogeneous
        return NestedReferenceWitnesses(
            np.asarray(self.coarsened_cell_ids, dtype=np.int64),
            np.asarray(self.coarsened_into_ids, dtype=np.int64),
            np.asarray(self.coarsened_reference_vertices, dtype=np.float64),
            None if bank is None else np.asarray(bank, dtype=np.float64),
        )


def exact_root_transition_id(
    *,
    source_topology_id: str,
    target_topology_id: str,
    source_geometry_id: str,
    target_geometry_id: str,
    target_cell_ids: ArrayLike,
    parent_cell_ids: ArrayLike,
    parent_reference_vertices: ArrayLike,
    parent_reference_homogeneous: ArrayLike | None,
    coarsened_cell_ids: ArrayLike,
    coarsened_into_ids: ArrayLike,
    coarsened_reference_vertices: ArrayLike,
    coarsened_reference_homogeneous: ArrayLike | None,
    evidence_id: str,
    policy_id: str,
) -> str:
    """Deterministic identity of a transition retaining rational source-owned corners.

    Every stored reference array, including the full binary64 homogeneous
    banks (not only their rational points), is bound, so replay recomputes and
    authenticates it. Transitions without banks keep their route identities.
    """
    return canonical_fingerprint(
        {
            "kind": "cell-geometry-transition-exact-roots",
            "source_topology": source_topology_id,
            "target_topology": target_topology_id,
            "source_geometry": source_geometry_id,
            "target_geometry": target_geometry_id,
            "target_cells": array_tree_fingerprint(np.asarray(target_cell_ids)),
            "parents": array_tree_fingerprint(np.asarray(parent_cell_ids)),
            "reference": array_tree_fingerprint(np.asarray(parent_reference_vertices)),
            "reference_homogeneous": None
            if parent_reference_homogeneous is None
            else array_tree_fingerprint(np.asarray(parent_reference_homogeneous)),
            "coarsened": array_tree_fingerprint(
                (np.asarray(coarsened_cell_ids), np.asarray(coarsened_into_ids))
            ),
            "coarsened_reference": array_tree_fingerprint(
                np.asarray(coarsened_reference_vertices)
            ),
            "coarsened_homogeneous": None
            if coarsened_reference_homogeneous is None
            else array_tree_fingerprint(np.asarray(coarsened_reference_homogeneous)),
            "evidence": evidence_id,
            "policy": policy_id,
        }
    )
