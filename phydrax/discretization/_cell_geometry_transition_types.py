#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Transition records and material correspondence contracts below transfer executors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal, NamedTuple, TypeAlias

import equinox as eqx
import numpy as np
from numpy.typing import NDArray

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_finite_float
from ..typing import Dim, Float64, Int32, Int64, parse
from ._cell_geometry import CellGeometrySpec
from ._cell_mesh import CellMesh


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
    """

    fine_cell_ids: NDArray[np.int64]
    coarse_cell_ids: NDArray[np.int64]
    fine_reference_vertices: NDArray[np.float64]


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
    A bounded chart deformation instead retains its     ``chart_deformation`` certificate. Its material correspondence and physical
    area inventories cannot be recovered from nested parent corners or the
    ``chart_correspondence_id`` alone.
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
