#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Metric-driven tetrahedral remeshing on the native constrained cavity owner.

This is host-only epoch preparation. Native edits own exact legality and atomic
rollback; this schedule owns the metric objective, source ancestry and budgets.
No scalar equivalent size is substituted for a tensor metric.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from time import monotonic
from typing import final, NamedTuple, TypeAlias, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, custom_jvp, lax
from jax.typing import ArrayLike

from .._bvh import bvh_overlap_pair_blocks, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import PredicateMode, PredicateSign
from .._meshcore import (
    current_native_execution_budget,
    exact_orient2d,
    exact_orient3d,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionBudget,
    TetMesh3D,
    TetMeshArrays,
    TetMeshEdgeConstruction,
    TetMeshInsertionPlan,
    TetMeshSourceComplex,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._exact_plc_geometry import (
    _ExactPlcBudget,
    _in_entity,
    _plc_storage_bounds,
    _point,
    ExactPlcCellGeometrySource,
    ExactPoint,
)
from ..discretization._reference_cell import reference_cell_topology
from ..linalg import (
    DenseQR,
    GeneralizedLSMR,
    JacobianLinearOperator,
    LinearSolvePolicy,
    orthonormal_frame,
    RankPolicy,
    SmallLinearSolvePlan,
    solve_small_linear,
    TolerancePolicy,
)
from ..nonlinear import quadratic_event_bound
from ..optim import (
    AbstractLeastSquaresTrialPolicy,
    compose_active_gradient_bank,
    least_squares,
    LeastSquaresTrialResult,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationStatus,
    OptimizationTermination,
)
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._lineage import EntityLineageKind
from ._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._metric import interpolate_mesh_metric, metric_edge_lengths, metric_simplex_quality
from ._sizing import SizeCompliancePolicy, SizeControlStrength, UniformSizeControl
from ._topology_edit import (
    CellTopologyEdit,
    entity_keys,
    EntityRelations,
    key_rows,
    source_family_blocks,
)
from ._trace import MeshingStageKind


_LOWER = 1.0 / np.sqrt(2.0)
_UPPER = np.sqrt(2.0)
_TET_EDGES = np.asarray(
    reference_cell_topology("tetrahedron").entities[1], dtype=np.int32
)
_FACES = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
_LINEAGE_RANK = {
    EntityLineageKind.PRESERVED: 0,
    EntityLineageKind.REFINED_FROM: 1,
    EntityLineageKind.RELOCATED: 2,
    EntityLineageKind.SWAPPED_FROM: 3,
    EntityLineageKind.COLLAPSED_INTO: 4,
}


def _lineage_operation_kind(
    kind: EntityLineageKind, previous: tuple[int, ...], /
) -> EntityLineageKind:
    return max(
        (kind, *(EntityLineageKind(value) for value in previous)),
        key=_LINEAGE_RANK.__getitem__,
    )


class MetricRemeshingStatus(StrEnum):
    COMPLETE = "complete"
    STALLED = "stalled"
    PASS_LIMIT = "pass_limit"
    RESOURCE_LIMIT = "resource_limit"


class MetricRemeshingCriterion(StrEnum):
    UNIT_MESH = "unit_mesh"
    RELOCATION_FIXED_POINT = "relocation_fixed_point"
    SIZE_STATISTICS = "size_statistics"


@final
class MetricRemeshingEvidence(StrictModule, NonTrainableState):
    """Closed-goal outcome and actual measurements; no feature edges excluded."""

    status: MetricRemeshingStatus = eqx.field(static=True)
    criterion: MetricRemeshingCriterion = eqx.field(static=True)
    passes: int = eqx.field(static=True)
    splits: int = eqx.field(static=True)
    compound_splits: int = eqx.field(static=True)
    expanded_insertions: int = eqx.field(static=True)
    collapses: int = eqx.field(static=True)
    flips: int = eqx.field(static=True)
    relocations: int = eqx.field(static=True)
    rejected_operations: int = eqx.field(static=True)
    work_units: int = eqx.field(static=True)
    out_of_range_edges: int = eqx.field(static=True)
    minimum_metric_length: float = eqx.field(static=True)
    maximum_metric_length: float = eqx.field(static=True)
    minimum_metric_quality: float = eqx.field(static=True)
    maximum_fidelity_bound: float = eqx.field(static=True)
    predicate_mode: PredicateMode = eqx.field(static=True)
    split_parameters: tuple[float, ...] = eqx.field(static=True)
    lower_metric_length: float = eqx.field(static=True)
    upper_metric_length: float = eqx.field(static=True)
    operation_attempts: int = eqx.field(static=True)
    native_memory_evidence: tuple[int, ...] = eqx.field(static=True)
    source_candidate_pairs: int = eqx.field(static=True)
    size_requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    size_achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    size_issues: tuple[str, ...] = eqx.field(static=True)
    metric_quality_floor: float | None = eqx.field(static=True)
    shape_radius_edge_bound: float | None = eqx.field(static=True)
    shape_minimum_dihedral_degrees: float | None = eqx.field(static=True)
    resource_message: str | None = eqx.field(static=True)
    resource_requested: tuple[tuple[str, float], ...] = eqx.field(static=True)
    resource_achieved: tuple[tuple[str, float], ...] = eqx.field(static=True)
    optimizer_status_counts: tuple[int, ...] = eqx.field(static=True)
    optimizer_terminations: tuple[tuple[int, int], ...] = eqx.field(static=True)
    optimizer_method_id: str | None = eqx.field(static=True)
    optimizer_tolerances: tuple[float, ...] = eqx.field(static=True)
    optimizer_iterations: int = eqx.field(static=True)
    optimizer_evaluations: int = eqx.field(static=True)
    optimizer_constraint_evaluations: int = eqx.field(static=True)
    optimizer_proposal_evaluations: int = eqx.field(static=True)
    optimizer_refused_proposals: int = eqx.field(static=True)
    optimizer_refusal_counts: tuple[tuple[str, int], ...] = eqx.field(static=True)
    optimizer_host_work_units: int = eqx.field(static=True)
    optimizer_scratch_required_bytes: int = eqx.field(static=True)
    native_work_units: int = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord | None
    metric_arc_bounds: Array | None
    metric_arc_edge_ids: Array | None
    metric_arc_binding: str | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: MetricRemeshingStatus,
        *,
        passes: int,
        counts: tuple[int, int, int, int],
        rejected_operations: int,
        work_units: int,
        lengths: np.ndarray,
        quality: np.ndarray,
        maximum_fidelity_bound: float = 0.0,
        criterion: MetricRemeshingCriterion = MetricRemeshingCriterion.UNIT_MESH,
        split_parameters: tuple[float, ...] = (),
        lower_metric_length: float = _LOWER,
        upper_metric_length: float = _UPPER,
        operation_attempts: int = 0,
        native_memory_evidence: tuple[int, ...] = (),
        size_requested: tuple[tuple[str, float], ...] = (),
        size_achieved: tuple[tuple[str, float], ...] = (),
        size_issues: tuple[str, ...] = (),
        source_candidate_pairs: int = 0,
        metric_quality_floor: float | None = None,
        shape_radius_edge_bound: float | None = None,
        shape_minimum_dihedral_degrees: float | None = None,
        resource_message: str | None = None,
        resource_requested: tuple[tuple[str, float], ...] = (),
        resource_achieved: tuple[tuple[str, float], ...] = (),
        compound_splits: int = 0,
        expanded_insertions: int = 0,
        optimizer_status_counts: tuple[int, ...] = (),
        optimizer_terminations: tuple[tuple[int, int], ...] = (),
        optimizer_method_id: str | None = None,
        optimizer_tolerances: tuple[float, ...] = (),
        optimizer_iterations: int = 0,
        optimizer_evaluations: int = 0,
        optimizer_constraint_evaluations: int = 0,
        optimizer_proposal_evaluations: int = 0,
        optimizer_refused_proposals: int = 0,
        optimizer_refusal_counts: tuple[tuple[str, int], ...] = (),
        optimizer_host_work_units: int = 0,
        optimizer_scratch_required_bytes: int = 0,
        native_work_units: int | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
        metric_arc_bounds: np.ndarray | None = None,
        metric_arc_edge_ids: np.ndarray | None = None,
        metric_arc_binding: str | None = None,
    ) -> None:
        if not isinstance(status, MetricRemeshingStatus):
            raise TypeError("status must be MetricRemeshingStatus.")
        if not isinstance(criterion, MetricRemeshingCriterion):
            raise TypeError("criterion must be MetricRemeshingCriterion.")
        if execution_evidence is not None and not isinstance(
            execution_evidence, NativeExecutionRecord
        ):
            raise TypeError("execution_evidence must be an actual ended native record.")
        if (metric_arc_bounds is None) != (metric_arc_edge_ids is None) or (
            metric_arc_bounds is None
        ) != (metric_arc_binding is None):
            raise ValueError(
                "Metric arc bounds, scientific edge IDs and source binding are required together."
            )
        if metric_arc_bounds is not None and metric_arc_edge_ids is not None:
            if (
                metric_arc_bounds.shape != (lengths.size, 2)
                or metric_arc_edge_ids.shape != (lengths.size, 2)
                or not np.issubdtype(metric_arc_edge_ids.dtype, np.integer)
                or not np.all(np.isfinite(metric_arc_bounds))
                or np.any(metric_arc_bounds[:, 0] < 0)
                or np.any(metric_arc_bounds[:, 1] < metric_arc_bounds[:, 0])
                or not metric_arc_binding
            ):
                raise ValueError(
                    "Metric arc evidence requires finite outward bounds bound to actual scientific edges."
                )
        if any(
            not np.isfinite(value) or not 0.0 < value < 1.0 for value in split_parameters
        ):
            raise ValueError(
                "Split parameters must be exact finite edge-interior fractions."
            )
        numbers = (
            expanded_insertions,
            passes,
            *counts,
            compound_splits,
            rejected_operations,
            work_units,
            operation_attempts,
            source_candidate_pairs,
            *optimizer_status_counts,
            optimizer_iterations,
            optimizer_evaluations,
            optimizer_refused_proposals,
            optimizer_host_work_units,
            optimizer_scratch_required_bytes,
            optimizer_constraint_evaluations,
            optimizer_proposal_evaluations,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in numbers
        ):
            raise ValueError("Operation and work counters must be nonnegative integers.")
        if (
            lengths.ndim != 1
            or quality.ndim != 1
            or not np.all(np.isfinite(lengths))
            or not np.all(np.isfinite(quality))
        ):
            raise ValueError("Metric measurements must be finite vectors.")
        if not np.isfinite(maximum_fidelity_bound) or maximum_fidelity_bound < 0:
            raise ValueError("The fidelity bound must be finite and nonnegative.")
        if (
            not np.isfinite(lower_metric_length)
            or not np.isfinite(upper_metric_length)
            or not 0.0 <= lower_metric_length <= upper_metric_length
            or upper_metric_length <= 0.0
        ):
            raise ValueError(
                "Metric edge bounds must form a finite nonnegative interval."
            )
        if len(native_memory_evidence) not in (0, 6) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in native_memory_evidence
        ):
            raise ValueError(
                "Native memory evidence must be the actual six-counter table."
            )
        if len(optimizer_status_counts) not in (0, len(OptimizationStatus)) or any(
            steps < 1 or evaluations < 1 for steps, evaluations in optimizer_terminations
        ):
            raise ValueError(
                "Optimizer status and termination evidence must be complete."
            )
        if optimizer_method_id is not None and not optimizer_method_id:
            raise ValueError("An optimizer method identifier must be non-empty.")
        if len(optimizer_tolerances) not in (0, 4) or any(
            not np.isfinite(value) or value < 0.0 for value in optimizer_tolerances
        ):
            raise ValueError(
                "Optimizer tolerances must be its four actual finite nonnegative thresholds."
            )
        native_work = work_units if native_work_units is None else native_work_units
        if (
            isinstance(native_work, bool)
            or not isinstance(native_work, int)
            or native_work < 0
            or native_work > work_units
        ):
            raise ValueError(
                "Native work must be a nonnegative part of total charged work."
            )
        if any(
            not name or not isinstance(count, int) or count < 0
            for name, count in optimizer_refusal_counts
        ):
            raise ValueError(
                "Optimizer refusal evidence must contain named nonnegative counts."
            )
        if any(
            not key or not np.isfinite(value)
            for key, value in (*size_requested, *size_achieved)
        ):
            raise ValueError("Physical size evidence must contain named finite values.")
        if criterion is MetricRemeshingCriterion.SIZE_STATISTICS and (
            not size_requested or not size_achieved
        ):
            raise ValueError(
                "Physical size remeshing must retain the original compliance evidence."
            )
        if status is MetricRemeshingStatus.COMPLETE and size_issues:
            raise ValueError(
                "A completed physical size goal cannot retain failed obligations."
            )
        if metric_quality_floor is not None and (
            not np.isfinite(metric_quality_floor)
            or not 0.0 <= metric_quality_floor <= 1.0
        ):
            raise ValueError("The metric quality floor must lie in [0, 1].")
        if shape_radius_edge_bound is not None and (
            not np.isfinite(shape_radius_edge_bound) or shape_radius_edge_bound <= 0.0
        ):
            raise ValueError(
                "The radius-edge proposal bound must be finite and positive."
            )
        if shape_minimum_dihedral_degrees is not None and (
            not np.isfinite(shape_minimum_dihedral_degrees)
            or not 0.0 <= shape_minimum_dihedral_degrees <= 180.0
        ):
            raise ValueError("The dihedral proposal bound must be finite and in degrees.")
        self.status = status
        self.criterion = criterion
        self.passes = passes
        self.splits, self.collapses, self.flips, self.relocations = counts
        self.compound_splits = compound_splits
        self.expanded_insertions = expanded_insertions
        self.rejected_operations = rejected_operations
        self.work_units = work_units
        self.lower_metric_length = float(lower_metric_length)
        self.upper_metric_length = float(upper_metric_length)
        self.operation_attempts = operation_attempts
        self.native_memory_evidence = native_memory_evidence
        self.size_requested = size_requested
        self.size_achieved = size_achieved
        self.size_issues = size_issues
        self.source_candidate_pairs = source_candidate_pairs
        self.metric_quality_floor = metric_quality_floor
        self.shape_radius_edge_bound = shape_radius_edge_bound
        self.shape_minimum_dihedral_degrees = shape_minimum_dihedral_degrees
        self.resource_message = resource_message
        self.resource_requested = resource_requested
        self.resource_achieved = resource_achieved
        self.optimizer_status_counts = optimizer_status_counts
        self.optimizer_terminations = optimizer_terminations
        self.optimizer_iterations = optimizer_iterations
        self.optimizer_evaluations = optimizer_evaluations
        self.optimizer_constraint_evaluations = optimizer_constraint_evaluations
        self.optimizer_proposal_evaluations = optimizer_proposal_evaluations
        self.optimizer_refused_proposals = optimizer_refused_proposals
        self.optimizer_refusal_counts = optimizer_refusal_counts
        self.optimizer_host_work_units = optimizer_host_work_units
        self.optimizer_scratch_required_bytes = optimizer_scratch_required_bytes
        self.native_work_units = native_work
        self.execution_evidence = execution_evidence
        self.metric_arc_bounds = (
            None
            if metric_arc_bounds is None
            else jnp.asarray(metric_arc_bounds, dtype=jnp.float64)
        )
        self.metric_arc_edge_ids = (
            None
            if metric_arc_edge_ids is None
            else jnp.asarray(metric_arc_edge_ids, dtype=jnp.int64)
        )
        self.metric_arc_binding = metric_arc_binding
        self.optimizer_method_id = optimizer_method_id
        self.optimizer_tolerances = optimizer_tolerances
        self.out_of_range_edges = int(
            np.count_nonzero(
                (lengths < lower_metric_length) | (lengths > upper_metric_length)
            )
        )
        self.minimum_metric_length = (
            float(np.min(lengths, initial=np.inf)) if lengths.size else 0.0
        )
        self.maximum_metric_length = float(np.max(lengths, initial=0.0))
        self.minimum_metric_quality = float(np.min(quality, initial=1.0))
        self.maximum_fidelity_bound = maximum_fidelity_bound
        self.predicate_mode = PredicateMode.EXACT
        self.split_parameters = split_parameters
        self.evidence_id = self._source_identity(metric_arc_bounds, metric_arc_edge_ids)

    def _source_identity(
        self, metric_arc_bounds: object, metric_arc_edge_ids: object
    ) -> str:
        return canonical_fingerprint(
            {
                "kind": "native-metric-remeshing",
                "status": self.status.value,
                "criterion": self.criterion.value,
                "passes": self.passes,
                "counts": (self.splits, self.collapses, self.flips, self.relocations),
                "rejected": self.rejected_operations,
                "work": self.work_units,
                "out_of_range": self.out_of_range_edges,
                "length_min": self.minimum_metric_length,
                "length_max": self.maximum_metric_length,
                "quality_min": self.minimum_metric_quality,
                "fidelity": self.maximum_fidelity_bound,
                "predicate_mode": self.predicate_mode.value,
                "split_parameters": self.split_parameters,
                "edge_band": (self.lower_metric_length, self.upper_metric_length),
                "operation_attempts": self.operation_attempts,
                "native_memory_evidence": self.native_memory_evidence,
                "size_requested": self.size_requested,
                "size_achieved": self.size_achieved,
                "size_issues": self.size_issues,
                "source_candidate_pairs": self.source_candidate_pairs,
                "metric_quality_floor": self.metric_quality_floor,
                "shape_radius_edge_bound": self.shape_radius_edge_bound,
                "shape_minimum_dihedral_degrees": self.shape_minimum_dihedral_degrees,
                "resource_message": self.resource_message,
                "resource_requested": self.resource_requested,
                "resource_achieved": self.resource_achieved,
                "compound_splits": self.compound_splits,
                "expanded_insertions": self.expanded_insertions,
                "optimizer_status_counts": self.optimizer_status_counts,
                "optimizer_terminations": self.optimizer_terminations,
                "optimizer_method_id": self.optimizer_method_id,
                "optimizer_tolerances": self.optimizer_tolerances,
                "optimizer_iterations": self.optimizer_iterations,
                "optimizer_evaluations": self.optimizer_evaluations,
                "optimizer_constraint_evaluations": self.optimizer_constraint_evaluations,
                "optimizer_proposal_evaluations": self.optimizer_proposal_evaluations,
                "optimizer_refused_proposals": self.optimizer_refused_proposals,
                "optimizer_refusal_counts": self.optimizer_refusal_counts,
                "optimizer_host_work_units": self.optimizer_host_work_units,
                "optimizer_scratch_required_bytes": self.optimizer_scratch_required_bytes,
                "native_work_units": self.native_work_units,
                "metric_arc_bounds": None
                if metric_arc_bounds is None
                else array_tree_fingerprint(metric_arc_bounds),
                "metric_arc_edge_ids": None
                if metric_arc_edge_ids is None
                else array_tree_fingerprint(metric_arc_edge_ids),
                "metric_arc_binding": self.metric_arc_binding,
            }
        )

    def validate_restored(self) -> None:
        """Authenticate retained aggregates, never invent discarded measurement rows."""
        if (
            type(self.status) is not MetricRemeshingStatus
            or type(self.criterion) is not MetricRemeshingCriterion
        ):
            raise TypeError(
                "Metric evidence requires its exact registered status and criterion."
            )
        if self.predicate_mode is not PredicateMode.EXACT:
            raise ValueError("Metric evidence changed its original exact predicate law.")
        counters = (
            self.passes,
            self.splits,
            self.collapses,
            self.flips,
            self.relocations,
            self.compound_splits,
            self.expanded_insertions,
            self.rejected_operations,
            self.work_units,
            self.out_of_range_edges,
            self.operation_attempts,
            self.source_candidate_pairs,
            *self.optimizer_status_counts,
            self.optimizer_iterations,
            self.optimizer_evaluations,
            self.optimizer_constraint_evaluations,
            self.optimizer_proposal_evaluations,
            self.optimizer_refused_proposals,
            self.optimizer_host_work_units,
            self.optimizer_scratch_required_bytes,
            self.native_work_units,
        )
        if (
            any(type(value) is not int or value < 0 for value in counters)
            or self.native_work_units > self.work_units
        ):
            raise ValueError(
                "Restored metric counters violate their original work domains."
            )
        if (
            not all(
                np.isfinite(value)
                for value in (
                    self.minimum_metric_length,
                    self.maximum_metric_length,
                    self.minimum_metric_quality,
                    self.maximum_fidelity_bound,
                    self.lower_metric_length,
                    self.upper_metric_length,
                )
            )
            or self.maximum_fidelity_bound < 0
        ):
            raise ValueError(
                "Restored metric measurements must be finite with nonnegative fidelity."
            )
        if (
            not 0 <= self.lower_metric_length <= self.upper_metric_length
            or self.upper_metric_length <= 0
        ):
            raise ValueError("Restored metric edge band is invalid.")
        if len(self.native_memory_evidence) not in (0, 6) or any(
            type(value) is not int or value < 0 for value in self.native_memory_evidence
        ):
            raise ValueError(
                "Restored metric memory evidence lacks the actual six-counter domain."
            )
        if len(self.optimizer_status_counts) not in (0, len(OptimizationStatus)) or any(
            type(steps) is not int
            or type(evaluations) is not int
            or steps < 1
            or evaluations < 1
            for steps, evaluations in self.optimizer_terminations
        ):
            raise ValueError("Restored optimizer evidence is incomplete.")
        if self.optimizer_method_id is not None and (
            type(self.optimizer_method_id) is not str or not self.optimizer_method_id
        ):
            raise ValueError("Restored optimizer identity must be nonempty.")
        if len(self.optimizer_tolerances) not in (0, 4) or any(
            not np.isfinite(value) or value < 0 for value in self.optimizer_tolerances
        ):
            raise ValueError("Restored optimizer tolerances are invalid.")
        if any(
            type(name) is not str or not name or type(count) is not int or count < 0
            for name, count in self.optimizer_refusal_counts
        ):
            raise ValueError("Restored optimizer refusal counts are invalid.")
        if any(
            not np.isfinite(value) or not 0 < value < 1 for value in self.split_parameters
        ):
            raise ValueError(
                "Restored split parameters leave the original edge interior."
            )
        if any(
            type(key) is not str or not key or not np.isfinite(value)
            for key, value in (
                *self.size_requested,
                *self.size_achieved,
                *self.resource_requested,
                *self.resource_achieved,
            )
        ):
            raise ValueError(
                "Restored physical/resource evidence requires named finite quantities."
            )
        if self.criterion is MetricRemeshingCriterion.SIZE_STATISTICS and (
            not self.size_requested or not self.size_achieved
        ):
            raise ValueError(
                "Restored physical size evidence omits its original obligations."
            )
        if self.status is MetricRemeshingStatus.COMPLETE and self.size_issues:
            raise ValueError(
                "Restored completed metric evidence retains failed obligations."
            )
        if self.metric_quality_floor is not None and (
            not np.isfinite(self.metric_quality_floor)
            or not 0 <= self.metric_quality_floor <= 1
        ):
            raise ValueError("Restored metric quality floor is invalid.")
        if self.shape_radius_edge_bound is not None and (
            not np.isfinite(self.shape_radius_edge_bound)
            or self.shape_radius_edge_bound <= 0
        ):
            raise ValueError("Restored radius-edge bound is invalid.")
        if self.shape_minimum_dihedral_degrees is not None and (
            not np.isfinite(self.shape_minimum_dihedral_degrees)
            or not 0 <= self.shape_minimum_dihedral_degrees <= 180
        ):
            raise ValueError("Restored dihedral bound is invalid.")
        if (self.metric_arc_bounds is None) != (self.metric_arc_edge_ids is None) or (
            self.metric_arc_bounds is None
        ) != (self.metric_arc_binding is None):
            raise ValueError("Restored metric arc source binding is incomplete.")
        if self.metric_arc_bounds is not None:
            bounds, identifiers = (
                np.asarray(self.metric_arc_bounds),
                np.asarray(self.metric_arc_edge_ids),
            )
            if (
                bounds.ndim != 2
                or bounds.shape[1] != 2
                or identifiers.shape != bounds.shape
                or identifiers.dtype.kind not in "iu"
                or not np.all(np.isfinite(bounds))
                or np.any(bounds[:, 0] < 0)
                or np.any(bounds[:, 1] < bounds[:, 0])
                or type(self.metric_arc_binding) is not str
                or not self.metric_arc_binding
            ):
                raise ValueError(
                    "Restored metric arcs changed their scientific bound/edge bank."
                )
        if self.execution_evidence is not None:
            if type(self.execution_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    "Restored metric execution evidence requires an actual native record."
                )
            self.execution_evidence.require_valid()
        if self.evidence_id != self._source_identity(
            self.metric_arc_bounds, self.metric_arc_edge_ids
        ):
            raise ValueError(
                "Restored metric evidence differs from its original aggregate identity."
            )

    @property
    def converged(self) -> bool:
        return self.status is MetricRemeshingStatus.COMPLETE

    @property
    def stalled(self) -> bool:
        return self.status is MetricRemeshingStatus.STALLED


def _source_location_budget(maximum_pairs: int, candidates: int, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.RESOURCE_EXHAUSTED,
        "Metric source-location candidate budget is exhausted.",
        stage=MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        requested=(("source_location_candidates", float(maximum_pairs)),),
        achieved=(("source_location_candidates", float(candidates)),),
    )


def _stage_source_limits(
    mesh: CellMesh, maximum_vertices: int, maximum_cells: int, /
) -> None:
    vertices = mesh.coordinates.shape[0]
    cells = sum(block.cell_count for block in mesh.blocks)
    if vertices > maximum_vertices or cells > maximum_cells:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The metric construction budget cannot stage the accepted source.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
            requested=(
                ("vertices", float(maximum_vertices)),
                ("cells", float(maximum_cells)),
            ),
            achieved=(("vertices", float(vertices)), ("cells", float(cells))),
        )


class TetraMetricOutcome(NamedTuple):
    edit: CellTopologyEdit
    metric: np.ndarray
    evidence: MetricRemeshingEvidence
    exact_source: ExactPlcCellGeometrySource | None = None


def _fresh_identifiers(first: int, count: int, /) -> np.ndarray:
    maximum = np.iinfo(np.int64).max
    if count < 0:
        raise RuntimeError("A native snapshot lost retained vertex slots.")
    if count and first + count - 1 > maximum:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Scientific global-identifier capacity is exhausted.",
            stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
            requested=(("new_global_ids", float(count)),),
            achieved=(("available_global_ids", float(max(0, maximum - first + 1))),),
        )
    return (
        np.arange(count, dtype=np.int64) + first
        if count
        else np.empty((0,), dtype=np.int64)
    )


@dataclass(slots=True)
class _Ancestry:
    cells: dict[tuple[int, ...], tuple[int, set[int]]]
    next_cell_id: int
    vertex_sources: list[set[int]]
    vertex_successor: dict[int, int] | None = None
    cell_kinds: dict[int, int] | None = None
    relocated_vertices: set[int] | None = None


def _edges(cells: np.ndarray, /) -> np.ndarray:
    local = np.asarray(tuple(combinations(range(cells.shape[1]), 2)), dtype=np.int32)
    return np.unique(np.sort(cells[:, local].reshape((-1, 2)), axis=1), axis=0)


def _resolved_collapses(
    identifiers: np.ndarray, source_count: int, successor: dict[int, int] | None, /
) -> dict[int, int]:
    result: dict[int, int] = {}
    if successor is None:
        return result
    for vertex in range(source_count):
        current = vertex
        visited: set[int] = set()
        while current in successor:
            if current in visited:
                raise RuntimeError("A collapse lineage contains a cycle.")
            visited.add(current)
            current = successor[current]
        if current != vertex:
            result[int(identifiers[vertex])] = int(identifiers[current])
    return result


def _subentity_relations(
    mesh: CellMesh,
    dimension: int,
    cells: np.ndarray,
    vertex_ids: np.ndarray,
    vertex_sources: list[set[int]],
    /,
    *,
    collapsed_vertices: dict[int, int] | None = None,
) -> EntityRelations:
    """Exact split/collapse ancestry from construction witnesses, not coordinates."""
    local = np.asarray(
        tuple(combinations(range(cells.shape[1]), dimension + 1)), dtype=np.int32
    )
    targets = np.unique(
        np.sort(cells[:, local].reshape((-1, dimension + 1)), axis=1), axis=0
    )
    source_keys = entity_keys(mesh, dimension)
    incidence: dict[int, set[int]] = {}
    for row, key in enumerate(source_keys.tolist()):
        for vertex in key:
            incidence.setdefault(vertex, set()).add(row)
    sources, descendants, kinds = [], [], []
    target_keys = {
        tuple(sorted(vertex_ids[target].tolist())) for target in targets.tolist()
    }
    for target in targets.tolist():
        key = sorted(vertex_ids[target].tolist())
        support = set().union(*(vertex_sources[vertex] for vertex in target))
        if len(support) > dimension + 1:
            continue
        candidates: set[int] | None = None
        for vertex in support:
            rows = incidence.get(vertex, set())
            candidates = rows.copy() if candidates is None else candidates & rows
        for row in sorted(candidates if candidates is not None else set()):
            source = source_keys[row].tolist()
            if key != source:
                sources.append(source)
                descendants.append(key)
                kinds.append(int(EntityLineageKind.REFINED_FROM))
    if collapsed_vertices:
        for source in source_keys.tolist():
            key = tuple(
                sorted(collapsed_vertices.get(vertex, vertex) for vertex in source)
            )
            if len(set(key)) == len(key) and key != tuple(source) and key in target_keys:
                sources.append(source)
                descendants.append(list(key))
                kinds.append(int(EntityLineageKind.COLLAPSED_INTO))
    width = dimension + 1
    return EntityRelations(
        dimension,
        np.asarray(sources, dtype=np.int64).reshape((-1, width)),
        np.asarray(descendants, dtype=np.int64).reshape((-1, width)),
        np.asarray(kinds, dtype=np.int8),
    )


def _bucket(rows: np.ndarray, /) -> np.ndarray:
    """Pad leading rows to a power-of-two count by repeating the first row.

    Rowwise compiled kernels then compile once per bucket instead of once per
    actual cell, edge, cavity or query count; callers keep only actual rows.
    """
    count = rows.shape[0]
    size = 8 if count <= 8 else 1 << (count - 1).bit_length()
    if count == 0 or size == count:
        return rows
    return np.concatenate((rows, np.repeat(rows[:1], size - count, axis=0)))


def _quality(points: np.ndarray, metric: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        metric_simplex_quality(_bucket(metric), _bucket(points), _bucket(cells)),
        dtype=np.float64,
    )[: cells.shape[0]]


def _lengths(points: np.ndarray, metric: np.ndarray, edges: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        metric_edge_lengths(_bucket(metric), _bucket(points), _bucket(edges)),
        dtype=np.float64,
    )[: edges.shape[0]]


def _interpolate(values: np.ndarray, weights: np.ndarray, /) -> np.ndarray:
    return np.asarray(
        interpolate_mesh_metric(_bucket(values), _bucket(weights)), dtype=np.float64
    )[: values.shape[0]]


def _positive(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    corners = _bucket(points[cells])
    signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    positive = signs == PredicateSign.POSITIVE
    return positive[: cells.shape[0]]


def _orient(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    result = cells.copy()
    positive = _positive(points, result)
    result[~positive, 0], result[~positive, 1] = (
        result[~positive, 1].copy(),
        result[~positive, 0].copy(),
    )
    return result


def _labels(values: ArrayLike | None, count: int, name: str, /) -> np.ndarray:
    if values is None:
        return np.zeros((count,), dtype=np.int32)
    array = np.asarray(values)
    if (
        array.shape != (count,)
        or not np.issubdtype(array.dtype, np.integer)
        or np.any(array < 0)
        or np.any(array > np.iinfo(np.int32).max)
    ):
        raise ValueError(
            f"{name} must be nonnegative int32-range labels aligned with entities."
        )
    return array.astype(np.int32)


def _constraints(
    mesh: CellMesh,
    cells: np.ndarray,
    regions: np.ndarray,
    facet_classes: ArrayLike | None,
    edge_classes: ArrayLike | None,
    protected_edges: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    faces = cells[:, _FACES].reshape((-1, 3))
    keys, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    labels = _labels(facet_classes, mesh.entity_set(2).count, "facet_classes")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = key_rows(entity_keys(mesh, 2), np.sort(vertex_ids[keys], axis=1))
    low = np.full((keys.shape[0],), np.iinfo(np.int32).max, dtype=np.int32)
    high = np.full((keys.shape[0],), -1, dtype=np.int32)
    np.minimum.at(low, inverse, np.repeat(regions, 4))
    np.maximum.at(high, inverse, np.repeat(regions, 4))
    constrained = (counts == 1) | (low != high) | (labels[rows] != 0)
    first = np.full((keys.shape[0],), faces.shape[0], dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(faces.shape[0], dtype=np.int64))
    selected = faces[first[constrained]]
    sources = np.arange(selected.shape[0], dtype=np.int32)
    boundary_edges = _edges(selected)
    edges = _edges(cells)
    edge_rows = key_rows(entity_keys(mesh, 1), np.sort(vertex_ids[edges], axis=1))
    feature = (
        _labels(edge_classes, mesh.entity_set(1).count, "edge_classes")[edge_rows] != 0
    )
    segments = np.unique(
        np.concatenate((boundary_edges, edges[feature | protected_edges[edge_rows]])),
        axis=0,
    )
    return selected, sources, segments, np.arange(segments.shape[0], dtype=np.int32)


def _plc_native_constraints(
    mesh: CellMesh,
    source: ExactPlcCellGeometrySource,
    faces: np.ndarray,
    segments: np.ndarray,
    protected: np.ndarray,
    edge_classes: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TetMeshSourceComplex]:
    """Rebuild source IDs from exact membership, not newly assigned face slots."""
    prepared = source.prepare(mesh.coordinates)
    points = np.asarray(source.source_points, dtype=np.float64)
    budget = _ExactPlcBudget(source.maximum_work, source.maximum_bits)
    execution = current_native_execution_budget()
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

    ledger = _COORDINATE_BUDGET.get()
    if ledger is None:
        raise RuntimeError(
            "An exact PLC constraint proof requires its original coordinate ledger."
        )
    bank_upper, _ = _plc_storage_bounds(points.shape[0], 1075)
    _, scratch_upper = _plc_storage_bounds(0, source.maximum_bits)
    row_upper = (
        512 + 88 * source.source_triangles.shape[0] + 80 * source.source_segments.shape[0]
    )
    with ledger.temporary_scope():
        ledger.reserve(0, bank_upper + scratch_upper + row_upper)
        source_points = tuple(_point(point) for point in points)
        triangles = tuple(
            tuple(source_points[int(index)] for index in row)
            for row in np.asarray(source.source_triangles, dtype=np.int32)
        )
        lines = tuple(
            tuple(source_points[int(index)] for index in row)
            for row in np.asarray(source.source_segments, dtype=np.int32)
        )
        triangle_ids = np.asarray(source.source_triangle_ids, dtype=np.int64)
        original_segment_ids = np.asarray(source.source_segment_ids, dtype=np.int64)
        face_ids, face_groups = np.unique(triangle_ids, return_inverse=True)
        segment_ids, segment_groups = np.unique(original_segment_ids, return_inverse=True)
        selected_faces = np.empty((faces.shape[0],), dtype=np.int64)
        for slot, face in enumerate(faces):
            for row, corners in enumerate(triangles):
                if execution is not None:
                    execution.charge(geometry_queries=1)
                budget.charge(192, tuple(value for point in corners for value in point))
                if all(_in_entity(prepared.vertices[index], corners) for index in face):
                    selected_faces[slot] = triangle_ids[row]
                    break
            else:
                raise ValueError(
                    "A constrained target face has no exact declared PLC source row."
                )
        selected_segments: list[np.ndarray] = []
        selected_ids: list[int] = []
        keys = entity_keys(mesh, 1)
        global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        classes = _labels(edge_classes, keys.shape[0], "edge_classes")
        for segment in segments:
            for row, corners in enumerate(lines):
                if execution is not None:
                    execution.charge(geometry_queries=1)
                budget.charge(128, tuple(value for point in corners for value in point))
                if all(
                    _in_entity(prepared.vertices[index], corners) for index in segment
                ):
                    selected_segments.append(segment)
                    selected_ids.append(int(original_segment_ids[row]))
                    break
            else:
                slot = key_rows(keys, np.sort(global_ids[segment])[None])[0]
                if protected[slot] or classes[slot] != 0:
                    raise ValueError(
                        "A protected edge requires an exact declared PLC source segment."
                    )
                # A triangulation diagonal is part of its facet, not a new feature.
        declared = TetMeshSourceComplex(
            points,
            np.asarray(source.source_triangles, dtype=np.int32),
            face_groups.astype(np.int32),
            np.asarray(source.source_triangle_bounds, dtype=np.float64),
            np.asarray(source.source_segments, dtype=np.int32),
            segment_groups.astype(np.int32),
            np.asarray(source.source_segment_bounds, dtype=np.float64),
            np.asarray(source.vertex_strata, dtype=np.int8),
            np.asarray(source.vertex_rows, dtype=np.int32),
            np.asarray(source.vertex_parameters, dtype=np.float64),
            face_ids,
            segment_ids,
        )
        return (
            selected_faces,
            np.asarray(selected_segments, dtype=np.int32).reshape((-1, 2)),
            np.asarray(selected_ids, dtype=np.int64),
            declared,
        )


def _record(
    ancestry: _Ancestry, before: np.ndarray, after: np.ndarray, kind: EntityLineageKind, /
) -> None:
    old = {
        tuple(sorted(row)): ancestry.cells[tuple(sorted(row))] for row in before.tolist()
    }
    keys = {tuple(sorted(row)) for row in after.tolist()}
    removed = [value for key, value in old.items() if key not in keys]
    parents = set().union(*(value[1] for value in removed)) if removed else set()
    if ancestry.cell_kinds is None:
        ancestry.cell_kinds = {}
    kind = _lineage_operation_kind(
        kind,
        tuple(
            ancestry.cell_kinds.get(identifier, int(EntityLineageKind.PRESERVED))
            for identifier, _ in removed
        ),
    )
    for key in old.keys() - keys:
        ancestry.cell_kinds.pop(old[key][0], None)
        del ancestry.cells[key]
    new_keys = sorted(keys - old.keys())
    identifiers = _fresh_identifiers(ancestry.next_cell_id, len(new_keys))
    ancestry.next_cell_id += len(new_keys)
    for key, value in zip(new_keys, identifiers, strict=True):
        identifier = int(value)
        ancestry.cells[key] = identifier, parents.copy()
        ancestry.cell_kinds[identifier] = int(kind)


def _collapse_cells(cells: np.ndarray, removed: int, kept: int, /) -> np.ndarray:
    result = cells.copy()
    result[result == removed] = kept
    unique = np.sort(result, axis=1)
    return result[np.all(np.diff(unique, axis=1) != 0, axis=1)]


@dataclass(frozen=True, slots=True)
class TetraMetricSource:
    """Immutable authoritative affine source tables for tensor sampling.

    Tables retain supplied identifiers and vertex-row order. An exact PLC
    geometry binds the original rational source S; without it the ordinary
    binary64 affine path and its operation order remain unchanged.
    """

    points: np.ndarray
    cells: np.ndarray
    vertex_global_ids: np.ndarray
    cell_global_ids: np.ndarray
    source_geometry: CellGeometrySpec | None = None
    corners: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Metric source points must have three coordinates.")
        if self.cells.ndim != 2 or self.cells.shape[1] != 4 or not self.cells.shape[0]:
            raise ValueError("Metric source cells must be nonempty tetrahedral rows.")
        if (
            not np.issubdtype(self.cells.dtype, np.integer)
            or np.any(self.cells < 0)
            or np.any(self.cells >= self.points.shape[0])
        ):
            raise ValueError("Metric source cells must index their actual source points.")
        if self.vertex_global_ids.shape != (
            self.points.shape[0],
        ) or self.cell_global_ids.shape != (self.cells.shape[0],):
            raise ValueError(
                "Metric source identifiers must align with their original rows."
            )
        if self.source_geometry is not None:
            if not isinstance(self.source_geometry, CellGeometrySpec):
                raise TypeError("source_geometry must be CellGeometrySpec or None.")
            if not isinstance(
                self.source_geometry.exact_source, ExactPlcCellGeometrySource
            ):
                raise ValueError(
                    "Exact tetrahedral metric sampling requires the original PLC source."
                )
            bank = self.source_geometry.source_coordinates()
            if len(bank) != self.points.shape[0] or any(
                len(point) != 3 for point in bank
            ):
                raise ValueError(
                    "Exact metric coordinates must bind their original source rows."
                )
            from .._geometry_predicates import exact_charge
            from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET

            # The temporary source object bank and gathered corner slots
            # coexist. The immutable Fraction values and corner table remain
            # owned by the ledger for the entire metric-source lifetime.
            with exact_charge(0, 8 * (3 * len(bank) + 3 * self.cells.size)):
                corners = np.asarray(bank, dtype=object)[self.cells]
                ledger = _COORDINATE_BUDGET.get()
                if ledger is not None:
                    ledger.retain_basis((bank, corners))
                object.__setattr__(self, "corners", corners)
            return
        object.__setattr__(self, "corners", self.points[self.cells])

    @classmethod
    def from_mesh(cls, mesh: CellMesh, /) -> TetraMetricSource:
        return cls(
            np.asarray(mesh.coordinates, dtype=np.float64),
            np.concatenate(
                [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
            ),
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            ),
        )


@contextmanager
def _exact_table_boxes(values: np.ndarray, /) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Outward boxes of exact rows using the canonical rational rounding owner."""
    from .._geometry_predicates import bigint_bytes, exact_bits, exact_charge
    from ..discretization._coordinate_enclosure import outward

    bits = exact_bits(values)
    # Each min/max comparison cross-multiplies two b-bit rational inputs;
    # two products and seven division/comparison temporaries fit 2b+1 bits.
    # Both float buffers persist only until the owning BVH has consumed them.
    storage = 48 * values.shape[0] + 7 * bigint_bytes(2 * bits + 1)
    with exact_charge(4 * values.size + 6 * values.shape[0], storage):
        lower = np.empty((values.shape[0], 3), dtype=np.float64)
        upper = np.empty_like(lower)
        for row, corners in enumerate(values):
            for axis in range(3):
                lower[row, axis] = outward(
                    min(corner[axis] for corner in corners), -np.inf
                )
                upper[row, axis] = outward(
                    max(corner[axis] for corner in corners), np.inf
                )
        yield lower, upper


def _metric_source_bvh(tables: TetraMetricSource, /) -> PackedBVH:
    if tables.source_geometry is None:
        return prepare_bvh(
            np.min(tables.corners, axis=1),
            np.max(tables.corners, axis=1),
            dtype=jnp.float64,
        )
    with _exact_table_boxes(tables.corners) as (lower, upper):
        return prepare_bvh(lower, upper, dtype=jnp.float64)


def _exact_source_weights(
    corners: np.ndarray,
    point: ExactPoint,
    /,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    from .._geometry_predicates import bigint_bytes, exact_bits, exact_charge
    from ..discretization._coordinate_enclosure import _COORDINATE_BUDGET
    from ..linalg._small_batched import prepare_exact_small_linear_actions

    with exact_charge(0, 8 * 15):
        bits = exact_bits(np.asarray((*corners, point), dtype=object))
    # Nine source-edge and three query differences each have numerator and
    # denominator bounded by 2b+1 bits. Elimination, minor bounds and actual
    # operation counts belong exclusively to the shared exact linear owner.
    with exact_charge(12, 12 * (2 * bigint_bytes(2 * bits + 1) + 64)):
        matrix = tuple(
            tuple(corners[column + 1, axis] - corners[0, axis] for column in range(3))
            for axis in range(3)
        )
        right = tuple((point[axis] - corners[0, axis],) for axis in range(3))
        solved = prepare_exact_small_linear_actions(
            matrix, right, coordinate_budget=_COORDINATE_BUDGET.get()
        )
        if solved.actions is None or solved.determinant <= 0:
            raise ValueError(
                "An exact tetrahedral metric source requires positive authoritative cells."
            )
        reference = tuple(row[0] for row in solved.actions)
        return (
            Fraction(1) - sum(reference, Fraction(0)),
            reference[0],
            reference[1],
            reference[2],
        )


def _locate_exact_source_stencil(
    tables: TetraMetricSource,
    rounded: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    require_coverage: bool,
    prepared_bvh: PackedBVH | None,
    work_counter: list[int] | None,
    exact_points: tuple[ExactPoint, ...] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select immutable original S cells and sample the tensor at authoritative S."""
    from .._geometry_predicates import exact_charge

    bank = tuple(_point(row) for row in rounded) if exact_points is None else exact_points
    if len(bank) != rounded.shape[0]:
        raise ValueError(
            "Exact metric query witnesses must align with their carrier rows."
        )
    tree = _metric_source_bvh(tables) if prepared_bvh is None else prepared_bvh
    with exact_charge(0, 8 * 3 * len(bank)):
        with _exact_table_boxes(np.asarray(bank, dtype=object)[:, None, :]) as (
            lower,
            upper,
        ):
            queries = prepare_bvh(lower, upper, dtype=jnp.float64)
    selected = np.full((len(bank),), -1, dtype=np.int64)
    selected_ids = np.full((len(bank),), np.iinfo(np.int64).max, dtype=np.int64)
    weights = np.zeros((len(bank), 4), dtype=np.float64)
    work = 0
    for cells, points in bvh_overlap_pair_blocks(tree, queries, include_touching=True):
        for cell, query in zip(cells, points, strict=True):
            work += 1
            if work_counter is not None:
                work_counter[0] += 1
            if work > maximum_pairs:
                raise _source_location_budget(maximum_pairs, work)
            coefficients = _exact_source_weights(tables.corners[cell], bank[query])
            if (
                min(coefficients) < 0
                or tables.cell_global_ids[cell] >= selected_ids[query]
            ):
                continue
            selected[query], selected_ids[query] = cell, tables.cell_global_ids[cell]
            weights[query] = tuple(float(value) for value in coefficients)
    covered = selected >= 0
    if require_coverage and np.any(~covered):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "A metric target source witness lacks an exactly certified original source host.",
            stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    weights[~covered, 0] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    return (
        tables.vertex_global_ids[tables.cells[np.maximum(selected, 0)]],
        weights,
        np.broadcast_to(covered[:, None], weights.shape),
    )


def _locate_stencil(
    mesh: CellMesh,
    points: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    require_coverage: bool = True,
    prepared_bvh: PackedBVH | None = None,
    work_counter: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate in affine source tets; a supplied BVH must bind this exact source/order."""
    return _locate_source_stencil(
        TetraMetricSource.from_mesh(mesh),
        points,
        maximum_pairs,
        require_coverage=require_coverage,
        prepared_bvh=prepared_bvh,
        work_counter=work_counter,
    )


def _locate_source_stencil(
    tables: TetraMetricSource,
    points: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    require_coverage: bool = True,
    prepared_bvh: PackedBVH | None = None,
    work_counter: list[int] | None = None,
    exact_points: tuple[ExactPoint, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shared exact affine kernel, bound to immutable original source rows."""
    if prepared_bvh is not None and not isinstance(prepared_bvh, PackedBVH):
        raise TypeError("prepared_bvh must be PackedBVH.")
    if work_counter is not None and (
        len(work_counter) != 1
        or isinstance(work_counter[0], bool)
        or not isinstance(work_counter[0], int)
        or work_counter[0] < 0
    ):
        raise ValueError("work_counter must contain one nonnegative integer.")
    if tables.source_geometry is not None:
        return _locate_exact_source_stencil(
            tables,
            points,
            maximum_pairs,
            require_coverage=require_coverage,
            prepared_bvh=prepared_bvh,
            work_counter=work_counter,
            exact_points=exact_points,
        )
    cells = tables.cells
    cell_ids = tables.cell_global_ids
    source = tables.corners
    ids = tables.vertex_global_ids
    tree = (
        prepare_bvh(np.min(source, axis=1), np.max(source, axis=1), dtype=jnp.float64)
        if prepared_bvh is None
        else prepared_bvh
    )
    padded = _bucket(points)
    queries = prepare_bvh(padded, padded, dtype=jnp.float64)
    selected = np.full((points.shape[0],), -1, dtype=np.int64)
    selected_ids = np.full((points.shape[0],), np.iinfo(np.int64).max, dtype=np.int64)
    weights = np.zeros((points.shape[0], 4), dtype=np.float64)
    work = 0
    for block_cell, block_query in bvh_overlap_pair_blocks(
        tree, queries, include_touching=True
    ):
        # Bucket padding repeats query zero; only actual query pairs are charged.
        actual = block_query < points.shape[0]
        cell, query = block_cell[actual], block_query[actual]
        if not cell.size:
            continue
        work += cell.size
        if work_counter is not None:
            work_counter[0] += cell.size
        if work > maximum_pairs:
            raise _source_location_budget(maximum_pairs, work)
        corners = _bucket(source[cell])
        target = _bucket(points[query])
        matrix = np.swapaxes(corners[:, 1:] - corners[:, :1], 1, 2)
        solve = solve_small_linear(
            SmallLinearSolvePlan(3), matrix, target - corners[:, 0]
        )
        reference = np.asarray(solve.value, dtype=np.float64)[: cell.size]
        valid = np.array(solve.successful, dtype=np.bool_, copy=True)[: cell.size]
        coefficients = np.concatenate(
            (1.0 - np.sum(reference, axis=1, keepdims=True), reference), axis=1
        )
        for slot in range(4):
            vertices = [corners[:, index] for index in range(4)]
            vertices[slot] = target
            signs = exact_orient3d(*vertices)[: cell.size]
            valid &= signs >= PredicateSign.ZERO
            coefficients[signs == PredicateSign.ZERO, slot] = 0.0
        accepted = np.flatnonzero(valid & (cell_ids[cell] < selected_ids[query]))
        order = accepted[np.lexsort((cell_ids[cell[accepted]], query[accepted]))]
        first = (
            np.concatenate(
                (np.asarray((True,), dtype=np.bool_), np.diff(query[order]) != 0)
            )
            if order.size
            else np.empty((0,), dtype=np.bool_)
        )
        chosen = order[first]
        selected[query[chosen]] = cell[chosen]
        selected_ids[query[chosen]] = cell_ids[cell[chosen]]
        weights[query[chosen]] = coefficients[chosen]
    covered = selected >= 0
    if require_coverage and np.any(~covered):
        raise MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED,
            "A metric target vertex lacks an exactly certified source host.",
            stage=MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    weights = np.maximum(weights, 0.0)
    weights[~covered, 0] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    return (
        ids[cells[np.maximum(selected, 0)]],
        weights,
        np.broadcast_to(covered[:, None], weights.shape),
    )


def _assemble(
    mesh: CellMesh,
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    ancestry: _Ancestry,
    maximum_pairs: int,
    /,
    *,
    source_tables: TetraMetricSource | None = None,
    exact_points: tuple[ExactPoint, ...] | None = None,
) -> tuple[CellTopologyEdit, np.ndarray]:
    live = np.flatnonzero(snapshot.vertex_dimension >= 0)
    row_of = np.full((snapshot.points.shape[0],), -1, dtype=np.int32)
    row_of[live] = np.arange(live.size, dtype=np.int32)
    source_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    ids = np.concatenate(
        (
            source_ids,
            _fresh_identifiers(
                int(np.max(source_ids)) + 1, snapshot.points.shape[0] - source_ids.size
            ),
        )
    )
    cells = row_of[snapshot.tetrahedra]
    cell_ids = np.asarray(
        [ancestry.cells[tuple(sorted(row))][0] for row in snapshot.tetrahedra.tolist()],
        dtype=np.int64,
    )
    source_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    block_owner = {
        int(identifier): block
        for block, values in enumerate(mesh.blocks)
        for identifier in np.asarray(values.global_ids).tolist()
    }
    owners = np.asarray(
        [
            block_owner[min(ancestry.cells[tuple(sorted(row))][1])]
            for row in snapshot.tetrahedra.tolist()
        ],
        dtype=np.int32,
    )
    blocks = source_family_blocks(
        tuple((block.name, block.cell_kind) for block in mesh.blocks),
        tuple(cells[owners == index] for index in range(len(mesh.blocks))),
        tuple(cell_ids[owners == index] for index in range(len(mesh.blocks))),
    )
    stencil = (
        _locate_stencil(mesh, snapshot.points[live], maximum_pairs)
        if source_tables is None or source_tables.source_geometry is None
        else _locate_source_stencil(
            source_tables, snapshot.points[live], maximum_pairs, exact_points=exact_points
        )
    )
    relations = []
    live_ids = ids[live]
    collapsed = _resolved_collapses(ids, source_ids.size, ancestry.vertex_successor)
    relocated = (
        set() if ancestry.relocated_vertices is None else ancestry.relocated_vertices
    )
    vertex_pairs = [
        (
            source,
            int(ids[vertex]),
            int(
                EntityLineageKind.RELOCATED
                if vertex in relocated
                else EntityLineageKind.REFINED_FROM
            ),
        )
        for vertex in live.tolist()
        if vertex >= source_ids.size
        for source in sorted(ancestry.vertex_sources[vertex])
    ]
    vertex_pairs.extend(
        (int(ids[vertex]), int(ids[vertex]), int(EntityLineageKind.RELOCATED))
        for vertex in live.tolist()
        if vertex < source_ids.size and vertex in relocated
    )
    vertex_pairs.extend(
        (source, target, int(EntityLineageKind.COLLAPSED_INTO))
        for source, target in sorted(collapsed.items())
    )
    cell_kinds = {} if ancestry.cell_kinds is None else ancestry.cell_kinds
    cell_pairs = [
        (parent, identifier, cell_kinds.get(identifier, int(EntityLineageKind.PRESERVED)))
        for identifier, parents in ancestry.cells.values()
        for parent in sorted(parents)
        if parent != identifier or identifier in cell_kinds
    ]
    for dimension in range(4):
        if dimension in (1, 2):
            relations.append(
                _subentity_relations(
                    mesh,
                    dimension,
                    snapshot.tetrahedra,
                    ids,
                    ancestry.vertex_sources,
                    collapsed_vertices=collapsed,
                )
            )
            continue
        width = 1 if dimension in (0, 3) else dimension + 1
        pairs = vertex_pairs if dimension == 0 else cell_pairs
        if pairs:
            values = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
            # Transient targets are not part of this accepted edit.
            valid = np.isin(values[:, 1], live_ids if dimension == 0 else cell_ids)
            values = values[valid]
            relations.append(
                EntityRelations(
                    dimension, values[:, :1], values[:, 1:2], values[:, 2].astype(np.int8)
                )
            )
        else:
            relations.append(
                EntityRelations(
                    dimension,
                    np.empty((0, width), dtype=np.int64),
                    np.empty((0, width), dtype=np.int64),
                    np.empty((0,), dtype=np.int8),
                )
            )
    operation = (
        "relocation"
        if np.array_equal(cell_ids, source_cell_ids) and live.size == source_ids.size
        else "local_reconnection"
    )
    return CellTopologyEdit(
        operation, snapshot.points[live], live_ids, blocks, *stencil, tuple(relations)
    ), metric[live]


_OperationResult = TypeVar("_OperationResult")


@dataclass(slots=True)
class _NativeMetricBudget:
    """Cumulative native work, source-pair and per-cavity epoch allowances."""

    native: TetMesh3D
    initial_work: int
    maximum_work_units: int
    maximum_cavity_work: int
    source_tables: TetraMetricSource
    maximum_source_pairs: int
    source_candidates: list[int] = field(default_factory=lambda: [0])
    source_bvh: PackedBVH | None = None
    exhausted: bool = False
    resource_message: str | None = None
    resource_requested: tuple[tuple[str, float], ...] = ()
    resource_achieved: tuple[tuple[str, float], ...] = ()
    optimizer_host_work: int = 0
    optimizer_evaluations: int = 0
    optimizer_iterations: int = 0
    optimizer_objective_evaluations: int = 0
    optimizer_constraint_evaluations: int = 0
    optimizer_refused_proposals: int = 0
    optimizer_status_counts: list[int] = field(
        default_factory=lambda: [0] * len(OptimizationStatus)
    )
    optimizer_terminations: list[tuple[int, int]] = field(default_factory=list)
    optimizer_scratch_required_bytes: int = 0
    maximum_optimizer_scratch_bytes: int | None = None
    maximum_cavity_cells: int = 0
    expanded_insertions: int = 0
    optimizer_refusals: dict[str, int] = field(default_factory=dict)
    optimizer_method_id: str | None = None
    optimizer_tolerances: tuple[float, ...] = ()
    record_phase: NativeMeshingPhaseRecorder | None = None
    execution_budget: NativeExecutionBudget | None = None
    operation_started: float | None = None
    maximum_wall_seconds: float | None = None

    def check_deadline(self) -> None:
        """Check host preparation/dispatch against the original operation clock."""
        if self.operation_started is None or self.maximum_wall_seconds is None:
            if self.execution_budget is not None:
                # A raw ambient scope owns its steady-clock deadline, but no
                # Python start timestamp. Observe that same native deadline;
                # zero is a dry bound, not invented charged work.
                self.execution_budget.admit_work_bound(0)
            return
        elapsed = monotonic() - self.operation_started
        if elapsed > self.maximum_wall_seconds:
            raise MeshingFailure(
                MeshingFailureCategory.TIMED_OUT,
                "Native meshing exceeded its enforced wall-time limit.",
                stage=MeshingStageKind.VOLUME_FILL.value,
                requested=(("maximum_wall_seconds", self.maximum_wall_seconds),),
                achieved=(("elapsed_seconds", elapsed),),
            )

    def refuse_optimizer(self, *reasons: str) -> None:
        self.optimizer_refused_proposals += 1
        for reason in reasons:
            self.optimizer_refusals[reason] = self.optimizer_refusals.get(reason, 0) + 1

    def refuse(
        self,
        message: str,
        requested: tuple[tuple[str, float], ...],
        achieved: tuple[tuple[str, float], ...],
        /,
    ) -> None:
        self.exhausted = True
        self.resource_message = message
        self.resource_requested = requested
        self.resource_achieved = achieved

    @property
    def work_units(self) -> int:
        return self.native.work_units() - self.initial_work + self.optimizer_host_work

    def optimizer_scratch(self, scratch_bytes: int, /) -> bool:
        """Admit a bounded host-table reservation before its actual allocation."""
        self.check_deadline()
        if self.exhausted:
            return False
        self.optimizer_scratch_required_bytes = max(
            self.optimizer_scratch_required_bytes, scratch_bytes
        )
        if (
            self.maximum_optimizer_scratch_bytes is not None
            and scratch_bytes > self.maximum_optimizer_scratch_bytes
        ):
            self.refuse(
                "The statistical optimizer's declared scratch plan exceeds its allowance.",
                (
                    (
                        "maximum_optimizer_scratch_bytes",
                        float(self.maximum_optimizer_scratch_bytes),
                    ),
                ),
                (("optimizer_scratch_required_bytes", float(scratch_bytes)),),
            )
            return False
        return True

    def optimizer_work(self, units: int, scratch_bytes: int, /) -> bool:
        """Reserve one finite host proposal, sharing the original total-work cap."""
        self.check_deadline()
        if self.exhausted:
            return False
        if units > self.maximum_work_units - self.work_units:
            self.refuse(
                "The cumulative statistical optimizer work allowance is exhausted.",
                (("maximum_work_units", float(self.maximum_work_units)),),
                (
                    ("work_units", float(self.work_units)),
                    ("optimizer_host_work_units", float(self.optimizer_host_work)),
                ),
            )
            return False
        if not self.optimizer_scratch(scratch_bytes):
            return False
        if self.execution_budget is not None:
            self.execution_budget.charge(work=units)
        self.optimizer_host_work += units
        self.optimizer_evaluations += 1
        return True

    def call(
        self, operation: Callable[[int], _OperationResult], /
    ) -> _OperationResult | None:
        self.check_deadline()
        allowance = min(
            self.maximum_cavity_work, self.maximum_work_units - self.work_units
        )
        if self.exhausted:
            return None
        if allowance <= 0:
            self.refuse(
                "The cumulative native/host metric work allowance is exhausted.",
                (("maximum_work_units", float(self.maximum_work_units)),),
                (
                    ("work_units", float(self.work_units)),
                    (
                        "native_work_units",
                        float(self.native.work_units() - self.initial_work),
                    ),
                    ("optimizer_host_work_units", float(self.optimizer_host_work)),
                ),
            )
            return None
        try:
            self.native.set_work_limit(allowance)
            result = operation(allowance)
            self.check_deadline()
            return result
        except MeshcoreError as error:
            if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
                raise
            self.refuse(
                str(error),
                (
                    ("native_operation_work_allowance", float(allowance)),
                    ("maximum_work_units", float(self.maximum_work_units)),
                ),
                (
                    ("work_units", float(self.work_units)),
                    (
                        "native_work_units",
                        float(self.native.work_units() - self.initial_work),
                    ),
                    ("optimizer_host_work_units", float(self.optimizer_host_work)),
                ),
            )
            return None

    def stencil(
        self,
        points: np.ndarray,
        /,
        *,
        exact_points: tuple[ExactPoint, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        self.check_deadline()
        remaining = self.maximum_source_pairs - self.source_candidates[0]
        if self.exhausted:
            return None
        if remaining <= 0:
            self.refuse(
                "The cumulative metric source-pair allowance is exhausted.",
                (("maximum_source_pairs", float(self.maximum_source_pairs)),),
                (("source_candidate_pairs", float(self.source_candidates[0])),),
            )
            return None
        if self.execution_budget is not None:
            # One actual affine source/field evaluation per submitted point.
            # Internal exact predicates are independent native telemetry.
            self.execution_budget.charge(geometry_queries=points.shape[0])
        if self.source_bvh is None:
            self.source_bvh = _metric_source_bvh(self.source_tables)
        try:
            return _locate_source_stencil(
                self.source_tables,
                points,
                remaining,
                require_coverage=False,
                prepared_bvh=self.source_bvh,
                work_counter=self.source_candidates,
                exact_points=exact_points,
            )
        except MeshingFailure as error:
            if error.evidence.category is not MeshingFailureCategory.RESOURCE_EXHAUSTED:
                raise
            self.refuse(
                error.evidence.message, error.evidence.requested, error.evidence.achieved
            )
            return None


@dataclass(frozen=True, slots=True)
class UniformSizeRemeshingGoal:
    """Original physical size obligations of one admitted hard background.

    ``minimum_relative_determinant`` is the publishing validity policy's
    relative determinant floor: no accepted proposal creates a cell whose
    exact determinant does not exceed it times its vertex-edge product.
    """

    control: UniformSizeControl
    policy: SizeCompliancePolicy
    minimum_relative_determinant: float
    maximum_radius_edge: float | None = None
    minimum_dihedral_degrees: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.control, UniformSizeControl):
            raise TypeError("A uniform size goal requires UniformSizeControl.")
        if self.control.strength is not SizeControlStrength.HARD:
            raise ValueError("A physical size-remeshing goal must be hard.")
        if not isinstance(self.policy, SizeCompliancePolicy):
            raise TypeError("A uniform size goal requires SizeCompliancePolicy.")
        if not np.isfinite(self.minimum_relative_determinant) or not (
            0.0 <= self.minimum_relative_determinant < 1.0
        ):
            raise ValueError("A relative determinant floor must lie in [0, 1).")
        if self.maximum_radius_edge is not None and (
            not np.isfinite(self.maximum_radius_edge) or self.maximum_radius_edge <= 0.0
        ):
            raise ValueError("A proposal radius-edge bound must be finite and positive.")
        if self.minimum_dihedral_degrees is not None and (
            not np.isfinite(self.minimum_dihedral_degrees)
            or not 0.0 <= self.minimum_dihedral_degrees <= 180.0
        ):
            raise ValueError("A proposal dihedral bound must be finite and in degrees.")


def _physical_size_compliance(
    snapshot: TetMeshArrays, goal: UniformSizeRemeshingGoal, /
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:
    from .providers._native_publication import (
        edge_size_evidence,
        uniform_size_compliance,
        unique_edges,
    )

    lengths, growth = edge_size_evidence(
        snapshot.points, unique_edges(snapshot.tetrahedra, "tetrahedron")
    )
    return uniform_size_compliance(goal.control, goal.policy, lengths, growth)


def _physical_size_goal_met(
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    goal: UniformSizeRemeshingGoal | None,
    minimum_quality: float,
    /,
) -> bool:
    if goal is None or _physical_size_compliance(snapshot, goal)[2]:
        return False
    return bool(
        np.all(_quality(snapshot.points, metric, snapshot.tetrahedra) >= minimum_quality)
    )


def _preferred_size_fraction(target: float, delta: np.ndarray, /) -> float:
    """Condition the target fraction from squared physical distance.

    Exact power-of-two scaling avoids overflow/underflow of the intermediate
    squares and a rounded square-root followed by a second rounded division.
    Native construction still owns exact represented incidence and admission.
    """
    target_mantissa, target_exponent = np.frexp(target)
    _, distance_exponent = np.frexp(np.max(np.abs(delta)))
    scaled_delta = np.ldexp(delta, -int(distance_exponent))
    squared = float(np.dot(scaled_delta, scaled_delta))
    return float(
        np.ldexp(
            np.sqrt(target_mantissa * target_mantissa / squared),
            int(target_exponent) - int(distance_exponent),
        )
    )


def _edge_incidence(cells: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Canonical tetrahedral edge keys and finite-cell incidence multiplicities."""
    pairs = np.sort(cells[:, _TET_EDGES].reshape((-1, 2)), axis=1).astype(np.uint64)
    encoded = (pairs[:, 0] << np.uint64(32)) | pairs[:, 1]
    return np.unique(encoded, return_counts=True)


def _edge_rows(keys: np.ndarray, /) -> np.ndarray:
    return np.column_stack((keys >> np.uint64(32), keys & np.uint64(0xFFFFFFFF))).astype(
        np.int64
    )


def _proposal_rows(
    values: np.ndarray,
    rows: np.ndarray,
    changed_vertex: int | None,
    changed_value: np.ndarray | None,
    /,
) -> np.ndarray:
    """Gather real rows, including one explicit append or relocation witness."""
    if changed_vertex is None:
        return values[rows]
    if changed_value is None:
        raise ValueError("A changed vertex requires its actual proposed value.")
    result = np.empty((*rows.shape, *values.shape[1:]), dtype=values.dtype)
    changed = rows == changed_vertex
    if np.any((rows[~changed] < 0) | (rows[~changed] >= values.shape[0])):
        raise ValueError("A proposal references no actual or explicitly proposed vertex.")
    result[~changed] = values[rows[~changed]]
    result[changed] = changed_value
    return result


@dataclass(frozen=True, slots=True)
class _PhysicalSizeMerit:
    deficits: np.ndarray
    directional_edge_penalty: float
    issues: tuple[str, ...]
    achieved: tuple[tuple[str, float], ...]
    requested: tuple[tuple[str, float], ...]

    @property
    def normalized_deficit(self) -> float:
        return float(np.sum(self.deficits))


def _physical_size_merit(
    lengths: np.ndarray, growth: float, goal: UniformSizeRemeshingGoal, /
) -> _PhysicalSizeMerit:
    """Numerical deficits from owning evidence, never an alternate acceptance."""
    from .providers._native_publication import uniform_size_compliance

    requested, achieved, issues = uniform_size_compliance(
        goal.control, goal.policy, lengths, growth
    )
    prefix = f"size:{goal.control.control_id}"
    request = dict(requested)
    actual = dict(achieved)
    failed = {issue.split(":", 1)[0] for issue in issues}
    quantities = [
        f"target_size_{statistic}" for statistic in goal.policy.target_statistics
    ]
    quantities.extend(
        name
        for name in ("minimum_size", "maximum_size", "maximum_growth_rate")
        if f"{prefix}:{name}" in request
    )
    deficits: list[float] = []
    penalty = 0.0
    for quantity in quantities:
        if quantity not in failed:
            deficits.append(0.0)
            continue
        if quantity.startswith("target_size_"):
            statistic = quantity.removeprefix("target_size_")
            desired = request[f"{prefix}:target_size"]
            measured = actual[f"{prefix}:{statistic}_edge"]
            deviation = abs(measured - desired)
            direction = lengths - desired if measured > desired else desired - lengths
        elif quantity == "minimum_size":
            desired = request[f"{prefix}:minimum_size"]
            deviation = desired - actual[f"{prefix}:minimum_edge"]
            direction = desired - lengths
        elif quantity == "maximum_size":
            desired = request[f"{prefix}:maximum_size"]
            deviation = actual[f"{prefix}:maximum_edge"] - desired
            direction = lengths - desired
        else:
            desired = request[f"{prefix}:maximum_growth_rate"]
            deviation = actual[f"{prefix}:maximum_local_edge_ratio"] - desired
            direction = np.abs(lengths / goal.control.target_size - 1.0)
        # Only the owning failed-quantity list decides whether a deficit exists.
        # The requested value normalizes its actual deviation; no quantile or
        # tolerance comparison is reimplemented by the optimizer.
        deficits.append(float(deviation / desired))
        penalty += float(np.sum(np.maximum(direction, 0.0)) / desired)
    return _PhysicalSizeMerit(
        np.asarray(deficits, dtype=np.float64),
        penalty,
        tuple(issues),
        tuple(achieved),
        tuple(requested),
    )


@dataclass(frozen=True, slots=True)
class _PhysicalEdgeLedger:
    """Sorted edge incidence and actual lengths of one immutable native state.

    A bounded cavity delta projects global edge evidence in linear edge space,
    without copying/sorting the complete tetrahedron table for each candidate.
    """

    keys: np.ndarray
    counts: np.ndarray
    lengths: np.ndarray
    vertex_count: int
    merit: _PhysicalSizeMerit

    @classmethod
    def from_snapshot(
        cls, snapshot: TetMeshArrays, goal: UniformSizeRemeshingGoal, /
    ) -> _PhysicalEdgeLedger:
        from .providers._native_publication import edge_size_evidence

        keys, counts = _edge_incidence(snapshot.tetrahedra)
        lengths, growth = edge_size_evidence(snapshot.points, _edge_rows(keys))
        return cls(
            keys,
            counts,
            lengths,
            snapshot.points.shape[0],
            _physical_size_merit(lengths, growth, goal),
        )

    def project(
        self,
        points: np.ndarray,
        removed: np.ndarray,
        proposed: np.ndarray,
        goal: UniformSizeRemeshingGoal,
        changed_vertex: int | None,
        changed_position: np.ndarray | None,
        /,
    ) -> _PhysicalEdgeLedger:
        from .providers._native_publication import edge_growth_evidence

        old_keys, old_counts = _edge_incidence(removed)
        new_keys, new_counts = _edge_incidence(proposed)
        affected = np.union1d(old_keys, new_keys)
        positions = np.searchsorted(self.keys, affected)
        present = positions < self.keys.size
        present[present] &= self.keys[positions[present]] == affected[present]
        after_counts = np.zeros((affected.size,), dtype=np.int64)
        after_counts[present] = self.counts[positions[present]]
        after_counts[np.searchsorted(affected, old_keys)] -= old_counts
        after_counts[np.searchsorted(affected, new_keys)] += new_counts
        if np.any(after_counts < 0):
            raise RuntimeError("A metric cavity delta exceeds its actual edge incidence.")
        edges = _edge_rows(affected)
        endpoints = _proposal_rows(points, edges, changed_vertex, changed_position)
        affected_lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
        keep = np.ones((self.keys.size,), dtype=np.bool_)
        keep[positions[present & (after_counts == 0)]] = False
        existing_keys = self.keys[keep]
        existing_counts = self.counts[keep]
        existing_lengths = self.lengths[keep]
        retained = present & (after_counts > 0)
        destinations = np.searchsorted(existing_keys, affected[retained])
        existing_counts[destinations] = after_counts[retained]
        existing_lengths[destinations] = affected_lengths[retained]
        added = ~present & (after_counts > 0)
        added_keys = affected[added]
        count = existing_keys.size + added_keys.size
        keys = np.empty((count,), dtype=np.uint64)
        counts = np.empty((count,), dtype=np.int64)
        lengths = np.empty((count,), dtype=np.float64)
        existing_destinations = np.arange(existing_keys.size) + np.searchsorted(
            added_keys, existing_keys
        )
        added_destinations = np.arange(added_keys.size) + np.searchsorted(
            existing_keys, added_keys
        )
        keys[existing_destinations], keys[added_destinations] = existing_keys, added_keys
        counts[existing_destinations], counts[added_destinations] = (
            existing_counts,
            after_counts[added],
        )
        lengths[existing_destinations], lengths[added_destinations] = (
            existing_lengths,
            affected_lengths[added],
        )
        vertex_count = max(
            self.vertex_count, 0 if changed_vertex is None else changed_vertex + 1
        )
        growth = edge_growth_evidence(lengths, _edge_rows(keys), vertex_count)
        return _PhysicalEdgeLedger(
            keys,
            counts,
            lengths,
            vertex_count,
            _physical_size_merit(lengths, growth, goal),
        )


@dataclass(frozen=True, slots=True)
class _StatisticalProposal:
    edges: _PhysicalEdgeLedger
    quality_sum: float
    cell_count: int


@dataclass(slots=True)
class _StatisticalObjective:
    """Closed size-goal merit and original native shape guards of one epoch."""

    goal: UniformSizeRemeshingGoal
    edges: _PhysicalEdgeLedger
    budget: _NativeMetricBudget
    minimum_quality: float
    quality_sum: float
    cell_count: int
    _active_edge_keys: np.ndarray | None = None

    def offending_edge_keys(self) -> np.ndarray:
        """Cache the current owning failed-statistic suffix or explicit violations."""
        if self._active_edge_keys is not None:
            return self._active_edge_keys
        ledger = self.edges
        actual = dict(ledger.merit.achieved)
        requested = dict(ledger.merit.requested)
        prefix = f"size:{self.goal.control.control_id}"
        failed = {issue.split(":", 1)[0] for issue in ledger.merit.issues}
        selected = np.zeros((ledger.keys.size,), dtype=np.bool_)
        for quantity in sorted(failed):
            if quantity.startswith("target_size_"):
                statistic = quantity.removeprefix("target_size_")
                measured = actual[f"{prefix}:{statistic}_edge"]
                desired = requested[f"{prefix}:target_size"]
                selected |= (
                    ledger.lengths >= measured
                    if measured > desired
                    else ledger.lengths <= measured
                )
            elif quantity == "minimum_size":
                selected |= ledger.lengths < requested[f"{prefix}:minimum_size"]
            elif quantity == "maximum_size":
                selected |= ledger.lengths > requested[f"{prefix}:maximum_size"]
            else:
                selected[:] = True
                continue
        self._active_edge_keys = ledger.keys[selected]
        return self._active_edge_keys

    def propose(
        self,
        snapshot: TetMeshArrays,
        metric: np.ndarray,
        removed: np.ndarray,
        proposed: np.ndarray,
        /,
        *,
        changed_vertex: int | None = None,
        changed_position: np.ndarray | None = None,
        changed_metric: np.ndarray | None = None,
    ) -> _StatisticalProposal | None:
        if not proposed.size:
            return None
        edge_proposal = self.edges.project(
            snapshot.points,
            removed,
            proposed,
            self.goal,
            changed_vertex,
            changed_position,
        )
        old_merit, new_merit = self.edges.merit, edge_proposal.merit
        if new_merit.normalized_deficit > old_merit.normalized_deficit:
            return None
        vertices = np.unique(proposed)
        local_points = _proposal_rows(
            snapshot.points, vertices, changed_vertex, changed_position
        )
        local_metric = _proposal_rows(metric, vertices, changed_vertex, changed_metric)
        local_cells = np.searchsorted(vertices, proposed).astype(np.int32)
        proposed_quality = _quality(local_points, local_metric, local_cells)
        if np.any(proposed_quality < self.minimum_quality):
            return None
        old_quality = _quality(snapshot.points, metric, removed)
        quality_sum = (
            self.quality_sum
            - float(np.sum(old_quality))
            + float(np.sum(proposed_quality))
        )
        cell_count = self.cell_count - removed.shape[0] + proposed.shape[0]
        if new_merit.normalized_deficit == old_merit.normalized_deficit:
            if new_merit.directional_edge_penalty > old_merit.directional_edge_penalty:
                return None
            if (
                new_merit.directional_edge_penalty == old_merit.directional_edge_penalty
                and (
                    quality_sum / cell_count
                    <= self.quality_sum / self.cell_count + 1.0e-6
                )
            ):
                return None
        floor = self.goal.minimum_relative_determinant
        if (
            floor > 0.0
            or self.goal.maximum_radius_edge is not None
            or self.goal.minimum_dihedral_degrees is not None
        ):
            # The native owner's vertex indices are this epoch's identities; a
            # proposed vertex carries the index it will receive. PLC publication
            # keeps their order (sorted-used compaction, arange global IDs) and
            # the native rows already start at their smallest index, so this is
            # the chart the publishing validity policy evaluates.
            shape = self.budget.call(
                lambda allowance: self.budget.native.proposal_shape(
                    local_points,
                    local_cells,
                    vertex_ids=vertices,
                    minimum_relative_determinant=floor,
                    work_limit=allowance,
                )
            )
            if shape is None:
                return None
            radius_edge, dihedral, below_floor = shape
            if below_floor:
                self.budget.refuse_optimizer("relative_determinant_floor")
                return None
            if (
                self.goal.maximum_radius_edge is not None
                and radius_edge > self.goal.maximum_radius_edge
            ) or (
                self.goal.minimum_dihedral_degrees is not None
                and dihedral < self.goal.minimum_dihedral_degrees
            ):
                return None
        return _StatisticalProposal(edge_proposal, quality_sum, cell_count)

    def commit(self, proposal: _StatisticalProposal, /) -> None:
        self.edges = proposal.edges
        self.quality_sum = proposal.quality_sum
        self.cell_count = proposal.cell_count
        self._active_edge_keys = None

    def complete(self, snapshot: TetMeshArrays, metric: np.ndarray, /) -> bool:
        return not self.edges.merit.issues and _physical_size_goal_met(
            snapshot, metric, self.goal, self.minimum_quality
        )


class NativeTetraMetricOutcome(NamedTuple):
    """In-place epoch output; native constraint-source identities are unchanged."""

    snapshot: TetMeshArrays
    metric: np.ndarray
    evidence: MetricRemeshingEvidence
    ancestry: _Ancestry


def execute_native_tetra_metric(
    native: TetMesh3D,
    source_tables: TetraMetricSource,
    metric: np.ndarray,
    /,
    *,
    fixed_vertices: np.ndarray,
    protected_edges: set[tuple[int, int]],
    maximum_passes: int,
    topology_operations: bool,
    relocation: bool,
    minimum_metric_quality: float,
    lower_metric_length: float,
    upper_metric_length: float,
    maximum_vertices: int,
    maximum_cells: int,
    maximum_operations: int,
    maximum_work_units: int,
    maximum_location_pairs: int,
    maximum_cavity_cells: int,
    maximum_cavity_work: int,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    size_goal: UniformSizeRemeshingGoal | None = None,
    maximum_optimizer_scratch_bytes: int | None = None,
    execution_budget: NativeExecutionBudget | None = None,
    operation_started: float | None = None,
    maximum_wall_seconds: float | None = None,
) -> NativeTetraMetricOutcome:
    """Run the canonical metric epoch on an existing native constrained owner.

    The affine source carrier supports tensor interpolation and edit ancestry;
    it never reconstructs the native state or reassigns material/facet/curve IDs.
    Numerical edge bounds and actual cumulative native work are explicit.
    """
    if (
        not np.isfinite(lower_metric_length)
        or not np.isfinite(upper_metric_length)
        or not 0.0 <= lower_metric_length <= upper_metric_length
        or upper_metric_length <= 0.0
    ):
        raise ValueError("Metric edge bounds must form a finite nonnegative interval.")
    budgets = (
        maximum_passes,
        maximum_vertices,
        maximum_cells,
        maximum_operations,
        maximum_work_units,
        maximum_location_pairs,
        maximum_cavity_cells,
        maximum_cavity_work,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in budgets
    ):
        raise ValueError("Metric schedule budgets must be nonnegative integers.")
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("Operation controls must be bool.")
    if not 0.0 <= minimum_metric_quality <= 1.0:
        raise ValueError("minimum_metric_quality must lie in [0, 1].")
    if metric.shape != (source_tables.points.shape[0], 3, 3):
        raise ValueError("metric must have one 3D SPD tensor per source vertex.")
    if fixed_vertices.shape != (source_tables.points.shape[0],):
        raise ValueError("Fixed vertices must align with the source carrier.")
    if size_goal is not None and (
        not isinstance(size_goal, UniformSizeRemeshingGoal) or not topology_operations
    ):
        raise TypeError(
            "Physical size-statistic remeshing requires its closed topology goal."
        )
    if maximum_optimizer_scratch_bytes is not None and (
        isinstance(maximum_optimizer_scratch_bytes, bool)
        or not isinstance(maximum_optimizer_scratch_bytes, int)
        or maximum_optimizer_scratch_bytes < 0
    ):
        raise ValueError("The optimizer scratch allowance must be a nonnegative integer.")
    if (operation_started is None) != (maximum_wall_seconds is None):
        raise ValueError(
            "Metric host deadlines require the original clock and wall allowance together."
        )
    if operation_started is not None and (
        not np.isfinite(operation_started)
        or maximum_wall_seconds is None
        or not np.isfinite(maximum_wall_seconds)
        or maximum_wall_seconds < 0.0
    ):
        raise ValueError(
            "Metric host deadlines require a finite clock and nonnegative wall allowance."
        )
    active_budget = current_native_execution_budget()
    if execution_budget is None:
        execution_budget = active_budget
    elif execution_budget is not active_budget:
        raise ValueError(
            "A metric epoch must borrow its actual active native execution budget."
        )
    budget = _NativeMetricBudget(
        native,
        native.work_units(),
        maximum_work_units,
        maximum_cavity_work,
        source_tables,
        maximum_location_pairs,
        record_phase=record_phase,
        execution_budget=execution_budget,
        operation_started=operation_started,
        maximum_wall_seconds=maximum_wall_seconds,
    )
    with _measure_statistical_phase(budget, "metric_preparation"):
        metric = _interpolate(
            metric[:, None], np.ones((metric.shape[0], 1), dtype=np.float64)
        )
    source_metric = metric
    values = metric.copy()
    cells = source_tables.cells
    source_cell_ids = source_tables.cell_global_ids
    ancestry = _Ancestry(
        {
            tuple(sorted(row)): (int(identifier), {int(identifier)})
            for row, identifier in zip(cells.tolist(), source_cell_ids, strict=True)
        },
        int(np.max(source_cell_ids)) + 1,
        [{int(identifier)} for identifier in source_tables.vertex_global_ids.tolist()],
    )
    budget.maximum_optimizer_scratch_bytes = maximum_optimizer_scratch_bytes
    budget.maximum_cavity_cells = maximum_cavity_cells
    split_parameters: list[float] = []
    counts = [0, 0, 0, 0]
    compound_splits = 0
    rejected, operations, passes = 0, 0, 0
    status = MetricRemeshingStatus.PASS_LIMIT
    snapshot = native.arrays()
    objective = (
        _StatisticalObjective(
            size_goal,
            _PhysicalEdgeLedger.from_snapshot(snapshot, size_goal),
            budget,
            minimum_metric_quality,
            float(np.sum(_quality(snapshot.points, values, snapshot.tetrahedra))),
            snapshot.tetrahedra.shape[0],
        )
        if size_goal is not None
        else None
    )
    for passes in range(1, maximum_passes + 1):
        budget.check_deadline()
        edges = _edges(snapshot.tetrahedra)
        lengths = _lengths(snapshot.points, values, edges)
        quality = _quality(snapshot.points, values, snapshot.tetrahedra)
        satisfied = (
            _physical_size_goal_met(snapshot, values, size_goal, minimum_metric_quality)
            if size_goal is not None
            else topology_operations
            and np.all(
                (lengths >= lower_metric_length) & (lengths <= upper_metric_length)
            )
            and np.all(quality >= minimum_metric_quality)
        )
        if satisfied:
            status = MetricRemeshingStatus.COMPLETE
            break
        applied = 0
        if topology_operations:
            with _measure_statistical_phase(budget, "metric_split"):
                (
                    snapshot,
                    values,
                    changes,
                    attempts,
                    refusals,
                    exhausted,
                    parameters,
                    compounds,
                ) = _split_pass(
                    native,
                    snapshot,
                    values,
                    source_tables,
                    source_metric,
                    ancestry,
                    protected_edges,
                    maximum_operations - operations,
                    maximum_vertices,
                    maximum_cells,
                    maximum_cavity_cells,
                    budget,
                    upper_metric_length,
                    objective,
                )
            counts[0] += changes
            compound_splits += compounds
            applied += changes
            operations += attempts
            rejected += refusals
            split_parameters.extend(parameters)
            if exhausted or budget.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            if _physical_size_goal_met(
                snapshot, values, size_goal, minimum_metric_quality
            ):
                status = MetricRemeshingStatus.COMPLETE
                break
            with _measure_statistical_phase(budget, "metric_collapse"):
                snapshot, changes, attempts, refusals = _collapse_pass(
                    native,
                    snapshot,
                    values,
                    fixed_vertices,
                    ancestry,
                    maximum_operations - operations,
                    minimum_metric_quality,
                    maximum_cavity_cells,
                    budget,
                    lower_metric_length,
                    objective,
                )
            counts[1] += changes
            applied += changes
            operations += attempts
            rejected += refusals
            if budget.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            if _physical_size_goal_met(
                snapshot, values, size_goal, minimum_metric_quality
            ):
                status = MetricRemeshingStatus.COMPLETE
                break
            with _measure_statistical_phase(budget, "metric_reconnection"):
                snapshot, changes, attempts = _flip_pass(
                    native,
                    snapshot,
                    values,
                    ancestry,
                    maximum_operations - operations,
                    budget,
                    objective,
                )
            counts[2] += changes
            applied += changes
            operations += attempts
            if budget.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            if _physical_size_goal_met(
                snapshot, values, size_goal, minimum_metric_quality
            ):
                status = MetricRemeshingStatus.COMPLETE
                break
            with _measure_statistical_phase(budget, "metric_reconnection"):
                snapshot, changes, attempts, exhausted = _ring_pass(
                    native,
                    snapshot,
                    values,
                    ancestry,
                    maximum_operations - operations,
                    maximum_cavity_cells,
                    budget,
                    maximum_cells,
                    objective,
                )
            counts[2] += changes
            applied += changes
            operations += attempts
            if exhausted or budget.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            if _physical_size_goal_met(
                snapshot, values, size_goal, minimum_metric_quality
            ):
                status = MetricRemeshingStatus.COMPLETE
                break
        if relocation and operations < maximum_operations:
            with _measure_statistical_phase(budget, "metric_relocation"):
                snapshot, changes, attempts = _relocate_pass(
                    native,
                    snapshot,
                    values,
                    fixed_vertices,
                    maximum_operations - operations,
                    source_tables,
                    source_metric,
                    ancestry,
                    budget,
                    objective,
                )
            counts[3] += changes
            applied += changes
            operations += attempts
            if budget.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            if _physical_size_goal_met(
                snapshot, values, size_goal, minimum_metric_quality
            ):
                status = MetricRemeshingStatus.COMPLETE
                break
        if applied == 0:
            status = (
                MetricRemeshingStatus.RESOURCE_LIMIT
                if operations >= maximum_operations
                else MetricRemeshingStatus.STALLED
            )
            if (
                not topology_operations
                and operations < maximum_operations
                and np.all(quality >= minimum_metric_quality)
            ):
                status = MetricRemeshingStatus.COMPLETE
            break
    edges = _edges(snapshot.tetrahedra)
    budget.check_deadline()
    lengths = _lengths(snapshot.points, values, edges)
    quality = _quality(snapshot.points, values, snapshot.tetrahedra)
    satisfied = (
        _physical_size_goal_met(snapshot, values, size_goal, minimum_metric_quality)
        if size_goal is not None
        else topology_operations
        and np.all((lengths >= lower_metric_length) & (lengths <= upper_metric_length))
        and np.all(quality >= minimum_metric_quality)
    )
    if status is not MetricRemeshingStatus.RESOURCE_LIMIT and satisfied:
        status = MetricRemeshingStatus.COMPLETE
    requested, achieved, issues = (
        _physical_size_compliance(snapshot, size_goal)
        if size_goal is not None
        else ([], [], [])
    )
    evidence = MetricRemeshingEvidence(
        status,
        passes=passes,
        counts=(counts[0], counts[1], counts[2], counts[3]),
        rejected_operations=rejected,
        work_units=budget.work_units,
        lengths=lengths,
        quality=quality,
        criterion=(
            MetricRemeshingCriterion.SIZE_STATISTICS
            if size_goal is not None
            else MetricRemeshingCriterion.UNIT_MESH
            if topology_operations
            else MetricRemeshingCriterion.RELOCATION_FIXED_POINT
        ),
        split_parameters=tuple(split_parameters),
        lower_metric_length=lower_metric_length,
        upper_metric_length=upper_metric_length,
        operation_attempts=operations,
        native_memory_evidence=tuple(int(value) for value in native.memory_evidence()),
        size_requested=tuple(requested),
        size_achieved=tuple(achieved),
        size_issues=tuple(issues),
        source_candidate_pairs=budget.source_candidates[0],
        metric_quality_floor=minimum_metric_quality,
        shape_radius_edge_bound=size_goal.maximum_radius_edge
        if size_goal is not None
        else None,
        shape_minimum_dihedral_degrees=(
            size_goal.minimum_dihedral_degrees if size_goal is not None else None
        ),
        resource_message=budget.resource_message,
        resource_requested=budget.resource_requested,
        resource_achieved=budget.resource_achieved,
        compound_splits=compound_splits,
        expanded_insertions=budget.expanded_insertions,
        optimizer_status_counts=(
            tuple(budget.optimizer_status_counts) if budget.optimizer_terminations else ()
        ),
        optimizer_terminations=tuple(budget.optimizer_terminations),
        optimizer_method_id=budget.optimizer_method_id,
        optimizer_tolerances=budget.optimizer_tolerances,
        optimizer_iterations=budget.optimizer_iterations,
        optimizer_evaluations=budget.optimizer_objective_evaluations,
        optimizer_constraint_evaluations=budget.optimizer_constraint_evaluations,
        optimizer_proposal_evaluations=budget.optimizer_evaluations,
        optimizer_refused_proposals=budget.optimizer_refused_proposals,
        optimizer_refusal_counts=tuple(sorted(budget.optimizer_refusals.items())),
        optimizer_host_work_units=budget.optimizer_host_work,
        optimizer_scratch_required_bytes=budget.optimizer_scratch_required_bytes,
        native_work_units=native.work_units() - budget.initial_work,
    )
    native.set_work_limit(max(0, maximum_work_units - budget.work_units))
    return NativeTetraMetricOutcome(snapshot, values, evidence, ancestry)


def execute_tetra_metric_adaptation(
    mesh: CellMesh,
    metric: ArrayLike,
    /,
    *,
    cell_classes: ArrayLike | None = None,
    facet_classes: ArrayLike | None = None,
    edge_classes: ArrayLike | None = None,
    protected_edges: ArrayLike | None = None,
    fixed_vertices: ArrayLike | None = None,
    maximum_passes: int = 16,
    topology_operations: bool = True,
    relocation: bool = True,
    minimum_metric_quality: float = 0.05,
    maximum_vertices: int = 1000000,
    maximum_cells: int = 6000000,
    maximum_operations: int = 100000,
    maximum_work_units: int = 1000000000,
    lower_metric_length: float = _LOWER,
    upper_metric_length: float = _UPPER,
    maximum_location_pairs: int = 10000000,
    maximum_cavity_cells: int = 512,
    maximum_cavity_work: int = 10000,
    maximum_scratch_bytes: int = 8000000000,
    record_phase: NativeMeshingPhaseRecorder | None = None,
    source_geometry: CellGeometrySpec | None = None,
) -> TetraMetricOutcome:
    """Native legal split/directional collapse, metric cavity improvement and relocation.

    Input is an affine tetrahedral carrier. Curved-map admission/publication is
    owned by the adaptation geometry transaction, not inferred here. All edges
    count toward the unit criterion, including immutable features. Resource
    exhaustion is distinct from a legal-operation fixed point.
    """
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if source_geometry is not None and not isinstance(source_geometry, CellGeometrySpec):
        raise TypeError("source_geometry must be CellGeometrySpec or None.")
    if mesh.ambient_dimension != 3 or any(
        block.cell_kind != "tetrahedron" for block in mesh.blocks
    ):
        raise ValueError(
            "Tetrahedral metric adaptation requires tetrahedron blocks in 3D."
        )
    budgets = (
        maximum_passes,
        maximum_vertices,
        maximum_cells,
        maximum_operations,
        maximum_work_units,
        maximum_location_pairs,
        maximum_cavity_cells,
        maximum_cavity_work,
        maximum_scratch_bytes,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in budgets
    ):
        raise ValueError("Metric schedule budgets must be nonnegative integers.")
    _stage_source_limits(mesh, maximum_vertices, maximum_cells)
    if not isinstance(topology_operations, bool) or not isinstance(relocation, bool):
        raise TypeError("Operation controls must be bool.")
    if record_phase is not None and not callable(record_phase):
        raise TypeError("record_phase must be a phase recorder callable.")
    if not 0.0 <= minimum_metric_quality <= 1.0:
        raise ValueError("minimum_metric_quality must lie in [0, 1].")
    if source_geometry is not None and isinstance(
        source_geometry.exact_source, ExactPlcCellGeometrySource
    ):
        from ..discretization._coordinate_enclosure import (
            _COORDINATE_BUDGET,
            CoordinateEnclosureBudget,
            CoordinateEnclosureResourceError,
        )

        ledger = _COORDINATE_BUDGET.get()
        native_owner = current_native_execution_budget()
        if ledger is None or native_owner is None:
            ledger = (
                CoordinateEnclosureBudget(maximum_work_units, maximum_scratch_bytes)
                if ledger is None
                else ledger
            )
            native_scope = (
                NativeExecutionBudget(
                    max_work=maximum_work_units,
                    max_geometry_queries=maximum_location_pairs,
                    max_cavity_cells=maximum_cavity_cells,
                    max_scratch_bytes=maximum_scratch_bytes,
                    max_wall_seconds=np.inf,
                )
                if native_owner is None
                else nullcontext(native_owner)
            )
            try:
                with native_scope, ledger.activate():
                    return execute_tetra_metric_adaptation(
                        mesh,
                        metric,
                        cell_classes=cell_classes,
                        facet_classes=facet_classes,
                        edge_classes=edge_classes,
                        protected_edges=protected_edges,
                        fixed_vertices=fixed_vertices,
                        maximum_passes=maximum_passes,
                        topology_operations=topology_operations,
                        relocation=relocation,
                        minimum_metric_quality=minimum_metric_quality,
                        maximum_vertices=maximum_vertices,
                        maximum_cells=maximum_cells,
                        maximum_operations=maximum_operations,
                        maximum_work_units=maximum_work_units,
                        lower_metric_length=lower_metric_length,
                        upper_metric_length=upper_metric_length,
                        maximum_location_pairs=maximum_location_pairs,
                        maximum_cavity_cells=maximum_cavity_cells,
                        maximum_cavity_work=maximum_cavity_work,
                        maximum_scratch_bytes=maximum_scratch_bytes,
                        record_phase=record_phase,
                        source_geometry=source_geometry,
                    )
            except CoordinateEnclosureResourceError as error:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "Exact tetrahedral source actions exhausted the original metric allowance.",
                    stage=MeshingStageKind.OPTIMIZATION.value,
                    requested=(
                        ("maximum_work_units", maximum_work_units),
                        ("maximum_scratch_bytes", maximum_scratch_bytes),
                    ),
                    achieved=(
                        ("coordinate_work_units", ledger.work_units),
                        ("coordinate_peak_bytes_upper", ledger.peak_bytes_upper),
                    ),
                ) from error
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (points.shape[0], 3, 3):
        raise ValueError("metric must have one 3D SPD tensor per vertex.")
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
    )
    source_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    rows = key_rows(entity_keys(mesh, 3), source_cell_ids[:, None])
    regions = _labels(cell_classes, mesh.entity_set(3).count, "cell_classes")[rows]
    block_rows = np.concatenate(
        [
            np.full((block.cell_count,), index, dtype=np.int32)
            for index, block in enumerate(mesh.blocks)
        ]
    )
    _, regions = np.unique(
        np.column_stack((block_rows, regions)), axis=0, return_inverse=True
    )
    regions = regions.astype(np.int32)
    protected = (
        np.zeros((mesh.entity_set(1).count,), dtype=np.bool_)
        if protected_edges is None
        else np.asarray(protected_edges, dtype=np.bool_)
    )
    fixed = (
        np.zeros((points.shape[0],), dtype=np.bool_)
        if fixed_vertices is None
        else np.asarray(fixed_vertices, dtype=np.bool_)
    )
    if protected.shape != (mesh.entity_set(1).count,) or fixed.shape != (
        points.shape[0],
    ):
        raise ValueError(
            "Protected-edge and fixed-vertex masks must align with the source."
        )
    faces, face_sources, segments, segment_sources = _constraints(
        mesh, cells, regions, facet_classes, edge_classes, protected
    )
    exact = None if source_geometry is None else source_geometry.exact_source
    declared = None
    if exact is not None:
        if not isinstance(exact, ExactPlcCellGeometrySource):
            raise ValueError(
                "Tetrahedral metric source ancestry requires exact PLC geometry."
            )
        if source_geometry is None:
            raise RuntimeError(
                "An exact PLC metric source lost its coordinate specification."
            )
        source_geometry.resolve(mesh)
        face_sources, segments, segment_sources, declared = _plc_native_constraints(
            mesh,
            exact,
            faces,
            segments,
            protected,
            edge_classes,
        )
    protected_keys = {
        tuple(row)
        for row in _edges(cells).tolist()
        if protected[
            key_rows(
                entity_keys(mesh, 1),
                np.sort(np.asarray(mesh.vertex_global_ids)[np.asarray(row)])[None],
            )[0]
        ]
    }
    source_tables = TetraMetricSource(
        points,
        cells,
        np.asarray(mesh.vertex_global_ids, dtype=np.int64),
        source_cell_ids,
        source_geometry=source_geometry
        if isinstance(exact, ExactPlcCellGeometrySource)
        else None,
    )
    with measure_phase(record_phase, "native_preparation"):
        native = TetMesh3D(
            points,
            cells,
            regions,
            faces,
            face_sources,
            segments,
            segment_sources,
            boundary_policy="conforming",
            max_vertices=maximum_vertices,
            max_tetrahedra=maximum_cells,
            max_scratch_bytes=maximum_scratch_bytes,
            source=declared,
        )
    try:
        outcome = execute_native_tetra_metric(
            native,
            source_tables,
            values,
            fixed_vertices=fixed,
            protected_edges=protected_keys,
            maximum_passes=maximum_passes,
            topology_operations=topology_operations,
            relocation=relocation,
            minimum_metric_quality=minimum_metric_quality,
            lower_metric_length=lower_metric_length,
            upper_metric_length=upper_metric_length,
            maximum_vertices=maximum_vertices,
            maximum_cells=maximum_cells,
            maximum_operations=maximum_operations,
            maximum_work_units=maximum_work_units,
            maximum_location_pairs=maximum_location_pairs,
            maximum_cavity_cells=maximum_cavity_cells,
            maximum_cavity_work=maximum_cavity_work,
            record_phase=record_phase,
        )
        successor = None
        target_source_points = None
        if isinstance(exact, ExactPlcCellGeometrySource):
            live = np.flatnonzero(outcome.snapshot.vertex_dimension >= 0)
            witnesses = native.source_evidence()
            successor = exact._with_witnesses(
                witnesses.witness_strata[live],
                witnesses.witness_entities[live],
                witnesses.witness_parameters[live],
            )
            target_source_points = successor.prepare(
                outcome.snapshot.points[live]
            ).vertices
        with measure_phase(record_phase, "lineage_construction"):
            edit, target_metric = _assemble(
                mesh,
                outcome.snapshot,
                outcome.metric,
                outcome.ancestry,
                maximum_location_pairs,
                source_tables=source_tables,
                exact_points=target_source_points,
            )
        return TetraMetricOutcome(edit, target_metric, outcome.evidence, successor)
    finally:
        native.close()


@dataclass(frozen=True, slots=True)
class _TangentSizePlan:
    """One sampled final point and its admitted complete-cavity objective."""

    position: np.ndarray
    metric: np.ndarray
    source_vertices: set[int]
    statistical: _StatisticalProposal
    insertion: TetMeshInsertionPlan


@dataclass(frozen=True, slots=True)
class _EdgeInsertionSeed:
    first: int
    second: int
    split_position: np.ndarray


def _compound_split_plan(
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    cavity: np.ndarray,
    children: np.ndarray,
    first: int,
    second: int,
    witness: np.ndarray,
    witness_metric: np.ndarray,
    source_metric: np.ndarray,
    objective: _StatisticalObjective,
    operation_budget: int,
    /,
) -> _TangentSizePlan | None:
    """Optimize one unpublished final split corner in its exact source stratum."""
    keys = objective.offending_edge_keys()
    segments = snapshot.segments
    on_curve = np.any(
        ((segments[:, 0] == first) & (segments[:, 1] == second))
        | ((segments[:, 0] == second) & (segments[:, 1] == first))
    )
    facets = snapshot.faces[
        np.any(snapshot.faces == first, axis=1) & np.any(snapshot.faces == second, axis=1)
    ]
    triangles = snapshot.points[facets]
    interval = np.empty((0, 3), dtype=np.float64)
    if on_curve:
        interval = snapshot.points[np.asarray((first, second), dtype=np.int32)]
        stratum = interval[1] - interval[0]
    elif triangles.size:
        stratum = np.cross(
            triangles[0, 1] - triangles[0, 0], triangles[0, 2] - triangles[0, 0]
        )
    else:
        stratum = None
    if stratum is None:
        basis = np.eye(3, dtype=np.float64)
    else:
        norm = float(np.max(np.abs(stratum)))
        if not np.isfinite(norm) or norm == 0.0:
            objective.budget.refuse_optimizer("unsupported_source_tangent")
            return None
        frame = orthonormal_frame((stratum / norm)[:, None])
        if not bool(np.asarray(frame.successful)):
            objective.budget.refuse_optimizer("unsupported_source_tangent")
            return None
        basis = np.asarray(
            frame.tangents if on_curve else frame.normal_basis, dtype=np.float64
        )
    neighbors = np.unique(cavity)
    scale = float(np.min(np.linalg.norm(snapshot.points[neighbors] - witness, axis=1)))
    if not np.isfinite(scale) or scale <= 0.0:
        objective.budget.refuse_optimizer("nonfinite_source_scale")
        return None
    coordinates = _TangentCoordinates(
        metric.shape[0], witness, basis, scale, triangles, interval
    )
    return _solve_tangent_size(
        snapshot,
        metric,
        source_metric,
        coordinates,
        cavity,
        children,
        witness_metric,
        objective,
        operation_budget,
        max(1, keys.size),
        insertion_seed=_EdgeInsertionSeed(first, second, witness),
    )


def _prepare_statistical_split(
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    source_metric: np.ndarray,
    edge: tuple[int, int],
    construction: TetMeshEdgeConstruction,
    sample: np.ndarray,
    source_vertices: set[int],
    objective: _StatisticalObjective,
    operation_budget: int,
    maximum_cells: int,
    maximum_cavity_cells: int,
    /,
) -> tuple[_TangentSizePlan | None, bool, int]:
    """Admit the actual native cavity before its one accepted-state commit."""
    budget = objective.budget
    first, second = edge
    # Every ghost is owned by a constrained external face. A cavity's old
    # cells and at most four cones per cell fit this immutable topology bound.
    capacity = min(
        maximum_cavity_cells, 5 * (snapshot.tetrahedra.shape[0] + snapshot.faces.shape[0])
    )
    if not budget.optimizer_scratch(32 * capacity + 80):
        return None, False, 0
    insertion = budget.call(
        lambda allowance: budget.native.inspect_edge_insertion(
            first,
            second,
            construction.position,
            construction.position,
            source_fraction=construction.parameter,
            maximum_cavity_cells=capacity,
            work_limit=allowance,
        )
    )
    if insertion is not None:
        target_cells = (
            snapshot.tetrahedra.shape[0]
            - insertion.removed.shape[0]
            + insertion.proposed.shape[0]
        )
        if target_cells > maximum_cells:
            budget.refuse(
                "The prepared statistical insertion exceeds its cell allowance.",
                (("maximum_cells", float(maximum_cells)),),
                (("target_cells", float(target_cells)),),
            )
            return None, False, 0
        statistical = objective.propose(
            snapshot,
            metric,
            insertion.removed,
            insertion.proposed,
            changed_vertex=metric.shape[0],
            changed_position=construction.position,
            changed_metric=sample,
        )
        if statistical is not None:
            return (
                _TangentSizePlan(
                    construction.position, sample, source_vertices, statistical, insertion
                ),
                False,
                0,
            )
    insertion = None
    if budget.exhausted:
        return None, False, 0
    incident = np.any(snapshot.tetrahedra == first, axis=1) & np.any(
        snapshot.tetrahedra == second, axis=1
    )
    cavity = snapshot.tetrahedra[incident]
    first_half, second_half = cavity.copy(), cavity.copy()
    first_half[first_half == second] = metric.shape[0]
    second_half[second_half == first] = metric.shape[0]
    children = np.concatenate((first_half, second_half))
    before = budget.optimizer_evaluations
    plan = _compound_split_plan(
        snapshot,
        metric,
        cavity,
        children,
        first,
        second,
        construction.position,
        sample,
        source_metric,
        objective,
        operation_budget,
    )
    evaluations = budget.optimizer_evaluations - before
    if plan is not None:
        target_cells = (
            snapshot.tetrahedra.shape[0]
            - plan.insertion.removed.shape[0]
            + plan.insertion.proposed.shape[0]
        )
        if target_cells > maximum_cells:
            budget.refuse(
                "The prepared statistical insertion exceeds its cell allowance.",
                (("maximum_cells", float(maximum_cells)),),
                (("target_cells", float(target_cells)),),
            )
            return None, True, evaluations
    return plan, True, evaluations


def _split_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    source_tables: TetraMetricSource,
    source_metric: np.ndarray,
    ancestry: _Ancestry,
    protected: set[tuple[int, int]],
    budget: int,
    maximum_vertices: int,
    maximum_cells: int,
    maximum_cavity_cells: int,
    native_budget: _NativeMetricBudget,
    upper_metric_length: float,
    objective: _StatisticalObjective | None,
    /,
) -> tuple[TetMeshArrays, np.ndarray, int, int, int, bool, tuple[float, ...], int]:
    edges = _edges(snapshot.tetrahedra)
    lengths = _lengths(snapshot.points, values, edges)
    candidates = np.argsort(-lengths, kind="stable")
    candidates = candidates[lengths[candidates] > upper_metric_length]
    if objective is not None:
        # Source-stratum rigidity matters: repeatedly adding curve-only degrees
        # of freedom cannot realize a global band of equal physical edges.
        # An unconstrained short edge is a legal construction witness for a
        # compound body insertion; its final position/cavity, not its midpoint,
        # must improve the original statistics and satisfy every hard guard.
        scratch = 8 * (20 * edges.shape[0] + 12 * snapshot.faces.shape[0])
        if not native_budget.optimizer_work(
            edges.shape[0] + 3 * snapshot.faces.shape[0], scratch
        ):
            return snapshot, values, 0, 0, 0, True, (), 0
        face_edges = np.sort(
            snapshot.faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1
        )
        constrained = np.concatenate((snapshot.segments, face_edges))
        encoded = (edges[:, 0].astype(np.uint64) << np.uint64(32)) | edges[:, 1].astype(
            np.uint64
        )
        constraint_keys = (
            constrained[:, 0].astype(np.uint64) << np.uint64(32)
        ) | constrained[:, 1].astype(np.uint64)
        body = ~np.isin(encoded, constraint_keys)
        structural = np.flatnonzero(body & (lengths <= upper_metric_length))
        incidence = objective.edges.counts[
            np.searchsorted(objective.edges.keys, encoded[structural])
        ]
        order = np.lexsort((-lengths[structural], incidence))
        structural = structural[order[: objective.offending_edge_keys().size]]
        candidates = np.concatenate(
            (candidates[body[candidates]], structural, candidates[~body[candidates]])
        )
    if candidates.size == 0:
        return snapshot, values, 0, 0, 0, False, (), 0
    proposals: list[tuple[int, int]] = []
    positions: list[np.ndarray] = []
    fractions: list[float] = []
    constructions: list[TetMeshEdgeConstruction] = []
    work, rejected = 0, 0
    exhausted = False
    for index in candidates:
        if work >= budget:
            exhausted = True
            break
        first, second = edges[index].tolist()
        if (first, second) in protected:
            rejected += 1
            continue
        preferred_fraction = None
        if objective is not None:
            key = (np.uint64(first) << np.uint64(32)) | np.uint64(second)
            row = int(np.searchsorted(objective.edges.keys, key))
            if row >= objective.edges.keys.size or objective.edges.keys[row] != key:
                raise RuntimeError("A split preference lacks its actual physical edge.")
            if lengths[index] > upper_metric_length:
                preferred_fraction = _preferred_size_fraction(
                    objective.goal.control.target_size,
                    snapshot.points[second] - snapshot.points[first],
                )
                if (
                    not np.isfinite(preferred_fraction)
                    or not 0.0 < preferred_fraction < 1.0
                ):
                    rejected += 1
                    continue
        work += 1
        construction = native_budget.call(
            lambda allowance: native.edge_split_point(
                first, second, preferred_fraction=preferred_fraction, work_limit=allowance
            )
        )
        if construction is None:
            rejected += 1
            continue
        point, fraction = construction.position, construction.parameter
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise RuntimeError(
                "Native edge construction returned an invalid exact parameter."
            )
        proposals.append((first, second))
        positions.append(point)
        fractions.append(fraction)
        constructions.append(construction)
    if native_budget.exhausted:
        return snapshot, values, 0, work, rejected, True, (), 0
    if not positions:
        return snapshot, values, 0, work, rejected, exhausted, (), 0
    points = np.stack(positions)
    authoritative = None
    if source_tables.source_geometry is not None:
        original_source = source_tables.source_geometry.exact_source
        if not isinstance(original_source, ExactPlcCellGeometrySource):
            raise RuntimeError("An exact metric source lost its canonical PLC authority.")
        authoritative = original_source._prepare_witnesses(
            points,
            np.asarray(
                [proposal.witness_stratum for proposal in constructions], dtype=np.int8
            ),
            np.asarray(
                [proposal.witness_entity for proposal in constructions], dtype=np.int32
            ),
            np.asarray(
                [proposal.witness_parameters for proposal in constructions],
                dtype=np.float64,
            ),
        )[0]
    stencil = native_budget.stencil(points, exact_points=authoritative)
    if stencil is None:
        return snapshot, values, 0, work, rejected, True, (), 0
    source_rows = key_rows(
        source_tables.vertex_global_ids[:, None],
        stencil[0].reshape((-1, 1)),
    ).reshape(stencil[0].shape)
    samples = _interpolate(source_metric[source_rows], stencil[1])
    applied = 0
    compounds = 0
    accepted: list[float] = []
    for index, (first, second) in enumerate(proposals):
        if not np.all(stencil[2][index]):
            rejected += 1
            continue
        before = snapshot.tetrahedra
        incident = np.any(before == first, axis=1) & np.any(before == second, axis=1)
        number = np.count_nonzero(incident)
        if number == 0:
            continue
        if (
            snapshot.points.shape[0] >= maximum_vertices
            or objective is None
            and (
                before.shape[0] + number > maximum_cells
                or 3 * number > maximum_cavity_cells
            )
        ):
            exhausted = True
            break
        plan = None
        compound = False
        if objective is not None:
            plan, compound, evaluations = _prepare_statistical_split(
                snapshot,
                values,
                source_metric,
                (first, second),
                constructions[index],
                samples[index],
                ancestry.vertex_sources[first] | ancestry.vertex_sources[second],
                objective,
                budget - work,
                maximum_cells,
                maximum_cavity_cells,
            )
            work += evaluations
            if plan is None:
                rejected += 1
                if native_budget.exhausted:
                    exhausted = True
                    break
                continue
        if plan is None:
            inserted = native_budget.call(
                lambda allowance: native.split_edge(
                    first,
                    second,
                    points[index],
                    target_size=0.0,
                    work_limit=allowance,
                    source_fraction=fractions[index],
                )
            )
        else:
            inserted = native_budget.call(
                lambda allowance: native.commit_edge_insertion(
                    plan.insertion,
                    target_size=0.0,
                    work_limit=allowance,
                )
            )
        if native_budget.exhausted:
            exhausted = True
            break
        if inserted is None:
            rejected += 1
            if compound:
                native_budget.refuse_optimizer("native_final_transaction")
            continue
        if inserted != values.shape[0]:
            raise RuntimeError("Native edge split violated append-only vertex identity.")
        sample = samples[index] if plan is None else plan.metric
        values = np.concatenate((values, sample[None]))
        ancestry.vertex_sources.append(
            ancestry.vertex_sources[first] | ancestry.vertex_sources[second]
            if plan is None
            else plan.source_vertices
        )
        compounds += int(compound)
        if plan is not None and plan.insertion.removed.shape[0] > number:
            native_budget.expanded_insertions += 1
        snapshot = native.arrays()
        _record(ancestry, before, snapshot.tetrahedra, EntityLineageKind.REFINED_FROM)
        accepted.append(fractions[index])
        applied += 1
        if objective is not None and plan is not None:
            objective.commit(plan.statistical)
            if objective.complete(snapshot, values):
                break
    return (
        snapshot,
        values,
        applied,
        work,
        rejected,
        exhausted,
        tuple(accepted),
        compounds,
    )


def _collapse_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    fixed: np.ndarray,
    ancestry: _Ancestry,
    budget: int,
    minimum_quality: float,
    maximum_cavity_cells: int,
    native_budget: _NativeMetricBudget,
    lower_metric_length: float,
    objective: _StatisticalObjective | None,
    /,
) -> tuple[TetMeshArrays, int, int, int]:
    edges = _edges(snapshot.tetrahedra)
    lengths = _lengths(snapshot.points, values, edges)
    applied, work, rejected = 0, 0, 0
    for index in np.argsort(lengths, kind="stable"):
        if (
            lengths[index] >= lower_metric_length
            or work >= budget
            or native_budget.exhausted
        ):
            break
        for removed, kept in (edges[index].tolist(), edges[index, ::-1].tolist()):
            eligible_dimension = (
                snapshot.vertex_dimension[removed] == 3
                if objective is None
                else snapshot.vertex_dimension[removed] >= 1
            )
            if not eligible_dimension or (removed < fixed.size and fixed[removed]):
                continue
            changed = np.any(snapshot.tetrahedra == removed, axis=1)
            cavity_cells = 2 * np.count_nonzero(changed)
            if cavity_cells > maximum_cavity_cells:
                raise MeshingFailure(
                    MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "A metric collapse exceeds its cavity-cell budget.",
                    stage=MeshingStageKind.TOPOLOGY_REPAIR.value,
                    requested=(("cavity_cells", float(maximum_cavity_cells)),),
                    achieved=(("cavity_cells", float(cavity_cells)),),
                )
            proposed = _collapse_cells(snapshot.tetrahedra[changed], removed, kept)
            if proposed.size == 0 or not np.all(_positive(snapshot.points, proposed)):
                rejected += 1
                continue
            statistical = None
            if objective is None:
                metric_lengths = _lengths(snapshot.points, values, _edges(proposed))
                if (
                    np.any(metric_lengths > 2.0)
                    or np.min(_quality(snapshot.points, values, proposed))
                    < minimum_quality
                ):
                    rejected += 1
                    continue
            else:
                statistical = objective.propose(
                    snapshot, values, snapshot.tetrahedra[changed], proposed
                )
                if statistical is None:
                    rejected += 1
                    continue
            work += 1
            before = snapshot.tetrahedra
            accepted = native_budget.call(
                lambda allowance: native.collapse_edge(
                    removed, kept, work_limit=allowance
                )
            )
            if not accepted:
                rejected += 1
                continue
            snapshot = native.arrays()
            if ancestry.vertex_successor is None:
                ancestry.vertex_successor = {}
            ancestry.vertex_successor[removed] = kept
            _record(
                ancestry, before, snapshot.tetrahedra, EntityLineageKind.COLLAPSED_INTO
            )
            applied += 1
            if objective is not None and statistical is not None:
                objective.commit(statistical)
                if objective.complete(snapshot, values):
                    return snapshot, applied, work, rejected
            break
    return snapshot, applied, work, rejected


def _flip_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    ancestry: _Ancestry,
    budget: int,
    native_budget: _NativeMetricBudget,
    objective: _StatisticalObjective | None,
    /,
) -> tuple[TetMeshArrays, int, int]:
    faces = np.unique(
        np.sort(snapshot.tetrahedra[:, _FACES].reshape((-1, 3)), axis=1), axis=0
    )
    applied, work = 0, 0
    for face in faces:
        if work >= budget or native_budget.exhausted:
            break
        # Explicit row selection; a face must have exactly two incident cells.
        rows = np.flatnonzero(np.sum(np.isin(snapshot.tetrahedra, face), axis=1) == 3)
        if rows.size != 2:
            continue
        cavity = snapshot.tetrahedra[rows]
        apex = cavity[~np.isin(cavity, face)]
        proposed = _orient(
            snapshot.points,
            np.asarray(
                (
                    (apex[0], apex[1], face[0], face[1]),
                    (apex[0], apex[1], face[1], face[2]),
                    (apex[0], apex[1], face[2], face[0]),
                ),
                dtype=np.int32,
            ),
        )
        if not np.all(_positive(snapshot.points, proposed)):
            continue
        statistical = None
        if objective is None:
            if (
                np.min(_quality(snapshot.points, values, proposed))
                <= np.min(_quality(snapshot.points, values, cavity)) + 1.0e-6
            ):
                continue
        else:
            statistical = objective.propose(snapshot, values, cavity, proposed)
            if statistical is None:
                continue
        work += 1
        before = snapshot.tetrahedra
        if native_budget.call(
            lambda _: native.flip_face(face, require_improvement=False)
        ):
            snapshot = native.arrays()
            _record(ancestry, before, snapshot.tetrahedra, EntityLineageKind.SWAPPED_FROM)
            applied += 1
            if objective is not None and statistical is not None:
                objective.commit(statistical)
                if objective.complete(snapshot, values):
                    break
    return snapshot, applied, work


def _polygon_triangulations(
    polygon: tuple[int, ...], /
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """All Catalan triangulations of a bounded edge link (at most seven vertices)."""
    if len(polygon) < 3:
        return ((),)
    result = []
    for split in range(1, len(polygon) - 1):
        for left in _polygon_triangulations(polygon[: split + 1]):
            for right in _polygon_triangulations(polygon[split:]):
                result.append((*left, *right, (polygon[0], polygon[split], polygon[-1])))
    return tuple(result)


def _edge_ring(cavity: np.ndarray, first: int, second: int, /) -> tuple[int, ...]:
    neighbors: dict[int, set[int]] = {}
    for cell in cavity.tolist():
        opposite = [vertex for vertex in cell if vertex not in (first, second)]
        if len(opposite) != 2:
            return ()
        neighbors.setdefault(opposite[0], set()).add(opposite[1])
        neighbors.setdefault(opposite[1], set()).add(opposite[0])
    if len(neighbors) != cavity.shape[0] or any(
        len(values) != 2 for values in neighbors.values()
    ):
        return ()
    ring = [min(neighbors)]
    previous, current = ring[0], min(neighbors[ring[0]])
    while current != ring[0]:
        if current in ring or len(ring) > 7:
            return ()
        ring.append(current)
        following = neighbors[current] - {previous}
        if len(following) != 1:
            return ()
        previous, current = current, next(iter(following))
    return tuple(ring) if len(ring) == len(neighbors) else ()


def _ring_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    ancestry: _Ancestry,
    budget: int,
    maximum_cavity_cells: int,
    native_budget: _NativeMetricBudget,
    maximum_cells: int,
    objective: _StatisticalObjective | None,
    /,
) -> tuple[TetMeshArrays, int, int, bool]:
    applied, work = 0, 0
    for first, second in _edges(snapshot.tetrahedra).tolist():
        if native_budget.exhausted:
            return snapshot, applied, work, True
        rows = np.flatnonzero(
            np.any(snapshot.tetrahedra == first, axis=1)
            & np.any(snapshot.tetrahedra == second, axis=1)
        )
        if rows.size < 3 or rows.size > 7:
            continue
        if np.unique(snapshot.tetrahedron_regions[rows]).size != 1:
            continue
        cavity = snapshot.tetrahedra[rows]
        ring = _edge_ring(cavity, first, second)
        if not ring:
            continue
        triangles = _polygon_triangulations(ring)
        cell_count = 2 * (len(ring) - 2)
        if len(triangles) * cell_count > maximum_cavity_cells:
            return snapshot, applied, work, True
        if work + len(triangles) > budget:
            return snapshot, applied, work, True
        if snapshot.tetrahedra.shape[0] - rows.size + cell_count > maximum_cells:
            return snapshot, applied, work, True
        proposed = np.asarray(
            [
                [
                    (apex, *triangle)
                    for triangle in triangulation
                    for apex in (first, second)
                ]
                for triangulation in triangles
            ],
            dtype=np.int32,
        )
        proposed = _orient(snapshot.points, proposed.reshape((-1, 4))).reshape(
            proposed.shape
        )
        positive = (
            _positive(snapshot.points, proposed.reshape((-1, 4)))
            .reshape((-1, cell_count))
            .all(axis=1)
        )
        quality = (
            _quality(snapshot.points, values, proposed.reshape((-1, 4)))
            .reshape((-1, cell_count))
            .min(axis=1)
        )
        work += len(triangles)
        worst = float(np.min(_quality(snapshot.points, values, cavity)))
        order = np.argsort(-quality, kind="stable")
        for candidate in order:
            if not positive[candidate]:
                continue
            statistical = None
            if objective is None:
                if (
                    quality[candidate] <= worst + 1.0e-6
                    or np.max(
                        _lengths(snapshot.points, values, _edges(proposed[candidate])),
                        initial=0.0,
                    )
                    > 2.0
                ):
                    continue
            else:
                statistical = objective.propose(
                    snapshot, values, cavity, proposed[candidate]
                )
                if statistical is None:
                    continue
            before = snapshot.tetrahedra
            if native_budget.call(
                lambda allowance: native.reconnect(
                    cavity, proposed[candidate], work_limit=allowance
                )
            ):
                snapshot = native.arrays()
                _record(
                    ancestry, before, snapshot.tetrahedra, EntityLineageKind.SWAPPED_FROM
                )
                applied += 1
                if objective is not None and statistical is not None:
                    objective.commit(statistical)
                    if objective.complete(snapshot, values):
                        return snapshot, applied, work, False
                break
    return snapshot, applied, work, False


@dataclass(frozen=True, slots=True)
class _TangentCoordinates:
    """Immutable numerical coordinates of one actual scientific source stratum."""

    vertex: int
    origin: np.ndarray
    basis: np.ndarray
    scale: float
    source_triangles: np.ndarray
    curve_endpoints: np.ndarray

    def position(self, coordinates: np.ndarray, /) -> np.ndarray:
        return self.origin + self.scale * (self.basis @ coordinates)

    def represented_source_contains(self, position: np.ndarray, /) -> bool:
        if self.curve_endpoints.size:
            first, second = self.curve_endpoints
            for axes in ((0, 1), (0, 2), (1, 2)):
                indices = np.asarray(axes, dtype=np.int32)
                sign = exact_orient2d(first[indices], second[indices], position[indices])
                if int(sign) != int(PredicateSign.ZERO):
                    return False
            return bool(
                np.all(position >= np.minimum(first, second))
                and np.all(position <= np.maximum(first, second))
                and not np.array_equal(position, first)
                and not np.array_equal(position, second)
            )
        if self.source_triangles.size:
            triangles = _bucket(self.source_triangles)
            signs = exact_orient3d(
                triangles[:, 0],
                triangles[:, 1],
                triangles[:, 2],
                np.broadcast_to(position, (triangles.shape[0], 3)),
            )[: self.source_triangles.shape[0]]
            return bool(np.all(signs == int(PredicateSign.ZERO)))
        return True


def _tangent_coordinates(
    snapshot: TetMeshArrays,
    vertex: int,
    cavity: np.ndarray,
    /,
    *,
    facet_rows: np.ndarray | None = None,
    segment_rows: np.ndarray | None = None,
) -> _TangentCoordinates | None:
    dimension = int(snapshot.vertex_dimension[vertex])
    triangles = np.empty((0, 3, 3), dtype=np.float64)
    interval = np.empty((0, 3), dtype=np.float64)
    if dimension == 3:
        basis = np.eye(3, dtype=np.float64)
    elif dimension == 1:
        rows = (
            np.flatnonzero(np.any(snapshot.segments == vertex, axis=1))
            if segment_rows is None
            else segment_rows
        )
        segments = snapshot.segments[rows]
        neighbors = np.unique(segments[segments != vertex])
        if neighbors.size != 2 or np.unique(snapshot.segment_sources[rows]).size != 1:
            return None
        interval = snapshot.points[neighbors]
        direction = interval[1] - interval[0]
        norm = float(np.max(np.abs(direction)))
        if norm == 0.0 or not np.isfinite(norm):
            return None
        frame = orthonormal_frame((direction / norm)[:, None])
        if not bool(np.asarray(frame.successful)):
            return None
        basis = np.asarray(frame.tangents, dtype=np.float64)
    elif dimension == 2:
        rows = (
            np.flatnonzero(np.any(snapshot.faces == vertex, axis=1))
            if facet_rows is None
            else facet_rows
        )
        faces = snapshot.faces[rows]
        if not faces.shape[0]:
            return None
        triangles = snapshot.points[faces]
        normal = np.cross(
            triangles[0, 1] - triangles[0, 0], triangles[0, 2] - triangles[0, 0]
        )
        norm = float(np.max(np.abs(normal)))
        if norm == 0.0 or not np.isfinite(norm):
            return None
        frame = orthonormal_frame((normal / norm)[:, None])
        if not bool(np.asarray(frame.successful)):
            return None
        basis = np.asarray(frame.normal_basis, dtype=np.float64)
    else:
        return None
    neighbors = np.unique(cavity)
    neighbors = neighbors[neighbors != vertex]
    scale = float(
        np.min(
            np.linalg.norm(snapshot.points[neighbors] - snapshot.points[vertex], axis=1)
        )
    )
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    return _TangentCoordinates(
        vertex, snapshot.points[vertex], basis, scale, triangles, interval
    )


@dataclass(slots=True)
class _TangentSizeObjective:
    """Closed original-size objective and separately evidenced hard refusals."""

    snapshot: TetMeshArrays
    metric: np.ndarray
    source_metric: np.ndarray
    coordinates: _TangentCoordinates
    initial_metric: np.ndarray
    objective: _StatisticalObjective
    scratch_bytes: int
    insertion_seed: _EdgeInsertionSeed
    previous: np.ndarray | None = None
    evaluation: tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    insertion_plan: TetMeshInsertionPlan | None = None

    def evaluate(
        self, coordinates: np.ndarray, /
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parameters = np.asarray(coordinates, dtype=np.float64)
        if self.previous is not None and np.array_equal(parameters, self.previous):
            if self.evaluation is None:
                raise RuntimeError(
                    "A cached insertion proposal lacks its evaluated evidence."
                )
            return self.evaluation
        budget = self.objective.budget
        position = self.coordinates.position(parameters)
        sample = self.initial_metric
        source_vertices = np.empty((0,), dtype=np.int64)
        constraints = np.zeros((5,), dtype=np.float64)
        scratch_bytes = self.scratch_bytes
        self.insertion_plan = None
        inspection_bytes = 32 * budget.maximum_cavity_cells + 80
        if not budget.optimizer_scratch(scratch_bytes + inspection_bytes):
            return np.inf, np.full((5,), np.inf), position, sample, source_vertices
        seed = self.insertion_seed
        plan = budget.call(
            lambda allowance: budget.native.inspect_edge_insertion(
                seed.first,
                seed.second,
                seed.split_position,
                position,
                maximum_cavity_cells=budget.maximum_cavity_cells,
                work_limit=allowance,
            )
        )
        if plan is None:
            budget.optimizer_work(0, scratch_bytes + inspection_bytes)
            budget.refuse_optimizer("native_insertion_cavity")
            return np.inf, np.full((5,), np.inf), position, sample, source_vertices
        self.insertion_plan = plan
        removed, proposed = plan.removed, plan.proposed
        vertices = np.unique(proposed)
        cells = np.searchsorted(vertices, proposed).astype(np.int32)
        cavity_edges = 6 * (removed.shape[0] + proposed.shape[0])
        host_work = (
            self.objective.edges.keys.size
            + cavity_edges
            + vertices.size
            + (
                3
                if self.coordinates.curve_endpoints.size
                else self.coordinates.source_triangles.shape[0]
            )
        )
        scratch_bytes = (
            max(
                scratch_bytes,
                (
                    8 * (9 * self.objective.edges.keys.size + 24 * cavity_edges)
                    + 8 * (6 * self.snapshot.points.shape[0] + 12 * vertices.size)
                    + cells.nbytes
                    + self.coordinates.source_triangles.nbytes
                ),
            )
            + plan.buffer_bytes
        )
        if not budget.optimizer_work(host_work, scratch_bytes):
            return np.inf, np.full((5,), np.inf), position, sample, source_vertices
        local_points = _proposal_rows(
            self.snapshot.points, vertices, self.coordinates.vertex, position
        )
        source_admitted = self.coordinates.represented_source_contains(position)
        constraints[0] = 0.0 if source_admitted else 1.0
        positive = bool(np.all(_positive(local_points, cells)))
        constraints[1] = 0.0 if positive else 1.0
        stencil = budget.stencil(position[None])
        if stencil is None or not bool(np.all(stencil[2])):
            constraints[0] = 1.0
        else:
            source_rows = key_rows(
                budget.source_tables.vertex_global_ids[:, None],
                stencil[0].reshape((-1, 1)),
            ).reshape(stencil[0].shape)
            sample = _interpolate(self.source_metric[source_rows], stencil[1])[0]
            source_vertices = stencil[0][0, stencil[1][0] > 0.0]
        local_metric = _proposal_rows(
            self.metric, vertices, self.coordinates.vertex, sample
        )
        quality = _quality(local_points, local_metric, cells)
        constraints[2] = self.objective.minimum_quality - float(np.min(quality))
        if positive:
            # The determinant floor is decided once, with its refusal reason,
            # by the owning objective proposal that admission always runs.
            shape = budget.call(
                lambda allowance: budget.native.proposal_shape(
                    local_points, cells, vertex_ids=vertices, work_limit=allowance
                )
            )
            if shape is None:
                constraints[3:] = np.inf
            else:
                radius, dihedral, _ = shape
                goal = self.objective.goal
                if goal.maximum_radius_edge is not None:
                    constraints[3] = radius - goal.maximum_radius_edge
                if goal.minimum_dihedral_degrees is not None:
                    constraints[4] = goal.minimum_dihedral_degrees - dihedral
        else:
            constraints[3:] = 1.0
        physical_refusal: tuple[str, ...] = ()
        try:
            ledger = self.objective.edges.project(
                self.snapshot.points,
                removed,
                proposed,
                self.objective.goal,
                self.coordinates.vertex,
                position,
            )
            value = ledger.merit.normalized_deficit
        except MeshingFailure as error:
            if error.evidence.category is not MeshingFailureCategory.COMPLIANCE_FAILED:
                raise
            # An optimizer-domain trial with zero/nonfinite physical edges is
            # a nonfinite evaluation, never substituted finite size evidence.
            value = np.inf
            constraints[1] = np.inf
            physical_refusal = ("nonfinite_physical_edges",)
        reasons = physical_refusal + tuple(
            name
            for name, violation in zip(
                (
                    "represented_source",
                    "positive_cells",
                    "metric_quality",
                    "radius_edge",
                    "minimum_dihedral",
                ),
                constraints,
                strict=True,
            )
            if not np.isfinite(violation) or violation > 0.0
        )
        if reasons:
            budget.refuse_optimizer(*reasons)
        self.previous = parameters.copy()
        self.evaluation = (value, constraints, position, sample, source_vertices)
        return self.evaluation


_EQUAL_LENGTH_METHOD = LevenbergMarquardt(linear_policy=LinearSolvePolicy(DenseQR()))
# Zero tolerances: each solve ends on its exact optimality/step stagnation, a
# failed trust region, or the declared step/residual-evaluation caps below.
_EQUAL_LENGTH_TERMINATION = OptimizationTermination(
    absolute_optimality=0.0,
    relative_optimality=0.0,
    absolute_step=0.0,
    relative_step=0.0,
    maximum_steps=32,
    maximum_evaluations=128,
)
_EQUAL_LENGTH_CANDIDATES = 7
_EQUAL_LENGTH_GOVERNING = 4


_EqualLengthArgs = tuple[Array, Array, Array, Array, Array, Array, Array, Array]


def _equal_length_residual(coordinates: Array, args: _EqualLengthArgs, /) -> Array:
    """Pinned edges at the target plus one-sided excess of the other star edges.

    ``pinned`` rows are driven to the target length exactly. ``longer`` and
    ``shorter`` weight only the failed statistic's offending side of every
    other incident edge, so a pin is never traded for an edge already inside
    its admissible side; padded rows carry zero weight.
    """
    origin, basis, scale, neighbors, target, pinned, longer, shorter = args
    position = origin + scale * (basis @ coordinates)
    relative = jnp.sqrt(jnp.sum((neighbors - position[None]) ** 2, axis=1)) / target - 1.0
    free = 1.0 - pinned
    return (
        pinned * relative
        + free * longer * jnp.maximum(relative, 0.0)
        + free * shorter * jnp.minimum(relative, 0.0)
    )


@eqx.filter_jit
def _equal_length_solve(
    args: _EqualLengthArgs, /
) -> tuple[Array, Array, Array, Array, Array]:
    """Bounded exact-derivative least squares of one star at one target length."""
    basis = args[1]
    result = least_squares(
        _equal_length_residual,
        jnp.zeros((basis.shape[1],), dtype=basis.dtype),
        method=_EQUAL_LENGTH_METHOD,
        termination=_EQUAL_LENGTH_TERMINATION,
        args=args,
    )
    diagnostics = result.diagnostics
    return (
        jnp.asarray(result.parameters),
        jnp.asarray(result.status),
        jnp.asarray(diagnostics.iterations),
        jnp.asarray(diagnostics.residual_evaluations),
        jnp.asarray(
            diagnostics.residual_evaluations
            + diagnostics.jvp_evaluations
            + diagnostics.vjp_evaluations
        ),
    )


def _equal_length_subsets(
    governing: np.ndarray, deviations: np.ndarray, dimension: int, /
) -> list[tuple[int, ...]]:
    """Bounded edge selections pinned at the target; each fixes a governing edge.

    Governing edges are ranked by largest deviation; the remaining incident
    edges by smallest, so an already compliant edge can stay at the target.
    The empty selection leaves only the one-sided statistic excess.
    """
    rows = np.arange(deviations.size)
    fixing = rows[governing][np.argsort(-deviations[governing], kind="stable")][
        :_EQUAL_LENGTH_GOVERNING
    ]
    keeping = rows[~governing][np.argsort(deviations[~governing], kind="stable")]
    ranked = np.concatenate((fixing, keeping))[:_EQUAL_LENGTH_CANDIDATES].tolist()
    allowed = set(fixing.tolist())
    return [()] + [
        subset
        for size in range(1, min(dimension, len(ranked)) + 1)
        for subset in combinations(ranked, size)
        if allowed.intersection(subset)
    ]


def _failed_statistic_sides(objective: _StatisticalObjective, /) -> tuple[bool, bool]:
    """Whether a failed owning target statistic lies above and/or below target."""
    merit = objective.edges.merit
    actual = dict(merit.achieved)
    prefix = f"size:{objective.goal.control.control_id}"
    target = objective.goal.control.target_size
    failed = {issue.split(":", 1)[0] for issue in merit.issues}
    measured = [
        actual[f"{prefix}:{statistic}_edge"]
        for statistic in objective.goal.policy.target_statistics
        if f"target_size_{statistic}" in failed
    ]
    return (
        any(value > target for value in measured),
        any(value < target for value in measured),
    )


def _solve_tangent_size(
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    source_metric: np.ndarray,
    coordinates: _TangentCoordinates,
    removed: np.ndarray,
    proposed: np.ndarray,
    initial_metric: np.ndarray,
    objective: _StatisticalObjective,
    operation_budget: int,
    remaining_candidates: int,
    /,
    *,
    insertion_seed: _EdgeInsertionSeed,
) -> _TangentSizePlan | None:
    """Own a bounded source-stratum final insertion solve, never vertex relocation.

    Bounded selections of the vertex's governing physical edges are solved to
    the original target length by exact-derivative Levenberg--Marquardt in the
    vertex's source stratum. Every solution is admitted only through the
    source, embedding, metric-quality and native shape refusals and the owning
    statistical merit; the best admitted proposal is returned.
    """
    budget = objective.budget
    vertices = np.unique(proposed)
    cells = np.searchsorted(vertices, proposed).astype(np.int32)
    edge_count = objective.edges.keys.size
    cavity_edge_capacity = 6 * (removed.shape[0] + proposed.shape[0])
    # This explicit host scratch plan does not claim a global JAX/device peak.
    scratch = (
        8 * (9 * edge_count + 24 * cavity_edge_capacity)
        + 8 * (6 * snapshot.points.shape[0] + 12 * vertices.size)
        + cells.nbytes
        + coordinates.source_triangles.nbytes
    )
    host_work = (
        edge_count
        + cavity_edge_capacity
        + vertices.size
        + (
            3
            if coordinates.curve_endpoints.size
            else coordinates.source_triangles.shape[0]
        )
    )
    native_shape_work = 3 * vertices.size + 5 * proposed.shape[0]
    remaining_work = budget.maximum_work_units - budget.work_units
    maximum_evaluations = min(
        operation_budget // remaining_candidates,
        remaining_work // ((host_work + native_shape_work) * remaining_candidates),
        (budget.maximum_source_pairs - budget.source_candidates[0])
        // (budget.source_tables.cells.shape[0] * remaining_candidates),
    )
    if maximum_evaluations < 1:
        budget.refuse(
            "The bounded statistical tangent proposal has no remaining evaluation allowance.",
            (
                ("maximum_work_units", float(budget.maximum_work_units)),
                ("maximum_source_pairs", float(budget.maximum_source_pairs)),
            ),
            (
                ("work_units", float(budget.work_units)),
                ("source_candidate_pairs", float(budget.source_candidates[0])),
            ),
        )
        return None
    evaluator = _TangentSizeObjective(
        snapshot,
        metric,
        source_metric,
        coordinates,
        initial_metric,
        objective,
        scratch,
        insertion_seed,
    )
    target = objective.goal.control.target_size
    resolution = objective.goal.policy.tolerance(target) / target
    dimension = coordinates.basis.shape[1]
    best: _TangentSizePlan | None = None
    best_rank: tuple[float, float, float] | None = None
    evaluations = 0

    def admit(parameters: np.ndarray, /) -> tuple[np.ndarray, np.ndarray] | None:
        nonlocal best, best_rank, evaluations
        evaluations += 1
        _, constraints, position, sample, sources = evaluator.evaluate(parameters)
        insertion = evaluator.insertion_plan
        if np.any(constraints > 0.0) or not np.all(np.isfinite(constraints)):
            return None
        if insertion is None:
            budget.refuse_optimizer("native_final_insertion_cavity")
            return None
        proposal = objective.propose(
            snapshot,
            metric,
            insertion.removed,
            insertion.proposed,
            changed_vertex=coordinates.vertex,
            changed_position=position,
            changed_metric=sample,
        )
        if proposal is not None:
            merit = proposal.edges.merit
            rank = (
                merit.normalized_deficit,
                merit.directional_edge_penalty,
                -proposal.quality_sum / proposal.cell_count,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = _TangentSizePlan(
                    position, sample, set(sources.tolist()), proposal, insertion
                )
        return position, insertion.proposed

    initial = admit(np.zeros((dimension,), dtype=np.float64))
    if budget.exhausted:
        return None
    star = proposed if initial is None else initial[1]
    origin = coordinates.origin if initial is None else initial[0]
    neighbors = np.unique(star[np.any(star == coordinates.vertex, axis=1)])
    neighbors = neighbors[neighbors != coordinates.vertex]
    deviations = np.abs(
        np.linalg.norm(snapshot.points[neighbors] - origin, axis=1) / target - 1.0
    )
    # Every edge of an inserted vertex is new; deviation beyond the owning
    # compliance tolerance governs.
    governing = deviations > resolution
    subsets = _equal_length_subsets(governing, deviations, dimension)
    budget.optimizer_terminations.append(
        (
            _EQUAL_LENGTH_TERMINATION.maximum_steps,
            int(_EQUAL_LENGTH_TERMINATION.maximum_evaluations or 0),
        )
    )
    budget.optimizer_method_id = _EQUAL_LENGTH_METHOD.method_id
    budget.optimizer_tolerances = (
        _EQUAL_LENGTH_TERMINATION.absolute_optimality,
        _EQUAL_LENGTH_TERMINATION.relative_optimality,
        _EQUAL_LENGTH_TERMINATION.absolute_step,
        _EQUAL_LENGTH_TERMINATION.relative_step,
    )
    longer, shorter = _failed_statistic_sides(objective)
    star_points = _bucket(snapshot.points[neighbors])
    rows = star_points.shape[0]
    # Pure pins, and pins with the failed side's one-sided star excess.
    variants = [
        (subset, sided)
        for subset in subsets
        for sided in ((True,) if not subset else (False, True))
    ]
    for subset, sided in variants:
        if evaluations >= maximum_evaluations or budget.exhausted:
            break
        pinned = np.zeros((rows,), dtype=np.float64)
        pinned[np.asarray(subset, dtype=np.int64)] = 1.0
        active = np.zeros((rows,), dtype=np.float64)
        active[: neighbors.size] = 1.0 if sided else 0.0
        solve_bytes = 8 * (star_points.size + 3 * dimension + 4 * rows + 8)
        bound = (
            4
            + _EQUAL_LENGTH_TERMINATION.maximum_steps
            * (2 + _EQUAL_LENGTH_METHOD.maximum_trials * (2 * dimension + 12))
        ) * rows
        if bound > budget.maximum_work_units - budget.work_units:
            budget.optimizer_work(bound, scratch + solve_bytes)
            return None
        if budget.execution_budget is not None:
            budget.execution_budget.admit_work_bound(bound)
        budget.check_deadline()
        parameters, status, iterations, residuals, actions = _equal_length_solve(
            (
                jnp.asarray(coordinates.origin),
                jnp.asarray(coordinates.basis),
                jnp.asarray(coordinates.scale),
                jnp.asarray(star_points),
                jnp.asarray(target),
                jnp.asarray(pinned),
                jnp.asarray(active * float(longer)),
                jnp.asarray(active * float(shorter)),
            )
        )
        residual_evaluations = int(residuals)
        budget.optimizer_status_counts[int(status)] += 1
        budget.optimizer_iterations += int(iterations)
        budget.optimizer_objective_evaluations += residual_evaluations
        if not budget.optimizer_work(int(actions) * rows, scratch + solve_bytes):
            return None
        solution = np.asarray(parameters, dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            budget.refuse_optimizer("nonfinite_equal_length_solution")
            continue
        admit(solution)
    if budget.exhausted:
        return None
    if best is None:
        budget.refuse_optimizer("nonimproving_final_proposal")
    return best


class _GlobalSizeArgs(NamedTuple):
    origins: Array
    basis: Array
    edges: Array
    cells: Array
    quantiles: Array
    edge_valid: Array
    cell_valid: Array
    initial_volume: Array
    target: Array
    tolerance: Array
    dihedral_cosine: Array
    radius_edge_bound: Array


_GLOBAL_LINEAR_STEPS = 32
_GLOBAL_SIZE_METHOD = LevenbergMarquardt(
    linear_policy=LinearSolvePolicy(
        GeneralizedLSMR(),
        tolerance=TolerancePolicy(
            relative=1.0e-10, absolute=1.0e-14, max_steps=_GLOBAL_LINEAR_STEPS
        ),
    ),
    initial_damping=100.0,
    maximum_trials=4,
)
_GLOBAL_SIZE_TERMINATION = OptimizationTermination(
    absolute_optimality=0.0,
    relative_optimality=0.0,
    absolute_step=0.0,
    relative_step=0.0,
    maximum_steps=32,
    maximum_evaluations=128,
)
_GLOBAL_TRIAL_SCALES = tuple(2.0**-level for level in range(16))


def _global_size_canonical_residual_bound() -> int:
    """Bound only real native LM primals, never its separately charged models."""
    trials = _GLOBAL_SIZE_METHOD.maximum_trials
    # One initial model, one model plus at most one primal per trial/step,
    # and one final model. The outer evaluation guard can overshoot by one
    # complete step before the final model; there is no auxiliary callback.
    bound = 2 + _GLOBAL_SIZE_TERMINATION.maximum_steps * (1 + trials)
    evaluations = _GLOBAL_SIZE_TERMINATION.maximum_evaluations
    return bound if evaluations is None else min(bound, evaluations + trials + 1)


def _size_quantile_parts(
    lengths: Array, quantiles: Array
) -> tuple[Array, Array, Array, Array, Array]:
    valid = ~jnp.isnan(lengths)
    order = jnp.argsort(jnp.where(valid, lengths, jnp.inf), stable=True)
    ordered = lengths[order]
    ranks = (jnp.sum(valid) - 1) * quantiles
    lower = jnp.floor(ranks).astype(jnp.int32)
    upper = jnp.ceil(ranks).astype(jnp.int32)
    fraction = ranks - lower
    # Match the canonical native nanquantile linear interpolation operation
    # order, rather than replacing it with an algebraically equal rounding.
    statistic = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return statistic, order, lower, upper, fraction


@custom_jvp
def _size_quantile(lengths: Array, quantiles: Array) -> Array:
    """Canonical primal with a permutation-symmetric linear tie selection.

    This is the average rank-permutation derivative in exact tied blocks,
    not the nonlinear one-sided directional derivative of an order statistic.
    Trial merit checks must select their outgoing ranks independently.
    """
    return _size_quantile_parts(lengths, quantiles)[0]


@_size_quantile.defjvp
def _size_quantile_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    lengths, quantiles = primals
    length_tangent, quantile_tangent = tangents
    statistic, order, lower, upper, fraction = _size_quantile_parts(lengths, quantiles)
    ordered = lengths[order]
    valid = ~jnp.isnan(ordered)
    # Exact equality only: distinct adjacent floating values remain distinct
    # branches. Groups and their coefficients belong to the prepared model.
    start = jnp.concatenate(
        (jnp.ones((1,), dtype=jnp.bool_), ordered[1:] != ordered[:-1])
    )
    group = jnp.cumsum(start.astype(jnp.int32)) - 1
    count = jnp.zeros_like(lengths).at[group].add(valid.astype(lengths.dtype))
    total = (
        jnp.zeros_like(lengths)
        .at[group]
        .add(jnp.where(valid, length_tangent[order], 0.0))
    )
    average = total / jnp.maximum(count, 1.0)
    selected = average[group]
    derivative = selected[lower] * (1.0 - fraction) + selected[upper] * fraction
    derivative += (
        (ordered[upper] - ordered[lower]) * (jnp.sum(valid) - 1) * quantile_tangent
    )
    return statistic, derivative


_GLOBAL_SIZE_TRIAL_POLICY_ID = "exact-linear-quantile/tie-barycenter/gap-scaled-component-rank-frontier/finite-source-trajectory-hard-shape-model/outgoing-events"


def _global_size_trial_work(
    vertex_count: int, edge_count: int, cell_count: int, quantile_count: int
) -> tuple[int, int, int]:
    sort_work = 4 * edge_count * (edge_count - 1).bit_length() ** 2
    fixed = 4 * vertex_count + 30 * edge_count + 40 * cell_count + sort_work
    model = vertex_count + 8 * edge_count + 400 * cell_count + sort_work
    event = 100 * quantile_count * edge_count
    return fixed, model, event


def _global_size_branch_scratch(vertex_count: int) -> int:
    # A single capacity-32 bank has at most 48 combined rounded inequality
    # and equality rows (32+16). Include its input rows and integer frontier.
    parameter_size = 3 * vertex_count
    active_bucket, equality_bucket, capacity = 32, 16, 32
    return 8 * (
        (4 * active_bucket + 4 * equality_bucket + 4 + capacity) * parameter_size
        + 12 * (active_bucket**2 + equality_bucket**2)
        + 8 * active_bucket * equality_bucket
        + 8 * capacity
    )


class _GlobalSizeBranchBank(NamedTuple):
    statistic: Array
    lower: Array
    upper: Array
    count: Array
    overflow: Array


_GlobalSizeFrontierCarry: TypeAlias = tuple[
    _GlobalSizeBranchBank, Array, Array, Array, Array, Array, Array
]


def _append_global_size_branches(
    bank: _GlobalSizeBranchBank,
    statistic: Array,
    lower: Array,
    upper: Array,
    active: Array,
) -> tuple[_GlobalSizeBranchBank, Array]:
    """Retain every requested branch or refuse the whole overflowing bank."""
    capacity = bank.statistic.size
    statistic = statistic.astype(bank.statistic.dtype)
    lower = lower.astype(bank.lower.dtype)
    upper = upper.astype(bank.upper.dtype)
    slots = jnp.arange(capacity)

    def append(
        index: Array, carry: tuple[_GlobalSizeBranchBank, Array]
    ) -> tuple[_GlobalSizeBranchBank, Array]:
        current, checks = carry

        def admitted(_: None) -> tuple[_GlobalSizeBranchBank, Array]:
            present = jnp.any(
                (slots < current.count)
                & (current.statistic == statistic[index])
                & (current.lower == lower[index])
                & (current.upper == upper[index])
            )

            def insert(_: None) -> _GlobalSizeBranchBank:
                def available(_: None) -> _GlobalSizeBranchBank:
                    slot = current.count
                    return _GlobalSizeBranchBank(
                        current.statistic.at[slot].set(statistic[index]),
                        current.lower.at[slot].set(lower[index]),
                        current.upper.at[slot].set(upper[index]),
                        current.count + 1,
                        current.overflow,
                    )

                return lax.cond(
                    current.count < capacity,
                    available,
                    lambda _: current._replace(overflow=jnp.asarray(True)),
                    None,
                )

            result = lax.cond(present, lambda _: current, insert, None)
            return result, checks + 1

        return lax.cond(
            active[index] & ~current.overflow, admitted, lambda _: (current, checks), None
        )

    result, checks = lax.fori_loop(
        0, statistic.size, append, (bank, jnp.asarray(0, dtype=jnp.int32))
    )
    return result, capacity + statistic.size + checks * (8 * capacity) + 4 * (
        result.count - bank.count
    )


def _global_size_branch_gradients(
    bank: _GlobalSizeBranchBank,
    delta: Array,
    lengths: Array,
    residual: Array,
    shape_gradient: Array,
    args: _GlobalSizeArgs,
) -> tuple[Array, Array, Array]:
    """Oriented component rows in the actual admitted parameter tangent space."""
    ranks = (jnp.sum(args.edge_valid > 0.0) - 1) * args.quantiles
    fraction = ranks - jnp.floor(ranks)
    active = jnp.arange(bank.statistic.size) < bank.count

    def row(statistic: Array, lower: Array, upper: Array, admitted: Array) -> Array:
        def evaluate(_: None) -> Array:
            def size_row(_: None) -> Array:
                gradient = jnp.zeros_like(args.origins)
                weight = fraction[statistic]
                for edge, coefficient in ((lower, 1.0 - weight), (upper, weight)):
                    vertices = args.edges[edge]
                    unit = coefficient * delta[edge] / lengths[edge]
                    for endpoint, sign in ((0, -1.0), (1, 1.0)):
                        vertex = vertices[endpoint]
                        value = jnp.sum(
                            args.basis[vertex] * (sign * unit)[:, None], axis=0
                        )
                        gradient = gradient.at[vertex].add(value)
                orientation = jnp.where(
                    residual[statistic] == 0.0, 1.0, -jnp.sign(residual[statistic])
                )
                return orientation * gradient

            return lax.cond(statistic < 0, lambda _: -shape_gradient, size_row, None)

        return lax.cond(admitted, evaluate, lambda _: jnp.zeros_like(args.origins), None)

    gradients = lax.map(
        lambda values: row(*values), (bank.statistic, bank.lower, bank.upper, active)
    )
    equality = (
        active & (bank.statistic >= 0) & (residual[jnp.maximum(bank.statistic, 0)] == 0.0)
    )
    return gradients, active & ~equality, equality


def _global_size_branch_targets(
    bank: _GlobalSizeBranchBank,
    residual: Array,
    inequality: Array,
    quantile_count: int,
) -> Array:
    """Raw progress targets owned by each actual signed scientific component."""
    size_targets = jnp.abs(residual[jnp.maximum(bank.statistic, 0)])
    shape_target = jnp.sum(residual[quantile_count:] ** 2)
    targets = jnp.where(bank.statistic < 0, shape_target, size_targets)
    return jnp.where(inequality, targets, 0.0)


def _global_size_bundle_direction(
    parameters: Array,
    direction: Array,
    residual: Array,
    shape_gradient: Array,
    remaining_work: Array,
    args: _GlobalSizeArgs,
) -> tuple[Array, Array, Array, Array, Array]:
    """Complete the generated current/full/outgoing rank frontier or refuse it."""
    capacity, maximum_refinements = 32, 8
    valid_edges = args.edge_valid > 0.0
    points = args.origins + args.target * jnp.sum(
        args.basis * parameters[:, None, :], axis=-1
    )
    delta = points[args.edges[:, 1]] - points[args.edges[:, 0]]
    lengths = jnp.sqrt(jnp.where(valid_edges, jnp.sum(delta * delta, axis=-1), 1.0))
    ranks = (jnp.sum(valid_edges) - 1) * args.quantiles
    lower_rank, upper_rank = (
        jnp.floor(ranks).astype(jnp.int32),
        jnp.ceil(ranks).astype(jnp.int32),
    )
    quantile_count, edge_count = args.quantiles.size, args.edges.shape[0]
    qids, edge_ids = (
        jnp.arange(quantile_count, dtype=jnp.int32),
        jnp.arange(edge_count, dtype=jnp.int32),
    )
    sort_work = 4 * edge_count * (edge_count - 1).bit_length() ** 2
    scan_work = (
        2 * sort_work
        + 2 * args.origins.shape[0]
        + 8 * edge_count
        + 8 * quantile_count * edge_count
    )
    empty = jnp.zeros((capacity,), dtype=jnp.int32)
    bank = _GlobalSizeBranchBank(empty, empty, empty, jnp.asarray(0), jnp.asarray(False))
    original_norm = jnp.sqrt(jnp.sum(direction * direction))
    tolerance = _GLOBAL_SIZE_METHOD.linear_policy.tolerance
    rank_policy = RankPolicy(
        relative_cutoff=tolerance.relative, absolute_cutoff=tolerance.absolute
    )

    def refine(carry: _GlobalSizeFrontierCarry) -> _GlobalSizeFrontierCarry:
        bank, candidate, admitted, finished, resource, work, visits = carry

        def scan(_: None) -> _GlobalSizeFrontierCarry:
            displacement = args.target * jnp.sum(
                args.basis * candidate[:, None, :], axis=-1
            )
            velocity = displacement[args.edges[:, 1]] - displacement[args.edges[:, 0]]
            tangent = jnp.sum(delta * velocity, axis=-1) / lengths
            acceleration = (
                jnp.sum(velocity * velocity, axis=-1) - tangent * tangent
            ) / lengths
            outgoing_order = jnp.lexsort(
                (acceleration, tangent, jnp.where(valid_edges, lengths, jnp.inf))
            )
            future_points = args.origins + args.target * jnp.sum(
                args.basis * (parameters + candidate)[:, None, :], axis=-1
            )
            future_delta = (
                future_points[args.edges[:, 1]] - future_points[args.edges[:, 0]]
            )
            future_lengths = jnp.sqrt(
                jnp.where(valid_edges, jnp.sum(future_delta * future_delta, axis=-1), 1.0)
            )
            future_order = jnp.argsort(jnp.where(valid_edges, future_lengths, jnp.inf))
            current_low, current_high = (
                outgoing_order[lower_rank],
                outgoing_order[upper_rank],
            )
            ties = (
                valid_edges[None, :]
                & (residual[:quantile_count, None] != 0.0)
                & (
                    (lengths[None, :] == lengths[current_low, None])
                    | (lengths[None, :] == lengths[current_high, None])
                )
            )
            statistics = jnp.concatenate(
                (
                    jnp.repeat(qids, edge_count),
                    qids,
                    qids,
                    jnp.asarray((-1,), dtype=jnp.int32),
                )
            )
            lows = jnp.concatenate(
                (
                    jnp.tile(edge_ids, quantile_count),
                    current_low,
                    future_order[lower_rank],
                    jnp.asarray((0,), dtype=jnp.int32),
                )
            )
            highs = jnp.concatenate(
                (
                    jnp.tile(edge_ids, quantile_count),
                    current_high,
                    future_order[upper_rank],
                    jnp.asarray((0,), dtype=jnp.int32),
                )
            )
            active = jnp.concatenate(
                (
                    ties.reshape(-1),
                    jnp.ones((2 * quantile_count,), dtype=jnp.bool_),
                    jnp.asarray((jnp.any(shape_gradient != 0.0),)),
                )
            )
            scanned_work = work + scan_work
            membership_bound = (
                capacity + statistics.size + jnp.sum(active) * (8 * capacity + 4)
            )

            def append(_: None) -> _GlobalSizeFrontierCarry:
                extended, append_work = _append_global_size_branches(
                    bank, statistics, lows, highs, active
                )
                spent = scanned_work + append_work
                stable = admitted & (extended.count == bank.count) & ~extended.overflow

                def complete(_: None) -> _GlobalSizeFrontierCarry:
                    return (
                        extended,
                        candidate,
                        admitted,
                        jnp.asarray(True),
                        resource,
                        spent,
                        visits + 1,
                    )

                def compose(_: None) -> _GlobalSizeFrontierCarry:
                    def bucket_step(bucket: int) -> _GlobalSizeFrontierCarry:
                        generation_work = 8 * bucket * args.origins.size

                        def generate(_: None) -> _GlobalSizeFrontierCarry:
                            selected_bank = _GlobalSizeBranchBank(
                                extended.statistic[:bucket],
                                extended.lower[:bucket],
                                extended.upper[:bucket],
                                extended.count,
                                extended.overflow,
                            )
                            gradients, inequality, equality = (
                                _global_size_branch_gradients(
                                    selected_bank,
                                    delta,
                                    lengths,
                                    residual,
                                    shape_gradient,
                                    args,
                                )
                            )
                            generated_work = spent + generation_work
                            projections = jnp.sum(
                                gradients * candidate[None, :, :], axis=(1, 2)
                            )
                            usable = (
                                jnp.any(inequality)
                                & jnp.all(~inequality | (projections > 0.0))
                                & jnp.all(
                                    ~equality
                                    | (jnp.abs(projections) <= tolerance.relative)
                                )
                                & jnp.all(jnp.isfinite(candidate))
                                & (original_norm > 0.0)
                            )

                            def keep(_: None) -> tuple[Array, Array, Array, Array]:
                                return (
                                    candidate,
                                    jnp.asarray(True),
                                    generated_work,
                                    jnp.asarray(False),
                                )

                            def change(_: None) -> tuple[Array, Array, Array, Array]:
                                target_work = 5 * bucket + 2 * (
                                    residual.size - quantile_count
                                )

                                def prepare_targets(
                                    _: None,
                                ) -> tuple[Array, Array, Array, Array]:
                                    # Select progress from the actual bank's scientific
                                    # component, not a unit margin that overmoves a
                                    # nearly satisfied quantile during frontier closure.
                                    targets = _global_size_branch_targets(
                                        selected_bank,
                                        residual,
                                        inequality,
                                        quantile_count,
                                    )
                                    prepared_work = generated_work + target_work
                                    result = compose_active_gradient_bank(
                                        gradients,
                                        targets,
                                        inequality,
                                        equality,
                                        original_norm,
                                        remaining_work - prepared_work,
                                        rank_policy=rank_policy,
                                        minimum_norm=tolerance.absolute,
                                        projection_tolerance=tolerance.relative,
                                    )
                                    return (
                                        result.direction,
                                        result.valid,
                                        prepared_work + result.work_units,
                                        result.resource_refused,
                                    )

                                return lax.cond(
                                    target_work <= remaining_work - generated_work,
                                    prepare_targets,
                                    lambda _: (
                                        candidate,
                                        jnp.asarray(False),
                                        generated_work,
                                        jnp.asarray(True),
                                    ),
                                    None,
                                )

                            chosen, accepted, total, denied = lax.cond(
                                usable, keep, change, None
                            )
                            return (
                                extended,
                                chosen,
                                accepted,
                                ~accepted,
                                denied,
                                total,
                                visits + 1,
                            )

                        return lax.cond(
                            generation_work <= remaining_work - spent,
                            generate,
                            lambda _: (
                                extended,
                                candidate,
                                jnp.asarray(False),
                                jnp.asarray(True),
                                jnp.asarray(True),
                                spent,
                                visits + 1,
                            ),
                            None,
                        )

                    def choose16(_: None) -> _GlobalSizeFrontierCarry:
                        return lax.cond(
                            extended.count <= 16,
                            lambda _: bucket_step(16),
                            lambda _: bucket_step(32),
                            None,
                        )

                    def choose8(_: None) -> _GlobalSizeFrontierCarry:
                        return lax.cond(
                            extended.count <= 8, lambda _: bucket_step(8), choose16, None
                        )

                    return lax.cond(
                        extended.count <= 4, lambda _: bucket_step(4), choose8, None
                    )

                return lax.cond(
                    extended.overflow,
                    lambda _: (
                        extended,
                        candidate,
                        jnp.asarray(False),
                        jnp.asarray(True),
                        resource,
                        spent,
                        visits + 1,
                    ),
                    lambda _: lax.cond(stable, complete, compose, None),
                    None,
                )

            return lax.cond(
                membership_bound <= remaining_work - scanned_work,
                append,
                lambda _: (
                    bank,
                    candidate,
                    jnp.asarray(False),
                    jnp.asarray(True),
                    jnp.asarray(True),
                    scanned_work,
                    visits + 1,
                ),
                None,
            )

        return lax.cond(
            scan_work <= remaining_work - work,
            scan,
            lambda _: (
                bank,
                candidate,
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(True),
                work,
                visits,
            ),
            None,
        )

    bank, selected, admitted, finished, resource, work, visits = lax.while_loop(
        lambda carry: (~carry[3]) & (~carry[4]) & (carry[6] < maximum_refinements),
        refine,
        (
            bank,
            direction,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return selected, admitted & finished & ~bank.overflow, work, visits, resource


class _GlobalSizeTrialPolicy(AbstractLeastSquaresTrialPolicy):
    def termination_valid(self, residual: Array, args: _GlobalSizeArgs) -> Array:
        # Zero selected generalized gradient is not a stationarity proof.
        # Source publication/admission remains with the scientific owner.
        return jnp.all(residual == 0.0)

    @eqx.filter_jit
    def __call__(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: JacobianLinearOperator,
        remaining_model_work: Array,
        args: _GlobalSizeArgs,
    ) -> LeastSquaresTrialResult:
        fixed_work, _, _ = _global_size_trial_work(
            args.origins.shape[0],
            args.edges.shape[0],
            args.cells.shape[0],
            args.quantiles.size,
        )
        return lax.cond(
            remaining_model_work >= fixed_work,
            lambda _: self._evaluate(
                parameters, direction, residual, jacobian, remaining_model_work, args
            ),
            lambda _: LeastSquaresTrialResult(
                direction=direction,
                scale=jnp.asarray(1.0, dtype=residual.dtype),
                model_image=jnp.zeros_like(residual),
                valid=jnp.asarray(False),
                jvp_actions=jnp.asarray(0, dtype=jnp.int32),
                vjp_actions=jnp.asarray(0, dtype=jnp.int32),
                model_work_units=jnp.asarray(0, dtype=jnp.int64),
                model_visits=jnp.asarray(0, dtype=jnp.int32),
                resource_refused=jnp.asarray(True),
            ),
            None,
        )

    def _evaluate(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: JacobianLinearOperator,
        remaining_model_work: Array,
        args: _GlobalSizeArgs,
    ) -> LeastSquaresTrialResult:
        fixed_work, model_work, event_work = _global_size_trial_work(
            args.origins.shape[0],
            args.edges.shape[0],
            args.cells.shape[0],
            args.quantiles.size,
        )
        shape_seed = residual.at[: args.quantiles.size].set(0.0)
        shape_gradient = jacobian.adjoint_mv(shape_seed)
        direction, bundle_valid, bundle_work, bundle_visits, bundle_resource = (
            _global_size_bundle_direction(
                parameters,
                direction,
                residual,
                shape_gradient,
                remaining_model_work - fixed_work,
                args,
            )
        )
        image = jacobian.mv(direction)
        valid = args.edge_valid > 0.0
        points = args.origins + args.target * jnp.sum(
            args.basis * parameters[:, None, :], axis=-1
        )
        displacement = args.target * jnp.sum(args.basis * direction[:, None, :], axis=-1)
        delta = points[args.edges[:, 1]] - points[args.edges[:, 0]]
        velocity = displacement[args.edges[:, 1]] - displacement[args.edges[:, 0]]
        constant = jnp.sum(delta * delta, axis=-1)
        linear = 2.0 * jnp.sum(delta * velocity, axis=-1)
        quadratic = jnp.sum(velocity * velocity, axis=-1)
        lengths = jnp.sqrt(jnp.where(valid, constant, 1.0))
        safe_length = jnp.where(constant > 0.0, lengths, 1.0)
        tangent = jnp.where(
            constant > 0.0, 0.5 * linear / safe_length, jnp.sqrt(quadratic)
        )
        acceleration = quadratic / safe_length - tangent * tangent / safe_length
        # Exact represented length ties are ordered by outgoing first and
        # second derivatives, never by SCI row identity or epsilon grouping.
        order = jnp.lexsort((acceleration, tangent, jnp.where(valid, lengths, jnp.inf)))
        ranks = (jnp.sum(valid) - 1) * args.quantiles
        lower = jnp.floor(ranks).astype(jnp.int32)
        upper = jnp.ceil(ranks).astype(jnp.int32)
        fraction = ranks - lower
        lower_rows, upper_rows = order[lower], order[upper]
        statistic = (
            lengths[lower_rows] * (1.0 - fraction) + lengths[upper_rows] * fraction
        )
        derivative = (
            tangent[lower_rows] * (1.0 - fraction) + tangent[upper_rows] * fraction
        ) / args.target
        relative = (statistic - args.target) / args.target
        size_image = jnp.where(
            args.tolerance == 0.0,
            derivative,
            jnp.where(
                (relative > args.tolerance) | (relative < -args.tolerance),
                derivative,
                jnp.where(
                    relative == args.tolerance,
                    jnp.maximum(derivative, 0.0),
                    jnp.where(
                        relative == -args.tolerance, jnp.minimum(derivative, 0.0), 0.0
                    ),
                ),
            ),
        )
        # At a zero max hinge JAX's linear selection is half its raw
        # derivative; recover the exact one-sided image only for this trial.
        shape_image = jnp.where(
            residual[args.quantiles.size :] == 0.0,
            jnp.maximum(2.0 * image[args.quantiles.size :], 0.0),
            image[args.quantiles.size :],
        )
        # Radius/shortest-edge also contains a nonsmooth rank. Differentiate
        # the native circumcenter equation and select the true outgoing
        # shortest-edge tangent, rather than reusing its averaged linear J.
        corners = points[args.cells] / args.target
        corner_velocity = displacement[args.cells] / args.target
        frame = corners[:, 1:] - corners[:, :1]
        frame_velocity = corner_velocity[:, 1:] - corner_velocity[:, :1]
        center = solve_small_linear(
            SmallLinearSolvePlan(3), frame, 0.5 * jnp.sum(frame * frame, axis=-1)
        ).value
        center_velocity = solve_small_linear(
            SmallLinearSolvePlan(3),
            frame,
            jnp.sum(frame * frame_velocity, axis=-1)
            - jnp.sum(frame_velocity * center[:, None, :], axis=-1),
        ).value
        radius = jnp.sqrt(jnp.sum(center * center, axis=-1))
        radius_velocity = jnp.sum(center * center_velocity, axis=-1) / radius
        pairs = jnp.asarray(_TET_EDGES)
        cell_delta = corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]]
        cell_velocity = corner_velocity[:, pairs[:, 1]] - corner_velocity[:, pairs[:, 0]]
        cell_lengths = jnp.sqrt(jnp.sum(cell_delta * cell_delta, axis=-1))
        cell_tangents = jnp.sum(cell_delta * cell_velocity, axis=-1) / cell_lengths
        shortest = jnp.min(cell_lengths, axis=-1)
        shortest_velocity = jnp.min(
            jnp.where(cell_lengths == shortest[:, None], cell_tangents, jnp.inf), axis=-1
        )
        radius_relative = radius / shortest - args.radius_edge_bound
        radius_derivative = (
            radius_velocity / shortest - radius * shortest_velocity / shortest**2
        )
        radius_image = (
            16.0
            * args.cell_valid
            * jnp.where(
                radius_relative > 0.0,
                radius_derivative,
                jnp.where(
                    radius_relative == 0.0, jnp.maximum(radius_derivative, 0.0), 0.0
                ),
            )
        )
        cell_count = args.cells.shape[0]
        shape_image = shape_image.at[cell_count : 2 * cell_count].set(radius_image)
        trial_image = jnp.concatenate((size_image, shape_image))

        # Finite source-point sizes reselect every rank, including
        # second-order movement of initially stationary edges. This is
        # a size/shape model, not an actual residual callback or Krylov J.
        def finite_model(scale: Array) -> tuple[Array, Array]:
            # Actual finite physical sizes and cell-shape constraints of the
            # source-point trajectory; this is not a residual callback.
            modeled_points = args.origins + args.target * jnp.sum(
                args.basis * (parameters + scale * direction)[:, None, :], axis=-1
            )
            modeled_delta = (
                modeled_points[args.edges[:, 1]] - modeled_points[args.edges[:, 0]]
            )
            modeled_lengths = jnp.sqrt(
                jnp.where(valid, jnp.sum(modeled_delta * modeled_delta, axis=-1), 1.0)
            )
            modeled_statistic = _size_quantile(
                jnp.where(valid, modeled_lengths, jnp.nan), args.quantiles
            )
            modeled_relative = (modeled_statistic - args.target) / args.target
            modeled_size = jnp.where(
                args.tolerance == 0.0,
                modeled_relative,
                modeled_relative
                - jnp.clip(modeled_relative, -args.tolerance, args.tolerance),
            )
            modeled_shape = _global_size_shape_residual(modeled_points, args)
            modeled = jnp.concatenate((modeled_size, modeled_shape))
            change = modeled - residual
            prediction = -jnp.sum(residual * change) - 0.5 * jnp.sum(change * change)
            admissible = (
                jnp.all(jnp.isfinite(modeled))
                & jnp.all((~valid) | (modeled_lengths > 0.0))
                & jnp.all(
                    jnp.abs(modeled_size) <= jnp.abs(residual[: args.quantiles.size])
                )
                & jnp.all(
                    modeled_shape[cell_count:]
                    <= residual[args.quantiles.size + cell_count :]
                )
                & jnp.isfinite(prediction)
                & (prediction > 0.0)
            )
            return change / scale, admissible

        merit_slope = jnp.sum(residual * trial_image)

        stage_budget = remaining_model_work - fixed_work - bundle_work

        def refused(visits: int) -> tuple[Array, Array, Array, Array, Array, Array]:
            return (
                jnp.asarray(1.0, dtype=residual.dtype),
                trial_image,
                jnp.asarray(False),
                jnp.asarray(visits, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
            )

        def select_trial(_: None) -> tuple[Array, Array, Array, Array, Array, Array]:
            def full_trial(_: None) -> tuple[Array, Array, Array, Array, Array, Array]:
                full_image, full_valid = finite_model(
                    jnp.asarray(1.0, dtype=residual.dtype)
                )

                def stationary_trial(
                    _: None,
                ) -> tuple[Array, Array, Array, Array, Array, Array]:
                    curvature = jnp.sum(trial_image * trial_image)
                    stationary_scale = jnp.minimum(
                        1.0, -merit_slope / jnp.where(curvature > 0.0, curvature, 1.0)
                    )
                    # The full finite model has already refused scale one.
                    # Repeating that point cannot recover hard-source feasibility.
                    stationary_scale = jnp.where(
                        stationary_scale == 1.0, 0.5, stationary_scale
                    )
                    image, admitted = finite_model(stationary_scale)
                    admitted = (
                        admitted
                        & jnp.isfinite(stationary_scale)
                        & (stationary_scale > 0.0)
                    )

                    def event_trial(
                        _: None,
                    ) -> tuple[Array, Array, Array, Array, Array, Array]:
                        support = jnp.concatenate((lower_rows, upper_rows))
                        alpha, accepted = quadratic_event_bound(
                            quadratic[None, :] - quadratic[support, None],
                            linear[None, :] - linear[support, None],
                            constant[None, :] - constant[support, None],
                            jnp.broadcast_to(valid[None, :], (support.size, valid.size)),
                        )
                        image, finite_valid = finite_model(alpha)
                        return (
                            alpha,
                            image,
                            accepted & finite_valid,
                            jnp.asarray(3, dtype=jnp.int32),
                            jnp.asarray(True),
                            jnp.asarray(False),
                        )

                    return lax.cond(
                        admitted,
                        lambda _: (
                            stationary_scale,
                            image,
                            admitted,
                            jnp.asarray(2, dtype=jnp.int32),
                            jnp.asarray(False),
                            jnp.asarray(False),
                        ),
                        lambda _: lax.cond(
                            stage_budget >= 3 * model_work + event_work,
                            event_trial,
                            lambda _: refused(2),
                            None,
                        ),
                        None,
                    )

                return lax.cond(
                    full_valid,
                    lambda _: (
                        jnp.asarray(1.0, dtype=residual.dtype),
                        full_image,
                        full_valid,
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(False),
                        jnp.asarray(False),
                    ),
                    lambda _: lax.cond(
                        stage_budget >= 2 * model_work,
                        stationary_trial,
                        lambda _: refused(1),
                        None,
                    ),
                    None,
                )

            return lax.cond(
                stage_budget >= model_work, full_trial, lambda _: refused(0), None
            )

        # An already-refused outgoing branch cannot benefit from model
        # sorting or event allocation. Charge only the executed branch.
        (
            scale,
            selected_trial_image,
            admissible,
            model_visits,
            event_executed,
            trial_resource,
        ) = lax.cond(
            bundle_valid
            & ~bundle_resource
            & jnp.isfinite(merit_slope)
            & (merit_slope < 0.0),
            select_trial,
            lambda _: (
                jnp.asarray(1.0, dtype=residual.dtype),
                trial_image,
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(False),
            ),
            None,
        )
        proposed_points = args.origins + args.target * jnp.sum(
            args.basis * (parameters + scale * direction)[:, None, :], axis=-1
        )
        represented_motion = jnp.any(proposed_points != points)
        # Exact outgoing slope admission remains independent of finite prediction.
        actual_work = (
            fixed_work
            + bundle_work
            + model_visits * model_work
            + event_executed.astype(jnp.int64) * event_work
        )
        return LeastSquaresTrialResult(
            direction=direction,
            scale=scale,
            model_image=selected_trial_image,
            valid=(
                bundle_valid
                & admissible
                & represented_motion
                & ~bundle_resource
                & ~trial_resource
                & jnp.isfinite(merit_slope)
                & (merit_slope < 0.0)
            ),
            jvp_actions=jnp.asarray(1, dtype=jnp.int32),
            vjp_actions=jnp.asarray(1, dtype=jnp.int32),
            model_work_units=actual_work,
            model_visits=bundle_visits + model_visits,
            resource_refused=bundle_resource | trial_resource,
        )


def _global_size_shape_residual(points: Array, args: _GlobalSizeArgs, /) -> Array:
    """One shared physical cell algebra for native primals and finite models."""
    corners = points[args.cells] / args.target
    frame = corners[:, 1:] - corners[:, :1]
    determinant = jnp.sum(frame[:, 0] * jnp.cross(frame[:, 1], frame[:, 2]), axis=-1)
    volume = jnp.maximum(0.1 * args.initial_volume - determinant, 0.0)
    faces = corners[:, jnp.asarray(_FACES)]
    normals = jnp.cross(faces[:, :, 1] - faces[:, :, 0], faces[:, :, 2] - faces[:, :, 0])
    squared = jnp.sum(normals * normals, axis=-1)
    pairs = jnp.asarray(_TET_EDGES)
    cosine = -jnp.sum(
        normals[:, pairs[:, 0]] * normals[:, pairs[:, 1]], axis=-1
    ) / jnp.sqrt(jnp.maximum(squared[:, pairs[:, 0]] * squared[:, pairs[:, 1]], 1.0e-30))
    angles = jnp.maximum(cosine - args.dihedral_cosine, 0.0)
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        frame,
        0.5 * jnp.sum(frame * frame, axis=-1),
    )
    cell_edges = corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]]
    shortest = jnp.sqrt(jnp.min(jnp.sum(cell_edges * cell_edges, axis=-1), axis=-1))
    radius = jnp.sqrt(jnp.sum(solve.value * solve.value, axis=-1))
    radius_excess = jnp.maximum(radius / shortest - args.radius_edge_bound, 0.0)
    return jnp.concatenate(
        (
            16.0 * args.cell_valid * volume,
            16.0 * args.cell_valid * radius_excess,
            (16.0 * args.cell_valid[:, None] * angles).reshape((-1,)),
        )
    )


def _global_size_residual(parameters: Array, args: _GlobalSizeArgs, /) -> Array:
    """Owning target quantiles and shape hinges on participating source strata.

    NumPy publication and this native proposal both use linear quantiles.
    Actual publication statistics and shape checks remain with their owners.
    Jacobian actions are differentiated edge gathers and cell-local algebra;
    no dense global Jacobian or normal matrix is formed.
    """
    points = args.origins + args.target * jnp.sum(
        args.basis * parameters[:, None, :], axis=-1
    )
    difference = points[args.edges[:, 1]] - points[args.edges[:, 0]]
    valid_edges = args.edge_valid > 0.0
    squared_lengths = jnp.sum(difference * difference, axis=-1)
    # Exclude bucket rows from the statistic itself, not merely its residual.
    # Mask before sqrt so reverse actions cannot encounter padded sqrt(0).
    lengths = jnp.where(
        valid_edges,
        jnp.sqrt(jnp.where(valid_edges, squared_lengths, 1.0)),
        jnp.nan,
    )
    # The primal keeps the owning linear-quantile convention. A prepared
    # exact-tie block has a linear, row-permutation-symmetric derivative;
    # outgoing trial derivatives are a separate nonlinear rank operation.
    statistic = _size_quantile(lengths, args.quantiles)
    relative = (statistic - args.target) / args.target
    size = jnp.where(
        args.tolerance == 0.0,
        relative,
        relative - jnp.clip(relative, -args.tolerance, args.tolerance),
    )
    return jnp.concatenate((size, _global_size_shape_residual(points, args)))


@eqx.filter_jit
def _global_size_solve(
    args: _GlobalSizeArgs, model_work_limit: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    problem = NonlinearLeastSquaresProblem(
        _global_size_residual,
        problem_id="source-stratum/global-size-linear-quantile",
        trial_policy=_GlobalSizeTrialPolicy(),
        trial_policy_id=_GLOBAL_SIZE_TRIAL_POLICY_ID,
        trial_model_work_limit=model_work_limit,
    )
    result = least_squares(
        problem,
        jnp.zeros_like(args.origins),
        method=_GLOBAL_SIZE_METHOD,
        termination=_GLOBAL_SIZE_TERMINATION,
        args=args,
    )
    diagnostics = result.diagnostics
    return (
        jnp.asarray(result.parameters),
        jnp.asarray(result.status),
        jnp.asarray(diagnostics.iterations),
        jnp.asarray(diagnostics.residual_evaluations),
        jnp.asarray(
            diagnostics.residual_evaluations
            + diagnostics.jvp_evaluations
            + diagnostics.vjp_evaluations
        ),
        jnp.asarray(result.method_evidence["trial_policy_evaluations"]),
        jnp.asarray(result.method_evidence["trial_policy_model_work_units"]),
        jnp.asarray(result.method_evidence["trial_policy_model_visits"]),
        jnp.asarray(result.method_evidence["trial_policy_receipts_valid"]),
        jnp.asarray(result.method_evidence["trial_policy_resource_refused"]),
    )


class _GlobalSizePlan(NamedTuple):
    args: _GlobalSizeArgs
    basis: np.ndarray
    vertices: np.ndarray
    edges: np.ndarray
    scratch_bytes: int
    preparation_work: int
    action_work: int
    sort_work: int
    group_work: int
    trial_work: int
    curve_vertices: np.ndarray
    curve_neighbors: np.ndarray


@contextmanager
def _measure_statistical_phase(
    budget: _NativeMetricBudget, phase: NativeMeshingPhase, /
) -> Iterator[None]:
    """Record actual charged work, including a refused phase, outside identity."""
    budget.check_deadline()
    started = phase_started(budget.record_phase)
    before = budget.work_units if budget.record_phase is not None else 0
    try:
        yield
    finally:
        work = budget.work_units - before if budget.record_phase is not None else None
        record_elapsed(budget.record_phase, phase, started, work_units=work)
    budget.check_deadline()


def _prepare_global_size(
    snapshot: TetMeshArrays,
    fixed: np.ndarray,
    objective: _StatisticalObjective,
    operation_budget: int,
    /,
) -> _GlobalSizePlan | None:
    """Reserve linear preparation and bind every movable source-stratum row."""
    budget = objective.budget
    vertex_count, cell_count = snapshot.points.shape[0], snapshot.tetrahedra.shape[0]
    edge_count = objective.edges.keys.size
    # Linear tables and Krylov vectors only: no dense global rigidity matrix.
    vertex_capacity = 1 << max(0, (vertex_count - 1).bit_length())
    edge_capacity = 1 << max(0, (edge_count - 1).bit_length())
    cell_capacity = 1 << max(0, (cell_count - 1).bit_length())
    # Include retained sort support, gather/scatter and derivative working rows.
    # Outgoing-event workspace retains three Q×E coefficient/root banks
    # and validity masks, as well as three-key sorting and tied groups.
    quantile_count = len(objective.goal.policy.target_statistics)
    scratch = 8 * (
        96 * vertex_capacity
        + 72 * edge_capacity
        + 240 * cell_capacity
        + 40 * quantile_count * edge_capacity
    ) + _global_size_branch_scratch(vertex_capacity)
    sort_levels = (edge_capacity - 1).bit_length()
    sort_work = 4 * edge_capacity * sort_levels * sort_levels
    preparation_work = (
        vertex_count
        + 6 * cell_count
        + edge_count
        + 3 * snapshot.faces.shape[0]
        + 2 * snapshot.segments.shape[0]
    )
    if operation_budget < 1 or not budget.optimizer_work(preparation_work, scratch):
        return None
    active = np.flatnonzero(snapshot.vertex_dimension > 0)
    frozen = np.zeros((vertex_count,), dtype=np.bool_)
    frozen[: fixed.size] = fixed
    active = active[~frozen[active]]
    if active.size == 0:
        return None
    basis = np.zeros((vertex_capacity, 3, 3), dtype=np.float64)
    # Immutable incidence preparation, not a numerical Python iteration.
    incidence: list[list[int]] = [[] for _ in range(vertex_count)]
    face_incidence: list[list[int]] = [[] for _ in range(vertex_count)]
    segment_incidence: list[list[int]] = [[] for _ in range(vertex_count)]
    for row, cell in enumerate(snapshot.tetrahedra.tolist()):
        for vertex in cell:
            incidence[vertex].append(row)
    for row, face in enumerate(snapshot.faces.tolist()):
        for vertex in face:
            face_incidence[vertex].append(row)
    for row, segment in enumerate(snapshot.segments.tolist()):
        for vertex in segment:
            segment_incidence[vertex].append(row)
    movable: list[int] = []
    curve_vertices: list[int] = []
    curve_neighbors: list[tuple[int, int]] = []
    for vertex in active.tolist():
        budget.check_deadline()
        cavity = snapshot.tetrahedra[np.asarray(incidence[vertex], dtype=np.int64)]
        coordinates = _tangent_coordinates(
            snapshot,
            vertex,
            cavity,
            facet_rows=np.asarray(face_incidence[vertex], dtype=np.int64),
            segment_rows=np.asarray(segment_incidence[vertex], dtype=np.int64),
        )
        if coordinates is None:
            budget.refuse_optimizer("unsupported_source_tangent")
            continue
        basis[vertex, :, : coordinates.basis.shape[1]] = coordinates.basis
        movable.append(vertex)
        if snapshot.vertex_dimension[vertex] == 1:
            segments = snapshot.segments[
                np.asarray(segment_incidence[vertex], dtype=np.int64)
            ]
            neighbors = np.unique(segments[segments != vertex])
            curve_vertices.append(vertex)
            curve_neighbors.append((int(neighbors[0]), int(neighbors[1])))
    if not movable:
        return None
    target = objective.goal.control.target_size
    ranks = np.arange(edge_capacity)
    quantiles = [
        {"p50": 0.5, "p95": 0.95}[name]
        for name in objective.goal.policy.target_statistics
    ]
    # Only the requested statistics enter the residual. Requiring every edge
    # between their ranks to equal the target adds unrequested constraints.
    edge_valid = (ranks < edge_count).astype(np.float64)
    cell_valid = (np.arange(cell_capacity) < cell_count).astype(np.float64)
    origins = _bucket(snapshot.points)
    cells = _bucket(snapshot.tetrahedra)
    corners = origins[cells]
    initial_volume = np.linalg.det(corners[:, 1:] - corners[:, :1]) / target**3
    edges = _edge_rows(objective.edges.keys)
    args = _GlobalSizeArgs(
        jnp.asarray(origins),
        jnp.asarray(basis),
        jnp.asarray(_bucket(edges)),
        jnp.asarray(cells),
        jnp.asarray(quantiles),
        jnp.asarray(edge_valid),
        jnp.asarray(cell_valid),
        jnp.asarray(initial_volume),
        jnp.asarray(target),
        jnp.asarray(objective.goal.policy.tolerance(target) / target),
        jnp.asarray(np.cos(np.deg2rad(objective.goal.minimum_dihedral_degrees or 0.0))),
        jnp.asarray(objective.goal.maximum_radius_edge or np.inf),
    )
    # Match the epoch's primitive/row-visit units, not floating-point FLOPs:
    # one stratum action and physical edge query; per cell six edge lengths,
    # four normals, six angle pairs, determinant, small solve and radius norm.
    # Padding executes real row work and is charged as well.
    # Every action pays edge processing and retained-support gather/scatter.
    # STORE linearization sorts only during primal residual/model evaluation;
    # its JVP and transposed VJP reuse the prepared sort permutation.
    action_work = vertex_capacity + 8 * edge_capacity + 20 * cell_capacity
    group_work = 4 * edge_capacity
    fixed_trial_work, model_trial_work, event_trial_work = _global_size_trial_work(
        vertex_capacity, edge_capacity, cell_capacity, quantile_count
    )
    trial_work = fixed_trial_work + 3 * model_trial_work + event_trial_work
    return _GlobalSizePlan(
        args,
        basis,
        np.asarray(movable, dtype=np.int64),
        edges,
        scratch,
        preparation_work,
        action_work,
        sort_work,
        group_work,
        trial_work,
        np.asarray(curve_vertices, dtype=np.int32),
        np.asarray(curve_neighbors, dtype=np.int32).reshape((-1, 2)),
    )


def _statistical_relocate_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    fixed: np.ndarray,
    operation_budget: int,
    source_metric: np.ndarray,
    ancestry: _Ancestry,
    objective: _StatisticalObjective,
    /,
) -> tuple[TetMeshArrays, int, int]:
    """One matrix-free coordinated solve, admitted as one native transaction."""
    budget = objective.budget
    with _measure_statistical_phase(budget, "metric_global_preparation"):
        plan = _prepare_global_size(snapshot, fixed, objective, operation_budget)
    if plan is None:
        return snapshot, 0, 0
    # Reserve the full bounded residual/JVP/VJP envelope before device dispatch.
    work_per_action, scratch = plan.action_work, plan.scratch_bytes
    # Each LSMR recurrence has one JVP and one VJP; twelve more
    # actions cover initialization, true-residual checks and trial/model work.
    action_bound = 4 + _GLOBAL_SIZE_TERMINATION.maximum_steps * (
        2 + _GLOBAL_SIZE_METHOD.maximum_trials * (2 * _GLOBAL_LINEAR_STEPS + 12)
    )
    # Native LM trial primals are distinct from the source policy's finite
    # models, whose sorts already consume its actual model-work receipt.
    residual_bound = _global_size_canonical_residual_bound()
    policy_bound = (
        _GLOBAL_SIZE_TERMINATION.maximum_steps * _GLOBAL_SIZE_METHOD.maximum_trials
    )
    action_bound += policy_bound * 2  # One actual prepared JVP and VJP per invocation.
    model_bound = _GLOBAL_SIZE_TERMINATION.maximum_steps + 2
    fixed_bound = (
        action_bound * work_per_action
        + residual_bound * plan.sort_work
        + model_bound * plan.group_work
    )
    remaining_work = budget.maximum_work_units - budget.work_units
    if budget.execution_budget is not None:
        remaining_work = min(
            remaining_work, budget.execution_budget.remaining().remaining_work_units
        )
    if fixed_bound > remaining_work:
        budget.optimizer_work(fixed_bound, scratch)
        return snapshot, 0, 0
    model_work_limit = remaining_work - fixed_bound
    budget.optimizer_terminations.append(
        (
            _GLOBAL_SIZE_TERMINATION.maximum_steps,
            int(_GLOBAL_SIZE_TERMINATION.maximum_evaluations or 0),
        )
    )
    budget.optimizer_method_id = (
        f"{_GLOBAL_SIZE_METHOD.method_id}/{_GLOBAL_SIZE_TRIAL_POLICY_ID}"
    )
    budget.optimizer_tolerances = (
        _GLOBAL_SIZE_TERMINATION.absolute_optimality,
        _GLOBAL_SIZE_TERMINATION.relative_optimality,
        _GLOBAL_SIZE_TERMINATION.absolute_step,
        _GLOBAL_SIZE_TERMINATION.relative_step,
    )
    if budget.execution_budget is not None:
        budget.execution_budget.admit_work_bound(fixed_bound + model_work_limit)
    with _measure_statistical_phase(budget, "metric_global_solve"):
        (
            parameters,
            status,
            iterations,
            residuals,
            actions,
            _,
            policy_work,
            _,
            receipts_valid,
            resource_refused,
        ) = _global_size_solve(plan.args, jnp.asarray(model_work_limit, dtype=jnp.int64))
        if not bool(receipts_valid):
            raise ValueError(
                "The native statistical trial-policy work receipts failed validation."
            )
        budget.optimizer_status_counts[int(status)] += 1
        budget.optimizer_iterations += int(iterations)
        budget.optimizer_objective_evaluations += int(residuals)
        # residual_evaluations includes prepared models and trial/final primals;
        # actions additionally includes every recorded JVP and VJP invocation.
        charged = budget.optimizer_work(
            int(actions) * work_per_action
            + int(residuals) * plan.sort_work
            + (int(iterations) + 2) * plan.group_work
            + int(policy_work),
            scratch,
        )
        if bool(resource_refused) and charged:
            budget.refuse(
                "The remaining original work allowance cannot admit the complete statistical rank model.",
                (("maximum_work_units", float(budget.maximum_work_units)),),
                (
                    ("work_units", float(budget.work_units)),
                    ("model_work_limit", float(model_work_limit)),
                    ("actual_model_work_units", float(policy_work)),
                ),
            )
            return snapshot, 0, 1
    if not charged:
        return snapshot, 0, 1
    displacement = np.asarray(parameters, dtype=np.float64)
    displacement = objective.goal.control.target_size * np.sum(
        plan.basis * displacement[:, None, :], axis=-1
    )
    if not np.all(np.isfinite(displacement)):
        budget.refuse_optimizer("nonfinite_global_size_solution")
        return snapshot, 0, 1
    with _measure_statistical_phase(budget, "metric_global_admission"):
        return _admit_global_size(
            native,
            snapshot,
            values,
            source_metric,
            ancestry,
            objective,
            plan,
            displacement,
            operation_budget,
        )


def _admit_global_size(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    source_metric: np.ndarray,
    ancestry: _Ancestry,
    objective: _StatisticalObjective,
    plan: _GlobalSizePlan,
    displacement: np.ndarray,
    operation_budget: int,
    /,
) -> tuple[TetMeshArrays, int, int]:
    """Check actual global merit/coverage and commit exactly once through native."""
    from .providers._native_publication import edge_size_evidence

    budget = objective.budget
    rows, scratch, preparation_work = (
        plan.vertices,
        plan.scratch_bytes,
        plan.preparation_work,
    )
    before = budget.optimizer_evaluations
    for scale in _GLOBAL_TRIAL_SCALES:
        if budget.exhausted or budget.optimizer_evaluations - before >= operation_budget:
            break
        if not budget.optimizer_work(preparation_work, scratch):
            break
        points = snapshot.points.copy()
        points[rows] += scale * displacement[rows]
        curve_mask = np.any(
            points[plan.curve_vertices] != snapshot.points[plan.curve_vertices],
            axis=1,
        )
        if np.any(curve_mask):
            curve_rows = plan.curve_vertices[curve_mask]
            curve_neighbors = plan.curve_neighbors[curve_mask]
            construction = budget.call(
                lambda allowance: native.construct_curve_points(
                    curve_rows,
                    curve_neighbors,
                    points[curve_rows],
                    work_limit=allowance,
                )
            )
            if construction is None or not bool(np.all(construction[2])):
                budget.refuse_optimizer("native_curve_stratum_construction")
                continue
            points[curve_rows] = construction[0]
        changed = rows[np.any(points[rows] != snapshot.points[rows], axis=1)]
        if not changed.size:
            budget.refuse_optimizer("unchanged_global_size_proposal")
            continue
        cavity_count = np.count_nonzero(
            np.any(np.isin(snapshot.tetrahedra, changed), axis=1)
        )
        if cavity_count > budget.maximum_cavity_cells:
            budget.refuse(
                "The coordinated statistical relocation exceeds its cavity-union allowance.",
                (("maximum_cavity_cells", float(budget.maximum_cavity_cells)),),
                (("cavity_union_cells", float(cavity_count)),),
            )
            return snapshot, 0, 1 + budget.optimizer_evaluations - before
        try:
            lengths, growth = edge_size_evidence(points, plan.edges)
        except MeshingFailure as error:
            if error.evidence.category is not MeshingFailureCategory.COMPLIANCE_FAILED:
                raise
            budget.refuse_optimizer("nonfinite_global_physical_edges")
            continue
        ledger = _PhysicalEdgeLedger(
            objective.edges.keys,
            objective.edges.counts,
            lengths,
            points.shape[0],
            _physical_size_merit(lengths, growth, objective.goal),
        )
        old_merit, new_merit = objective.edges.merit, ledger.merit
        if new_merit.normalized_deficit > old_merit.normalized_deficit or (
            new_merit.normalized_deficit == old_merit.normalized_deficit
            and new_merit.directional_edge_penalty >= old_merit.directional_edge_penalty
        ):
            budget.refuse_optimizer("nonimproving_global_size_proposal")
            continue
        stencil = budget.stencil(points[changed])
        if stencil is None or not bool(np.all(stencil[2])):
            budget.refuse_optimizer("uncovered_global_size_proposal")
            continue
        source_rows = key_rows(
            budget.source_tables.vertex_global_ids[:, None],
            stencil[0].reshape((-1, 1)),
        ).reshape(stencil[0].shape)
        sampled = _interpolate(source_metric[source_rows], stencil[1])
        trial_metric = values.copy()
        trial_metric[changed] = sampled
        quality = _quality(points, trial_metric, snapshot.tetrahedra)
        if np.any(quality < objective.minimum_quality):
            budget.refuse_optimizer("global_metric_quality")
            continue
        floor = objective.goal.minimum_relative_determinant
        if floor > 0.0:
            touched = snapshot.tetrahedra[
                np.any(np.isin(snapshot.tetrahedra, changed), axis=1)
            ]
            vertices = np.unique(touched)
            shape = budget.call(
                lambda allowance: native.proposal_shape(
                    points[vertices],
                    np.searchsorted(vertices, touched).astype(np.int32),
                    vertex_ids=vertices,
                    minimum_relative_determinant=floor,
                    work_limit=allowance,
                )
            )
            if shape is None:
                budget.refuse_optimizer("native_global_size_transaction")
                continue
            if shape[2]:
                budget.refuse_optimizer("relative_determinant_floor")
                continue
        if not budget.call(
            lambda allowance: native.relocate_vertices(
                changed,
                points[changed],
                radius_edge_bound=objective.goal.maximum_radius_edge or 0.0,
                minimum_dihedral_degrees=objective.goal.minimum_dihedral_degrees or 0.0,
                work_limit=allowance,
            )
        ):
            budget.refuse_optimizer("native_global_size_transaction")
            continue
        values[changed] = sampled
        for vertex, sources, weights in zip(
            changed.tolist(), stencil[0], stencil[1], strict=True
        ):
            ancestry.vertex_sources[vertex] = set(sources[weights > 0.0].tolist())
        if ancestry.relocated_vertices is None:
            ancestry.relocated_vertices = set()
        ancestry.relocated_vertices.update(changed.tolist())
        if ancestry.cell_kinds is None:
            ancestry.cell_kinds = {}
        for cell in snapshot.tetrahedra[
            np.any(np.isin(snapshot.tetrahedra, changed), axis=1)
        ].tolist():
            identifier, _ = ancestry.cells[tuple(sorted(cell))]
            ancestry.cell_kinds[identifier] = int(
                _lineage_operation_kind(
                    EntityLineageKind.RELOCATED,
                    (
                        ancestry.cell_kinds.get(
                            identifier, int(EntityLineageKind.PRESERVED)
                        ),
                    ),
                )
            )
        objective.commit(
            _StatisticalProposal(
                ledger, float(np.sum(quality)), snapshot.tetrahedra.shape[0]
            )
        )
        return native.arrays(), changed.size, 1 + budget.optimizer_evaluations - before
    return snapshot, 0, 1 + budget.optimizer_evaluations - before


def _relocate_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    fixed: np.ndarray,
    budget: int,
    source_tables: TetraMetricSource,
    source_metric: np.ndarray,
    ancestry: _Ancestry,
    native_budget: _NativeMetricBudget,
    objective: _StatisticalObjective | None,
    /,
) -> tuple[TetMeshArrays, int, int]:
    if objective is not None:
        return _statistical_relocate_pass(
            native, snapshot, values, fixed, budget, source_metric, ancestry, objective
        )
    edges = _edges(snapshot.tetrahedra)
    lengths = _lengths(snapshot.points, values, edges)
    delta = snapshot.points[edges[:, 1]] - snapshot.points[edges[:, 0]]
    force = (1.0 - 1.0 / np.maximum(lengths, 1.0e-30))[:, None] * delta
    total = np.zeros_like(snapshot.points)
    degree = np.zeros((snapshot.points.shape[0],), dtype=np.float64)
    np.add.at(total, edges[:, 0], force)
    np.add.at(total, edges[:, 1], -force)
    np.add.at(degree, edges.reshape((-1,)), 1.0)
    vertices = np.flatnonzero(snapshot.vertex_dimension == 3)
    vertices = vertices[
        ~np.asarray(
            [vertex < fixed.size and fixed[vertex] for vertex in vertices], dtype=np.bool_
        )
    ][:budget]
    if vertices.size == 0:
        return snapshot, 0, 0
    fractions = np.asarray((1.0, 0.5, 0.25), dtype=np.float64)
    directions = total[vertices] / np.maximum(degree[vertices, None], 1.0)
    targets = (
        snapshot.points[vertices, None] + fractions[None, :, None] * directions[:, None]
    )
    stencil = native_budget.stencil(targets.reshape((-1, 3)))
    if stencil is None:
        return snapshot, 0, 0
    source_rows = key_rows(
        source_tables.vertex_global_ids[:, None],
        stencil[0].reshape((-1, 1)),
    ).reshape(stencil[0].shape)
    samples = _interpolate(source_metric[source_rows], stencil[1]).reshape((-1, 3, 3, 3))
    covered = stencil[2].reshape((-1, 3, 4)).all(axis=-1)
    applied, work = 0, 0
    for index, vertex in enumerate(vertices.tolist()):
        cavity = snapshot.tetrahedra[np.any(snapshot.tetrahedra == vertex, axis=1)]
        old = np.min(_quality(snapshot.points, values, cavity))
        for step in range(3):
            if work >= budget or native_budget.exhausted:
                break
            work += 1
            if not covered[index, step]:
                continue
            proposed = snapshot.points.copy()
            proposed[vertex] = targets[index, step]
            metrics = values.copy()
            metrics[vertex] = samples[index, step]
            if (
                not np.all(_positive(proposed, cavity))
                or np.min(_quality(proposed, metrics, cavity)) <= old + 1.0e-6
            ):
                continue
            if native_budget.call(
                lambda _: native.relocate(
                    vertex, targets[index, step], require_improvement=False
                )
            ):
                snapshot = native.arrays()
                values[vertex] = samples[index, step]
                ancestry.vertex_sources[vertex] = set(
                    stencil[0][
                        index * 3 + step, stencil[1][index * 3 + step] > 0.0
                    ].tolist()
                )
                if ancestry.relocated_vertices is None:
                    ancestry.relocated_vertices = set()
                ancestry.relocated_vertices.add(vertex)
                if ancestry.cell_kinds is None:
                    ancestry.cell_kinds = {}
                for cell in cavity.tolist():
                    identifier, _ = ancestry.cells[tuple(sorted(cell))]
                    ancestry.cell_kinds[identifier] = int(
                        _lineage_operation_kind(
                            EntityLineageKind.RELOCATED,
                            (
                                ancestry.cell_kinds.get(
                                    identifier, int(EntityLineageKind.PRESERVED)
                                ),
                            ),
                        )
                    )
                applied += 1
                break
    return snapshot, applied, work


__all__ = [
    "MetricRemeshingCriterion",
    "MetricRemeshingEvidence",
    "MetricRemeshingStatus",
    "NativeTetraMetricOutcome",
    "TetraMetricOutcome",
    "TetraMetricSource",
    "UniformSizeRemeshingGoal",
    "execute_native_tetra_metric",
    "execute_tetra_metric_adaptation",
]
