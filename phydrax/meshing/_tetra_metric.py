#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Metric-driven tetrahedral remeshing on the native constrained cavity owner.

This is host-only epoch preparation. Native edits certify legality and atomic
rollback; this schedule owns the metric objective, source ancestry and budgets.
No scalar equivalent size is substituted for a tensor metric.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import dataclass, field, fields, InitVar, is_dataclass
from enum import StrEnum
from fractions import Fraction
from itertools import combinations
from time import monotonic
from typing import final, NamedTuple, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, custom_jvp, lax
from jax.typing import ArrayLike

import phydrax._geometry_predicates as _dependency_phydrax__geometry_predicates
import phydrax._geometry_predicates as _owner_phydrax__geometry_predicates
import phydrax.discretization._cell_geometry as _dependency_phydrax_discretization__cell_geometry
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.discretization._exact_plc_geometry as _dependency_phydrax_discretization__exact_plc_geometry
import phydrax.geometry.implicit._projection_cover as implicit_projection_owner
import phydrax.linalg._small_batched as _owner_phydrax_linalg__small_batched
import phydrax.meshing._entity_keys as entity_keys_owner
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._metric as _dependency_phydrax_meshing__metric
import phydrax.meshing._sizing as _dependency_phydrax_meshing__sizing
import phydrax.meshing._topology_edit as _dependency_phydrax_meshing__topology_edit
import phydrax.meshing._trace as _dependency_phydrax_meshing__trace
import phydrax.meshing.providers._native_route_checks as _owner_phydrax_meshing_providers__native_publication
import phydrax.optim._least_squares as _owner_phydrax_optim__least_squares
from phydrax.geometry._rational import (
    fraction_endpoint,
    rational_vector as plc_rational_point,
)

from .._bvh import bvh_overlap_pair_blocks, PackedBVH, prepare_bvh
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import (
    exact_orient2d,
    exact_orient3d,
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    TetMesh3D,
    TetMeshArrays,
    TetMeshCurvedSources,
    TetMeshEdgeConstruction,
    TetMeshInsertionPlan,
    TetMeshSourceComplex,
    TetMeshUnmet,
)
from .._resource_ticket import BatchTicket, ManagedBytesContributor
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer
from ..discretization import CellMesh
from ..discretization._reference_cell import reference_cell_topology
from ..geometry._embedding_state import MeshCertificateLimits
from ..geometry.implicit._analytic_profile import AnalyticImplicitProfile
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
from ..linalg._dense_pseudoinverse import apply_pseudoinverse, factor_pseudoinverse
from ..nonlinear import quadratic_event_bound
from ..optim import (
    AbstractLeastSquaresTrialPolicy,
    Bounds,
    compose_active_gradient_bank,
    ConvexSolvePolicy,
    ConvexTermination,
    DensePrimalDualQP,
    LeastSquaresResult,
    LeastSquaresState,
    LeastSquaresTrialResult,
    LevenbergMarquardt,
    NonlinearLeastSquaresProblem,
    OptimizationStatus,
    OptimizationTermination,
    QuadraticProgram,
    solve_quadratic_program,
)
from ..typing import checked, parse
from ._lineage_records import EntityLineageKind
from ._measurements import (
    measure_phase,
    NativeExecutionRecord,
    NativeMeshingPhase,
    NativeMeshingPhaseRecorder,
    phase_started,
    record_elapsed,
)
from ._metric_execution import (
    metric_execution_passes,
    MetricExecutionSchedule,
    validate_metric_operation_controls,
    validate_metric_phase_recorder,
    validate_metric_quality_floor,
)


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
_METRIC_TRAJECTORY_STRATEGY_ID = canonical_fingerprint(
    {
        "kind": "native-metric-operation-order",
        "ordinary_coarsening": ("collapse", "flip", "ring", "split", "relocate"),
        "statistical_coarsening": ("collapse", "split", "flip", "ring", "relocate"),
        "refinement": ("split", "collapse", "flip", "ring", "relocate"),
        "existing_quality_preface": ("focused-flip", "focused-ring"),
        "pass_bound": "authored-maximum-passes",
    }
)


def lineage_operation_kind(
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
class MetricOptimizerEvidence(StrictModule, NonTrainableState):
    """Optimizer termination, objective, proposal, and work counters."""

    status_counts: tuple[int, ...] = eqx.field(static=True, default=())
    terminations: tuple[tuple[int, int], ...] = eqx.field(static=True, default=())
    method_id: str | None = eqx.field(static=True, default=None)
    tolerances: tuple[float, ...] = eqx.field(static=True, default=())
    iterations: int = eqx.field(static=True, default=0)
    evaluations: int = eqx.field(static=True, default=0)
    constraint_evaluations: int = eqx.field(static=True, default=0)
    proposal_evaluations: int = eqx.field(static=True, default=0)
    refused_proposals: int = eqx.field(static=True, default=0)
    refusal_counts: tuple[tuple[str, int], ...] = eqx.field(static=True, default=())
    host_work_units: int = eqx.field(static=True, default=0)

    def __check_init__(self) -> None:
        self.validate_restored()

    def validate_restored(self) -> None:
        """Validate optimizer counter domains and retained termination policy."""
        counters = (
            *self.status_counts,
            self.iterations,
            self.evaluations,
            self.constraint_evaluations,
            self.proposal_evaluations,
            self.refused_proposals,
            self.host_work_units,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError(
                "MetricOptimizerEvidence.counters must be nonnegative integers; "
                f"got {counters!r}"
            )
        if len(self.status_counts) not in (0, len(OptimizationStatus)) or any(
            type(steps) is not int
            or type(evaluations) is not int
            or steps < 1
            or evaluations < 1
            for steps, evaluations in self.terminations
        ):
            raise ValueError(
                "MetricOptimizerEvidence.terminations must retain positive integer "
                f"limits and a full status table; got {self.terminations!r}"
            )
        if self.method_id is not None and (
            type(self.method_id) is not str or not self.method_id
        ):
            raise ValueError(
                "MetricOptimizerEvidence.method_id must be nonempty or None; "
                f"got {self.method_id!r}"
            )
        if len(self.tolerances) not in (0, 4) or any(
            not np.isfinite(value) or value < 0 for value in self.tolerances
        ):
            raise ValueError(
                "MetricOptimizerEvidence.tolerances must contain four finite "
                f"nonnegative thresholds or be empty; got {self.tolerances!r}"
            )
        if any(
            type(name) is not str or not name or type(count) is not int or count < 0
            for name, count in self.refusal_counts
        ):
            raise ValueError(
                "MetricOptimizerEvidence.refusal_counts must contain named "
                f"nonnegative integer counts; got {self.refusal_counts!r}"
            )


@final
class MetricMemoryEvidence(StrictModule, NonTrainableState):
    """Named native allocator measurements and optimizer scratch requirement."""

    native_counters: NativeExecutionMemoryCounters | None = None
    optimizer_scratch_required_bytes: int = eqx.field(static=True, default=0)

    def __check_init__(self) -> None:
        self.validate_restored()

    def validate_restored(self) -> None:
        """Validate memory counters without reconstructing discarded allocations."""
        if (
            self.native_counters is not None
            and type(self.native_counters) is not NativeExecutionMemoryCounters
        ):
            raise ValueError(
                "MetricMemoryEvidence.native_counters must be NativeExecutionMemoryCounters "
                f"or None; got {type(self.native_counters).__name__}."
            )
        nonnegative_integer(
            self.optimizer_scratch_required_bytes,
            "MetricMemoryEvidence.optimizer_scratch_required_bytes",
        )


@final
class _MetricMeasurements(StrictModule):
    out_of_range: int = eqx.field(static=True)
    minimum_length: float = eqx.field(static=True)
    maximum_length: float = eqx.field(static=True)
    minimum_quality: float = eqx.field(static=True)

    @classmethod
    def prepare(
        cls,
        lengths: np.ndarray,
        quality: np.ndarray,
        fidelity: float,
        lower: float,
        upper: float,
    ) -> _MetricMeasurements:
        if (
            lengths.ndim != 1
            or quality.ndim != 1
            or not np.all(np.isfinite(lengths))
            or not np.all(np.isfinite(quality))
        ):
            raise ValueError(
                "MetricRemeshingEvidence.measurements must contain finite vectors; got an invalid measurement bank"
            )
        if not np.isfinite(fidelity) or fidelity < 0:
            raise ValueError(
                f"MetricRemeshingEvidence.maximum_fidelity_bound must be finite and nonnegative; got {fidelity!r}"
            )
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or not 0.0 <= lower <= upper
            or upper <= 0.0
        ):
            raise ValueError(
                "tetra_metric.metric_edge_band must be a finite nonnegative interval with positive upper bound; got invalid bounds"
            )
        return cls(
            int(np.count_nonzero((lengths < lower) | (lengths > upper))),
            float(np.min(lengths, initial=np.inf)) if lengths.size else 0.0,
            float(np.max(lengths, initial=0.0)),
            float(np.min(quality, initial=1.0)),
        )


@final
class _MetricArcPreparation(StrictModule):
    bounds: np.ndarray | None
    edge_ids: np.ndarray | None
    binding: str | None = eqx.field(static=True)

    @classmethod
    def prepare(
        cls,
        bounds: np.ndarray | None,
        edge_ids: np.ndarray | None,
        binding: str | None,
        count: int | None,
    ) -> _MetricArcPreparation:
        if (bounds is None) != (edge_ids is None) or (bounds is None) != (
            binding is None
        ):
            raise ValueError(
                "MetricRemeshingEvidence.metric_arc_binding must accompany bounds and scientific edge IDs; got a partial bank"
            )
        if bounds is not None and edge_ids is not None:
            if (
                bounds.ndim != 2
                or bounds.shape[1] != 2
                or count is not None
                and bounds.shape[0] != count
                or edge_ids.shape != bounds.shape
                or not np.issubdtype(edge_ids.dtype, np.integer)
                or not np.all(np.isfinite(bounds))
                or np.any(bounds[:, 0] < 0)
                or np.any(bounds[:, 1] < bounds[:, 0])
                or type(binding) is not str
                or not binding
            ):
                raise ValueError(
                    f"MetricRemeshingEvidence.metric_arc_bounds must contain finite outward bounds aligned with scientific edges; got bounds.shape={bounds.shape}, edge_ids.shape={edge_ids.shape}"
                )
        return cls(bounds, edge_ids, binding)


@final
class _MetricShapeLimits(StrictModule):
    quality_floor: float | None = eqx.field(static=True)
    radius_edge_bound: float | None = eqx.field(static=True)
    minimum_dihedral_degrees: float | None = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.quality_floor is not None and (
            not np.isfinite(self.quality_floor) or not 0.0 <= self.quality_floor <= 1.0
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.metric_quality_floor must lie in [0, 1]; got {self.quality_floor!r}"
            )
        if self.radius_edge_bound is not None and (
            not np.isfinite(self.radius_edge_bound) or self.radius_edge_bound <= 0.0
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.shape_radius_edge_bound must be finite and positive; got {self.radius_edge_bound!r}"
            )
        if self.minimum_dihedral_degrees is not None and (
            not np.isfinite(self.minimum_dihedral_degrees)
            or not 0.0 <= self.minimum_dihedral_degrees <= 180.0
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.shape_minimum_dihedral_degrees must lie in [0, 180]; got {self.minimum_dihedral_degrees!r}"
            )


@final
class MetricRemeshingEvidence(StrictModule, NonTrainableState):
    """Remeshing goal, measurements, optimizer termination, and resource evidence."""

    status: MetricRemeshingStatus = eqx.field(static=True)
    criterion: MetricRemeshingCriterion = eqx.field(static=True)
    trajectory_strategy_id: str = eqx.field(static=True)
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
    predicate_mode: _dependency_phydrax__geometry_predicates.PredicateMode = eqx.field(
        static=True
    )
    split_parameters: tuple[float, ...] = eqx.field(static=True)
    lower_metric_length: float = eqx.field(static=True)
    upper_metric_length: float = eqx.field(static=True)
    operation_attempts: int = eqx.field(static=True)
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
    optimizer: MetricOptimizerEvidence
    memory: MetricMemoryEvidence
    native_work_units: int = eqx.field(static=True)
    execution_evidence: NativeExecutionRecord | None
    metric_arc_bounds: Array | None
    metric_arc_edge_ids: Array | None
    metric_arc_binding: str | None = eqx.field(static=True)
    source_lower_exemptions: tuple[tuple[int, int, float], ...] = eqx.field(static=True)
    source_lower_binding: str | None = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @checked
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
        optimizer: MetricOptimizerEvidence | None = None,
        memory: MetricMemoryEvidence | None = None,
        native_work_units: int | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
        metric_arc_bounds: np.ndarray | None = None,
        metric_arc_edge_ids: np.ndarray | None = None,
        metric_arc_binding: str | None = None,
        source_lower_exemptions: tuple[tuple[int, int, float], ...] = (),
        source_lower_binding: str | None = None,
    ) -> None:
        status = parse(status, MetricRemeshingStatus, "status")
        criterion = parse(criterion, MetricRemeshingCriterion, "criterion")
        arcs = _MetricArcPreparation.prepare(
            metric_arc_bounds, metric_arc_edge_ids, metric_arc_binding, lengths.size
        )
        if any(
            not np.isfinite(value) or not 0.0 < value < 1.0 for value in split_parameters
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.split_parameters must contain finite edge-interior fractions; got {split_parameters!r}"
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
        )
        for index, value in enumerate(numbers):
            nonnegative_integer(value, f"tetra_metric.counter[{index}]")
        measurements = _MetricMeasurements.prepare(
            lengths,
            quality,
            maximum_fidelity_bound,
            lower_metric_length,
            upper_metric_length,
        )
        optimizer = MetricOptimizerEvidence() if optimizer is None else optimizer
        memory = MetricMemoryEvidence() if memory is None else memory
        native_work = work_units if native_work_units is None else native_work_units
        if (
            isinstance(native_work, bool)
            or not isinstance(native_work, int)
            or native_work < 0
            or native_work > work_units
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.native_work_units must lie between zero and work_units; got {native_work!r} of {work_units!r}"
            )
        if any(
            not key or not np.isfinite(value)
            for key, value in (*size_requested, *size_achieved)
        ):
            raise ValueError(
                "MetricRemeshingEvidence.size_evidence must contain named finite quantities; got an invalid entry"
            )
        if criterion is MetricRemeshingCriterion.SIZE_STATISTICS and (
            not size_requested or not size_achieved
        ):
            raise ValueError(
                "MetricRemeshingEvidence.size_evidence must retain requested and achieved quantities for SIZE_STATISTICS; got an empty bank"
            )
        if status is MetricRemeshingStatus.COMPLETE and size_issues:
            raise ValueError(
                f"MetricRemeshingEvidence.size_issues must be empty for COMPLETE status; got {size_issues!r}"
            )
        shape = _MetricShapeLimits(
            metric_quality_floor, shape_radius_edge_bound, shape_minimum_dihedral_degrees
        )
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
        self.size_requested = size_requested
        self.size_achieved = size_achieved
        self.size_issues = size_issues
        self.source_candidate_pairs = source_candidate_pairs
        self.metric_quality_floor = shape.quality_floor
        self.shape_radius_edge_bound = shape.radius_edge_bound
        self.shape_minimum_dihedral_degrees = shape.minimum_dihedral_degrees
        self.resource_message = resource_message
        self.resource_requested = resource_requested
        self.resource_achieved = resource_achieved
        self.optimizer = optimizer
        self.memory = memory
        self.native_work_units = native_work
        self.execution_evidence = execution_evidence
        _host_metric_arc_bounds = (
            None if arcs.bounds is None else np.asarray(arcs.bounds, dtype=np.float64)
        )
        _host_metric_arc_edge_ids = (
            None if arcs.edge_ids is None else np.asarray(arcs.edge_ids, dtype=np.int64)
        )
        (
            self.metric_arc_bounds,
            self.metric_arc_edge_ids,
        ) = jax.device_put(
            (
                _host_metric_arc_bounds,
                _host_metric_arc_edge_ids,
            )
        )
        self.metric_arc_binding = metric_arc_binding
        self.out_of_range_edges = measurements.out_of_range
        self.minimum_metric_length = measurements.minimum_length
        self.maximum_metric_length = measurements.maximum_length
        self.minimum_metric_quality = measurements.minimum_quality
        self.maximum_fidelity_bound = maximum_fidelity_bound
        self.predicate_mode = _dependency_phydrax__geometry_predicates.PredicateMode.EXACT
        self.split_parameters = split_parameters
        self.source_lower_exemptions = source_lower_exemptions
        self.source_lower_binding = source_lower_binding
        self._validate_source_lower_exemptions()
        self.trajectory_strategy_id = _METRIC_TRAJECTORY_STRATEGY_ID
        self.evidence_id = self._source_identity(metric_arc_bounds, metric_arc_edge_ids)

    def _source_identity(
        self, metric_arc_bounds: object, metric_arc_edge_ids: object
    ) -> str:
        return canonical_fingerprint(
            {
                "kind": "native-metric-remeshing",
                "status": self.status.value,
                "criterion": self.criterion.value,
                "trajectory_strategy_id": self.trajectory_strategy_id,
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
                "source_lower_exemptions": self.source_lower_exemptions,
                "source_lower_binding": self.source_lower_binding,
                "operation_attempts": self.operation_attempts,
                "native_memory_counters": (
                    None
                    if self.memory.native_counters is None
                    else dict(self.memory.native_counters.items())
                ),
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
                "optimizer_status_counts": self.optimizer.status_counts,
                "optimizer_terminations": self.optimizer.terminations,
                "optimizer_method_id": self.optimizer.method_id,
                "optimizer_tolerances": self.optimizer.tolerances,
                "optimizer_iterations": self.optimizer.iterations,
                "optimizer_evaluations": self.optimizer.evaluations,
                "optimizer_constraint_evaluations": self.optimizer.constraint_evaluations,
                "optimizer_proposal_evaluations": self.optimizer.proposal_evaluations,
                "optimizer_refused_proposals": self.optimizer.refused_proposals,
                "optimizer_refusal_counts": self.optimizer.refusal_counts,
                "optimizer_host_work_units": self.optimizer.host_work_units,
                "optimizer_scratch_required_bytes": self.memory.optimizer_scratch_required_bytes,
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
            canonical_identifier(
                self.trajectory_strategy_id, "MetricRemeshingEvidence.trajectory_strategy_id"
            )
            != self.trajectory_strategy_id
        ):
            raise ValueError(
                "MetricRemeshingEvidence.trajectory_strategy_id must retain its canonical stored strategy identity; got changed identifier"
            )
        if (
            type(self.status) is not MetricRemeshingStatus
            or type(self.criterion) is not MetricRemeshingCriterion
        ):
            raise TypeError(
                "MetricRemeshingEvidence.selectors must use registered status and criterion types; got foreign selectors"
            )
        if (
            self.predicate_mode
            is not _dependency_phydrax__geometry_predicates.PredicateMode.EXACT
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.predicate_mode must be EXACT; got {self.predicate_mode!r}"
            )
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
            self.native_work_units,
        )
        if (
            any(type(value) is not int or value < 0 for value in counters)
            or self.native_work_units > self.work_units
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.counters must be nonnegative integers with native work no greater than total work; got {counters!r}"
            )
        self._validate_restored_measurements()
        self._validate_source_lower_exemptions()
        if type(self.memory) is not MetricMemoryEvidence:
            raise TypeError(
                "MetricRemeshingEvidence.memory must be MetricMemoryEvidence; "
                f"got {type(self.memory).__name__}"
            )
        self.memory.validate_restored()
        if type(self.optimizer) is not MetricOptimizerEvidence:
            raise TypeError(
                "MetricRemeshingEvidence.optimizer must be MetricOptimizerEvidence; "
                f"got {type(self.optimizer).__name__}"
            )
        self.optimizer.validate_restored()
        self._validate_restored_obligations()
        self._validate_restored_arcs()
        if self.execution_evidence is not None:
            if type(self.execution_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    f"MetricRemeshingEvidence.execution_evidence must be NativeExecutionRecord; got {type(self.execution_evidence).__name__}"
                )
            self.execution_evidence.require_valid()
        if self.evidence_id != self._source_identity(
            self.metric_arc_bounds, self.metric_arc_edge_ids
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.evidence_id must match the restored aggregate fingerprint; got {self.evidence_id!r}"
            )

    @property
    def unmet_metric_edges(self) -> int:
        """Unit-band failures not owned by an authenticated source fidelity constraint."""
        return self.out_of_range_edges - len(self.source_lower_exemptions)

    def _validate_source_lower_exemptions(self) -> None:
        rows = self.source_lower_exemptions
        if bool(rows) != (self.source_lower_binding is not None):
            raise ValueError(
                "MetricRemeshingEvidence.source_lower_binding must accompany exemptions; got a partial bank"
            )
        if self.source_lower_binding is not None:
            canonical_identifier(
                self.source_lower_binding, "MetricRemeshingEvidence.source_lower_binding"
            )
        if (
            len(rows) > self.out_of_range_edges
            or tuple(sorted(rows)) != rows
            or len({(first, second) for first, second, _ in rows}) != len(rows)
            or any(
                type(first) is not int or type(second) is not int
                or not 0 <= first < second
                or not np.isfinite(length) or not 0 <= length < self.lower_metric_length
                for first, second, length in rows
            )
        ):
            raise ValueError(
                "MetricRemeshingEvidence.source_lower_exemptions must retain unique ordered short edge rows; got invalid support"
            )

    def _validate_restored_measurements(self) -> None:
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
                "MetricRemeshingEvidence.measurements must be finite with nonnegative fidelity; got invalid retained measurements"
            )
        if (
            not 0 <= self.lower_metric_length <= self.upper_metric_length
            or self.upper_metric_length <= 0
        ):
            raise ValueError(
                "MetricRemeshingEvidence.metric_edge_band must be finite, nonnegative, and ordered; got invalid retained bounds"
            )

    def _validate_restored_obligations(self) -> None:
        if any(
            not np.isfinite(value) or not 0 < value < 1 for value in self.split_parameters
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.split_parameters must contain finite edge-interior fractions; got {self.split_parameters!r}"
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
                "MetricRemeshingEvidence.resource_evidence must contain named finite quantities; got an invalid retained entry"
            )
        if self.criterion is MetricRemeshingCriterion.SIZE_STATISTICS and (
            not self.size_requested or not self.size_achieved
        ):
            raise ValueError(
                "MetricRemeshingEvidence.size_evidence must retain requested and achieved quantities for SIZE_STATISTICS; got an empty bank"
            )
        if self.status is MetricRemeshingStatus.COMPLETE and self.size_issues:
            raise ValueError(
                f"MetricRemeshingEvidence.size_issues must be empty for COMPLETE status; got {self.size_issues!r}"
            )
        if self.metric_quality_floor is not None and (
            not np.isfinite(self.metric_quality_floor)
            or not 0 <= self.metric_quality_floor <= 1
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.metric_quality_floor must lie in [0, 1]; got {self.metric_quality_floor!r}"
            )
        if self.shape_radius_edge_bound is not None and (
            not np.isfinite(self.shape_radius_edge_bound)
            or self.shape_radius_edge_bound <= 0
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.shape_radius_edge_bound must be finite and positive; got {self.shape_radius_edge_bound!r}"
            )
        if self.shape_minimum_dihedral_degrees is not None and (
            not np.isfinite(self.shape_minimum_dihedral_degrees)
            or not 0 <= self.shape_minimum_dihedral_degrees <= 180
        ):
            raise ValueError(
                f"MetricRemeshingEvidence.shape_minimum_dihedral_degrees must lie in [0, 180]; got {self.shape_minimum_dihedral_degrees!r}"
            )

    def _validate_restored_arcs(self) -> None:
        if (self.metric_arc_bounds is None) != (self.metric_arc_edge_ids is None) or (
            self.metric_arc_bounds is None
        ) != (self.metric_arc_binding is None):
            raise ValueError(
                "MetricRemeshingEvidence.metric_arc_binding must accompany bounds and scientific edge IDs; got a partial retained bank"
            )
        if self.metric_arc_bounds is not None:
            host_bounds, host_identifiers = jax.device_get(
                (self.metric_arc_bounds, self.metric_arc_edge_ids)
            )
            bounds, identifiers = np.asarray(host_bounds), np.asarray(host_identifiers)
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
                    "MetricRemeshingEvidence.metric_arc_bounds must align finite outward bounds with scientific edges; got an invalid retained bank"
                )

    @property
    def converged(self) -> bool:
        return self.status is MetricRemeshingStatus.COMPLETE

    @property
    def stalled(self) -> bool:
        return self.status is MetricRemeshingStatus.STALLED


def source_location_budget(
    maximum_pairs: int, candidates: int, /
) -> _leaf_phydrax_meshing__failures.MeshingFailure:
    return _leaf_phydrax_meshing__failures.MeshingFailure(
        _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
        "Metric source-location candidate budget is exhausted.",
        stage=_dependency_phydrax_meshing__trace.MeshingStageKind.GEOMETRY_ASSOCIATION.value,
        requested=(("source_location_candidates", float(maximum_pairs)),),
        achieved=(("source_location_candidates", float(candidates)),),
    )


def stage_source_limits(
    mesh: CellMesh, maximum_vertices: int, maximum_cells: int, /
) -> None:
    vertices = mesh.coordinates.shape[0]
    cells = sum(block.cell_count for block in mesh.blocks)
    if vertices > maximum_vertices or cells > maximum_cells:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "The metric construction budget cannot stage the accepted source.",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.SOURCE_INSPECTION.value,
            requested=(
                ("vertices", float(maximum_vertices)),
                ("cells", float(maximum_cells)),
            ),
            achieved=(("vertices", float(vertices)), ("cells", float(cells))),
        )


class TetraMetricOutcome(NamedTuple):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    metric: np.ndarray
    evidence: MetricRemeshingEvidence
    exact_source: (
        _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource
        | None
    ) = None


def fresh_identifiers(first: int, count: int, /) -> np.ndarray:
    maximum = np.iinfo(np.int64).max
    if count < 0:
        raise RuntimeError(
            f"tetra_metric.vertex_count must not drop retained source slots; got {count!r}"
        )
    if count and first + count - 1 > maximum:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Scientific global-identifier capacity is exhausted.",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
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
    constrained_faces: dict[
        tuple[int, ...], dict[tuple[int, ...], int]
    ] | None = None
    curved_collapses: int = 0


def _edges(cells: np.ndarray, /) -> np.ndarray:
    local = np.asarray(tuple(combinations(range(cells.shape[1]), 2)), dtype=np.int32)
    return np.unique(np.sort(cells[:, local].reshape((-1, 2)), axis=1), axis=0)


def resolved_collapses(
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
                raise RuntimeError(
                    f"tetra_metric.collapse_lineage must be acyclic; got a cycle at vertex {current!r}"
                )
            visited.add(current)
            current = successor[current]
        if current != vertex:
            result[int(identifiers[vertex])] = int(identifiers[current])
    return result


def subentity_relations(
    mesh: CellMesh,
    dimension: int,
    cells: np.ndarray,
    vertex_ids: np.ndarray,
    vertex_sources: list[set[int]],
    /,
    *,
    collapsed_vertices: dict[int, int] | None = None,
    compose_collapses: bool = False,
) -> _dependency_phydrax_meshing__topology_edit.EntityRelations:
    """Return split/collapse ancestry derived from construction witnesses.

    ``compose_collapses`` matches targets against the collapsed images of
    source entities, so later refinements of a collapsed entity (for example
    a curved boundary face) keep their source parent as COLLAPSED_INTO.
    """
    local = np.asarray(
        tuple(combinations(range(cells.shape[1]), dimension + 1)), dtype=np.int32
    )
    targets, target_counts = np.unique(
        np.sort(cells[:, local].reshape((-1, dimension + 1)), axis=1),
        axis=0,
        return_counts=True,
    )
    source_keys = entity_keys_owner.entity_keys(mesh, dimension)
    source_boundary: set[tuple[int, ...]] | None = None
    target_boundary: set[tuple[int, ...]] | None = None
    if compose_collapses and dimension == mesh.topological_dimension - 1:
        # Collapse images preserve the boundary subcomplex. An interior
        # source facet may acquire the same vertex image as a boundary
        # target, but is not that target's boundary-source lineage.
        original_cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
        )
        original_facets, original_counts = np.unique(
            np.sort(
                np.asarray(mesh.vertex_global_ids)[
                    original_cells[:, local].reshape((-1, dimension + 1))
                ],
                axis=1,
            ),
            axis=0,
            return_counts=True,
        )
        source_boundary = {
            tuple(row) for row in original_facets[original_counts == 1].tolist()
        }
        target_boundary = {
            tuple(sorted(vertex_ids[row].tolist()))
            for row in targets[target_counts == 1]
        }
    collapse = collapsed_vertices if compose_collapses and collapsed_vertices else {}
    images = [
        tuple(sorted({collapse.get(vertex, vertex) for vertex in key}))
        for key in source_keys.tolist()
    ]
    incidence: dict[int, set[int]] = {}
    for row, image in enumerate(images):
        if len(image) != dimension + 1:
            continue
        for vertex in image:
            incidence.setdefault(vertex, set()).add(row)
    sources, descendants, kinds = [], [], []
    target_keys = {
        tuple(sorted(vertex_ids[target].tolist())) for target in targets.tolist()
    }
    for target in targets.tolist():
        key = sorted(vertex_ids[target].tolist())
        support = {
            collapse.get(vertex, vertex)
            for vertex in set().union(*(vertex_sources[vertex] for vertex in target))
        }
        if len(support) > dimension + 1:
            continue
        candidates: set[int] | None = None
        for vertex in support:
            rows = incidence.get(vertex, set())
            candidates = rows.copy() if candidates is None else candidates & rows
        for row in sorted(candidates if candidates is not None else set()):
            source = source_keys[row].tolist()
            if (
                source_boundary is not None
                and target_boundary is not None
                and ((tuple(source) in source_boundary) != (tuple(key) in target_boundary))
            ):
                continue
            if key != source:
                sources.append(source)
                descendants.append(key)
                kinds.append(
                    int(
                        EntityLineageKind.REFINED_FROM
                        if list(images[row]) == source
                        else EntityLineageKind.COLLAPSED_INTO
                    )
                )
    if collapsed_vertices and not compose_collapses:
        for source in source_keys.tolist():
            key = tuple(
                sorted(collapsed_vertices.get(vertex, vertex) for vertex in source)
            )
            if len(set(key)) == len(key) and key != tuple(source) and key in target_keys:
                sources.append(source)
                descendants.append(list(key))
                kinds.append(int(EntityLineageKind.COLLAPSED_INTO))
    width = dimension + 1
    return _dependency_phydrax_meshing__topology_edit.EntityRelations(
        dimension,
        np.asarray(sources, dtype=np.int64).reshape((-1, width)),
        np.asarray(descendants, dtype=np.int64).reshape((-1, width)),
        np.asarray(kinds, dtype=np.int8),
    )


def _bucket_size(count: int, /) -> int:
    return 8 if count <= 8 else 1 << (count - 1).bit_length()


def _bucket(rows: np.ndarray, /) -> np.ndarray:
    """Pad leading rows to a power-of-two count by repeating the first row.

    Rowwise compiled kernels then compile once per bucket instead of once per
    cell, edge, cavity, or query count; callers retain only unpadded rows.
    """
    count = rows.shape[0]
    size = _bucket_size(count)
    if count == 0 or size == count:
        return rows
    return np.concatenate((rows, np.repeat(rows[:1], size - count, axis=0)))


@jax.jit
def _cavity_quality(tensors: Array, corners: Array, /) -> Array:
    """Evaluate the existing metric algebra on only the participating corners."""
    weights = jnp.full(tensors.shape[:-2], 0.25, dtype=tensors.dtype)
    mean = _dependency_phydrax_meshing__metric.interpolate_mesh_metric(tensors, weights)
    return _dependency_phydrax_meshing__metric.metric_shape_quality(
        mean, corners, dimension=3
    )


def _quality(points: np.ndarray, metric: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    result = _cavity_quality(_bucket(metric[cells]), _bucket(points[cells]))
    return np.asarray(jax.device_get(result), dtype=np.float64)[: cells.shape[0]]


def _lengths(points: np.ndarray, metric: np.ndarray, edges: np.ndarray, /) -> np.ndarray:
    if 2 * edges.shape[0] < points.shape[0]:
        local_edges = np.arange(2 * edges.shape[0], dtype=np.int32).reshape((-1, 2))
        result = _dependency_phydrax_meshing__metric.metric_edge_lengths(
            _bucket(metric[edges].reshape((-1, 3, 3))),
            _bucket(points[edges].reshape((-1, 3))),
            _bucket(local_edges),
        )
    else:
        result = _dependency_phydrax_meshing__metric.metric_edge_lengths(
            _bucket(metric), _bucket(points), _bucket(edges)
        )
    return np.asarray(jax.device_get(result), dtype=np.float64)[: edges.shape[0]]


def _interpolate(values: np.ndarray, weights: np.ndarray, /) -> np.ndarray:
    result = _dependency_phydrax_meshing__metric.interpolate_mesh_metric(
        _bucket(values), _bucket(weights)
    )
    return np.asarray(jax.device_get(result), dtype=np.float64)[: values.shape[0]]


def _positive(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    corners = _bucket(points[cells])
    signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    positive = signs == _dependency_phydrax__geometry_predicates.PredicateSign.POSITIVE
    return positive[: cells.shape[0]]


def _collapse_coplanar(
    points: np.ndarray, faces: np.ndarray, kept: int, /
) -> bool:
    """Apply the native boundary-collapse necessary plane test in one bank."""
    if faces.shape[0] == 0:
        return True
    corners = _bucket(points[faces])
    signs = exact_orient3d(
        corners[:, 0], corners[:, 1], corners[:, 2],
        np.broadcast_to(points[kept], corners[:, 0].shape),
    )
    return bool(np.all(signs[: faces.shape[0]] == 0))


def _oriented_positive(
    points: np.ndarray, cells: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Orient and certify a proposal with one exact predicate bank."""
    corners = _bucket(points[cells])
    signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    signs = signs[: cells.shape[0]]
    result = cells.copy()
    swap = signs != _dependency_phydrax__geometry_predicates.PredicateSign.POSITIVE
    result[swap, 0], result[swap, 1] = result[swap, 1].copy(), result[swap, 0].copy()
    positive = signs == _dependency_phydrax__geometry_predicates.PredicateSign.POSITIVE
    negative = signs == _dependency_phydrax__geometry_predicates.PredicateSign.NEGATIVE
    return result, positive | negative


def _orient(points: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    return _oriented_positive(points, cells)[0]


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
            f"tetra_metric.{name} must contain aligned nonnegative int32-range labels; got shape={array.shape}, dtype={array.dtype}"
        )
    return array.astype(np.int32)


def constraints(
    mesh: CellMesh,
    cells: np.ndarray,
    regions: np.ndarray,
    facet_classes: ArrayLike | None,
    edge_classes: ArrayLike | None,
    protected_edges: np.ndarray,
    /,
    *,
    smooth_facets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Constrained faces, their native sources, segments and segment sources.

    Every constrained face is its own source and every face edge a segment,
    unless ``smooth_facets`` (a mask over facet entity rows) names faces of
    one explicit smooth source. Those share one source per material pair and
    facet class; only their edges between distinct sources, open edges, real
    feature edges and protected edges remain segments.
    """
    faces = cells[:, _FACES].reshape((-1, 3))
    keys, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    labels = _labels(facet_classes, mesh.entity_set(2).count, "facet_classes")
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(mesh, 2),
        np.sort(vertex_ids[keys], axis=1),
    )
    low = np.full((keys.shape[0],), np.iinfo(np.int32).max, dtype=np.int32)
    high = np.full((keys.shape[0],), -1, dtype=np.int32)
    np.minimum.at(low, inverse, np.repeat(regions, 4))
    np.maximum.at(high, inverse, np.repeat(regions, 4))
    constrained = (counts == 1) | (low != high) | (labels[rows] != 0)
    first = np.full((keys.shape[0],), faces.shape[0], dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(faces.shape[0], dtype=np.int64))
    selected = faces[first[constrained]]
    sources = np.arange(selected.shape[0], dtype=np.int32)
    face_edges = np.sort(selected[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1)
    boundary_edges = np.unique(face_edges, axis=0)
    if smooth_facets is not None:
        smooth_mask = np.asarray(smooth_facets, dtype=np.bool_)
        if smooth_mask.shape != (mesh.entity_set(2).count,):
            raise ValueError(
                f"tetra_metric.smooth_facets must mark facet entity rows; got shape={smooth_mask.shape}"
            )
        smooth = smooth_mask[rows[constrained]]
        if np.any(smooth):
            _, groups = np.unique(
                np.column_stack(
                    (
                        low[constrained][smooth],
                        high[constrained][smooth],
                        labels[rows[constrained]][smooth],
                    )
                ),
                axis=0,
                return_inverse=True,
            )
            sources[smooth] = selected.shape[0] + groups.reshape(-1).astype(np.int32)
            edge_keys, edge_inverse, edge_counts = np.unique(
                face_edges, axis=0, return_inverse=True, return_counts=True
            )
            edge_inverse = edge_inverse.reshape(-1)
            first_source = np.full((edge_keys.shape[0],), np.iinfo(np.int32).max, dtype=np.int32)
            last_source = np.full((edge_keys.shape[0],), -1, dtype=np.int32)
            np.minimum.at(first_source, edge_inverse, np.repeat(sources, 3))
            np.maximum.at(last_source, edge_inverse, np.repeat(sources, 3))
            all_smooth = np.ones((edge_keys.shape[0],), dtype=np.bool_)
            np.logical_and.at(all_smooth, edge_inverse, np.repeat(smooth, 3))
            interior = (edge_counts == 2) & (first_source == last_source) & all_smooth
            boundary_edges = edge_keys[~interior]
    edges = _edges(cells)
    edge_rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(mesh, 1),
        np.sort(vertex_ids[edges], axis=1),
    )
    feature = (
        _labels(edge_classes, mesh.entity_set(1).count, "edge_classes")[edge_rows] != 0
    )
    segments = np.unique(
        np.concatenate((boundary_edges, edges[feature | protected_edges[edge_rows]])),
        axis=0,
    )
    return selected, sources, segments, np.arange(segments.shape[0], dtype=np.int32)


@final
class CurvedBoundarySource(StrictModule):
    """Explicit smooth source of tetrahedral boundary facets under analytic fidelity.

    ``facets`` marks the carrier's facet entity rows on the source's smooth
    zero set. Those faces share one native source per material pair and
    facet class; real feature curves, protected edges and fixed vertices stay
    constrained. Every curved face, initial or new, must carry a whole-facet
    bound of ``implicit_facet_projection_bounds`` under the original
    ``profile``, ``tolerance`` and certificate ``limits``, outward-oriented
    along the source gradient. ``digest`` binds that identity for native claim
    comparison; it is never itself a proof.
    """

    profile: AnalyticImplicitProfile
    limits: MeshCertificateLimits
    facets: np.ndarray
    tolerance: float = eqx.field(static=True)
    digest: bytes = eqx.field(static=True)

    def __init__(
        self,
        profile: AnalyticImplicitProfile,
        tolerance: float,
        limits: MeshCertificateLimits,
        facets: np.ndarray,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        mask = np.asarray(facets)
        if mask.dtype != np.bool_ or mask.ndim != 1:
            raise TypeError(
                f"CurvedBoundarySource.facets must be a boolean facet-row mask; got dtype={mask.dtype}, shape={mask.shape}"
            )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"CurvedBoundarySource.tolerance must be finite and nonnegative; got {tolerance!r}"
            )
        profile.validate_source_integrity(budget=budget)
        self.profile = profile
        self.limits = limits
        self.facets = mask.copy()
        self.facets.setflags(write=False)
        self.tolerance = float(tolerance)
        self.digest = self._source_digest()

    def _source_digest(self) -> bytes:
        return bytes.fromhex(
            canonical_fingerprint(
                {
                    "kind": "tetra-metric-curved-boundary-source",
                    "source_id": self.profile.source_id,
                    "source_revision": self.profile.source_revision,
                    "profile_id": self.profile.profile_id,
                    "profile_arrays": array_tree_fingerprint(self.profile),
                    "profile_facts": {
                        entry.name: getattr(self.profile, entry.name)
                        for entry in fields(self.profile)
                        if entry.metadata.get("static", False)
                    },
                    "coordinate_contract_id": self.profile.coordinate_contract.spatial_id,
                    "tolerance": float(self.tolerance).hex(),
                    "limits_id": self.limits.limits_id,
                    "limits_facts": {
                        entry.name: getattr(self.limits, entry.name)
                        for entry in fields(self.limits)
                    },
                }
            )
        )

    def _integrity_digest(self) -> bytes | None:
        """Bind nested static metadata as well as numerical source leaves."""
        def facts(value: object) -> object:
            if is_dataclass(value) and not isinstance(value, type):
                return (
                    type(value).__module__, type(value).__qualname__,
                    tuple((entry.name, facts(getattr(value, entry.name))) for entry in fields(value)),
                )
            if isinstance(value, dict):
                return tuple((facts(key), facts(item)) for key, item in value.items())
            if isinstance(value, (tuple, list)):
                return tuple(facts(item) for item in value)
            if isinstance(value, bytes):
                return ("bytes", value.hex())
            if isinstance(value, float):
                return ("float", value.hex())
            if value is None or isinstance(value, (str, int, bool, np.ndarray, np.generic, Array)):
                return value
            raise TypeError("Source metadata has no immutable content binding.")

        try:
            return bytes.fromhex(canonical_fingerprint(facts(self)))
        except TypeError:
            # Unsupported source metadata is never silently left out of an
            # authentication key: such sources retain full validation per use.
            return None

    def validate_source_integrity(self) -> None:
        """Refuse a changed source or policy before issuing or reusing evidence."""
        # The constructor proved these profile facts on the caller's ledger.
        # Rebinding their complete arrays/static facts detects mutation without
        # reconstructing an analytic profile under a fresh helper query scope.
        self.profile.field_bounds.validate_source_integrity()
        self.profile.map_bounds.validate_source_integrity()
        if self.digest != self._source_digest():
            raise ValueError(
                "CurvedBoundarySource.digest must bind the current source and policy; got changed source claims"
            )

    def bounds(
        self,
        triangles: np.ndarray,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Issue whole-facet deviations and outward orientation of (k, 3, 3) triangles."""
        self.validate_source_integrity()
        return self._bounds_authenticated(triangles, budget=budget)

    def _bounds_authenticated(
        self,
        triangles: np.ndarray,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate after the request owner has authenticated current source facts."""
        corners = np.ascontiguousarray(triangles, dtype=np.float64)
        if corners.ndim != 3 or corners.shape[1:] != (3, 3):
            raise ValueError(
                f"CurvedBoundarySource.triangles must have shape (k, 3, 3); got {corners.shape}"
            )
        if corners.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.bool_)
        # The field evaluator charges its own source queries to ``budget``.
        deviation, oriented, _ = (
            implicit_projection_owner.implicit_facet_projection_bounds(
                corners, self.profile, self.tolerance, self.limits, budget=budget
            )
        )
        return deviation, oriented

    def witnesses(
        self,
        triangles: np.ndarray,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> np.ndarray:
        """A certified field interval at a selected exact point of each facet.

        Outward boxes enclose the rational centroid and edge midpoints.
        Select the point with the strongest certified distance from zero.
        A point beyond the band disproves admission of the whole facet;
        an unrefuted facet still needs its independent whole-facet proof.
        """
        self.validate_source_integrity()
        return self._witnesses_authenticated(triangles, budget=budget)

    def _witnesses_authenticated(
        self,
        triangles: np.ndarray,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> np.ndarray:
        """Evaluate after the request owner has authenticated current source facts."""
        corners = np.ascontiguousarray(triangles, dtype=np.float64)
        if corners.ndim != 3 or corners.shape[1:] != (3, 3):
            raise ValueError(
                f"CurvedBoundarySource.triangles must have shape (k, 3, 3); got {corners.shape}"
            )
        if corners.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)
        pair = corners[:, 0] + corners[:, 1]
        low = np.nextafter(np.nextafter(pair, -np.inf) + corners[:, 2], -np.inf)
        high = np.nextafter(np.nextafter(pair, np.inf) + corners[:, 2], np.inf)
        centroid_low = np.nextafter(low / 3.0, -np.inf)
        centroid_high = np.nextafter(high / 3.0, np.inf)
        pairs = corners + corners[:, (1, 2, 0)]
        midpoint_low = np.nextafter(np.nextafter(pairs, -np.inf) * 0.5, -np.inf)
        midpoint_high = np.nextafter(np.nextafter(pairs, np.inf) * 0.5, np.inf)
        field = self.profile.field_bounds.boxes(
            np.concatenate((centroid_low[:, None], midpoint_low), axis=1).reshape((-1, 3)),
            np.concatenate((centroid_high[:, None], midpoint_high), axis=1).reshape((-1, 3)),
            budget=budget,
        )
        intervals = np.stack((field.value_lower, field.value_upper), axis=1).reshape(
            (-1, 4, 2)
        )
        distance_lower = np.maximum(intervals[:, :, 0], -intervals[:, :, 1])
        selected = np.argmax(distance_lower, axis=1)
        return intervals[np.arange(corners.shape[0]), selected]

    def refutes(self, witnesses: np.ndarray, /) -> np.ndarray:
        """Facets with a certified point outside the admitted fidelity band.

        Every whole-facet bound encloses the field at that point, so a refuted
        facet can never be admitted; an unrefuted facet still needs its bound.
        """
        band = min(self.tolerance, self.profile.tube_radius)
        return (witnesses[:, 0] > band) | (witnesses[:, 1] < -band)

    def admits(self, deviation: np.ndarray, oriented: np.ndarray, /) -> np.ndarray:
        """Admit issued bounds under the original tolerance and the source tube."""
        return (
            oriented
            & np.isfinite(deviation)
            & (deviation <= self.tolerance)
            & (deviation <= self.profile.tube_radius)
        )

    def certify(
        self,
        triangles: np.ndarray,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Whole-facet deviations of outward (k, 3, 3) triangles and their admission."""
        deviation, oriented = self.bounds(triangles, budget=budget)
        return deviation, self.admits(deviation, oriented)


def _oriented_facet_bytes(triangle: np.ndarray, /) -> bytes:
    """Canonicalize cyclic corner order, preserving all bits and orientation.

    A cyclic rotation has the same closed affine triangle, exact centroid
    and oriented area vector. Whole-facet and centroid enclosures therefore
    prove the same obligations; a reversal or any changed corner does not.
    """
    raw = triangle.tobytes()
    return min(raw, raw[24:] + raw[:24], raw[48:] + raw[:48])


def _plc_native_constraints(
    mesh: CellMesh,
    source: _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource,
    faces: np.ndarray,
    segments: np.ndarray,
    protected: np.ndarray,
    edge_classes: ArrayLike | None,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TetMeshSourceComplex]:
    """Rebuild source IDs from predicate-certified membership in declared PLC entities."""
    prepared = source.prepare(mesh.coordinates, budget=budget)
    points = np.asarray(source.source_points, dtype=np.float64)
    exact_work = _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcBudget(
        source.maximum_work, source.maximum_bits, budget=budget
    )
    bank_upper, _ = (
        _dependency_phydrax_discretization__exact_plc_geometry.plc_storage_bounds(
            points.shape[0], 1075
        )
    )
    _, scratch_upper = (
        _dependency_phydrax_discretization__exact_plc_geometry.plc_storage_bounds(
            0, source.maximum_bits
        )
    )
    row_upper = (
        512 + 88 * source.source_triangles.shape[0] + 80 * source.source_segments.shape[0]
    )
    with budget.temporary_scope():
        budget.reserve(0, bank_upper + scratch_upper + row_upper)
        source_points = tuple(plc_rational_point(point) for point in points)
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
            if slot % 64 == 0:
                budget.ticket.checkpoint()
            for row, corners in enumerate(triangles):
                budget.reserve_geometry_queries(1)
                exact_work.charge(
                    192, tuple(value for point in corners for value in point)
                )
                if all(
                    _dependency_phydrax_discretization__exact_plc_geometry.in_entity(
                        prepared.vertices[index], corners
                    )
                    for index in face
                ):
                    selected_faces[slot] = triangle_ids[row]
                    break
            else:
                raise ValueError(
                    "tetra_metric.constrained_faces must belong to declared PLC source rows; got an unassigned face"
                )
        selected_segments: list[np.ndarray] = []
        selected_ids: list[int] = []
        keys = entity_keys_owner.entity_keys(mesh, 1)
        global_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        classes = _labels(edge_classes, keys.shape[0], "edge_classes")
        for slot, segment in enumerate(segments):
            if slot % 64 == 0:
                budget.ticket.checkpoint()
            for row, corners in enumerate(lines):
                budget.reserve_geometry_queries(1)
                exact_work.charge(
                    128, tuple(value for point in corners for value in point)
                )
                if all(
                    _dependency_phydrax_discretization__exact_plc_geometry.in_entity(
                        prepared.vertices[index], corners
                    )
                    for index in segment
                ):
                    selected_segments.append(segment)
                    selected_ids.append(int(original_segment_ids[row]))
                    break
            else:
                slot = entity_keys_owner.key_rows(
                    keys, np.sort(global_ids[segment])[None]
                )[0]
                if protected[slot] or classes[slot] != 0:
                    raise ValueError(
                        "tetra_metric.protected_edges must belong to declared PLC source segments; got an unassigned edge"
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
    kind = lineage_operation_kind(
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
    identifiers = fresh_identifiers(ancestry.next_cell_id, len(new_keys))
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
    """Retain immutable affine source tables for tensor sampling.

    Supplied identifiers and vertex-row order remain fixed. PLC geometry binds
    rational source coordinates; otherwise the binary64 affine path retains
    its floating-point operation order.
    """

    points: np.ndarray
    cells: np.ndarray
    vertex_global_ids: np.ndarray
    cell_global_ids: np.ndarray
    source_geometry: (
        _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec | None
    ) = None
    budget: InitVar[
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    ] = field(kw_only=True)
    corners: np.ndarray = field(init=False, repr=False)

    def __post_init__(
        self,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(
                f"TetraMetricSource.points must have three coordinate columns; got shape={self.points.shape}"
            )
        if self.cells.ndim != 2 or self.cells.shape[1] != 4 or not self.cells.shape[0]:
            raise ValueError(
                f"TetraMetricSource.cells must contain nonempty tetrahedral rows; got shape={self.cells.shape}"
            )
        if (
            not np.issubdtype(self.cells.dtype, np.integer)
            or np.any(self.cells < 0)
            or np.any(self.cells >= self.points.shape[0])
        ):
            raise ValueError(
                f"TetraMetricSource.cells must index source points; got shape={self.cells.shape}"
            )
        if self.vertex_global_ids.shape != (
            self.points.shape[0],
        ) or self.cell_global_ids.shape != (self.cells.shape[0],):
            raise ValueError(
                "TetraMetricSource.identifiers must align with source rows; got incompatible lengths"
            )
        if self.source_geometry is not None:
            if not isinstance(
                self.source_geometry,
                _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec,
            ):
                raise TypeError(
                    f"TetraMetricSource.source_geometry must be CellGeometrySpec or None; got {type(self.source_geometry).__name__}"
                )
            if not isinstance(
                self.source_geometry.exact_source,
                _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource,
            ):
                raise ValueError(
                    "TetraMetricSource.source_geometry must retain an ExactPlcCellGeometrySource; got another source kind"
                )
            bank = self.source_geometry.source_coordinates(budget=budget)
            if len(bank) != self.points.shape[0] or any(
                len(point) != 3 for point in bank
            ):
                raise ValueError(
                    "TetraMetricSource.source_coordinates must align with source rows; got incompatible coordinates"
                )

            # The temporary source object bank and gathered corner slots
            # coexist. The immutable Fraction values and corner table remain
            # owned by the ledger for the entire metric-source lifetime.
            with _owner_phydrax__geometry_predicates.exact_charge(
                0, 8 * (3 * len(bank) + 3 * self.cells.size), budget=budget
            ):
                corners = np.asarray(bank, dtype=object)[self.cells]
                budget.retain_basis((bank, corners))
                object.__setattr__(self, "corners", corners)
            return
        object.__setattr__(self, "corners", self.points[self.cells])

    @classmethod
    def from_mesh(
        cls,
        mesh: CellMesh,
        /,
        *,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    ) -> TetraMetricSource:
        return cls(
            np.asarray(mesh.coordinates, dtype=np.float64),
            np.concatenate(
                [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
            ),
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            ),
            budget=budget,
        )


@contextmanager
def _exact_table_boxes(
    values: np.ndarray,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Return outward boxes of rational rows using the shared rounding functions."""

    bits = _owner_phydrax__geometry_predicates.exact_bits(values, budget=budget)
    # Each min/max comparison cross-multiplies two b-bit rational inputs;
    # two products and seven division/comparison temporaries fit 2b+1 bits.
    # Both float buffers persist until the BVH has consumed them.
    storage = 48 * values.shape[0] + 7 * _owner_phydrax__geometry_predicates.bigint_bytes(
        2 * bits + 1
    )
    with _owner_phydrax__geometry_predicates.exact_charge(
        4 * values.size + 6 * values.shape[0], storage, budget=budget
    ):
        lower = np.empty((values.shape[0], 3), dtype=np.float64)
        upper = np.empty_like(lower)
        for row, corners in enumerate(values):
            for axis in range(3):
                lower[row, axis] = fraction_endpoint(
                    min(corner[axis] for corner in corners), upper=False, saturate=True
                )
                upper[row, axis] = fraction_endpoint(
                    max(corner[axis] for corner in corners), upper=True, saturate=True
                )
        yield lower, upper


def _metric_source_bvh(
    tables: TetraMetricSource,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> PackedBVH:
    if tables.source_geometry is None:
        return prepare_bvh(
            np.min(tables.corners, axis=1),
            np.max(tables.corners, axis=1),
            dtype=jnp.float64,
        )
    with _exact_table_boxes(tables.corners, budget=budget) as (lower, upper):
        return prepare_bvh(lower, upper, dtype=jnp.float64)


def _exact_source_weights(
    corners: np.ndarray,
    point: _dependency_phydrax_discretization__exact_plc_geometry.ExactPoint,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:

    with _owner_phydrax__geometry_predicates.exact_charge(0, 8 * 15, budget=budget):
        bits = _owner_phydrax__geometry_predicates.exact_bits(
            np.asarray((*corners, point), dtype=object), budget=budget
        )
    # Nine source-edge and three query differences each have numerator and
    # denominator bounded by 2b+1 bits. Elimination, minor bounds, and
    # arithmetic operation counts belong to the shared rational linear owner.
    with _owner_phydrax__geometry_predicates.exact_charge(
        12,
        12 * (2 * _owner_phydrax__geometry_predicates.bigint_bytes(2 * bits + 1) + 64),
        budget=budget,
    ):
        matrix = tuple(
            tuple(corners[column + 1, axis] - corners[0, axis] for column in range(3))
            for axis in range(3)
        )
        right = tuple((point[axis] - corners[0, axis],) for axis in range(3))
        solved = _owner_phydrax_linalg__small_batched.prepare_exact_small_linear_actions(
            matrix, right, budget=budget
        )
        if solved.actions is None or solved.determinant <= 0:
            raise ValueError(
                "TetraMetricSource.cells must have positive rational determinants; got a nonpositive source cell"
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
    exact_points: tuple[
        _dependency_phydrax_discretization__exact_plc_geometry.ExactPoint, ...
    ]
    | None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select immutable source cells and sample the tensor in source coordinates."""

    bank = (
        tuple(plc_rational_point(row) for row in rounded)
        if exact_points is None
        else exact_points
    )
    if len(bank) != rounded.shape[0]:
        raise ValueError(
            "tetra_metric.query_witnesses must align with carrier rows; got incompatible lengths"
        )
    tree = (
        _metric_source_bvh(tables, budget=budget)
        if prepared_bvh is None
        else prepared_bvh
    )
    with _owner_phydrax__geometry_predicates.exact_charge(
        0, 8 * 3 * len(bank), budget=budget
    ):
        with _exact_table_boxes(
            np.asarray(bank, dtype=object)[:, None, :], budget=budget
        ) as (
            lower,
            upper,
        ):
            queries = prepare_bvh(lower, upper, dtype=jnp.float64)
    selected = np.full((len(bank),), -1, dtype=np.int64)
    selected_ids = np.full((len(bank),), np.iinfo(np.int64).max, dtype=np.int64)
    weights = np.zeros((len(bank), 4), dtype=np.float64)
    work = 0
    for cells, points in bvh_overlap_pair_blocks(tree, queries, include_touching=True):
        budget.ticket.checkpoint()
        for cell, query in zip(cells, points, strict=True):
            work += 1
            if work_counter is not None:
                work_counter[0] += 1
            if work > maximum_pairs:
                raise source_location_budget(maximum_pairs, work)
            coefficients = _exact_source_weights(
                tables.corners[cell], bank[query], budget=budget
            )
            if (
                min(coefficients) < 0
                or tables.cell_global_ids[cell] >= selected_ids[query]
            ):
                continue
            selected[query], selected_ids[query] = cell, tables.cell_global_ids[cell]
            weights[query] = tuple(float(value) for value in coefficients)
    covered = selected >= 0
    if require_coverage and np.any(~covered):
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "tetra_metric.source_location must find a predicate-certified source host; got an uncovered target point",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    weights[~covered, 0] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    return (
        tables.vertex_global_ids[tables.cells[np.maximum(selected, 0)]],
        weights,
        np.broadcast_to(covered[:, None], weights.shape),
    )


@checked
def locate_stencil(
    mesh: CellMesh,
    points: np.ndarray,
    maximum_pairs: int,
    /,
    *,
    require_coverage: bool = True,
    prepared_bvh: PackedBVH | None = None,
    work_counter: list[int] | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate points in affine tetrahedra with a BVH bound to source row order."""
    return _locate_source_stencil(
        TetraMetricSource.from_mesh(mesh, budget=budget),
        points,
        maximum_pairs,
        require_coverage=require_coverage,
        prepared_bvh=prepared_bvh,
        work_counter=work_counter,
        budget=budget,
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
    exact_points: tuple[
        _dependency_phydrax_discretization__exact_plc_geometry.ExactPoint, ...
    ]
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Locate points with the affine kernel bound to immutable source rows."""
    if work_counter is not None and (
        len(work_counter) != 1
        or isinstance(work_counter[0], bool)
        or not isinstance(work_counter[0], int)
        or work_counter[0] < 0
    ):
        raise ValueError(
            "tetra_metric.work_counter must contain one nonnegative integer; got an invalid counter bank"
        )
    if tables.source_geometry is not None:
        return _locate_exact_source_stencil(
            tables,
            points,
            maximum_pairs,
            require_coverage=require_coverage,
            prepared_bvh=prepared_bvh,
            work_counter=work_counter,
            exact_points=exact_points,
            budget=budget,
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
        # Bucket padding repeats query zero; only unpadded query pairs are charged.
        actual = block_query < points.shape[0]
        cell, query = block_cell[actual], block_query[actual]
        if not cell.size:
            continue
        work += cell.size
        if work_counter is not None:
            work_counter[0] += cell.size
        if work > maximum_pairs:
            raise source_location_budget(maximum_pairs, work)
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
            valid &= signs >= _dependency_phydrax__geometry_predicates.PredicateSign.ZERO
            coefficients[
                signs == _dependency_phydrax__geometry_predicates.PredicateSign.ZERO, slot
            ] = 0.0
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
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.LINEAGE_FAILED,
            "A metric target vertex lacks an exactly certified source host.",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.LINEAGE_CONSTRUCTION.value,
        )
    weights = np.maximum(weights, 0.0)
    weights[~covered, 0] = 1.0
    weights /= np.sum(weights, axis=1, keepdims=True)
    return (
        ids[cells[np.maximum(selected, 0)]],
        weights,
        np.broadcast_to(covered[:, None], weights.shape),
    )


def _collapse_constrained_faces(
    parents: dict[tuple[int, ...], dict[tuple[int, ...], int]],
    removed: int,
    kept: int,
    /,
) -> dict[tuple[int, ...], dict[tuple[int, ...], int]]:
    """Take the accepted collapse's simplicial image, dropping degenerate faces."""
    result: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
    for face, sources in parents.items():
        image = tuple(sorted(kept if vertex == removed else vertex for vertex in face))
        if len(set(image)) != len(image):
            continue
        retained = result.setdefault(image, {})
        for source, kind in sources.items():
            inherited = int(lineage_operation_kind(
                EntityLineageKind.COLLAPSED_INTO, (kind,)
            )) if removed in face else kind
            retained[source] = int(lineage_operation_kind(
                EntityLineageKind(inherited), (retained[source],)
            )) if source in retained else inherited
    return result


def _insert_constrained_faces(
    parents: dict[tuple[int, ...], dict[tuple[int, ...], int]],
    before_faces: np.ndarray,
    before_sources: np.ndarray,
    after_faces: np.ndarray,
    after_sources: np.ndarray,
    removed_cells: np.ndarray,
    proposed_cells: np.ndarray,
    inserted: int,
    /,
) -> dict[tuple[int, ...], dict[tuple[int, ...], int]]:
    """Use the committed conflict cavity's crossed-face rim ancestry.

    Native ``TetMesh::collect`` reconstructs each constrained face as
    ``(inserted, rim_edge)``. The rim edge occurs once among crossed faces
    of that source; internal edges cancel. The crossed face owning that
    edge, not the initially proposed split edge or a point stencil, owns
    the new face's construction ancestry.
    """
    before = {
        tuple(sorted(face)): int(source)
        for face, source in zip(before_faces.tolist(), before_sources, strict=True)
    }
    after = {
        tuple(sorted(face)): int(source)
        for face, source in zip(after_faces.tolist(), after_sources, strict=True)
    }
    if (
        len(before) != len(before_faces) or len(after) != len(after_faces)
        or before.keys() != parents.keys()
    ):
        raise ValueError("tetra_metric.insert_face_lineage requires unique complete pre/post face snapshots")
    removed = before.keys() - after.keys()
    added = after.keys() - before.keys()
    old_facets = {
        tuple(sorted(face))
        for cell in removed_cells.tolist() for face in combinations(cell, 3)
    }
    new_facets = {
        tuple(sorted(face))
        for cell in proposed_cells.tolist() for face in combinations(cell, 3)
    }
    if not removed <= old_facets or not added <= new_facets:
        raise ValueError("tetra_metric.insert_face_lineage faces must belong to the committed cavity")
    rim: dict[tuple[tuple[int, ...], int], list[tuple[int, ...]]] = {}
    for face in removed:
        for edge in combinations(face, 2):
            rim.setdefault((edge, before[face]), []).append(face)
    result = {face: parents[face] for face in before.keys() & after.keys()}
    if any(before[face] != after[face] for face in result):
        raise ValueError("tetra_metric.insert_face_lineage cannot change a surviving face source")
    for face in added:
        edge = tuple(vertex for vertex in face if vertex != inserted)
        owners = rim.get((edge, after[face]), ())
        if len(edge) != 2 or len(owners) != 1:
            raise ValueError(
                f"tetra_metric.insert_face_lineage requires one crossed face owning the committed rim edge; got face={face!r}, source={after[face]}, parents={owners!r}"
            )
        result[face] = {
            source: int(lineage_operation_kind(EntityLineageKind.REFINED_FROM, (kind,)))
            for source, kind in parents[owners[0]].items()
        }
    return result


def _constrained_face_relations(
    relations: _dependency_phydrax_meshing__topology_edit.EntityRelations,
    snapshot: TetMeshArrays,
    identifiers: np.ndarray,
    parents: dict[tuple[int, ...], dict[tuple[int, ...], int]],
    /,
) -> _dependency_phydrax_meshing__topology_edit.EntityRelations:
    """Publish accepted face ancestry independently of interpolation support."""
    faces = tuple(tuple(sorted(face)) for face in snapshot.faces.tolist())
    if len(set(faces)) != len(faces) or set(faces) != parents.keys():
        raise ValueError(
            "tetra_metric.constrained_face_lineage must cover exactly the current constrained faces"
        )
    targets = np.sort(identifiers[snapshot.faces], axis=1)
    target_keys = {tuple(row) for row in targets.tolist()}
    keep = np.asarray(
        [tuple(row) not in target_keys for row in relations.target_keys.tolist()],
        dtype=np.bool_,
    )
    sources, descendants, kinds = [], [], []
    for face, target in zip(faces, targets.tolist(), strict=True):
        if not parents[face]:
            raise ValueError("tetra_metric.constrained_face_lineage requires declared source parents")
        for source, kind in sorted(parents[face].items()):
            sources.append(source)
            descendants.append(target)
            kinds.append(kind)
    return _dependency_phydrax_meshing__topology_edit.EntityRelations(
        2,
        np.concatenate((relations.source_keys[keep], np.asarray(sources, dtype=np.int64).reshape(-1, 3))),
        np.concatenate((relations.target_keys[keep], np.asarray(descendants, dtype=np.int64).reshape(-1, 3))),
        np.concatenate((relations.kinds[keep], np.asarray(kinds, dtype=np.int32))),
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
    exact_points: tuple[
        _dependency_phydrax_discretization__exact_plc_geometry.ExactPoint, ...
    ]
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[_dependency_phydrax_meshing__topology_edit.CellTopologyEdit, np.ndarray]:
    live = np.flatnonzero(snapshot.vertex_dimension >= 0)
    row_of = np.full((snapshot.points.shape[0],), -1, dtype=np.int32)
    row_of[live] = np.arange(live.size, dtype=np.int32)
    source_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    ids = np.concatenate(
        (
            source_ids,
            fresh_identifiers(
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
    blocks = _dependency_phydrax_meshing__topology_edit.source_family_blocks(
        tuple((block.name, block.cell_kind) for block in mesh.blocks),
        tuple(cells[owners == index] for index in range(len(mesh.blocks))),
        tuple(cell_ids[owners == index] for index in range(len(mesh.blocks))),
    )
    stencil = (
        locate_stencil(mesh, snapshot.points[live], maximum_pairs, budget=budget)
        if source_tables is None or source_tables.source_geometry is None
        else _locate_source_stencil(
            source_tables,
            snapshot.points[live],
            maximum_pairs,
            exact_points=exact_points,
            budget=budget,
        )
    )
    relations = []
    live_ids = ids[live]
    collapsed = resolved_collapses(ids, source_ids.size, ancestry.vertex_successor)
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
            relation = subentity_relations(
                mesh,
                dimension,
                snapshot.tetrahedra,
                ids,
                ancestry.vertex_sources,
                collapsed_vertices=collapsed,
                compose_collapses=ancestry.curved_collapses > 0,
            )
            if dimension == 2 and ancestry.constrained_faces is not None:
                relation = _constrained_face_relations(
                    relation, snapshot, ids, ancestry.constrained_faces
                )
            relations.append(relation)
            continue
        width = 1 if dimension in (0, 3) else dimension + 1
        pairs = vertex_pairs if dimension == 0 else cell_pairs
        if pairs:
            values = np.unique(np.asarray(pairs, dtype=np.int64), axis=0)
            # Transient targets are not part of this accepted edit.
            valid = np.isin(values[:, 1], live_ids if dimension == 0 else cell_ids)
            values = values[valid]
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
                    dimension, values[:, :1], values[:, 1:2], values[:, 2].astype(np.int8)
                )
            )
        else:
            relations.append(
                _dependency_phydrax_meshing__topology_edit.EntityRelations(
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
    return _dependency_phydrax_meshing__topology_edit.CellTopologyEdit(
        operation, snapshot.points[live], live_ids, blocks, *stencil, tuple(relations)
    ), metric[live]


_OperationResult = TypeVar("_OperationResult")


@contextmanager
def _metric_coordinate_phase(
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    /,
    *,
    maximum_work_units: int,
    maximum_cavity_cells: int,
) -> Iterator[NativeBatchTicket]:
    """Reserve a bounded host/native phase and retain its request counters."""
    remaining = scope.remaining()
    storage = budget.storage_ticket
    retained_bytes = 0 if storage is None else storage.bytes_upper
    ticket = scope.batch(
        min(
            maximum_work_units,
            max(0, budget.maximum_work_units - budget.work_units),
            remaining.remaining_work_units,
        ),
        queries_upper=remaining.remaining_geometry_queries,
        bytes_upper=min(
            max(0, budget.maximum_memory_bytes - retained_bytes),
            remaining.remaining_managed_bytes,
        ),
        cavity_upper=min(maximum_cavity_cells, remaining.remaining_cavity_cells),
    )
    with budget.measurement(), budget.bind_ticket(ticket):
        with ticket.native_target():
            yield ticket


@dataclass(slots=True)
class _NativeMetricBudget:
    """Cumulative native work, source-pair and per-cavity epoch allowances."""

    native: TetMesh3D
    maximum_work_units: int
    maximum_cavity_work: int
    source_tables: TetraMetricSource
    maximum_source_pairs: int
    scope: NativeExecutionScope
    coordinate_budget: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    )
    native_work_units: int = 0
    coordinate_work_units: int = 0
    phase_ticket: NativeBatchTicket | None = None
    optimizer_storage: ManagedBytesContributor | None = None
    band_snapshot: TetMeshArrays | None = None
    band_incidence: dict[int, int] | None = None
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
    operation_started: float | None = None
    maximum_wall_seconds: float | None = None
    # Issued curved-source evidence of this epoch's candidates, keyed by the
    # source digest and the cyclic-canonical oriented corner bytes:
    # whole-facet bounds (deviation, outward orientation) and certified
    # facet-point field intervals. Admission and refutation are recomputed
    # from the source policy on every use; equal facets share issued bounds.
    curved_bounds: dict[tuple[bytes, bytes], tuple[float, bool]] = field(
        default_factory=dict
    )
    curved_witnesses: dict[tuple[bytes, bytes], tuple[float, float]] = field(
        default_factory=dict
    )
    curved_storage: (
        _dependency_phydrax_discretization__coordinate_budget.CoordinateLiveStorage | None
    ) = None
    authenticated_curved: tuple[CurvedBoundarySource, bytes] | None = None

    @contextmanager
    def phase(self) -> Iterator[None]:
        """Share one explicit reservation across a bounded epoch phase."""
        if self.phase_ticket is not None:
            yield
            return
        with _metric_coordinate_phase(
            self.scope,
            self.coordinate_budget,
            maximum_work_units=max(0, self.maximum_work_units - self.work_units),
            maximum_cavity_cells=self.maximum_cavity_cells,
        ) as ticket:
            with ticket.managed_storage() as storage:
                self.phase_ticket = ticket
                self.optimizer_storage = storage
                try:
                    yield
                finally:
                    self.band_snapshot = None
                    self.band_incidence = None
                    self.optimizer_storage = None
                    self.phase_ticket = None

    def commit_band(
        self, snapshot: TetMeshArrays, removed: np.ndarray, proposed: np.ndarray, /
    ) -> None:
        """Advance phase-local incidence with the accepted cavity delta."""
        incidence = self.band_incidence
        if incidence is None:
            return
        if not self.optimizer_work(
            6 * (removed.shape[0] + proposed.shape[0]),
            self.optimizer_scratch_required_bytes,
        ):
            self.band_snapshot = None
            self.band_incidence = None
            return
        for cells, sign in ((removed, -1), (proposed, 1)):
            keys, counts = _edge_incidence(cells)
            for key, count in zip(keys.tolist(), counts.tolist(), strict=True):
                remaining = incidence.get(key, 0) + sign * count
                if remaining < 0:
                    raise RuntimeError(
                        f"tetra_metric.band_incidence must remain nonnegative; got edge={key}, incidence={remaining}"
                    )
                if remaining:
                    incidence[key] = remaining
                else:
                    del incidence[key]
        self.band_snapshot = snapshot

    def admit_optimizer_bound(self, units: int, /) -> bool:
        """Admit a dispatch against both the epoch and its live shared phase."""
        self.check_deadline()
        if self.exhausted:
            return False
        remaining = min(
            self.maximum_work_units - self.work_units,
            self.coordinate_budget.ticket.remaining_work_units,
        )
        if units > remaining:
            self.refuse(
                "The shared metric phase work allowance is exhausted.",
                (("maximum_work_units", float(self.maximum_work_units)),),
                (
                    ("work_units", float(self.work_units)),
                    ("remaining_phase_work_units", float(remaining)),
                    ("next_work_units", float(units)),
                ),
            )
            return False
        return True

    def check_deadline(self) -> None:
        """Check host preparation and dispatch against the request clock."""
        if self.maximum_wall_seconds is None:
            return
        started = (
            self.scope.operation_started
            if self.operation_started is None
            else self.operation_started
        )
        elapsed = monotonic() - started
        if elapsed > self.maximum_wall_seconds:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.TIMED_OUT,
                "Native meshing exceeded its enforced wall-time limit.",
                stage=_dependency_phydrax_meshing__trace.MeshingStageKind.VOLUME_FILL.value,
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
        return (
            self.native_work_units + self.optimizer_host_work + self.coordinate_work_units
        )

    def optimizer_scratch(self, scratch_bytes: int, /) -> bool:
        """Admit a bounded host-table reservation before allocating the table."""
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
        storage = self.optimizer_storage
        if storage is None:
            raise RuntimeError(
                "tetra_metric.optimizer_storage must belong to an admitted phase; got None"
            )
        ticket = self.phase_ticket
        if ticket is None:
            raise RuntimeError(
                "tetra_metric.phase_ticket must belong to an admitted phase; got None"
            )
        available = storage.bound + ticket.remaining_managed_bytes
        if scratch_bytes > available:
            self.refuse(
                "The statistical optimizer's scratch plan exceeds the live phase allowance.",
                (("optimizer_scratch_required_bytes", float(scratch_bytes)),),
                (("remaining_optimizer_scratch_bytes", float(available)),),
            )
            return False

        storage.set_bound(max(storage.bound, scratch_bytes))
        return True

    def optimizer_work(
        self, units: int, scratch_bytes: int, /, *, evaluation: bool = True
    ) -> bool:
        """Reserve a finite host proposal against the shared total-work cap.

        ``evaluation=False`` charges a later stage of an already counted
        proposal evaluation.
        """
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
        ticket = self.coordinate_budget.ticket

        if not self.admit_optimizer_bound(units):
            return False
        ticket.consume(units)
        self.optimizer_host_work += units
        self.optimizer_evaluations += int(evaluation)
        return True

    def call(
        self, operation: Callable[[int], _OperationResult], /
    ) -> _OperationResult | None:
        self.check_deadline()
        if self.exhausted:
            return None
        ticket = self.phase_ticket
        if ticket is None:
            raise RuntimeError(
                "tetra_metric.phase_ticket must belong to an admitted phase; got None"
            )
        allowance = min(
            self.maximum_cavity_work,
            self.maximum_work_units - self.work_units,
            ticket.remaining_work_units,
        )
        if allowance <= 0:
            self.refuse(
                "The cumulative native/host metric work allowance is exhausted.",
                (("maximum_work_units", float(self.maximum_work_units)),),
                (
                    ("work_units", float(self.work_units)),
                    (
                        "native_work_units",
                        float(self.native_work_units),
                    ),
                    ("optimizer_host_work_units", float(self.optimizer_host_work)),
                ),
            )
            return None

        before = ticket.native_work_units
        try:
            try:
                self.native.set_work_limit(allowance)
                result = operation(allowance)
                self.check_deadline()
                return result
            finally:
                # The native-only local delta excludes host rational arithmetic
                # and source predicates outside this mesh-operation callback.
                self.native_work_units += ticket.native_work_units - before
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
                    ("native_work_units", float(self.native_work_units)),
                    ("optimizer_host_work_units", float(self.optimizer_host_work)),
                ),
            )
            return None

    def stencil(
        self,
        points: np.ndarray,
        /,
        *,
        exact_points: tuple[
            _dependency_phydrax_discretization__exact_plc_geometry.ExactPoint, ...
        ]
        | None = None,
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
        self.coordinate_budget.reserve_geometry_queries(points.shape[0])
        if self.source_bvh is None:
            self.source_bvh = _metric_source_bvh(
                self.source_tables, budget=self.coordinate_budget
            )
        try:
            return _locate_source_stencil(
                self.source_tables,
                points,
                remaining,
                require_coverage=False,
                prepared_bvh=self.source_bvh,
                work_counter=self.source_candidates,
                exact_points=exact_points,
                budget=self.coordinate_budget,
            )
        except _leaf_phydrax_meshing__failures.MeshingFailure as error:
            if (
                error.evidence.category
                is not _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED
            ):
                raise
            self.refuse(
                error.evidence.message, error.evidence.requested, error.evidence.achieved
            )
            return None

    def _charge_curved(
        self,
        issue: Callable[
            [_dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget],
            _OperationResult,
        ],
        /,
    ) -> _OperationResult | None:
        """Charge one host curved-source issuance to the shared epoch work."""
        budget = self.coordinate_budget
        before = budget.work_units
        try:
            result = issue(budget)
        except _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError as error:
            self.coordinate_work_units += budget.work_units - before
            self.refuse(
                f"The curved whole-facet source proof exceeds its allowance: {error}",
                (("maximum_work_units", float(self.maximum_work_units)),),
                (("work_units", float(self.work_units)),),
            )
            return None
        self.coordinate_work_units += budget.work_units - before
        if self.work_units > self.maximum_work_units:
            self.refuse(
                "The curved whole-facet source proofs exhaust the metric work allowance.",
                (("maximum_work_units", float(self.maximum_work_units)),),
                (("work_units", float(self.work_units)),),
            )
            return None
        return result

    def _admit_curved_cache(self, count: int, /) -> bool:
        """Preadmit both proof tables and the next bounded candidate batch."""
        storage = self.curved_storage
        if storage is None:
            raise RuntimeError(
                "tetra_metric.curved_storage must belong to the active epoch; got None"
            )
        # Each row owns its oriented 72-byte corners, key/result tuples,
        # scalar bounds and dict capacity. Both tables and the candidate
        # batch coexist; the lease lasts across phases, not a fresh quota.
        rows = len(self.curved_bounds) + len(self.curved_witnesses) + count

        def reserve(
            _: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
        ) -> bool:
            storage.set_bound(1024 + 512 * rows)
            return True

        return self._charge_curved(reserve) is not None

    def _authenticate_curved(self, curved: CurvedBoundarySource, /) -> None:
        """Authenticate once, then rebind every use to all current source facts."""
        authenticated = self.authenticated_curved
        digest = curved._integrity_digest()
        if authenticated is None or authenticated[0] is not curved or digest is None:
            curved.validate_source_integrity()
            self.authenticated_curved = None if digest is None else (curved, digest)
        elif digest != authenticated[1]:
            raise ValueError(
                "CurvedBoundarySource.digest must bind the current source and policy; got changed source claims"
            )

    def refute_curved(
        self, curved: CurvedBoundarySource, triangles: np.ndarray, /
    ) -> np.ndarray | None:
        """Refute candidate facets by certified facet-point witnesses before any bound."""
        self.check_deadline()
        if self.exhausted:
            return None
        self._authenticate_curved(curved)
        if not self._admit_curved_cache(triangles.shape[0]):
            return None
        corners = np.ascontiguousarray(triangles, dtype=np.float64)
        keys = [(curved.digest, _oriented_facet_bytes(row)) for row in corners]
        fresh = [index for index, key in enumerate(keys) if key not in self.curved_witnesses]
        if fresh:
            witnesses = self._charge_curved(
                lambda budget: curved._witnesses_authenticated(corners[fresh], budget=budget)
            )
            if witnesses is None:
                return None
            for index, (low, high) in zip(fresh, witnesses.tolist(), strict=True):
                self.curved_witnesses[keys[index]] = (low, high)
        return curved.refutes(
            np.asarray([self.curved_witnesses[key] for key in keys], dtype=np.float64).reshape(
                (-1, 2)
            )
        )

    def certify_curved(
        self, curved: CurvedBoundarySource, triangles: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Charge host whole-facet bounds of unbounded facets to the epoch work."""
        self.check_deadline()
        if self.exhausted:
            return None
        self._authenticate_curved(curved)
        if not self._admit_curved_cache(triangles.shape[0]):
            return None
        corners = np.ascontiguousarray(triangles, dtype=np.float64)
        keys = [(curved.digest, _oriented_facet_bytes(row)) for row in corners]
        fresh = [index for index, key in enumerate(keys) if key not in self.curved_bounds]
        if fresh:
            issued = self._charge_curved(
                lambda budget: curved._bounds_authenticated(corners[fresh], budget=budget)
            )
            if issued is None:
                return None
            for index, value, outward in zip(
                fresh, issued[0].tolist(), issued[1].tolist(), strict=True
            ):
                self.curved_bounds[keys[index]] = (value, outward)
        bounds = [self.curved_bounds[key] for key in keys]
        deviation = np.asarray([value for value, _ in bounds], dtype=np.float64)
        oriented = np.asarray([outward for _, outward in bounds], dtype=np.bool_)
        return deviation, curved.admits(deviation, oriented)


@dataclass(frozen=True, slots=True)
class UniformSizeRemeshingGoal:
    """Retain physical size obligations for an admitted hard background.

    ``minimum_relative_determinant`` is the publication validity policy's
    relative determinant floor. Accepted cells must exceed the floor times
    their vertex-edge product under the determinant predicate.
    """

    control: _dependency_phydrax_meshing__sizing.UniformSizeControl
    policy: _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
    minimum_relative_determinant: float
    maximum_radius_edge: float | None = None
    minimum_dihedral_degrees: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.control, _dependency_phydrax_meshing__sizing.UniformSizeControl
        ):
            raise TypeError(
                f"UniformSizeRemeshingGoal.control must be UniformSizeControl; got {type(self.control).__name__}"
            )
        if (
            self.control.strength
            is not _dependency_phydrax_meshing__sizing.SizeControlStrength.HARD
        ):
            raise ValueError(
                f"UniformSizeRemeshingGoal.control.strength must be HARD; got {self.control.strength!r}"
            )
        if not isinstance(
            self.policy, _dependency_phydrax_meshing__sizing.SizeCompliancePolicy
        ):
            raise TypeError(
                f"UniformSizeRemeshingGoal.policy must be SizeCompliancePolicy; got {type(self.policy).__name__}"
            )
        if not np.isfinite(self.minimum_relative_determinant) or not (
            0.0 <= self.minimum_relative_determinant < 1.0
        ):
            raise ValueError(
                f"UniformSizeRemeshingGoal.minimum_relative_determinant must lie in [0, 1); got {self.minimum_relative_determinant!r}"
            )
        if self.maximum_radius_edge is not None and (
            not np.isfinite(self.maximum_radius_edge) or self.maximum_radius_edge <= 0.0
        ):
            raise ValueError(
                f"UniformSizeRemeshingGoal.maximum_radius_edge must be finite and positive; got {self.maximum_radius_edge!r}"
            )
        if self.minimum_dihedral_degrees is not None and (
            not np.isfinite(self.minimum_dihedral_degrees)
            or not 0.0 <= self.minimum_dihedral_degrees <= 180.0
        ):
            raise ValueError(
                f"UniformSizeRemeshingGoal.minimum_dihedral_degrees must lie in [0, 180]; got {self.minimum_dihedral_degrees!r}"
            )


def physical_size_compliance(
    snapshot: TetMeshArrays,
    goal: UniformSizeRemeshingGoal,
    /,
    *,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], list[str]]:

    lengths, growth = (
        _owner_phydrax_meshing_providers__native_publication.edge_size_evidence(
            snapshot.points,
            _owner_phydrax_meshing_providers__native_publication.unique_edges(
                snapshot.tetrahedra, "tetrahedron"
            ),
            geometry=geometry,
            mesh=None
            if geometry is None
            else CellMesh.from_tetrahedra(snapshot.points, snapshot.tetrahedra),
            budget=budget,
        )
    )
    return _owner_phydrax_meshing_providers__native_publication.uniform_size_compliance(
        goal.control, goal.policy, lengths, growth
    )


def _physical_size_goal_met(
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    goal: UniformSizeRemeshingGoal | None,
    minimum_quality: float,
    /,
    *,
    geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
    | None = None,
) -> bool:
    if (
        goal is None
        or physical_size_compliance(snapshot, goal, geometry=geometry, budget=budget)[2]
    ):
        return False
    return bool(
        np.all(_quality(snapshot.points, metric, snapshot.tetrahedra) >= minimum_quality)
    )


def _preferred_size_fraction(target: float, delta: np.ndarray, /) -> float:
    """Condition the target fraction from squared physical distance.

    Binary power-of-two scaling avoids overflow/underflow in intermediate
    squares and a rounded square-root followed by a second rounded division.
    Native construction still certifies represented incidence and admission.
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
    """Return sorted tetrahedral edge keys and finite-cell incidence counts."""
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
    """Gather retained rows together with an appended or relocated vertex value."""
    if changed_vertex is None:
        return values[rows]
    if changed_value is None:
        raise ValueError(
            "tetra_metric.changed_value must accompany a changed vertex; got None"
        )
    result = np.empty((*rows.shape, *values.shape[1:]), dtype=values.dtype)
    changed = rows == changed_vertex
    if np.any((rows[~changed] < 0) | (rows[~changed] >= values.shape[0])):
        raise ValueError(
            "tetra_metric.proposal_rows must index retained or proposed vertices; got an invalid row"
        )
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
    """Return numerical deficits from size-compliance evidence."""

    requested, achieved, issues = (
        _owner_phydrax_meshing_providers__native_publication.uniform_size_compliance(
            goal.control, goal.policy, lengths, growth
        )
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
        # A failed bound is repaired only by edges beyond its compliance
        # acceptance limit; edges already inside the band cannot change it.
        tolerance = goal.policy.tolerance
        if quantity.startswith("target_size_"):
            statistic = quantity.removeprefix("target_size_")
            desired = request[f"{prefix}:target_size"]
            measured = actual[f"{prefix}:{statistic}_edge"]
            deviation = abs(measured - desired)
            direction = (
                lengths - (desired + tolerance(desired))
                if measured > desired
                else desired - tolerance(desired) - lengths
            )
        elif quantity == "minimum_size":
            desired = request[f"{prefix}:minimum_size"]
            deviation = desired - actual[f"{prefix}:minimum_edge"]
            direction = desired - tolerance(desired) - lengths
        elif quantity == "maximum_size":
            desired = request[f"{prefix}:maximum_size"]
            deviation = actual[f"{prefix}:maximum_edge"] - desired
            direction = lengths - (desired + tolerance(desired))
        else:
            desired = request[f"{prefix}:maximum_growth_rate"]
            deviation = actual[f"{prefix}:maximum_local_edge_ratio"] - desired
            direction = np.abs(lengths / goal.control.target_size - 1.0)
        # The size-compliance failure list determines whether a deficit exists.
        # Requested values normalize measured deviations; the optimizer
        # does not reimplement quantile comparisons.
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
    """Retain sorted edge incidence and physical lengths for a native state.

    Bounded cavity deltas project global edge evidence in linear edge space
    without copying or sorting the tetrahedron table for every candidate.
    """

    keys: np.ndarray
    counts: np.ndarray
    lengths: np.ndarray
    vertex_count: int
    merit: _PhysicalSizeMerit

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TetMeshArrays,
        goal: UniformSizeRemeshingGoal,
        /,
        *,
        geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
        | None = None,
        budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget
        | None = None,
    ) -> _PhysicalEdgeLedger:

        keys, counts = _edge_incidence(snapshot.tetrahedra)
        lengths, growth = (
            _owner_phydrax_meshing_providers__native_publication.edge_size_evidence(
                snapshot.points,
                _edge_rows(keys),
                geometry=geometry,
                mesh=None
                if geometry is None
                else CellMesh.from_tetrahedra(snapshot.points, snapshot.tetrahedra),
                budget=budget,
            )
        )
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
            raise RuntimeError(
                "tetra_metric.edge_incidence must remain nonnegative after a cavity delta; got a negative count"
            )
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

        growth = (
            _owner_phydrax_meshing_providers__native_publication.edge_growth_evidence(
                lengths, _edge_rows(keys), vertex_count
            )
        )
        return _PhysicalEdgeLedger(
            keys,
            counts,
            lengths,
            vertex_count,
            _physical_size_merit(lengths, growth, goal),
        )


@dataclass(frozen=True, slots=True)
class TetraShapeAims:
    """Soft shape-search targets, distinct from earned hard admissibility.

    ``search_allowance(remaining_work_units, cell_count)`` is called once, after
    the complete entry shape measurement (charged under the epoch's own cap),
    and returns the work the guarded search may spend. Reaching it ends the
    search with a ``resource_limit`` terminal reason and leaves the earned hard
    state and its complete measured bank intact.
    If the entry measurement itself exhausts that cap, its evidence is explicitly
    incomplete; this advisory owner still cannot revoke hard admissibility.
    """

    radius_edge_bound: float
    minimum_dihedral_degrees: float
    refinement_rounds: int
    improvement_passes: int
    search_allowance: Callable[[int, int], int]

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius_edge_bound) or self.radius_edge_bound <= 0.0:
            raise ValueError("TetraShapeAims.radius_edge_bound must be finite and positive.")
        if not np.isfinite(self.minimum_dihedral_degrees) or not (
            0.0 <= self.minimum_dihedral_degrees <= 180.0
        ):
            raise ValueError("TetraShapeAims.minimum_dihedral_degrees must lie in [0, 180].")
        nonnegative_integer(self.refinement_rounds, "TetraShapeAims.refinement_rounds")
        nonnegative_integer(self.improvement_passes, "TetraShapeAims.improvement_passes")
        if not callable(self.search_allowance):
            raise TypeError("TetraShapeAims.search_allowance must be callable.")

    def deficit(self, shape: tuple[float, float], /) -> float:
        radius, angle = shape
        if not np.isfinite(radius) or not np.isfinite(angle):
            return np.inf
        return max(
            0.0, radius / self.radius_edge_bound - 1.0,
            (self.minimum_dihedral_degrees - angle)
            / max(self.minimum_dihedral_degrees, 1.0),
        )


@dataclass(frozen=True, slots=True)
class NativeTetraAimEvidence:
    """Final-state AIM measurements, separate from hard-goal convergence.

    Attempts and recorded rejections are deltas of the existing epoch operation
    counters; accepted_operations counts committed guarded transactions.
    """

    requested_radius_edge_bound: float
    requested_minimum_dihedral_degrees: float
    maximum_radius_edge: float | None
    minimum_dihedral_degrees: float | None
    radius_unmet_cells: int | None
    dihedral_unmet_cells: int | None
    refinement_rounds: int
    improvement_passes: int
    attempts: int
    accepted_operations: int
    refused_proposals: int
    terminal_reason: str
    work_units: int
    unmet: TetMeshUnmet | None
    state_binding: str
    measurement_complete: bool

# None denotes an issued zero-deficit membership proof, never a fabricated
# per-cell radius or dihedral measurement.
type _AimShapeValue = tuple[float, float] | None
_AIM_SHAPE_INITIAL_CELLS = 32


class _AimShapeMeasurements(NamedTuple):
    shapes: dict[tuple[int, ...], _AimShapeValue]
    extrema: tuple[float, float]



@dataclass(frozen=True, slots=True)
class _AimShapeDelta:
    removed: tuple[tuple[int, ...], ...]
    shapes: dict[tuple[int, ...], _AimShapeValue]
    extrema: tuple[float, float]


def _measure_aim_cells(
    points: np.ndarray,
    cells: np.ndarray,
    budget: _NativeMetricBudget,
    floor: float,
    aims: TetraShapeAims,
    changed_vertex: int | None = None,
    changed_position: np.ndarray | None = None,
) -> _AimShapeMeasurements | None:
    """Share native point validation across bounded source-identity cell tiles.

    Aggregate extrema certify every tile cell's zero deficit when both aims
    hold. Only tiles containing an unmet aim need individual measurements.
    The native owner still checks the relative floor of every cell.
    """
    measured: dict[tuple[int, ...], _AimShapeValue] = {}
    maximum_radius, minimum_angle = 0.0, 180.0
    local_cell = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    start, tile_cells = 0, _AIM_SHAPE_INITIAL_CELLS
    while start < cells.shape[0]:
        tile = cells[start:start + tile_cells]
        start += tile.shape[0]
        if not budget.optimizer_work(
            4 * tile.shape[0], 1024 * cells.shape[0], evaluation=False
        ):
            return None
        identities: dict[int, int] = {}
        local = np.empty(tile.shape, dtype=np.int32)
        for row, cell in enumerate(tile.tolist()):
            for column, vertex in enumerate(cell):
                local[row, column] = identities.setdefault(vertex, len(identities))
        vertices = np.fromiter(identities, dtype=np.int64, count=len(identities))
        corners = _proposal_rows(points, vertices, changed_vertex, changed_position)
        shape = budget.call(
            lambda allowance: budget.native.proposal_shape(
                corners, local, vertex_ids=vertices,
                minimum_relative_determinant=floor, work_limit=allowance,
            )
        )
        if shape is None:
            return None
        if shape[2]:
            budget.refuse_optimizer("relative_determinant_floor")
            return None
        maximum_radius = max(maximum_radius, shape[0])
        minimum_angle = min(minimum_angle, shape[1])
        if aims.deficit((shape[0], shape[1])) == 0.0:
            for cell in tile.tolist():
                measured[tuple(sorted(cell))] = None
            # Successful membership proofs allow larger validation tiles;
            # a failed tile returns to the small local measurement schedule.
            tile_cells = min(2 * tile_cells, cells.shape[0])
            continue
        tile_cells = _AIM_SHAPE_INITIAL_CELLS
        if tile.shape[0] == 1:
            measured[tuple(sorted(tile[0].tolist()))] = (shape[0], shape[1])
            continue
        for cell in tile:
            corners = _proposal_rows(points, cell, changed_vertex, changed_position)
            shape = budget.call(
                lambda allowance: budget.native.proposal_shape(
                    corners, local_cell, vertex_ids=cell,
                    minimum_relative_determinant=floor, work_limit=allowance,
                )
            )
            if shape is None:
                return None
            if shape[2]:
                budget.refuse_optimizer("relative_determinant_floor")
                return None
            measured[tuple(sorted(cell.tolist()))] = (shape[0], shape[1])
    return _AimShapeMeasurements(measured, (maximum_radius, minimum_angle))


@dataclass(slots=True)
class _AimShapeLedger:
    aims: TetraShapeAims
    shapes: dict[tuple[int, ...], _AimShapeValue]
    native: TetMesh3D
    extrema: tuple[float, float] | None
    accepted: int = 0

    def score(self, shapes: list[_AimShapeValue], /) -> tuple[float, float]:
        deficits = [0.0 if shape is None else self.aims.deficit(shape) for shape in shapes]
        return max(deficits, default=0.0), sum(deficits)

    @property
    def satisfied(self) -> bool:
        return all(
            shape is None or self.aims.deficit(shape) == 0.0
            for shape in self.shapes.values()
        )

    def offending_edges(self) -> np.ndarray:
        cells = np.asarray([
            cell for cell, shape in self.shapes.items()
            if shape is not None and self.aims.deficit(shape) > 0.0
        ], dtype=np.int32).reshape((-1, 4))
        return _edge_incidence(cells)[0]

    def has_unmet(self, cells: np.ndarray, /) -> bool:
        return any(
            shape is not None and self.aims.deficit(shape) > 0.0
            for shape in (self.shapes[tuple(sorted(cell))] for cell in cells.tolist())
        )

    def propose(
        self, snapshot: TetMeshArrays, removed: np.ndarray, proposed: np.ndarray,
        budget: _NativeMetricBudget, floor: float,
        changed_vertex: int | None, changed_position: np.ndarray | None,
    ) -> _AimShapeDelta | None:
        if not budget.optimizer_scratch(1024 * (len(self.shapes) + proposed.shape[0])):
            return None
        measured = _measure_aim_cells(
            snapshot.points, proposed, budget, floor, self.aims,
            changed_vertex, changed_position,
        )
        if measured is None:
            return None
        shapes = measured.shapes
        removed_keys = tuple(tuple(sorted(cell)) for cell in removed.tolist())
        before = self.score([self.shapes[cell] for cell in removed_keys])
        after = self.score(list(shapes.values()))
        if after >= before:
            return None
        return _AimShapeDelta(removed_keys, shapes, measured.extrema)

    def commit(self, delta: _AimShapeDelta, /) -> None:
        for cell in delta.removed:
            del self.shapes[cell]
        self.shapes.update(delta.shapes)
        self.accepted += 1
        self.extrema = None

    def unmet(self, snapshot: TetMeshArrays, reason: str, /) -> TetMeshUnmet:
        cells: list[list[int]] = []
        criteria: list[int] = []
        values: list[float] = []
        for cell in snapshot.tetrahedra.tolist():
            shape = self.shapes[tuple(sorted(cell))]
            if shape is None:
                continue
            radius, angle = shape
            if radius > self.aims.radius_edge_bound:
                cells.append(cell)
                criteria.append(0)
                values.append(radius)
            if angle < self.aims.minimum_dihedral_degrees:
                cells.append(cell)
                criteria.append(2)
                values.append(angle)
        return TetMeshUnmet(
            np.asarray(cells, dtype=np.int32).reshape((-1, 4)),
            np.asarray(criteria, dtype=np.int32),
            np.full((len(cells),), 5 if reason == "constraint_limited" else 0, dtype=np.int32),
            np.asarray(values, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class _StatisticalProposal:
    edges: _PhysicalEdgeLedger
    quality_sum: float
    cell_count: int
    aim_delta: _AimShapeDelta | None = None


@dataclass(slots=True)
class _StatisticalObjective:
    """Retain size-goal merit and native shape guards for one epoch."""

    goal: UniformSizeRemeshingGoal
    edges: _PhysicalEdgeLedger
    budget: _NativeMetricBudget
    minimum_quality: float
    quality_sum: float
    cell_count: int
    _active_edge_keys: np.ndarray | None = None
    aim: _AimShapeLedger | None = None

    def offending_edge_keys(self) -> np.ndarray:
        """Cache the current failed-statistic suffix or explicit edge violations."""
        if self._active_edge_keys is not None:
            return self._active_edge_keys
        if self.aim is not None:
            self._active_edge_keys = self.aim.offending_edges()
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
        projected: _PhysicalEdgeLedger | None = None,
        proposed_quality: np.ndarray | None = None,
        removed_quality: np.ndarray | None = None,
    ) -> _StatisticalProposal | None:
        """Admit one cavity delta under the unchanged hard guards.

        ``projected``, ``proposed_quality`` and ``removed_quality`` let a caller
        that already evaluated this exact delta (same rows, changed vertex and
        position) reuse those values instead of recomputing them.
        """
        if self.aim is not None:
            if not self.aim.has_unmet(removed):
                return None
        if not proposed.size:
            return None
        vertices = np.unique(proposed)
        local_points = _proposal_rows(
            snapshot.points, vertices, changed_vertex, changed_position
        )
        local_cells = np.searchsorted(vertices, proposed).astype(np.int32)
        if proposed_quality is None:
            local_metric = _proposal_rows(metric, vertices, changed_vertex, changed_metric)
            proposed_quality = _quality(local_points, local_metric, local_cells)
        # Every new cell must meet the unchanged floor; this rejection is
        # independent of the size merit, so it is decided before projection.
        if not np.all(np.isfinite(proposed_quality)) or np.any(proposed_quality < self.minimum_quality):
            return None
        edge_proposal = (
            self.edges.project(
                snapshot.points,
                removed,
                proposed,
                self.goal,
                changed_vertex,
                changed_position,
            )
            if projected is None
            else projected
        )
        old_merit, new_merit = self.edges.merit, edge_proposal.merit
        if new_merit.normalized_deficit > old_merit.normalized_deficit:
            return None
        if self.aim is not None and new_merit.issues:
            return None
        old_quality = (
            _quality(snapshot.points, metric, removed)
            if removed_quality is None
            else removed_quality
        )
        quality_sum = (
            self.quality_sum
            - float(np.sum(old_quality))
            + float(np.sum(proposed_quality))
        )
        cell_count = self.cell_count - removed.shape[0] + proposed.shape[0]
        repairing_quality = bool(np.any(old_quality < self.minimum_quality))
        if (
            self.aim is None
            and not repairing_quality
            and new_merit.normalized_deficit == old_merit.normalized_deficit
        ):
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
        aim_delta = None
        if self.aim is not None:
            aim_delta = self.aim.propose(
                snapshot, removed, proposed, self.budget, floor,
                changed_vertex, changed_position,
            )
            if aim_delta is None:
                return None
        if (
            aim_delta is not None
            or floor > 0.0
            or self.goal.maximum_radius_edge is not None
            or self.goal.minimum_dihedral_degrees is not None
        ):
            # The native owner's vertex indices are this epoch's identities; a
            # proposed vertex carries the index it will receive. PLC publication
            # keeps their order (sorted-used compaction, arange global IDs) and
            # the native rows already start at their smallest index, so this is
            # the chart the publishing validity policy evaluates.
            shape = (
                (
                    aim_delta.extrema[0],
                    aim_delta.extrema[1],
                    0,
                )
                if aim_delta is not None
                else self.budget.call(
                    lambda allowance: self.budget.native.proposal_shape(
                        local_points,
                        local_cells,
                        vertex_ids=vertices,
                        minimum_relative_determinant=floor,
                        work_limit=allowance,
                    )
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
        return _StatisticalProposal(edge_proposal, quality_sum, cell_count, aim_delta)

    def commit(self, proposal: _StatisticalProposal, /) -> None:
        if self.aim is not None:
            if proposal.aim_delta is None:
                raise RuntimeError("A guarded AIM commit must retain its measured cavity delta.")
            self.aim.commit(proposal.aim_delta)
        self.edges = proposal.edges
        self.quality_sum = proposal.quality_sum
        self.cell_count = proposal.cell_count
        self._active_edge_keys = None

    def complete(self, snapshot: TetMeshArrays, metric: np.ndarray, /) -> bool:
        if self.aim is not None and not self.aim.satisfied:
            return False
        return not self.edges.merit.issues and _physical_size_goal_met(
            snapshot, metric, self.goal, self.minimum_quality
        )


class NativeTetraMetricOutcome(NamedTuple):
    """In-place epoch output; native constraint-source identities are unchanged."""

    snapshot: TetMeshArrays
    metric: np.ndarray
    evidence: MetricRemeshingEvidence
    ancestry: _Ancestry
    aim_evidence: NativeTetraAimEvidence | None = None


class _CurvedAuthority(StrictModule):
    """A declared curved source and the native face sources it owns."""

    source: CurvedBoundarySource
    face_sources: np.ndarray


class _NativeMetricDriver(StrictModule):
    native: TetMesh3D
    source: TetraMetricSource
    epoch: _NativeMetricBudget
    fixed_vertices: np.ndarray
    protected_edges: set[tuple[int, int]]
    maximum_vertices: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_operations: int = eqx.field(static=True)
    maximum_cavity_cells: int = eqx.field(static=True)
    minimum_quality: float = eqx.field(static=True)
    lower_length: float = eqx.field(static=True)
    upper_length: float = eqx.field(static=True)
    optimizer_scratch_bytes: int | None = eqx.field(static=True)
    curved: _CurvedAuthority | None = None


class _NativeMetricState(StrictModule):
    snapshot: TetMeshArrays
    metric: np.ndarray
    source_metric: np.ndarray
    ancestry: _Ancestry
    objective: _StatisticalObjective | None
    counts: tuple[int, int, int, int] = eqx.field(static=True, default=(0, 0, 0, 0))
    operations: int = eqx.field(static=True, default=0)
    rejected: int = eqx.field(static=True, default=0)
    compounds: int = eqx.field(static=True, default=0)
    split_parameters: tuple[float, ...] = eqx.field(static=True, default=())


class _NativeMetricStep(StrictModule):
    state: _NativeMetricState
    applied: int = eqx.field(static=True)
    exhausted: bool = eqx.field(static=True)


class _NativeMetricOperation(StrEnum):
    SPLIT = "split"
    COLLAPSE = "collapse"
    FLIP = "flip"
    RING = "ring"
    RELOCATE = "relocate"


def _metric_topology_operation_order(
    *, coarsening_first: bool, statistical: bool
) -> tuple[_NativeMetricOperation, ...]:
    """Offer each existing topology operation once under the bounded pass.

    Ordinary band coarsening first offers connectivity repair before allocating
    refinement points. Statistical sizing retains its separately governed order.
    Neither case adds a pass, sweep, proposal type or repair requeue.
    """
    if not coarsening_first:
        return (
            _NativeMetricOperation.SPLIT,
            _NativeMetricOperation.COLLAPSE,
            _NativeMetricOperation.FLIP,
            _NativeMetricOperation.RING,
        )
    if statistical:
        return (
            _NativeMetricOperation.COLLAPSE,
            _NativeMetricOperation.SPLIT,
            _NativeMetricOperation.FLIP,
            _NativeMetricOperation.RING,
        )
    return (
        _NativeMetricOperation.COLLAPSE,
        _NativeMetricOperation.FLIP,
        _NativeMetricOperation.RING,
        _NativeMetricOperation.SPLIT,
    )


def _prepare_native_metric_state(
    driver: _NativeMetricDriver,
    metric: np.ndarray,
    size_goal: UniformSizeRemeshingGoal | None,
    initial_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
) -> _NativeMetricState:
    with _measure_statistical_phase(driver.epoch, "metric_preparation"):
        source_metric = _interpolate(
            metric[:, None], np.ones((metric.shape[0], 1), dtype=np.float64)
        )
    values = source_metric.copy()
    cells = driver.source.cells
    identifiers = driver.source.cell_global_ids
    ancestry = _Ancestry(
        {
            tuple(sorted(row)): (int(identifier), {int(identifier)})
            for row, identifier in zip(cells.tolist(), identifiers, strict=True)
        },
        int(np.max(identifiers)) + 1,
        [{int(identifier)} for identifier in driver.source.vertex_global_ids.tolist()],
    )
    # Metric interpolation precedes cavity and optimizer allowance binding.
    driver.epoch.maximum_optimizer_scratch_bytes = driver.optimizer_scratch_bytes
    driver.epoch.maximum_cavity_cells = driver.maximum_cavity_cells
    snapshot = driver.native.arrays()
    ancestry.constrained_faces = {
        tuple(sorted(face)): {
            tuple(sorted(driver.source.vertex_global_ids[face].tolist())):
                int(EntityLineageKind.PRESERVED)
        }
        for face in snapshot.faces
    }
    objective = (
        _StatisticalObjective(
            size_goal,
            _PhysicalEdgeLedger.from_snapshot(
                snapshot,
                size_goal,
                geometry=initial_geometry,
                budget=driver.epoch.coordinate_budget,
            ),
            driver.epoch,
            driver.minimum_quality,
            float(np.sum(_quality(snapshot.points, values, snapshot.tetrahedra))),
            snapshot.tetrahedra.shape[0],
        )
        if size_goal is not None
        else None
    )
    return _NativeMetricState(snapshot, values, source_metric, ancestry, objective)


def _run_native_metric_operation(
    operation: _NativeMetricOperation,
    driver: _NativeMetricDriver,
    state: _NativeMetricState,
    /,
    *,
    focus: np.ndarray | None = None,
) -> _NativeMetricStep:
    """Run one epoch phase; ``focus`` restricts a reconnection to these cells."""
    native, epoch = driver.native, driver.epoch
    snapshot, values, ancestry = state.snapshot, state.metric, state.ancestry
    objective = state.objective
    remaining = driver.maximum_operations - state.operations
    refusals, compounds, parameters, exhausted = 0, 0, (), False
    if focus is not None and operation not in (
        _NativeMetricOperation.FLIP, _NativeMetricOperation.RING,
    ):
        raise ValueError(
            f"tetra_metric.focus must select a reconnection phase; got {operation!r}"
        )
    match operation:
        case _NativeMetricOperation.SPLIT:
            with _measure_statistical_phase(epoch, "metric_split"):
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
                    driver.source,
                    state.source_metric,
                    ancestry,
                    driver.protected_edges,
                    remaining,
                    driver.maximum_vertices,
                    driver.maximum_cells,
                    driver.maximum_cavity_cells,
                    epoch,
                    driver.upper_length,
                    objective,
                    minimum_quality=driver.minimum_quality,
                    metric_band=(driver.lower_length, driver.upper_length),
                )
            count_index = 0
        case _NativeMetricOperation.COLLAPSE:
            with _measure_statistical_phase(epoch, "metric_collapse"):
                snapshot, changes, attempts, refusals = _collapse_pass(
                    native,
                    snapshot,
                    values,
                    driver.fixed_vertices,
                    ancestry,
                    remaining,
                    driver.minimum_quality,
                    driver.maximum_cavity_cells,
                    epoch,
                    driver.lower_length,
                    objective,
                    metric_band=(driver.lower_length, driver.upper_length),
                    curved=driver.curved,
                )
            count_index = 1
        case _NativeMetricOperation.FLIP:
            with _measure_statistical_phase(epoch, "metric_reconnection"):
                snapshot, changes, attempts = _flip_pass(
                    native,
                    snapshot,
                    values,
                    ancestry,
                    remaining,
                    epoch,
                    objective,
                    minimum_quality=driver.minimum_quality,
                    metric_band=(driver.lower_length, driver.upper_length),
                    focus=focus,
                )
            count_index = 2
        case _NativeMetricOperation.RING:
            with _measure_statistical_phase(epoch, "metric_reconnection"):
                snapshot, changes, attempts, exhausted = _ring_pass(
                    native,
                    snapshot,
                    values,
                    ancestry,
                    remaining,
                    driver.maximum_cavity_cells,
                    epoch,
                    driver.maximum_cells,
                    objective,
                    minimum_quality=driver.minimum_quality,
                    metric_band=(driver.lower_length, driver.upper_length),
                    focus=focus,
                )
            count_index = 2
        case _NativeMetricOperation.RELOCATE:
            with _measure_statistical_phase(epoch, "metric_relocation"):
                snapshot, changes, attempts = _relocate_pass(
                    native,
                    snapshot,
                    values,
                    driver.fixed_vertices,
                    remaining,
                    driver.source,
                    state.source_metric,
                    ancestry,
                    epoch,
                    objective,
                    minimum_quality=driver.minimum_quality,
                    metric_band=(driver.lower_length, driver.upper_length),
                )
            count_index = 3
        case _:
            raise ValueError(
                f"tetra_metric.operation must select an epoch phase; got {operation!r}"
            )
    counts = list(state.counts)
    counts[count_index] += changes
    return _NativeMetricStep(
        _NativeMetricState(
            snapshot,
            values,
            state.source_metric,
            ancestry,
            objective,
            (counts[0], counts[1], counts[2], counts[3]),
            state.operations + attempts,
            state.rejected + refusals,
            state.compounds + compounds,
            state.split_parameters + parameters,
        ),
        changes,
        exhausted or epoch.exhausted,
    )


def _shape_aim_evidence(
    state: _NativeMetricState,
    aims: TetraShapeAims,
    ledger: _AimShapeLedger | None,
    rounds: int,
    passes: int,
    reason: str,
    work: int,
    attempts: int,
    rejected: int,
) -> NativeTetraAimEvidence:
    snapshot = state.snapshot
    binding = canonical_fingerprint({
        "state": array_tree_fingerprint((
            snapshot.points, snapshot.tetrahedra, snapshot.tetrahedron_regions,
            snapshot.faces, snapshot.face_sources, snapshot.segments, snapshot.segment_sources,
        )),
        "radius_edge_bound": aims.radius_edge_bound,
        "minimum_dihedral_degrees": aims.minimum_dihedral_degrees,
    })
    unmet = None if ledger is None else ledger.unmet(snapshot, reason)
    extrema = None if ledger is None else ledger.extrema
    if ledger is not None and extrema is None:
        # Accepted edits invalidate the entry aggregate. Renew actual global
        # extrema from the same native shape owner, not per-cell placeholders.
        quality = ledger.native.quality(sliver_degrees=aims.minimum_dihedral_degrees)
        extrema = (quality.maximum_radius_edge, quality.minimum_dihedral)
    return NativeTetraAimEvidence(
        requested_radius_edge_bound=aims.radius_edge_bound,
        requested_minimum_dihedral_degrees=aims.minimum_dihedral_degrees,
        maximum_radius_edge=None if extrema is None else extrema[0],
        minimum_dihedral_degrees=None if extrema is None else extrema[1],
        radius_unmet_cells=None if unmet is None else int(np.count_nonzero(unmet.criteria == 0)),
        dihedral_unmet_cells=None if unmet is None else int(np.count_nonzero(unmet.criteria == 2)),
        refinement_rounds=rounds,
        improvement_passes=passes,
        attempts=attempts,
        accepted_operations=0 if ledger is None else ledger.accepted,
        refused_proposals=rejected,
        terminal_reason=reason,
        work_units=work,
        unmet=unmet,
        state_binding=binding,
        measurement_complete=ledger is not None,
    )


def _run_shape_aims(
    driver: _NativeMetricDriver,
    state: _NativeMetricState,
    aims: TetraShapeAims,
    /,
) -> tuple[_NativeMetricState, NativeTetraAimEvidence]:
    # Retain the measured shape bank under one live storage reservation across
    # all guarded subphases; no new epoch or source ancestry is reconstructed.
    epoch = driver.epoch
    maximum = epoch.maximum_work_units
    try:
        with epoch.phase():
            try:
                return _search_shape_aims(driver, state, aims)
            finally:
                if state.objective is not None:
                    state.objective.aim = None
                    state.objective._active_edge_keys = None
    finally:
        epoch.maximum_work_units = maximum


def _search_shape_aims(
    driver: _NativeMetricDriver,
    state: _NativeMetricState,
    aims: TetraShapeAims,
    /,
) -> tuple[_NativeMetricState, NativeTetraAimEvidence]:
    """Search soft shape targets without losing any earned hard obligation."""
    objective, budget = state.objective, driver.epoch
    if objective is None:
        raise RuntimeError("A guarded shape AIM epoch must retain its hard statistical objective.")
    before = budget.work_units
    initial_operations = state.operations
    initial_rejected = state.rejected
    with _measure_statistical_phase(budget, "metric_preparation"):
        shapes = (
            _measure_aim_cells(
                state.snapshot.points, state.snapshot.tetrahedra,
                budget, objective.goal.minimum_relative_determinant, aims,
            )
            if budget.optimizer_scratch(1024 * state.snapshot.tetrahedra.shape[0])
            else None
        )
    if shapes is None:
        if not budget.exhausted:
            raise _leaf_phydrax_meshing__failures.MeshingFailure(
                _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED,
                "The AIM entry state has not earned the hard determinant floor.",
                stage="optimization",
                requested=(("minimum_relative_determinant", objective.goal.minimum_relative_determinant),),
                achieved=(("observed_below_floor", 1.0),),
            )
        return state, _shape_aim_evidence(
            state, aims, None, 0, 0, "resource_limit", budget.work_units - before, 0, 0
        )
    ledger = _AimShapeLedger(aims, shapes.shapes, budget.native, shapes.extrema)
    objective.aim = ledger
    objective._active_edge_keys = None
    # The complete measured bank is mandatory evidence, charged under the
    # epoch's own cap. Only the soft search below runs under the caller's
    # allowance; its resource stop is AIM evidence, so the refusal state the
    # epoch held before the search is restored. Every spent unit stays charged.
    refusal = (
        budget.exhausted,
        budget.resource_message,
        budget.resource_requested,
        budget.resource_achieved,
    )
    allowance = nonnegative_integer(
        aims.search_allowance(
            max(0, budget.maximum_work_units - budget.work_units),
            int(state.snapshot.tetrahedra.shape[0]),
        ),
        "TetraShapeAims.search_allowance",
    )
    budget.maximum_work_units = min(
        budget.maximum_work_units, budget.work_units + allowance
    )
    rounds, passes = 0, 0
    stalled, exhausted = False, False
    for _ in range(aims.refinement_rounds):
        if ledger.satisfied:
            break
        if state.operations >= driver.maximum_operations:
            exhausted = True
            break
        with _measure_statistical_phase(budget, "refinement"):
            step = _run_native_metric_operation(_NativeMetricOperation.SPLIT, driver, state)
        rounds += 1
        state = step.state
        exhausted = step.exhausted
        stalled = step.applied == 0
        if exhausted or stalled:
            break
    if not exhausted:
        for _ in range(aims.improvement_passes):
            if ledger.satisfied:
                break
            applied = 0
            with _measure_statistical_phase(budget, "improvement"):
                for operation in (
                    _NativeMetricOperation.COLLAPSE, _NativeMetricOperation.FLIP,
                    _NativeMetricOperation.RING, _NativeMetricOperation.RELOCATE,
                ):
                    if state.operations >= driver.maximum_operations:
                        exhausted = True
                        break
                    step = _run_native_metric_operation(operation, driver, state)
                    state = step.state
                    applied += step.applied
                    exhausted = step.exhausted
                    if exhausted or ledger.satisfied:
                        break
            passes += 1
            stalled = applied == 0
            if exhausted or stalled:
                break
    reason = (
        "resource_limit" if exhausted or budget.exhausted
        else "satisfied" if ledger.satisfied
        else "constraint_limited" if stalled
        else "pass_limit"
    )
    (
        budget.exhausted,
        budget.resource_message,
        budget.resource_requested,
        budget.resource_achieved,
    ) = refusal
    return state, _shape_aim_evidence(
        state, aims, ledger, rounds, passes, reason, budget.work_units - before,
        state.operations - initial_operations, state.rejected - initial_rejected,
    )


def _source_owned_short_edges(
    snapshot: TetMeshArrays,
    curved: _CurvedAuthority | None,
    epoch: _NativeMetricBudget,
    edges: np.ndarray,
    lengths: np.ndarray,
    lower: float,
) -> tuple[tuple[tuple[int, int, float], ...], str | None, float, bool]:
    """Authenticate source-owned lower sizing, never exempt a long boundary edge.

    Both boundary facets incident to an exempt edge must be current exterior
    facets owned by this source, with fresh or exactly cached whole-facet proof.
    Row keys are bound to the complete current topology and coordinates.
    """
    if curved is None:
        return (), None, 0.0, True
    if not epoch.optimizer_work(
        4 * snapshot.tetrahedra.shape[0] + 3 * snapshot.faces.shape[0] + edges.shape[0],
        1024 * (snapshot.tetrahedra.shape[0] + snapshot.faces.shape[0] + edges.shape[0]),
    ):
        return (), None, 0.0, False
    curved.source.validate_source_integrity()
    selected = np.isin(snapshot.face_sources, curved.face_sources)
    faces = snapshot.faces[selected]
    if faces.size == 0:
        return (), None, 0.0, False
    keys, counts = np.unique(
        np.sort(snapshot.tetrahedra[:, _FACES].reshape((-1, 3)), axis=1),
        axis=0, return_counts=True,
    )
    exterior = {tuple(row) for row in keys[counts == 1].tolist()}
    owned = [tuple(row) for row in np.sort(faces, axis=1).tolist()]
    if len(set(owned)) != len(owned) or any(key not in exterior for key in owned):
        return (), None, 0.0, False
    proof = epoch.certify_curved(curved.source, snapshot.points[faces])
    if proof is None:
        return (), None, 0.0, False
    deviation, admitted = proof
    maximum = float(np.max(deviation, initial=0.0))
    if not np.all(admitted):
        return (), None, maximum, False
    support: dict[tuple[int, int], int] = {}
    for first, second, third in owned:
        for edge in ((first, second), (first, third), (second, third)):
            support[edge] = support.get(edge, 0) + 1
    exterior_support: dict[tuple[int, int], int] = {}
    for first, second, third in exterior:
        for edge in ((first, second), (first, third), (second, third)):
            exterior_support[edge] = exterior_support.get(edge, 0) + 1
    rows = tuple(
        (int(first), int(second), float(length))
        for (first, second), length in zip(edges.tolist(), lengths.tolist(), strict=True)
        if length < lower and support.get((first, second), 0) == 2
        and exterior_support.get((first, second), 0) == 2
    )
    binding = None if not rows else canonical_fingerprint({
        "kind": "source-owned-metric-lower-bound",
        "source": curved.source.digest.hex(),
        "points": snapshot.points,
        "cells": snapshot.tetrahedra,
        "faces": faces,
        "face_sources": snapshot.face_sources[selected],
        "exemptions": rows,
        "deviation": deviation,
    })
    return rows, binding, maximum, True


def _certify_native_metric_evidence(
    driver: _NativeMetricDriver,
    state: _NativeMetricState,
    status: MetricRemeshingStatus,
    passes: int,
    topology_operations: bool,
    size_goal: UniformSizeRemeshingGoal | None,
    initial_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    *,
    advisory_aim_resource_stop: bool = False,
) -> MetricRemeshingEvidence:
    epoch = driver.epoch
    snapshot, values = state.snapshot, state.metric
    edges = _edges(snapshot.tetrahedra)
    epoch.check_deadline()
    coordinate_before = epoch.coordinate_budget.work_units
    lengths = _lengths(snapshot.points, values, edges)
    quality = _quality(snapshot.points, values, snapshot.tetrahedra)
    with epoch.phase() if driver.curved is not None else nullcontext():
        exemptions, source_binding, fidelity, source_valid = (
            _source_owned_short_edges(
                snapshot, driver.curved, epoch, edges, lengths, driver.lower_length
            )
            if topology_operations and size_goal is None
            else ((), None, 0.0, True)
        )
    unmet = int(np.count_nonzero(
        (lengths < driver.lower_length) | (lengths > driver.upper_length)
    )) - len(exemptions)
    satisfied = (
        _physical_size_goal_met(
            snapshot,
            values,
            size_goal,
            driver.minimum_quality,
            geometry=initial_geometry,
            budget=epoch.coordinate_budget,
        )
        if size_goal is not None
        else topology_operations
        and unmet == 0 and source_valid
        and np.all(quality >= driver.minimum_quality)
    )
    # Soft shape measurements cannot revoke an already earned hard goal.
    # Keep their refusal facts and every spent unit in the returned evidence.
    if epoch.exhausted and not advisory_aim_resource_stop:
        status = MetricRemeshingStatus.RESOURCE_LIMIT
    elif status is not MetricRemeshingStatus.RESOURCE_LIMIT:
        if satisfied:
            status = MetricRemeshingStatus.COMPLETE
        elif not source_valid or (
            topology_operations and status is MetricRemeshingStatus.COMPLETE
        ):
            status = MetricRemeshingStatus.STALLED
    requested, achieved, issues = (
        physical_size_compliance(
            state.snapshot,
            size_goal,
            geometry=initial_geometry,
            budget=epoch.coordinate_budget,
        )
        if size_goal is not None
        else ([], [], [])
    )
    if not source_valid:
        issues.append("source_boundary_fidelity_or_support")
    arc_bounds, arc_ids, arc_binding = None, None, None
    if initial_geometry is not None:
        if size_goal is None:
            raise ValueError(
                "tetra_metric.initial_geometry must bind a physical size goal; got none"
            )
        mesh = CellMesh.from_tetrahedra(snapshot.points, snapshot.tetrahedra)
        physical_lengths, _ = (
            _owner_phydrax_meshing_providers__native_publication.edge_size_evidence(
                snapshot.points,
                edges,
                geometry=initial_geometry,
                mesh=mesh,
                budget=epoch.coordinate_budget,
            )
        )
        lengths = physical_lengths / size_goal.control.target_size
        arc_bounds = (
            _owner_phydrax_meshing_providers__native_publication.source_edge_arc_bounds(
                mesh,
                initial_geometry,
                edges,
                length_scale=size_goal.control.target_size,
                budget=epoch.coordinate_budget,
            )
        )
        arc_ids = driver.source.vertex_global_ids[edges]
        arc_binding = canonical_fingerprint(
            {
                "coordinate_source": initial_geometry.geometry_layout_id,
                "metric": array_tree_fingerprint(values),
                "edges": array_tree_fingerprint(arc_ids),
            }
        )
        achieved.extend(
            (
                (f"coordinate_source:{initial_geometry.geometry_layout_id}", 1.0),
                (
                    "coordinate_source_execution_error_upper",
                    initial_geometry.source_execution_error(
                        mesh, budget=epoch.coordinate_budget
                    ),
                ),
            )
        )
        epoch.coordinate_work_units += (
            epoch.coordinate_budget.work_units - coordinate_before
        )
        achieved.append(("coordinate_source_work_units", epoch.coordinate_work_units))
    return MetricRemeshingEvidence(
        status,
        passes=passes,
        counts=state.counts,
        rejected_operations=state.rejected,
        work_units=epoch.work_units,
        lengths=lengths,
        quality=quality,
        maximum_fidelity_bound=fidelity,
        source_lower_exemptions=exemptions,
        source_lower_binding=source_binding,
        criterion=(
            MetricRemeshingCriterion.SIZE_STATISTICS
            if size_goal is not None
            else MetricRemeshingCriterion.UNIT_MESH
            if topology_operations
            else MetricRemeshingCriterion.RELOCATION_FIXED_POINT
        ),
        split_parameters=state.split_parameters,
        lower_metric_length=driver.lower_length,
        upper_metric_length=driver.upper_length,
        operation_attempts=state.operations,
        memory=MetricMemoryEvidence(
            native_counters=driver.native.memory_counters(),
            optimizer_scratch_required_bytes=epoch.optimizer_scratch_required_bytes,
        ),
        size_requested=tuple(requested),
        size_achieved=tuple(achieved),
        size_issues=tuple(issues),
        source_candidate_pairs=epoch.source_candidates[0],
        metric_quality_floor=driver.minimum_quality,
        shape_radius_edge_bound=size_goal.maximum_radius_edge
        if size_goal is not None
        else None,
        shape_minimum_dihedral_degrees=(
            size_goal.minimum_dihedral_degrees if size_goal is not None else None
        ),
        resource_message=epoch.resource_message,
        resource_requested=epoch.resource_requested,
        resource_achieved=epoch.resource_achieved,
        compound_splits=state.compounds,
        expanded_insertions=epoch.expanded_insertions,
        optimizer=MetricOptimizerEvidence(
            status_counts=tuple(epoch.optimizer_status_counts)
            if epoch.optimizer_terminations
            else (),
            terminations=tuple(epoch.optimizer_terminations),
            method_id=epoch.optimizer_method_id,
            tolerances=epoch.optimizer_tolerances,
            iterations=epoch.optimizer_iterations,
            evaluations=epoch.optimizer_objective_evaluations,
            constraint_evaluations=epoch.optimizer_constraint_evaluations,
            proposal_evaluations=epoch.optimizer_evaluations,
            refused_proposals=epoch.optimizer_refused_proposals,
            refusal_counts=tuple(sorted(epoch.optimizer_refusals.items())),
            host_work_units=epoch.optimizer_host_work,
        ),
        native_work_units=epoch.native_work_units,
        metric_arc_bounds=arc_bounds,
        metric_arc_edge_ids=arc_ids,
        metric_arc_binding=arc_binding,
    )


class _NativeMetricAdmission(StrictModule):
    optimizer_scratch_bytes: int | None = eqx.field(static=True)
    operation_started: float | None = eqx.field(static=True)
    maximum_wall_seconds: float | None = eqx.field(static=True)

    @classmethod
    def prepare(
        cls,
        source: TetraMetricSource,
        metric: np.ndarray,
        fixed_vertices: np.ndarray,
        minimum_quality: float,
        size_goal: UniformSizeRemeshingGoal | None,
        topology_operations: bool,
        optimizer_scratch_bytes: int | None,
        operation_started: float | None,
        maximum_wall_seconds: float | None,
    ) -> _NativeMetricAdmission:
        validate_metric_quality_floor(
            minimum_quality,
            message="tetra_metric.minimum_metric_quality must lie in [0, 1]; got a value outside the unit interval",
        )
        if metric.shape != (source.points.shape[0], 3, 3):
            raise ValueError(
                f"tetra_metric.metric must have one 3D SPD tensor per source vertex; got shape={metric.shape}"
            )
        if fixed_vertices.shape != (source.points.shape[0],):
            raise ValueError(
                f"tetra_metric.fixed_vertices must align with source vertices; got shape={fixed_vertices.shape}"
            )
        if size_goal is not None and not topology_operations:
            raise TypeError(
                "tetra_metric.size_goal must enable topology operations; got a physical-size goal without topology operations"
            )
        if optimizer_scratch_bytes is not None:
            optimizer_scratch_bytes = nonnegative_integer(
                optimizer_scratch_bytes, "maximum_optimizer_scratch_bytes"
            )
        if (operation_started is None) != (maximum_wall_seconds is None):
            raise ValueError(
                "tetra_metric.deadline must provide the request clock and wall allowance together; got a partial deadline"
            )
        if operation_started is not None and (
            not np.isfinite(operation_started)
            or maximum_wall_seconds is None
            or not np.isfinite(maximum_wall_seconds)
            or maximum_wall_seconds < 0.0
        ):
            raise ValueError(
                "tetra_metric.deadline must contain a finite clock and nonnegative wall allowance; got invalid deadline values"
            )
        return cls(optimizer_scratch_bytes, operation_started, maximum_wall_seconds)


@checked
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
    shape_aims: TetraShapeAims | None = None,
    maximum_optimizer_scratch_bytes: int | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    operation_started: float | None = None,
    maximum_wall_seconds: float | None = None,
    initial_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    curved: _CurvedAuthority | None = None,
) -> NativeTetraMetricOutcome:
    """Run a metric epoch on an existing native constrained mesh.

    The epoch retains native material, facet, and curve IDs. The source
    carrier supplies tensor interpolation and edit ancestry; numerical edge
    bounds and cumulative native work remain explicit.
    """
    if (
        not np.isfinite(lower_metric_length)
        or not np.isfinite(upper_metric_length)
        or not 0.0 <= lower_metric_length <= upper_metric_length
        or upper_metric_length <= 0.0
    ):
        raise ValueError(
            "tetra_metric.metric_edge_band must be a finite nonnegative interval with positive upper bound; got invalid bounds"
        )
    if shape_aims is not None:
        if not isinstance(shape_aims, TetraShapeAims):
            raise TypeError("execute_native_tetra_metric.shape_aims must be TetraShapeAims.")
        if size_goal is None or initial_geometry is not None:
            raise ValueError("Guarded shape AIMs require an affine native hard size goal.")
        if not topology_operations or not relocation:
            raise ValueError("Guarded shape AIMs require enabled topology and relocation proposals.")
    schedule = MetricExecutionSchedule(
        maximum_passes=maximum_passes,
        maximum_vertices=maximum_vertices,
        maximum_cells=maximum_cells,
        maximum_operations=maximum_operations,
        maximum_work_units=maximum_work_units,
        maximum_location_pairs=maximum_location_pairs,
        maximum_cavity_cells=maximum_cavity_cells,
        maximum_cavity_work=maximum_cavity_work,
        maximum_arc_work_units=0,
    )
    (
        maximum_passes,
        maximum_vertices,
        maximum_cells,
        maximum_operations,
        maximum_work_units,
        maximum_location_pairs,
        maximum_cavity_cells,
        maximum_cavity_work,
    ) = (
        schedule.maximum_passes,
        schedule.maximum_vertices,
        schedule.maximum_cells,
        schedule.maximum_operations,
        schedule.maximum_work_units,
        schedule.maximum_location_pairs,
        schedule.maximum_cavity_cells,
        schedule.maximum_cavity_work,
    )
    validate_metric_operation_controls(topology_operations, relocation)
    admission = _NativeMetricAdmission.prepare(
        source_tables,
        metric,
        fixed_vertices,
        minimum_metric_quality,
        size_goal,
        topology_operations,
        maximum_optimizer_scratch_bytes,
        operation_started,
        maximum_wall_seconds,
    )
    epoch = _NativeMetricBudget(
        native,
        maximum_work_units,
        maximum_cavity_work,
        source_tables,
        maximum_location_pairs,
        record_phase=record_phase,
        scope=scope,
        coordinate_budget=budget,
        operation_started=admission.operation_started,
        maximum_wall_seconds=admission.maximum_wall_seconds,
    )
    driver = _NativeMetricDriver(
        native,
        source_tables,
        epoch,
        fixed_vertices,
        protected_edges,
        maximum_vertices,
        maximum_cells,
        maximum_operations,
        maximum_cavity_cells,
        float(minimum_metric_quality),
        float(lower_metric_length),
        float(upper_metric_length),
        admission.optimizer_scratch_bytes,
        curved,
    )
    with budget.live_storage() as proof_storage:
        epoch.curved_storage = proof_storage
        try:
            return _run_native_metric_epoch(
                driver, metric, schedule, size_goal, shape_aims, initial_geometry,
                topology_operations, relocation,
            )
        finally:
            epoch.curved_bounds.clear()
            epoch.curved_witnesses.clear()
            epoch.curved_storage = None


def _run_native_metric_epoch(
    driver: _NativeMetricDriver,
    metric: np.ndarray,
    schedule: MetricExecutionSchedule,
    size_goal: UniformSizeRemeshingGoal | None,
    shape_aims: TetraShapeAims | None,
    initial_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None,
    topology_operations: bool,
    relocation: bool,
) -> NativeTetraMetricOutcome:
    """Execute one admitted epoch with request-owned curved proof storage."""
    native, epoch = driver.native, driver.epoch
    budget = epoch.coordinate_budget
    minimum_metric_quality = driver.minimum_quality
    lower_metric_length, upper_metric_length = driver.lower_length, driver.upper_length
    maximum_operations = driver.maximum_operations
    maximum_work_units = epoch.maximum_work_units
    if initial_geometry is not None:
        if size_goal is None:
            raise ValueError(
                "execute_native_tetra_metric.initial_geometry must bind a physical size goal; got none"
            )
        inverse = 1.0 / size_goal.control.target_size
        expected = np.eye(3, dtype=np.float64) * (inverse * inverse)
        if not np.array_equal(metric, np.broadcast_to(expected, metric.shape)):
            raise ValueError(
                "execute_native_tetra_metric.initial_geometry must bind its global uniform metric; got another metric field"
            )
    coordinate_before = budget.work_units
    with epoch.phase() if initial_geometry is not None else nullcontext():
        state = _prepare_native_metric_state(driver, metric, size_goal, initial_geometry)
    if initial_geometry is not None:
        epoch.coordinate_work_units += budget.work_units - coordinate_before
    passes = 0
    status = MetricRemeshingStatus.PASS_LIMIT
    relocation_checked = False
    for passes in metric_execution_passes(schedule, epoch.check_deadline):
        snapshot, values = state.snapshot, state.metric
        edges = _edges(snapshot.tetrahedra)
        lengths = _lengths(snapshot.points, values, edges)
        quality = _quality(snapshot.points, values, snapshot.tetrahedra)
        coordinate_before = budget.work_units
        with epoch.phase() if initial_geometry is not None else nullcontext():
            satisfied = (
                _physical_size_goal_met(
                    snapshot,
                    values,
                    size_goal,
                    minimum_metric_quality,
                    geometry=initial_geometry,
                    budget=budget,
                )
                if size_goal is not None
                else topology_operations
                and np.all(
                    (lengths >= lower_metric_length) & (lengths <= upper_metric_length)
                )
                and np.all(quality >= minimum_metric_quality)
            )
        if initial_geometry is not None:
            epoch.coordinate_work_units += budget.work_units - coordinate_before
        if satisfied:
            if size_goal is not None and relocation and initial_geometry is None:
                step = _run_native_metric_operation(
                    _NativeMetricOperation.RELOCATE, driver, state
                )
                relocation_checked = True
                state = step.state
                status = (
                    MetricRemeshingStatus.RESOURCE_LIMIT
                    if step.exhausted else MetricRemeshingStatus.COMPLETE
                )
            else:
                status = MetricRemeshingStatus.COMPLETE
            break
        if initial_geometry is not None:
            raise ValueError(
                "execute_native_tetra_metric.initial_geometry must satisfy the initial goal before native mutations; got unresolved immutable source coordinates"
            )
        applied = 0
        initial_deficit = (
            None if state.objective is None
            else state.objective.edges.merit.normalized_deficit
        )
        operations: tuple[_NativeMetricOperation, ...] = ()
        coarsening_first = bool(
            np.count_nonzero(lengths < lower_metric_length)
            > np.count_nonzero(lengths > upper_metric_length)
        )
        if topology_operations and state.objective is not None:
            longer, shorter = _failed_statistic_sides(state.objective)
            coarsening_first = bool(shorter and not longer)
        if topology_operations:
            operations = _metric_topology_operation_order(
                coarsening_first=coarsening_first,
                statistical=state.objective is not None,
            )
        schedule_steps: list[tuple[_NativeMetricOperation, np.ndarray | None]] = []
        below_floor = quality < minimum_metric_quality
        if topology_operations and np.any(below_floor):
            # Repair an actual hard-quality deficit first, on reconnections of
            # the below-floor cells only, before any whole-mesh or insertion
            # pass. Every new cavity still meets the unchanged floor; the full
            # requested schedule follows unchanged.
            repair = snapshot.tetrahedra[below_floor]
            schedule_steps.extend((
                (_NativeMetricOperation.FLIP, repair),
                (_NativeMetricOperation.RING, repair),
            ))
        schedule_steps.extend((operation, None) for operation in operations)
        schedule_steps.append((_NativeMetricOperation.RELOCATE, None))
        for operation, focus in schedule_steps:
            if operation is _NativeMetricOperation.RELOCATE:
                if not relocation:
                    continue
                if (
                    state.objective is None
                    and state.operations >= maximum_operations
                ):
                    continue
                if (
                    state.objective is not None
                    and initial_deficit is not None
                    and applied > 0
                    and passes < schedule.maximum_passes
                    and state.objective.edges.merit.normalized_deficit < initial_deficit
                ):
                    # A transient topology still makes strict requested-size
                    # progress. The stalled/closing pass must run relocation;
                    # early success below also executes its requested check.
                    continue
            step = _run_native_metric_operation(operation, driver, state, focus=focus)
            relocation_checked |= operation is _NativeMetricOperation.RELOCATE
            state = step.state
            applied += step.applied
            if step.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
                break
            if _physical_size_goal_met(
                state.snapshot, state.metric, size_goal, minimum_metric_quality
            ):
                if relocation and operation is not _NativeMetricOperation.RELOCATE:
                    closure = _run_native_metric_operation(
                        _NativeMetricOperation.RELOCATE, driver, state
                    )
                    relocation_checked = True
                    state = closure.state
                    status = (
                        MetricRemeshingStatus.RESOURCE_LIMIT
                        if closure.exhausted else MetricRemeshingStatus.COMPLETE
                    )
                else:
                    status = MetricRemeshingStatus.COMPLETE
                break
        if status in (
            MetricRemeshingStatus.COMPLETE,
            MetricRemeshingStatus.RESOURCE_LIMIT,
        ):
            break
        if applied == 0:
            status = (
                MetricRemeshingStatus.RESOURCE_LIMIT
                if state.operations >= maximum_operations
                else MetricRemeshingStatus.STALLED
            )
            if (
                not topology_operations
                and state.operations < maximum_operations
                and np.all(quality >= minimum_metric_quality)
            ):
                status = MetricRemeshingStatus.COMPLETE
            break
    aim_evidence = None
    if (
        size_goal is not None
        and initial_geometry is None
        and status is not MetricRemeshingStatus.RESOURCE_LIMIT
        and (shape_aims is not None or (relocation and not relocation_checked))
        and _physical_size_goal_met(
            state.snapshot, state.metric, size_goal, minimum_metric_quality
        )
    ):
        status = MetricRemeshingStatus.COMPLETE
        if relocation and not relocation_checked:
            closure = _run_native_metric_operation(
                _NativeMetricOperation.RELOCATE, driver, state
            )
            state = closure.state
            if closure.exhausted:
                status = MetricRemeshingStatus.RESOURCE_LIMIT
        if shape_aims is not None and status is MetricRemeshingStatus.COMPLETE:
            state, aim_evidence = _run_shape_aims(driver, state, shape_aims)
    snapshot, values = state.snapshot, state.metric
    with epoch.phase() if initial_geometry is not None else nullcontext():
        evidence = _certify_native_metric_evidence(
            driver,
            state,
            status,
            passes,
            topology_operations,
            size_goal,
            initial_geometry,
            advisory_aim_resource_stop=(
                status is MetricRemeshingStatus.COMPLETE
                and aim_evidence is not None
                and aim_evidence.terminal_reason == "resource_limit"
            ),
        )
    native.set_work_limit(max(0, maximum_work_units - epoch.work_units))
    return NativeTetraMetricOutcome(snapshot, values, evidence, state.ancestry, aim_evidence)


class _TetraMetricInput(StrictModule):
    points: np.ndarray
    metric: np.ndarray
    cells: np.ndarray
    cell_ids: np.ndarray
    regions: np.ndarray
    protected: np.ndarray
    fixed: np.ndarray
    faces: np.ndarray
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray


def _prepare_tetra_metric_input(
    mesh: CellMesh,
    metric: ArrayLike,
    cell_classes: ArrayLike | None,
    facet_classes: ArrayLike | None,
    edge_classes: ArrayLike | None,
    protected_edges: ArrayLike | None,
    fixed_vertices: ArrayLike | None,
) -> _TetraMetricInput:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    values = np.asarray(metric, dtype=np.float64)
    if values.shape != (points.shape[0], 3, 3):
        raise ValueError(
            f"tetra_metric.metric must have one 3D SPD tensor per vertex; got shape={values.shape}"
        )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks]
    )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
    )
    rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(mesh, 3),
        cell_ids[:, None],
    )
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
            "tetra_metric.masks must align protected edges and fixed vertices with the source; got incompatible shapes"
        )
    faces, face_sources, segments, segment_sources = constraints(
        mesh, cells, regions, facet_classes, edge_classes, protected
    )
    return _TetraMetricInput(
        points,
        values,
        cells,
        cell_ids,
        regions,
        protected,
        fixed,
        faces,
        face_sources,
        segments,
        segment_sources,
    )


class _CurvedDeclaration(StrictModule):
    authority: _CurvedAuthority
    native: TetMeshCurvedSources
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray


def _declare_curved_sources(
    mesh: CellMesh,
    request: _TetraMetricInput,
    facet_classes: ArrayLike | None,
    edge_classes: ArrayLike | None,
    curved: CurvedBoundarySource,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> _CurvedDeclaration | None:
    """Group only whole-facet-certified smooth faces into curved native sources.

    A face of the smooth source enters the curved group only with its own
    outward whole-facet bound under the original source and limits; any
    other face stays its own planar source with segment edges.
    """
    if curved.facets.shape != (mesh.entity_set(2).count,):
        raise ValueError(
            f"tetra_metric.curved_boundary.facets must mark facet entity rows; got shape={curved.facets.shape}"
        )
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    rows = entity_keys_owner.key_rows(
        entity_keys_owner.entity_keys(mesh, 2), np.sort(vertex_ids[request.faces], axis=1)
    )
    candidates = np.flatnonzero(curved.facets[rows])
    if candidates.size == 0:
        return None
    deviation, admitted = curved.certify(
        request.points[request.faces[candidates]], budget=budget
    )
    certified = candidates[admitted]
    if certified.size == 0:
        return None
    smooth = np.zeros((mesh.entity_set(2).count,), dtype=np.bool_)
    smooth[rows[certified]] = True
    faces, face_sources, segments, segment_sources = constraints(
        mesh,
        request.cells,
        request.regions,
        facet_classes,
        edge_classes,
        request.protected,
        smooth_facets=smooth,
    )
    if not np.array_equal(faces, request.faces):
        raise RuntimeError(
            "tetra_metric.curved_constraints must retain the constrained face rows; got a different face set"
        )
    labels = np.unique(face_sources[certified]).astype(np.int64)
    claims = np.full((faces.shape[0],), np.nan, dtype=np.float64)
    claims[certified] = deviation[admitted]
    native = TetMeshCurvedSources(
        labels,
        np.full((labels.size,), curved.tolerance, dtype=np.float64),
        np.tile(np.frombuffer(curved.digest, dtype=np.uint8), (labels.size, 1)),
        claims,
    )
    return _CurvedDeclaration(
        _CurvedAuthority(curved, labels), native, face_sources, segments, segment_sources
    )


class _TetraMetricSourcePreparation(StrictModule):
    source_tables: TetraMetricSource
    exact: (
        _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource
        | None
    )
    declared: TetMeshSourceComplex | None
    protected_keys: set[tuple[int, int]]
    face_sources: np.ndarray
    segments: np.ndarray
    segment_sources: np.ndarray
    curved: _CurvedDeclaration | None = None


def _prepare_tetra_metric_source(
    mesh: CellMesh,
    request: _TetraMetricInput,
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None,
    facet_classes: ArrayLike | None,
    edge_classes: ArrayLike | None,
    curved_boundary: CurvedBoundarySource | None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    maximum_work_units: int,
    maximum_cavity_cells: int,
) -> _TetraMetricSourcePreparation:
    declared, exact, curved = None, None, None
    face_sources, segments, segment_sources = (
        request.face_sources,
        request.segments,
        request.segment_sources,
    )
    with _metric_coordinate_phase(
        scope,
        budget,
        maximum_work_units=maximum_work_units,
        maximum_cavity_cells=maximum_cavity_cells,
    ):
        if source_geometry is not None:
            match source_geometry.exact_source:
                case None:
                    pass
                case (
                    _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource() as exact
                ):
                    source_geometry.resolve(mesh, budget=budget)
                    face_sources, segments, segment_sources, declared = (
                        _plc_native_constraints(
                            mesh,
                            exact,
                            request.faces,
                            segments,
                            request.protected,
                            edge_classes,
                            budget=budget,
                        )
                    )
                case _:
                    raise ValueError(
                        "tetra_metric.source_geometry must retain an ExactPlcCellGeometrySource; got another source kind"
                    )
        if curved_boundary is not None:
            if exact is not None:
                raise ValueError(
                    "tetra_metric.curved_boundary must use the own source complex; got an exact PLC source"
                )
            curved = _declare_curved_sources(
                mesh, request, facet_classes, edge_classes, curved_boundary, budget=budget
            )
            if curved is not None:
                face_sources, segments, segment_sources = (
                    curved.face_sources,
                    curved.segments,
                    curved.segment_sources,
                )
        edges = _edges(request.cells)
        edge_rows = entity_keys_owner.key_rows(
            entity_keys_owner.entity_keys(mesh, 1),
            np.sort(np.asarray(mesh.vertex_global_ids)[edges], axis=1),
        )
        protected_keys = {
            tuple(row) for row in edges[request.protected[edge_rows]].tolist()
        }
        source_tables = TetraMetricSource(
            request.points,
            request.cells,
            np.asarray(mesh.vertex_global_ids, dtype=np.int64),
            request.cell_ids,
            source_geometry=source_geometry if exact is not None else None,
            budget=budget,
        )
    return _TetraMetricSourcePreparation(
        source_tables,
        exact,
        declared,
        protected_keys,
        face_sources,
        segments,
        segment_sources,
        curved,
    )


class _TetraMetricPublication(StrictModule):
    edit: _dependency_phydrax_meshing__topology_edit.CellTopologyEdit
    metric: np.ndarray
    successor: (
        _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource
        | None
    )


def _reprove_curved_publication(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    authority: _CurvedAuthority,
    /,
    *,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> None:
    """Recompute every delegated claim and published curved facet from the source.

    Native retained the host claims as evidence only. Publication reissues a
    whole-facet bound for each distinct oriented corner array among all
    retained claims and live curved faces under the original profile,
    tolerance and limits. Bitwise-identical facets share that fresh bound;
    a foreign source or any facet that no longer certifies refuses. Bounds
    are recomputed, not compared bitwise with the retained claim.
    """
    curved = authority.source
    claims = native.curved_claims()
    if np.any(~np.isin(claims.source_ids, authority.face_sources)):
        raise ValueError(
            "tetra_metric.curved_claims must belong to the declared curved sources; got a foreign source"
        )
    live = np.isin(snapshot.face_sources, authority.face_sources)
    count = claims.coordinates.shape[0] + int(np.count_nonzero(live))
    with budget.temporary_scope():
        # Combined corners, integer sorting copies, indices and admission
        # flags coexist with the native-owned claim and published banks.
        budget.reserve(0, 512 * count)
        corners = np.concatenate(
            (claims.coordinates, snapshot.points[snapshot.faces[live]]), axis=0
        )
        _, rows, inverse = np.unique(
            np.asarray(
                [_oriented_facet_bytes(row) for row in corners], dtype=np.dtype("V72")
            ),
            return_index=True,
            return_inverse=True,
        )
        _, certified = curved.certify(corners[rows], budget=budget)
        admitted = certified[inverse[: claims.rows.size]]
        if not np.all(admitted):
            stale = np.flatnonzero(~admitted)
            raise ValueError(
                f"tetra_metric.curved_claims must recertify from the original source; got {stale.size} unproved claim rows {claims.rows[stale[:8]].tolist()!r}"
            )
        published = certified[inverse[claims.rows.size :]]
        if not np.all(published):
            raise ValueError(
                f"tetra_metric.curved_faces must each carry a whole-facet source proof; got {int(np.count_nonzero(~published))} unproved published faces"
            )


def _publish_tetra_metric(
    mesh: CellMesh,
    native: TetMesh3D,
    source: _TetraMetricSourcePreparation,
    outcome: NativeTetraMetricOutcome,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
    maximum_work_units: int,
    maximum_cavity_cells: int,
    maximum_location_pairs: int,
    record_phase: NativeMeshingPhaseRecorder | None,
) -> _TetraMetricPublication:
    successor, target_source_points = None, None
    with _metric_coordinate_phase(
        scope,
        budget,
        maximum_work_units=maximum_work_units,
        maximum_cavity_cells=maximum_cavity_cells,
    ):
        if source.exact is not None:
            live = np.flatnonzero(outcome.snapshot.vertex_dimension >= 0)
            witnesses = native.source_evidence()
            successor = source.exact.with_witnesses(
                witnesses.witness_strata[live],
                witnesses.witness_entities[live],
                witnesses.witness_parameters[live],
                budget=budget,
            )
            target_source_points = successor.prepare(
                outcome.snapshot.points[live], budget=budget
            ).vertices
        if source.curved is not None:
            _reprove_curved_publication(
                native, outcome.snapshot, source.curved.authority, budget=budget
            )
        with measure_phase(record_phase, "lineage_construction"):
            edit, target_metric = _assemble(
                mesh,
                outcome.snapshot,
                outcome.metric,
                outcome.ancestry,
                maximum_location_pairs,
                source_tables=source.source_tables,
                exact_points=target_source_points,
                budget=budget,
            )
    return _TetraMetricPublication(edit, target_metric, successor)


@checked
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
    source_geometry: _dependency_phydrax_discretization__cell_geometry.CellGeometrySpec
    | None = None,
    curved_boundary: CurvedBoundarySource | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> TetraMetricOutcome:
    """Native legal split/directional collapse, metric cavity improvement and relocation.

    Input is an affine tetrahedral carrier. Curved-map admission/publication is
    owned by the adaptation geometry transaction, not inferred here. A
    ``curved_boundary`` declares one explicit smooth source: its facets with
    a whole-facet source proof share curved native sources whose interior
    surface vertices may collapse only under a fresh host proof of every new
    facet; publication reproves every claim and curved face. All edges
    count toward the unit criterion, including immutable features. Resource
    exhaustion is distinct from a legal-operation fixed point.
    """
    if mesh.ambient_dimension != 3 or any(
        block.cell_kind != "tetrahedron" for block in mesh.blocks
    ):
        raise ValueError(
            "tetra_metric.mesh must contain tetrahedron blocks in 3D; got another cell kind or dimension"
        )
    schedule = MetricExecutionSchedule(
        maximum_passes=maximum_passes,
        maximum_vertices=maximum_vertices,
        maximum_cells=maximum_cells,
        maximum_operations=maximum_operations,
        maximum_work_units=maximum_work_units,
        maximum_location_pairs=maximum_location_pairs,
        maximum_cavity_cells=maximum_cavity_cells,
        maximum_cavity_work=maximum_cavity_work,
        maximum_scratch_bytes=maximum_scratch_bytes,
        maximum_arc_work_units=0,
    )
    (
        maximum_passes,
        maximum_vertices,
        maximum_cells,
        maximum_operations,
        maximum_work_units,
        maximum_location_pairs,
        maximum_cavity_cells,
        maximum_cavity_work,
        maximum_scratch_bytes,
    ) = (
        schedule.maximum_passes,
        schedule.maximum_vertices,
        schedule.maximum_cells,
        schedule.maximum_operations,
        schedule.maximum_work_units,
        schedule.maximum_location_pairs,
        schedule.maximum_cavity_cells,
        schedule.maximum_cavity_work,
        schedule.maximum_scratch_bytes,
    )
    stage_source_limits(mesh, maximum_vertices, maximum_cells)
    validate_metric_operation_controls(topology_operations, relocation)
    validate_metric_phase_recorder(record_phase)
    validate_metric_quality_floor(
        minimum_metric_quality,
        message="tetra_metric.minimum_metric_quality must lie in [0, 1]; got a value outside the unit interval",
    )
    request = _prepare_tetra_metric_input(
        mesh,
        metric,
        cell_classes,
        facet_classes,
        edge_classes,
        protected_edges,
        fixed_vertices,
    )
    try:
        source = _prepare_tetra_metric_source(
            mesh,
            request,
            source_geometry,
            facet_classes,
            edge_classes,
            curved_boundary,
            scope,
            budget,
            maximum_work_units,
            maximum_cavity_cells,
        )
        with budget.measurement(), ExitStack() as storage:
            if budget.storage_ticket is None:

                def retain_storage(bytes_upper: int) -> BatchTicket:
                    return scope.batch(0, bytes_upper=bytes_upper)

                storage.enter_context(budget.retained_storage(retain_storage))
            with measure_phase(record_phase, "native_preparation"):
                native = TetMesh3D(
                    request.points,
                    request.cells,
                    request.regions,
                    request.faces,
                    source.face_sources,
                    source.segments,
                    source.segment_sources,
                    boundary_policy="conforming",
                    max_vertices=maximum_vertices,
                    max_tetrahedra=maximum_cells,
                    max_scratch_bytes=maximum_scratch_bytes,
                    source=source.declared,
                    curved=None if source.curved is None else source.curved.native,
                )
            try:
                outcome = execute_native_tetra_metric(
                    native,
                    source.source_tables,
                    request.metric,
                    fixed_vertices=request.fixed,
                    protected_edges=source.protected_keys,
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
                    scope=scope,
                    budget=budget,
                    curved=None if source.curved is None else source.curved.authority,
                )
                publication = _publish_tetra_metric(
                    mesh,
                    native,
                    source,
                    outcome,
                    scope,
                    budget,
                    maximum_work_units,
                    maximum_cavity_cells,
                    maximum_location_pairs,
                    record_phase,
                )
                return TetraMetricOutcome(
                    publication.edit,
                    publication.metric,
                    outcome.evidence,
                    publication.successor,
                )
            finally:
                native.close()
    except _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureResourceError as error:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "tetra_metric.source_actions must fit the coordinate allowance; got resource exhaustion",
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION.value,
            requested=(
                ("maximum_work_units", maximum_work_units),
                ("maximum_scratch_bytes", maximum_scratch_bytes),
            ),
            achieved=(
                ("coordinate_work_units", budget.work_units),
                ("coordinate_peak_bytes_upper", budget.peak_bytes_upper),
            ),
        ) from error
    except MeshcoreError as error:
        if error.status is not MeshcoreStatus.CAPACITY_EXCEEDED:
            raise
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            str(error),
            stage=_dependency_phydrax_meshing__trace.MeshingStageKind.OPTIMIZATION.value,
            requested=(
                ("maximum_work_units", maximum_work_units),
                ("maximum_scratch_bytes", maximum_scratch_bytes),
            ),
            achieved=(
                ("coordinate_work_units", budget.work_units),
                ("coordinate_peak_bytes_upper", budget.peak_bytes_upper),
            ),
        ) from error


@dataclass(frozen=True, slots=True)
class _TangentSizePlan:
    """Retain a sampled final point and its admitted cavity objective."""

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
    source_fraction: float


def _compound_split_plan(
    snapshot: TetMeshArrays,
    metric: np.ndarray,
    cavity: np.ndarray,
    children: np.ndarray,
    first: int,
    second: int,
    witness: np.ndarray,
    witness_fraction: float,
    witness_metric: np.ndarray,
    source_metric: np.ndarray,
    objective: _StatisticalObjective,
    operation_budget: int,
    /,
) -> _TangentSizePlan | None:
    """Optimize an unpublished final split corner within its declared source stratum."""
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
        insertion_seed=_EdgeInsertionSeed(first, second, witness, witness_fraction),
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
    *,
    tangent: bool = True,
) -> tuple[_TangentSizePlan | None, bool, int]:
    """Admit the native insertion cavity before its accepted-state commit.

    ``tangent=False`` offers only the native construction and its balanced
    midpoint; the tangent trajectory search is deferred to a later stage.
    """
    budget = objective.budget
    first, second = edge
    # Every ghost is owned by a constrained external face. A cavity's old
    # cells and at most four cones per cell fit this immutable topology bound.
    capacity = min(
        maximum_cavity_cells, 5 * (snapshot.tetrahedra.shape[0] + snapshot.faces.shape[0])
    )
    if not budget.optimizer_scratch(32 * capacity + 80):
        return None, False, 0
    used = 0
    insertion = None
    for balanced in (False, True):
        if balanced:
            if construction.parameter == 0.5 or operation_budget < 1:
                break
            used = 1
            midpoint = budget.call(
                lambda allowance: budget.native.edge_split_point(
                    first, second, preferred_fraction=0.5, work_limit=allowance
                )
            )
            if midpoint is None:
                break
            sampled = _sample_metric_splits(
                midpoint.position[None], (midpoint,), budget.source_tables, source_metric, budget
            )
            if sampled is None or not np.all(sampled.valid):
                break
            construction, sample = midpoint, sampled.metric[0]
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
                return None, False, used
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
                    used,
                )
        if budget.exhausted:
            return None, False, used
    if budget.exhausted or not tangent:
        return None, False, used
    if insertion is None:
        incident = np.any(snapshot.tetrahedra == first, axis=1) & np.any(
            snapshot.tetrahedra == second, axis=1
        )
        cavity = snapshot.tetrahedra[incident]
        first_half, second_half = cavity.copy(), cavity.copy()
        first_half[first_half == second] = metric.shape[0]
        second_half[second_half == first] = metric.shape[0]
        children = np.concatenate((first_half, second_half))
    else:
        cavity, children = insertion.removed, insertion.proposed
    before = budget.optimizer_evaluations
    plan = _compound_split_plan(
        snapshot,
        metric,
        cavity,
        children,
        first,
        second,
        construction.position,
        construction.parameter,
        sample,
        source_metric,
        objective,
        operation_budget - used,
    )
    evaluations = used + budget.optimizer_evaluations - before
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


class _SplitCandidates(StrictModule):
    edges: np.ndarray
    lengths: np.ndarray
    indices: np.ndarray
    exhausted: bool = eqx.field(static=True)


def _prepare_split_candidates(
    snapshot: TetMeshArrays,
    values: np.ndarray,
    upper_metric_length: float,
    native_budget: _NativeMetricBudget,
    objective: _StatisticalObjective | None,
) -> _SplitCandidates:
    edges = _edges(snapshot.tetrahedra)
    lengths = _lengths(snapshot.points, values, edges)
    candidates = np.argsort(-lengths, kind="stable")
    if objective is not None and objective.aim is not None:
        if not native_budget.optimizer_work(edges.shape[0], 160 * edges.shape[0]):
            return _SplitCandidates(edges, lengths, candidates, True)
        encoded = (
            edges[:, 0].astype(np.uint64) << np.uint64(32)
        ) | edges[:, 1].astype(np.uint64)
        candidates = candidates[
            np.isin(encoded[candidates], objective.offending_edge_keys())
        ]
        return _SplitCandidates(edges, lengths, candidates, False)
    candidates = candidates[lengths[candidates] > upper_metric_length]
    if objective is not None:
        # Interior insertion degrees of freedom may improve size statistics
        # when subdividing only constrained curve edges cannot realize them.
        scratch = 8 * (20 * edges.shape[0] + 12 * snapshot.faces.shape[0])
        if not native_budget.optimizer_work(
            edges.shape[0] + 3 * snapshot.faces.shape[0], scratch
        ):
            return _SplitCandidates(edges, lengths, candidates, True)
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
    return _SplitCandidates(edges, lengths, candidates, False)


class _SplitConstructions(StrictModule):
    proposals: tuple[tuple[int, int], ...] = eqx.field(static=True)
    points: np.ndarray | None
    fractions: tuple[float, ...] = eqx.field(static=True)
    constructions: tuple[TetMeshEdgeConstruction, ...]
    work: int = eqx.field(static=True)
    rejected: int = eqx.field(static=True)
    exhausted: bool = eqx.field(static=True)


def _construct_metric_splits(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    candidates: _SplitCandidates,
    protected: set[tuple[int, int]],
    budget: int,
    native_budget: _NativeMetricBudget,
    upper_metric_length: float,
    objective: _StatisticalObjective | None,
) -> _SplitConstructions:
    proposals: list[tuple[int, int]] = []
    positions: list[np.ndarray] = []
    fractions: list[float] = []
    constructions: list[TetMeshEdgeConstruction] = []
    work, rejected, exhausted = 0, 0, False
    for index in candidates.indices:
        if work >= budget:
            exhausted = True
            break
        first, second = candidates.edges[index].tolist()
        if (first, second) in protected:
            rejected += 1
            continue
        preferred_fraction = None
        if objective is not None:
            key = (np.uint64(first) << np.uint64(32)) | np.uint64(second)
            row = int(np.searchsorted(objective.edges.keys, key))
            if row >= objective.edges.keys.size or objective.edges.keys[row] != key:
                raise RuntimeError(
                    "tetra_metric.split_preference must bind a physical edge; got a missing edge key"
                )
            if candidates.lengths[index] > upper_metric_length:
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
                f"tetra_metric.split_parameter must lie strictly inside the edge; got {fraction!r}"
            )
        proposals.append((first, second))
        positions.append(point)
        fractions.append(fraction)
        constructions.append(construction)
    return _SplitConstructions(
        tuple(proposals),
        np.stack(positions) if positions and not native_budget.exhausted else None,
        tuple(fractions),
        tuple(constructions),
        work,
        rejected,
        exhausted or native_budget.exhausted,
    )


class _SplitSamples(StrictModule):
    metric: np.ndarray
    valid: np.ndarray


def _sample_metric_splits(
    points: np.ndarray,
    constructions: tuple[TetMeshEdgeConstruction, ...],
    source_tables: TetraMetricSource,
    source_metric: np.ndarray,
    native_budget: _NativeMetricBudget,
) -> _SplitSamples | None:
    authoritative = None
    if source_tables.source_geometry is not None:
        original_source = source_tables.source_geometry.exact_source
        if not isinstance(
            original_source,
            _dependency_phydrax_discretization__exact_plc_geometry.ExactPlcCellGeometrySource,
        ):
            raise RuntimeError(
                "TetraMetricSource.source_geometry must retain an ExactPlcCellGeometrySource; got another source kind"
            )
        authoritative = original_source.prepare_witnesses(
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
            budget=native_budget.coordinate_budget,
        ).vertices
    stencil = native_budget.stencil(points, exact_points=authoritative)
    if stencil is None:
        return None
    source_rows = entity_keys_owner.key_rows(
        source_tables.vertex_global_ids[:, None],
        stencil[0].reshape((-1, 1)),
    ).reshape(stencil[0].shape)
    return _SplitSamples(_interpolate(source_metric[source_rows], stencil[1]), stencil[2])


def _exact_on_edge(first: np.ndarray, second: np.ndarray, point: np.ndarray, /) -> bool:
    """Whether binary64 ``point`` lies exactly strictly inside segment first-second."""
    a = [Fraction(float(value)) for value in first]
    b = [Fraction(float(value)) for value in second]
    p = [Fraction(float(value)) for value in point]
    direction = [b[axis] - a[axis] for axis in range(3)]
    offset = [p[axis] - a[axis] for axis in range(3)]
    cross = (
        direction[1] * offset[2] - direction[2] * offset[1],
        direction[2] * offset[0] - direction[0] * offset[2],
        direction[0] * offset[1] - direction[1] * offset[0],
    )
    along = sum(direction[axis] * offset[axis] for axis in range(3))
    length = sum(direction[axis] * direction[axis] for axis in range(3))
    return all(value == 0 for value in cross) and 0 < along < length


def _insertion_support(
    ancestry: _Ancestry,
    before: np.ndarray,
    after: np.ndarray,
    points: np.ndarray,
    edge: tuple[int, int],
    position: np.ndarray,
    /,
) -> set[int]:
    """Construction support of a split vertex from its actual geometry.

    A point exactly inside its edge refines that edge. A rounded off-edge
    carrier never claims nested edge/face parents: its support is the actual
    replaced cavity, so only the conservative cell-union relation remains.
    """
    first, second = edge
    if _exact_on_edge(points[first], points[second], position):
        return ancestry.vertex_sources[first] | ancestry.vertex_sources[second]
    row_bytes = np.dtype((np.void, 4 * np.dtype(np.int64).itemsize))
    old = np.ascontiguousarray(np.sort(before, axis=1), dtype=np.int64).view(row_bytes)
    new = np.ascontiguousarray(np.sort(after, axis=1), dtype=np.int64).view(row_bytes)
    replaced = np.unique(before[~np.isin(old.reshape(-1), new.reshape(-1))])
    return set().union(
        *(ancestry.vertex_sources[vertex] for vertex in replaced.tolist())
    )




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
    *,
    minimum_quality: float,
    metric_band: tuple[float, float],
) -> tuple[TetMeshArrays, np.ndarray, int, int, int, bool, tuple[float, ...], int]:
    candidates = _prepare_split_candidates(
        snapshot, values, upper_metric_length, native_budget, objective
    )
    if candidates.exhausted or candidates.indices.size == 0:
        return snapshot, values, 0, 0, 0, candidates.exhausted, (), 0
    applied, work, rejected, compounds = 0, 0, 0, 0
    accepted: list[float] = []
    exhausted = False
    # Preserve the initial deterministic candidate order, but construct/source-
    # sample only the next bounded bank. Accepted edits can remove later edges,
    # and a completed goal need not query the unvisited tail. A statistical
    # pass first offers every live candidate its native construction and
    # balanced midpoint, then revisits the still-live candidates with the
    # tangent trajectory search, so costly searches never precede cheap
    # admissible insertions elsewhere. Every requested proposal remains.
    stages = (False, True) if objective is not None else (True,)
    for tangent in stages:
        for start in range(0, candidates.indices.size, 64):
            native_budget.check_deadline()
            if work >= budget or native_budget.exhausted:
                exhausted = True
                break
            indices = candidates.indices[start : start + 64]
            live_keys, _ = _edge_incidence(snapshot.tetrahedra)
            edges = candidates.edges[indices].astype(np.uint64)
            keys = (edges[:, 0] << np.uint64(32)) | edges[:, 1]
            indices = indices[np.isin(keys, live_keys)]
            if indices.size == 0:
                continue
            batch = _SplitCandidates(candidates.edges, candidates.lengths, indices, False)
            (
                snapshot, values, changes, attempts, refusals, exhausted, parameters, compound,
            ) = _split_batch(
                native, snapshot, values, source_tables, source_metric, ancestry,
                protected, budget - work, maximum_vertices, maximum_cells,
                maximum_cavity_cells, native_budget, upper_metric_length, objective, batch,
                tangent=tangent,
                minimum_quality=minimum_quality,
                metric_band=metric_band,
            )
            applied += changes
            work += attempts
            rejected += refusals
            compounds += compound
            accepted.extend(parameters)
            if exhausted or (
                objective is not None and changes and objective.complete(snapshot, values)
            ):
                break
        if exhausted or (
            objective is not None and applied and objective.complete(snapshot, values)
        ):
            break
    return snapshot, values, applied, work, rejected, exhausted, tuple(accepted), compounds


def _split_batch(
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
    candidates: _SplitCandidates,
    /,
    *,
    minimum_quality: float,
    metric_band: tuple[float, float],
    tangent: bool = True,
) -> tuple[TetMeshArrays, np.ndarray, int, int, int, bool, tuple[float, ...], int]:
    constructed = _construct_metric_splits(
        native,
        snapshot,
        candidates,
        protected,
        budget,
        native_budget,
        upper_metric_length,
        objective,
    )
    work, rejected, exhausted = (
        constructed.work,
        constructed.rejected,
        constructed.exhausted,
    )
    points = constructed.points
    if points is None:
        return snapshot, values, 0, work, rejected, exhausted, (), 0
    sampled = _sample_metric_splits(
        points, constructed.constructions, source_tables, source_metric, native_budget
    )
    if sampled is None:
        return snapshot, values, 0, work, rejected, True, (), 0
    proposals, fractions, constructions = (
        constructed.proposals,
        constructed.fractions,
        constructed.constructions,
    )
    samples = sampled.metric
    applied = 0
    compounds = 0
    accepted: list[float] = []
    for index, (first, second) in enumerate(proposals):
        if not np.all(sampled.valid[index]):
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
                tangent=tangent,
            )
            work += evaluations
            if plan is None:
                rejected += 1
                if native_budget.exhausted:
                    exhausted = True
                    break
                continue
        if plan is None:
            capacity = min(
                maximum_cavity_cells,
                5 * (snapshot.tetrahedra.shape[0] + snapshot.faces.shape[0]),
            )
            if not native_budget.optimizer_scratch(32 * capacity + 80):
                exhausted = True
                break
            insertion = native_budget.call(
                lambda allowance: native.inspect_edge_insertion(
                    first, second, points[index], points[index],
                    source_fraction=fractions[index],
                    maximum_cavity_cells=capacity, work_limit=allowance,
                )
            )
            if insertion is None:
                rejected += 1
                if native_budget.exhausted:
                    exhausted = True
                    break
                continue
            target_cells = (
                before.shape[0] - insertion.removed.shape[0] + insertion.proposed.shape[0]
            )
            if target_cells > maximum_cells:
                exhausted = True
                break
            # Ordinary metric-band insertion follows the same strict descent
            # strategy as collapse/reconnection. This is an admission choice,
            # not a claim that every intermediate worsening step is invalid.
            # Inspect the complete native cavity and use the already issued
            # source sample once; rejected trials never mutate the mesh.
            if not _metric_cavity_merit(
                snapshot, values, insertion.removed, insertion.proposed,
                minimum_quality, metric_band, native_budget,
                changed_vertex=values.shape[0], changed_position=points[index],
                changed_metric=samples[index],
                scratch_extra=insertion.buffer_bytes,
                refinement=True,
                required_long_split=(
                    constructions[index].witness_stratum == 0
                    and not np.any(
                        np.any(snapshot.faces == first, axis=1)
                        & np.any(snapshot.faces == second, axis=1)
                    )
                ),
            ):
                rejected += 1
                if native_budget.exhausted:
                    exhausted = True
                    break
                continue
        else:
            insertion = plan.insertion
        inserted = native_budget.call(
            lambda allowance: native.commit_edge_insertion(
                insertion, target_size=0.0, work_limit=allowance,
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
            raise RuntimeError(
                f"tetra_metric.inserted_vertex must append to retained vertex rows; got row {inserted!r} for count {values.shape[0]}"
            )
        sample = samples[index] if plan is None else plan.metric
        values = np.concatenate((values, sample[None]))
        compounds += int(compound)
        if insertion.removed.shape[0] > number:
            native_budget.expanded_insertions += 1
        points_before = snapshot.points
        previous_faces, previous_sources = snapshot.faces, snapshot.face_sources
        snapshot = native.arrays()
        removed, proposed = insertion.removed, insertion.proposed
        if ancestry.constrained_faces is not None:
            ancestry.constrained_faces = _insert_constrained_faces(
                ancestry.constrained_faces, previous_faces, previous_sources,
                snapshot.faces, snapshot.face_sources, removed, proposed, int(inserted),
            )
        if objective is None:
            native_budget.commit_band(snapshot, removed, proposed)
        ancestry.vertex_sources.append(
            plan.source_vertices
            if compound and plan is not None
            else _insertion_support(
                ancestry,
                removed,
                proposed,
                points_before,
                (first, second),
                points[index] if plan is None else plan.position,
            )
        )
        _record(ancestry, removed, proposed, EntityLineageKind.REFINED_FROM)
        accepted.append(fractions[index] if plan is None else plan.insertion.source_fraction)
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


def _metric_cavity_merit(
    snapshot: TetMeshArrays,
    values: np.ndarray,
    removed: np.ndarray,
    proposed: np.ndarray,
    minimum_quality: float,
    metric_band: tuple[float, float],
    native_budget: _NativeMetricBudget,
    /,
    *,
    quality: tuple[float, float] | None = None,
    changed_vertex: int | None = None,
    changed_position: np.ndarray | None = None,
    changed_metric: np.ndarray | None = None,
    scratch_extra: int = 0,
    refinement: bool = False,
    required_long_split: bool = False,
) -> bool:
    """Compare the global edge delta under the epoch's unit-band goal."""
    scratch = 8 * (
        64 * snapshot.tetrahedra.shape[0]
        + 128 * (removed.shape[0] + proposed.shape[0])
        + 32 * snapshot.points.shape[0]
    ) + scratch_extra
    if native_budget.band_snapshot is not snapshot:
        if not native_budget.optimizer_work(12 * snapshot.tetrahedra.shape[0], scratch):
            return False
        keys, counts = _edge_incidence(snapshot.tetrahedra)
        native_budget.band_incidence = dict(
            zip(keys.tolist(), counts.tolist(), strict=True)
        )
        native_budget.band_snapshot = snapshot
    incidence = native_budget.band_incidence
    if incidence is None:
        raise RuntimeError(
            "tetra_metric.band_incidence must bind the accepted state; got no incidence"
        )
    if not native_budget.optimizer_work(
        12 * (removed.shape[0] + proposed.shape[0]), scratch
    ):
        return False
    old_keys, old_counts = _edge_incidence(removed)
    new_keys, new_counts = _edge_incidence(proposed)
    affected = np.union1d(old_keys, new_keys)
    before = np.asarray(
        [incidence.get(key, 0) for key in affected.tolist()], dtype=np.int64
    )
    present = before > 0
    after = before.copy()
    after[np.searchsorted(affected, old_keys)] -= old_counts
    after[np.searchsorted(affected, new_keys)] += new_counts
    if np.any(after < 0):
        raise RuntimeError(
            "tetra_metric.band_incidence must remain nonnegative; got a negative count"
        )
    if not native_budget.optimizer_work(affected.size, scratch):
        return False
    changed = (present & (after == 0)) | (~present & (after > 0))
    edges = _edge_rows(affected[changed])
    if changed_vertex is None:
        points, metric = snapshot.points, values
        old_cells, new_cells = removed, proposed
    else:
        vertices = np.unique(np.concatenate((removed, proposed)).reshape(-1))
        points = _proposal_rows(snapshot.points, vertices, changed_vertex, changed_position)
        metric = _proposal_rows(values, vertices, changed_vertex, changed_metric)
        old_cells = np.searchsorted(vertices, removed).astype(np.int32)
        new_cells = np.searchsorted(vertices, proposed).astype(np.int32)
        edges = np.searchsorted(vertices, edges).astype(np.int32)
    lengths = (
        _lengths(points, metric, edges)
        if edges.size else np.empty((0,), dtype=np.float64)
    )
    old_quality, new_quality = (
        (
            float(np.min(_quality(points, metric, old_cells))),
            float(np.min(_quality(points, metric, new_cells))),
        )
        if quality is None else quality
    )
    if new_quality < minimum_quality:
        return False
    if refinement and not native_budget.optimizer_work(
        lengths.size, scratch + 64 * lengths.size, evaluation=False
    ):
        # Rank only the bounded changed unique-edge bank, not global edges.
        return False
    return _metric_relocation_merit(
        lengths[present[changed]],
        lengths[~present[changed]],
        old_quality,
        new_quality,
        minimum_quality,
        metric_band,
        refinement=refinement,
        required_long_split=required_long_split,
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
    *,
    metric_band: tuple[float, float],
    curved: _CurvedAuthority | None = None,
) -> tuple[TetMeshArrays, int, int, int]:
    """Directional collapses; curved surface vertices need a fresh facet proof.

    A surface vertex whose incident constrained faces all belong to declared
    curved sources is offered only through native inspection, the host's
    whole-facet source proof of every new face and the claimed native commit.
    """
    edges = _edges(snapshot.tetrahedra)
    lengths = _lengths(snapshot.points, values, edges)
    applied, work, rejected = 0, 0, 0
    workset = _ReconnectionWorkset.prepare(snapshot, native_budget)
    if workset is None:
        return snapshot, applied, work, rejected
    for index in np.argsort(lengths, kind="stable"):
        if (
            lengths[index] >= lower_metric_length
            or work >= budget
            or native_budget.exhausted
        ):
            break
        for removed, kept in (edges[index].tolist(), edges[index, ::-1].tolist()):
            native_budget.check_deadline()
            if (
                snapshot.vertex_dimension[kept] < 0
                or snapshot.vertex_dimension[kept] > snapshot.vertex_dimension[removed]
            ):
                continue
            surface_faces = (
                workset.face_rows(removed)
                if snapshot.vertex_dimension[removed] == 2
                else np.empty((0,), dtype=np.int64)
            )
            curved_surface = (
                curved is not None
                and surface_faces.size > 0
                and bool(
                    np.all(
                        np.isin(snapshot.face_sources[surface_faces], curved.face_sources)
                    )
                )
            )
            eligible_dimension = (
                snapshot.vertex_dimension[removed] == 3 or curved_surface
                if objective is None
                else snapshot.vertex_dimension[removed] >= 1
            )
            if not eligible_dimension or (removed < fixed.size and fixed[removed]):
                continue
            if (
                snapshot.vertex_dimension[removed] == 1
                and tuple(sorted((removed, kept))) not in workset.constrained_segments
            ):
                continue
            changed = workset.rows((removed,))
            incident = workset.rows((removed, kept))
            if incident.size == 0:
                continue
            if not curved_surface and not _collapse_coplanar(
                snapshot.points, snapshot.faces[workset.face_rows(removed)], kept
            ):
                rejected += 1
                continue
            cavity_cells = 2 * changed.size
            if cavity_cells > maximum_cavity_cells:
                raise _leaf_phydrax_meshing__failures.MeshingFailure(
                    _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                    "A metric collapse exceeds its cavity-cell budget.",
                    stage=_dependency_phydrax_meshing__trace.MeshingStageKind.TOPOLOGY_REPAIR.value,
                    requested=(("cavity_cells", float(maximum_cavity_cells)),),
                    achieved=(("cavity_cells", float(cavity_cells)),),
                )
            proposed = _collapse_cells(snapshot.tetrahedra[changed], removed, kept)
            if proposed.size == 0 or not np.all(_positive(snapshot.points, proposed)):
                rejected += 1
                continue
            statistical = None
            if objective is None:
                if not _metric_cavity_merit(
                    snapshot,
                    values,
                    snapshot.tetrahedra[changed],
                    proposed,
                    minimum_quality,
                    metric_band,
                    native_budget,
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
            if curved_surface and curved is not None:
                # Positivity and cavity merit are independent necessary gates.
                # Do not query the source for a delta they already rule out;
                # every remaining curved candidate still earns its own proof.
                refuted = _refute_curved_collapse(
                    snapshot,
                    removed,
                    kept,
                    snapshot.faces[surface_faces],
                    curved,
                    native_budget,
                )
                if refuted is None:
                    break
                if refuted:
                    rejected += 1
                    continue
            work += 1
            before = snapshot.tetrahedra
            if curved_surface and curved is not None:
                accepted = _commit_curved_collapse(
                    native,
                    snapshot,
                    removed,
                    kept,
                    surface_faces.size,
                    curved,
                    native_budget,
                )
                if accepted is None:
                    break
            else:
                accepted = native_budget.call(
                    lambda allowance: native.collapse_edge(
                        removed, kept, work_limit=allowance
                    )
                )
            if not accepted:
                rejected += 1
                continue
            if ancestry.constrained_faces is not None:
                ancestry.constrained_faces = _collapse_constrained_faces(
                    ancestry.constrained_faces, int(removed), int(kept)
                )
            if curved_surface:
                ancestry.curved_collapses += 1
            snapshot = native.arrays()
            if ancestry.constrained_faces is not None and set(
                tuple(sorted(face)) for face in snapshot.faces.tolist()
            ) != ancestry.constrained_faces.keys():
                raise ValueError(
                    "tetra_metric.collapse_face_lineage must match the actual committed faces"
                )
            native_budget.commit_band(snapshot, before[changed], proposed)
            if ancestry.vertex_successor is None:
                ancestry.vertex_successor = {}
            ancestry.vertex_successor[removed] = kept
            _record(
                ancestry, before[changed], proposed, EntityLineageKind.COLLAPSED_INTO
            )
            applied += 1
            if objective is not None and statistical is not None:
                objective.commit(statistical)
                if objective.complete(snapshot, values):
                    return snapshot, applied, work, rejected
            workset = _ReconnectionWorkset.prepare(snapshot, native_budget, workset)
            if workset is None:
                return snapshot, applied, work, rejected
            break
    return snapshot, applied, work, rejected


def _refute_curved_collapse(
    snapshot: TetMeshArrays,
    removed: int,
    kept: int,
    star: np.ndarray,
    curved: _CurvedAuthority,
    native_budget: _NativeMetricBudget,
    /,
) -> bool | None:
    """Whether a facet the collapse would create is refuted by its source.

    The collapse of ``removed`` onto ``kept`` replaces ``removed`` in every
    face of its constrained ``star`` that does not contain ``kept``. A
    certified facet-point witness outside the fidelity band proves such a facet
    can never be admitted. None means the shared epoch allowance is exhausted.
    """
    created = star[~np.any(star == kept, axis=1)]
    refuted = native_budget.refute_curved(
        curved.source, snapshot.points[np.where(created == removed, kept, created)]
    )
    return None if refuted is None else bool(np.any(refuted))


def _commit_curved_collapse(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    removed: int,
    kept: int,
    maximum_faces: int,
    curved: _CurvedAuthority,
    native_budget: _NativeMetricBudget,
    /,
) -> bool | None:
    """Inspect natively, prove every new whole facet from the source, then commit.

    None means the shared epoch allowance is exhausted. The proof binds the
    exact inspected faces, their bitwise current coordinates, the accepted
    generation and the declared source digest; native refuses any change.
    """
    plan = native_budget.call(
        lambda allowance: native.inspect_curved_collapse(
            removed, kept, maximum_faces=maximum_faces, work_limit=allowance
        )
    )
    if native_budget.exhausted:
        return None
    if plan is None or plan.source not in curved.face_sources:
        return False
    triangles = snapshot.points[plan.faces]
    proof = native_budget.certify_curved(curved.source, triangles)
    if proof is None:
        return None
    deviation, admitted = proof
    if not np.all(admitted):
        return False
    accepted = native_budget.call(
        lambda allowance: native.collapse_curved_edge(
            plan,
            triangles,
            deviation,
            digest=curved.source.digest,
            work_limit=allowance,
        )
    )
    if native_budget.exhausted:
        return None
    return bool(accepted)


@dataclass(frozen=True, slots=True)
class _ReconnectionWorkset:
    """Snapshot-local incidence keyed by stable, oriented native cell tuples."""

    vertex_cells: tuple[set[tuple[int, ...]], ...]
    cell_rows: dict[tuple[int, ...], int]
    faces: np.ndarray
    segments: np.ndarray
    vertex_faces: tuple[set[int], ...]
    constrained_faces: set[tuple[int, ...]]
    constrained_edges: set[tuple[int, ...]]
    constrained_segments: set[tuple[int, ...]]

    @classmethod
    def prepare(
        cls,
        snapshot: TetMeshArrays,
        budget: _NativeMetricBudget,
        /,
        previous: _ReconnectionWorkset | None = None,
    ) -> _ReconnectionWorkset | None:
        # Include Python set entries and the old bank retained during replacement.
        scratch = 512 * (
            2 * snapshot.points.shape[0]
            + 4 * snapshot.tetrahedra.shape[0]
            + 6 * snapshot.faces.shape[0]
            + snapshot.segments.shape[0]
        )
        if not budget.optimizer_scratch(scratch):
            return None
        cell_rows = {
            tuple(cell): row for row, cell in enumerate(snapshot.tetrahedra.tolist())
        }
        budget.check_deadline()
        if previous is None:
            incidence: list[set[tuple[int, ...]]] = [
                set() for _ in range(snapshot.points.shape[0])
            ]
            for cell in cell_rows:
                for vertex in cell:
                    incidence[vertex].add(cell)
        else:
            # Native commits globally reorder rows. Stable tuple incidence
            # survives that permutation; only the row lookup is replaced.
            incidence = list(previous.vertex_cells)
            incidence.extend(set() for _ in range(len(incidence), snapshot.points.shape[0]))
            removed = previous.cell_rows.keys() - cell_rows.keys()
            added = cell_rows.keys() - previous.cell_rows.keys()
            affected = {vertex for cell in removed | added for vertex in cell}
            for vertex in affected:
                incidence[vertex] = incidence[vertex].copy()
            for cell in removed:
                for vertex in cell:
                    incidence[vertex].remove(cell)
            for cell in added:
                for vertex in cell:
                    incidence[vertex].add(cell)
            if (
                len(previous.vertex_faces) == snapshot.points.shape[0]
                and np.array_equal(previous.faces, snapshot.faces)
                and np.array_equal(previous.segments, snapshot.segments)
            ):
                return cls(
                    tuple(incidence), cell_rows, previous.faces, previous.segments,
                    previous.vertex_faces, previous.constrained_faces,
                    previous.constrained_edges, previous.constrained_segments,
                )
        face_incidence: tuple[set[int], ...] = tuple(
            set() for _ in range(snapshot.points.shape[0])
        )
        for row, face in enumerate(snapshot.faces.tolist()):
            if row % 256 == 0:
                budget.check_deadline()
            for vertex in face:
                face_incidence[vertex].add(row)
        faces = {tuple(sorted(face)) for face in snapshot.faces.tolist()}
        segments = {tuple(sorted(edge)) for edge in snapshot.segments.tolist()}
        edges = set(segments)
        for face in faces:
            edges.update(combinations(face, 2))
        return cls(
            tuple(incidence), cell_rows, snapshot.faces.copy(), snapshot.segments.copy(),
            face_incidence, faces, edges, segments,
        )

    def rows(self, vertices: tuple[int, ...], /) -> np.ndarray:
        incident = self.vertex_cells[vertices[0]].intersection(
            *(self.vertex_cells[vertex] for vertex in vertices[1:])
        )
        return np.asarray(sorted(self.cell_rows[cell] for cell in incident), dtype=np.int64)

    def face_rows(self, vertex: int, /) -> np.ndarray:
        return np.asarray(sorted(self.vertex_faces[vertex]), dtype=np.int64)


def _flip_pass(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    ancestry: _Ancestry,
    budget: int,
    native_budget: _NativeMetricBudget,
    objective: _StatisticalObjective | None,
    /,
    *,
    minimum_quality: float,
    metric_band: tuple[float, float],
    focus: np.ndarray | None = None,
) -> tuple[TetMeshArrays, int, int]:
    workset = _ReconnectionWorkset.prepare(snapshot, native_budget)
    if workset is None:
        return snapshot, 0, 0
    cells = snapshot.tetrahedra if focus is None else focus
    faces = np.unique(np.sort(cells[:, _FACES].reshape((-1, 3)), axis=1), axis=0)
    applied, work = 0, 0
    for face in faces:
        if work >= budget or native_budget.exhausted:
            break
        native_budget.check_deadline()
        key = tuple(face.tolist())
        if key in workset.constrained_faces:
            continue
        rows = workset.rows(key)
        if rows.size != 2 or np.unique(snapshot.tetrahedron_regions[rows]).size != 1:
            continue
        cavity = snapshot.tetrahedra[rows]
        apex = cavity[~np.isin(cavity, face)]
        proposed, positive = _oriented_positive(
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
        if not np.all(positive):
            continue
        statistical = None
        if objective is None:
            quality = (
                float(np.min(_quality(snapshot.points, values, cavity))),
                float(np.min(_quality(snapshot.points, values, proposed))),
            )
            # A 2-to-3 flip retains every old edge, so band deficit cannot fall.
            if quality[1] < minimum_quality or quality[1] <= quality[0] + 1.0e-6:
                continue
            if not _metric_cavity_merit(
                snapshot,
                values,
                cavity,
                proposed,
                minimum_quality,
                metric_band,
                native_budget,
                quality=quality,
            ):
                continue
        else:
            statistical = objective.propose(snapshot, values, cavity, proposed)
            if statistical is None:
                continue
        work += 1
        if native_budget.call(
            lambda _: native.flip_face(face, require_improvement=False)
        ):
            snapshot = native.arrays()
            native_budget.commit_band(snapshot, cavity, proposed)
            _record(ancestry, cavity, proposed, EntityLineageKind.SWAPPED_FROM)
            applied += 1
            if objective is not None and statistical is not None:
                objective.commit(statistical)
                if objective.complete(snapshot, values):
                    break
            workset = _ReconnectionWorkset.prepare(snapshot, native_budget, workset)
            if workset is None:
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


class _RingCandidates(StrictModule):
    proposed: np.ndarray
    positive: np.ndarray
    quality: np.ndarray
    cell_quality: np.ndarray


def _prepare_ring_candidates(
    snapshot: TetMeshArrays,
    values: np.ndarray,
    triangles: tuple[tuple[tuple[int, int, int], ...], ...],
    first: int,
    second: int,
    cell_count: int,
) -> _RingCandidates:
    proposed = np.asarray(
        [
            [(apex, *triangle) for triangle in triangulation for apex in (first, second)]
            for triangulation in triangles
        ],
        dtype=np.int32,
    )
    # Catalan alternatives reuse the same oriented cone cells. Certify each
    # distinct row once, then restore the original candidate and cell order.
    unique, inverse = np.unique(
        proposed.reshape((-1, 4)), axis=0, return_inverse=True
    )
    oriented, positive = _oriented_positive(snapshot.points, unique)
    quality = _quality(snapshot.points, values, oriented)
    proposed = oriented[inverse].reshape(proposed.shape)
    positive = positive[inverse].reshape((-1, cell_count)).all(axis=1)
    cell_quality = quality[inverse].reshape((-1, cell_count))
    return _RingCandidates(proposed, positive, cell_quality.min(axis=1), cell_quality)


class _RingCommit(StrictModule):
    snapshot: TetMeshArrays
    applied: bool = eqx.field(static=True)
    complete: bool = eqx.field(static=True)


def _commit_ring_reconnection(
    native: TetMesh3D,
    snapshot: TetMeshArrays,
    values: np.ndarray,
    ancestry: _Ancestry,
    cavity: np.ndarray,
    candidates: _RingCandidates,
    native_budget: _NativeMetricBudget,
    objective: _StatisticalObjective | None,
    *,
    minimum_quality: float,
    metric_band: tuple[float, float],
    removed_edge_length: float,
) -> _RingCommit:
    removed_quality = _quality(snapshot.points, values, cavity)
    worst = float(np.min(removed_quality))
    order = np.argsort(-candidates.quality, kind="stable")
    for candidate in order:
        if not candidates.positive[candidate]:
            continue
        statistical = None
        proposed = candidates.proposed[candidate]
        if objective is None:
            proposed_quality = float(candidates.quality[candidate])
            if proposed_quality < minimum_quality:
                break
            # Only the central edge disappears. An in-band central edge cannot
            # reduce deficit, irrespective of the polygon's new diagonals.
            if (
                metric_band[0] <= removed_edge_length <= metric_band[1]
                and proposed_quality <= worst + 1.0e-6
            ):
                break
            if not _metric_cavity_merit(
                snapshot,
                values,
                cavity,
                proposed,
                minimum_quality,
                metric_band,
                native_budget,
                quality=(worst, proposed_quality),
            ):
                continue
        else:
            # Candidates are ordered by their minimum new-cell quality, the
            # same values the objective's all-new-cell floor would test.
            if candidates.quality[candidate] < minimum_quality:
                break
            statistical = objective.propose(
                snapshot,
                values,
                cavity,
                proposed,
                proposed_quality=candidates.cell_quality[candidate],
                removed_quality=removed_quality,
            )
            if statistical is None:
                continue
        if native_budget.call(
            lambda allowance: native.reconnect(cavity, proposed, work_limit=allowance)
        ):
            snapshot = native.arrays()
            native_budget.commit_band(snapshot, cavity, proposed)
            _record(ancestry, cavity, proposed, EntityLineageKind.SWAPPED_FROM)
            complete = False
            if objective is not None and statistical is not None:
                objective.commit(statistical)
                complete = objective.complete(snapshot, values)
            return _RingCommit(snapshot, True, complete)
    return _RingCommit(snapshot, False, False)


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
    *,
    minimum_quality: float,
    metric_band: tuple[float, float],
    focus: np.ndarray | None = None,
) -> tuple[TetMeshArrays, int, int, bool]:
    applied, work = 0, 0
    workset = _ReconnectionWorkset.prepare(snapshot, native_budget)
    if workset is None:
        return snapshot, applied, work, True
    edges = _edges(snapshot.tetrahedra if focus is None else focus)
    lengths = _lengths(snapshot.points, values, edges)
    for (first, second), length in zip(edges.tolist(), lengths.tolist(), strict=True):
        if native_budget.exhausted:
            return snapshot, applied, work, True
        native_budget.check_deadline()
        if (first, second) in workset.constrained_edges:
            continue
        rows = workset.rows((first, second))
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
        candidates = _prepare_ring_candidates(
            snapshot, values, triangles, first, second, cell_count
        )
        work += len(triangles)
        committed = _commit_ring_reconnection(
            native,
            snapshot,
            values,
            ancestry,
            cavity,
            candidates,
            native_budget,
            objective,
            minimum_quality=minimum_quality,
            metric_band=metric_band,
            removed_edge_length=length,
        )
        snapshot = committed.snapshot
        applied += int(committed.applied)
        if committed.complete:
            return snapshot, applied, work, False
        if committed.applied:
            workset = _ReconnectionWorkset.prepare(snapshot, native_budget, workset)
            if workset is None:
                return snapshot, applied, work, True
    return snapshot, applied, work, False


@dataclass(frozen=True, slots=True)
class _TangentCoordinates:
    """Retain immutable numerical coordinates for one source stratum."""

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
                if int(sign) != int(
                    _dependency_phydrax__geometry_predicates.PredicateSign.ZERO
                ):
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
            return bool(
                np.all(
                    signs
                    == int(_dependency_phydrax__geometry_predicates.PredicateSign.ZERO)
                )
            )
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
    """Retain the size objective and evidence for hard refusals."""

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
    # Exact projection and new-cell quality of the last evaluated insertion,
    # reused by its admission rather than recomputed.
    projected: _PhysicalEdgeLedger | None = None
    proposed_quality: np.ndarray | None = None

    def evaluate(
        self, coordinates: np.ndarray, /
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parameters = np.asarray(coordinates, dtype=np.float64)
        if self.previous is not None and np.array_equal(parameters, self.previous):
            if self.evaluation is None:
                raise RuntimeError(
                    "tetra_metric.cached_proposal must retain evaluated evidence; got None"
                )
            return self.evaluation
        budget = self.objective.budget
        position = self.coordinates.position(parameters)
        sample = self.initial_metric
        source_vertices = np.empty((0,), dtype=np.int64)
        constraints = np.zeros((5,), dtype=np.float64)
        scratch_bytes = self.scratch_bytes
        self.insertion_plan = None
        self.projected = None
        self.proposed_quality = None
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
                source_fraction=seed.source_fraction,
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
        # The global edge projection (one visit per retained edge) is charged
        # separately, only when the local constraints admit this trial.
        host_work = (
            cavity_edges
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
            source_rows = entity_keys_owner.key_rows(
                budget.source_tables.vertex_global_ids[:, None],
                stencil[0].reshape((-1, 1)),
            ).reshape(stencil[0].shape)
            sample = _interpolate(self.source_metric[source_rows], stencil[1])[0]
            source_vertices = stencil[0][0, stencil[1][0] > 0.0]
        local_metric = _proposal_rows(
            self.metric, vertices, self.coordinates.vertex, sample
        )
        quality = _quality(local_points, local_metric, cells)
        self.proposed_quality = quality
        constraints[2] = self.objective.minimum_quality - float(np.min(quality))
        goal = self.objective.goal
        if positive and (
            goal.maximum_radius_edge is not None
            or goal.minimum_dihedral_degrees is not None
        ):
            # The determinant floor is decided once, with its refusal reason,
            # by the objective proposal that admission always runs.
            shape = budget.call(
                lambda allowance: budget.native.proposal_shape(
                    local_points, cells, vertex_ids=vertices, work_limit=allowance
                )
            )
            if shape is None:
                constraints[3:] = np.inf
            else:
                radius, dihedral, _ = shape
                if goal.maximum_radius_edge is not None:
                    constraints[3] = radius - goal.maximum_radius_edge
                if goal.minimum_dihedral_degrees is not None:
                    constraints[4] = goal.minimum_dihedral_degrees - dihedral
        elif not positive:
            constraints[3:] = 1.0
        physical_refusal: tuple[str, ...] = ()
        value = np.inf
        try:
            if not np.any(constraints > 0.0) and np.all(np.isfinite(constraints)):
                if not budget.optimizer_work(
                    self.objective.edges.keys.size, scratch_bytes, evaluation=False
                ):
                    return np.inf, np.full((5,), np.inf), position, sample, source_vertices
                ledger = self.objective.edges.project(
                    self.snapshot.points,
                    removed,
                    proposed,
                    self.objective.goal,
                    self.coordinates.vertex,
                    position,
                )
                value = ledger.merit.normalized_deficit
                self.projected = ledger
        except _leaf_phydrax_meshing__failures.MeshingFailure as error:
            if (
                error.evidence.category
                is not _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED
            ):
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
# Zero tolerances: each solve ends at zero optimality or step stagnation, a
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


def _equal_length_solve(
    args: _EqualLengthArgs, budget: _NativeMetricBudget, scratch: int, /
) -> LeastSquaresResult | None:
    """Continue the small native LM state under per-dispatch admission."""
    basis, rows = args[1], args[3].shape[0]
    model_work = 2 * rows

    def admit(required: int) -> bool:
        budget.check_deadline()
        remaining = min(
            budget.maximum_work_units - budget.work_units,
            budget.coordinate_budget.ticket.remaining_work_units,
        )
        if required > remaining:
            budget.refuse(
                "The remaining work allowance cannot admit the next tangent solve phase.",
                (("maximum_work_units", float(budget.maximum_work_units)),),
                (
                    ("work_units", float(budget.work_units)),
                    ("remaining_work_units", float(remaining)),
                    ("solve_phase_work_upper", float(required)),
                ),
            )
            return False
        return budget.admit_optimizer_bound(required) and budget.optimizer_scratch(scratch)

    if not admit(2 * model_work):
        return None
    problem = NonlinearLeastSquaresProblem(
        _equal_length_residual, problem_id="source-stratum/equal-length"
    )
    parameters, state = _initialize_metric_least_squares(
        args, problem, jnp.zeros((basis.shape[1],), dtype=basis.dtype),
        _EQUAL_LENGTH_METHOD,
    )
    if not budget.optimizer_work(model_work, scratch):
        return None
    budget.optimizer_objective_evaluations += 1
    step_work = (
        2 + _EQUAL_LENGTH_METHOD.maximum_trials * (2 * basis.shape[1] + 12)
    ) * rows
    if not admit(step_work + model_work):
        return None
    remaining = min(
        budget.maximum_work_units - budget.work_units,
        budget.coordinate_budget.ticket.remaining_work_units,
    )
    if not budget.admit_optimizer_bound(remaining - model_work):
        return None
    previous = state
    parameters, state, refused = _advance_metric_least_squares(
        args, problem, parameters, state, _EQUAL_LENGTH_METHOD,
        _EQUAL_LENGTH_TERMINATION,
        _MetricSolveWork.from_counts(rows, 0, 0, step_work, model_work, remaining),
    )
    residuals = int(state.residual_evaluations - previous.residual_evaluations)
    actions = residuals + int(
        state.jvp_evaluations - previous.jvp_evaluations
        + state.vjp_evaluations - previous.vjp_evaluations
    )
    budget.optimizer_iterations += int(state.iteration - previous.iteration)
    budget.optimizer_objective_evaluations += residuals
    if not budget.optimizer_work(actions * rows, scratch):
        return None
    if bool(refused):
        admit(step_work + model_work)
        return None
    status = OptimizationStatus(int(state.metrics.status))
    evaluation_limit = _EQUAL_LENGTH_TERMINATION.maximum_evaluations
    if status is OptimizationStatus.ITERATING:
        status = (
            OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
            if evaluation_limit is not None and int(state.residual_evaluations) >= evaluation_limit
            else OptimizationStatus.MAXIMUM_STEPS_REACHED
        )
    if not admit(model_work):
        return None
    result = _finish_metric_least_squares(
        args, problem, parameters, state, jnp.asarray(int(status)),
        _EQUAL_LENGTH_METHOD, _EQUAL_LENGTH_TERMINATION,
    )
    if not budget.optimizer_work(model_work, scratch):
        return None
    budget.optimizer_objective_evaluations += 1
    budget.optimizer_status_counts[int(result.status)] += 1
    return result


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
    """Return whether failed target statistics lie above or below the target."""
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
    /,
    *,
    insertion_seed: _EdgeInsertionSeed,
) -> _TangentSizePlan | None:
    """Return the best admitted final insertion point in a source stratum.

    Bounded selections of governing physical edges are solved to the requested
    target length by Levenberg--Marquardt with AD derivatives in the vertex's
    source stratum. Source, embedding, metric-quality, native-shape, and
    statistical-merit admission govern every returned proposal.
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
    remaining_work = min(
        budget.maximum_work_units - budget.work_units,
        budget.coordinate_budget.ticket.remaining_work_units,
    )
    # Candidates are visited sequentially; future edges do not own reservations
    # in this proposal. The evaluator settles actual BVH/native/host work before
    # the next proposal is admitted against the same request ledger.
    maximum_evaluations = min(
        operation_budget,
        remaining_work // (host_work + native_shape_work),
        (budget.maximum_source_pairs - budget.source_candidates[0])
        // budget.source_tables.cells.shape[0],
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
            projected=evaluator.projected,
            proposed_quality=evaluator.proposed_quality,
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

    # The witness was already inspected and rejected by objective.propose in
    # _prepare_statistical_split. Reuse that cavity rather than re-inspecting,
    # source-sampling and shape-testing the identical zero tangent.
    star = proposed
    origin = coordinates.origin
    neighbors = np.unique(star[np.any(star == coordinates.vertex, axis=1)])
    neighbors = neighbors[neighbors != coordinates.vertex]
    deviations = np.abs(
        np.linalg.norm(snapshot.points[neighbors] - origin, axis=1) / target - 1.0
    )
    # Every edge of an inserted vertex is new; deviation beyond the
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
    # Pure pins, and pins with the failed side's one-sided star excess. With no
    # failed target side the one-sided weights vanish, so a sided variant is
    # the identical problem to its pure pin, and the empty selection is the
    # zero residual whose solution is the already rejected witness.
    sides = longer or shorter
    variants = [
        (subset, sided)
        for subset in subsets
        for sided in (
            ((True,) if sides else ())
            if not subset
            else ((False, True) if sides else (False,))
        )
    ]
    evaluated: set[bytes] = set()
    for subset, sided in variants:
        if evaluations >= maximum_evaluations or budget.exhausted:
            break
        pinned = np.zeros((rows,), dtype=np.float64)
        pinned[np.asarray(subset, dtype=np.int64)] = 1.0
        active = np.zeros((rows,), dtype=np.float64)
        active[: neighbors.size] = 1.0 if sided else 0.0
        solve_bytes = 8 * (star_points.size + 3 * dimension + 4 * rows + 8)
        result = _equal_length_solve(
            (
                jnp.asarray(coordinates.origin),
                jnp.asarray(coordinates.basis),
                jnp.asarray(coordinates.scale),
                jnp.asarray(star_points),
                jnp.asarray(target),
                jnp.asarray(pinned),
                jnp.asarray(active * float(longer)),
                jnp.asarray(active * float(shorter)),
            ),
            budget,
            scratch + solve_bytes,
        )
        if result is None:
            return None
        solution = np.asarray(result.parameters, dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            budget.refuse_optimizer("nonfinite_equal_length_solution")
            continue
        # Distinct selections may converge to the identical final point; its
        # deterministic evaluation and admission were already decided.
        key = solution.tobytes()
        if key in evaluated:
            continue
        evaluated.add(key)
        admit(solution)
    if budget.exhausted:
        return None
    if best is None:
        budget.refuse_optimizer("nonimproving_final_proposal")
    return best


class GlobalSizeArgs(NamedTuple):
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
_GLOBAL_SHAPE_PROJECTION_POLICY = ConvexSolvePolicy(
    DensePrimalDualQP(max_kkt_dimension=3 * 32),
    termination=ConvexTermination(
        absolute=_GLOBAL_SIZE_METHOD.linear_policy.tolerance.absolute,
        relative=0.0,
        primal_infeasible=_GLOBAL_SIZE_METHOD.linear_policy.tolerance.absolute,
        dual_infeasible=_GLOBAL_SIZE_METHOD.linear_policy.tolerance.absolute,
        maximum_steps=_GLOBAL_LINEAR_STEPS,
    ),
)


def global_size_canonical_residual_bound() -> int:
    """Bound only real native LM primals, never its separately charged models."""
    trials = _GLOBAL_SIZE_METHOD.maximum_trials
    # One initial model, one model plus at most one primal per trial/step,
    # and one final model. The outer evaluation guard can overshoot by one
    # full step before the final model; there is no auxiliary callback.
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
    # Match the native nanquantile linear-interpolation operation order,
    # rather than replacing it with an algebraically equal rounding.
    statistic = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return statistic, order, lower, upper, fraction


@custom_jvp
def size_quantile(lengths: Array, quantiles: Array) -> Array:
    """Return linear quantiles with a permutation-symmetric tie derivative.

    Equal-value rank blocks use the average rank-permutation derivative.
    One-sided trial derivatives select their outgoing ranks independently.
    """
    return _size_quantile_parts(lengths, quantiles)[0]


@size_quantile.defjvp
def _size_quantile_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    lengths, quantiles = primals
    length_tangent, quantile_tangent = tangents
    statistic, order, lower, upper, fraction = _size_quantile_parts(lengths, quantiles)
    ordered = lengths[order]
    valid = ~jnp.isnan(ordered)
    # Equal floating values share a rank block; adjacent distinct values
    # remain distinct branches with coefficients in the prepared model.
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


_GLOBAL_SIZE_TRIAL_POLICY_ID = "exact-linear-quantile/tie-barycenter/retained-joint-size-deficit-rank-frontier/finite-source-trajectory-hard-shape-bracket/raw-affine-third-trial/outgoing-events"


def _global_size_trial_work(
    vertex_count: int, edge_count: int, cell_count: int, quantile_count: int
) -> tuple[int, int, int, int]:
    sort_work = 4 * edge_count * (edge_count - 1).bit_length() ** 2
    fixed = 4 * vertex_count + 30 * edge_count + 40 * cell_count + sort_work
    model = (
        vertex_count + 8 * edge_count + 400 * cell_count + sort_work + 4 * quantile_count
    )
    event = 100 * quantile_count * edge_count
    return fixed, model, event, 440 * cell_count


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


class GlobalSizeBranchBank(NamedTuple):
    statistic: Array
    lower: Array
    upper: Array
    count: Array
    overflow: Array


class _GlobalSizeJointRows(NamedTuple):
    keys: Array
    gradients: Array
    count: Array
    overflow: Array


_GlobalSizeFrontierCarry: TypeAlias = tuple[
    GlobalSizeBranchBank,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    _GlobalSizeJointRows,
]
_GlobalSizeTrialCarry: TypeAlias = tuple[
    Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
]


def append_global_size_branches(
    bank: GlobalSizeBranchBank,
    statistic: Array,
    lower: Array,
    upper: Array,
    active: Array,
) -> tuple[GlobalSizeBranchBank, Array]:
    """Retain every requested branch or refuse the whole overflowing bank."""
    capacity = bank.statistic.size
    statistic = statistic.astype(bank.statistic.dtype)
    lower = lower.astype(bank.lower.dtype)
    upper = upper.astype(bank.upper.dtype)
    slots = jnp.arange(capacity)

    def append(
        index: Array, carry: tuple[GlobalSizeBranchBank, Array]
    ) -> tuple[GlobalSizeBranchBank, Array]:
        current, checks = carry

        def admitted(_: None) -> tuple[GlobalSizeBranchBank, Array]:
            present = jnp.any(
                (slots < current.count)
                & (current.statistic == statistic[index])
                & (current.lower == lower[index])
                & (current.upper == upper[index])
            )

            def insert(_: None) -> GlobalSizeBranchBank:
                def available(_: None) -> GlobalSizeBranchBank:
                    slot = current.count
                    return GlobalSizeBranchBank(
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


def global_size_branch_gradients(
    bank: GlobalSizeBranchBank,
    delta: Array,
    lengths: Array,
    residual: Array,
    shape_gradient: Array,
    joint_gradients: Array,
    args: GlobalSizeArgs,
) -> tuple[Array, Array, Array]:
    """Return source-tangent equalities, shape rows, and retained size-merit rows."""
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

            return lax.cond(
                statistic < -1,
                lambda _: joint_gradients[lower],
                lambda _: lax.cond(
                    statistic < 0, lambda _: -shape_gradient, size_row, None
                ),
                None,
            )

        return lax.cond(admitted, evaluate, lambda _: jnp.zeros_like(args.origins), None)

    gradients = lax.map(
        lambda values: row(*values), (bank.statistic, bank.lower, bank.upper, active)
    )
    equality = (
        active & (bank.statistic >= 0) & (residual[jnp.maximum(bank.statistic, 0)] == 0.0)
    )
    return gradients, active & ~equality, equality


def _global_size_joint_gradient(
    delta: Array,
    lengths: Array,
    residual: Array,
    lower: Array,
    upper: Array,
    args: GlobalSizeArgs,
    /,
) -> Array:
    """Differentiate the failed-statistic merit on selected rank support pairs."""
    qids = jnp.arange(args.quantiles.size, dtype=jnp.int32)
    bank = GlobalSizeBranchBank(
        qids, lower, upper, jnp.asarray(qids.size), jnp.asarray(False)
    )
    zero = jnp.zeros_like(args.origins)
    gradients, failed, _ = global_size_branch_gradients(
        bank,
        delta,
        lengths,
        residual,
        zero,
        jnp.zeros((1, *args.origins.shape), dtype=args.origins.dtype),
        args,
    )
    return jnp.sum(jnp.where(failed[:, None, None], gradients, 0.0), axis=0)


def _append_global_size_joint_rows(
    bank: _GlobalSizeJointRows,
    lower: Array,
    upper: Array,
    active: Array,
    delta: Array,
    lengths: Array,
    residual: Array,
    remaining_work: Array,
    args: GlobalSizeArgs,
    /,
) -> tuple[_GlobalSizeJointRows, Array, Array, Array]:
    """Retain failed-rank support keys and gradients in one invocation-local bank."""
    capacity = bank.keys.shape[0]
    failed = residual[: args.quantiles.size] != 0.0
    keys = jnp.stack((lower, upper), axis=-1).astype(jnp.int32)
    keys = jnp.where(failed[None, :, None], keys, -1).reshape((lower.shape[0], -1))
    metadata_work = capacity * (keys.shape[1] + 4)
    gradient_work = 10 * args.quantiles.size * args.origins.size

    def append(
        index: Array, carry: tuple[_GlobalSizeJointRows, Array, Array, Array]
    ) -> tuple[_GlobalSizeJointRows, Array, Array, Array]:
        current, destinations, work, resource = carry

        def inspect(_: None) -> tuple[_GlobalSizeJointRows, Array, Array, Array]:
            present = (jnp.arange(capacity) < current.count) & jnp.all(
                current.keys == keys[index], axis=1
            )
            exists = jnp.any(present)
            destination = jnp.argmax(present.astype(jnp.int32)).astype(jnp.int32)
            spent = work + metadata_work

            def retained(_: None) -> tuple[_GlobalSizeJointRows, Array, Array, Array]:
                return current, destinations.at[index].set(destination), spent, resource

            def insert(_: None) -> tuple[_GlobalSizeJointRows, Array, Array, Array]:
                def admitted(_: None) -> tuple[_GlobalSizeJointRows, Array, Array, Array]:
                    gradient = _global_size_joint_gradient(
                        delta, lengths, residual, lower[index], upper[index], args
                    )
                    slot = current.count
                    return (
                        _GlobalSizeJointRows(
                            current.keys.at[slot].set(keys[index]),
                            current.gradients.at[slot].set(gradient),
                            current.count + 1,
                            current.overflow,
                        ),
                        destinations.at[index].set(slot),
                        spent + gradient_work,
                        resource,
                    )

                return lax.cond(
                    current.count < capacity,
                    lambda _: lax.cond(
                        gradient_work <= remaining_work - spent,
                        admitted,
                        lambda _: (
                            current,
                            destinations,
                            spent,
                            jnp.asarray(True),
                        ),
                        None,
                    ),
                    lambda _: (
                        current._replace(overflow=jnp.asarray(True)),
                        destinations,
                        spent,
                        resource,
                    ),
                    None,
                )

            return lax.cond(exists, retained, insert, None)

        return lax.cond(
            active[index] & ~current.overflow & ~resource,
            lambda _: lax.cond(
                metadata_work <= remaining_work - work,
                inspect,
                lambda _: (current, destinations, work, jnp.asarray(True)),
                None,
            ),
            lambda _: carry,
            None,
        )

    return lax.fori_loop(
        0,
        lower.shape[0],
        append,
        (
            bank,
            jnp.zeros((lower.shape[0],), dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(False),
        ),
    )


def global_size_branch_targets(
    bank: GlobalSizeBranchBank,
    residual: Array,
    inequality: Array,
    quantile_count: int,
) -> Array:
    """Return size-deficit or shape progress targets for active merit rows."""
    size_targets = jnp.abs(residual[jnp.maximum(bank.statistic, 0)])
    shape_target = jnp.sum(residual[quantile_count:] ** 2)
    joint_target = jnp.sum(jnp.abs(residual[:quantile_count]))
    targets = jnp.where(
        bank.statistic < -1,
        joint_target,
        jnp.where(bank.statistic < 0, shape_target, size_targets),
    )
    return jnp.where(inequality, targets, 0.0)


def _global_size_raw_shape_rows(
    points: Array, bank: GlobalSizeBranchBank, args: GlobalSizeArgs, /
) -> tuple[Array, Array]:
    """Gather bounded cell-local raw tangents, never a global shape Jacobian."""
    cells = args.cells.shape[0]
    active = (jnp.arange(bank.statistic.size) < bank.count) & (bank.statistic == -4)

    def row(values: tuple[Array, Array]) -> tuple[Array, Array]:
        index, admitted = values

        def evaluate(_: None) -> tuple[Array, Array]:
            cell = jnp.where(index < cells, index, (index - cells) // 6)
            kind = jnp.where(index < cells, 1, 2 + (index - cells) % 6)
            vertices = args.cells[cell]
            local = args._replace(
                cells=jnp.arange(4, dtype=jnp.int32)[None, :],
                cell_valid=jnp.ones((1,), dtype=args.cell_valid.dtype),
                initial_volume=args.initial_volume[cell][None],
            )

            def margin(corners: Array) -> Array:
                return _global_size_shape_margins(corners, local)[kind]

            value, corners = jax.value_and_grad(margin)(points[vertices])
            gradient = jnp.zeros_like(points)
            for corner in range(4):
                vertex = vertices[corner]
                tangent = args.target * jnp.sum(
                    args.basis[vertex] * corners[corner, :, None], axis=0
                )
                gradient = gradient.at[vertex].add(tangent)
            return gradient, value

        return lax.cond(
            admitted,
            evaluate,
            lambda _: (jnp.zeros_like(points), jnp.asarray(0.0, dtype=points.dtype)),
            None,
        )

    return lax.map(row, (bank.lower, active))


def _global_size_equality_projection(
    rows: Array, direction: Array, equality: Array, remaining_work: Array, /
) -> tuple[Array, Array, Array, Array, Array]:
    """Project a bounded row bank into the existing satisfied-size null space."""
    count = jnp.sum(equality)
    parameter_size = direction.size
    capacity = rows.shape[0]

    def project(bucket: int) -> tuple[Array, Array, Array, Array, Array]:
        work = (
            8 * (bucket * bucket + bucket * capacity) * parameter_size
            + 16 * (bucket + capacity) * parameter_size
            + 64 * bucket**3
            + 16 * bucket * bucket
        )

        def admitted(_: None) -> tuple[Array, Array, Array, Array, Array]:
            indices = jnp.nonzero(equality, size=bucket, fill_value=0)[0]
            included = jnp.arange(bucket) < count
            equal = jnp.where(included[:, None, None], rows[indices], 0.0)
            flat_equal = equal.reshape((bucket, -1))
            flat_rows = rows.reshape((capacity, -1))
            tolerance = _GLOBAL_SIZE_METHOD.linear_policy.tolerance
            factors = factor_pseudoinverse(
                flat_equal @ flat_equal.T,
                RankPolicy(
                    relative_cutoff=tolerance.relative,
                    absolute_cutoff=tolerance.absolute,
                ),
                hermitian=True,
            )
            coefficients = apply_pseudoinverse(factors, flat_equal @ flat_rows.T)
            projected = flat_rows - coefficients.T @ flat_equal
            seed_coefficients = apply_pseudoinverse(
                factors, flat_equal @ direction.reshape((-1,))
            )
            seed = direction.reshape((-1,)) - seed_coefficients @ flat_equal
            return (
                projected.reshape(rows.shape),
                seed.reshape(direction.shape),
                factors.finite,
                jnp.asarray(work, dtype=jnp.int64),
                jnp.asarray(False),
            )

        return lax.cond(
            remaining_work >= work,
            admitted,
            lambda _: (
                rows,
                direction,
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int64),
                jnp.asarray(True),
            ),
            None,
        )

    def bounded(_: None) -> tuple[Array, Array, Array, Array, Array]:
        return lax.cond(
            count <= 4,
            lambda _: project(4),
            lambda _: lax.cond(
                count <= 8, lambda _: project(8), lambda _: project(16), None
            ),
            None,
        )

    return lax.cond(
        (count == 0) | (count > 16),
        lambda _: (
            rows,
            direction,
            count == 0,
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(False),
        ),
        bounded,
        None,
    )


def _global_size_affine_projection(
    gradients: Array,
    inequality: Array,
    equality: Array,
    raw: Array,
    bounds: Array,
    direction: Array,
    remaining_work: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Solve an affine projection in row/dual space within the 32-row bank."""
    rows, seed, equal_valid, equal_work, equal_resource = (
        _global_size_equality_projection(gradients, direction, equality, remaining_work)
    )
    count = jnp.sum(inequality)
    capacity = gradients.shape[0]
    projections = jnp.sum(rows * seed[None, :, :], axis=(1, 2))
    norms = jnp.sqrt(jnp.sum(rows * rows, axis=(1, 2)))
    progress = 0.5 * jnp.where(
        projections > 0.0, projections, norms * jnp.sqrt(jnp.sum(seed * seed))
    )
    right = jnp.where(raw, bounds, -progress)
    signed = jnp.where(raw[:, None, None], rows, -rows)

    def project(bucket: int) -> tuple[Array, Array, Array, Array]:
        preparation = (8 * bucket * bucket + 12 * bucket) * direction.size + 16 * capacity
        step_work = 64 * bucket**3 + 32 * bucket * bucket + 16 * bucket
        bound = preparation + (_GLOBAL_LINEAR_STEPS + 2) * step_work

        def admitted(_: None) -> tuple[Array, Array, Array, Array]:
            indices = jnp.nonzero(inequality, size=bucket, fill_value=0)[0]
            included = jnp.arange(bucket) < count
            local = jnp.where(included[:, None, None], signed[indices], 0.0)
            lengths = norms[indices]
            usable = (~included) | (
                jnp.isfinite(lengths)
                & (lengths > _GLOBAL_SIZE_METHOD.linear_policy.tolerance.absolute)
            )
            denominators = jnp.where(included & usable, lengths, 1.0)
            normalized = local / denominators[:, None, None]
            flat = normalized.reshape((bucket, -1))
            rhs = jnp.where(included, right[indices] / denominators, 1.0)
            quadratic = flat @ flat.T + jnp.diag((~included).astype(flat.dtype))
            program = QuadraticProgram(
                quadratic,
                rhs - flat @ seed.reshape((-1,)),
                bounds=Bounds(0.0, np.inf),
                problem_id="source-stratum/raw-hard-shape-affine-projection",
                convexity_evidence="normalized-gradient-gram",
            )
            result = solve_quadratic_program(
                program, policy=_GLOBAL_SHAPE_PROJECTION_POLICY
            )
            corrected = (seed.reshape((-1,)) - result.primal @ flat).reshape(
                direction.shape
            )
            actual = jnp.sum(gradients * corrected[None, :, :], axis=(1, 2))
            valid = (
                result.valid
                & jnp.all(usable)
                & jnp.all(jnp.isfinite(corrected))
                & jnp.all(~(inequality & raw) | (actual <= bounds))
                & jnp.all(~(inequality & ~raw) | (actual > 0.0))
                & jnp.all(
                    ~equality
                    | (
                        jnp.abs(actual)
                        <= _GLOBAL_SIZE_METHOD.linear_policy.tolerance.relative
                    )
                )
            )
            work = equal_work + preparation + (result.iterations + 2) * step_work
            return corrected, valid, work, jnp.asarray(False)

        return lax.cond(
            remaining_work - equal_work >= bound,
            admitted,
            lambda _: (direction, jnp.asarray(False), equal_work, jnp.asarray(True)),
            None,
        )

    def bounded(_: None) -> tuple[Array, Array, Array, Array]:
        return lax.cond(
            count <= 4,
            lambda _: project(4),
            lambda _: lax.cond(
                count <= 8,
                lambda _: project(8),
                lambda _: lax.cond(
                    count <= 16, lambda _: project(16), lambda _: project(32), None
                ),
                None,
            ),
            None,
        )

    return lax.cond(
        equal_valid & (count > 0) & (count <= 32),
        bounded,
        lambda _: (direction, jnp.asarray(False), equal_work, equal_resource),
        None,
    )


def _global_size_outgoing_slope(
    parameters: Array, direction: Array, residual: Array, args: GlobalSizeArgs, /
) -> Array:
    """Evaluate the true outgoing size slope for a feasible-point correction."""
    points = args.origins + args.target * jnp.sum(
        args.basis * parameters[:, None, :], axis=-1
    )
    displacement = args.target * jnp.sum(args.basis * direction[:, None, :], axis=-1)
    delta = points[args.edges[:, 1]] - points[args.edges[:, 0]]
    velocity = displacement[args.edges[:, 1]] - displacement[args.edges[:, 0]]
    valid = args.edge_valid > 0.0
    lengths = jnp.sqrt(jnp.where(valid, jnp.sum(delta * delta, axis=-1), 1.0))
    tangent = jnp.sum(delta * velocity, axis=-1) / lengths
    acceleration = (jnp.sum(velocity * velocity, axis=-1) - tangent * tangent) / lengths
    order = jnp.lexsort((acceleration, tangent, jnp.where(valid, lengths, jnp.inf)))
    ranks = (jnp.sum(valid) - 1) * args.quantiles
    lower = order[jnp.floor(ranks).astype(jnp.int32)]
    upper = order[jnp.ceil(ranks).astype(jnp.int32)]
    fraction = ranks - jnp.floor(ranks)
    derivative = (
        tangent[lower] * (1.0 - fraction) + tangent[upper] * fraction
    ) / args.target
    # Incoming compliant statistics retain equality rows and finite admission.
    # Their zero residual contributes no least-squares slope.
    return jnp.sum(residual[: args.quantiles.size] * derivative)


class _GlobalShapeDirectionState(NamedTuple):
    bank: GlobalSizeBranchBank
    direction: Array
    valid: Array
    finished: Array
    resource: Array
    work: Array
    visits: Array


def _global_size_shape_direction(
    parameters: Array,
    direction: Array,
    residual: Array,
    remaining_work: Array,
    remaining_visits: Array,
    args: GlobalSizeArgs,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Restore a refused finite trajectory using raw hard-shape inequalities.

    Kernel rationale: the bounded working-set loop owns physical row ordering,
    shared rank-visit capacity, local AD tables, and budget-before-solve admission.
    Full finite shape and native publication remain independent final guards.
    """
    capacity = 32
    points = args.origins + args.target * jnp.sum(
        args.basis * parameters[:, None, :], axis=-1
    )
    delta = points[args.edges[:, 1]] - points[args.edges[:, 0]]
    valid_edges = args.edge_valid > 0.0
    lengths = jnp.sqrt(jnp.where(valid_edges, jnp.sum(delta * delta, axis=-1), 1.0))
    quantile_count, edge_count = args.quantiles.size, args.edges.shape[0]
    cell_count = args.cells.shape[0]
    allowed_margin = residual[quantile_count + cell_count :]
    qids = jnp.arange(quantile_count, dtype=jnp.int32)
    hard_ids = jnp.arange(7 * cell_count, dtype=jnp.int32)
    incoming_compliant = residual[:quantile_count] == 0.0
    ranks = (jnp.sum(valid_edges) - 1) * args.quantiles
    low_rank, high_rank = (
        jnp.floor(ranks).astype(jnp.int32),
        jnp.ceil(ranks).astype(jnp.int32),
    )
    sort_work = 4 * edge_count * (edge_count - 1).bit_length() ** 2
    scan_work = (
        2 * sort_work
        + 2 * args.origins.shape[0]
        + 8 * edge_count
        + 400 * cell_count
        + 12 * quantile_count
    )
    empty = jnp.zeros((capacity,), dtype=jnp.int32)
    bank = GlobalSizeBranchBank(empty, empty, empty, jnp.asarray(0), jnp.asarray(False))

    def refine(state: _GlobalShapeDirectionState) -> _GlobalShapeDirectionState:
        def scan(_: None) -> _GlobalShapeDirectionState:
            displacement = args.target * jnp.sum(
                args.basis * state.direction[:, None, :], axis=-1
            )
            velocity = displacement[args.edges[:, 1]] - displacement[args.edges[:, 0]]
            tangent = jnp.sum(delta * velocity, axis=-1) / lengths
            acceleration = (
                jnp.sum(velocity * velocity, axis=-1) - tangent * tangent
            ) / lengths
            order = jnp.lexsort(
                (acceleration, tangent, jnp.where(valid_edges, lengths, jnp.inf))
            )
            current_low, current_high = order[low_rank], order[high_rank]
            future_points = args.origins + args.target * jnp.sum(
                args.basis * (parameters + state.direction)[:, None, :], axis=-1
            )
            future_delta = (
                future_points[args.edges[:, 1]] - future_points[args.edges[:, 0]]
            )
            future_lengths = jnp.sqrt(
                jnp.where(valid_edges, jnp.sum(future_delta * future_delta, axis=-1), 1.0)
            )
            future_order = jnp.argsort(jnp.where(valid_edges, future_lengths, jnp.inf))
            future_low, future_high = future_order[low_rank], future_order[high_rank]
            future_margin = _global_size_shape_margins(future_points, args)[cell_count:]
            violation = future_margin - allowed_margin
            raw_active = violation > 0.0
            # A working set is not a truncation of the shape obligation: every
            # row is rescanned and completion requires no outstanding violation.
            # Add the most violated row, then let the finite recheck decide
            # which remaining constraints actually need independent gradients.
            selected_raw = raw_active & (hard_ids == jnp.argmax(violation))
            statistics = jnp.concatenate(
                (
                    qids,
                    qids,
                    jnp.asarray((-2,), dtype=jnp.int32),
                    -4 * jnp.ones_like(hard_ids),
                )
            )
            lows = jnp.concatenate(
                (current_low, future_low, jnp.asarray((0,), dtype=jnp.int32), hard_ids)
            )
            highs = jnp.concatenate(
                (
                    current_high,
                    future_high,
                    jnp.asarray((0,), dtype=jnp.int32),
                    jnp.zeros_like(hard_ids),
                )
            )
            active = jnp.concatenate(
                (incoming_compliant, incoming_compliant, jnp.asarray((True,)), selected_raw)
            )
            spent = state.work + scan_work
            membership_bound = (
                capacity + statistics.size + jnp.sum(active) * (8 * capacity + 4)
            )

            def append(_: None) -> _GlobalShapeDirectionState:
                extended, append_work = append_global_size_branches(
                    state.bank, statistics, lows, highs, active
                )
                appended = spent + append_work

                def generate(_: None) -> _GlobalShapeDirectionState:
                    raw = (jnp.arange(capacity) < extended.count) & (
                        extended.statistic == -4
                    )
                    generation_work = (
                        8 * capacity + 10 * quantile_count
                    ) * args.origins.size + 400 * jnp.sum(raw)

                    def admitted(_: None) -> _GlobalShapeDirectionState:
                        raw_gradients, current_margin = _global_size_raw_shape_rows(
                            points, extended, args
                        )
                        joint = _global_size_joint_gradient(
                            delta, lengths, residual, current_low, current_high, args
                        )
                        # Raw slots are not size or hinge objective rows.
                        merit_bank = GlobalSizeBranchBank(
                            jnp.where(raw, -1, extended.statistic),
                            jnp.where(raw, 0, extended.lower),
                            jnp.where(raw, 0, extended.upper),
                            extended.count,
                            extended.overflow,
                        )
                        zero = jnp.zeros_like(direction)
                        gradients, inequality, equality = global_size_branch_gradients(
                            merit_bank,
                            delta,
                            lengths,
                            residual,
                            zero,
                            joint[None, :, :],
                            args,
                        )
                        gradients = jnp.where(
                            raw[:, None, None], raw_gradients, gradients
                        )
                        slope = jnp.sum(
                            gradients * state.direction[None, :, :], axis=(1, 2)
                        )
                        future = future_margin[
                            jnp.minimum(extended.lower, hard_ids.size - 1)
                        ]
                        allowed = allowed_margin[
                            jnp.minimum(extended.lower, hard_ids.size - 1)
                        ]
                        # Preserve the same incoming per-cell bound used by
                        # finite admission, including existing unresolved defects.
                        remainder = jnp.maximum(future - current_margin - slope, 0.0)
                        bounds = jnp.minimum(
                            allowed - current_margin - remainder,
                            -0.5 * jnp.maximum(future - allowed, 0.0),
                        )
                        generated = appended + generation_work
                        corrected, valid, work, denied = _global_size_affine_projection(
                            gradients,
                            inequality,
                            equality,
                            raw,
                            bounds,
                            direction,
                            remaining_work - generated,
                        )
                        return _GlobalShapeDirectionState(
                            extended,
                            corrected,
                            valid,
                            ~valid,
                            denied,
                            generated + work,
                            state.visits + 1,
                        )

                    return lax.cond(
                        generation_work <= remaining_work - appended,
                        admitted,
                        lambda _: state._replace(
                            bank=extended,
                            valid=jnp.asarray(False),
                            finished=jnp.asarray(True),
                            resource=jnp.asarray(True),
                            work=appended,
                            visits=state.visits + 1,
                        ),
                        None,
                    )

                slope = _global_size_outgoing_slope(
                    parameters, state.direction, residual, args
                )
                finished = state.valid & ~jnp.any(raw_active) & (slope < 0.0)
                return lax.cond(
                    finished | extended.overflow,
                    lambda _: state._replace(
                        bank=extended,
                        valid=finished & ~extended.overflow,
                        finished=jnp.asarray(True),
                        work=appended,
                        visits=state.visits + 1,
                    ),
                    generate,
                    None,
                )

            return lax.cond(
                membership_bound <= remaining_work - spent,
                append,
                lambda _: state._replace(
                    valid=jnp.asarray(False),
                    finished=jnp.asarray(True),
                    resource=jnp.asarray(True),
                    work=spent,
                    visits=state.visits + 1,
                ),
                None,
            )

        return lax.cond(
            scan_work <= remaining_work - state.work,
            scan,
            lambda _: state._replace(
                valid=jnp.asarray(False),
                finished=jnp.asarray(True),
                resource=jnp.asarray(True),
            ),
            None,
        )

    state = lax.while_loop(
        lambda value: (
            ~value.finished & ~value.resource & (value.visits < remaining_visits)
        ),
        refine,
        _GlobalShapeDirectionState(
            bank,
            direction,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int64),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return (
        state.direction,
        state.valid & state.finished,
        state.work,
        state.visits,
        state.resource,
    )


# Kernel rationale: the nested lax branches encode charged rank-frontier
# admission. Keep their comparison, branch, and floating-point reduction order
# together; extracting Python lifecycle phases would not reduce this device
# kernel's numerical responsibility and could alter its traced closure.
# Keep the resource-gated frontier in one traced while loop with fixed carry and branch order.
def _global_size_bundle_direction(
    parameters: Array,
    direction: Array,
    residual: Array,
    shape_gradient: Array,
    remaining_work: Array,
    args: GlobalSizeArgs,
) -> tuple[Array, Array, Array, Array, Array]:
    """Close joint size-merit ranks and satisfied-statistic equalities."""
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
    qids = jnp.arange(quantile_count, dtype=jnp.int32)
    incoming_compliant = residual[:quantile_count] == 0.0
    any_failed = jnp.any(~incoming_compliant)
    sort_work = 4 * edge_count * (edge_count - 1).bit_length() ** 2
    scan_work = (
        2 * sort_work + 2 * args.origins.shape[0] + 8 * edge_count + 12 * quantile_count
    )
    empty = jnp.zeros((capacity,), dtype=jnp.int32)
    bank = GlobalSizeBranchBank(empty, empty, empty, jnp.asarray(0), jnp.asarray(False))
    original_norm = jnp.sqrt(jnp.sum(direction * direction))
    tolerance = _GLOBAL_SIZE_METHOD.linear_policy.tolerance
    rank_policy = RankPolicy(
        relative_cutoff=tolerance.relative, absolute_cutoff=tolerance.absolute
    )

    # This while body owns one atomic frontier visit and its resource-refusal carry.
    def refine(carry: _GlobalSizeFrontierCarry) -> _GlobalSizeFrontierCarry:
        bank, candidate, admitted, finished, resource, work, visits, previous, joint = (
            carry
        )

        # Keep rank scanning and gated bank composition fused in the same traced visit.
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
            future_low, future_high = (
                future_order[lower_rank],
                future_order[upper_rank],
            )
            fraction = ranks - lower_rank
            future_statistic = (
                future_lengths[future_low] * (1.0 - fraction)
                + future_lengths[future_high] * fraction
            )
            future_relative = (future_statistic - args.target) / args.target
            future_size = jnp.where(
                args.tolerance == 0.0,
                future_relative,
                future_relative
                - jnp.clip(future_relative, -args.tolerance, args.tolerance),
            )
            # Future failed ranks matter only when the full scalar size merit
            # does not improve. Individual failed components may trade progress
            # under the same summed deficit used by accepted-state admission.
            future_joint_active = any_failed & (
                jnp.sum(jnp.abs(future_size))
                >= jnp.sum(jnp.abs(residual[:quantile_count]))
            )
            current_low, current_high = (
                outgoing_order[lower_rank],
                outgoing_order[upper_rank],
            )
            supports = jnp.stack(
                (current_low, current_high, future_low, future_high), axis=-1
            ).astype(jnp.int32)
            joint_rows, joint_slots, joint_work, joint_resource = (
                _append_global_size_joint_rows(
                    joint,
                    jnp.stack((current_low, future_low)),
                    jnp.stack((current_high, future_high)),
                    jnp.asarray((any_failed, future_joint_active)),
                    delta,
                    lengths,
                    residual,
                    remaining_work - work - scan_work,
                    args,
                )
            )
            statistics = jnp.concatenate(
                (
                    qids,
                    qids,
                    jnp.asarray((-2, -2, -1), dtype=jnp.int32),
                )
            )
            lows = jnp.concatenate(
                (
                    current_low,
                    future_low,
                    joint_slots,
                    jnp.asarray((0,), dtype=jnp.int32),
                )
            )
            highs = jnp.concatenate(
                (
                    current_high,
                    future_high,
                    jnp.asarray((0, 0, 0), dtype=jnp.int32),
                )
            )
            active = jnp.concatenate(
                (
                    incoming_compliant,
                    incoming_compliant,
                    jnp.asarray(
                        (any_failed, future_joint_active, jnp.any(shape_gradient != 0.0))
                    ),
                )
            )
            scanned_work = work + scan_work + joint_work
            membership_bound = (
                capacity + statistics.size + jnp.sum(active) * (8 * capacity + 4)
            )

            def append(_: None) -> _GlobalSizeFrontierCarry:
                extended, append_work = append_global_size_branches(
                    bank, statistics, lows, highs, active
                )
                spent = scanned_work + append_work
                stable = (
                    admitted
                    & (extended.count == bank.count)
                    & jnp.all(supports == previous)
                    & (joint_rows.count == joint.count)
                    & ~extended.overflow
                )

                def complete(_: None) -> _GlobalSizeFrontierCarry:
                    return (
                        extended,
                        candidate,
                        admitted,
                        jnp.asarray(True),
                        resource,
                        spent,
                        visits + 1,
                        supports,
                        joint_rows,
                    )

                def compose(_: None) -> _GlobalSizeFrontierCarry:
                    def bucket_step(bucket: int) -> _GlobalSizeFrontierCarry:
                        generation_work = 8 * bucket * args.origins.size

                        def generate(_: None) -> _GlobalSizeFrontierCarry:
                            selected_bank = GlobalSizeBranchBank(
                                extended.statistic[:bucket],
                                extended.lower[:bucket],
                                extended.upper[:bucket],
                                extended.count,
                                extended.overflow,
                            )
                            gradients, inequality, equality = (
                                global_size_branch_gradients(
                                    selected_bank,
                                    delta,
                                    lengths,
                                    residual,
                                    shape_gradient,
                                    joint_rows.gradients,
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
                                target_work = (
                                    5 * bucket
                                    + 2 * (residual.size - quantile_count)
                                    + 2 * quantile_count
                                )

                                def prepare_targets(
                                    _: None,
                                ) -> tuple[Array, Array, Array, Array]:
                                    # Preserve summed failed-size progress rather
                                    # than impose unrequested Pareto descent.
                                    # Satisfied rank rows remain equalities.
                                    targets = global_size_branch_targets(
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
                                supports,
                                joint_rows,
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
                                supports,
                                joint_rows,
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
                        supports,
                        joint_rows,
                    ),
                    lambda _: lax.cond(stable, complete, compose, None),
                    None,
                )

            return lax.cond(
                ~joint_resource
                & ~joint_rows.overflow
                & (membership_bound <= remaining_work - scanned_work),
                append,
                lambda _: (
                    bank,
                    candidate,
                    jnp.asarray(False),
                    jnp.asarray(True),
                    joint_resource
                    | (
                        ~joint_rows.overflow
                        & (membership_bound > remaining_work - scanned_work)
                    ),
                    scanned_work,
                    visits + 1,
                    supports,
                    joint_rows,
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
                previous,
                joint,
            ),
            None,
        )

    bank, selected, admitted, finished, resource, work, visits, _, _ = lax.while_loop(
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
            jnp.full((quantile_count, 4), -1, dtype=jnp.int32),
            _GlobalSizeJointRows(
                jnp.full((capacity, 2 * quantile_count), -1, dtype=jnp.int32),
                jnp.zeros((capacity, *args.origins.shape), dtype=args.origins.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
            ),
        ),
    )
    return selected, admitted & finished & ~bank.overflow, work, visits, resource


class GlobalSizeTrialPolicy(AbstractLeastSquaresTrialPolicy):
    def termination_valid(self, residual: Array, args: GlobalSizeArgs) -> Array:
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
        args: GlobalSizeArgs,
    ) -> LeastSquaresTrialResult:
        fixed_work, _, _, _ = _global_size_trial_work(
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

    # Kernel rationale: outgoing size/shape tangents, finite source models, and
    # rank events form one ordered trial computation. Retain explicit algebra
    # and lax branches so model visits, native solve actions, and work refusal
    # remain in the same compiled order; host lifecycle phases live above it.
    # Keep outgoing rank derivatives and ordered finite trials in one resource-accounted traced kernel.
    def _evaluate(
        self,
        parameters: Array,
        direction: Array,
        residual: Array,
        jacobian: JacobianLinearOperator,
        remaining_model_work: Array,
        args: GlobalSizeArgs,
    ) -> LeastSquaresTrialResult:
        fixed_work, model_work, event_work, margin_work = _global_size_trial_work(
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
        # Equal represented lengths are ordered by outgoing first and second
        # derivatives, rather than by SCI row identity or epsilon grouping.
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
        # derivative; recover the one-sided hinge derivative for this trial.
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
        # a finite size/shape model, separate from residual callbacks and Krylov J.
        def finite_model(
            scale: Array, model_direction: Array
        ) -> tuple[Array, Array, Array]:
            # Physical sizes and cell-shape constraints of the finite
            # source-point trajectory have separate model-work accounting.
            modeled_points = args.origins + args.target * jnp.sum(
                args.basis * (parameters + scale * model_direction)[:, None, :], axis=-1
            )
            modeled_delta = (
                modeled_points[args.edges[:, 1]] - modeled_points[args.edges[:, 0]]
            )
            modeled_lengths = jnp.sqrt(
                jnp.where(valid, jnp.sum(modeled_delta * modeled_delta, axis=-1), 1.0)
            )
            modeled_statistic = size_quantile(
                jnp.where(valid, modeled_lengths, jnp.nan), args.quantiles
            )
            modeled_relative = (modeled_statistic - args.target) / args.target
            modeled_size = jnp.where(
                args.tolerance == 0.0,
                modeled_relative,
                modeled_relative
                - jnp.clip(modeled_relative, -args.tolerance, args.tolerance),
            )
            modeled_margin = _global_size_shape_margins(modeled_points, args)
            modeled_shape = jnp.maximum(modeled_margin, 0.0)
            modeled = jnp.concatenate((modeled_size, modeled_shape))
            change = modeled - residual
            prediction = -jnp.sum(residual * change) - 0.5 * jnp.sum(change * change)
            admissible = (
                jnp.all(jnp.isfinite(modeled))
                & jnp.all((~valid) | (modeled_lengths > 0.0))
                & (
                    jnp.sum(jnp.abs(modeled_size))
                    <= jnp.sum(jnp.abs(residual[: args.quantiles.size]))
                )
                & jnp.all(
                    (residual[: args.quantiles.size] != 0.0) | (modeled_size == 0.0)
                )
                & jnp.all(
                    modeled_shape[cell_count:]
                    <= residual[args.quantiles.size + cell_count :]
                )
                & jnp.isfinite(prediction)
                & (prediction > 0.0)
            )
            return change / scale, admissible, modeled_margin

        merit_slope = jnp.sum(residual * trial_image)

        stage_budget = remaining_model_work - fixed_work - bundle_work

        def refused(visits: int, margin_evaluated: bool = False) -> _GlobalSizeTrialCarry:
            return (
                jnp.asarray(1.0, dtype=residual.dtype),
                trial_image,
                jnp.asarray(False),
                jnp.asarray(visits, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(margin_evaluated),
                direction,
                jnp.asarray(0, dtype=jnp.int64),
                jnp.asarray(0, dtype=jnp.int32),
                merit_slope,
            )

        def select_trial(_: None) -> _GlobalSizeTrialCarry:
            def full_trial(
                _: None,
            ) -> _GlobalSizeTrialCarry:
                full_image, full_valid, full_margin = finite_model(
                    jnp.asarray(1.0, dtype=residual.dtype), direction
                )

                def stationary_trial(
                    _: None,
                ) -> _GlobalSizeTrialCarry:
                    curvature = jnp.sum(trial_image * trial_image)
                    stationary_scale = jnp.minimum(
                        1.0, -merit_slope / jnp.where(curvature > 0.0, curvature, 1.0)
                    )
                    # The full finite model has already refused scale one.
                    # Repeating that point cannot recover hard-source feasibility.
                    stationary_scale = jnp.where(
                        stationary_scale == 1.0, 0.5, stationary_scale
                    )
                    # Clipped zero hinges contain no measure of incoming
                    # feasibility. Use signed margins of the same cell algebra
                    # to bracket a hard-shape crossing before another model.
                    current_margin = _global_size_shape_margins(points, args)
                    shape_scale = _hard_shape_trial_scale(
                        current_margin[cell_count:],
                        full_margin[cell_count:],
                        residual[args.quantiles.size + cell_count :],
                        stationary_scale,
                    )
                    stationary_scale = shape_scale
                    image, admitted, _source_valid = finite_model(
                        stationary_scale, direction
                    )
                    admitted = (
                        admitted
                        & jnp.isfinite(stationary_scale)
                        & (stationary_scale > 0.0)
                    )

                    def event_trial(
                        _: None,
                    ) -> _GlobalSizeTrialCarry:
                        support = jnp.concatenate((lower_rows, upper_rows))
                        alpha, accepted = quadratic_event_bound(
                            quadratic[None, :] - quadratic[support, None],
                            linear[None, :] - linear[support, None],
                            constant[None, :] - constant[support, None],
                            jnp.broadcast_to(valid[None, :], (support.size, valid.size)),
                        )
                        alpha = jnp.minimum(alpha, shape_scale)
                        image, finite_valid, _source_valid = finite_model(
                            alpha, direction
                        )
                        return (
                            alpha,
                            image,
                            accepted & finite_valid,
                            jnp.asarray(3, dtype=jnp.int32),
                            jnp.asarray(True),
                            jnp.asarray(False),
                            jnp.asarray(True),
                            direction,
                            jnp.asarray(0, dtype=jnp.int64),
                            jnp.asarray(0, dtype=jnp.int32),
                            merit_slope,
                        )

                    def shape_trial(_: None) -> _GlobalSizeTrialCarry:
                        slope_work = (
                            4
                            * args.edges.shape[0]
                            * (args.edges.shape[0] - 1).bit_length() ** 2
                            + 4 * args.origins.shape[0]
                            + 12 * args.edges.shape[0]
                            + 10 * args.quantiles.size
                        )
                        allowance = (
                            stage_budget - 3 * model_work - margin_work - slope_work
                        )
                        (
                            corrected,
                            shape_valid,
                            shape_work,
                            shape_visits,
                            shape_resource,
                        ) = _global_size_shape_direction(
                            parameters,
                            direction,
                            residual,
                            allowance,
                            jnp.asarray(8, dtype=jnp.int32) - bundle_visits,
                            args,
                        )

                        def evaluate(_: None) -> tuple[Array, Array, Array]:
                            corrected_image, finite_valid, _source_valid = finite_model(
                                jnp.asarray(1.0, dtype=residual.dtype), corrected
                            )
                            corrected_slope = _global_size_outgoing_slope(
                                parameters, corrected, residual, args
                            )
                            return corrected_image, finite_valid, corrected_slope

                        corrected_image, finite_valid, corrected_slope = lax.cond(
                            shape_valid & ~shape_resource,
                            evaluate,
                            lambda _: (trial_image, jnp.asarray(False), merit_slope),
                            None,
                        )
                        executed = (shape_valid & ~shape_resource).astype(jnp.int32)
                        return (
                            jnp.asarray(1.0, dtype=residual.dtype),
                            corrected_image,
                            finite_valid & shape_valid & ~shape_resource,
                            jnp.asarray(2, dtype=jnp.int32) + executed,
                            jnp.asarray(False),
                            shape_resource,
                            jnp.asarray(True),
                            corrected,
                            shape_work + executed.astype(jnp.int64) * slope_work,
                            shape_visits,
                            corrected_slope,
                        )

                    def third_trial(_: None) -> _GlobalSizeTrialCarry:
                        needs_shape = jnp.any(
                            full_margin[cell_count:]
                            > residual[args.quantiles.size + cell_count :]
                        )
                        return lax.cond(
                            needs_shape,
                            lambda _: lax.cond(
                                stage_budget >= 3 * model_work + margin_work,
                                shape_trial,
                                lambda _: refused(2, True),
                                None,
                            ),
                            lambda _: lax.cond(
                                stage_budget >= 3 * model_work + event_work + margin_work,
                                event_trial,
                                lambda _: refused(2, True),
                                None,
                            ),
                            None,
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
                            jnp.asarray(True),
                            direction,
                            jnp.asarray(0, dtype=jnp.int64),
                            jnp.asarray(0, dtype=jnp.int32),
                            merit_slope,
                        ),
                        third_trial,
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
                        jnp.asarray(False),
                        direction,
                        jnp.asarray(0, dtype=jnp.int64),
                        jnp.asarray(0, dtype=jnp.int32),
                        merit_slope,
                    ),
                    lambda _: lax.cond(
                        stage_budget >= 2 * model_work + margin_work,
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
            margin_evaluated,
            selected_direction,
            shape_work,
            shape_visits,
            selected_slope,
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
                jnp.asarray(False),
                direction,
                jnp.asarray(0, dtype=jnp.int64),
                jnp.asarray(0, dtype=jnp.int32),
                merit_slope,
            ),
            None,
        )
        proposed_points = args.origins + args.target * jnp.sum(
            args.basis * (parameters + scale * selected_direction)[:, None, :], axis=-1
        )
        represented_motion = jnp.any(proposed_points != points)
        # One-sided outgoing slope admission is independent of finite prediction.
        actual_work = (
            fixed_work
            + bundle_work
            + model_visits * model_work
            + event_executed.astype(jnp.int64) * event_work
            + margin_evaluated.astype(jnp.int64) * margin_work
            + shape_work
        )
        return LeastSquaresTrialResult(
            direction=selected_direction,
            scale=scale,
            model_image=selected_trial_image,
            valid=(
                bundle_valid
                & admissible
                & represented_motion
                & ~bundle_resource
                & ~trial_resource
                & jnp.isfinite(selected_slope)
                & (selected_slope < 0.0)
            ),
            jvp_actions=jnp.asarray(1, dtype=jnp.int32),
            vjp_actions=jnp.asarray(1, dtype=jnp.int32),
            model_work_units=actual_work,
            model_visits=bundle_visits + model_visits + shape_visits,
            resource_refused=bundle_resource | trial_resource,
        )


def _hard_shape_trial_scale(
    current: Array, future: Array, allowed: Array, scale: Array, /
) -> Array:
    """Interiorize a finite hard-shape crossing without changing its bound."""
    slack = allowed - current
    excess = future - allowed
    violated = excess > 0.0
    denominator = jnp.where(violated, slack + excess, 1.0)
    crossing = jnp.where(violated, jnp.maximum(slack, 0.0) / denominator, jnp.inf)
    bound = jnp.min(crossing)
    # Zero incoming slack does not establish a positive crossing bracket.
    # Retain the selected scales under unchanged finite hard-shape admission.
    return jnp.where((bound > 0.0) & (bound < scale), 0.5 * bound, scale)


def _global_size_shape_margins(points: Array, args: GlobalSizeArgs, /) -> Array:
    """Retain signed excesses in the existing physical shape residual order."""
    corners = points[args.cells] / args.target
    frame = corners[:, 1:] - corners[:, :1]
    determinant = jnp.sum(frame[:, 0] * jnp.cross(frame[:, 1], frame[:, 2]), axis=-1)
    volume = 0.1 * args.initial_volume - determinant
    faces = corners[:, jnp.asarray(_FACES)]
    normals = jnp.cross(faces[:, :, 1] - faces[:, :, 0], faces[:, :, 2] - faces[:, :, 0])
    squared = jnp.sum(normals * normals, axis=-1)
    pairs = jnp.asarray(_TET_EDGES)
    cosine = -jnp.sum(
        normals[:, pairs[:, 0]] * normals[:, pairs[:, 1]], axis=-1
    ) / jnp.sqrt(jnp.maximum(squared[:, pairs[:, 0]] * squared[:, pairs[:, 1]], 1.0e-30))
    angles = cosine - args.dihedral_cosine
    solve = solve_small_linear(
        SmallLinearSolvePlan(3),
        frame,
        0.5 * jnp.sum(frame * frame, axis=-1),
    )
    cell_edges = corners[:, pairs[:, 1]] - corners[:, pairs[:, 0]]
    shortest = jnp.sqrt(jnp.min(jnp.sum(cell_edges * cell_edges, axis=-1), axis=-1))
    radius = jnp.sqrt(jnp.sum(solve.value * solve.value, axis=-1))
    radius_excess = radius / shortest - args.radius_edge_bound
    return jnp.concatenate(
        (
            jnp.where(args.cell_valid > 0.0, 16.0 * volume, 0.0),
            jnp.where(args.cell_valid > 0.0, 16.0 * radius_excess, 0.0),
            jnp.where(args.cell_valid[:, None] > 0.0, 16.0 * angles, 0.0).reshape((-1,)),
        )
    )


def global_size_shape_residual(points: Array, args: GlobalSizeArgs, /) -> Array:
    """Evaluate the shared physical cell algebra for native primals and finite models."""
    return jnp.maximum(_global_size_shape_margins(points, args), 0.0)


def global_size_residual(parameters: Array, args: GlobalSizeArgs, /) -> Array:
    """Evaluate target quantiles and shape hinges on participating source strata.

    Publication and native proposals use linear quantiles. Publication
    statistics and shape checks remain separate from this numerical model.
    Jacobian actions use differentiated edge gathers and cell-local algebra.
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
    # The primal uses linear quantiles. Equal-value rank blocks have a
    # row-permutation-symmetric linear derivative; outgoing trial derivatives
    # remain a separate nonlinear rank operation.
    statistic = size_quantile(lengths, args.quantiles)
    relative = (statistic - args.target) / args.target
    size = jnp.where(
        args.tolerance == 0.0,
        relative,
        relative - jnp.clip(relative, -args.tolerance, args.tolerance),
    )
    return jnp.concatenate((size, global_size_shape_residual(points, args)))


def _global_size_problem(model_work_limit: Array, /) -> NonlinearLeastSquaresProblem:
    return NonlinearLeastSquaresProblem(
        global_size_residual,
        problem_id="source-stratum/global-size-linear-quantile",
        trial_policy=GlobalSizeTrialPolicy(),
        trial_policy_id=_GLOBAL_SIZE_TRIAL_POLICY_ID,
        trial_model_work_limit=model_work_limit,
    )


@eqx.filter_jit
def _initialize_metric_least_squares(
    args: GlobalSizeArgs | _EqualLengthArgs,
    problem: NonlinearLeastSquaresProblem,
    parameters: Array,
    method: LevenbergMarquardt,
    /,
) -> tuple[Array, LeastSquaresState]:
    residual = _owner_phydrax_optim__least_squares._PolicyResidual(problem, args)
    return parameters, method.prepare_state(residual, parameters)


class _MetricSolveWork(NamedTuple):
    action: Array
    sort: Array
    group: Array
    step: Array
    final: Array
    available: Array

    @classmethod
    def from_counts(
        cls, action: int, sort: int, group: int, step: int, final: int, available: int, /
    ) -> _MetricSolveWork:
        return cls(*(jnp.asarray(value, dtype=jnp.int64) for value in (
            action, sort, group, step, final, available,
        )))


@eqx.filter_jit
def _advance_metric_least_squares(
    args: GlobalSizeArgs | _EqualLengthArgs,
    problem: NonlinearLeastSquaresProblem,
    parameters: Array,
    state: LeastSquaresState,
    method: LevenbergMarquardt,
    termination: OptimizationTermination,
    work: _MetricSolveWork,
    /,
) -> tuple[Array, LeastSquaresState, Array]:
    """Continue native solver state with device-side next-step admission.

    Prefix counters debit the same finite envelope before each numerical step;
    unused earlier upper bounds become available without a host synchronization.
    """
    dynamic, static = eqx.partition(state, eqx.is_array)

    def spent(current: LeastSquaresState) -> Array:
        residuals = current.residual_evaluations - state.residual_evaluations
        actions = (
            residuals + current.jvp_evaluations - state.jvp_evaluations
            + current.vjp_evaluations - state.vjp_evaluations
        )
        return (
            actions * work.action + residuals * work.sort
            + (current.iteration - state.iteration) * work.group
            + current.trial_policy_model_work_units - state.trial_policy_model_work_units
        )

    def condition(carry: tuple[Array, LeastSquaresState, Array, Array]) -> Array:
        _, current, status, refused = carry
        within_evaluations = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else current.residual_evaluations < termination.maximum_evaluations
        )
        return (
            (status == int(OptimizationStatus.ITERATING))
            & (current.iteration < termination.maximum_steps)
            & within_evaluations & ~refused
        )

    def body(
        carry: tuple[Array, LeastSquaresState, Array, Array],
    ) -> tuple[Array, LeastSquaresState, Array, Array]:
        parameters, dynamic, status, _ = carry
        current = eqx.combine(dynamic, static)
        remaining = work.available - spent(current)

        def admitted(_unused: None) -> tuple[Array, LeastSquaresState, Array, Array]:
            bounded = problem
            if problem.trial_policy is not None:
                bounded = eqx.tree_at(
                    lambda value: value.trial_model_work_limit, problem,
                    current.trial_policy_model_work_units + remaining - work.step - work.final,
                )
            residual = _owner_phydrax_optim__least_squares._PolicyResidual(bounded, args)
            next_parameters, next_state, _ = method.step(
                residual, parameters, current, termination=termination
            )
            dynamic_next, _ = eqx.partition(next_state, eqx.is_array)
            return (
                next_parameters, dynamic_next, method.step_metrics(next_state).status,
                next_state.trial_policy_resource_refused,
            )

        return lax.cond(
            remaining >= work.step + work.final,
            admitted,
            lambda _: (parameters, dynamic, status, jnp.asarray(True)),
            None,
        )

    parameters, dynamic, _, refused = lax.while_loop(
        condition, body,
        (parameters, dynamic, jnp.asarray(int(OptimizationStatus.ITERATING), dtype=jnp.int32),
         jnp.asarray(False)),
    )
    return parameters, eqx.combine(dynamic, static), refused


@eqx.filter_jit
def _finish_metric_least_squares(
    args: GlobalSizeArgs | _EqualLengthArgs,
    problem: NonlinearLeastSquaresProblem,
    parameters: Array,
    state: LeastSquaresState,
    status: Array,
    method: LevenbergMarquardt,
    termination: OptimizationTermination,
    /,
) -> LeastSquaresResult:
    residual = _owner_phydrax_optim__least_squares._PolicyResidual(problem, args)
    run = _owner_phydrax_optim__least_squares._LeastSquaresRun(parameters, state, status)
    return _owner_phydrax_optim__least_squares._package_least_squares_result(
        method, problem, run, residual, termination, args
    )


class _GlobalSizePlan(NamedTuple):
    args: GlobalSizeArgs
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
    """Record charged work and settle the explicit reservation on every exit."""
    budget.check_deadline()
    started = phase_started(budget.record_phase)
    before = budget.work_units if budget.record_phase is not None else 0
    try:
        with budget.phase():
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
    # Reuse compiled numerical banks across nearby topology epochs; validity
    # masks preserve exact support and every executed padded row remains charged.
    vertex_capacity = _bucket_size(vertex_count)
    edge_capacity = _bucket_size(edge_count)
    cell_capacity = _bucket_size(cell_count)
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
    quantiles = [
        _dependency_phydrax_meshing__sizing.size_statistic_quantile(name)
        for name in objective.goal.policy.target_statistics
    ]
    # Only the requested statistics enter the residual. Requiring every edge
    # between their ranks to equal the target adds unrequested constraints.
    edge_valid = (np.arange(edge_capacity) < edge_count).astype(np.float64)
    cell_valid = (np.arange(cell_capacity) < cell_count).astype(np.float64)
    origins = _bucket(snapshot.points)
    cells = _bucket(snapshot.tetrahedra)
    corners = origins[cells]
    initial_volume = np.linalg.det(corners[:, 1:] - corners[:, :1]) / target**3
    edges = _edge_rows(objective.edges.keys)
    args = GlobalSizeArgs(
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
    # Every action pays edge processing and retained-support gather/scatter.
    # STORE linearization sorts only during primal residual/model evaluation;
    # its JVP and transposed VJP reuse the prepared sort permutation.
    action_work = vertex_capacity + 8 * edge_capacity + 20 * cell_capacity
    group_work = 4 * edge_capacity
    fixed_trial_work, model_trial_work, event_trial_work, margin_trial_work = (
        _global_size_trial_work(
            vertex_capacity, edge_capacity, cell_capacity, quantile_count
        )
    )
    trial_work = (
        fixed_trial_work + 3 * model_trial_work + event_trial_work + margin_trial_work
    )
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


def _admit_global_size_work(
    plan: _GlobalSizePlan, budget: _NativeMetricBudget, required: int, /
) -> int | None:
    budget.check_deadline()
    remaining = min(
        budget.maximum_work_units - budget.work_units,
        budget.coordinate_budget.ticket.remaining_work_units,
    )
    if required > remaining:
        budget.refuse(
            "The remaining work allowance cannot admit the next statistical solve phase.",
            (("maximum_work_units", float(budget.maximum_work_units)),),
            (
                ("work_units", float(budget.work_units)),
                ("remaining_work_units", float(remaining)),
                ("solve_phase_work_upper", float(required)),
                ("solve_vertices", float(plan.args.origins.shape[0])),
                ("solve_edges", float(plan.args.edges.shape[0])),
                ("solve_cells", float(plan.args.cells.shape[0])),
            ),
        )
        return None
    if not budget.admit_optimizer_bound(required):
        return None
    return remaining


def _global_size_solve(
    plan: _GlobalSizePlan, budget: _NativeMetricBudget, /
) -> LeastSquaresResult | None:
    """Continue one LM state, admitting and settling each bounded dispatch.

    Damping, refresh state, parameters and all counters survive every phase.
    Only the final native transaction can mutate the mesh.
    """
    model_work = 2 * plan.action_work + plan.sort_work + plan.group_work
    if _admit_global_size_work(plan, budget, 2 * model_work) is None:
        return None
    parameters, state = _initialize_metric_least_squares(
        plan.args, _global_size_problem(jnp.asarray(0, dtype=jnp.int64)),
        jnp.zeros_like(plan.args.origins), _GLOBAL_SIZE_METHOD,
    )
    if not budget.optimizer_work(model_work, plan.scratch_bytes):
        return None
    budget.optimizer_objective_evaluations += 1
    trials = _GLOBAL_SIZE_METHOD.maximum_trials
    step_actions = 2 + trials * (2 * _GLOBAL_LINEAR_STEPS + 12) + 2 * trials
    step_work = (
        step_actions * plan.action_work
        + (1 + trials) * plan.sort_work
        + plan.group_work
    )
    remaining = _admit_global_size_work(plan, budget, step_work + model_work)
    if remaining is None:
        return None
    if not budget.admit_optimizer_bound(remaining - model_work):
        return None
    previous = state
    parameters, state, refused = _advance_metric_least_squares(
        plan.args, _global_size_problem(jnp.asarray(0, dtype=jnp.int64)),
        parameters, state, _GLOBAL_SIZE_METHOD, _GLOBAL_SIZE_TERMINATION,
        _MetricSolveWork.from_counts(
            plan.action_work, plan.sort_work, plan.group_work,
            step_work, model_work, remaining,
        ),
    )
    if not bool(state.trial_policy_receipts_valid):
        raise ValueError(
            "tetra_metric.trial_policy_work must pass native accounting validation; got invalid work evidence"
        )
    residuals = int(state.residual_evaluations - previous.residual_evaluations)
    actions = residuals + int(
        state.jvp_evaluations - previous.jvp_evaluations
        + state.vjp_evaluations - previous.vjp_evaluations
    )
    iterations = int(state.iteration - previous.iteration)
    policy_work = int(
        state.trial_policy_model_work_units - previous.trial_policy_model_work_units
    )
    budget.optimizer_iterations += iterations
    budget.optimizer_objective_evaluations += residuals
    if not budget.optimizer_work(
        actions * plan.action_work + residuals * plan.sort_work
        + iterations * plan.group_work + policy_work,
        plan.scratch_bytes,
    ):
        return None
    if bool(refused):
        budget.refuse(
            "The remaining original work allowance cannot admit the next statistical solve phase.",
            (("maximum_work_units", float(budget.maximum_work_units)),),
            (
                ("work_units", float(budget.work_units)),
                ("next_step_work_upper", float(step_work + model_work)),
                ("actual_model_work_units", float(policy_work)),
            ),
        )
        return None
    status = OptimizationStatus(int(state.metrics.status))
    evaluation_limit = _GLOBAL_SIZE_TERMINATION.maximum_evaluations
    if status is OptimizationStatus.ITERATING:
        status = (
            OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
            if evaluation_limit is not None and int(state.residual_evaluations) >= evaluation_limit
            else OptimizationStatus.MAXIMUM_STEPS_REACHED
        )
    if _admit_global_size_work(plan, budget, model_work) is None:
        return None
    result = _finish_metric_least_squares(
        plan.args, _global_size_problem(state.trial_policy_model_work_units),
        parameters, state, jnp.asarray(int(status)),
        _GLOBAL_SIZE_METHOD, _GLOBAL_SIZE_TERMINATION,
    )
    if not budget.optimizer_work(model_work, plan.scratch_bytes):
        return None
    budget.optimizer_objective_evaluations += 1
    budget.optimizer_status_counts[int(result.status)] += 1
    return result


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
        if objective.complete(snapshot, values):
            # This is the requested relocation's exact goal check, not a
            # skipped phase or a successful label on an unexecuted failed solve.
            return snapshot, 0, 0
        if operation_budget < 1:
            budget.refuse(
                "The requested statistical relocation has no operation allowance.",
                (("remaining_operations", float(operation_budget)),),
                (("work_units", float(budget.work_units)),),
            )
            return snapshot, 0, 0
        plan = _prepare_global_size(snapshot, fixed, objective, operation_budget)
    if plan is None:
        return snapshot, 0, 0
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
    with _measure_statistical_phase(budget, "metric_global_solve"):
        result = _global_size_solve(plan, budget)
    if result is None:
        return snapshot, 0, 1
    displacement = np.asarray(result.parameters, dtype=np.float64)
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
    """Check global merit and source coverage before one native commit."""

    budget = objective.budget
    rows, scratch, preparation_work = (
        plan.vertices,
        plan.scratch_bytes,
        plan.preparation_work,
    )
    if not np.any(displacement[rows]):
        budget.refuse_optimizer("unchanged_global_size_proposal")
        return snapshot, 0, 1
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
            lengths, growth = (
                _owner_phydrax_meshing_providers__native_publication.edge_size_evidence(
                    points, plan.edges
                )
            )
        except _leaf_phydrax_meshing__failures.MeshingFailure as error:
            if (
                error.evidence.category
                is not _leaf_phydrax_meshing__failures.MeshingFailureCategory.COMPLIANCE_FAILED
            ):
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
        source_rows = entity_keys_owner.key_rows(
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
                lineage_operation_kind(
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


def _metric_relocation_merit(
    old_lengths: np.ndarray,
    new_lengths: np.ndarray,
    old_quality: float,
    new_quality: float,
    minimum_quality: float,
    metric_band: tuple[float, float],
    /,
    *,
    refinement: bool = False,
    required_long_split: bool = False,
) -> bool:
    """Admit band descent or monotone repair of an initially deficient cavity.

    Repair steps remain inside the metric epoch. Completion independently
    requires every cell to meet the floor; an admissible cavity cannot cross it.
    Refined cavities compare their descending positive band-violation
    multisets: adding shorter edges must not be penalized merely for their
    multiplicity. Common unchanged global edges cancel before this call.
    """
    if (refinement or required_long_split) and (
        not np.isfinite(old_quality) or not np.isfinite(new_quality)
        or not np.all(np.isfinite(old_lengths)) or not np.all(np.isfinite(new_lengths))
        or new_quality < minimum_quality
    ):
        return False
    if old_quality < minimum_quality and not required_long_split:
        return new_quality > old_quality + 1.0e-6
    if new_quality < minimum_quality:
        return False
    lower, upper = metric_band
    if required_long_split:
        # The inspected interior insertion must strictly reduce the longest
        # changed edge. New short edges are secondary to this required upper
        # sizing repair; the final certificate still measures their obligations.
        # Common unchanged global edges already cancelled in the cavity delta.
        old_max = float(np.max(old_lengths, initial=0.0))
        new_max = float(np.max(new_lengths, initial=0.0))
        return bool(old_max > upper and new_max < old_max)
    if refinement:
        old_violations = np.maximum(lower - old_lengths, 0.0) + np.maximum(
            old_lengths - upper, 0.0
        )
        new_violations = np.maximum(lower - new_lengths, 0.0) + np.maximum(
            new_lengths - upper, 0.0
        )
        old_ranked = np.sort(old_violations[old_violations > 0.0])[::-1]
        new_ranked = np.sort(new_violations[new_violations > 0.0])[::-1]
        common = min(old_ranked.size, new_ranked.size)
        different = np.flatnonzero(old_ranked[:common] != new_ranked[:common])
        if different.size:
            first = int(different[0])
            return bool(new_ranked[first] < old_ranked[first])
        if new_ranked.size != old_ranked.size:
            return new_ranked.size < old_ranked.size
        return new_quality > old_quality + 1.0e-6
    old_deficit = float(
        np.sum(
            np.maximum(lower - old_lengths, 0.0) + np.maximum(old_lengths - upper, 0.0)
        )
    )
    new_deficit = float(
        np.sum(
            np.maximum(lower - new_lengths, 0.0) + np.maximum(new_lengths - upper, 0.0)
        )
    )
    return new_deficit < old_deficit or (
        new_deficit == old_deficit and new_quality > old_quality + 1.0e-6
    )


def _aim_relocation_targets(
    snapshot: TetMeshArrays,
    vertices: np.ndarray,
    directions: np.ndarray,
    budget: _NativeMetricBudget,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    workset = _ReconnectionWorkset.prepare(snapshot, budget)
    targets = np.full((vertices.size, 3, 3), np.nan, dtype=np.float64)
    eligible = np.zeros((vertices.size, 3), dtype=np.bool_)
    if workset is None:
        return targets, eligible
    fractions = np.asarray((1.0, 0.5, 0.25), dtype=np.float64)
    curve_rows: list[int] = []
    curve_neighbors: list[tuple[int, int]] = []
    for row, vertex in enumerate(vertices.tolist()):
        budget.check_deadline()
        segment_rows = np.flatnonzero(np.any(snapshot.segments == vertex, axis=1))
        coordinates = _tangent_coordinates(
            snapshot, vertex, snapshot.tetrahedra[workset.rows((vertex,))],
            facet_rows=workset.face_rows(vertex), segment_rows=segment_rows,
        )
        if coordinates is None:
            budget.refuse_optimizer("unsupported_source_tangent")
            continue
        tangent = coordinates.basis @ (coordinates.basis.T @ directions[row])
        targets[row] = snapshot.points[vertex] + fractions[:, None] * tangent
        eligible[row] = np.all(np.isfinite(targets[row]), axis=1)
        if snapshot.vertex_dimension[vertex] == 1:
            segments = snapshot.segments[segment_rows]
            neighbors = np.unique(segments[segments != vertex])
            curve_rows.append(row)
            curve_neighbors.append((int(neighbors[0]), int(neighbors[1])))
    if curve_rows:
        rows = np.asarray(curve_rows, dtype=np.int64)
        curve_targets = targets[rows].reshape((-1, 3))
        curve_eligible = eligible[rows].reshape((-1,))
        queries = np.flatnonzero(curve_eligible)
        if queries.size:
            construction = budget.call(
                lambda allowance: budget.native.construct_curve_points(
                    np.repeat(vertices[rows], 3)[queries],
                    np.repeat(np.asarray(curve_neighbors, dtype=np.int32), 3, axis=0)[queries],
                    curve_targets[queries],
                    work_limit=allowance,
                )
            )
            if construction is None:
                curve_eligible[:] = False
            else:
                curve_targets[queries] = construction[0]
                curve_eligible[queries] &= construction[2]
            targets[rows] = curve_targets.reshape((-1, 3, 3))
            eligible[rows] = curve_eligible.reshape((-1, 3))
    return targets, eligible


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
    *,
    minimum_quality: float,
    metric_band: tuple[float, float],
) -> tuple[TetMeshArrays, int, int]:
    if objective is not None and objective.aim is None:
        return _statistical_relocate_pass(
            native, snapshot, values, fixed, budget, source_metric, ancestry, objective
        )
    guarded = objective is not None
    if guarded and not native_budget.optimizer_scratch(
        256 * (snapshot.points.shape[0] + 12 * snapshot.tetrahedra.shape[0] + 6 * snapshot.faces.shape[0])
    ):
        return snapshot, 0, 0
    edges = _edges(snapshot.tetrahedra)
    lengths = _lengths(snapshot.points, values, edges)
    if guarded and not native_budget.optimizer_work(
        snapshot.points.shape[0] + 6 * snapshot.tetrahedra.shape[0] + edges.shape[0]
        + 3 * snapshot.faces.shape[0] + 2 * snapshot.segments.shape[0],
        native_budget.optimizer_scratch_required_bytes,
    ):
        return snapshot, 0, 0
    delta = snapshot.points[edges[:, 1]] - snapshot.points[edges[:, 0]]
    lower, upper = metric_band
    target = 1.0 if lower < 1.0 < upper else 0.5 * (lower + upper)
    active = (lengths < lower) | (lengths > upper)
    if guarded or not np.any(active) or np.any(
        _quality(snapshot.points, values, snapshot.tetrahedra) < minimum_quality
    ):
        active[:] = True
    force = (active * (1.0 - target / np.maximum(lengths, 1.0e-30)))[:, None] * delta
    total = np.zeros_like(snapshot.points)
    degree = np.zeros((snapshot.points.shape[0],), dtype=np.float64)
    np.add.at(total, edges[:, 0], force)
    np.add.at(total, edges[:, 1], -force)
    np.add.at(degree, edges.reshape((-1,)), np.repeat(active, 2))
    vertices = np.flatnonzero(
        snapshot.vertex_dimension > 0 if guarded else snapshot.vertex_dimension == 3
    )
    if objective is not None and objective.aim is not None:
        affected = np.unique(_edge_rows(objective.offending_edge_keys()))
        vertices = vertices[np.isin(vertices, affected)]
    vertices = vertices[
        ~np.asarray(
            [vertex < fixed.size and fixed[vertex] for vertex in vertices], dtype=np.bool_
        )
    ][:budget]
    if vertices.size == 0:
        return snapshot, 0, 0
    # Backtracking steps of the unit-band force: a merit-descent direction
    # admits a short enough step whenever the other incident edges are strictly
    # inside the band; the strict merit itself is unchanged.
    fractions = np.asarray((1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125), dtype=np.float64)
    directions = total[vertices] / np.maximum(degree[vertices, None], 1.0)
    targets = (
        snapshot.points[vertices, None] + fractions[None, :, None] * directions[:, None]
    )
    if guarded:
        targets, eligible = _aim_relocation_targets(
            snapshot, vertices, directions, native_budget
        )
        source_ids = np.full((vertices.size * 3, 4), -1, dtype=np.int64)
        source_weights = np.full((vertices.size * 3, 4), np.nan, dtype=np.float64)
        samples = np.full((vertices.size * 3, 3, 3), np.nan, dtype=np.float64)
        covered = np.zeros((vertices.size * 3,), dtype=np.bool_)
        rows = np.flatnonzero(eligible.reshape((-1,)))
        if rows.size:
            stencil = native_budget.stencil(targets.reshape((-1, 3))[rows])
            if stencil is None:
                return snapshot, 0, 0
            valid = stencil[2].all(axis=1)
            selected = rows[valid]
            source_ids[selected] = stencil[0][valid]
            source_weights[selected] = stencil[1][valid]
            if selected.size:
                source_rows = entity_keys_owner.key_rows(
                    source_tables.vertex_global_ids[:, None],
                    source_ids[selected].reshape((-1, 1)),
                ).reshape((-1, 4))
                samples[selected] = _interpolate(
                    source_metric[source_rows], source_weights[selected]
                )
                covered[selected] = True
        samples = samples.reshape((-1, 3, 3, 3))
        covered = covered.reshape((-1, 3))
    else:
        stencil = native_budget.stencil(targets.reshape((-1, 3)))
        if stencil is None:
            return snapshot, 0, 0
        source_ids, source_weights = stencil[:2]
        source_rows = entity_keys_owner.key_rows(
            source_tables.vertex_global_ids[:, None],
            source_ids.reshape((-1, 1)),
        ).reshape(source_ids.shape)
        samples = _interpolate(source_metric[source_rows], source_weights).reshape(
            (-1, fractions.size, 3, 3)
        )
        covered = stencil[2].reshape((-1, fractions.size, 4)).all(axis=-1)
    applied, work = 0, 0
    for index, vertex in enumerate(vertices.tolist()):
        cavity = snapshot.tetrahedra[np.any(snapshot.tetrahedra == vertex, axis=1)]
        old = float(np.min(_quality(snapshot.points, values, cavity)))
        incident_edges = edges[np.any(edges == vertex, axis=1)]
        old_lengths = _lengths(snapshot.points, values, incident_edges)
        for step in range(targets.shape[1]):
            if work >= budget or native_budget.exhausted:
                break
            work += 1
            if not covered[index, step]:
                continue
            statistical = None
            if objective is not None:
                local_vertices = np.unique(cavity)
                local_points = _proposal_rows(
                    snapshot.points, local_vertices, vertex, targets[index, step]
                )
                if not np.all(_positive(local_points, np.searchsorted(local_vertices, cavity))):
                    continue
                statistical = objective.propose(
                    snapshot, values, cavity, cavity,
                    changed_vertex=vertex,
                    changed_position=targets[index, step],
                    changed_metric=samples[index, step],
                )
                if statistical is None:
                    continue
            else:
                proposed = snapshot.points.copy()
                proposed[vertex] = targets[index, step]
                metrics = values.copy()
                metrics[vertex] = samples[index, step]
                if not np.all(_positive(proposed, cavity)):
                    continue
                new_quality = float(np.min(_quality(proposed, metrics, cavity)))
                if not _metric_relocation_merit(
                    old_lengths,
                    _lengths(proposed, metrics, incident_edges),
                    old,
                    new_quality,
                    minimum_quality,
                    metric_band,
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
                    source_ids[
                        index * targets.shape[1] + step,
                        source_weights[index * targets.shape[1] + step] > 0.0,
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
                        lineage_operation_kind(
                            EntityLineageKind.RELOCATED,
                            (
                                ancestry.cell_kinds.get(
                                    identifier, int(EntityLineageKind.PRESERVED)
                                ),
                            ),
                        )
                    )
                applied += 1
                if objective is not None and statistical is not None:
                    objective.commit(statistical)
                    if objective.complete(snapshot, values):
                        return snapshot, applied, work
                break
    return snapshot, applied, work


__all__ = [
    "MetricMemoryEvidence",
    "MetricOptimizerEvidence",
    "MetricRemeshingCriterion",
    "MetricRemeshingEvidence",
    "MetricRemeshingStatus",
    "NativeTetraAimEvidence",
    "NativeTetraMetricOutcome",
    "TetraMetricOutcome",
    "TetraMetricSource",
    "TetraShapeAims",
    "UniformSizeRemeshingGoal",
    "execute_native_tetra_metric",
    "execute_tetra_metric_adaptation",
]
