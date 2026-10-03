# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Q15 adaptive/learned/calibrated/hybrid and Q17 long-run restart workloads.

Every workload calls the public package API (and the public builders of the
examples that are the documented consumers of those APIs), measures its
preparation, solve, transfer, epoch-commit, restart and output phases
separately through ``benchmarks.meshfree_scaling.PhaseRecorder``, declares its
planning reservation before any work, and returns measured scientific evidence
beside timing. Workloads are registered in ``benchmarks.meshfree_closure``;
nothing here writes a record or decides a qualification gate.

Examples are imported inside the workloads: importing this module never
mutates JAX configuration.
"""

from __future__ import annotations

import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from math import comb, isqrt
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import phydrax.linalg as la
from benchmarks._runtime import logical_array_bytes
from benchmarks.meshfree_scaling import (
    declare_reservation,
    fill_distance,
    MeshfreeConfig,
    PhaseRecorder,
    PROJECT_ROOT,
    unit_cube_probes,
)
from phydrax import ArrayArchiveCorruptionError, execution, solver, uq
from phydrax.discretization import (
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    PointCollocationStability,
    PreparedPointCloudDiscretization,
    TopologyEpoch,
)
from phydrax.discretization.meshfree import (
    adaptation_acceptance,
    commit_meshfree_epoch,
    degree_difference_indicator,
    DistributedMeshfreeOperator,
    EdgeCoverageStatus,
    EdgeFeatureCoverage,
    EdgeFeatureField,
    EdgeFrameFeatures,
    LipschitzCoupledEdgeFlux,
    LocalStencilPolicy,
    mark_points,
    meshfree_runtime_inventory,
    MeshfreeAdaptationAcceptance,
    MeshfreeAdaptationPolicy,
    MeshfreeAdaptationProposal,
    MeshfreeAdaptationStatus,
    MeshfreeAdaptiveSupport,
    MeshfreeCorrectionStatus,
    MeshfreeDiffusionLaw,
    MeshfreeEpochChange,
    MeshfreeErrorIndicator,
    MeshfreeEvolutionPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeFunctional,
    MeshfreeMarkingPolicy,
    MeshfreeMetricCorrectionPlan,
    MeshfreeMetricCorrectionPolicy,
    MeshfreeMotion,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    MonotoneCoupledEdgeFlux,
    moving_archive_policy,
    MovingSurfaceFixedStepMethod,
    MovingSurfacePlan,
    MovingSurfaceStatus,
    O3EdgeInvariants,
    O3EdgeNetwork,
    ownership_migration_relation,
    prepare_adaptation_transfer,
    prepare_local_stencils,
    PreparedMeshfreeEvolution,
    PreparedSurfacePointCloud,
    probe_residual_indicator,
    propose_adaptation,
    stage_meshfree_epoch,
    support_epoch_relation,
    SurfaceEllipticSystem,
)
from phydrax.discretization.spatial import (
    DistributedOwnershipPlan,
    DistributedPointLayout,
    MortonAddressPlan,
)
from phydrax.interfacial_transport import FilmStepStatus
from phydrax.lifecycle import (
    Composition,
    CompositionEntry,
    CompositionRole,
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    ResolvedRunSpec,
)
from phydrax.nn.models import PartiallyInputConvexNetwork
from phydrax.nn.operator.representations import O3Representation
from phydrax.qualification import SupportDependency
from phydrax.solver.coupling import (
    MeshfreeBulkSurfaceMethod,
    prepare_overlap_schwarz,
    solve_coupled_problem,
)
from phydrax.sparse import SparseCoordinateOperator
from phydrax.typing import parse


if TYPE_CHECKING:
    from examples.meshfree_hybrid_calibrated import CaseBatch
    from examples.meshfree_learned_metric_correction import LearnedMetricTraining


_FUSED_SEARCH = "PointCloudPlan.prepare fuses the neighbor search with the local fit"


def _require_float64(config: MeshfreeConfig, workload: str, /) -> None:
    if config.precision != "float64":
        raise ValueError(
            f"{workload} is declared for float64 only; precision is not silently changed."
        )


def _record(
    workload: str,
    capacity: int,
    requested: int,
    seed: int,
    dimension: int,
    recorder: PhaseRecorder,
    metrics: dict[str, Any],
    /,
    *,
    retained: Any,
    oracle: str,
    consumer: str,
    reserved: int,
    series: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "workload": workload,
        "capacity": capacity,
        "requested_capacity": requested,
        "seed": seed,
        "dimension": dimension,
        "status": "measured",
        "phases": recorder.record(),
        "metrics": {key: value for key, value in metrics.items() if value is not None},
        "series": {} if series is None else series,
        "retained_bytes": logical_array_bytes(retained),
        "reserved_working_set_bytes": reserved,
        "oracle_provenance": oracle,
        "consumer": consumer,
        **extra,
    }


def _scope_work(phases: dict[str, Any], scopes: set[str], /) -> tuple[float, float]:
    """Wall and process-CPU seconds of every measured occurrence in ``scopes``."""
    wall = cpu = 0.0
    for record in phases.values():
        if record["status"] != "measured":
            continue
        for occurrence in record["occurrences"]:
            if occurrence["scope"] in scopes:
                wall += occurrence["wall_seconds"]
                cpu += occurrence["cpu_seconds"]
    return wall, cpu


def _loglog_at(abscissae: list[float], values: list[float], at: float, /) -> float | None:
    """Log-log interpolation inside a strictly increasing measured range, else None.

    No extrapolation: outside the measured range, or on a non-monotone abscissa
    (for example compile-dominated work), the comparison is not measured.
    """
    x = np.log(np.asarray(abscissae, dtype=np.float64))
    if x.size < 2 or not np.all(np.diff(x) > 0) or not x[0] <= math.log(at) <= x[-1]:
        return None
    y = np.log(np.asarray(values, dtype=np.float64))
    return float(np.exp(np.interp(math.log(at), x, y)))


def _gain(reference: float | None, achieved: float, /) -> float | None:
    return None if reference is None else reference / achieved


def _square_fill(points: np.ndarray, seed: int, /) -> dict[str, Any]:
    return fill_distance(
        points,
        unit_cube_probes(2, 4 * points.shape[0], seed),
        provenance="max nearest-sample distance over scrambled Sobol probes of [0,1]^2 "
        "(a lower estimate of the supremum)",
    )


def _adaptive_comparison(
    uniform_points: list[int],
    uniform_errors: list[float],
    uniform_cpu: list[float],
    adaptive_points: list[int],
    adaptive_errors: list[float],
    adaptive_cpu: list[float],
    /,
) -> dict[str, float | bool | None]:
    """Uniform error at the adaptive point count; uniform work at the adaptive error.

    ``*_cpu`` are warmed per-scope CPU seconds: the identical deterministic
    clouds were executed once untimed before, so compilation is excluded and
    reported separately (``*_cold``). The adaptive route must solve every
    level, so its work is cumulative; a uniform solve is one solve at its size.

    The uniform sweep extends over declared capacities until its error is at
    most the adaptive final error. The uniform CPU work at that error is then
    interpolated (log-log, error decreasing with work) inside the measured
    uniform range, never extrapolated; the equal-error work gain is that work
    over the adaptive cumulative work. If the declared sweep never reaches the
    adaptive error, ``uniform_reaches_adaptive_error`` is False and the work
    comparison is unmeasured.
    """
    final_points, final_error = adaptive_points[-1], adaptive_errors[-1]
    work = float(sum(adaptive_cpu))
    equal_points = _loglog_at(
        [float(n) for n in uniform_points], uniform_errors, final_points
    )
    reached = min(uniform_errors) <= final_error
    # Reversed: the uniform error must strictly increase as work decreases.
    equal_error_work = (
        _loglog_at(uniform_errors[::-1], uniform_cpu[::-1], final_error)
        if reached
        else None
    )
    # Same-work envelope: a measured uniform row that already spends at least
    # the adaptive cumulative work and still has a worse error.
    worse_at_same_work = any(
        cpu >= work and error > final_error
        for cpu, error in zip(uniform_cpu, uniform_errors, strict=True)
    )
    return {
        "uniform_error_at_equal_points": equal_points,
        "adaptive_gain_at_equal_points": _gain(equal_points, final_error),
        "uniform_reaches_adaptive_error": reached,
        "uniform_points_max": int(max(uniform_points)),
        "uniform_cpu_at_equal_error": equal_error_work,
        "adaptive_work_gain_at_equal_error": None
        if equal_error_work is None
        else equal_error_work / work,
        "adaptive_cumulative_cpu_seconds": work,
        "uniform_worse_at_or_above_adaptive_work": worse_at_same_work,
        # The work comparison is measured only inside a strictly monotone
        # uniform error range; these facts explain an unmeasured comparison.
        "uniform_cpu_seconds_max": float(max(uniform_cpu)),
        "uniform_errors_decreasing": bool(
            np.all(np.diff(np.asarray(uniform_errors)) < 0)
        ),
    }


# ---------------------------------------------------------------------------
# Q15 bulk boundary layer: h-adaptivity versus uniform refinement


_LAYER_DEGREE = 3
_LAYER_NEIGHBORS = 30
_LAYER_START = 12
# Two adaptive levels: with the default grading 2.05 a third level would need
# more points than the finest uniform cloud (measured 1219 > 1156) and is
# capacity-refused by the declared adaptive cap.
_LAYER_LEVELS = 2
_LAYER_OWNER = "q15-adaptive-boundary-layer"
_LAYER_ORACLE = (
    "analytic u = exp(-x/0.04)(1 + sin(pi y)/4) of -Laplace(u) = f on the unit square; "
    "max nodal error"
)
_LAYER_CONSUMER = (
    "phydrax.discretization.meshfree.{probe_residual_indicator, mark_points, "
    "propose_adaptation, prepare_adaptation_transfer, adaptation_acceptance, "
    "stage_meshfree_epoch, commit_meshfree_epoch} via examples.meshfree_adaptive_learning builders"
)


def _layer_axes(capacity: int, /) -> tuple[int, ...]:
    """The scenario's four uniform axis counts 12 * sqrt(2)^k whose squares fit."""
    axes: list[int] = []
    while len(axes) < 4:
        axis = round(_LAYER_START * math.sqrt(2.0) ** len(axes))
        if axis * axis > capacity:
            break
        axes.append(axis)
    return tuple(axes)


# Equal-error work comparison (Main's P16 decision): uniform axes continuing
# 12 * sqrt(2)^k beyond the four scenario resolutions, declared before
# execution, run only until the uniform error reaches the adaptive error.
_LAYER_EXTENSION_AXES = (48, 68)


def _layer_reservation(capacity: int, config: MeshfreeConfig, /) -> int:
    basis = comb(2 + _LAYER_DEGREE, _LAYER_DEGREE)
    # Two live clouds (source and candidate): stencil relations and local systems,
    # 40 GMRES Krylov vectors and an ILU with up to three times the stencil fill.
    per_point = _LAYER_NEIGHBORS * (basis + 2 + 12) + 40 + 3 * _LAYER_NEIGHBORS
    return declare_reservation(
        2 * 8 * capacity * per_point, config, scope="adaptive boundary-layer clouds"
    )


@dataclass(frozen=True)
class _LayerSolve:
    values: Array
    successful: bool
    stability: PointCollocationStability | None

    @property
    def outcome(self) -> str:
        return "unassessed" if self.stability is None else str(self.stability.outcome)


def _layer_solve(
    recorder: PhaseRecorder, cloud: PreparedPointCloudDiscretization, scope: str, /
) -> _LayerSolve:
    """Square collocation with the owner's spectral stability as evidence."""
    from examples.meshfree_adaptive_learning import layer_exact, layer_source

    points = np.asarray(cloud.points)
    rows = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    plan = PointCloudPoissonPlan(
        cloud,
        PointBoundaryPlan(
            (
                PointBoundaryCondition(
                    "dirichlet", rows, layer_exact(points)[rows], label="walls"
                ),
            ),
            row_count=points.shape[0],
        ),
        linear_policy=la.LinearSolvePolicy(
            la.GMRES(restart=min(40, points.shape[0])),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-10, max_steps=1000),
            preconditioning=la.PreconditioningPolicy(
                la.ILUPreconditionerBuilder(), refresh="numeric"
            ),
            failure=la.FailurePolicy("status"),
        ),
        stability="diagnostic",
    )
    prepared = recorder.run("assembly", lambda: plan.prepare(1.0), scope=scope)
    source = layer_source(points)
    result = recorder.run("solve", lambda: prepared.solve(source), scope=scope)
    return _LayerSolve(result.values, bool(result.successful), prepared.stability)


def _layer_error(cloud: PreparedPointCloudDiscretization, values: Array, /) -> float:
    from examples.meshfree_adaptive_learning import layer_exact

    points = np.asarray(cloud.points)
    return float(np.max(np.abs(np.asarray(values) - layer_exact(points))))


def _entry(
    value: object,
    entry_id: str,
    role: CompositionRole,
    structure: str,
    revision: str,
    *dependencies: CompositionEntry,
) -> CompositionEntry:
    return CompositionEntry(
        value,
        entry_id=entry_id,
        role=role,
        owner_id=_LAYER_OWNER,
        structure_id=structure,
        revision_id=revision,
        semantics_id=entry_id.split("/")[0],
        dependencies=tuple(item.binding("structure") for item in dependencies),
    )


def _layer_epoch(index: int, /) -> TopologyEpoch:
    return TopologyEpoch(index, f"square-cloud-{index}", "boundary-layer", "serial")


def _layer_composition(
    epoch: TopologyEpoch, cloud: PreparedPointCloudDiscretization, values: Array, /
) -> Composition:
    root = _entry(epoch, "cloud/epoch", "topology", epoch.epoch_id, "initial")
    return Composition(
        (
            root,
            _entry(
                values,
                "field/solution",
                "physical-state",
                epoch.epoch_id,
                "solution",
                root,
            ),
            _entry(
                cloud,
                "support/cloud",
                "discretization",
                epoch.epoch_id,
                cloud.prepared_id,
                root,
            ),
        ),
        boundary_id=f"adaptive-{epoch.index}",
    )


@dataclass(frozen=True)
class _LayerTransaction:
    composition: Composition
    published: bool
    solve: _LayerSolve | None
    acceptance: MeshfreeAdaptationAcceptance
    transfer_status: str
    conservation_residual: float | None
    # The candidate's residual indicator: the next level's marking indicator.
    after: MeshfreeErrorIndicator | None


def _layer_transaction(
    recorder: PhaseRecorder,
    source: Composition,
    proposal: MeshfreeAdaptationProposal,
    indicator: MeshfreeErrorIndicator,
    scope: str,
    /,
) -> _LayerTransaction:
    """Candidate cloud, transfer, solve, indicator, acceptance and epoch commit."""
    from examples.meshfree_adaptive_learning import bulk_cloud, layer_residual

    cloud = source.value("support/cloud")
    epoch = source.value("cloud/epoch")
    target = _layer_epoch(epoch.index + 1)
    candidate = recorder.run(
        "local-fit",
        lambda: bulk_cloud(
            np.asarray(proposal.target_points),
            np.asarray(proposal.target_boundary),
            np.asarray(proposal.target_normals),
            np.asarray(proposal.target_ids),
            degree=proposal.target_degree,
            neighbors=proposal.target_neighbors,
        ),
        scope=scope,
    )
    transfer = recorder.run(
        "transfer", lambda: prepare_adaptation_transfer(cloud, proposal), scope=scope
    )
    solve: _LayerSolve | None = None
    after: MeshfreeErrorIndicator | None = None
    if candidate.report.refused_rows == 0:
        solve = _layer_solve(recorder, candidate, scope)
        solved = solve.values
        after = recorder.run(
            "local-fit",
            lambda: probe_residual_indicator(candidate, solved, layer_residual),
            scope=scope,
        )
    acceptance = adaptation_acceptance(
        proposal,
        transfer,
        candidate.report,
        solve_successful=solve is not None and solve.successful,
        stability=None if solve is None else solve.stability,
        indicator_before=indicator if after is not None else None,
        indicator_after=after,
    )
    if not transfer.admitted:
        return _LayerTransaction(
            source, False, solve, acceptance, transfer.evidence.status.name, None, after
        )
    change = MeshfreeEpochChange(
        epoch, target, cause="adaptive-refinement", proposal=proposal
    )
    root = CompositionEntry(
        target,
        entry_id="cloud/epoch",
        role="topology",
        owner_id=_LAYER_OWNER,
        structure_id=target.epoch_id,
        revision_id=change.change_id,
        semantics_id="cloud",
    )
    rebuilt = _entry(
        candidate,
        "support/cloud",
        "discretization",
        target.epoch_id,
        candidate.prepared_id,
        root,
    )

    def commit() -> Any:
        staged = stage_meshfree_epoch(
            source,
            change,
            epoch_entry="cloud/epoch",
            remap={"field/solution": transfer.epoch_transition(epoch, target)},
            reprepare=(rebuilt,),
        )
        return commit_meshfree_epoch(staged, accepted_boundary=acceptance.accepted)

    receipt = recorder.run("epoch-commit", commit, scope=scope)
    return _LayerTransaction(
        receipt.composition,
        bool(receipt.published),
        solve,
        acceptance,
        transfer.evidence.status.name,
        float(np.max(np.abs(np.asarray(receipt.conservation_residuals)))),
        after,
    )


def _layer_initial(
    recorder: PhaseRecorder, axis: int, seed: int, scope: str, /
) -> tuple[PreparedPointCloudDiscretization, _LayerSolve]:
    from examples.meshfree_adaptive_learning import bulk_cloud, square_cloud

    points, boundary, normals = square_cloud(axis, seed)
    cloud = recorder.run(
        "local-fit",
        lambda: bulk_cloud(points, boundary, normals, np.arange(points.shape[0])),
        scope=scope,
    )
    return cloud, _layer_solve(recorder, cloud, scope)


@dataclass(frozen=True)
class _LayerPass:
    """One complete uniform-plus-adaptive pass over the deterministic clouds."""

    uniform_points: list[int]
    uniform_errors: list[float]
    uniform_stability: list[str]
    uniform_successful: list[bool]
    uniform_cpu: list[float]
    finest: PreparedPointCloudDiscretization
    cloud: PreparedPointCloudDiscretization
    values: Array
    adaptive_points: list[int]
    adaptive_errors: list[float]
    adaptive_cpu: list[float]
    published: list[bool]
    stabilities: list[str]
    successful: list[bool]
    residuals: list[float]
    statuses: list[str]


def _layer_pass(
    recorder: PhaseRecorder,
    axes: tuple[int, ...],
    extension: tuple[int, ...],
    seed: int,
    /,
) -> _LayerPass:
    from examples.meshfree_adaptive_learning import layer_residual, square_projection

    uniform_points: list[int] = []
    uniform_errors: list[float] = []
    uniform_stability: list[str] = []
    uniform_successful: list[bool] = []
    clouds: list[tuple[PreparedPointCloudDiscretization, _LayerSolve]] = []
    for axis in axes:
        cloud, solve = _layer_initial(recorder, axis, seed, f"uniform-{axis * axis}")
        clouds.append((cloud, solve))
        uniform_points.append(axis * axis)
        uniform_errors.append(_layer_error(cloud, solve.values))
        uniform_stability.append(solve.outcome)
        uniform_successful.append(solve.successful)

    cloud, solve = clouds[0]
    values = solve.values
    composition = _layer_composition(_layer_epoch(0), cloud, values)
    adaptive_points = [uniform_points[0]]
    adaptive_errors = [uniform_errors[0]]
    scopes = [f"uniform-{uniform_points[0]}"]
    published: list[bool] = []
    stabilities: list[str] = []
    successful: list[bool] = [solve.successful]
    residuals: list[float] = []
    statuses: list[str] = []
    marking = MeshfreeMarkingPolicy("dorfler", fraction=0.9, maximum_marked=800)
    # The adaptive route may never exceed the finest uniform cloud of the scenario.
    adaptation = MeshfreeAdaptationPolicy(
        total_measure=1.0, maximum_points=uniform_points[-1]
    )
    indicator: MeshfreeErrorIndicator | None = None
    for level in range(1, _LAYER_LEVELS + 1):
        scope = f"adaptive-{level}"
        current, current_values = cloud, values
        if indicator is None:
            indicator = recorder.run(
                "local-fit",
                lambda: probe_residual_indicator(current, current_values, layer_residual),
                scope=scope,
            )
        marked = indicator
        proposal = recorder.run(
            "geometry",
            lambda: propose_adaptation(
                MeshfreeAdaptiveSupport.from_point_cloud(current),
                marked,
                mark_points(marked, marking),
                adaptation,
                boundary_projection=square_projection,
            ),
            scope=scope,
        )
        statuses.append(proposal.status.name)
        scopes.append(scope)
        if not proposal.admitted:
            break
        outcome = _layer_transaction(recorder, composition, proposal, marked, scope)
        published.append(outcome.published)
        stabilities.append(str(outcome.acceptance.stability_outcome))
        successful.append(outcome.solve is not None and outcome.solve.successful)
        if outcome.conservation_residual is not None:
            residuals.append(outcome.conservation_residual)
        if not outcome.published or outcome.solve is None:
            break
        composition = outcome.composition
        cloud, values = composition.value("support/cloud"), outcome.solve.values
        # The accepted candidate's residual indicator is the next marking input.
        indicator = outcome.after
        adaptive_points.append(int(np.asarray(cloud.points).shape[0]))
        adaptive_errors.append(_layer_error(cloud, values))
    finest = clouds[-1][0]
    for axis in extension:
        if min(uniform_errors) <= adaptive_errors[-1]:
            break
        finest, extended = _layer_initial(recorder, axis, seed, f"uniform-{axis * axis}")
        uniform_points.append(axis * axis)
        uniform_errors.append(_layer_error(finest, extended.values))
        uniform_stability.append(extended.outcome)
        uniform_successful.append(extended.successful)
    phases = recorder.record()
    uniform_cpu = [_scope_work(phases, {f"uniform-{n}"})[1] for n in uniform_points]
    return _LayerPass(
        uniform_points,
        uniform_errors,
        uniform_stability,
        uniform_successful,
        uniform_cpu,
        finest,
        cloud,
        values,
        adaptive_points,
        adaptive_errors,
        [_scope_work(phases, {scope})[1] for scope in scopes],
        published,
        stabilities,
        successful,
        residuals,
        statuses,
    )


def measure_adaptive_bulk_boundary_layer(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Residual-indicator h-adaptivity versus uniform refinement at equal points/work.

    The deterministic scenario runs twice: the first (cold) pass compiles every
    program of the identical clouds; work is compared on the second (warm)
    pass, and the cold per-scope CPU is retained as evidence.
    """
    _require_float64(config, "adaptive-bulk-boundary-layer")
    axes = _layer_axes(capacity)
    if len(axes) < 4:
        raise ValueError(
            "The boundary-layer comparison needs four uniform resolutions: capacity >= 34^2."
        )
    # Declared up front over the largest uniform extension cloud.
    extension = tuple(axis for axis in _LAYER_EXTENSION_AXES if axis > axes[-1])
    finest_axis = max((*axes, *extension))
    reserved = _layer_reservation(finest_axis * finest_axis, config)
    cold_recorder = PhaseRecorder()
    cold_recorder.unavailable("search", _FUSED_SEARCH)
    cold = _layer_pass(cold_recorder, axes, extension, seed)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    warm = _layer_pass(recorder, axes, extension, seed)
    comparison = _adaptive_comparison(
        warm.uniform_points,
        warm.uniform_errors,
        warm.uniform_cpu,
        warm.adaptive_points,
        warm.adaptive_errors,
        warm.adaptive_cpu,
    )
    requested = _LAYER_LEVELS
    metrics: dict[str, Any] = {
        **comparison,
        "uniform_finest_points": warm.uniform_points[-1],
        "uniform_finest_max_error": warm.uniform_errors[-1],
        "adaptive_final_points": warm.adaptive_points[-1],
        "adaptive_final_max_error": warm.adaptive_errors[-1],
        "adaptive_levels_requested": requested,
        "adaptive_levels_published": sum(warm.published),
        "all_epochs_published": len(warm.published) == requested and all(warm.published),
        "all_epoch_stability_admitted": len(warm.stabilities) == requested
        and all(item == "admitted" for item in warm.stabilities),
        "uniform_stability_admitted": all(
            item == "admitted" for item in warm.uniform_stability
        ),
        "all_solves_successful": all(warm.successful) and all(warm.uniform_successful),
        "max_epoch_conservation_residual": max(warm.residuals)
        if warm.residuals
        else None,
        "adaptive_error_decreased": warm.adaptive_errors[-1] < warm.adaptive_errors[0],
        "warm_pass_reproduces_cold": warm.adaptive_errors == cold.adaptive_errors
        and warm.uniform_errors == cold.uniform_errors,
        "uniform_cpu_seconds_cold_max": float(max(cold.uniform_cpu)),
        "adaptive_cumulative_cpu_seconds_cold": float(sum(cold.adaptive_cpu)),
    }
    return _record(
        "adaptive-bulk-boundary-layer",
        max(warm.uniform_points[-1], warm.adaptive_points[-1]),
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(warm.finest, warm.cloud, warm.values),
        oracle=_LAYER_ORACLE,
        consumer=_LAYER_CONSUMER,
        reserved=reserved,
        series={
            "uniform_points": warm.uniform_points,
            "uniform_max_errors": warm.uniform_errors,
            "uniform_cpu_seconds": warm.uniform_cpu,
            "uniform_cpu_seconds_cold": cold.uniform_cpu,
            "uniform_stability": warm.uniform_stability,
            "adaptive_points": warm.adaptive_points,
            "adaptive_max_errors": warm.adaptive_errors,
            "adaptive_level_cpu_seconds": warm.adaptive_cpu,
            "adaptive_level_cpu_seconds_cold": cold.adaptive_cpu,
            "adaptive_proposal_statuses": warm.statuses,
            "adaptive_epoch_stability": warm.stabilities,
            "adaptive_epoch_published": warm.published,
            "uniform_finest_fill": _square_fill(np.asarray(warm.finest.points), seed),
            "adaptive_final_fill": _square_fill(np.asarray(warm.cloud.points), seed),
        },
    )


def measure_adaptive_bulk_rejected_coarsening(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Blind coarsening raises the residual indicator; the epoch is refused whole."""
    from examples.meshfree_adaptive_learning import layer_residual, square_projection

    _require_float64(config, "adaptive-bulk-rejected-coarsening")
    axis = min(isqrt(capacity), _LAYER_START)
    if axis < _LAYER_START:
        raise ValueError("The rejected-coarsening scenario needs capacity >= 144.")
    reserved = _layer_reservation(axis * axis, config)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    cloud, solve = _layer_initial(recorder, axis, seed, "source")
    source = _layer_composition(_layer_epoch(0), cloud, solve.values)
    values = solve.values
    indicator = recorder.run(
        "local-fit",
        lambda: probe_residual_indicator(cloud, values, layer_residual),
        scope="source",
    )
    # A zero indicator marks nothing and makes every interior point a removal
    # candidate; the staged epoch is judged against the residual indicator.
    blind = MeshfreeErrorIndicator(
        np.zeros(indicator.values.shape[0], dtype=np.float64),
        indicator.stable_ids,
        kind="probe-residual",
        probe_count=0,
    )
    proposal = recorder.run(
        "geometry",
        lambda: propose_adaptation(
            MeshfreeAdaptiveSupport.from_point_cloud(cloud),
            blind,
            mark_points(blind, MeshfreeMarkingPolicy()),
            MeshfreeAdaptationPolicy(
                total_measure=1.0, coarsen_fraction=1.0, maximum_removed=10_000
            ),
            boundary_projection=square_projection,
        ),
        scope="candidate",
    )
    outcome = _layer_transaction(recorder, source, proposal, indicator, "candidate")
    metrics = {
        "proposal_admitted": proposal.admitted,
        "removed_points": int(np.asarray(proposal.removed_ids).shape[0]),
        "epoch_published": outcome.published,
        "acceptance_refused": not outcome.acceptance.accepted,
        "refused_indicator_not_reduced": "indicator-not-reduced"
        in outcome.acceptance.refusals,
        "source_composition_unchanged": outcome.composition is source,
        "indicator_before": outcome.acceptance.indicator_before,
        "indicator_after": outcome.acceptance.indicator_after,
    }
    return _record(
        "adaptive-bulk-rejected-coarsening",
        axis * axis,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(cloud, values),
        oracle="adaptation_acceptance indicator comparison (indicator, not error bound)",
        consumer=_LAYER_CONSUMER,
        reserved=reserved,
        series={"refusals": list(outcome.acceptance.refusals)},
    )


def measure_adaptive_bulk_capacity_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A proposal beyond its declared insertion capacity is refused before staging."""
    from examples.meshfree_adaptive_learning import layer_residual, square_projection

    _require_float64(config, "adaptive-bulk-capacity-refusal")
    axis = min(isqrt(capacity), _LAYER_START)
    if axis < _LAYER_START:
        raise ValueError("The capacity-refusal scenario needs capacity >= 144.")
    reserved = _layer_reservation(axis * axis, config)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    cloud, solve = _layer_initial(recorder, axis, seed, "source")
    values = solve.values
    indicator = recorder.run(
        "local-fit",
        lambda: probe_residual_indicator(cloud, values, layer_residual),
        scope="source",
    )
    maximum_inserted = 4
    proposal = recorder.run(
        "geometry",
        lambda: propose_adaptation(
            MeshfreeAdaptiveSupport.from_point_cloud(cloud),
            indicator,
            mark_points(indicator, MeshfreeMarkingPolicy("dorfler", fraction=0.9)),
            MeshfreeAdaptationPolicy(
                total_measure=1.0, maximum_inserted=maximum_inserted
            ),
            boundary_projection=square_projection,
        ),
        scope="candidate",
    )
    metrics = {
        "capacity_refused": proposal.status is MeshfreeAdaptationStatus.CAPACITY_REFUSED,
        "proposal_admitted": proposal.admitted,
        "declared_maximum_inserted": maximum_inserted,
        "refined_points": proposal.refined_points,
    }
    return _record(
        "adaptive-bulk-capacity-refusal",
        axis * axis,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(cloud, values),
        oracle="declared MeshfreeAdaptationPolicy.maximum_inserted",
        consumer=_LAYER_CONSUMER,
        reserved=reserved,
        series={"proposal_status": proposal.status.name},
    )


# ---------------------------------------------------------------------------
# Q15 curved surface: degree-difference indicator on the sphere


_SPHERE_START = 192
_SPHERE_NEIGHBORS = 30
# Equal-error work comparison (Main's P16 decision): uniform counts doubling
# beyond the three scenario resolutions, declared before execution, run only
# until the uniform error reaches the adaptive error.
_SPHERE_EXTENSION_COUNTS = (1536, 3072)
_SPHERE_ORACLE = (
    "analytic zonal u = exp(6 x.a), a = (0.6, 0, 0.8), of -Laplace_S(u) + u = f on the "
    "unit sphere; measure-weighted relative L2 error"
)


def _sphere_solve(
    recorder: PhaseRecorder, surface: PreparedSurfacePointCloud, scope: str, /
) -> tuple[Array, bool]:
    from examples.meshfree_adaptive_learning import sphere_source

    points = np.asarray(surface.points)
    system = recorder.run(
        "assembly",
        lambda: SurfaceEllipticSystem(
            (surface,), diffusivity=1.0, reaction=1.0, system_id="screened-sphere"
        ),
        scope=scope,
    )
    rhs = system.rhs((jnp.asarray(sphere_source(points)),), (None,))
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=min(200, points.shape[0])),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=4000),
        failure=la.FailurePolicy("status"),
    )
    result = recorder.run(
        "solve", lambda: la.solve(system.linear_system, rhs, policy=policy), scope=scope
    )
    return system.split(result.value)[0], bool(result.successful)


@dataclass(frozen=True)
class _SpherePass:
    """One complete uniform-plus-adaptive pass over the deterministic spheres."""

    uniform_errors: list[float]
    uniform_cpu: list[float]
    adaptive_points: list[int]
    adaptive_errors: list[float]
    adaptive_cpu: list[float]
    statuses: list[str]
    successful: list[bool]
    surface: PreparedSurfacePointCloud
    values: Array
    uniform_points: list[int]


def _sphere_uniform(recorder: PhaseRecorder, count: int, /) -> tuple[float, bool]:
    from examples.meshfree_adaptive_learning import (
        fibonacci_sphere,
        sphere_cloud,
        sphere_error,
    )

    scope = f"uniform-{count}"
    points = fibonacci_sphere(count)
    surface = recorder.run(
        "local-fit", lambda: sphere_cloud(points, 3, _SPHERE_NEIGHBORS), scope=scope
    )
    values, ok = _sphere_solve(recorder, surface, scope)
    return sphere_error(surface, values), ok


def _sphere_pass(
    recorder: PhaseRecorder, counts: list[int], extension: tuple[int, ...], /
) -> _SpherePass:
    from examples.meshfree_adaptive_learning import (
        fibonacci_sphere,
        sphere_adaptation,
        sphere_cloud,
        sphere_error,
        sphere_marking,
        sphere_projection,
    )

    uniform_points = list(counts)
    uniform_errors: list[float] = []
    successful: list[bool] = []
    for count in counts:
        error, ok = _sphere_uniform(recorder, count)
        uniform_errors.append(error)
        successful.append(ok)

    points = fibonacci_sphere(counts[0])
    ids = np.arange(counts[0], dtype=np.int64)
    adaptive_points: list[int] = []
    adaptive_errors: list[float] = []
    statuses: list[str] = []
    scopes: list[str] = []
    marking = sphere_marking()
    adaptation = sphere_adaptation(maximum_points=counts[-1])
    for level in range(len(counts)):
        scope = f"adaptive-{level}"
        scopes.append(scope)
        level_points = points
        surface = recorder.run(
            "local-fit",
            lambda: sphere_cloud(level_points, 3, _SPHERE_NEIGHBORS),
            scope=scope,
        )
        values, ok = _sphere_solve(recorder, surface, scope)
        successful.append(ok)
        adaptive_points.append(points.shape[0])
        adaptive_errors.append(sphere_error(surface, values))
        if level == len(counts) - 1:
            break
        higher = recorder.run(
            "local-fit",
            lambda: sphere_cloud(level_points, 4, _SPHERE_NEIGHBORS),
            scope=scope,
        )
        level_values, level_ids = values, ids
        indicator = recorder.run(
            "local-fit",
            lambda: degree_difference_indicator(
                surface.laplace_beltrami.mv,
                higher.laplace_beltrami.mv,
                level_values,
                np.asarray(surface.measures),
                level_ids,
            ),
            scope=scope,
        )
        support = MeshfreeAdaptiveSupport(
            np.asarray(surface.points),
            ids,
            kind="surface",
            intrinsic_dimension=2,
            degree=3,
            neighbors=_SPHERE_NEIGHBORS,
        )
        proposal = recorder.run(
            "geometry",
            lambda: propose_adaptation(
                support,
                indicator,
                mark_points(indicator, marking),
                adaptation,
                manifold_projection=sphere_projection,
            ),
            scope=scope,
        )
        statuses.append(proposal.status.name)
        if not proposal.admitted:
            break
        points = np.asarray(proposal.target_points)
        ids = np.asarray(proposal.target_ids)
    for count in extension:
        if min(uniform_errors) <= adaptive_errors[-1]:
            break
        error, ok = _sphere_uniform(recorder, count)
        uniform_points.append(count)
        uniform_errors.append(error)
        successful.append(ok)
    phases = recorder.record()
    return _SpherePass(
        uniform_errors,
        [_scope_work(phases, {f"uniform-{count}"})[1] for count in uniform_points],
        adaptive_points,
        adaptive_errors,
        [_scope_work(phases, {scope})[1] for scope in scopes],
        statuses,
        successful,
        surface,
        values,
        uniform_points,
    )


def measure_adaptive_sphere_zonal_peak(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Degree-3 versus degree-4 indicator h-adaptivity on the sphere versus uniform.

    The deterministic scenario runs twice; work is compared on the warm pass
    and the cold (compiling) per-scope CPU is retained as evidence.
    """
    _require_float64(config, "adaptive-sphere-zonal-peak")
    config.check_point_limit(capacity)
    counts: list[int] = []
    while len(counts) < 3 and _SPHERE_START * 2 ** len(counts) <= capacity:
        counts.append(_SPHERE_START * 2 ** len(counts))
    if len(counts) < 3:
        raise ValueError(
            "The sphere comparison needs three uniform resolutions: capacity >= 768."
        )
    basis = comb(2 + 4, 4)
    extension = tuple(
        count
        for count in _SPHERE_EXTENSION_COUNTS
        if counts[-1] < count <= config.max_points
    )
    # Declared up front over the largest uniform extension cloud.
    reserved = declare_reservation(
        2 * 8 * max((*counts, *extension)) * (_SPHERE_NEIGHBORS * (basis + 3 + 12) + 200),
        config,
        scope="adaptive sphere clouds (degree 3 and 4)",
    )
    fused = "SurfacePointCloudPlan.prepare fuses search, geometry and the local fit"
    cold_recorder = PhaseRecorder()
    cold_recorder.unavailable("search", fused)
    cold = _sphere_pass(cold_recorder, counts, extension)
    recorder = PhaseRecorder()
    recorder.unavailable("search", fused)
    warm = _sphere_pass(recorder, counts, extension)
    comparison = _adaptive_comparison(
        warm.uniform_points,
        warm.uniform_errors,
        warm.uniform_cpu,
        warm.adaptive_points,
        warm.adaptive_errors,
        warm.adaptive_cpu,
    )
    errors = warm.adaptive_errors
    metrics: dict[str, Any] = {
        **comparison,
        "uniform_finest_points": warm.uniform_points[-1],
        "uniform_finest_relative_l2": warm.uniform_errors[-1],
        "adaptive_final_points": warm.adaptive_points[-1],
        "adaptive_final_relative_l2": errors[-1],
        "all_proposals_admitted": len(warm.statuses) == len(counts) - 1
        and all(item == "ADMITTED" for item in warm.statuses),
        "all_solves_successful": all(warm.successful),
        "adaptive_error_monotone": all(
            later < earlier for earlier, later in zip(errors, errors[1:])
        ),
        "warm_pass_reproduces_cold": warm.adaptive_errors == cold.adaptive_errors
        and warm.uniform_errors == cold.uniform_errors,
        "uniform_cpu_seconds_cold_max": float(max(cold.uniform_cpu)),
        "adaptive_cumulative_cpu_seconds_cold": float(sum(cold.adaptive_cpu)),
    }
    return _record(
        "adaptive-sphere-zonal-peak",
        max(warm.uniform_points[-1], warm.adaptive_points[-1]),
        capacity,
        seed,
        3,
        recorder,
        metrics,
        retained=(warm.surface, warm.values),
        oracle=_SPHERE_ORACLE,
        consumer="phydrax.discretization.meshfree.{degree_difference_indicator, "
        "propose_adaptation(manifold_projection)} with SurfacePointCloudPlan/SurfaceEllipticSystem",
        reserved=reserved,
        series={
            "uniform_points": warm.uniform_points,
            "uniform_relative_l2": warm.uniform_errors,
            "uniform_cpu_seconds": warm.uniform_cpu,
            "uniform_cpu_seconds_cold": cold.uniform_cpu,
            "adaptive_points": warm.adaptive_points,
            "adaptive_relative_l2": errors,
            "adaptive_level_cpu_seconds": warm.adaptive_cpu,
            "adaptive_level_cpu_seconds_cold": cold.adaptive_cpu,
            "adaptive_proposal_statuses": warm.statuses,
            "sampling": "deterministic Fibonacci sphere; the seed does not change the cloud",
        },
    )


# ---------------------------------------------------------------------------
# Q15 learned metric correction: projection, audit refusal, training


_CORRECTION_MARGIN = 1e-2
_CORRECTION_SIZE = 6  # The documented 6 x 6 learned-correction lattice.
_COUPLED_EDGES = 64
_CORRECTION_CONSUMER = (
    "phydrax.discretization.meshfree.MeshfreeMetricCorrectionPlan.correct "
    "(full-moment projection with separate moment/sign/coercivity audits)"
)


def _correction_lattice(
    capacity: int, seed: int, /
) -> tuple[int, np.ndarray, np.ndarray]:
    size = min(isqrt(capacity), _CORRECTION_SIZE)
    if size < 4:
        raise ValueError("The learned-correction lattice needs capacity >= 16 nodes.")
    spacing = 1.0 / (size - 1)
    grid = (
        np.stack(
            np.meshgrid(np.arange(size), np.arange(size), indexing="ij"), axis=-1
        ).reshape((-1, 2))
        * spacing
    )
    interior = np.all((grid > 1e-9) & (grid < 1.0 - 1e-9), axis=1)
    jitter = np.random.default_rng(seed).uniform(
        -0.1 * spacing, 0.1 * spacing, grid.shape
    )
    return size, grid + interior[:, None] * jitter, interior


def _learned_projection(
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    sign: Literal["refuse", "constrained"],
    /,
) -> dict[str, Any]:
    _require_float64(config, f"learned-correction-{sign}")
    size, points, interior = _correction_lattice(capacity, seed)
    spacing = 1.0 / (size - 1)
    edges = 16 * size * size
    reserved = declare_reservation(
        8 * edges * (edges // 4 + 64), config, scope="edge-moment projection workspace"
    )
    recorder = PhaseRecorder()
    exterior = recorder.run(
        "assembly",
        lambda: MeshfreeExteriorCalculusPlan(
            points,
            2.05 * spacing,
            edges,
            node_volumes=np.full(size * size, spacing * spacing, dtype=np.float64),
            dirichlet=~interior,
        ).prepare(),
        scope="exterior-metric",
    )
    lengths = np.asarray(exterior.lengths)
    # A frozen edge-feature model stands in for any learned law: its output is
    # only a candidate, never trusted before projection and audit.
    candidate = np.asarray(exterior.metric_result.weights) + 1e-2 * np.tanh(
        3.0 * lengths / spacing
    )
    plan = MeshfreeMetricCorrectionPlan(
        exterior,
        policy=MeshfreeMetricCorrectionPolicy(margin=_CORRECTION_MARGIN, sign=sign),
    )
    corrected = recorder.run(
        "conic" if sign == "constrained" else "solve",
        lambda: plan.correct(candidate),
        scope=f"correction-{sign}",
    )
    evidence = corrected.evidence
    metrics: dict[str, Any] = {
        "admitted": corrected.status is MeshfreeCorrectionStatus.ADMITTED,
        "positivity_conflict": corrected.status
        is MeshfreeCorrectionStatus.POSITIVITY_CONFLICT,
        "weights_published": corrected.weights is not None,
        "moments_exact": bool(evidence.moments_exact),
        "max_moment_residual": float(evidence.maximum_moment_residual),
        "sign_margin": float(evidence.sign_margin),
        "minimum_weight_over_margin": float(evidence.minimum_weight) / _CORRECTION_MARGIN,
        "coercive": bool(evidence.coercive),
        "derivative_available": bool(evidence.derivative_available),
        "provider_conic": evidence.provider == "conic",
        "edges": int(lengths.shape[0]),
    }
    return _record(
        "learned-correction-constrained"
        if sign == "constrained"
        else "learned-correction-signed-refusal",
        size * size,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(exterior, corrected),
        oracle="original full edge-moment equations of the prepared exterior metric",
        consumer=_CORRECTION_CONSUMER,
        reserved=reserved,
        series={"status": corrected.status.name, "provider": str(evidence.provider)},
    )


def measure_learned_correction_constrained(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Explicitly selected nonnegative conic correction: exact moments and margin."""
    return _learned_projection(capacity, seed, config, "constrained")


def measure_learned_correction_signed_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Signed minimum-norm projection breaks the margin and is refused, never clipped."""
    return _learned_projection(capacity, seed, config, "refuse")


_TRAINING_TRUTH = (0.6, 0.3)
_TRAINING_CONSUMER = (
    "phydrax.solver.train_components over MeshfreeMetricCorrectionPlan.project and the "
    "implicit MeshfreeConservationProblem adjoint (examples.meshfree_learned_metric_correction)"
)


def _training(
    recorder: PhaseRecorder, capacity: int, config: MeshfreeConfig, workload: str, /
) -> tuple[int, int, LearnedMetricTraining]:
    from examples.meshfree_learned_metric_correction import prepare_training

    _require_float64(config, workload)
    size = min(isqrt(capacity), _CORRECTION_SIZE)
    if size < 4:
        raise ValueError(
            "The learned-correction training lattice needs capacity >= 16 nodes."
        )
    edges = 8 * size * size
    reserved = declare_reservation(
        8 * edges * (edges // 4 + 256),
        config,
        scope="projection, conservation and adjoint",
    )
    training = recorder.run(
        "assembly",
        lambda: prepare_training(size=size, truth=_TRAINING_TRUTH),
        scope="training",
    )
    return size, reserved, training


def measure_learned_correction_training(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Gradient through projection and implicit adjoint, native training, audits."""
    import optax

    from examples.meshfree_learned_metric_correction import (
        objective,
        train,
        training_loss,
    )

    recorder = PhaseRecorder()
    size, reserved, training = _training(
        recorder, capacity, config, "learned-correction-training"
    )
    target = objective(training)

    def loss(coefficients: Array) -> Array:
        return training_loss(training, target, coefficients)

    point = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    direction = jnp.asarray([0.8, 0.6], dtype=jnp.float64)
    gradient = recorder.run(
        "vjp", lambda: jax.grad(loss)(point), scope="training-gradient"
    )
    directional = float(jnp.vdot(gradient, direction))
    step = 1e-5
    central = recorder.run(
        "solve",
        lambda: (
            (loss(point + step * direction) - loss(point - step * direction)) / (2 * step)
        ),
        scope="central-difference",
    )
    initial = jnp.zeros((2,), dtype=jnp.float64)
    before = float(recorder.run("solve", lambda: loss(initial), scope="loss"))
    trained, coefficients = recorder.run(
        "solve",
        lambda: train(
            training, target, initial, optimizer=optax.lbfgs(), steps=6, seed=seed
        ),
        scope="train-components",
    )
    after = float(recorder.run("solve", lambda: loss(coefficients), scope="loss"))
    candidate = training.model(coefficients)(0.0)
    corrected = recorder.run(
        "solve", lambda: training.plan.correct(candidate), scope="trained-correction"
    )
    weights = training.plan.project(candidate).weights
    parameters = {"corrected-metric": weights}
    solution = recorder.run(
        "solve",
        lambda: training.prepared.solve(parameters=parameters),
        scope="trained-primal",
    )
    misfit = (solution.state - training.reference)[
        training.prepared.residual.free_indices
    ]
    adjoint = recorder.run(
        "vjp",
        lambda: training.prepared.adjoint(solution, misfit, parameters=parameters),
        scope="trained-adjoint",
    )
    central_value = float(central)
    metrics: dict[str, Any] = {
        "gradient_fd_relative_error": abs(directional - central_value)
        / max(abs(central_value), 1e-14),
        "loss_before": before,
        "loss_after": after,
        "loss_reduction_ratio": after / before,
        "accepted_updates": int(trained.accepted_updates),
        "trained_correction_admitted": corrected.status
        is MeshfreeCorrectionStatus.ADMITTED,
        "trained_max_moment_residual": float(corrected.evidence.maximum_moment_residual),
        "trained_primal_successful": bool(solution.accepted),
        "trained_adjoint_successful": bool(adjoint.accepted),
        "coefficient_recovery_error": float(
            np.max(np.abs(np.asarray(coefficients) - np.asarray(_TRAINING_TRUTH)))
        ),
    }
    return _record(
        "learned-correction-training",
        size * size,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(training, coefficients),
        oracle="central difference of the native objective; discrete identification of "
        "projected truth coefficients (0.6, 0.3) on one cloud (no continuum claim)",
        consumer=_TRAINING_CONSUMER,
        reserved=reserved,
        series={
            "directional_gradient": directional,
            "central_difference": central_value,
            "coefficients": np.asarray(coefficients).tolist(),
        },
    )


def measure_learned_correction_failed_step(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A candidate whose projection breaks the margin is rejected, never stepped on."""
    import optax

    from examples.meshfree_learned_metric_correction import (
        objective,
        train,
        training_loss,
    )

    recorder = PhaseRecorder()
    size, reserved, training = _training(
        recorder, capacity, config, "learned-correction-failed-step"
    )
    target = objective(training)
    failing = jnp.asarray([6.0, 0.0], dtype=jnp.float64)
    projection = recorder.run(
        "solve",
        lambda: training.plan.project(training.model(failing)(0.0)),
        scope="projection",
    )
    failed = float(
        recorder.run(
            "solve", lambda: training_loss(training, target, failing), scope="loss"
        )
    )
    rejected, kept = recorder.run(
        "solve",
        lambda: train(
            training,
            target,
            failing,
            optimizer=optax.sgd(1e-3),
            steps=1,
            seed=seed,
            rejection_budget=1,
        ),
        scope="train-components",
    )
    metrics: dict[str, Any] = {
        "projection_admissible": bool(projection.admissible),
        "failed_loss_nonfinite": not math.isfinite(failed),
        "accepted_updates": int(rejected.accepted_updates),
        "nonfinite_rejections": int(rejected.nonfinite_rejections),
        "coefficients_unchanged": bool(jnp.all(kept == failing)),
    }
    metrics["step_rejected"] = (
        metrics["accepted_updates"] == 0
        and metrics["nonfinite_rejections"] == 1
        and metrics["coefficients_unchanged"]
    )
    return _record(
        "learned-correction-failed-step",
        size * size,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(training, kept),
        oracle="declared sign margin of MeshfreeMetricCorrectionPolicy; reject-attempt objective",
        consumer=_TRAINING_CONSUMER,
        reserved=reserved,
    )


def _edge_geometry(size: int, /) -> np.ndarray:
    """Edge length, axis cosine and midpoint modulation of one lattice's edges."""
    from examples.meshfree_learned_metric_correction import training_lattice

    exterior = training_lattice(size)
    points = np.asarray(exterior.points)
    pairs = np.asarray(exterior.pairs)
    offsets = points[pairs[:, 1]] - points[pairs[:, 0]]
    lengths = np.linalg.norm(offsets, axis=1)
    midpoint = 0.5 * (points[pairs[:, 0]] + points[pairs[:, 1]])
    modulation = np.sin(np.pi * midpoint[:, 0]) * np.cos(np.pi * midpoint[:, 1])
    return np.stack((lengths, np.abs(offsets[:, 0]) / lengths, modulation), axis=1)


def measure_learned_correction_coverage_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A cloud outside the train-fitted edge-geometry support is refused, not clipped.

    ``capacity`` bounds the refined query lattice (the controlling count); the
    training lattice has twice its spacing.
    """
    _require_float64(config, "learned-correction-coverage-refusal")
    size = min((isqrt(capacity) + 1) // 2, _CORRECTION_SIZE)
    if size < 4:
        raise ValueError("The coverage scenario needs capacity >= 49 query nodes.")
    refined = 2 * size - 1
    reserved = declare_reservation(
        8 * 8 * refined * refined * 64, config, scope="edge features and coverage solves"
    )
    recorder = PhaseRecorder()
    training = recorder.run(
        "geometry", lambda: _edge_geometry(size), scope="training-edges"
    )
    # The same family at the same spacing is an in-support control; the
    # refined lattice (half spacing) has edges shorter than any training edge.
    control = recorder.run(
        "geometry", lambda: _edge_geometry(size), scope="control-edges"
    )
    shifted = recorder.run(
        "geometry", lambda: _edge_geometry(refined), scope="refined-edges"
    )
    coverage = recorder.run(
        "rank-certificate",
        lambda: EdgeFeatureCoverage.fit(
            training,
            training_domain=f"learned-correction-lattice-{size}x{size}",
            feature_names=("edge-length", "axis-cosine", "midpoint-modulation"),
        ),
        scope="coverage-fit",
    )
    inside = recorder.run("first", lambda: coverage.assess(control), scope="control")
    outside = recorder.run("first", lambda: coverage.assess(shifted), scope="refined")
    status = np.asarray(outside.status)
    metrics: dict[str, Any] = {
        "control_admitted": bool(inside.admitted),
        "refined_admitted": bool(outside.admitted),
        "refined_refused": not bool(outside.admitted),
        "refined_marginal_outside_edges": int(
            np.sum(status == int(EdgeCoverageStatus.MARGINAL_OUTSIDE))
        ),
        "refined_joint_outside_edges": int(
            np.sum(status == int(EdgeCoverageStatus.JOINT_OUTSIDE))
        ),
        "refined_covered_edges": int(np.sum(status == int(EdgeCoverageStatus.COVERED))),
        "refined_edges": int(status.shape[0]),
    }
    metrics["refused_outside_marginal_support"] = (
        metrics["refined_refused"] and metrics["refined_marginal_outside_edges"] > 0
    )
    return _record(
        "learned-correction-coverage-refusal",
        refined * refined,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(coverage, outside),
        oracle="train-only empirical quantile box and Mahalanobis support (EdgeFeatureCoverage)",
        consumer="phydrax.discretization.meshfree.EdgeFeatureCoverage.fit/assess (policy refuse)",
        reserved=reserved,
    )


# ---------------------------------------------------------------------------
# Q15 learned coupled O(3) laws: reversal parity and frame covariance


_COUPLED_STATE = O3Representation(
    scalars=1, pseudoscalars=1, vectors=2, pseudovectors=1, tensors=1
)
_PROPER = np.asarray(
    [[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]], dtype=np.float64
)
CoupledFamily: TypeAlias = Literal["monotone", "lipschitz"]


def _coupled_features(
    edges: int, seed: int, /, *, reversed_pairs: bool = False
) -> EdgeFrameFeatures:
    """A ring of random 3-D edge frames; ``reversed_pairs`` lists every edge backwards."""
    rng = np.random.default_rng(seed + 17)
    points = rng.normal(size=(edges, 3))
    nodes = np.arange(edges, dtype=np.int32)
    pairs = np.stack((nodes, (nodes + 1) % edges), axis=1).astype(np.int32)
    schema = (EdgeFeatureField("s", "scalar"), EdgeFeatureField("v", "vector"))
    values = (rng.normal(size=edges), rng.normal(size=(edges, 3)))
    return EdgeFrameFeatures(
        points, pairs[:, ::-1] if reversed_pairs else pairs, schema, values
    )


def _coupled_law(
    family: CoupledFamily, features: EdgeFrameFeatures, seed: int, /
) -> MonotoneCoupledEdgeFlux | LipschitzCoupledEdgeFlux:
    even, odd = features.even.shape[1], features.odd.shape[1]
    match parse(family, CoupledFamily, "family"):
        case "monotone":
            invariants = O3EdgeInvariants(
                _COUPLED_STATE,
                quadratic_representation=O3Representation(
                    scalars=2, pseudoscalars=1, vectors=2, pseudovectors=1, tensors=1
                ),
                linear_count=3,
                key=jr.key(seed),
            )
            potential = PartiallyInputConvexNetwork(
                context_size=even + odd,
                convex_size=invariants.size,
                width_size=6,
                depth=2,
                input_monotonicity="nondecreasing",
                key=jr.key(seed + 1),
            )
            return MonotoneCoupledEdgeFlux(
                invariants, potential, background_conductance=0.6, odd_size=odd
            )
        case "lipschitz":
            network = O3EdgeNetwork(
                _COUPLED_STATE,
                O3Representation(
                    scalars=3, pseudoscalars=1, vectors=2, pseudovectors=2, tensors=1
                ),
                context_size=even + odd,
                key=jr.key(seed + 2),
            )
            return LipschitzCoupledEdgeFlux(
                network, even_size=even, odd_size=odd, background_conductance=2.0
            )


def _states(edges: int, seed: int, scale: float, /) -> Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(scale * rng.normal(size=(edges, _COUPLED_STATE.packed_size)))


def _coupled_invariance(
    capacity: int, seed: int, config: MeshfreeConfig, family: CoupledFamily, /
) -> dict[str, Any]:
    _require_float64(config, f"learned-coupled-{family}")
    requested, capacity = capacity, min(capacity, _COUPLED_EDGES)
    if capacity < 3:
        raise ValueError("A coupled-law edge ring needs at least three edges.")
    reserved = declare_reservation(
        8 * capacity * _COUPLED_STATE.packed_size**2 * 64,
        config,
        scope="coupled-law Jacobian blocks and samples",
    )
    recorder = PhaseRecorder()
    features = recorder.run(
        "geometry", lambda: _coupled_features(capacity, seed), scope="edge-frames"
    )
    law = recorder.run(
        "assembly", lambda: _coupled_law(family, features, seed), scope="law"
    )
    states = _states(capacity, seed + 3, 1.0)
    _, compiled = recorder.compiled_action(
        lambda values: law.flux(values, features),
        states,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
        scope="flux",
    )

    def checks() -> dict[str, float]:
        # Every comparison uses the same (eager) evaluation path: compiled and
        # eager evaluations may legitimately differ by round-off.
        flux = law.flux(states, features)
        scale = float(jnp.max(jnp.abs(flux)))
        reversed_flux = law.flux(-states, features.reoriented())
        swapped = _coupled_features(capacity, seed, reversed_pairs=True)
        result = {
            "reversal_parity_max_abs": float(jnp.max(jnp.abs(reversed_flux + flux))),
            "independent_reversal_relative_error": float(
                jnp.max(jnp.abs(law.flux(-states, swapped) + flux))
            )
            / scale,
        }
        for label, orthogonal in (("rotation", _PROPER), ("rotoreflection", -_PROPER)):
            q = jnp.asarray(orthogonal)
            rotated = law.flux(
                _COUPLED_STATE.transform(states, q),
                features.transformed_frame(orthogonal),
            )
            result[f"{label}_covariance_relative_error"] = (
                float(jnp.max(jnp.abs(rotated - _COUPLED_STATE.transform(flux, q))))
                / scale
            )
        return result

    metrics: dict[str, Any] = recorder.run("warm", checks, scope="invariance")
    if isinstance(law, MonotoneCoupledEdgeFlux):
        bound = float(law.monotonicity_lower_bound())

        def monotone() -> dict[str, float]:
            ratios = []
            for sample in range(8):
                first = _states(capacity, 10 + sample, 3.0)
                second = _states(capacity, 20 + sample, 0.5)
                change = first - second
                gap = jnp.sum(
                    (law.flux(first, features) - law.flux(second, features)) * change,
                    axis=1,
                )
                ratios.append(float(jnp.min(gap / jnp.sum(change * change, axis=1))))
            blocks = law.derivative(_states(capacity, seed + 4, 2.0), features)
            return {
                "sampled_monotonicity_over_bound": min(ratios) / bound,
                "jacobian_symmetry_max_abs": float(
                    jnp.max(jnp.abs(blocks - blocks.transpose(0, 2, 1)))
                ),
                "jacobian_min_eigenvalue_over_bound": float(
                    jnp.min(
                        jnp.linalg.eigvalsh(0.5 * (blocks + blocks.transpose(0, 2, 1)))
                    )
                )
                / bound,
            }

        metrics.update(recorder.run("jvp", monotone, scope="monotonicity"))
    else:
        bound = float(law.lipschitz_bound())

        def lipschitz() -> dict[str, float]:
            worst = 0.0
            for sample in range(12):
                first = _states(capacity, 100 + sample, 4.0 / (sample + 1))
                second = first + _states(capacity, 200 + sample, 10.0 ** (-(sample % 4)))
                change = jnp.linalg.norm(
                    law.flux(first, features) - law.flux(second, features), axis=1
                )
                worst = max(
                    worst,
                    float(jnp.max(change / jnp.linalg.norm(first - second, axis=1))),
                )
            return {"sampled_lipschitz_over_bound": worst / bound}

        metrics.update(recorder.run("jvp", lipschitz, scope="lipschitz"))
    metrics["certified_bound"] = bound
    return _record(
        f"learned-coupled-{family}-invariance",
        capacity,
        requested,
        seed,
        3,
        recorder,
        metrics,
        retained=(law, features),
        oracle="exact O(3) action on the packed representation and independently reversed "
        "edge lists; the law's own certified bound",
        consumer=f"phydrax.discretization.meshfree.{type(law).__name__}.flux",
        reserved=reserved,
        compiler=compiled["compiler"],
    )


def measure_learned_coupled_monotone_invariance(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _coupled_invariance(capacity, seed, config, "monotone")


def measure_learned_coupled_lipschitz_invariance(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _coupled_invariance(capacity, seed, config, "lipschitz")


# ---------------------------------------------------------------------------
# Q15 hybrid point-cloud / finite-volume overlap Schwarz and calibration


_HYBRID_CONSUMER = (
    "phydrax.solver.coupling.{OverlapDirichletLaw, prepare_overlap_schwarz, "
    "solve_coupled_problem} with phydrax.linalg subspace-correction builders "
    "(examples.meshfree_hybrid_calibrated builders)"
)


def _hybrid_unknowns(cells: int, /) -> int:
    """Cloud points of the frame (lattice minus the open hole) plus grid cells."""
    hole = cells // 2 - 1
    return (cells + 1) ** 2 - hole * hole + (3 * cells // 4) ** 2


def _hybrid_level(recorder: PhaseRecorder, cells: int, seed: int, /) -> dict[str, Any]:
    import examples.meshfree_hybrid_calibrated as hybrid

    scope = f"cells-{cells}"
    setup = recorder.run(
        "local-fit",
        lambda: hybrid.prepare_setup(cells, hybrid.BASE_GEOMETRY, seed=seed),
        scope=scope,
    )
    prepared = recorder.run(
        "assembly",
        lambda: hybrid.hybrid_problem(setup, hybrid.REFERENCE_PARAMETERS),
        scope=scope,
    )
    schwarz = recorder.run(
        "ordering-fill",
        lambda: prepare_overlap_schwarz(prepared, hybrid.LAW),
        scope=scope,
    )
    builders: dict[str, Any] = {
        "unpreconditioned": None,
        "additive-schwarz": la.AdditiveSubspaceCorrectionBuilder(schwarz.terms),
        "multiplicative-schwarz": la.MultiplicativeSubspaceCorrectionBuilder(
            schwarz.terms
        ),
    }
    solutions = {
        name: recorder.run(
            "solve",
            lambda: solve_coupled_problem(
                prepared,
                policy=hybrid.gmres_policy(
                    builder, max_steps=300 if builder is None else 400
                ),
            ),
            scope=f"{scope}-{name}",
        )
        for name, builder in builders.items()
    }
    accepted = solutions["multiplicative-schwarz"]
    theta = jnp.asarray(hybrid.REFERENCE_PARAMETERS)
    cloud_error = float(
        jnp.max(
            jnp.abs(
                accepted.field(hybrid.CLOUD, hybrid.FIELD)
                - hybrid.exact(setup.cloud.points, theta)
            )
        )
    )
    grid_error = float(
        jnp.max(
            jnp.abs(
                accepted.field(hybrid.GRID, hybrid.FIELD)
                - hybrid.exact(jnp.asarray(setup.cell_centers), theta)
            )
        )
    )
    monolithic = recorder.run(
        "solve",
        lambda: hybrid.meshfree_reference(cells, seed=seed),
        scope=f"{scope}-monolithic",
    )
    report = accepted.interface(hybrid.LAW)
    defects = {
        name: float(value)
        for name, value in zip(report.names, report.values, strict=True)
    }
    return {
        "cells": cells,
        "spacing": 1.0 / cells,
        "unknowns": int(setup.cloud.points.shape[0] + setup.cell_centers.size // 2),
        "cloud_max_error": cloud_error,
        "grid_max_error": grid_error,
        "monolithic_max_error": float(monolithic),
        "overlap_mismatch_max": defects["overlap-mismatch-max"],
        "component_residual_max": max(
            float(item.residual_norm / item.scale) for item in accepted.components
        ),
        "converged": {
            name: bool(item.native_successful) for name, item in solutions.items()
        },
        "accepted": {name: bool(item.accepted) for name, item in solutions.items()},
        "iterations": {
            name: None if item.linear is None else int(item.linear.diagnostics.iterations)
            for name, item in solutions.items()
        },
        "transfer_refused_rows": int(
            sum(item.refused_rows for item in setup.law.transfers.evidence)
        ),
        "retained": (setup.cloud, accepted),
    }


def measure_hybrid_overlap_schwarz(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Meshfree/FV overlap solve against the analytic solution under refinement."""
    _require_float64(config, "hybrid-overlap-schwarz")
    levels = [
        cells for cells in range(16, 1024, 8) if _hybrid_unknowns(cells) <= capacity
    ]
    if len(levels) < 4:
        raise ValueError(
            "The hybrid refinement study needs four levels: capacity >= 2220."
        )
    levels = levels[:4]
    reserved = declare_reservation(
        8 * _hybrid_unknowns(levels[-1]) * (24 * (10 + 2 + 12) + 2 * 100 + 4 * 24),
        config,
        scope="hybrid cloud/grid owners, Krylov basis and local factors",
    )
    recorder = PhaseRecorder()
    recorder.unavailable(
        "search", "prepare_setup fuses the cloud search with its local fits"
    )
    records = [_hybrid_level(recorder, cells, seed) for cells in levels]
    retained = records[-1].pop("retained")
    for record in records[:-1]:
        record.pop("retained")
    converged = [record["converged"] for record in records]
    metrics: dict[str, Any] = {
        "additive_schwarz_converged_all_levels": all(
            item["additive-schwarz"] for item in converged
        ),
        "multiplicative_schwarz_converged_all_levels": all(
            item["multiplicative-schwarz"] for item in converged
        ),
        "multiplicative_schwarz_accepted_all_levels": all(
            record["accepted"]["multiplicative-schwarz"] for record in records
        ),
        "unpreconditioned_converged_levels": sum(
            int(item["unpreconditioned"]) for item in converged
        ),
        "component_residual_max": max(
            record["component_residual_max"] for record in records
        ),
        "hybrid_over_monolithic_error_max": max(
            record["cloud_max_error"] / record["monolithic_max_error"]
            for record in records
        ),
        "finest_overlap_mismatch_max": records[-1]["overlap_mismatch_max"],
        "transfer_refused_rows": sum(
            record["transfer_refused_rows"] for record in records
        ),
        "finest_cloud_max_error": records[-1]["cloud_max_error"],
        "finest_grid_max_error": records[-1]["grid_max_error"],
    }
    return _record(
        "hybrid-overlap-schwarz",
        records[-1]["unknowns"],
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=retained,
        oracle="analytic u(x, y; 0.2, 2.0, 0.5) of -div(K grad u) = f (exact automatic "
        "derivatives) plus a monolithic point-cloud solve on the whole square",
        consumer=_HYBRID_CONSUMER,
        reserved=reserved,
        series={
            "levels": records,
            "spacing_provenance": "nominal lattice spacing 1/cells of both owners "
            "(jittered interior cloud nodes, uniform FV cells)",
        },
    )


def _case_batch_slice(batch: CaseBatch, rows: slice, /) -> CaseBatch:
    from examples.meshfree_hybrid_calibrated import CaseBatch

    return CaseBatch(
        batch.predictions[rows],
        batch.targets[rows],
        batch.accepted[rows],
        batch.iterations[rows],
    )


def _shift_coverage(
    calibrator: uq.ProcessConformalCalibrator, batch: CaseBatch, scale: np.ndarray, /
) -> float:
    center = jnp.asarray(batch.predictions)
    interval = calibrator.interval(
        center, jnp.broadcast_to(jnp.asarray(scale), center.shape)
    )
    pointwise = uq.interval_coverage(
        interval.lower.data,
        interval.upper.data,
        jnp.asarray(batch.targets),
        reduction="none",
    )
    return float(jnp.mean(jnp.all(pointwise > 0.5, axis=1).astype(jnp.float64)))


def measure_calibrated_hybrid_held_out(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Split-conformal bands of the coarse hybrid solve on disjoint complete cases."""
    import examples.meshfree_hybrid_calibrated as hybrid

    _require_float64(config, "calibrated-hybrid-held-out")
    cells = 16
    unknowns = _hybrid_unknowns(cells)
    if unknowns > capacity:
        raise ValueError(f"The calibration level needs capacity >= {unknowns}.")
    train_cases, calibration_cases, test_cases, shift_cases, alpha = 6, 19, 20, 10, 0.1
    cases = train_cases + calibration_cases + test_cases
    reserved = declare_reservation(
        8 * unknowns * (24 * (10 + 2 + 12) + (cases + 2 * shift_cases) * 100),
        config,
        scope="multi-right-hand-side hybrid solves",
    )
    recorder = PhaseRecorder()
    rng = np.random.default_rng(seed + 1)
    in_family = hybrid.draw_parameters(rng, cases, hybrid.IN_FAMILY_DELTA)
    regime = hybrid.draw_parameters(rng, shift_cases, hybrid.SHIFTED_DELTA)
    geometry_cases = hybrid.draw_parameters(rng, shift_cases, hybrid.IN_FAMILY_DELTA)
    base = recorder.run(
        "local-fit",
        lambda: hybrid.prepare_setup(cells, hybrid.BASE_GEOMETRY, seed=seed),
        scope="base-geometry",
    )
    shifted = recorder.run(
        "local-fit",
        lambda: hybrid.prepare_setup(cells, hybrid.SHIFTED_GEOMETRY, seed=seed),
        scope="shifted-geometry",
    )
    family = recorder.run(
        "solve",
        lambda: hybrid.solve_cases(base, np.concatenate((in_family, regime), axis=0)),
        scope="family-and-regime-cases",
    )
    geometry = recorder.run(
        "solve",
        lambda: hybrid.solve_cases(shifted, geometry_cases),
        scope="geometry-cases",
    )
    ids = [f"case-{index:03d}" for index in range(cases)]
    train = slice(0, train_cases)
    calibration = slice(train_cases, train_cases + calibration_cases)
    test = slice(train_cases + calibration_cases, cases)
    split = uq.ProcessValidationSplit(ids[train], ids[calibration], ids[test])
    predictions, targets = family.predictions, family.targets
    # Train cases fit only the per-probe score scale; they never enter the radius.
    scale = np.sqrt(np.mean((predictions[train] - targets[train]) ** 2, axis=0))

    def calibrate() -> tuple[uq.ProcessConformalCalibrator, Any]:
        calibrator = uq.ProcessConformalCalibrator.calibrate_observable(
            jnp.asarray(predictions[calibration]),
            jnp.asarray(targets[calibration]),
            split,
            observable_name="probe-values",
            alpha=alpha,
            scale=jnp.broadcast_to(jnp.asarray(scale), (calibration_cases, scale.size)),
        )
        center = jnp.asarray(predictions[test])
        diagnostics = uq.process_conformal_diagnostics(
            calibrator,
            center,
            jnp.asarray(targets[test]),
            scale=jnp.broadcast_to(jnp.asarray(scale), center.shape),
        )
        return calibrator, diagnostics

    calibrator, diagnostics = recorder.run("output", calibrate, scope="split-conformal")
    expected, deviation = hybrid.finite_sample_band(calibration_cases, test_cases, alpha)
    measured = float(diagnostics.empirical_coverage)
    disjoint = not (
        set(split.train_case_ids) & set(split.calibration_case_ids)
        or set(split.train_case_ids) & set(split.test_case_ids)
        or set(split.calibration_case_ids) & set(split.test_case_ids)
    )
    metrics: dict[str, Any] = {
        "split_disjoint": disjoint,
        "all_family_cases_accepted": bool(np.all(family.accepted[:cases])),
        "nominal_coverage": float(diagnostics.nominal_coverage),
        "held_out_coverage": measured,
        "finite_sample_expected_coverage": expected,
        "finite_sample_coverage_deviation": deviation,
        "held_out_coverage_z": (measured - expected) / deviation,
        "radius": float(calibrator.calibrator.radius),
        "regime_shift_coverage": _shift_coverage(
            calibrator, _case_batch_slice(family, slice(cases, None)), scale
        ),
        "geometry_shift_coverage": _shift_coverage(calibrator, geometry, scale),
        "shifted_cases_accepted": bool(np.all(family.accepted[cases:]))
        and bool(np.all(geometry.accepted)),
        "distribution_free_guarantee_claimed_under_shift": False,
    }
    return _record(
        "calibrated-hybrid-held-out",
        unknowns,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(base.cloud, family.predictions),
        oracle="analytic manufactured family targets at fixed probes; exchangeable i.i.d. "
        "complete cases; Beta-binomial finite-sample coverage band",
        consumer="phydrax.uq.{ProcessValidationSplit, ProcessConformalCalibrator, "
        "process_conformal_diagnostics} over the hybrid overlap solve",
        reserved=reserved,
        series={
            "exchangeability": "complete cases are i.i.d. draws of the manufactured family "
            "on one fixed coarse hybrid discretization",
            "multi_rhs_iterations": [int(value) for value in family.iterations],
            "split": {
                "train": len(split.train_case_ids),
                "calibration": len(split.calibration_case_ids),
                "test": len(split.test_case_ids),
            },
        },
    )


# ---------------------------------------------------------------------------
# Q17 durable production runs: interrupt, restart, changed ownership, refusals


_RESTART_SOURCE = "meshfree-closure-q17-build"
_RESTART_RNG = "threefry:fold-in(accepted-step)"
_RESTART_AXIS = 8
_RESTART_RETRY = solver.RobustRetryPolicy(maximum_retries=2, reduction_factor=0.5)
_PERIODIC = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=(True, True))
_RESTART_CONSUMER = (
    "phydrax.solver.{ProductionRunPlan, PreparedProductionRun, ArtifactCheckpointStore, "
    "DurableCheckpointStore} with phydrax.discretization.meshfree.{meshfree_runtime_inventory, "
    "support_epoch_relation}"
)
StaleRole: TypeAlias = Literal["geometry", "capacity", "precision", "source", "program"]


def _restart_axis(capacity: int, /) -> int:
    axis = min(isqrt(capacity), _RESTART_AXIS)
    if axis < 6:
        raise ValueError("The restart cloud needs capacity >= 36 points.")
    return axis


def _restart_reservation(points: int, config: MeshfreeConfig, /) -> int:
    # Stencil relations, three checkpoint generations of state/evidence and the
    # staged archive buffer (16 MiB declared by the resource request below).
    return declare_reservation(
        8 * points * 13 * (6 + 2 + 12) + 3 * 8 * points * 64 + (16 << 20),
        config,
        scope="production run, checkpoint generations and staging",
    )


def _restart_cloud(axis: int, seed: int, /) -> PreparedPointCloudDiscretization:
    spacing = 1.0 / axis
    grid = (np.arange(axis, dtype=np.float64) + 0.5) * spacing
    x, y = np.meshgrid(grid, grid, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    jitter = np.random.default_rng(seed + 3).uniform(-1.0, 1.0, points.shape)
    return PointCloudPlan(
        np.mod(points + 0.005 * spacing * jitter, 1.0),
        np.full(points.shape[0], spacing**2, dtype=np.float64),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=13,
        address=_PERIODIC,
    ).prepare()


def _drift(time: Array, points: Array, args: Any) -> Array:
    del time, args
    return jnp.broadcast_to(jnp.asarray([1.0, 0.0], dtype=points.dtype), points.shape)


def _ale_evolution(
    cloud: PreparedPointCloudDiscretization, /
) -> PreparedMeshfreeEvolution:
    """ALE drift: the fixed support exhausts its trust after a few steps."""
    return MeshfreeEvolutionPlan(
        cloud,
        diffusion=MeshfreeDiffusionLaw(0.05, law_id="diffusivity:0.05"),
        motion=MeshfreeMotion("ale", mesh_velocity=_drift, law_id="uniform-drift"),
        plan_id="meshfree-closure-q17-ale",
    ).prepare()


def _ale_step(cloud: PreparedPointCloudDiscretization, /) -> float:
    return float(jnp.min(cloud.trust_radius)) / 5.5


@dataclass(frozen=True)
class _Leg:
    evolution: PreparedMeshfreeEvolution
    plan: solver.ProductionRunPlan
    inventory: solver.RuntimeIdentityInventory
    manifest: solver.ProductionCaseManifest


def _leg(
    evolution: PreparedMeshfreeEvolution,
    step: float,
    /,
    *,
    end_steps: int = 10,
    source: str = _RESTART_SOURCE,
    precision: str = "float64",
) -> _Leg:
    method = evolution.ssprk_method("ssprk33")
    end = end_steps * step
    plan = solver.ProductionRunPlan(
        method,
        _RESTART_RETRY,
        step_size=step,
        end_time=end,
        maximum_steps=4 * end_steps,
        checkpoint_interval=2,
        segment_steps=2,
        output_schedule=solver.ExactTimeSchedule(
            jnp.arange(1, end_steps // 4 + 1, dtype=jnp.float64) * 4 * step
        ),
        moments=(
            solver.StreamingMomentPlan(
                lambda time, state, args: jnp.sum(evolution.fields(state).content),
                value_shape=(),
                plan_id="total-content",
            ),
        ),
    )
    inventory = meshfree_runtime_inventory(
        evolution,
        source=source,
        program=plan.plan_id,
        method=method,
        controller=_RESTART_RETRY.policy_id,
        precision=precision,
        rng=_RESTART_RNG,
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="meshfree-ale-drift", dtype="float64"
    )
    return _Leg(evolution, plan, inventory, manifest)


@dataclass(frozen=True)
class _Repository:
    repository: POSIXArtifactRepository
    resolved: ResolvedRunSpec
    request: execution.ResourceRequest
    policy: solver.CheckpointGenerationPolicy


def _repository(root: Path, label: str, /) -> _Repository:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    repository = POSIXArtifactRepository(
        root.resolve(),
        POSIXRepositoryPolicy(
            HPCFilesystemProfile(
                f"{label}-posix",
                "benchmark-filesystem",
                atomic_rename_same_filesystem=True,
                file_fsync=True,
                directory_fsync=True,
                advisory_locking=True,
                attempt_private_staging=True,
            ),
            maximum_chunk_bytes=1 << 16,
            maximum_metadata_bytes=1 << 20,
        ),
    )
    request = execution.ResourceRequest(
        cpu_cores=1,
        memory_bytes=64 << 20,
        maximum_checkpoint_staging_bytes=16 << 20,
        maximum_output_backlog_bytes=8 << 20,
    )
    policy = solver.CheckpointGenerationPolicy(3)
    dependency = SupportDependency(
        f"{label}-repository", repository.support_tuple.support_tuple_id
    )
    resolved = ResolvedRunSpec(
        (),
        (dependency,),
        release_index_id="release-index",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust-policy",
        valid_at=10,
        valid_from=0,
        valid_until=20,
        prepared_configuration_id=f"{label}-configuration",
        precision_policy_id="float64",
        resource_policy_id=request.resource_id,
        checkpoint_policy_id=policy.policy_id,
        output_policy_id="ordered-outbox",
        repository_id=repository.provider_id,
        scheduler_id="scheduler",
        auth_policy_id="auth-policy",
    )
    return _Repository(repository, resolved, request, policy)


def _publisher(events: list[str], /) -> solver.ByteBoundedAsyncPublisher:
    def archive(event_id: str, snapshot: Any) -> None:
        del snapshot
        events.append(event_id)

    return solver.ByteBoundedAsyncPublisher(
        archive, maximum_pending=2, maximum_pending_bytes=1 << 20
    )


def _artifact_runtime(
    repository: _Repository,
    leg: _Leg,
    publisher: solver.ByteBoundedAsyncPublisher,
    /,
    *,
    relation: solver.RuntimeRestartRelation | None = None,
) -> solver.PreparedProductionRun:
    """Fresh store and runtime objects, as a restarted process builds them."""
    store = solver.ArtifactCheckpointStore(
        repository.repository,
        leg.manifest,
        repository.policy,
        repository.resolved,
        writer_id="meshfree-closure-q17",
        resource_request=repository.request,
        artifact_id="meshfree-ale-drift",
    )
    return solver.PreparedProductionRun(
        leg.manifest,
        leg.plan,
        store,
        publisher=publisher,
        resolved_run_spec=repository.resolved,
        restart_relation=relation,
    )


def _initial(runtime: solver.PreparedProductionRun, state: Any, seed: int, /) -> Any:
    return runtime.initial_state(
        state,
        controller_state=jnp.asarray(0, dtype=jnp.int64),
        rng_state=jax.random.key_data(jax.random.key(11 + seed)),
    )


def _interrupt(
    recorder: PhaseRecorder,
    runtime: solver.PreparedProductionRun,
    state: Any,
    steps: int,
    scope: str,
    /,
) -> Any:
    """Advance accepted steps, commit, and return the committed state; the caller
    abandons every live object."""

    def advance() -> Any:
        current = state
        for _ in range(steps):
            current, transition = runtime.step(current)
            if not bool(transition.successful):
                raise RuntimeError(
                    "An interrupted leg must stop after accepted steps only."
                )
        return current

    advanced = recorder.run("solve", advance, scope=scope)
    recorder.run("epoch-commit", lambda: runtime.checkpoint(advanced), scope=scope)
    return advanced


@dataclass(frozen=True)
class _EpochCampaign:
    held: Any
    held_failure: solver.ProductionFailureRecord
    final: solver.ProductionRunResult
    replay_classification: str | None
    output_events: int


def _epoch_campaign(
    recorder: PhaseRecorder, root: Path, axis: int, seed: int, interrupt: bool, /
) -> _EpochCampaign:
    """Support-exhaustion refusal ends epoch 0; a rebased epoch completes the run."""
    label = "interrupted" if interrupt else "uninterrupted"
    cloud = recorder.run("local-fit", lambda: _restart_cloud(axis, seed), scope=label)
    step = _ale_step(cloud)
    repository = _repository(root, f"q17-{label}")
    first = recorder.run(
        "assembly", lambda: _leg(_ale_evolution(cloud), step), scope=label
    )
    events: list[str] = []
    publisher = _publisher(events)
    runtime = _artifact_runtime(repository, first, publisher)
    start = _initial(
        runtime,
        first.evolution.initial_state(
            1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * cloud.points[:, 0])
        ),
        seed,
    )
    if interrupt:
        _interrupt(recorder, runtime, start, 3, f"{label}-epoch-0")
        runtime = _artifact_runtime(repository, first, publisher)
        template = start
        start = recorder.run(
            "restart", lambda: runtime.resume(template), scope=f"{label}-epoch-0"
        )
    epoch_runtime = runtime
    epoch_start = start
    held = recorder.run(
        "solve", lambda: epoch_runtime.run(epoch_start), scope=f"{label}-epoch-0"
    )
    if held.failure is None:
        raise RuntimeError(
            "The ALE drift must exhaust its fixed-support trust in epoch 0."
        )
    successor = recorder.run(
        "transfer",
        lambda: first.evolution.rebase(held.state.accepted_state),
        scope=f"{label}-rebase",
    )
    second = _leg(successor, step)
    relation = recorder.run(
        "transfer",
        lambda: support_epoch_relation(
            first.inventory,
            second.inventory,
            first.evolution,
            successor,
            held.state.accepted_state,
            source_template=held.state.accepted_state,
        ),
        scope=f"{label}-rebase",
    )
    runtime = _artifact_runtime(repository, second, publisher, relation=relation)
    template = _initial(
        runtime, first.evolution.repacked(held.state.accepted_state), seed
    )
    relation_runtime = runtime
    resumed = recorder.run(
        "restart",
        lambda: relation_runtime.resume(template),
        scope=f"{label}-epoch-relation",
    )
    classification = runtime.last_replay_classification
    if interrupt:
        _interrupt(recorder, runtime, resumed, 2, f"{label}-epoch-1")
        # Same-epoch restart binds the identity relation of the second epoch.
        runtime = _artifact_runtime(repository, second, publisher)
        same_runtime = runtime
        resumed = recorder.run(
            "restart", lambda: same_runtime.resume(template), scope=f"{label}-epoch-1"
        )
    final_runtime = runtime
    final_start = resumed
    final = recorder.run(
        "solve", lambda: final_runtime.run(final_start), scope=f"{label}-epoch-1"
    )
    recorder.run("output", publisher.close, scope=label)
    return _EpochCampaign(held.state, held.failure, final, classification, len(events))


def _bitwise(first: Any, second: Any, /) -> bool:
    """Bit-pattern equality: the bitwise replay class compares stored bytes.

    Fail-closed evidence legitimately retains nonfinite values; NaN payloads are
    equal when their bits are, which value comparison (NaN != NaN) would deny.
    """
    left, right = np.asarray(first), np.asarray(second)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _arrays_equal(first: Any, second: Any, /) -> bool:
    return _bitwise(first, second)


def _trees_equal(first: Any, second: Any, /) -> bool:
    """Same structure (static fields included) and bitwise-equal array leaves."""
    left, left_tree = jax.tree.flatten(first)
    right, right_tree = jax.tree.flatten(second)
    return left_tree == right_tree and all(
        _bitwise(a, b) for a, b in zip(left, right, strict=True)
    )


def _unequal_paths(first: Any, second: Any, /) -> list[str]:
    """Leaf paths whose bits differ (diagnostic evidence for a failed replay)."""
    left = jax.tree_util.tree_flatten_with_path(first)[0]
    right = jax.tree_util.tree_flatten_with_path(second)[0]
    if len(left) != len(right):
        return ["<structure>"]
    return [
        jax.tree_util.keystr(path)
        for (path, a), (_, b) in zip(left, right, strict=True)
        if not _bitwise(a, b)
    ]


def measure_restart_refused_epoch_interrupt(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Uninterrupted versus interrupted+resumed run across a refused support epoch."""
    _require_float64(config, "restart-refused-epoch-interrupt")
    axis = _restart_axis(capacity)
    reserved = _restart_reservation(axis * axis, config)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    with tempfile.TemporaryDirectory(prefix="q17-epoch-") as directory:
        root = Path(directory)
        reference = _epoch_campaign(recorder, root / "uninterrupted", axis, seed, False)
        restarted = _epoch_campaign(recorder, root / "interrupted", axis, seed, True)
    held, failure = reference.held, reference.held_failure
    held_evidence = held.evidence
    final_reference, final_restarted = reference.final.state, restarted.final.state
    final_evidence = final_reference.evidence
    metrics: dict[str, Any] = {
        "refused_epoch_category_step_rejected": failure.category == "step-rejected",
        "refused_attempts_retained": held_evidence is not None
        and int(held_evidence.refused_attempts) >= _RESTART_RETRY.maximum_retries + 1
        and int(held_evidence.refused_step) == int(held.step_index),
        "refusal_evidence_survives_epoch": final_evidence is not None
        and held_evidence is not None
        and int(final_evidence.refused_step) == int(held.step_index)
        and int(final_evidence.refused_attempts) == int(held_evidence.refused_attempts),
        "held_state_equal": _arrays_equal(
            restarted.held.accepted_state, held.accepted_state
        ),
        "held_evidence_equal": _trees_equal(restarted.held.evidence, held_evidence),
        "both_completed": final_reference.status == final_restarted.status == "completed",
        "step_index_equal": int(final_reference.step_index)
        == int(final_restarted.step_index),
        "time_equal": _arrays_equal(final_reference.time, final_restarted.time),
        "accepted_state_equal": _arrays_equal(
            final_reference.accepted_state, final_restarted.accepted_state
        ),
        "moment_states_equal": _trees_equal(
            final_reference.moment_states, final_restarted.moment_states
        ),
        "evidence_equal": _trees_equal(
            final_reference.evidence, final_restarted.evidence
        ),
        "trigger_states_equal": _trees_equal(
            final_reference.trigger_states, final_restarted.trigger_states
        ),
        "schedule_cursor_equal": int(final_reference.schedule_cursor)
        == int(final_restarted.schedule_cursor),
        "output_cursor_equal": int(final_reference.output_cursor)
        == int(final_restarted.output_cursor),
        "rng_state_equal": _arrays_equal(
            final_reference.rng_state, final_restarted.rng_state
        ),
        "controller_state_equal": _arrays_equal(
            final_reference.controller_state, final_restarted.controller_state
        ),
        "replay_bitwise_uninterrupted": reference.replay_classification == "bitwise",
        "replay_bitwise_interrupted": restarted.replay_classification == "bitwise",
        "final_step_index": int(final_reference.step_index),
        "output_cursor": int(final_reference.output_cursor),
    }
    metrics["replay_class_bitwise_match"] = all(
        metrics[key]
        for key in (
            "held_state_equal",
            "held_evidence_equal",
            "step_index_equal",
            "time_equal",
            "accepted_state_equal",
            "moment_states_equal",
            "evidence_equal",
            "trigger_states_equal",
            "schedule_cursor_equal",
            "output_cursor_equal",
            "rng_state_equal",
            "controller_state_equal",
        )
    )
    return _record(
        "restart-refused-epoch-interrupt",
        axis * axis,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(final_reference, final_restarted),
        oracle="the uninterrupted production run of the identical plan (bitwise replay class)",
        consumer=_RESTART_CONSUMER,
        reserved=reserved,
        series={
            "replay_classification": [
                reference.replay_classification,
                restarted.replay_classification,
            ],
            "output_events": [reference.output_events, restarted.output_events],
            "held_evidence_unequal_leaves": _unequal_paths(
                restarted.held.evidence, held_evidence
            ),
            "final_evidence_unequal_leaves": _unequal_paths(
                final_restarted.evidence, final_reference.evidence
            ),
        },
    )


def _children_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


_CLI_FIELDS = (
    "status",
    "step",
    "time",
    "state_sha256",
    "moments_sha256",
    "evidence_sha256",
    "accepted_step",
    "refused_attempts",
    "output_cursor",
)


def _cli(root: Path, log: Path, *extra: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "examples.meshfree_production_restart",
            "--root",
            str(root),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "JAX_ENABLE_X64": "1"},
        stdout=subprocess.PIPE,
        stderr=log.open("w", encoding="utf-8"),
        text=True,
    )


def _cli_records(lines: list[str], /) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines if line.startswith("{")]


def _cli_complete(root: Path, log: Path, *extra: str) -> list[dict[str, Any]]:
    process = _cli(root, log, *extra)
    output, _ = process.communicate(timeout=3600)
    if process.returncode != 0:
        raise RuntimeError(
            f"meshfree_production_restart {' '.join(extra)} exited {process.returncode}: "
            + log.read_text(encoding="utf-8")[-4000:]
        )
    return _cli_records(output.splitlines())


def measure_restart_cli_sigkill(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """SIGKILL the production example mid-run; --resume must reach the reference result."""
    _require_float64(config, "restart-cli-sigkill")
    points = 100  # examples.meshfree_production_restart: a fixed 10 x 10 periodic cloud.
    if capacity < points:
        raise ValueError("The production-restart example runs a fixed 100-point cloud.")
    reserved = _restart_reservation(points, config)
    recorder = PhaseRecorder()
    for phase in ("search", "local-fit", "assembly", "epoch-commit"):
        recorder.unavailable(phase, "fused inside the separate example process")
    with tempfile.TemporaryDirectory(prefix="q17-cli-") as directory:
        root = Path(directory)
        children = _children_cpu()
        reference = recorder.run(
            "end-to-end",
            lambda: _cli_complete(
                root / "reference", root / "reference.log", "--archive-seconds", "0.0"
            ),
            scope="reference-process",
        )
        recorder.annotate("end-to-end", child_cpu_seconds=_children_cpu() - children)

        def killed_run() -> tuple[list[str], int]:
            process = _cli(root / "killed", root / "killed.log")
            stdout = process.stdout
            if stdout is None:
                raise RuntimeError("The CLI subprocess has no stdout pipe.")
            observed: list[str] = []
            while sum('"output"' in line for line in observed) < 3:
                line = stdout.readline()
                if not line:
                    raise RuntimeError(
                        "The CLI ended before three outputs: "
                        + (root / "killed.log").read_text(encoding="utf-8")[-4000:]
                    )
                observed.append(line)
            process.send_signal(signal.SIGKILL)
            process.wait(timeout=60)
            return observed, process.returncode

        children = _children_cpu()
        observed, returncode = recorder.run("solve", killed_run, scope="killed-process")
        recorder.annotate("solve", child_cpu_seconds=_children_cpu() - children)
        children = _children_cpu()
        resumed = recorder.run(
            "restart",
            lambda: _cli_complete(
                root / "killed",
                root / "resumed.log",
                "--resume",
                "--archive-seconds",
                "0.0",
            ),
            scope="resumed-process",
        )
        recorder.annotate("restart", child_cpu_seconds=_children_cpu() - children)
    expected, final, start = reference[-1], resumed[-1], resumed[0]
    metrics: dict[str, Any] = {
        "reference_completed": expected.get("status") == "completed",
        "killed_by_sigkill": returncode == -signal.SIGKILL,
        "killed_before_completion": not any('"status"' in line for line in observed),
        "resumed_from_committed_step": start.get("start") == "resume"
        and int(start.get("step", 0)) > 0,
        "resume_step": int(start.get("step", 0)),
        "sampled_peak_resident_reported": final.get("sampled_peak_resident_bytes")
        is not None,
        **{
            f"{name}_equal": final.get(name) == expected.get(name) for name in _CLI_FIELDS
        },
    }
    metrics["final_record_equal"] = all(metrics[f"{name}_equal"] for name in _CLI_FIELDS)
    return _record(
        "restart-cli-sigkill",
        points,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(),
        oracle="the uninterrupted run of the same CLI (state/moment/evidence SHA-256)",
        consumer="python -m examples.meshfree_production_restart --root DIR [--resume]",
        reserved=reserved,
        series={"reference_final": expected, "resumed_final": final},
    )


_MIGRATION_KAPPA = 0.01
_MIGRATION_STEP = 0.01
_MIGRATION_STEPS = 20
_MIGRATION_DEVICES = 4
_MIGRATION_POINTS = 48


def _migration_operator(points: np.ndarray, /) -> SparseCoordinateOperator:
    cloud = jnp.asarray(points)
    neighborhood = MeshfreeNeighborhoodPlan(cloud, 12).prepare()
    stencils = prepare_local_stencils(
        neighborhood,
        cloud,
        cloud,
        (MeshfreeFunctional(((2, 0), (0, 2)), (1.0, 1.0), name="laplacian"),),
        LocalStencilPolicy(polynomial_degree=2),
    )
    return MeshfreeOperator(stencils).operator


def _migration_method(
    operator: DistributedMeshfreeOperator, /
) -> solver.CallableFixedStepMethod:
    """Explicit diffusion on owner-blocked values; one identity on every partition."""

    def step(
        step_index: Array, time: Array, state: Array, step_size: Array, args: Any
    ) -> solver.FixedStepResult:
        del step_index, time, args
        candidate = state + step_size * _MIGRATION_KAPPA * operator.apply(state)
        finite = jnp.all(jnp.isfinite(candidate))
        return solver.FixedStepResult(
            candidate,
            jnp.where(finite, candidate, state),
            finite,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
        )

    return solver.CallableFixedStepMethod(
        step, f"distributed-explicit-diffusion:{operator.operator_id}:{_MIGRATION_KAPPA}"
    )


def _migration_runtime(
    repository: _Repository,
    operator: DistributedMeshfreeOperator,
    /,
    *,
    relation: solver.RuntimeRestartRelation | None = None,
) -> tuple[solver.PreparedProductionRun, solver.RuntimeIdentityInventory]:
    method = _migration_method(operator)
    retry = solver.RobustRetryPolicy(maximum_retries=0)
    plan = solver.ProductionRunPlan(
        method,
        retry,
        step_size=_MIGRATION_STEP,
        end_time=_MIGRATION_STEPS * _MIGRATION_STEP,
        maximum_steps=_MIGRATION_STEPS + 2,
        checkpoint_interval=4,
        segment_steps=4,
    )
    inventory = meshfree_runtime_inventory(
        operator,
        source="meshfree-closure-q17-ownership-migration",
        program=plan.plan_id,
        method=method,
        controller=retry.policy_id,
        precision="float64",
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="distributed-diffusion", dtype="float64"
    )
    store = solver.ArtifactCheckpointStore(
        repository.repository,
        manifest,
        repository.policy,
        repository.resolved,
        writer_id="meshfree-closure-q17-migration",
        resource_request=repository.request,
        artifact_id="distributed-diffusion",
    )
    runtime = solver.PreparedProductionRun(
        manifest,
        plan,
        store,
        resolved_run_spec=repository.resolved,
        restart_relation=relation,
    )
    return runtime, inventory


def measure_restart_ownership_migration_worker(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Restart into a changed owner partition through an explicit migration receipt.

    Runs only inside a process with exactly four (forced host) devices; the
    forced CPU devices prove the restart relation and partition parity only,
    never accelerator performance.
    """
    _require_float64(config, "restart-ownership-migration-worker")
    if len(jax.devices()) != _MIGRATION_DEVICES:
        raise ValueError(
            "restart-ownership-migration-worker requires exactly four devices; launch it "
            "through restart-ownership-migration."
        )
    requested, capacity = capacity, min(capacity, _MIGRATION_POINTS)
    if capacity < 4 * _MIGRATION_DEVICES:
        raise ValueError("The migration cloud needs capacity >= 16 points.")
    owner_capacity = capacity // 2
    reserved = declare_reservation(
        8 * capacity * 12 * (6 + 2 + 12)
        + 3 * 8 * _MIGRATION_DEVICES * owner_capacity * 64
        + (16 << 20),
        config,
        scope="distributed operator, halos, migration packets and checkpoints",
    )
    recorder = PhaseRecorder()
    group = execution.ExecutionRuntime.current().child_groups(1)[0]
    rng = np.random.default_rng(seed + 5)
    points = rng.uniform(0.0, 1.0, (capacity, 2))
    by_x = np.minimum((4 * points[:, 0]).astype(np.int32), 3)
    by_y = np.minimum((4 * points[:, 1]).astype(np.int32), 3)
    ownership = DistributedOwnershipPlan(
        MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10), group, owner_capacity
    )
    source = recorder.run(
        "communication-migration",
        lambda: DistributedPointLayout.from_global(ownership, points, by_x),
        scope="initial-distribution",
    )
    reference_operator = recorder.run(
        "local-fit", lambda: _migration_operator(points), scope="operator"
    )

    def bind(layout: DistributedPointLayout) -> DistributedMeshfreeOperator:
        return DistributedMeshfreeOperator.bind(
            reference_operator, layout, layout, halo_capacity=owner_capacity
        )

    values = np.sin(2.0 * np.pi * points[:, 0]) * np.cos(2.0 * np.pi * points[:, 1])
    initial = source.distribute(jnp.asarray(values))
    with tempfile.TemporaryDirectory(prefix="q17-migration-") as directory:
        root = Path(directory)
        source_operator = recorder.run("assembly", lambda: bind(source), scope="source")
        reference_runtime, _ = _migration_runtime(
            _repository(root / "reference", "q17-migration-reference"), source_operator
        )
        reference = recorder.run(
            "solve",
            lambda: reference_runtime.run(reference_runtime.initial_state(initial)),
            scope="reference",
        )
        logical_reference = np.asarray(source.collect(reference.state.accepted_state))
        repository = _repository(root / "migrated", "q17-migration")
        runtime, source_inventory = _migration_runtime(repository, source_operator)
        state = runtime.initial_state(initial)
        interrupted_state = _interrupt(recorder, runtime, state, 8, "interrupted")
        interrupted = np.asarray(source.collect(interrupted_state.accepted_state))
        logical = np.asarray(jax.device_get(source.logical_indices))
        active = np.asarray(jax.device_get(source.active))
        destinations = np.where(active, by_y[np.where(active, logical, 0)], 0).astype(
            np.int32
        )
        migrated = recorder.run(
            "communication-migration",
            lambda: source.migrate(
                jnp.asarray(destinations), packet_capacity=owner_capacity
            ),
            scope="ownership-migration",
        )
        committed = bool(np.asarray(jax.device_get(migrated.evidence.committed)))
        target = migrated.layout
        target_operator = recorder.run("assembly", lambda: bind(target), scope="target")
        unrelated, target_inventory = _migration_runtime(repository, target_operator)
        refused: list[str] = []
        try:
            unrelated.resume(
                unrelated.initial_state(target.distribute(jnp.zeros(capacity)))
            )
        except solver.StaleRuntimeCheckpointError as error:
            refused = list(error.roles)
        relation = recorder.run(
            "transfer",
            lambda: ownership_migration_relation(
                source_inventory,
                target_inventory,
                source,
                destinations,
                packet_capacity=owner_capacity,
                source_template=interrupted_state.accepted_state,
            ),
            scope="migration-relation",
        )
        restarted, _ = _migration_runtime(repository, target_operator, relation=relation)
        template = restarted.initial_state(target.distribute(jnp.zeros(capacity)))
        resumed = recorder.run(
            "restart", lambda: restarted.resume(template), scope="changed-partition"
        )
        restored = np.asarray(target.collect(resumed.accepted_state))
        final = recorder.run("solve", lambda: restarted.run(resumed), scope="continued")
        logical_final = np.asarray(target.collect(final.state.accepted_state))
    migration = relation.migration
    if migration is None:
        raise RuntimeError("An ownership relation carries its migration receipt.")
    migrated_roles = list(migration.migrated_roles)
    metrics: dict[str, Any] = {
        "devices": len(jax.devices()),
        "migration_committed": committed,
        "partition_changed": source.partition_fingerprint()
        != target.partition_fingerprint(),
        "partition_role_migrated": "partition" in migrated_roles,
        "migrated_roles_placement_only": set(migrated_roles)
        <= {"partition", "program", "capacity"},
        "replay_bitwise": restarted.last_replay_classification == "bitwise",
        "restored_bitwise": bool(np.array_equal(restored, interrupted)),
        "final_completed": final.state.status == "completed",
        "step_index_equal": int(final.state.step_index)
        == int(reference.state.step_index),
        "final_maximum_difference": float(
            np.max(np.abs(logical_final - logical_reference))
        ),
        "identity_restart_refused": bool(refused),
        "identity_restart_refused_partition": "partition" in refused,
    }
    return _record(
        "restart-ownership-migration-worker",
        capacity,
        requested,
        seed,
        2,
        recorder,
        metrics,
        retained=(final.state.accepted_state,),
        oracle="the uninterrupted run on the source partition (logical owner-collected state)",
        consumer="phydrax.discretization.meshfree.ownership_migration_relation + "
        "solver.RuntimeRestartRelation into a changed DistributedPointLayout",
        reserved=reserved,
        series={
            "migrated_roles": migrated_roles,
            "identity_restart_refused_roles": refused,
            "replay_classification": restarted.last_replay_classification,
        },
    )


def measure_restart_ownership_migration(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Launch the migration worker in a dedicated four-forced-device process."""
    from benchmarks.meshfree_closure_distributed import run_forced_device_workloads

    _require_float64(config, "restart-ownership-migration")
    with tempfile.TemporaryDirectory(prefix="q17-migration-launch-") as directory:
        record = run_forced_device_workloads(
            ("restart-ownership-migration-worker",),
            MeshfreeConfig(
                sizes=(capacity,),
                seeds=(seed,),
                dimension=config.dimension,
                repeats=config.repeats,
                neighbors=config.neighbors,
                degree=config.degree,
                chunk_rows=config.chunk_rows,
                working_set_bytes=config.working_set_bytes,
                resource_bytes=config.resource_bytes,
                max_points=config.max_points,
                steps=config.steps,
                precision=config.precision,
            ),
            devices=_MIGRATION_DEVICES,
            output=Path(directory) / "migration.json",
        )
    rows = record["rows"]
    if len(rows) != 1 or rows[0]["status"] != "measured":
        raise RuntimeError(f"The migration worker produced no measured row: {rows}")
    return {
        **rows[0],
        "workload": "restart-ownership-migration",
        "worker_workload": rows[0]["workload"],
        "subprocess": record["subprocess"],
    }


def _durable(
    root: Path, plan: solver.ProductionRunPlan, manifest: solver.ProductionCaseManifest, /
) -> solver.PreparedProductionRun:
    return solver.PreparedProductionRun(
        manifest,
        plan,
        solver.DurableCheckpointStore(
            root, manifest, solver.CheckpointGenerationPolicy(2)
        ),
        publisher=_publisher([]),
    )


def _stale_leg(change: StaleRole, axis: int, seed: int, /) -> _Leg:
    reference = _restart_cloud(axis, seed)
    match parse(change, StaleRole, "change"):
        case "geometry":
            return _leg(
                _ale_evolution(_restart_cloud(axis, seed + 1)), _ale_step(reference)
            )
        case "capacity":
            return _leg(
                _ale_evolution(_restart_cloud(axis - 1, seed)), _ale_step(reference)
            )
        case "precision":
            return _leg(
                _ale_evolution(reference),
                _ale_step(reference),
                precision="float64-mixed-policy",
            )
        case "source":
            return _leg(
                _ale_evolution(reference), _ale_step(reference), source="another-build"
            )
        case "program":
            return _leg(_ale_evolution(reference), _ale_step(reference), end_steps=12)


def _committed(
    recorder: PhaseRecorder, root: Path, axis: int, seed: int, /
) -> tuple[_Leg, Any]:
    cloud = recorder.run(
        "local-fit", lambda: _restart_cloud(axis, seed), scope="committed"
    )
    leg = recorder.run(
        "assembly",
        lambda: _leg(_ale_evolution(cloud), _ale_step(cloud)),
        scope="committed",
    )
    runtime = _durable(root, leg.plan, leg.manifest)
    state = runtime.initial_state(
        leg.evolution.initial_state(jnp.ones(cloud.points.shape[0]))
    )
    recorder.run("epoch-commit", lambda: runtime.checkpoint(state), scope="committed")
    return leg, state


def _store_listing(root: Path, /) -> list[tuple[str, int]]:
    return sorted((path.name, path.stat().st_mtime_ns) for path in root.iterdir())


def _stale_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, change: StaleRole, /
) -> dict[str, Any]:
    _require_float64(config, f"restart-stale-{change}")
    axis = _restart_axis(capacity)
    reserved = _restart_reservation(axis * axis, config)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    with tempfile.TemporaryDirectory(prefix=f"q17-stale-{change}-") as directory:
        root = Path(directory) / "checkpoints"
        _committed(recorder, root, axis, seed)
        before = _store_listing(root)
        leg = recorder.run(
            "assembly", lambda: _stale_leg(change, axis, seed), scope=change
        )
        runtime = _durable(root, leg.plan, leg.manifest)
        template = runtime.initial_state(
            leg.evolution.initial_state(jnp.ones(leg.evolution.point_count))
        )
        roles: tuple[str, ...] = ()

        def attempt() -> tuple[str, ...]:
            try:
                runtime.resume(template)
            except solver.StaleRuntimeCheckpointError as error:
                return tuple(error.roles)
            return ()

        roles = recorder.run("restart", attempt, scope=f"stale-{change}")
        unchanged = _store_listing(root) == before
    metrics: dict[str, Any] = {
        "refused": bool(roles),
        "changed_role_named": change in roles,
        "store_unchanged": unchanged,
    }
    return _record(
        f"restart-stale-{change}",
        axis * axis,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(),
        oracle="checkpoint identity inventory of the committed run",
        consumer="phydrax.solver.PreparedProductionRun.resume -> StaleRuntimeCheckpointError",
        reserved=reserved,
        series={"refused_roles": list(roles)},
    )


def measure_restart_stale_geometry(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _stale_refusal(capacity, seed, config, "geometry")


def measure_restart_stale_capacity(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _stale_refusal(capacity, seed, config, "capacity")


def measure_restart_stale_precision(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _stale_refusal(capacity, seed, config, "precision")


def measure_restart_stale_source(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _stale_refusal(capacity, seed, config, "source")


def measure_restart_stale_program(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return _stale_refusal(capacity, seed, config, "program")


def measure_restart_truncated_archive(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A checkpoint archive cut in half refuses before any value is used."""
    _require_float64(config, "restart-truncated-archive")
    axis = _restart_axis(capacity)
    reserved = _restart_reservation(axis * axis, config)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    with tempfile.TemporaryDirectory(prefix="q17-truncated-") as directory:
        root = Path(directory) / "checkpoints"
        leg, state = _committed(recorder, root, axis, seed)
        truncated = Path(directory) / "truncated"
        truncated.mkdir(mode=0o700)
        for path in root.iterdir():
            (truncated / path.name).write_bytes(path.read_bytes())
        archive = next(truncated.glob("generation-*.phx"))
        archive.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
        runtime = _durable(truncated, leg.plan, leg.manifest)

        def attempt() -> str:
            # The archive owner's documented typed refusal; anything else re-raises.
            try:
                runtime.resume(state)
            except ArrayArchiveCorruptionError as error:
                return type(error).__name__
            return ""

        refusal = recorder.run("restart", attempt, scope="truncated-archive")
    metrics = {"refused_as_archive_corruption": refusal == "ArrayArchiveCorruptionError"}
    return _record(
        "restart-truncated-archive",
        axis * axis,
        capacity,
        seed,
        2,
        recorder,
        metrics,
        retained=(),
        oracle="canonical array-archive framing of the committed generation",
        consumer="phydrax.solver.DurableCheckpointStore.latest via PreparedProductionRun.resume",
        reserved=reserved,
        series={"refusal": refusal},
    )


def measure_restart_foreign_case(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A checkpoint of another case with identical discretization identities refuses."""
    _require_float64(config, "restart-foreign-case")
    axis = _restart_axis(capacity)
    reserved = _restart_reservation(axis * axis, config)
    recorder = PhaseRecorder()
    recorder.unavailable("search", _FUSED_SEARCH)
    with tempfile.TemporaryDirectory(prefix="q17-foreign-") as directory:
        leg, state = _committed(recorder, Path(directory) / "checkpoints", axis, seed)
        foreign_manifest = solver.ProductionCaseManifest.from_inventory(
            leg.inventory, problem_id="another-case", dtype="float64"
        )
        foreign = Path(directory) / "foreign"
        recorder.run(
            "epoch-commit",
            lambda: _durable(foreign, leg.plan, foreign_manifest).checkpoint(state),
            scope="foreign",
        )
        runtime = _durable(foreign, leg.plan, leg.manifest)

        def attempt() -> bool:
            try:
                runtime.resume(state)
            except ValueError as error:
                if "another store" not in str(error):
                    raise
                return True
            return False

        refused = recorder.run("restart", attempt, scope="foreign-case")
    return _record(
        "restart-foreign-case",
        axis * axis,
        capacity,
        seed,
        2,
        recorder,
        {"refused_foreign_store": refused},
        retained=(),
        oracle="case manifest identity of the committed store",
        consumer="phydrax.solver.DurableCheckpointStore via PreparedProductionRun.resume",
        reserved=reserved,
    )


def _same_topology_restart(
    recorder: PhaseRecorder,
    root: Path,
    plan: solver.ProductionRunPlan,
    manifest: solver.ProductionCaseManifest,
    initial: Any,
    steps: int,
    /,
) -> tuple[solver.ProductionRunResult, solver.ProductionRunResult]:
    reference = _durable(root / "reference", plan, manifest)
    uninterrupted = recorder.run(
        "solve",
        lambda: reference.run(reference.initial_state(initial)),
        scope="reference",
    )
    runtime = _durable(root / "restarted", plan, manifest)
    start = runtime.initial_state(initial)
    _interrupt(recorder, runtime, start, steps, "interrupted")
    restarted = _durable(root / "restarted", plan, manifest)
    resumed = recorder.run(
        "restart", lambda: restarted.resume(start), scope="interrupted"
    )
    return uninterrupted, recorder.run(
        "solve", lambda: restarted.run(resumed), scope="resumed"
    )


def _equal_runs(reference: Any, restarted: Any, /) -> dict[str, bool]:
    return {
        "status_equal": restarted.status == reference.status,
        "step_index_equal": int(restarted.step_index) == int(reference.step_index),
        "time_equal": _arrays_equal(restarted.time, reference.time),
        "accepted_state_equal": _trees_equal(
            restarted.accepted_state, reference.accepted_state
        ),
        "evidence_equal": _trees_equal(restarted.evidence, reference.evidence),
        "moment_states_equal": _trees_equal(
            restarted.moment_states, reference.moment_states
        ),
    }


# The current state occupies one ring slot: the acknowledged archive window is
# history_capacity - 1 accepted steps (moving_archive_policy owner contract).
_MOVING_HISTORY = 5
_MOVING_STEPS = 12
_MOVING_SIZE = 48
_BULK_SURFACE_SIZE = 128


def _moving_plan(capacity: int, seed: int, /) -> tuple[MovingSurfacePlan, Any]:
    from examples.meshfree_moving_surface_reaction_diffusion import prepare_workflow

    rolling, initial = prepare_workflow(
        size=min(capacity, _MOVING_SIZE), dimension=3, seed=seed
    )
    plan = MovingSurfacePlan(
        rolling.geometry,
        rolling.motion,
        rolling.reaction,
        method=rolling.tableau,
        epoch=rolling.epoch,
        capacity=rolling.capacity,
        history_capacity=_MOVING_HISTORY,
        archive="acknowledged",
        plan_id="growing-sphere:acknowledged-archive",
    )
    # The live-history ring is part of the state layout: initialize on this plan.
    return plan, plan.initialize(initial.points, initial.concentration)


def measure_restart_moving_surface_history(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Moving-surface restart keeps the live-history ring and acknowledged archive cursor."""
    _require_float64(config, "restart-moving-surface-history")
    size = min(capacity, _MOVING_SIZE)
    reserved = _restart_reservation(size * (_MOVING_HISTORY + 1), config)
    recorder = PhaseRecorder()
    plan, state = recorder.run(
        "geometry", lambda: _moving_plan(capacity, seed), scope="moving-surface"
    )
    method = MovingSurfaceFixedStepMethod(plan)
    archive = moving_archive_policy(plan)
    retry = solver.RobustRetryPolicy(maximum_retries=0)
    run_plan = solver.ProductionRunPlan(
        method,
        retry,
        step_size=0.01,
        end_time=0.01 * _MOVING_STEPS,
        maximum_steps=_MOVING_STEPS,
        checkpoint_interval=_MOVING_HISTORY - 1,
        segment_steps=2,
        archive=archive,
    )
    inventory = meshfree_runtime_inventory(
        plan,
        source=_RESTART_SOURCE,
        program=run_plan.plan_id,
        method=method,
        controller=archive.policy_id,
        precision="float64",
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="growing-sphere", dtype="float64"
    )
    with tempfile.TemporaryDirectory(prefix="q17-moving-") as directory:
        reference, restarted = _same_topology_restart(
            recorder, Path(directory), run_plan, manifest, state, 3
        )
    final = reference.state.accepted_state
    evidence = reference.state.evidence
    metrics: dict[str, Any] = {
        "reference_completed": reference.state.status == "completed",
        # Twelve steps through a four-entry window: only acknowledgement admits them.
        "accepted_steps": int(final.accepted_steps),
        "history_window_full": int(final.history_count) == _MOVING_HISTORY,
        "archive_cursor_after_all_steps": int(final.archive_cursor) == _MOVING_STEPS + 1,
        "accepted_status_retained": evidence is not None
        and int(evidence.accepted.status) == int(MovingSurfaceStatus.ACCEPTED),
        **_equal_runs(reference.state, restarted.state),
    }
    metrics["replay_class_bitwise_match"] = all(
        value for key, value in metrics.items() if key.endswith("_equal")
    )
    return _record(
        "restart-moving-surface-history",
        size,
        capacity,
        seed,
        3,
        recorder,
        metrics,
        retained=(reference.state, restarted.state),
        oracle="the uninterrupted production run of the same moving-surface plan",
        consumer="phydrax.discretization.meshfree.{MovingSurfaceFixedStepMethod, "
        "moving_archive_policy} in solver.PreparedProductionRun",
        reserved=reserved,
    )


def measure_restart_moving_history_window_refusal(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """A checkpoint interval longer than the live-history window refuses at planning."""
    _require_float64(config, "restart-moving-history-window-refusal")
    size = min(capacity, _MOVING_SIZE)
    reserved = _restart_reservation(size * (_MOVING_HISTORY + 1), config)
    recorder = PhaseRecorder()
    plan, _ = recorder.run(
        "geometry", lambda: _moving_plan(capacity, seed), scope="moving-surface"
    )
    method = MovingSurfaceFixedStepMethod(plan)
    archive = moving_archive_policy(plan)

    def attempt() -> bool:
        try:
            solver.ProductionRunPlan(
                method,
                solver.RobustRetryPolicy(maximum_retries=0),
                step_size=0.01,
                end_time=0.01 * _MOVING_STEPS,
                maximum_steps=_MOVING_STEPS,
                checkpoint_interval=_MOVING_HISTORY,
                archive=archive,
            )
        except ValueError as error:
            if "history window" not in str(error):
                raise
            return True
        return False

    refused = recorder.run("assembly", attempt, scope="run-plan")
    return _record(
        "restart-moving-history-window-refusal",
        size,
        capacity,
        seed,
        3,
        recorder,
        {"refused_history_window": refused},
        retained=(),
        oracle="declared MovingSurfacePlan.history_capacity",
        consumer="phydrax.solver.ProductionRunPlan(archive=moving_archive_policy(plan))",
        reserved=reserved,
    )


def measure_restart_coupled_bulk_surface(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Coupled bulk-surface restart keeps refused-Newton evidence across the interrupt."""
    from examples.meshfree_bulk_surface_exchange import prepare_workflow

    _require_float64(config, "restart-coupled-bulk-surface")
    size = min(capacity, _BULK_SURFACE_SIZE)
    reserved = _restart_reservation(size, config)
    recorder = PhaseRecorder()
    workflow = recorder.run(
        "assembly",
        lambda: prepare_workflow(
            size=min(capacity, _BULK_SURFACE_SIZE), dimension=3, seed=seed
        ),
        scope="bulk-surface",
    )
    original = workflow.method
    kinetics = original.transport.structure.kinetics
    if kinetics is None:
        raise TypeError("Langmuir exchange requires its actual kinetic owner.")
    # A tight native Newton budget refuses long steps; reduced steps converge.
    method = MeshfreeBulkSurfaceMethod(
        original.query,
        workflow.graph,
        original.bulk_volumes,
        kinetics,
        surface_diffusivity=0.02,
        maximum_iterations=6,
    )
    retry = solver.RobustRetryPolicy(maximum_retries=4, reduction_factor=0.125)
    plan = solver.ProductionRunPlan(
        method,
        retry,
        step_size=8.0,
        end_time=4.0,
        maximum_steps=16,
        checkpoint_interval=2,
        segment_steps=2,
    )
    inventory = meshfree_runtime_inventory(
        method,
        source=_RESTART_SOURCE,
        program=plan.plan_id,
        method=method,
        controller=retry.policy_id,
        precision="float64",
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="bulk-surface-langmuir", dtype="float64"
    )
    with tempfile.TemporaryDirectory(prefix="q17-bulk-surface-") as directory:
        reference, restarted = _same_topology_restart(
            recorder, Path(directory), plan, manifest, workflow.initial, 2
        )
    evidence = reference.state.evidence
    metrics: dict[str, Any] = {
        "reference_completed": reference.state.status == "completed",
        "refused_newton_attempts": 0
        if evidence is None
        else int(evidence.refused_attempts),
        "refused_status_solve_failed": evidence is not None
        and int(evidence.refused.status) == int(FilmStepStatus.SOLVE_FAILED),
        "accepted_status_retained": evidence is not None
        and int(evidence.accepted.status) == int(FilmStepStatus.ACCEPTED),
        **_equal_runs(reference.state, restarted.state),
    }
    metrics["replay_class_bitwise_match"] = all(
        value for key, value in metrics.items() if key.endswith("_equal")
    )
    return _record(
        "restart-coupled-bulk-surface",
        size,
        capacity,
        seed,
        3,
        recorder,
        metrics,
        retained=(reference.state, restarted.state),
        oracle="the uninterrupted production run of the same coupled plan",
        consumer="phydrax.solver.coupling.MeshfreeBulkSurfaceMethod in solver.PreparedProductionRun",
        reserved=reserved,
    )


ADAPTIVE_RESTART_WORKLOADS: dict[
    str, Callable[[int, int, MeshfreeConfig], dict[str, Any]]
] = {
    "adaptive-bulk-boundary-layer": measure_adaptive_bulk_boundary_layer,
    "adaptive-bulk-rejected-coarsening": measure_adaptive_bulk_rejected_coarsening,
    "adaptive-bulk-capacity-refusal": measure_adaptive_bulk_capacity_refusal,
    "adaptive-sphere-zonal-peak": measure_adaptive_sphere_zonal_peak,
    "learned-correction-constrained": measure_learned_correction_constrained,
    "learned-correction-signed-refusal": measure_learned_correction_signed_refusal,
    "learned-correction-training": measure_learned_correction_training,
    "learned-correction-failed-step": measure_learned_correction_failed_step,
    "learned-correction-coverage-refusal": measure_learned_correction_coverage_refusal,
    "learned-coupled-monotone-invariance": measure_learned_coupled_monotone_invariance,
    "learned-coupled-lipschitz-invariance": measure_learned_coupled_lipschitz_invariance,
    "hybrid-overlap-schwarz": measure_hybrid_overlap_schwarz,
    "calibrated-hybrid-held-out": measure_calibrated_hybrid_held_out,
    "restart-refused-epoch-interrupt": measure_restart_refused_epoch_interrupt,
    "restart-cli-sigkill": measure_restart_cli_sigkill,
    "restart-ownership-migration": measure_restart_ownership_migration,
    "restart-ownership-migration-worker": measure_restart_ownership_migration_worker,
    "restart-stale-geometry": measure_restart_stale_geometry,
    "restart-stale-capacity": measure_restart_stale_capacity,
    "restart-stale-precision": measure_restart_stale_precision,
    "restart-stale-source": measure_restart_stale_source,
    "restart-stale-program": measure_restart_stale_program,
    "restart-truncated-archive": measure_restart_truncated_archive,
    "restart-foreign-case": measure_restart_foreign_case,
    "restart-moving-surface-history": measure_restart_moving_surface_history,
    "restart-moving-history-window-refusal": measure_restart_moving_history_window_refusal,
    "restart-coupled-bulk-surface": measure_restart_coupled_bulk_surface,
}
