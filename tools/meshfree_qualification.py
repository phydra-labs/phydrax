# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run selectable meshfree qualification campaigns Q1–Q17; evidence is not release.

Accepted campaigns (``Q<N>``) and declared expected-refusal campaigns
(``Q<N>-refusal``) are separate selections: a refusal never satisfies an
accepted gate. Every planned row names one exact declared support tuple;
requested support, capacity, dimension, precision and device are validated
before anything executes. Rows become typed qualification criteria, campaign
start/observation records, unreviewed producer evidence, and one fail-closed
coverage matrix per campaign. Nothing here reviews, signs, or releases.

Generic stencil controls govern bulk operators. Q2 has explicit elliptic
stencil, degree, and neighbor controls, independent of the generic policy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from math import comb
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    capture_benchmark_identity,
    capture_environment,
    logical_array_bytes,
)
from benchmarks.meshfree_scaling import (
    add_config_arguments,
    BenchmarkExecutionScope,
    cloud_points,
    declare_reservation,
    DeclaredCapacityRefusal,
    failure_record,
    fill_distance,
    MeshfreeConfig,
    PhaseRecorder,
    PROJECT_ROOT,
    unit_cube_probes,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import PointDiffusionForm
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    meshfree_candidate_profiles,
    meshfree_support,
    MeshfreeApproximation,
    PointTransferStatus,
)
from phydrax.linalg import SparseAssemblyPolicy
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    CapabilityProfile,
    QualificationCriterion,
    QualificationEvidence,
    QualificationMatrix,
    SupportTuple,
    validate_qualification_causality,
)
from phydrax.typing import parse


type SupportValue = str | int | bool
type GateRecord = dict[str, Any]
type RowRecord = dict[str, Any]

CampaignScenario: TypeAlias = Literal["accepted", "expected-refusal"]
GateStatus: TypeAlias = Literal["passed", "failed", "unexecuted", "blocked"]
GateComparison: TypeAlias = Literal[
    "less-than-or-equal", "greater-than-or-equal", "equal"
]
GateEvidenceKind: TypeAlias = Literal["scientific", "performance", "smoke", "operational"]
MeshfreeDevice: TypeAlias = Literal[
    "cpu",
    "cpu-forced-multi-device",
    "gpu",
    "multi-device-gpu",
    "multi-host-cpu",
    "multi-host-gpu",
]
EllipticBoundary: TypeAlias = Literal["dirichlet", "neumann", "robin", "mixed"]

# Thresholds are declared by the closure plan (P16); approval names the plan,
# not a reviewer. Producer evidence is explicitly unreviewed.
APPROVAL_ID = "meshfree-closure-plan-p16-declared-thresholds"
REVIEWER_ID = "unreviewed-automated-producer"
EVIDENCE_VALIDITY_NS = 30 * 24 * 3600 * 1_000_000_000
_SHARED: dict[tuple[Hashable, ...], dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Gate vocabulary


def gate(
    name: str,
    status: GateStatus,
    /,
    *,
    metric: str,
    comparison: GateComparison,
    target: float,
    observed: Any,
    unit: str = "1",
    evidence_kind: GateEvidenceKind = "scientific",
    reason: str | None = None,
    support_tuple_id: str | None = None,
    coordinates: dict[str, Any] | None = None,
) -> GateRecord:
    """One declared criterion observation; missing, blocked and failed stay distinct."""
    record: GateRecord = {
        "gate": name,
        "status": parse(status, GateStatus, "status"),
        "metric": metric,
        "comparison": parse(comparison, GateComparison, "comparison"),
        "target": float(target),
        "unit": unit,
        "observed": observed,
        "evidence_kind": parse(evidence_kind, GateEvidenceKind, "evidence_kind"),
        "reason": reason,
    }
    if support_tuple_id is not None:
        record["support_tuple_id"] = support_tuple_id
    if coordinates is not None:
        record["coordinates"] = coordinates
    return record


def _number(metrics: Mapping[str, object], key: str, /) -> float | None:
    """Read a native integer/float measurement; missing leaves remain unavailable."""
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{key} must be a numerical measurement, not a Boolean.")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a native numerical measurement or unavailable.")
    return float(value)


def upper_gate(
    name: str,
    metrics: Mapping[str, object],
    key: str,
    tolerance: float,
    /,
    *,
    unit: str = "1",
    evidence_kind: GateEvidenceKind = "scientific",
) -> GateRecord:
    """Pass iff the measured magnitude is finite and at most ``tolerance``."""
    value = _number(metrics, key)
    status: GateStatus = (
        "unexecuted"
        if value is None
        else "passed"
        if math.isfinite(value) and abs(value) <= tolerance
        else "failed"
    )
    return gate(
        name,
        status,
        metric=key,
        comparison="less-than-or-equal",
        target=tolerance,
        observed=value,
        unit=unit,
        evidence_kind=evidence_kind,
        reason=f"Workload did not measure {key}" if value is None else None,
    )


def lower_gate(
    name: str,
    metrics: dict[str, Any],
    key: str,
    minimum: float,
    /,
    *,
    unit: str = "1",
    evidence_kind: GateEvidenceKind = "scientific",
) -> GateRecord:
    """Pass iff the measured value is finite and at least ``minimum``."""
    value = _number(metrics, key)
    status: GateStatus = (
        "unexecuted"
        if value is None
        else "passed"
        if math.isfinite(value) and value >= minimum
        else "failed"
    )
    return gate(
        name,
        status,
        metric=key,
        comparison="greater-than-or-equal",
        target=minimum,
        observed=value,
        unit=unit,
        evidence_kind=evidence_kind,
        reason=f"Workload did not measure {key}" if value is None else None,
    )


def boolean_gate(
    name: str,
    metrics: Mapping[str, object],
    key: str,
    /,
    *,
    expected: bool = True,
    evidence_kind: GateEvidenceKind = "scientific",
) -> GateRecord:
    """Pass iff the measured Boolean equals ``expected``."""
    value = metrics.get(key)
    if value is not None and type(value) is not bool:
        raise TypeError(f"{key} must be a measured Boolean or unavailable.")
    status: GateStatus = (
        "unexecuted" if value is None else "passed" if value is expected else "failed"
    )
    return gate(
        name,
        status,
        metric=key,
        comparison="equal",
        target=float(expected),
        observed=value,
        evidence_kind=evidence_kind,
        reason=f"Workload did not measure {key}" if value is None else None,
    )


def blocked_gate(
    name: str,
    /,
    *,
    metric: str,
    comparison: GateComparison,
    target: float,
    reason: str,
    unit: str = "1",
    evidence_kind: GateEvidenceKind = "scientific",
) -> GateRecord:
    """A declared criterion that cannot execute here (hardware, owner, reference)."""
    if not reason:
        raise ValueError("A blocked gate needs an explicit reason.")
    return gate(
        name,
        "blocked",
        metric=metric,
        comparison=comparison,
        target=target,
        observed=None,
        unit=unit,
        evidence_kind=evidence_kind,
        reason=reason,
    )


def rate_gate(
    name: str,
    scales: Sequence[float],
    errors: Sequence[float],
    minimum: float,
    /,
    *,
    metric: str,
    provenance: str,
    support_tuple_id: str,
    coordinates: dict[str, Any],
) -> GateRecord:
    """Least-squares observed order over at least four measured resolutions."""
    pairs = [
        (float(scale), float(error))
        for scale, error in zip(scales, errors, strict=True)
        if scale is not None and error is not None
    ]
    measured = {
        **coordinates,
        "scales": [scale for scale, _ in pairs],
        "errors": [error for _, error in pairs],
        "scale_provenance": provenance,
    }
    distinct = len({scale for scale, _ in pairs})
    if distinct < 4 or any(
        not (math.isfinite(scale) and math.isfinite(error) and scale > 0 and error > 0)
        for scale, error in pairs
    ):
        return gate(
            name,
            "unexecuted",
            metric=metric,
            comparison="greater-than-or-equal",
            target=minimum,
            observed=None,
            reason="An order claim needs >= 4 positive finite measured resolutions; "
            f"got {distinct}",
            support_tuple_id=support_tuple_id,
            coordinates=measured,
        )
    slope, _ = np.polyfit(
        np.log([scale for scale, _ in pairs]), np.log([error for _, error in pairs]), 1
    )
    rate = float(slope)
    return gate(
        name,
        "passed" if rate >= minimum else "failed",
        metric=metric,
        comparison="greater-than-or-equal",
        target=minimum,
        observed=rate,
        support_tuple_id=support_tuple_id,
        coordinates=measured,
    )


# ---------------------------------------------------------------------------
# Campaign contract


@dataclass(frozen=True, slots=True)
class PlannedRow:
    """One validated-before-execution row bound to one exact declared support tuple."""

    capability: str
    support: dict[str, SupportValue]
    capacity: int
    seed: int
    label: str
    execute: Callable[[], dict[str, Any]]


type CampaignPlanner = Callable[[MeshfreeConfig], tuple[PlannedRow, ...]]
type CampaignAggregate = Callable[[Sequence[RowRecord], MeshfreeConfig], list[GateRecord]]


@dataclass(frozen=True, slots=True)
class Campaign:
    """A finite selectable campaign with declared default capacities and dimension."""

    description: str
    scenario: CampaignScenario
    sizes: tuple[int, ...]
    dimension: int
    plan: CampaignPlanner
    aggregate: CampaignAggregate | None = None


def shared(
    key: tuple[Hashable, ...], compute: Callable[[], dict[str, Any]], /
) -> dict[str, Any]:
    """Execute one consumer once for every support tuple/campaign it observes.

    Later observers receive the same measured record marked ``reused`` so its
    phases, wall/CPU time and compile counts are never counted twice.
    """
    if key in _SHARED:
        return {**_SHARED[key], "shared_execution": "reused"}
    _SHARED[key] = {**compute(), "shared_execution": "first"}
    return _SHARED[key]


def _consumer_record(
    module_file: str | None, consumer: str, metrics: dict[str, Any], /
) -> dict[str, Any]:
    if module_file is None:
        raise TypeError(f"{consumer} must expose a concrete source file.")
    return {
        "consumer": consumer,
        "metrics": metrics,
        "consumer_source_fingerprint": benchmark_driver_fingerprint(
            PROJECT_ROOT, Path(module_file)
        ),
        "oracle_provenance": metrics.get("oracle_provenance"),
        "domain": metrics.get("domain", metrics.get("training_domain")),
    }


def _fill(points: np.ndarray, dimension: int, seed: int, /) -> dict[str, Any]:
    return fill_distance(
        points,
        unit_cube_probes(dimension, 4 * points.shape[0], seed),
        provenance="max nearest-sample distance over scrambled Sobol probes of [0,1]^d "
        "(a lower estimate of the supremum)",
    )


# ---------------------------------------------------------------------------
# Q1 strong-form derivatives


_DERIVATIVE_ORDERS = {"first-derivative": 1, "laplacian": 2}


def _q1_support(method: str, config: MeshfreeConfig, /) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method=method,
        boundary="none",
        geometry_authority="unit-cube-stratified-samples",
        derivative_class="coordinate-orders-1-2",
        capacity=16384,
        precision=config.precision,
    )


def _degenerate_cloud(capacity: int, dimension: int, /) -> np.ndarray:
    """Outside polynomial unisolvency: collinear (d>=2) or two-site (d=1) samples."""
    line = np.zeros((capacity, dimension))
    if dimension == 1:
        line[:, 0] = np.arange(capacity) % 2
    else:
        line[:, 0] = np.linspace(0, 1, capacity)
    return line


def _q1_functionals(dimension: int, /) -> tuple[Any, ...]:
    from phydrax.discretization.meshfree import MeshfreeFunctional

    gradient = tuple(
        MeshfreeFunctional(
            (tuple(int(d == axis) for d in range(dimension)),),
            np.asarray([1.0]),
            name=f"partial-{axis}",
        )
        for axis in range(dimension)
    )
    laplacian = MeshfreeFunctional(
        tuple(
            tuple(2 * int(d == axis) for d in range(dimension))
            for axis in range(dimension)
        ),
        np.ones(dimension),
        name="laplacian",
    )
    return (*gradient, laplacian)


def _q1_row(
    method: MeshfreeApproximation, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from phydrax.discretization.meshfree import (
        MeshfreeNeighborhoodPlan,
        MeshfreePrecisionPolicy,
        prepare_local_stencils,
    )

    reservation = config.check_capacity(capacity)
    recorder = PhaseRecorder()
    # float32 rows keep 64-bit Morton addressing and declare every precision
    # role float32 (geometry, fit, coefficients, certification).
    dtype = np.float32 if config.precision == "float32" else np.float64
    points = cloud_points(capacity, config.dimension, seed).astype(dtype)
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(
            points,
            config.neighbors,
            maximum_candidates=config.candidates,
            target_chunk_size=config.chunk_rows,
            precision=MeshfreePrecisionPolicy(geometry_dtype=dtype),
        ).prepare(),
    )
    prepared = recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            points,
            points,
            _q1_functionals(config.dimension),
            LocalStencilPolicy(
                approximation=method,
                polynomial_degree=config.degree,
                chunk_rows=config.chunk_rows,
            ),
        ),
    )
    relation = prepared.neighborhood.relation
    weights = jnp.stack(tuple(prepared.weights))

    def action(values: Array) -> Array:
        gathered = values[relation.source_indices]
        return jnp.sum(
            jnp.where(relation.valid[None], weights * gathered[None], 0), axis=2
        ).T

    polynomial = jnp.asarray(np.sum(points**2, axis=1) + points[:, 0])
    actual, compiled = recorder.compiled_action(
        action, polynomial, budget_bytes=config.resource_bytes, repeats=config.repeats
    )
    expected = np.column_stack(
        (
            2 * points + np.eye(config.dimension)[0],
            np.full(capacity, 2 * config.dimension),
        )
    )
    polynomial_error = float(np.max(np.abs(np.asarray(actual) - expected)))
    smooth = np.exp(np.sum(points, axis=1))
    smooth_actual = np.asarray(action(jnp.asarray(smooth)))
    smooth_expected = np.column_stack(
        (np.broadcast_to(smooth[:, None], points.shape), config.dimension * smooth)
    )
    # Interior rows isolate the interior consistency order from one-sided rows.
    interior = np.all((points >= 0.25) & (points <= 0.75), axis=1)
    defects = (smooth_actual - smooth_expected)[interior]
    fill = _fill(points, config.dimension, seed)
    tolerance = 5e-3 if config.precision == "float32" else 1e-7
    metrics = {
        "polynomial_error": polynomial_error,
        "first-derivative": float(np.sqrt(np.mean(defects[:, :-1] ** 2))),
        "laplacian": float(np.sqrt(np.mean(defects[:, -1] ** 2))),
        "interior_rows": int(np.count_nonzero(interior)),
        "refused_rows": prepared.report.refused_rows,
        "maximum_condition_number": prepared.report.maximum_condition_number,
        "moment_residual": prepared.report.maximum_moment_residual,
        "fill_distance": fill["fill_distance"],
        "separation_distance": fill["separation_distance"],
    }
    return {
        "metrics": metrics,
        "fill": fill,
        "phases": recorder.record(),
        **compiled,
        "retained_bytes": logical_array_bytes(prepared),
        "reserved_working_set_bytes": reservation,
        "prepared_id": prepared.prepared_id,
        "consumer": "phydrax.discretization.meshfree.prepare_local_stencils",
        "oracle_provenance": "independent host NumPy: polynomial sum(x^2)+x0 exact "
        "derivatives; exp(sum x) analytic gradient/Laplacian on interior [0.25,0.75]^d",
        "gates": [
            gate(
                "analytic",
                "passed" if polynomial_error <= tolerance else "failed",
                metric="polynomial_error",
                comparison="less-than-or-equal",
                target=tolerance,
                observed=polynomial_error,
            ),
            gate(
                "row-acceptance",
                "passed" if prepared.report.refused_rows == 0 else "failed",
                metric="refused_rows",
                comparison="equal",
                target=0,
                observed=prepared.report.refused_rows,
                unit="rows",
            ),
        ],
    }


def _q1_methods(config: MeshfreeConfig, /) -> tuple[MeshfreeApproximation, ...]:
    # PHS saddles on these clouds have condition 1e4-1e6, beyond the float32 fit
    # admission (condition * eps32 <= 1e-3, rounding-level moments): float32 PHS
    # is not a declared support tuple; the stencil owner refuses its rows.
    return ("gmls",) if config.precision == "float32" else ("gmls", "phs-rbf-fd")


def _plan_q1(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    methods = _q1_methods(config)
    return tuple(
        PlannedRow(
            "meshfree.strong-form",
            _q1_support(method, config),
            size,
            seed,
            method,
            partial(_q1_row, method, size, seed, config),
        )
        for method in methods
        for size in config.sizes
        for seed in config.seeds
    )


def _q1_convergence(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /
) -> list[GateRecord]:
    gates: list[GateRecord] = []
    for support_id in sorted({row["support_tuple_id"] for row in rows}):
        for seed in config.seeds:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["support_tuple_id"] == support_id
                    and row["seed"] == seed
                    and row["status"] != "failed"
                    and "metrics" in row
                ),
                key=lambda row: row["capacity"],
            )
            for derivative, order in _DERIVATIVE_ORDERS.items():
                # Degree-p reproduction: order-k derivative error O(h^(p+1-k)).
                minimum = config.degree + 1 - order - 0.5
                gates.append(
                    rate_gate(
                        f"convergence-{derivative}",
                        [row["metrics"]["fill_distance"] for row in selected],
                        [row["metrics"][derivative] for row in selected],
                        minimum,
                        metric=f"{derivative}-interior-rms-order",
                        provenance="measured fill distance (scrambled Sobol probes)",
                        support_tuple_id=support_id,
                        coordinates={
                            "seed": seed,
                            "derivative": derivative,
                            "derivative_order": order,
                            "polynomial_degree": config.degree,
                            "capacities": [row["capacity"] for row in selected],
                        },
                    )
                )
    return gates


def _q1_refusal_row(
    method: MeshfreeApproximation, capacity: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from phydrax.discretization.meshfree import (
        MeshfreeNeighborhoodPlan,
        prepare_local_stencils,
    )

    config.check_capacity(capacity)
    recorder = PhaseRecorder()
    line = _degenerate_cloud(capacity, config.dimension)
    functionals = _q1_functionals(config.dimension)
    if config.dimension == 1:
        # Distinct 1-D samples are always unisolvent; the declared 1-D refusal
        # is duplicate-sample rejection at neighborhood preparation.
        duplicate_message: str | None = None
        try:
            MeshfreeNeighborhoodPlan(
                line, config.neighbors, target_chunk_size=config.chunk_rows
            ).prepare()
        except ValueError as error:
            if "exact duplicates" not in str(error):
                raise
            duplicate_message = str(error)
        duplicate_metrics = {"declared_refusal": duplicate_message is not None}
        return {
            "metrics": duplicate_metrics,
            "refusal_message": duplicate_message,
            "degenerate_cloud": "two-site-duplicates",
            "phases": recorder.record(),
            "consumer": "phydrax.discretization.meshfree.MeshfreeNeighborhoodPlan",
            "oracle_provenance": "duplicate samples cannot define a local polynomial fit",
            "gates": [
                boolean_gate("declared-refusal", duplicate_metrics, "declared_refusal")
            ],
        }
    neighborhood = recorder.run(
        "search",
        lambda: MeshfreeNeighborhoodPlan(
            line, config.neighbors, target_chunk_size=config.chunk_rows
        ).prepare(),
    )
    masked = recorder.run(
        "local-fit",
        lambda: prepare_local_stencils(
            neighborhood,
            line,
            line,
            functionals,
            LocalStencilPolicy(
                approximation=method,
                polynomial_degree=config.degree,
                chunk_rows=config.chunk_rows,
                acceptance="mask",
            ),
        ),
        scope="acceptance=mask",
    )
    statuses, counts = np.unique(np.asarray(masked.evidence.status), return_counts=True)
    refusal_message: str | None = None
    try:
        prepare_local_stencils(
            neighborhood,
            line,
            line,
            functionals,
            LocalStencilPolicy(
                approximation=method,
                polynomial_degree=config.degree,
                chunk_rows=config.chunk_rows,
            ),
        )
    except ValueError as error:
        if not str(error).startswith(
            "Meshfree stencil row "
        ) or "RANK_DEFICIENT" not in str(error):
            raise
        refusal_message = str(error)
    metrics = {
        "declared_refusal": refusal_message is not None,
        "masked_refused_rows": masked.report.refused_rows,
        "unrefused_rows": capacity - masked.report.refused_rows,
    }
    return {
        "metrics": metrics,
        "refusal_message": refusal_message,
        "degenerate_cloud": "collinear",
        "row_status_counts": {
            str(int(status)): int(count)
            for status, count in zip(statuses, counts, strict=True)
        },
        "phases": recorder.record(),
        "consumer": "phydrax.discretization.meshfree.prepare_local_stencils",
        "oracle_provenance": "degenerate sample geometry is not unisolvent for the declared basis",
        "gates": [
            boolean_gate("declared-refusal", metrics, "declared_refusal"),
            upper_gate("masked-row-refusal", metrics, "unrefused_rows", 0, unit="rows"),
        ],
    }


def _plan_q1_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    methods = _q1_methods(config)
    return tuple(
        PlannedRow(
            "meshfree.strong-form",
            _q1_support(method, config),
            size,
            0,
            f"{method}-degenerate",
            partial(_q1_refusal_row, method, size, config),
        )
        for method in methods
        for size in config.sizes
    )


# ---------------------------------------------------------------------------
# Q2 boundary elliptic solves


@dataclass(frozen=True, slots=True)
class EllipticSelection:
    """Q2 elliptic route, explicit and separate from the generic stencil policy."""

    boundaries: tuple[EllipticBoundary, ...] = ("dirichlet", "neumann", "robin", "mixed")
    forms: tuple[PointDiffusionForm, ...] = ("collocated",)
    stencil: MeshfreeApproximation = "phs-rbf-fd"
    degree: int = 3
    neighbors: int | None = None

    def __post_init__(self) -> None:
        for name, values, alias in (
            ("boundaries", self.boundaries, EllipticBoundary),
            ("forms", self.forms, PointDiffusionForm),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"Select distinct elliptic {name}.")
            for value in values:
                parse(value, alias, name)


def _q2_row(
    boundary: EllipticBoundary,
    form: PointDiffusionForm,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    selection: EllipticSelection,
    /,
) -> dict[str, Any]:
    import examples.point_cloud_poisson as consumer

    # The dissipative consumer binds actual second-order native tensor SBP
    # families, not constrained GMLS coefficients. Its separate local GMLS
    # cloud preparation remains reserved, but does not own the weak derivative.
    explicit_sbp = form == "dissipative"
    local_degree = _Q2_SBP_LOCAL_DEGREE if explicit_sbp else selection.degree
    policy = LocalStencilPolicy(
        approximation="gmls" if explicit_sbp else selection.stencil,
        polynomial_degree=local_degree,
        chunk_rows=config.chunk_rows,
    )
    basis = comb(config.dimension + local_degree, local_degree)
    neighbor_capacity = (
        min(capacity, 3 * basis)
        if explicit_sbp
        else min(capacity, 2 * basis)
        if selection.neighbors is None
        else selection.neighbors
    )
    stencil_reservation = config.check_capacity(
        capacity, degree=local_degree, neighbors=neighbor_capacity
    )
    assembly_reservation = (
        64
        * capacity
        * neighbor_capacity
        * (
            config.dimension * neighbor_capacity
            if explicit_sbp
            else 2 * config.dimension + 4
        )
    )
    reservation = declare_reservation(
        stencil_reservation + assembly_reservation,
        config,
        scope="elliptic stencil/saddle/assembly",
    )
    assembly_policy = SparseAssemblyPolicy(
        max_workspace_bytes=min(config.working_set_bytes, config.resource_bytes)
        - stencil_reservation
    )
    recorder = PhaseRecorder()
    metrics = recorder.run(
        "end-to-end",
        lambda: consumer.run_workflow(
            size=capacity,
            dimension=config.dimension,
            seed=seed,
            manufactured="exponential",
            boundary_kind=boundary,
            form=form,
            stencil=None if explicit_sbp else policy,
            neighbors=None if explicit_sbp else selection.neighbors,
            target_chunk_size=config.chunk_rows,
            assembly_policy=assembly_policy,
        ),
        scope="point_cloud_poisson.run_workflow",
    )
    recorder.unavailable(
        "solve", "The public workflow fuses preparation, assembly and solve"
    )
    gates = [
        upper_gate("analytic", metrics, "maximum_solution_error", 5e-2),
        upper_gate("boundary-equations", metrics, "boundary_residual_norm", 1e-5),
        upper_gate("boundary-field-error", metrics, "maximum_boundary_error", 5e-2),
        boolean_gate("solve-status", metrics, "successful"),
    ]
    if boundary == "dirichlet":
        gates.append(
            upper_gate("exact-Dirichlet-trace", metrics, "maximum_boundary_error", 1e-5)
        )
    if explicit_sbp:
        metrics = {**metrics, **_q2_sbp_ratios(metrics, boundary)}
        gates.extend(_q2_sbp_gates(metrics, boundary))
    else:
        gates.append(upper_gate("true-residual", metrics, "residual_norm", 1e-5))
    # The example selects the ghost-layer PDE+BC route for every collocated
    # flux (Neumann/Robin/mixed) boundary; the row records the route it used.
    ghost_route = form == "collocated" and boundary != "dirichlet"
    metrics = {
        **metrics,
        "declared_ghost_route": ghost_route,
        "ghost_route_used": metrics["boundary_route"] == "ghost-collocation",
    }
    if ghost_route:
        gates.append(boolean_gate("ghost-route-selected", metrics, "ghost_route_used"))
        # Extension law evidence max_g |u_g - (E u)(x_g)|: u_g and the one-sided
        # cloud reconstruction E u both approximate the same smooth continuation
        # one boundary spacing outside the domain, so the defect is bounded by
        # the solution error plus the reconstruction truncation. Declared at the
        # same absolute tolerance as the analytic solution gate (O(1)-scale
        # exponential manufactured solution).
        gates.append(
            upper_gate("ghost-extension-defect", metrics, "ghost_extension_defect", 5e-2)
        )
    return {
        **_consumer_record(
            consumer.__file__, "examples.point_cloud_poisson.run_workflow", metrics
        ),
        "phases": recorder.record(),
        "reserved_working_set_bytes": reservation,
        "boundary_route": metrics["boundary_route"],
        "elliptic_policy": {
            "derivative_realization": "native-tensor-sbp"
            if explicit_sbp
            else "local-meshfree-collocation",
            "sbp_interior_order": 2 if explicit_sbp else None,
            "approximation": policy.approximation,
            "polynomial_degree": policy.polynomial_degree,
            "neighbor_capacity": neighbor_capacity,
            "neighbor_policy": "native-support-count"
            if selection.neighbors is None
            else "explicit",
            "stencil_saddle_reservation_bytes": stencil_reservation,
            "assembly_reservation_bytes": assembly_reservation,
            "native_assembly_workspace_budget": assembly_policy.max_workspace_bytes,
        },
        "qualification_scope": "projected-source-point-gauged-Neumann"
        if boundary == "neumann"
        else "independent-continuum-MMS",
        "gates": gates,
    }


_Q2_SBP_LOCAL_DEGREE = 2
_Q2_SBP_REPRODUCTION_DEGREE = 1
# Driver-side criterion: the original (unprojected) weak equations assembled
# from the physical operator and analytic load must hold to 1e-6 relative.
# The native solve stops at its own residual_tolerance (checked separately at
# ratio <= 1); 1e-6 allows assembly roundoff without admitting an unsolved
# system. Pure Neumann projects the source, so its original residual is
# reported, not gated (the projected-source correction is the claim there).
_Q2_ORIGINAL_RELATIVE_RESIDUAL = 1e-6


def _ratio(numerator: float | None, denominator: float | None, /) -> float | None:
    if numerator is None or denominator is None:
        return None
    return numerator / denominator if denominator > 0 else math.inf


def _q2_sbp_ratios(
    metrics: Mapping[str, object], boundary: str, /
) -> dict[str, float | bool | None]:
    """Normalize validated native residuals by their declared native tolerances."""
    degree = metrics.get("sbp_reproduction_degree")
    if degree is not None and (isinstance(degree, bool) or not isinstance(degree, int)):
        raise TypeError(
            "sbp_reproduction_degree must be a native integer or unavailable."
        )
    ratios: dict[str, float | bool | None] = {
        "weak_residual_ratio": _ratio(
            _number(metrics, "residual_norm"), _number(metrics, "residual_tolerance")
        ),
        "sbp_reproduction_ratio": _ratio(
            _number(metrics, "sbp_reproduction_residual"),
            _number(metrics, "sbp_tolerance"),
        ),
        "sbp_green_ratio": _ratio(
            _number(metrics, "sbp_green_residual"), _number(metrics, "sbp_tolerance")
        ),
        "sbp_conservation_ratio": _ratio(
            _number(metrics, "sbp_conservation_residual"),
            _number(metrics, "sbp_tolerance"),
        ),
        "sbp_declared_degree": degree == _Q2_SBP_REPRODUCTION_DEGREE,
    }
    if boundary != "neumann":
        ratios["original_relative_residual"] = _ratio(
            _number(metrics, "true_original_residual"),
            _number(metrics, "original_load_norm"),
        )
    return ratios


def _q2_sbp_gates(metrics: dict[str, Any], boundary: str, /) -> list[GateRecord]:
    """Native tensor SBP binding, consistency and original weak equations."""
    gates = [
        boolean_gate("sbp-admitted", metrics, "sbp_successful"),
        boolean_gate("native-sbp-stable-realization", metrics, "sbp_stable_realization"),
        boolean_gate("sbp-declared-degree", metrics, "sbp_declared_degree"),
        boolean_gate("continuum-consistent", metrics, "continuum_consistent"),
        upper_gate("sbp-reproduction", metrics, "sbp_reproduction_ratio", 1.0),
        upper_gate("sbp-green-identity", metrics, "sbp_green_ratio", 1.0),
        upper_gate("sbp-conservation", metrics, "sbp_conservation_ratio", 1.0),
        # Canonical profile gate: the executed weak system's true residual
        # against its declared native solver tolerance. Original unprojected-
        # load diagnostics and Neumann source corrections stay separate.
        upper_gate("true-residual", metrics, "weak_residual_ratio", 1.0),
    ]
    if boundary != "neumann":
        gates.append(
            upper_gate(
                "original-weak-residual",
                metrics,
                "original_relative_residual",
                _Q2_ORIGINAL_RELATIVE_RESIDUAL,
            )
        )
    return gates


def _q2_support(
    boundary: str, form: str, config: MeshfreeConfig, selection: EllipticSelection, /
) -> dict[str, SupportValue]:
    if form == "dissipative":
        return meshfree_support(
            dimension=config.dimension,
            method="native-tensor-sbp-interior-order-2-dissipative",
            boundary=boundary,
            geometry_authority="native-point-primary-tensor-grid-sbp-norm",
            derivative_class="none",
            capacity=16384,
            precision=config.precision,
        )
    return meshfree_support(
        dimension=config.dimension,
        method=f"{selection.stencil}-{form}",
        boundary=boundary,
        geometry_authority="cartesian-box-perturbed-samples",
        derivative_class="none",
        capacity=16384,
        precision=config.precision,
    )


def _plan_q2(
    config: MeshfreeConfig, /, *, selection: EllipticSelection = EllipticSelection()
) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.elliptic-solve",
            _q2_support(boundary, form, config, selection),
            size,
            seed,
            f"native-tensor-sbp-{form}-{boundary}"
            if form == "dissipative"
            else f"{selection.stencil}-{form}-{boundary}",
            partial(_q2_row, boundary, form, size, seed, config, selection),
        )
        for boundary in selection.boundaries
        for form in selection.forms
        for size in config.sizes
        for seed in config.seeds
    )


def _q2_convergence(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /, *, degree: int = 3
) -> list[GateRecord]:
    gates: list[GateRecord] = []
    for support_id in sorted({row["support_tuple_id"] for row in rows}):
        scoped = [row for row in rows if row["support_tuple_id"] == support_id]
        method = str(scoped[0]["support"]["method"])
        # Collocated degree-p Laplacian consistency is O(h^(p-1)). The supplied
        # native second-order tensor SBP family owns a degree-one closure;
        # arbitrary constrained reproduction does not establish global order.
        minimum = (
            degree - 1 - 0.5
            if method.endswith("collocated")
            else _Q2_SBP_REPRODUCTION_DEGREE - 0.5
        )
        for seed in config.seeds:
            selected = sorted(
                (
                    row
                    for row in scoped
                    if row["seed"] == seed
                    and row["status"] != "failed"
                    and "metrics" in row
                ),
                key=lambda row: row["capacity"],
            )
            for key in ("maximum_solution_error", "maximum_boundary_error"):
                if (
                    key == "maximum_boundary_error"
                    and scoped[0]["support"]["boundary"] == "dirichlet"
                ):
                    continue
                gates.append(
                    rate_gate(
                        f"convergence-{key}",
                        [row["metrics"].get("fill_distance") for row in selected],
                        [row["metrics"].get(key) for row in selected],
                        minimum,
                        metric=f"{key}-order",
                        provenance=str(
                            selected[0]["metrics"].get(
                                "fill_distance_provenance", "missing fill distance"
                            )
                        )
                        if selected
                        else "no executed rows",
                        support_tuple_id=support_id,
                        coordinates={
                            "seed": seed,
                            "polynomial_degree": degree,
                            "capacities": [row["capacity"] for row in selected],
                        },
                    )
                )
    return gates


# ---------------------------------------------------------------------------
# Q3 fine-system preconditioner comparison


_Q3_PROVIDERS = (
    "native-ilu",
    "native-smoothed-aggregation",
    "native-meshfree-multilevel",
)


def _q3_row(
    provider: str, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from benchmarks.meshfree_multilevel import measure_capacity

    measured = shared(
        ("Q3", capacity, seed, config), lambda: measure_capacity(capacity, seed, config)
    )
    entry = next(item for item in measured["solvers"] if item["provider"] == provider)
    # The shared measurement records a provider failure and continues; only
    # that provider's row fails, with the recorded failure retained verbatim.
    if entry.get("status") == "failed":
        raise RuntimeError(
            f"{provider} failed in the shared measurement: {entry['failure']}"
        )
    reproduction = [
        value for level in measured["reproduction_residuals"] for value in level
    ]
    metrics = {
        "relative_residual": entry["relative_residual"],
        "refreshed_relative_residual": entry["refreshed_relative_residual"],
        "solution_error": entry["solution_error"],
        "iterations": entry["iterations"],
        "refreshed_iterations": entry["refreshed_iterations"],
        "maximum_reproduction_residual": max(
            (abs(value) for value in reproduction), default=0.0
        ),
        "grid_complexity": measured["grid_complexity"],
        "derivative_solves_accepted": all(
            status == "measured" for status in entry["derivatives"].values()
        ),
        "setup_seconds": entry["phases"]["hierarchy"]["wall_seconds"],
        "fine_nnz": measured["fine_nnz"],
    }
    return {
        "metrics": metrics,
        "phases": entry["phases"],
        "shared_preparation_phases": measured["phases"],
        "compiler": entry["compiler"],
        "retained_bytes": entry["retained_bytes"],
        "level_sizes": measured["level_sizes"],
        "consumer": "benchmarks.meshfree_multilevel.measure_capacity",
        "oracle_provenance": measured["oracle"],
        "gates": [
            upper_gate("true-residual", metrics, "relative_residual", 1e-5),
            upper_gate("refresh-residual", metrics, "refreshed_relative_residual", 1e-5),
            boolean_gate("derivative-solve", metrics, "derivative_solves_accepted"),
            upper_gate("analytic", metrics, "solution_error", 1e-5),
            upper_gate(
                "transfer-reproduction", metrics, "maximum_reproduction_residual", 1e-7
            ),
            lower_gate(
                "fine-system-cost",
                metrics,
                "setup_seconds",
                0.0,
                unit="s",
                evidence_kind="performance",
            ),
        ],
    }


def _plan_q3(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    labels = {
        "native-ilu": "native-ilu",
        "native-smoothed-aggregation": "native-smoothed-aggregation",
        "native-meshfree-multilevel": "meshfree-multilevel",
    }
    return tuple(
        PlannedRow(
            "meshfree.multilevel",
            meshfree_support(
                dimension=config.dimension,
                method=f"gmres-{labels[provider]}",
                boundary="dirichlet-convex-hull",
                geometry_authority="unit-cube-stratified-samples",
                derivative_class="none",
                capacity=16384,
                precision=config.precision,
            ),
            size,
            seed,
            provider,
            partial(_q3_row, provider, size, seed, config),
        )
        for provider in _Q3_PROVIDERS
        for size in config.sizes
        for seed in config.seeds
    )


# ---------------------------------------------------------------------------
# Q4 conservative exterior calculus (and its declared nonnegative refusal)


def _q4_metrics(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    import examples.meshfree_conservative_diffusion as consumer

    def compute() -> dict[str, Any]:
        recorder = PhaseRecorder()
        metrics = recorder.run(
            "end-to-end",
            lambda: consumer.run_workflow(
                size=capacity, dimension=config.dimension, seed=seed
            ),
            scope="meshfree_conservative_diffusion.run_workflow",
        )
        return {
            **_consumer_record(
                consumer.__file__,
                "examples.meshfree_conservative_diffusion.run_workflow",
                metrics,
            ),
            "phases": recorder.record(),
        }

    return shared(("Q4", capacity, seed, config), compute)


def _q4_support(config: MeshfreeConfig, /) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method="edge-moment-metric-exact-signed-and-nonnegative-conic",
        boundary="dirichlet-incomplete-axial-neighborhood",
        geometry_authority="cartesian-nodal-control-volumes",
        derivative_class="fixed-topology-coordinate",
        capacity=4096,
        precision=config.precision,
    )


def _q4_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    record = dict(_q4_metrics(capacity, seed, config))
    metrics = record["metrics"]
    record["gates"] = [
        upper_gate("moment", metrics, "moment_residual", 1e-6),
        upper_gate("analytic", metrics, "quadratic_laplacian_error", 1e-6),
        upper_gate("conservation", metrics, "conservation_residual", 1e-7),
        upper_gate(
            "advection-conservation", metrics, "advection_conservation_residual", 1e-7
        ),
        boolean_gate("coercivity", metrics, "reduced_spd"),
        boolean_gate("feasibility", metrics, "metric_accepted"),
        boolean_gate(
            "coordinate-derivative-availability", metrics, "derivative_available"
        ),
        upper_gate("coordinate-derivative", metrics, "derivative_error", 1e-3),
    ]
    return record


def _q4_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = dict(_q4_metrics(capacity, seed, config))
    metrics = record["metrics"]
    record["gates"] = [
        boolean_gate("nonnegative-refusal", metrics, "nonnegative_refused"),
    ]
    return record


def _plan_q4(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.conservative-exterior",
            _q4_support(config),
            size,
            seed,
            "exterior",
            partial(_q4_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


def _plan_q4_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.conservative-exterior",
            _q4_support(config),
            size,
            seed,
            "nonnegative-infeasible",
            partial(_q4_refusal_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


# ---------------------------------------------------------------------------
# Q5 intrinsic closed-surface Laplace–Beltrami


Q5Surface: TypeAlias = Literal["sphere", "torus"]
Q5Geometry: TypeAlias = Literal["implicit-level-set", "sampled-fit"]
_TORUS_MAJOR, _TORUS_MINOR = 2.0, 0.75


def _surface_samples(
    surface: Q5Surface, capacity: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Deterministic samples, analytic normals, density, LB(z) reference, area."""
    index = np.arange(capacity, dtype=np.float64) + 0.5
    match surface:
        case "sphere":
            polar = np.arccos(1 - 2 * index / capacity)
            azimuth = math.pi * (1 + math.sqrt(5)) * index
            points = np.column_stack(
                (
                    np.sin(polar) * np.cos(azimuth),
                    np.sin(polar) * np.sin(azimuth),
                    np.cos(polar),
                )
            )
            return points, points, np.ones(capacity), -2 * points[:, 2], 4 * math.pi
        case "torus":
            theta = 2 * math.pi * index / capacity
            phi = 2 * math.pi * np.mod(index * (math.sqrt(5) - 1) / 2, 1)
            radial = _TORUS_MAJOR + _TORUS_MINOR * np.cos(theta)
            points = np.column_stack(
                (radial * np.cos(phi), radial * np.sin(phi), _TORUS_MINOR * np.sin(theta))
            )
            normals = np.column_stack(
                (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta))
            )
            reference = (
                -np.sin(theta) / _TORUS_MINOR - np.sin(theta) * np.cos(theta) / radial
            )
            return (
                points,
                normals,
                1 / (_TORUS_MINOR * radial),
                reference,
                4 * math.pi**2 * _TORUS_MAJOR * _TORUS_MINOR,
            )


def _surface_probes(surface: Q5Surface, count: int, seed: int, /) -> np.ndarray:
    """Independent uniform-area probes of the analytic surface."""
    generator = np.random.default_rng(seed + 7919)
    match surface:
        case "sphere":
            normal = generator.standard_normal((count, 3))
            return normal / np.linalg.norm(normal, axis=1, keepdims=True)
        case "torus":
            accepted: list[np.ndarray] = []
            total = 0
            while total < count:
                theta = generator.uniform(0, 2 * math.pi, 2 * count)
                keep = generator.uniform(0, 1, 2 * count) <= (
                    _TORUS_MAJOR + _TORUS_MINOR * np.cos(theta)
                ) / (_TORUS_MAJOR + _TORUS_MINOR)
                accepted.append(theta[keep])
                total += int(np.count_nonzero(keep))
            theta = np.concatenate(accepted)[:count]
            phi = generator.uniform(0, 2 * math.pi, count)
            radial = _TORUS_MAJOR + _TORUS_MINOR * np.cos(theta)
            return np.column_stack(
                (radial * np.cos(phi), radial * np.sin(phi), _TORUS_MINOR * np.sin(theta))
            )


def _q5_row(
    surface: Q5Surface,
    geometry_kind: Q5Geometry,
    degree: int,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    from phydrax.discretization.meshfree import (
        ImplicitSurfaceGeometry,
        SampledSurfaceGeometry,
        SurfacePointCloudPlan,
        SurfaceQuadraturePolicy,
    )
    from phydrax.metrix import RegularLevelSetManifold

    # SurfaceGeomP6 measured admission: 12 neighbors for degree 2, 24 for degree 4.
    neighbors = 12 if degree == 2 else 24
    declare_reservation(
        8 * capacity * neighbors * (3 + comb(2 + degree, degree) + 16),
        config,
        scope="surface stencil and geometry fit",
    )
    points, normals, density, reference, area = _surface_samples(surface, capacity)
    recorder = PhaseRecorder()
    match geometry_kind:
        case "sampled-fit":
            geometry: Any = SampledSurfaceGeometry(
                reference_normals=jnp.asarray(normals),
                fit_degree=degree,
                oversampling=1.5 if degree == 4 else 1.0,
                geometry_id=f"sampled-{surface}",
            )
        case "implicit-level-set":

            def constraint(point: Array) -> Array:
                if surface == "sphere":
                    return jnp.asarray([jnp.sum(point * point) - 1])
                return jnp.asarray(
                    [
                        (jnp.sqrt(point[0] ** 2 + point[1] ** 2) - _TORUS_MAJOR) ** 2
                        + point[2] ** 2
                        - _TORUS_MINOR**2
                    ]
                )

            geometry = ImplicitSurfaceGeometry(
                RegularLevelSetManifold(
                    constraint, ambient_dimension=3, codimension=1, manifold_id=surface
                ),
                certified_tube_radius=0.5 if surface == "sphere" else 0.25,
                geometry_id=f"implicit-{surface}",
            )
    prepared = recorder.run(
        "geometry",
        lambda: SurfacePointCloudPlan(
            jnp.asarray(points),
            geometry,
            neighbors,
            quadrature=SurfaceQuadraturePolicy(
                "normalized-density", density=jnp.asarray(density), total_area=area
            ),
            stencil_policy=LocalStencilPolicy(polynomial_degree=degree, chunk_rows=256),
            maximum_candidates=min(capacity, 4096),
            target_chunk_size=128,
        ).prepare(),
        scope="SurfacePointCloudPlan.prepare (search, geometry fit, local fit fused)",
    )
    for phase in ("search", "local-fit"):
        recorder.unavailable(
            phase, "SurfacePointCloudPlan.prepare fuses search, geometry and local fit"
        )
    field = jnp.asarray(points[:, 2])
    actual, compiled = recorder.compiled_action(
        prepared.laplace_beltrami.mv,
        field,
        budget_bytes=config.resource_bytes,
        repeats=config.repeats,
    )
    measures = np.asarray(prepared.measures)
    difference = np.asarray(actual) - reference
    error = float(
        np.sqrt(np.sum(measures * difference**2) / np.sum(measures * reference**2))
    )
    fill = fill_distance(
        points,
        _surface_probes(surface, 4 * capacity, seed),
        provenance="max chordal nearest-sample distance over independent uniform-area "
        "analytic surface probes",
    )
    metrics = {
        "laplace_beltrami_relative_error": error,
        "normal_error": float(
            np.max(np.linalg.norm(np.asarray(prepared.normals) - normals, axis=1))
        ),
        "area_relative_error": float(abs(np.sum(measures) - area) / area),
        "geometry_valid": bool(np.all(np.asarray(prepared.geometry_evidence.valid))),
        "fill_distance": fill["fill_distance"],
        "separation_distance": fill["separation_distance"],
        "polynomial_degree": degree,
        "neighbors": neighbors,
    }
    return {
        "metrics": metrics,
        "fill": fill,
        "phases": recorder.record(),
        **compiled,
        "retained_bytes": logical_array_bytes(prepared),
        "consumer": "phydrax.discretization.meshfree.SurfacePointCloudPlan",
        "oracle_provenance": "independent analytic Laplace-Beltrami of z: sphere -2z; "
        "torus -sin(t)/r - sin(t)cos(t)/(R + r cos t); analytic normals",
        "gates": [
            # A bounded relative error is a smoke gate; order is the accuracy claim.
            upper_gate("analytic", metrics, "laplace_beltrami_relative_error", 0.1),
            upper_gate("normal-comparison", metrics, "normal_error", 0.3),
            boolean_gate("geometry-admission", metrics, "geometry_valid"),
        ],
    }


def _q5_support(
    surface: str, geometry_kind: str, degree: int, config: MeshfreeConfig, /
) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method=f"surface-gmls-laplace-beltrami-degree-{degree}",
        boundary="closed-surface",
        geometry_authority=f"{surface}-{geometry_kind}",
        derivative_class="none",
        capacity=16384,
        precision=config.precision,
    )


def _plan_q5(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    surfaces: tuple[Q5Surface, ...] = get_args(Q5Surface)
    geometries: tuple[Q5Geometry, ...] = get_args(Q5Geometry)
    return tuple(
        PlannedRow(
            "meshfree.surface-operators",
            _q5_support(surface, geometry_kind, degree, config),
            size,
            seed,
            f"{surface}-{geometry_kind}-degree-{degree}",
            partial(_q5_row, surface, geometry_kind, degree, size, seed, config),
        )
        for surface in surfaces
        for geometry_kind in geometries
        for degree in (2, 4)
        for size in config.sizes
        for seed in config.seeds
    )


def _q5_convergence(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /
) -> list[GateRecord]:
    gates: list[GateRecord] = []
    for support_id in sorted({row["support_tuple_id"] for row in rows}):
        scoped = [row for row in rows if row["support_tuple_id"] == support_id]
        degree = int(str(scoped[0]["support"]["method"]).rsplit("-", 1)[1])
        for seed in config.seeds:
            selected = sorted(
                (
                    row
                    for row in scoped
                    if row["seed"] == seed
                    and row["status"] != "failed"
                    and "metrics" in row
                ),
                key=lambda row: row["capacity"],
            )
            # Second-order operator fitted with degree p: O(h^(p-1)); no
            # superconvergence is assumed for irregular torus samples.
            gates.append(
                rate_gate(
                    "convergence",
                    [row["metrics"]["fill_distance"] for row in selected],
                    [
                        row["metrics"]["laplace_beltrami_relative_error"]
                        for row in selected
                    ],
                    degree - 1 - 0.5,
                    metric="laplace-beltrami-relative-l2-order",
                    provenance="measured chordal fill distance over analytic surface probes",
                    support_tuple_id=support_id,
                    coordinates={
                        "seed": seed,
                        "polynomial_degree": degree,
                        "capacities": [row["capacity"] for row in selected],
                    },
                )
            )
    return gates


# ---------------------------------------------------------------------------
# Q6–Q8 public consumer workflows


def _q6_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    import examples.meshfree_moving_surface_reaction_diffusion as consumer

    recorder = PhaseRecorder()
    metrics = recorder.run(
        "end-to-end",
        lambda: consumer.run_workflow(
            size=capacity, dimension=config.dimension, seed=seed
        ),
        scope="meshfree_moving_surface_reaction_diffusion.run_workflow",
    )
    return {
        **_consumer_record(
            consumer.__file__,
            "examples.meshfree_moving_surface_reaction_diffusion.run_workflow",
            metrics,
        ),
        "phases": recorder.record(),
        "gates": [
            upper_gate("dilution", metrics, "concentration_error", 5e-3),
            upper_gate("conservation", metrics, "conservation_residual", 1e-6),
            upper_gate("repeated-shift", metrics, "shift_conservation_residual", 1e-6),
            upper_gate("epoch-remap", metrics, "epoch_conservation_residual", 1e-6),
            upper_gate(
                "repeated-remap", metrics, "repeated_cycle_conservation_residual", 1e-6
            ),
            upper_gate("rollback", metrics, "rollback_error", 1e-8),
            boolean_gate("step-status", metrics, "successful"),
        ],
    }


def _plan_q6(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.moving-surface",
            meshfree_support(
                dimension=config.dimension,
                method="surface-material-ale-reaction-diffusion",
                boundary="closed-surface",
                geometry_authority="prescribed-expanding-sphere",
                derivative_class="none",
                capacity=4096,
                precision=config.precision,
                temporal_method="additive-imex",
            ),
            size,
            seed,
            "expanding-sphere",
            partial(_q6_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


def _q7_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    import examples.meshfree_bulk_surface_exchange as consumer

    recorder = PhaseRecorder()
    metrics = recorder.run(
        "end-to-end",
        lambda: consumer.run_workflow(
            size=capacity, dimension=config.dimension, seed=seed
        ),
        scope="meshfree_bulk_surface_exchange.run_workflow",
    )
    return {
        **_consumer_record(
            consumer.__file__,
            "examples.meshfree_bulk_surface_exchange.run_workflow",
            metrics,
        ),
        "phases": recorder.record(),
        "gates": [
            upper_gate("analytic", metrics, "equilibrium_error", 1e-4),
            upper_gate("conservation", metrics, "conservation_error", 1e-6),
            upper_gate("window-lag", metrics, "window_lag_error", 1e-4),
            boolean_gate("window-status", metrics, "accepted"),
        ],
    }


def _plan_q7(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.bulk-surface-exchange",
            meshfree_support(
                dimension=config.dimension,
                method="langmuir-bulk-surface-exchange",
                boundary="bulk-surface-interface",
                geometry_authority="sphere-interface-in-box",
                derivative_class="none",
                capacity=4096,
                precision=config.precision,
                temporal_method="explicit-window-buffer",
            ),
            size,
            seed,
            "langmuir",
            partial(_q7_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


def _q8_metrics(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    import examples.meshfree_learned_edge_flux as consumer

    if config.precision != "float64":
        raise ValueError(
            "Learned implicit recovery requires float64; precision is not silently changed."
        )
    neighbors = min(config.neighbors, capacity)
    reservation = config.check_capacity(
        capacity, neighbors=neighbors
    ) + config.check_capacity(capacity + 2 * config.dimension, neighbors=neighbors)
    declare_reservation(
        reservation, config, scope="training and unseen-cloud preparation"
    )

    def compute() -> dict[str, Any]:
        recorder = PhaseRecorder()
        metrics = recorder.run(
            "end-to-end",
            lambda: consumer.run_workflow(
                size=capacity, dimension=config.dimension, seed=seed, steps=config.steps
            ),
            scope="meshfree_learned_edge_flux.run_workflow",
        )
        return {
            **_consumer_record(
                consumer.__file__,
                "examples.meshfree_learned_edge_flux.run_workflow",
                metrics,
            ),
            "phases": recorder.record(),
            "reserved_working_set_bytes": reservation,
        }

    return shared(("Q8", capacity, seed, config), compute)


def _q8_support(config: MeshfreeConfig, /) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method="learned-monotone-edge-flux-native-newton",
        boundary="dirichlet-incomplete-axial-neighborhood",
        geometry_authority="cartesian-nodal-control-volumes",
        derivative_class="implicit-adjoint",
        capacity=1024,
        precision=config.precision,
    )


def _q8_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    record = dict(_q8_metrics(capacity, seed, config))
    metrics = record["metrics"]
    record["gates"] = [
        upper_gate("law-recovery", metrics, "parameter_error", 5e-2),
        upper_gate("implicit-derivative", metrics, "gradient_relative_error", 1e-3),
        boolean_gate("primal-status", metrics, "primal_successful"),
        boolean_gate("adjoint-status", metrics, "adjoint_successful"),
        upper_gate("conservation", metrics, "conservation_defect", 1e-7),
    ]
    return record


def _q8_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = dict(_q8_metrics(capacity, seed, config))
    metrics = record["metrics"]
    record["gates"] = [
        boolean_gate("coverage", metrics, "coverage_refused"),
        boolean_gate("newton-refusal", metrics, "failure_rejected"),
    ]
    return record


def _plan_q8(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.learned-constitutive-flux",
            _q8_support(config),
            size,
            seed,
            "learned-flux",
            partial(_q8_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


def _plan_q8_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.learned-constitutive-flux",
            _q8_support(config),
            size,
            seed,
            "out-of-coverage-and-failed-newton",
            partial(_q8_refusal_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


# ---------------------------------------------------------------------------
# Q9 bulk evolution/transport and Q13 joint transfer/physical topology campaigns.

from benchmarks.meshfree_closure_transport_transfer import (
    BulkTemporalMethod,
    lattice_side,
    measure_bulk_adr_spatial,
    measure_bulk_adr_temporal,
    measure_bulk_ale,
    measure_cfl_refusal,
    measure_graph_closed,
    measure_graph_inflow,
    measure_history_route_refusal,
    measure_merge_rollback,
    measure_point_transfer,
    measure_positivity_refusal,
    measure_support_refusal,
    measure_surface_merge,
    measure_surface_pinch,
    measure_topology_refusal,
    measure_transfer_refusal,
    merge_face_capacity,
    pinch_face_capacity,
    TRANSFER_LEBESGUE_BOUND,
    transfer_routes,
    TransferCase,
    TransferRefusal,
)
from phydrax.discretization.meshfree import TransportScheme


_Q9_TEMPORAL: dict[BulkTemporalMethod, str] = {
    "ssprk33": "ssprk33",
    "ars-222": "imex-ars-222",
}
_Q9_METHODS: tuple[BulkTemporalMethod, ...] = ("ssprk33", "ars-222")
_Q9_SCHEMES: tuple[TransportScheme, ...] = ("upwind", "reconstructed", "limited")
# Declared orders (formal order minus the plan's 0.5 margin, as in Q1):
# degree-3 GMLS, Laplacian truncation O(h^(p+1-2)) = O(h^2) dominates -> 1.5;
# SSPRK33 order 3 -> 2.5; ARS-222 order 2 -> 1.5; donor-cell upwind order 1 -> 0.5;
# unlimited MUSCL order 2 -> 1.5; the limiter reverts to the donor flux at
# extrema, so only the low-order guarantee (1 -> 0.5) is declared for 'limited'.
_Q9_SPATIAL_ORDER = 1.5
_Q9_TEMPORAL_ORDER: dict[BulkTemporalMethod, float] = {"ssprk33": 2.5, "ars-222": 1.5}
_Q9_GRAPH_ORDER: dict[TransportScheme, float] = {
    "upwind": 0.5,
    "reconstructed": 1.5,
    "limited": 0.5,
}
# Content telescopes exactly across graph edges: float64 summation over <= 4096
# nodes and <= 64 steps bounds the explicit drift by ~3e-11 -> 1e-10. Implicit
# stages solve to relative 1e-12 (absolute 1e-14): drift 1e-9 is declared.
_Q9_EXPLICIT_MASS = 1e-10
_Q9_IMPLICIT_MASS = 1e-9
# Certified forward-Euler bounds hold exactly; only roundoff is admitted.
# P16 (Main, TransportP5): the closed swirl's velocity-derived flux is not
# discretely solenoidal, so 'bound-preservation' is measured against the
# compression-aware bound max(c_0) (1 + dt max(-div_h u, 0))^n per step n
# (threshold unchanged); graph inflow uses strong inflow rows and the lattice
# declares boundary_area_vectors; sizes >= 256 resolve the reconstructed order.
_Q9_BOUND_ROUNDOFF = 1e-12


def _q9_collocation_support(
    method: BulkTemporalMethod, config: MeshfreeConfig, /
) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method="gmls-degree3-collocation-adr",
        boundary="periodic",
        geometry_authority="periodic-unit-square-jittered-lattice",
        derivative_class="none",
        capacity=4096,
        precision=config.precision,
        temporal_method=_Q9_TEMPORAL[method],
    )


def _q9_ale_support(config: MeshfreeConfig, /) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method="gmls-degree3-collocation-ale-adr",
        boundary="periodic",
        geometry_authority="periodic-shell-lattice",
        derivative_class="none",
        capacity=4096,
        precision=config.precision,
        temporal_method="ssprk33",
    )


def _q9_graph_support(
    method: str, boundary: str, temporal: str, config: MeshfreeConfig, /
) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method=method,
        boundary=boundary,
        geometry_authority="cartesian-lattice-control-volumes",
        derivative_class="none",
        capacity=4096,
        precision=config.precision,
        temporal_method=temporal,
    )


def _with_gates(row: dict[str, Any], gates: list[GateRecord], /) -> dict[str, Any]:
    return {**row, "gates": gates}


def _q9_spatial_row(
    method: BulkTemporalMethod, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_bulk_adr_spatial(method, capacity, seed, config)
    return _with_gates(
        row, [boolean_gate("rollout-status", row["metrics"], "successful")]
    )


def _q9_temporal_row(
    method: BulkTemporalMethod, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_bulk_adr_temporal(method, capacity, seed, config)
    refinement = row["temporal_refinement"]
    support = SupportTuple(
        "meshfree.bulk-evolution", _q9_collocation_support(method, config)
    )
    return _with_gates(
        row,
        [
            boolean_gate("rollout-status", row["metrics"], "successful"),
            rate_gate(
                "temporal-order",
                refinement["step_sizes"],
                refinement["errors"],
                _Q9_TEMPORAL_ORDER[method],
                metric=f"{method}-max-error-temporal-order",
                provenance="declared step sizes T/n of the fixed-step rollout",
                support_tuple_id=support.support_tuple_id,
                coordinates={"seed": seed, "capacity": row["capacity"], "method": method},
            ),
        ],
    )


def _q9_ale_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    row = measure_bulk_ale(capacity, seed, config)
    metrics = row["metrics"]
    # Translation GCL is exact (zero discrete divergence of a uniform mesh
    # velocity, exact position update); the material-minus-mesh rate is an
    # algebraic identity. Only float64 roundoff is admitted.
    return _with_gates(
        row,
        [
            upper_gate("ale-gcl-free-stream", metrics, "gcl_free_stream_error", 1e-12),
            upper_gate("ale-gcl-volume", metrics, "gcl_volume_relative_error", 1e-12),
            upper_gate("ale-gcl-position", metrics, "gcl_position_error", 1e-12),
            upper_gate(
                "ale-material-minus-mesh",
                metrics,
                "material_minus_mesh_rate_error",
                1e-10,
            ),
            boolean_gate("rollout-status", metrics, "motion_successful"),
            boolean_gate("rebase-continuation", metrics, "rebased_step_accepted"),
        ],
    )


def _q9_inflow_row(
    scheme: TransportScheme, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_graph_inflow(scheme, capacity, seed, config)
    metrics = row["metrics"]
    gates = [
        boolean_gate("rollout-status", metrics, "successful"),
        boolean_gate("metric-admission", metrics, "metric_accepted"),
        upper_gate(
            "conservation-ledger", metrics, "maximum_ledger_residual", _Q9_EXPLICIT_MASS
        ),
        # 'reconstructed' declares no positivity certificate; the others do.
        boolean_gate(
            "cfl-certificate",
            metrics,
            "cfl_certified",
            expected=scheme != "reconstructed",
        ),
    ]
    if scheme != "reconstructed":
        gates.append(
            lower_gate(
                "positivity", metrics, "minimum_concentration", -_Q9_BOUND_ROUNDOFF
            )
        )
    return _with_gates(row, gates)


def _q9_closed_row(
    method: BulkTemporalMethod, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_graph_closed(method, capacity, seed, config)
    metrics = row["metrics"]
    gates = [
        boolean_gate("rollout-status", metrics, "successful"),
        boolean_gate("metric-admission", metrics, "metric_accepted"),
    ]
    match method:
        case "ssprk33":
            gates += [
                upper_gate(
                    "mass-conservation", metrics, "relative_mass_drift", _Q9_EXPLICIT_MASS
                ),
                lower_gate(
                    "positivity", metrics, "minimum_concentration", -_Q9_BOUND_ROUNDOFF
                ),
                upper_gate(
                    "bound-preservation", metrics, "maximum_overshoot", _Q9_BOUND_ROUNDOFF
                ),
            ]
        case "ars-222":
            # An IMEX method inherits no positivity certificate; only mass is declared.
            gates.append(
                upper_gate(
                    "mass-conservation", metrics, "relative_mass_drift", _Q9_IMPLICIT_MASS
                )
            )
    return _with_gates(row, gates)


def _plan_q9(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    smallest = min(config.sizes)
    rows: list[PlannedRow] = []
    for seed in config.seeds:
        for method in _Q9_METHODS:
            support = _q9_collocation_support(method, config)
            rows += [
                PlannedRow(
                    "meshfree.bulk-evolution",
                    support,
                    lattice_side(size, 8) ** 2,
                    seed,
                    f"adr-spatial-{method}",
                    partial(_q9_spatial_row, method, size, seed, config),
                )
                for size in config.sizes
            ]
            rows.append(
                PlannedRow(
                    "meshfree.bulk-evolution",
                    support,
                    lattice_side(smallest, 8) ** 2,
                    seed,
                    f"adr-temporal-{method}",
                    partial(_q9_temporal_row, method, smallest, seed, config),
                )
            )
        rows.append(
            PlannedRow(
                "meshfree.bulk-evolution",
                _q9_ale_support(config),
                lattice_side(smallest, 8) ** 2,
                seed,
                "ale-seam-motion",
                partial(_q9_ale_row, smallest, seed, config),
            )
        )
        for scheme in _Q9_SCHEMES:
            rows += [
                PlannedRow(
                    "meshfree.bulk-transport",
                    _q9_graph_support(
                        f"edge-graph-{scheme}", "inflow-outflow", "ssprk33", config
                    ),
                    lattice_side(size, 9) ** 2,
                    seed,
                    f"graph-inflow-{scheme}",
                    partial(_q9_inflow_row, scheme, size, seed, config),
                )
                for size in config.sizes
            ]
        rows += [
            PlannedRow(
                "meshfree.bulk-transport",
                _q9_graph_support(
                    "edge-graph-limited", "closed-tangential", "ssprk33", config
                ),
                lattice_side(size, 9) ** 2,
                seed,
                "graph-closed-limited-ssprk33",
                partial(_q9_closed_row, "ssprk33", size, seed, config),
            )
            for size in config.sizes
        ]
        rows.append(
            PlannedRow(
                "meshfree.bulk-transport",
                _q9_graph_support(
                    "edge-graph-limited-graph-diffusion",
                    "closed-tangential",
                    "imex-ars-222",
                    config,
                ),
                lattice_side(smallest, 9) ** 2,
                seed,
                "graph-closed-limited-ars222",
                partial(_q9_closed_row, "ars-222", smallest, seed, config),
            )
        )
    return tuple(rows)


def _rate_groups(
    rows: Sequence[RowRecord], prefix: str, /
) -> dict[tuple[str, int, str], list[RowRecord]]:
    """Executed rows of one label family, grouped by tuple, seed and label."""
    groups: dict[tuple[str, int, str], list[RowRecord]] = {}
    for row in rows:
        if row["label"].startswith(prefix) and "metrics" in row:
            groups.setdefault(
                (row["support_tuple_id"], row["seed"], row["label"]), []
            ).append(row)
    return {
        key: sorted(value, key=lambda row: row["capacity"])
        for key, value in groups.items()
    }


def _q9_convergence(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /
) -> list[GateRecord]:
    del config
    gates: list[GateRecord] = []
    for (support_id, seed, label), selected in sorted(
        _rate_groups(rows, "adr-spatial-").items()
    ):
        gates.append(
            rate_gate(
                "convergence-spatial",
                [row["metrics"]["fill_distance"] for row in selected],
                [row["metrics"]["max_error"] for row in selected],
                _Q9_SPATIAL_ORDER,
                metric="adr-max-error-order-in-measured-fill-distance",
                provenance="measured fill distance (scrambled Sobol probes; non-periodic nearest)",
                support_tuple_id=support_id,
                coordinates={
                    "seed": seed,
                    "label": label,
                    "capacities": [row["capacity"] for row in selected],
                },
            )
        )
    for (support_id, seed, label), selected in sorted(
        _rate_groups(rows, "graph-inflow-").items()
    ):
        scheme = parse(label.removeprefix("graph-inflow-"), TransportScheme, "scheme")
        gates.append(
            rate_gate(
                "convergence-l1",
                [row["metrics"]["fill_distance"] for row in selected],
                [row["metrics"]["l1_error"] for row in selected],
                _Q9_GRAPH_ORDER[scheme],
                metric=f"{scheme}-pulse-l1-order-at-fixed-cfl",
                provenance="measured lattice fill distance; space-time refinement at the "
                "scheme's forward-Euler step bound",
                support_tuple_id=support_id,
                coordinates={
                    "seed": seed,
                    "label": label,
                    "capacities": [row["capacity"] for row in selected],
                },
            )
        )
    return gates


def _q9_graph_refusal_row(
    measure: Callable[[int, int, MeshfreeConfig], dict[str, Any]],
    certificate_expected: bool,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    row = measure(capacity, seed, config)
    metrics = row["metrics"]
    first = (
        boolean_gate("cfl-refusal", metrics, "cfl_admitted", expected=False)
        if certificate_expected
        else boolean_gate("cfl-certificate", metrics, "cfl_certified", expected=False)
    )
    return _with_gates(
        row,
        [
            first,
            boolean_gate("step-refusal", metrics, "step_refused"),
            boolean_gate("negative-state-status", metrics, "negative_state_status"),
            boolean_gate("candidate-kept-raw", metrics, "candidate_kept_raw"),
            boolean_gate("state-held", metrics, "accepted_state_held"),
            boolean_gate("rollback", metrics, "rolled_back"),
        ],
    )


def _q9_support_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_support_refusal(capacity, seed, config)
    metrics = row["metrics"]
    return _with_gates(
        row,
        [
            boolean_gate("step-refusal", metrics, "step_refused"),
            boolean_gate("support-refusal-status", metrics, "support_exceeded_status"),
            boolean_gate("state-held", metrics, "accepted_state_held"),
            boolean_gate("rollback", metrics, "rolled_back"),
        ],
    )


def _q9_topology_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_topology_refusal(capacity, seed, config)
    metrics = row["metrics"]
    return _with_gates(
        row,
        [
            boolean_gate("refresh-refused", metrics, "refresh_refused"),
            boolean_gate("topology-refusal-status", metrics, "topology_trust_status"),
            boolean_gate("owner-unchanged", metrics, "owner_unchanged"),
        ],
    )


def _plan_q9_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    rows: list[PlannedRow] = []
    for size in config.sizes:
        for seed in config.seeds:
            graph = lattice_side(size, 9) ** 2
            rows += [
                PlannedRow(
                    "meshfree.bulk-transport",
                    _q9_graph_support(
                        "edge-graph-upwind", "closed-tangential", "ssprk33", config
                    ),
                    graph,
                    seed,
                    "cfl-admission-refusal",
                    partial(
                        _q9_graph_refusal_row,
                        measure_cfl_refusal,
                        True,
                        size,
                        seed,
                        config,
                    ),
                ),
                PlannedRow(
                    "meshfree.bulk-transport",
                    _q9_graph_support(
                        "edge-graph-reconstructed", "closed-tangential", "ssprk33", config
                    ),
                    graph,
                    seed,
                    "positivity-refusal",
                    partial(
                        _q9_graph_refusal_row,
                        measure_positivity_refusal,
                        False,
                        size,
                        seed,
                        config,
                    ),
                ),
                PlannedRow(
                    "meshfree.bulk-transport",
                    _q9_graph_support(
                        "edge-graph-fixed-topology-refresh",
                        "lattice-boundary-nodes",
                        "none",
                        config,
                    ),
                    graph,
                    seed,
                    "topology-trust-refusal",
                    partial(_q9_topology_refusal_row, size, seed, config),
                ),
                PlannedRow(
                    "meshfree.bulk-evolution",
                    _q9_ale_support(config),
                    lattice_side(size, 8) ** 2,
                    seed,
                    "support-trust-refusal",
                    partial(_q9_support_refusal_row, size, seed, config),
                ),
            ]
    return tuple(rows)


_Q13_CASES: tuple[TransferCase, ...] = (
    "conservative-signed",
    "conservative-positive",
    "joint-nonnegative-constant",
    "joint-signed-degree2",
)
# Audited transfer tolerance (PointTransferPlan default, relative per source);
# the independent host recomputation adds float64 summation over <= 4e4 terms.
_Q13_AUDIT = 1e-10
_Q13_INDEPENDENT = 1e-9
# Frozen linear operator and its coordinate dual: pairing roundoff only.
_Q13_DUAL = 1e-11
# A constant(quadratic)-exact remap of a smooth field errs O(h) (O(h^3)).
_Q13_REMAP_ORDER: dict[TransferCase, float] = {
    "joint-nonnegative-constant": 0.5,
    "joint-signed-degree2": 2.5,
}
# Physical events reuse the authority's conservative transfer, audited at the
# declared transfer tolerance.
_Q13_EVENT = 1e-10
_Q13_REFUSALS: tuple[TransferRefusal, ...] = (
    "uncovered-source",
    "uncovered-target",
    "measure-obstruction",
    "infeasible-left-null",
    "infeasible-farkas",
)
_Q13_REFUSAL_SUPPORT: dict[TransferRefusal, tuple[str, str]] = {
    "uncovered-source": (
        "point-transfer-conservative-signed",
        "tensor-simpson-unit-cube-3x-refined-source",
    ),
    "uncovered-target": (
        "point-transfer-joint-radius-routes",
        "tensor-simpson-unit-cube-with-exterior-target",
    ),
    "measure-obstruction": (
        "point-transfer-joint-nonnegative-constant",
        "tensor-simpson-unit-cube-dilated-target-measure",
    ),
    "infeasible-left-null": (
        "point-transfer-joint-signed-degree2",
        "tensor-midpoint-unit-cube",
    ),
    "infeasible-farkas": (
        "point-transfer-joint-nonnegative-degree1",
        "tensor-midpoint-unit-cube",
    ),
}


def _q13_transfer_support(
    method: str, geometry: str, config: MeshfreeConfig, /
) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=config.dimension,
        method=method,
        boundary="none",
        geometry_authority=geometry,
        derivative_class="frozen-transfer-value",
        capacity=65536,
        precision=config.precision,
    )


def _q13_event_support(
    kind: Literal["merge", "pinch"], config: MeshfreeConfig, /
) -> dict[str, SupportValue]:
    match kind:
        case "merge":
            method, boundary, geometry = (
                "multiregion-merge-meshfree-epoch",
                "closed-foam-films",
                "multiregion-icosphere-bubbles",
            )
        case "pinch":
            method, boundary, geometry = (
                "multiregion-pinch-region-split-meshfree-epoch",
                "fixed-ring-boundary",
                "multiregion-catenoid",
            )
    return meshfree_support(
        dimension=config.dimension,
        method=method,
        boundary=boundary,
        geometry_authority=geometry,
        derivative_class="frozen-transfer-value",
        capacity=8192,
        precision=config.precision,
    )


def _q13_transfer_row(
    case: TransferCase, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_point_transfer(case, capacity, seed, config)
    metrics = row["metrics"]
    gates = [
        boolean_gate("admission", metrics, "admitted"),
        upper_gate(
            "conservation-audit", metrics, "audited_conservation_residual", _Q13_AUDIT
        ),
        upper_gate(
            "conservation-independent",
            metrics,
            "independent_content_defect",
            _Q13_INDEPENDENT,
        ),
        upper_gate("dual-identity", metrics, "dual_identity_residual", _Q13_DUAL),
        upper_gate(
            "bounded-amplification", metrics, "lebesgue_constant", TRANSFER_LEBESGUE_BOUND
        ),
    ]
    if case in ("conservative-positive", "joint-nonnegative-constant"):
        gates.append(
            lower_gate("positivity", metrics, "minimum_coefficient", -_Q13_AUDIT)
        )
    match case:
        case "joint-nonnegative-constant":
            gates.append(
                upper_gate(
                    "constant-reproduction",
                    metrics,
                    "constant_reproduction_error",
                    _Q13_INDEPENDENT,
                )
            )
        case "joint-signed-degree2":
            gates += [
                upper_gate(
                    "constant-reproduction",
                    metrics,
                    "constant_reproduction_error",
                    _Q13_INDEPENDENT,
                ),
                upper_gate(
                    "moment-reproduction",
                    metrics,
                    "quadratic_reproduction_error",
                    _Q13_INDEPENDENT,
                ),
            ]
        case "conservative-signed" | "conservative-positive":
            pass
    return _with_gates(row, gates)


def _q13_merge_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    row = measure_surface_merge(capacity, seed, config)
    metrics = row["metrics"]
    return _with_gates(
        row,
        [
            boolean_gate("epoch-published", metrics, "published"),
            boolean_gate("continuation-published", metrics, "continuation_published"),
            boolean_gate("physical-lineage", metrics, "merge_in_lineage"),
            boolean_gate("ccd-certified", metrics, "ccd_certified"),
            boolean_gate("all-history-remap", metrics, "every_history_remapped"),
            boolean_gate("receipt-ledger", metrics, "receipt_residuals_within_tolerance"),
            boolean_gate("controller-carried", metrics, "controller_carried"),
            upper_gate("content-conservation", metrics, "content_drift", _Q13_EVENT),
            upper_gate("history-conservation", metrics, "history_drift", _Q13_EVENT),
            upper_gate("authority-agreement", metrics, "authority_mismatch", _Q13_EVENT),
            boolean_gate("value-derivative", metrics, "value_jvp_finite"),
            upper_gate("value-vjp", metrics, "value_vjp_matches_pullback", 1e-12),
        ],
    )


def _q13_pinch_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    row = measure_surface_pinch(capacity, seed, config)
    metrics = row["metrics"]
    return _with_gates(
        row,
        [
            boolean_gate("physical-lineage", metrics, "pinched"),
            boolean_gate("epoch-published", metrics, "every_epoch_published"),
            boolean_gate("region-split-lineage", metrics, "core_split_into_children"),
            boolean_gate("region-removed", metrics, "core_removed"),
            boolean_gate("receipt-ledger", metrics, "receipt_residuals_within_tolerance"),
            upper_gate("content-conservation", metrics, "content_drift", _Q13_EVENT),
            upper_gate("history-conservation", metrics, "history_drift", _Q13_EVENT),
        ],
    )


def _q13_rollback_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_merge_rollback(capacity, seed, config)
    metrics = row["metrics"]
    return _with_gates(
        row,
        [
            boolean_gate("rollback-refused", metrics, "epoch_refused"),
            boolean_gate(
                "rollback-failed-history", metrics, "failed_only_poisoned_history"
            ),
            boolean_gate(
                "rollback-source-returned", metrics, "source_composition_returned"
            ),
            boolean_gate("rollback-values-unchanged", metrics, "current_field_unchanged"),
            boolean_gate(
                "rollback-derivative-withheld", metrics, "value_derivative_withheld"
            ),
        ],
    )


def _q13_route_neighbors(
    config: MeshfreeConfig, case: TransferCase, /
) -> tuple[int, ...]:
    """Vary admitted route widths independently of point capacity.

    Joint quadratic reproduction and global conservation require more support
    than row-local unisolvency. The smaller infeasible/over-amplifying routes
    remain separate refusal scenarios rather than accepted-target measurements.
    """
    basis = comb(config.dimension + 2, 2)
    if case == "joint-signed-degree2":
        return (4 * basis, 5 * basis, 6 * basis)
    return (2 * basis, 3 * basis, 4 * basis)


def _q13_size_neighbors(config: MeshfreeConfig, /) -> int:
    # Four times the quadratic basis keeps the joint degree-2 routes'
    # amplification (Lebesgue constant) bounded across the size sweep.
    return max(config.neighbors, 4 * comb(config.dimension + 2, 2))


def _plan_q13(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    rows: list[PlannedRow] = []
    middle = sorted(config.sizes)[(len(config.sizes) - 1) // 2]
    sized = replace(config, neighbors=_q13_size_neighbors(config))
    for seed in config.seeds:
        for case in _Q13_CASES:
            support = _q13_transfer_support(
                f"point-transfer-{case}", "tensor-simpson-unit-cube", config
            )
            rows += [
                PlannedRow(
                    "meshfree.joint-transfer",
                    support,
                    transfer_routes(size, config.dimension, sized.neighbors),
                    seed,
                    f"size-sweep-{case}",
                    partial(_q13_transfer_row, case, size, seed, sized),
                )
                for size in config.sizes
            ]
            # Edge (route) capacity varies independently of the point count.
            for neighbors in _q13_route_neighbors(config, case):
                swept = replace(config, neighbors=neighbors)
                rows.append(
                    PlannedRow(
                        "meshfree.joint-transfer",
                        support,
                        transfer_routes(middle, config.dimension, neighbors),
                        seed,
                        f"route-sweep-{case}-k{neighbors}",
                        partial(_q13_transfer_row, case, middle, seed, swept),
                    )
                )
        if config.dimension != 3:
            continue
        for level in (1, 2):
            rows.append(
                PlannedRow(
                    "meshfree.physical-topology",
                    _q13_event_support("merge", config),
                    merge_face_capacity(level),
                    seed,
                    f"merge-s{level}",
                    partial(_q13_merge_row, merge_face_capacity(level), seed, config),
                )
            )
        for pinch_rows in (8, 10):
            rows.append(
                PlannedRow(
                    "meshfree.physical-topology",
                    _q13_event_support("pinch", config),
                    pinch_face_capacity(pinch_rows),
                    seed,
                    f"pinch-rows{pinch_rows}",
                    partial(
                        _q13_pinch_row, pinch_face_capacity(pinch_rows), seed, config
                    ),
                )
            )
        rows.append(
            PlannedRow(
                "meshfree.physical-topology",
                _q13_event_support("merge", config),
                merge_face_capacity(1),
                seed,
                "merge-all-history-rollback",
                partial(_q13_rollback_row, merge_face_capacity(1), seed, config),
            )
        )
    return tuple(rows)


def _plan_q13_signed(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    """Replay only the revised signed-quadratic support envelope."""
    return tuple(
        row
        for row in _plan_q13(config)
        if row.label == "size-sweep-joint-signed-degree2"
        or row.label.startswith("route-sweep-joint-signed-degree2-")
    )


def _q13_convergence(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /
) -> list[GateRecord]:
    del config
    gates: list[GateRecord] = []
    for (support_id, seed, label), selected in sorted(
        _rate_groups(rows, "size-sweep-joint-").items()
    ):
        case = parse(label.removeprefix("size-sweep-"), TransferCase, "case")
        gates.append(
            rate_gate(
                "convergence-remap",
                [row["metrics"]["fill_distance"] for row in selected],
                # A refused transfer has no remap; rate_gate drops that resolution
                # and reports the order unexecuted below four measured ones.
                [row["metrics"].get("remap_max_error") for row in selected],
                _Q13_REMAP_ORDER[case],
                metric=f"{case}-smooth-remap-max-error-order",
                provenance="measured source fill distance (scrambled Sobol probes)",
                support_tuple_id=support_id,
                coordinates={
                    "seed": seed,
                    "label": label,
                    "capacities": [row["capacity"] for row in selected],
                },
            )
        )
    return gates


def _q13_route_refusal_row(
    expected: PointTransferStatus,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    row = measure_point_transfer("joint-signed-degree2", capacity, seed, config)
    metrics = row["metrics"]
    metrics["expected_status"] = metrics["status"] == int(expected)
    metrics["transfer_withheld"] = not metrics["admitted"]
    gates = [
        boolean_gate("declared-status", metrics, "expected_status"),
        boolean_gate("transfer-withheld", metrics, "transfer_withheld"),
    ]
    match expected:
        case PointTransferStatus.INFEASIBLE:
            margin = _number(metrics, "witness_margin")
            metrics["witness_margin_magnitude"] = None if margin is None else abs(margin)
            gates.extend(
                (
                    upper_gate("witness-residual", metrics, "witness_residual", 1e-8),
                    lower_gate(
                        "witness-margin",
                        metrics,
                        "witness_margin_magnitude",
                        10 * _Q13_AUDIT,
                    ),
                )
            )
        case PointTransferStatus.AMPLIFICATION_EXCEEDED:
            gates.append(
                lower_gate(
                    "declared-amplification-limit", metrics, "lebesgue_constant", 10.0
                )
            )
        case _:
            raise ValueError("Unsupported joint-transfer route refusal scenario.")
    return _with_gates(row, gates)


def _q13_refusal_row(
    case: TransferRefusal, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_transfer_refusal(case, capacity, seed, config)
    metrics = row["metrics"]
    gates = [
        boolean_gate("declared-status", metrics, "expected_status"),
        boolean_gate("transfer-withheld", metrics, "transfer_withheld"),
    ]
    match case:
        case "uncovered-source":
            gates.append(
                lower_gate("coverage-witness", metrics, "uncovered_source_count", 1.0)
            )
        case "uncovered-target":
            gates.append(
                lower_gate("coverage-witness", metrics, "uncovered_target_count", 1.0)
            )
        case "measure-obstruction" | "infeasible-left-null" | "infeasible-farkas":
            # A certified witness annihilates every declared row (residual at the
            # provider tolerance scale) and separates the right side by more
            # than ten audit tolerances.
            gates += [
                upper_gate("witness-residual", metrics, "witness_residual", 1e-8),
                lower_gate(
                    "witness-margin", metrics, "witness_margin_magnitude", 10 * _Q13_AUDIT
                ),
            ]
    return _with_gates(row, gates)


def _q13_history_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    row = measure_history_route_refusal(capacity, seed, config)
    return _with_gates(
        row,
        [
            boolean_gate(
                "history-route-refusal", row["metrics"], "refused_naming_missing_history"
            )
        ],
    )


def _plan_q13_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    rows: list[PlannedRow] = []
    for size in config.sizes:
        for seed in config.seeds:
            for case in _Q13_REFUSALS:
                method, geometry = _Q13_REFUSAL_SUPPORT[case]
                # Routes are bounded by targets times neighbors (plus one exterior target).
                routes = (
                    transfer_routes(size, config.dimension, config.neighbors)
                    + config.neighbors
                )
                rows.append(
                    PlannedRow(
                        "meshfree.joint-transfer",
                        _q13_transfer_support(method, geometry, config),
                        routes,
                        seed,
                        f"{case}-refusal",
                        partial(_q13_refusal_row, case, size, seed, config),
                    )
                )
    if config.dimension == 3:
        # These exact supports have independently observed incompatibility and
        # amplification refusals; they are not accepted-target measurements.
        basis = comb(config.dimension + 2, 2)
        for seed in config.seeds:
            for multiple, expected in (
                (2, PointTransferStatus.INFEASIBLE),
                (3, PointTransferStatus.AMPLIFICATION_EXCEEDED),
            ):
                swept = replace(config, neighbors=multiple * basis)
                rows.append(
                    PlannedRow(
                        "meshfree.joint-transfer",
                        _q13_transfer_support(
                            "point-transfer-joint-signed-degree2",
                            "tensor-simpson-unit-cube",
                            config,
                        ),
                        transfer_routes(343, config.dimension, swept.neighbors),
                        seed,
                        f"joint-quadratic-route-k{swept.neighbors}-refusal",
                        partial(_q13_route_refusal_row, expected, 343, seed, swept),
                    )
                )
        rows += [
            PlannedRow(
                "meshfree.physical-topology",
                _q13_event_support("merge", config),
                merge_face_capacity(1),
                seed,
                "missing-history-route-refusal",
                partial(_q13_history_refusal_row, merge_face_capacity(1), seed, config),
            )
            for seed in config.seeds
        ]
    return tuple(rows)


# ---------------------------------------------------------------------------
# Q10 incompressible/Lagrangian flow and Q11 mechanics/surface Stokes/FSI campaigns.

from benchmarks.meshfree_closure_flow_mechanics import (
    FLOW_MECHANICS_WORKLOADS,
    MIXED_COMPRESSIBILITIES,
)


type FlowGates = Callable[[dict[str, Any]], list[GateRecord]]

_PLATE = "unit-square-jittered-lattice-trapezoid-measure"
# Checkerboard index h^2 ||lap_h p|| / ||p - mean p|| of the mixed owners: a
# smooth manufactured pressure cos(pi x) cos(pi y) gives ~ 2 pi^2 h^2 <= 0.31
# for h <= 1/8, an odd-even mode ~ 8 (|lap_h| ~ 8 / h^2); 1 separates them.
_CHECKERBOARD_INDEX = 1.0


def _flow_support(
    method: str,
    boundary: str,
    geometry_authority: str,
    capacity: int,
    /,
    *,
    derivative_class: str = "none",
    temporal_method: str = "none",
    dimension: int = 2,
) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=dimension,
        method=method,
        boundary=boundary,
        geometry_authority=geometry_authority,
        derivative_class=derivative_class,
        capacity=capacity,
        precision="float64",
        temporal_method=temporal_method,
    )


# Exact Q10/Q11 support tuples (declared in meshfree._profiles; capacities are
# the largest admitted controlling point counts of each route).
_TAYLOR_GREEN = _flow_support(
    "exterior-projection-reconstructed-transport-phs3",
    "periodic",
    "periodic-square-lattice-minimum-image-charts",
    1024,
    derivative_class="pressure-projection-jvp-vjp",
    temporal_method="ars-222",
)
_BOUNDED_STOKES = _flow_support(
    "generalized-stokes-pspg-phs3-collocated", "dirichlet", _PLATE, 1024
)
_LAGRANGIAN_VOLUME = _flow_support(
    "lagrangian-gmls-weak-projection-quadrature-volume",
    "periodic",
    "periodic-unit-square-cell-lattice",
    1024,
    temporal_method="explicit-predictor-projection",
)
_LAGRANGIAN_MATERIAL = _flow_support(
    "lagrangian-gmls-weak-projection-material-mass-sph-wendland-c2",
    "periodic",
    "periodic-unit-square-cell-lattice",
    1024,
    temporal_method="explicit-predictor-projection",
)
_MEASURE_TRANSFER = _flow_support(
    "measure-transfer-conservative-positive-material-to-quadrature",
    "periodic",
    "periodic-unit-square-jittered-cell-lattice",
    1024,
)
# Declared to the 65^2-point lattice of Q11-elasticity-order (four resolutions).
_SMALL_STRAIN_TRACTION = _flow_support(
    "small-strain-phs3-collocated-gmres-auxiliary",
    "traction-face-dirichlet",
    _PLATE,
    4225,
)
_SMALL_STRAIN_ROLLERS = _flow_support(
    "small-strain-phs3-collocated-gmres-auxiliary", "rollers-traction", _PLATE, 1024
)
_NEO_HOOKEAN = _flow_support(
    "neo-hookean-incremental-newton-phs3",
    "rollers-traction",
    _PLATE,
    1024,
    temporal_method="incremental-load-steps",
)
_HERRMANN = _flow_support("herrmann-mixed-pspg-phs3", "dirichlet-clamped", _PLATE, 1024)
_FLUID_STRUCTURE = _flow_support(
    "monolithic-mortar-vector-transmission-phs3-ghost-dense-lu",
    "paired-interface-dirichlet-walls",
    "polygon-chart-authorized-adjacent-rectangles",
    1024,
    temporal_method="implicit-quasi-static-step",
)
_FLUID_STRUCTURE_MANUFACTURED = _flow_support(
    "monolithic-mortar-vector-transmission-phs3-ghost-dense-lu",
    "paired-interface-manufactured-dirichlet-walls",
    "polygon-chart-authorized-adjacent-rectangles",
    2178,
    temporal_method="implicit-quasi-static-step",
)
_SURFACE_STOKES = _flow_support(
    "surface-strain-form-stokes-brinkman-gmls-degree-4",
    "closed-surface",
    "sphere-exact-implicit-normals",
    4096,
    dimension=3,
)


def _flow_row(
    workload: str, gates: FlowGates, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    """Execute one workload once per (capacity, seed, config); gate its metrics."""
    record = dict(
        shared(
            ("flow-mechanics", workload, capacity, seed, config),
            lambda: FLOW_MECHANICS_WORKLOADS[workload](capacity, seed, config),
        )
    )
    record["gates"] = gates(record["metrics"])
    return record


def _flow_rows(
    capability: str,
    support: dict[str, SupportValue],
    label: str,
    workload: str,
    gates: FlowGates,
    config: MeshfreeConfig,
    /,
) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            capability,
            support,
            size,
            seed,
            label,
            partial(_flow_row, workload, gates, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


def _flow_rates(
    rows: Sequence[RowRecord],
    config: MeshfreeConfig,
    label: str,
    orders: tuple[tuple[str, str, float], ...],
    provenance: str,
    /,
) -> list[GateRecord]:
    """Least-squares orders against measured fill distance, per support tuple and seed.

    Every executed row contributes its measurement (a failed sanity gate does not
    invalidate a measured error); rows without a measurement are omitted and the
    order stays unexecuted below four resolutions.
    """
    gates: list[GateRecord] = []
    scoped = [row for row in rows if row["label"] == label]
    for support_id in sorted({row["support_tuple_id"] for row in scoped}):
        for seed in config.seeds:
            selected = sorted(
                (
                    row
                    for row in scoped
                    if row["support_tuple_id"] == support_id
                    and row["seed"] == seed
                    and "metrics" in row
                ),
                key=lambda row: row["capacity"],
            )
            for name, key, minimum in orders:
                gates.append(
                    rate_gate(
                        name,
                        [row["metrics"]["fill_distance"] for row in selected],
                        [row["metrics"].get(key) for row in selected],
                        minimum,
                        metric=f"{key}-order",
                        provenance=provenance,
                        support_tuple_id=support_id,
                        coordinates={
                            "seed": seed,
                            "capacities": [row["capacity"] for row in selected],
                        },
                    )
                )
    return gates


# --- Q10 accepted gates. Projection/conservation tolerances follow the owners'
# native solve tolerances (pressure/GMRES relative 1e-10); conservation that is
# exact by construction is held to round-off. "resolution" gates are sanity
# bounds (an O(1) relative error is no approximation), never accuracy claims:
# accuracy is claimed only by the observed-order gates below.


def _taylor_green_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("steps-accepted", metrics, "all_steps_accepted"),
        # Each step ends with a native projection solved to ~1e-10 relative
        # on an O(1) velocity: the committed graph divergence is solve error.
        upper_gate("graph-divergence", metrics, "max_graph_divergence", 1e-10),
        # Conservative edge transport and the mean-free graph gradient leave
        # total momentum of the zero-mean vortex at round-off.
        upper_gate("momentum", metrics, "max_momentum", 1e-10),
        upper_gate("mass", metrics, "relative_mass_defect", 1e-12),
        # Unforced viscous flow cannot gain kinetic energy.
        boolean_gate("energy-dissipation", metrics, "energy_nonincreasing"),
        boolean_gate(
            "physical-cycle-content", metrics, "smooth_flow_classified_physical"
        ),
        # A random graph-solenoidal field is mostly artificial cycle content.
        boolean_gate(
            "spurious-cycle-detection",
            metrics,
            "noise_classified_physical",
            expected=False,
        ),
        # P(Pu) = Pu up to the projection's solve tolerance.
        upper_gate(
            "projection-idempotence", metrics, "projection_idempotence_defect", 1e-8
        ),
        upper_gate("energy-resolution", metrics, "relative_energy_error", 0.1),
    ]


def _bounded_stokes_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("accepted", metrics, "accepted"),
        boolean_gate("original-residual", metrics, "residual_within_tolerance"),
        upper_gate(
            "pressure-oscillation", metrics, "pressure_oscillation", _CHECKERBOARD_INDEX
        ),
        upper_gate("velocity-resolution", metrics, "velocity_relative_error", 0.1),
    ]


def _lagrangian_volume_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        # The weak projection annihilates D_h u to the GMRES tolerance (1e-10).
        upper_gate(
            "projection-divergence", metrics, "projection_weak_divergence_ratio", 1e-8
        ),
        upper_gate(
            "projection-gradient-removal",
            metrics,
            "projection_removed_gradient_error",
            1e-6,
        ),
        # M-orthogonality: the projection removes exactly the gradient energy.
        upper_gate(
            "projection-energy", metrics, "projection_energy_relative_error", 1e-8
        ),
        boolean_gate("steps-accepted", metrics, "all_steps_accepted"),
        # Material masses never change: exact mass ledger.
        upper_gate("mass", metrics, "max_relative_mass_defect", 1e-14),
        # Starting from a projected vortex the predicted divergence is already
        # small, so a reduction ratio is meaningless: the committed weak
        # divergence is GMRES error (relative 1e-10) on the O(k |u|) = O(2 pi)
        # divergence scale, declared with three orders of margin.
        upper_gate("step-divergence", metrics, "max_weak_divergence_after", 1e-7),
        boolean_gate(
            "projection-energy-nonincreasing", metrics, "projection_energy_nonincreasing"
        ),
        # sum_i V_i D_h u_i = -(G 1)^T M u = 0 since GMLS reproduces constants.
        upper_gate("volume-gcl", metrics, "max_relative_volume_defect", 1e-12),
    ]


def _lagrangian_material_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    # SPH/GMLS lattice tolerances are the owner's declared contract: Wendland
    # C2 at h = 1.3 dx carries a ~1% partition-of-unity defect, SPH divergence
    # is first order, degree-3 GMLS resolves the 12-point wavelength to 1%.
    return [
        boolean_gate("sph-neighbors", metrics, "sph_neighbors_successful"),
        upper_gate("sph-density", metrics, "sph_partition_density_mismatch", 2e-2),
        upper_gate("gmls-divergence", metrics, "gmls_divergence_relative_error", 1e-2),
        upper_gate("sph-divergence", metrics, "sph_divergence_relative_error", 0.15),
        upper_gate("conversion-mass", metrics, "conversion_mass_defect", 1e-14),
        boolean_gate("conversion-masses", metrics, "conversion_masses_unchanged"),
        boolean_gate("material-steps-accepted", metrics, "material_steps_accepted"),
        boolean_gate("material-mass", metrics, "material_mass_exact"),
        upper_gate("material-volume", metrics, "material_sph_volume_mismatch", 1e-12),
    ]


def _measure_transfer_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    # 1e-10 is the declared audit tolerance of MeshfreeMeasureTransferPlan.
    return [
        boolean_gate("admitted", metrics, "transfer_admitted"),
        boolean_gate("positive", metrics, "transfer_positive"),
        upper_gate("mass", metrics, "relative_mass_defect", 1e-10),
        upper_gate("momentum", metrics, "momentum_defect", 1e-10),
    ]


def _plan_q10(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return (
        *_flow_rows(
            "meshfree.incompressible-flow",
            _TAYLOR_GREEN,
            "taylor-green",
            "incompressible-taylor-green",
            _taylor_green_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.incompressible-flow",
            _BOUNDED_STOKES,
            "bounded-stokes",
            "incompressible-stokes-bounded",
            _bounded_stokes_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.lagrangian-flow",
            _LAGRANGIAN_VOLUME,
            "quadrature-volume",
            "lagrangian-quadrature-volume",
            _lagrangian_volume_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.lagrangian-flow",
            _LAGRANGIAN_MATERIAL,
            "material-mass-sph",
            "lagrangian-material-mass-sph",
            _lagrangian_material_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.lagrangian-flow",
            _MEASURE_TRANSFER,
            "measure-transfer",
            "lagrangian-measure-transfer",
            _measure_transfer_gates,
            config,
        ),
    )


def _q10_orders(rows: Sequence[RowRecord], config: MeshfreeConfig, /) -> list[GateRecord]:
    return [
        # Exact-metric graph Laplacian/projection on the lattice, PHS3
        # reconstruction and ARS-222 with dt ~ h: second order for velocity
        # and energy, less a 0.5 margin. Incremental projection guarantees
        # only first-order pressure in general (no superconvergence assumed).
        *_flow_rates(
            rows,
            config,
            "taylor-green",
            (
                ("velocity-order", "velocity_rms_error", 1.5),
                ("energy-order", "relative_energy_error", 1.5),
                ("pressure-order", "pressure_rms_error", 1.0),
            ),
            "periodic lattice scaled to [0,1)^2: measured Sobol-probe fill distance "
            "times the period 2 pi",
        ),
        # PHS degree 3: second-derivative consistency p - 1 = 2 for velocity,
        # one order less for the stabilized equal-order pressure and for the
        # boundary traction (a first derivative of an O(h^2) field), each
        # less a 0.5 margin.
        *_flow_rates(
            rows,
            config,
            "bounded-stokes",
            (
                ("velocity-order", "velocity_rms_error", 1.5),
                ("pressure-order", "pressure_rms_error", 0.5),
                ("traction-order", "traction_face_relative_error", 0.5),
            ),
            "measured Sobol-probe fill distance of the jittered unit square",
        ),
    ]


def _incompressible_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("cfl-refusal", metrics, "cfl_refused"),
        boolean_gate("cfl-rollback", metrics, "refused_state_unchanged"),
        boolean_gate("cfl-resume", metrics, "resumed_accepted"),
        boolean_gate(
            "incompatible-source-refusal", metrics, "incompatible_source_refused"
        ),
        boolean_gate(
            "incompatible-velocity-kept", metrics, "incompatible_velocity_unchanged"
        ),
        boolean_gate("incompatible-pressure-nan", metrics, "incompatible_pressure_nan"),
    ]


def _coercivity_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("coercivity-budget-refusal", metrics, "coercivity_refused"),
        boolean_gate("refused-before-allocation", metrics, "refused_before_allocation"),
    ]


def _courant_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("courant-refusal", metrics, "courant_refused"),
        boolean_gate("courant-rollback", metrics, "refused_state_kept"),
        boolean_gate("courant-resume", metrics, "continued_accepted"),
    ]


def _uncovered_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("uncovered-source-status", metrics, "uncovered_source_refused"),
        boolean_gate("uncovered-apply-refusal", metrics, "uncovered_apply_refused"),
    ]


def _plan_q10_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return (
        *_flow_rows(
            "meshfree.incompressible-flow",
            _TAYLOR_GREEN,
            "cfl-and-incompatible-source",
            "incompressible-refusals",
            _incompressible_refusal_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.incompressible-flow",
            _TAYLOR_GREEN,
            "coercivity-symbolic-budget",
            "exterior-coercivity-refusal",
            _coercivity_refusal_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.lagrangian-flow",
            _LAGRANGIAN_VOLUME,
            "courant",
            "lagrangian-refusals",
            _courant_refusal_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.lagrangian-flow",
            _MEASURE_TRANSFER,
            "uncovered-source",
            "lagrangian-refusals",
            _uncovered_refusal_gates,
            config,
        ),
    )


# --- Q11 accepted gates.


def _elasticity_manufactured_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("accepted", metrics, "accepted"),
        # PHS3 rows reproduce linear fields exactly: a rigid motion has zero
        # strain/stress/energy up to stencil-conditioned round-off.
        upper_gate("rigid-strain", metrics, "rigid_max_strain", 1e-9),
        upper_gate("rigid-stress", metrics, "rigid_max_stress", 1e-9),
        upper_gate("rigid-energy", metrics, "rigid_strain_energy", 1e-16),
        upper_gate(
            "displacement-resolution", metrics, "displacement_relative_error", 0.1
        ),
    ]


def _elasticity_linear_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    # A homogeneous (linear) field is reproduced exactly by PHS3 and integrated
    # exactly by the trapezoid measure: closed forms hold to solve tolerance.
    return [
        boolean_gate("accepted", metrics, "linear_accepted"),
        upper_gate("displacement", metrics, "linear_displacement_error", 1e-8),
        upper_gate("strain-energy", metrics, "linear_energy_relative_error", 1e-8),
        upper_gate("external-work", metrics, "linear_work_relative_error", 1e-8),
        upper_gate("clapeyron", metrics, "clapeyron_defect", 1e-8),
        upper_gate("force-balance", metrics, "force_balance_defect", 1e-8),
    ]


def _neo_hookean_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("accepted", metrics, "finite_accepted"),
        boolean_gate("small-load-accepted", metrics, "small_load_accepted"),
        # Homogeneous stretch: exact in the PHS3 space and the trapezoid measure.
        upper_gate("finite-displacement", metrics, "finite_displacement_error", 1e-8),
        upper_gate("stored-energy", metrics, "finite_energy_relative_error", 1e-8),
        boolean_gate("jacobian-positive", metrics, "finite_jacobian_positive"),
        # Load-path work is a 4-step trapezoid rule on a smooth load curve:
        # O(1/steps^2) consistency, declared at 1%.
        upper_gate("energy-work", metrics, "energy_work_defect", 1e-2),
        # A 1e-3 load is in the linear regime: O(load) relative gap.
        upper_gate("small-load-linearization", metrics, "small_load_relative_gap", 1e-2),
    ]


def _herrmann_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    gates: list[GateRecord] = []
    for label, _ in MIXED_COMPRESSIBILITIES:
        gates += [
            boolean_gate(f"{label}-accepted", metrics, f"{label}_accepted"),
            upper_gate(
                f"{label}-displacement-resolution",
                metrics,
                f"{label}_displacement_relative_error",
                0.1,
            ),
            upper_gate(
                f"{label}-pressure-resolution",
                metrics,
                f"{label}_pressure_relative_error",
                0.1,
            ),
            upper_gate(
                f"{label}-pressure-oscillation",
                metrics,
                f"{label}_pressure_oscillation",
                _CHECKERBOARD_INDEX,
            ),
        ]
    # Locking-free mixed form: kappa-uniform error; locking grows it ~ 1/kappa.
    gates.append(upper_gate("no-locking", metrics, "locking_ratio", 10.0))
    return gates


def _fluid_structure_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("accepted", metrics, "accepted"),
        # Gated certificate defects are solve-precision residuals of the mortar.
        upper_gate("interface-certificate", metrics, "max_gated_interface_defect", 1e-10),
        # The multiplier pairs with the kinematic defect, which the mortar rows
        # annihilate: delivered + received power is a solve residual.
        upper_gate("power-balance", metrics, "power_balance_defect", 1e-10),
        # The stored-versus-received power gap of the physical channel is
        # reported, not gated: its mixed Dirichlet/traction corners carry an
        # r^(1/2) strain singularity, so the energy identity is gated on the
        # smooth manufactured step instead (fluid-structure-manufactured).
    ]


def _fluid_structure_manufactured_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("accepted", metrics, "accepted"),
        upper_gate("power-balance", metrics, "power_balance_defect", 1e-10),
    ]


def _plan_q11(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return (
        *_flow_rows(
            "meshfree.elasticity",
            _SMALL_STRAIN_TRACTION,
            "manufactured",
            "elasticity-manufactured",
            _elasticity_manufactured_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.elasticity",
            _SMALL_STRAIN_ROLLERS,
            "linear-tension",
            "elasticity-closed-form",
            _elasticity_linear_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.elasticity",
            _NEO_HOOKEAN,
            "neo-hookean-tension",
            "elasticity-closed-form",
            _neo_hookean_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.elasticity",
            _HERRMANN,
            "herrmann-mixed",
            "elasticity-mixed-locking",
            _herrmann_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.fluid-structure",
            _FLUID_STRUCTURE,
            "monolithic-step",
            "fluid-structure-step",
            _fluid_structure_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.fluid-structure",
            _FLUID_STRUCTURE_MANUFACTURED,
            "manufactured-step",
            "fluid-structure-manufactured",
            _fluid_structure_manufactured_gates,
            config,
        ),
    )


def _plan_q11_fsi(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    """Replay the revised block-ghost fluid/solid tuples without other mechanics."""
    return tuple(
        row for row in _plan_q11(config) if row.capability == "meshfree.fluid-structure"
    )


def _plan_q11_fsi_energy(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    """Resolve the unchanged manufactured balance beyond its coarse cancellation."""
    return _flow_rows(
        "meshfree.fluid-structure",
        _FLUID_STRUCTURE_MANUFACTURED,
        "manufactured-step",
        "fluid-structure-manufactured",
        _fluid_structure_manufactured_gates,
        config,
    )


def _manufactured_elasticity_orders(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /
) -> list[GateRecord]:
    # PHS3: displacement O(h^(p-1)) = O(h^2) for the second-order operator and
    # the boundary traction one order lower, each less a 0.5 margin.
    return _flow_rates(
        rows,
        config,
        "manufactured",
        (
            ("displacement-order", "displacement_relative_error", 1.5),
            ("traction-order", "traction_face_relative_error", 0.5),
        ),
        "measured Sobol-probe fill distance of the jittered unit square",
    )


def _q11_orders(rows: Sequence[RowRecord], config: MeshfreeConfig, /) -> list[GateRecord]:
    # The manufactured FSI step uses the same cubic-augmented collocation: field
    # error and the quadrature energy identity at second order less 0.5.
    return [
        *_manufactured_elasticity_orders(rows, config),
        *_flow_rates(
            rows,
            config,
            "manufactured-step",
            (
                ("solid-rate-order", "solid_rate_relative_error", 1.5),
                ("energy-balance-order", "energy_balance_gap", 1.5),
            ),
            "measured Sobol-probe fill distance of the fluid unit-square lattice",
        ),
    ]


def _plan_q11_elasticity_order(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    """Traction-face manufactured elasticity alone, on the finer order lattices."""
    return _flow_rows(
        "meshfree.elasticity",
        _SMALL_STRAIN_TRACTION,
        "manufactured",
        "elasticity-manufactured",
        _elasticity_manufactured_gates,
        config,
    )


def _newton_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("newton-budget-refusal", metrics, "newton_solve_refused"),
        boolean_gate("rollback-load", metrics, "rolled_back_to_zero_load"),
        upper_gate("rollback-state", metrics, "committed_max_displacement", 0.0),
    ]


def _floating_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [boolean_gate("floating-body-refusal", metrics, "floating_body_refused")]


def _unstabilized_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [boolean_gate("inf-sup-refusal", metrics, "unstabilized_mixed_refused")]


def _plan_q11_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return (
        *_flow_rows(
            "meshfree.elasticity",
            _NEO_HOOKEAN,
            "newton-budget",
            "mechanics-refusals",
            _newton_refusal_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.elasticity",
            _SMALL_STRAIN_TRACTION,
            "floating-body",
            "mechanics-refusals",
            _floating_refusal_gates,
            config,
        ),
        *_flow_rows(
            "meshfree.elasticity",
            _HERRMANN,
            "unstabilized-equal-order",
            "mechanics-refusals",
            _unstabilized_refusal_gates,
            config,
        ),
    )


def _surface_stokes_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    # Tolerances are the owner's declared surface-Stokes contract (N = 400).
    return [
        boolean_gate("solved", metrics, "successful"),
        # The tangency multiplier enforces N^T U = 0 to the solve tolerance.
        upper_gate("tangency", metrics, "normal_residual", 1e-8),
        upper_gate("gauge", metrics, "gauge_residual", 1e-3),
        # Rigid rotations are Killing fields: zero strain and dissipation.
        upper_gate("killing-strain", metrics, "killing_max_strain", 5e-4),
        upper_gate("killing-dissipation", metrics, "killing_dissipation", 1e-6),
        upper_gate("dissipation", metrics, "dissipation_relative_error", 2e-2),
        upper_gate("velocity-resolution", metrics, "velocity_relative_error", 0.1),
    ]


def _plan_q11_surface(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return _flow_rows(
        "meshfree.surface-stokes",
        _SURFACE_STOKES,
        "sphere-strain-form",
        "surface-stokes-sphere",
        _surface_stokes_gates,
        config,
    )


def _q11_surface_orders(
    rows: Sequence[RowRecord], config: MeshfreeConfig, /
) -> list[GateRecord]:
    # Degree-4 GMLS: the strain divergence is a second-order operator,
    # O(h^(p-1)) = O(h^3) less 0.5 for velocity; the saddle pressure one
    # order lower. No superconvergence is assumed on spiral samples.
    return _flow_rates(
        rows,
        config,
        "sphere-strain-form",
        (
            ("velocity-order", "velocity_relative_error", 2.5),
            ("pressure-order", "pressure_relative_error", 1.5),
        ),
        "measured chordal fill distance over uniform random sphere probes",
    )


def _surface_refusal_gates(metrics: dict[str, Any], /) -> list[GateRecord]:
    return [
        boolean_gate("maximum-steps-status", metrics, "maximum_steps_reported"),
        boolean_gate("not-accepted", metrics, "refused_not_successful"),
        boolean_gate("finite-evidence", metrics, "refused_fields_finite"),
    ]


def _plan_q11_surface_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return _flow_rows(
        "meshfree.surface-stokes",
        _SURFACE_STOKES,
        "starved-krylov",
        "surface-stokes-refusal",
        _surface_refusal_gates,
        config,
    )


# ---------------------------------------------------------------------------
# Q12 higher forms and Q14 sensitivity campaigns.

from benchmarks.meshfree_closure_forms_sensitivity import (
    FORMS_SENSITIVITY_WORKLOADS,
    higher_form_dimension,
    higher_form_level,
    HigherFormGeometry,
)


_Q12_TUPLES: dict[HigherFormGeometry, tuple[str, str, int]] = {
    "holed-square": (
        "absolute-and-relative-boundary-complex",
        "cellmesh-holed-square-domain",
        4096,
    ),
    "tunneled-slab": (
        "absolute-and-relative-boundary-complex",
        "cellmesh-tunneled-slab-domain",
        1024,
    ),
    "icosphere": ("closed-surface", "cellmesh-icosphere-closed-surface", 4096),
}


def _q12_support(
    geometry: HigherFormGeometry, config: MeshfreeConfig, /
) -> dict[str, SupportValue]:
    boundary, authority, capacity = _Q12_TUPLES[geometry]
    return meshfree_support(
        dimension=higher_form_dimension(geometry),
        method="gmls-p1-k-form-moment-reconstruction-sparse-hodge",
        boundary=boundary,
        geometry_authority=authority,
        derivative_class="none",
        capacity=capacity,
        precision=config.precision,
    )


def _clique_support(config: MeshfreeConfig, /) -> dict[str, SupportValue]:
    return meshfree_support(
        dimension=2,
        method="radius-clique-exact-integer-chain",
        boundary="none",
        geometry_authority="abstract-radius-clique-no-geometry",
        derivative_class="none",
        capacity=4096,
        precision=config.precision,
    )


def _q12_geometries(config: MeshfreeConfig, /) -> tuple[HigherFormGeometry, ...]:
    geometries = tuple(
        geometry
        for geometry in _Q12_TUPLES
        if higher_form_dimension(geometry) == config.dimension
    )
    if not geometries:
        raise ValueError(f"Q12 declares no {config.dimension}-D authoritative complex.")
    # Validate every requested top-cell capacity before anything executes.
    for geometry in geometries:
        for size in config.sizes:
            higher_form_level(geometry, size)
    return geometries


def _workload(
    name: str, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    return FORMS_SENSITIVITY_WORKLOADS[name](capacity, seed, config)


def _q12_row(
    geometry: HigherFormGeometry, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = _workload(f"higher-forms-{geometry}", capacity, seed, config)
    metrics = record["metrics"]
    gates = [
        boolean_gate("geometry-admission", metrics, "admitted"),
        boolean_gate("geometry-authorized-fidelity", metrics, "geometry_authorized"),
        # Order-6 simplex cubature is exact for the cubic coefficients, so the
        # de Rham map commutes with d to round-off (owner audit tolerance 1e-11).
        upper_gate("commutation", metrics, "commutation_relative_residual", 1e-10),
        # Discrete Green/Stokes with exact integer incidence and outward trace:
        # only floating summation round-off remains.
        upper_gate("stokes", metrics, "stokes_relative_defect", 1e-11),
        # d∘d has exactly cancelling ±1 incidence; a few ulps of the intermediate.
        upper_gate("d-squared-zero", metrics, "d_squared_relative", 1e-13),
        boolean_gate("hodge-positivity", metrics, "hodge_admitted"),
        # P1 forms are reproduced exactly on every admitted patch.
        upper_gate("hodge-reproduction", metrics, "reproduction_residual", 1e-9),
        # δ applies M_{k-1}^{-1} through a native sparse factorization admitted at
        # condition <= 1e10; owner contract tolerance 1e-7.
        upper_gate(
            "hodge-adjoint", metrics, "codifferential_adjoint_relative_defect", 1e-7
        ),
        boolean_gate("harmonic-absolute", metrics, "harmonic_matches_betti"),
        boolean_gate("hodge-laplace-status", metrics, "hodge_laplace_successful"),
        # Mixed Hodge–Laplace original-equation residual (owner solve contract 1e-6).
        upper_gate(
            "hodge-laplace-consumer", metrics, "hodge_laplace_relative_residual", 1e-6
        ),
    ]
    if geometry != "icosphere":
        gates += [
            boolean_gate(
                "harmonic-relative", metrics, "harmonic_relative_matches_lefschetz"
            ),
            # Flat simplices: the degree-2m node cubature integrates |ω|² of a
            # linear form exactly, as does the consistent P1 Hodge.
            upper_gate(
                "hodge-linear-norm", metrics, "hodge_linear_norm_relative_defect", 1e-9
            ),
            upper_gate(
                "linear-reconstruction",
                metrics,
                "linear_reconstruction_relative_error",
                1e-9,
            ),
        ]
    if geometry == "tunneled-slab":
        # B = dA and the leapfrog update -dE keep dB = 0 to round-off (|B| = O(1)).
        gates.append(
            upper_gate(
                "maxwell-divergence", metrics, "maxwell_magnetic_divergence", 1e-11
            )
        )
    return {**record, "gates": gates}


def _clique_row(capacity: int, seed: int, config: MeshfreeConfig, /) -> dict[str, Any]:
    record = _workload("abstract-clique-forms", capacity, seed, config)
    metrics = record["metrics"]
    return {
        **record,
        "gates": [
            boolean_gate("betti-circle", metrics, "betti_matches_circle"),
            # Exact integer chains: boundary∘boundary has no nonzero entry.
            upper_gate("d-squared-zero", metrics, "boundary_composition_nonzeros", 0.0),
            boolean_gate("abstract-fidelity", metrics, "abstract_research_fidelity"),
            boolean_gate("work-bound", metrics, "work_within_declared"),
        ],
    }


def _plan_q12(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    rows = [
        PlannedRow(
            "meshfree.higher-forms",
            _q12_support(geometry, config),
            size,
            seed,
            geometry,
            partial(_q12_row, geometry, size, seed, config),
        )
        for geometry in _q12_geometries(config)
        for size in config.sizes
        for seed in config.seeds
    ]
    if config.dimension == 2:
        rows += [
            PlannedRow(
                "meshfree.abstract-clique-forms",
                _clique_support(config),
                size,
                seed,
                "abstract-clique-circle",
                partial(_clique_row, size, seed, config),
            )
            for size in config.sizes
            for seed in config.seeds
        ]
    return tuple(rows)


def _q12_order(rows: Sequence[RowRecord], config: MeshfreeConfig, /) -> list[GateRecord]:
    gates: list[GateRecord] = []
    for support_id in sorted(
        {
            row["support_tuple_id"]
            for row in rows
            if row["capability"] == "meshfree.higher-forms"
        }
    ):
        for seed in config.seeds:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["support_tuple_id"] == support_id
                    and row["seed"] == seed
                    and "metrics" in row
                ),
                key=lambda row: row["capacity"],
            )
            # P1 GMLS sample maps: vertex values and edge moments of the fitted
            # gradient both err by O(h^2); order 2 with 0.5 pre-asymptotic slack.
            gates.append(
                rate_gate(
                    "sampled-commutation-order",
                    [row["metrics"]["mesh_size_h"] for row in selected],
                    [row["metrics"]["sampled_commutation_error"] for row in selected],
                    1.5,
                    metric="max-abs-d-sample0-minus-sample1-order",
                    provenance="maximum edge length of the authoritative mesh",
                    support_tuple_id=support_id,
                    coordinates={
                        "seed": seed,
                        "capacities": [row["capacity"] for row in selected],
                    },
                )
            )
    return gates


def _q12_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = _workload("higher-forms-refusals", capacity, seed, config)
    metrics = record["metrics"]
    return {
        **record,
        "gates": [
            boolean_gate(
                "abstract-clique-refusal", metrics, "abstract_clique_plan_refused"
            ),
            boolean_gate(
                "abstract-clique-authority-refusal",
                metrics,
                "abstract_clique_authority_refused",
            ),
            boolean_gate("measure-audit-refusal", metrics, "measure_audit_refused"),
            boolean_gate("betti-audit-refusal", metrics, "betti_audit_refused"),
            boolean_gate(
                "orientation-audit-refusal", metrics, "orientation_audit_refused"
            ),
            boolean_gate("patch-capacity-refusal", metrics, "patch_capacity_refused"),
        ],
    }


def _clique_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = _workload("abstract-clique-refusals", capacity, seed, config)
    metrics = record["metrics"]
    return {
        **record,
        "gates": [
            boolean_gate("simplex-capacity-refusal", metrics, "simplex_capacity_refused"),
            boolean_gate("work-capacity-refusal", metrics, "work_capacity_refused"),
        ],
    }


def _plan_q12_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    if config.dimension != 2:
        raise ValueError("Q12-refusal audits the 2-D holed-square authority.")
    for size in config.sizes:
        higher_form_level("holed-square", size)
    return tuple(
        row
        for size in config.sizes
        for seed in config.seeds
        for row in (
            PlannedRow(
                "meshfree.higher-forms",
                _q12_support("holed-square", config),
                size,
                seed,
                "authority-audit-refusals",
                partial(_q12_refusal_row, size, seed, config),
            ),
            PlannedRow(
                "meshfree.abstract-clique-forms",
                _clique_support(config),
                size,
                seed,
                "clique-capacity-refusals",
                partial(_clique_refusal_row, size, seed, config),
            ),
        )
    )


_Q14_TUPLES: dict[str, tuple[str, str, str, str, int]] = {
    "fixed-support": (
        "gmls-wendland-c2-smooth-envelope-point-cloud",
        "none",
        "unit-cube-jittered-lattice",
        "fixed-support-coordinate-jvp-vjp",
        4096,
    ),
    "cutoff-crossing": (
        "gmls-wendland-c2-smooth-envelope-local-stencil",
        "none",
        "ball-sources-oblique-cutoff-crossing",
        "smooth-support-cutoff-crossing",
        1024,
    ),
    "strict-active": (
        "edge-moment-metric-nonnegative-conic",
        "dirichlet-lattice-boundary",
        "cartesian-lattice-nodal-control-volumes",
        "strict-active-fixed-set-kkt-jvp",
        4096,
    ),
    "frozen-remap": (
        "conservative-positive-point-transfer-live-history-remap",
        "none",
        "unit-cube-stratified-supports",
        "frozen-remap-value-tangent-adjoint",
        4096,
    ),
    "knn-tie": (
        "gmls-knn-local-stencil",
        "none",
        "axis-equidistant-selection-tie",
        "knn-selection-gap",
        64,
    ),
}


def _q14_support(route: str, config: MeshfreeConfig, /) -> dict[str, SupportValue]:
    method, boundary, authority, derivative, capacity = _Q14_TUPLES[route]
    return meshfree_support(
        dimension=config.dimension,
        method=method,
        boundary=boundary,
        geometry_authority=authority,
        derivative_class=derivative,
        capacity=capacity,
        precision=config.precision,
    )


def _q14_gates(route: str, metrics: dict[str, Any], /) -> list[GateRecord]:
    match route:
        case "fixed-support":
            return [
                boolean_gate("derivative-availability", metrics, "accepted"),
                boolean_gate("refresh-status", metrics, "status_accepted"),
                boolean_gate("rank", metrics, "rank_full"),
                # Central difference (step 1e-3 of a 0.05h motion): O(step^2)
                # truncation of a C^2-on-envelope map; owner contract 1e-6.
                upper_gate(
                    "jvp-finite-difference", metrics, "jvp_fd_relative_error", 1e-6
                ),
                # <c, J t> = <J^T c, t> up to summation round-off over every weight.
                upper_gate("vjp-duality", metrics, "vjp_duality_relative_error", 1e-10),
                # The linearization's primal is the published refresh map itself.
                upper_gate(
                    "contract-identity",
                    metrics,
                    "linearization_primal_identity_error",
                    1e-12,
                ),
            ]
        case "cutoff-crossing":
            return [
                boolean_gate(
                    "derivative-availability", metrics, "accepted_along_trajectory"
                ),
                boolean_gate("rank", metrics, "rank_full_along_trajectory"),
                boolean_gate("cutoff-zero-weight", metrics, "cutoff_weight_vanishes"),
                # Independent NumPy normal equations agree to conditioning round-off.
                upper_gate(
                    "weight-oracle", metrics, "weight_oracle_relative_error", 1e-9
                ),
                # Oracle central differences at step 1e-5: O(1e-10) truncation.
                upper_gate(
                    "jvp-finite-difference",
                    metrics,
                    "weight_tangent_relative_error",
                    1e-6,
                ),
                # Second-order one-sided differences at h = 1e-4 on each side of a
                # C^1-at-cutoff map: O(h^2) truncation.
                upper_gate(
                    "cutoff-two-sided", metrics, "cutoff_two_sided_relative_error", 1e-5
                ),
                upper_gate(
                    "field-derivative",
                    metrics,
                    "field_derivative_tangent_relative_error",
                    1e-6,
                ),
                # Quadratics are reproduced exactly by the degree-2 GMLS fit.
                upper_gate(
                    "quadratic-exactness",
                    metrics,
                    "quadratic_exactness_relative_error",
                    1e-10,
                ),
                upper_gate("vjp-duality", metrics, "vjp_duality_relative_error", 1e-12),
            ]
        case "strict-active":
            return [
                boolean_gate("metric-acceptance", metrics, "metric_accepted"),
                boolean_gate("fd-solve-acceptance", metrics, "fd_solves_accepted"),
                boolean_gate("active-set-regular", metrics, "active_set_regular"),
                boolean_gate("derivative-availability", metrics, "derivative_available"),
                boolean_gate("tangent-availability", metrics, "tangent_available"),
                boolean_gate("contract-identity", metrics, "derivative_contract_matches"),
                # Forward conic tolerance 1e-9 over step 1e-4 bounds FD noise by
                # ~1e-5; smooth on the fixed active set (O(step^2) truncation).
                upper_gate(
                    "jvp-finite-difference",
                    metrics,
                    "strict_active_fd_relative_error",
                    1e-3,
                ),
            ]
        case "frozen-remap":
            return [
                boolean_gate("remap-acceptance", metrics, "remap_successful"),
                boolean_gate(
                    "derivative-availability", metrics, "value_derivative_available"
                ),
                # A linear map: central differences carry only cancellation
                # round-off eps*|f|/step.
                upper_gate(
                    "jvp-finite-difference", metrics, "jvp_fd_relative_error", 1e-10
                ),
                # Each history's JVP is its own route's primal action.
                upper_gate(
                    "contract-identity", metrics, "jvp_route_action_identity_error", 1e-12
                ),
                upper_gate(
                    "published-pullback",
                    metrics,
                    "published_pullback_relative_error",
                    1e-12,
                ),
                # Inner products accumulate O(nnz eps) round-off.
                upper_gate("vjp-duality", metrics, "vjp_duality_relative_error", 1e-10),
                upper_gate(
                    "hilbert-adjoint", metrics, "hilbert_adjoint_relative_error", 1e-10
                ),
                upper_gate(
                    "conservation", metrics, "content_conservation_relative_error", 1e-10
                ),
            ]
        case _:
            raise ValueError(f"No accepted Q14 route {route!r}.")


_Q14_WORKLOADS = {
    "fixed-support": "fixed-support-sensitivity",
    "cutoff-crossing": "smooth-support-crossing",
    "strict-active": "strict-active-metric-sensitivity",
    "frozen-remap": "frozen-remap-sensitivity",
}
_Q14_REFUSALS = {
    "fixed-support": "fixed-support-envelope-exit",
    "cutoff-crossing": "smooth-support-envelope-exit",
    "strict-active": "strict-active-change-refusal",
    "frozen-remap": "frozen-remap-failure-refusal",
    "knn-tie": "knn-tie-refusal",
}


def _q14_row(
    route: str, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = _workload(_Q14_WORKLOADS[route], capacity, seed, config)
    return {**record, "gates": _q14_gates(route, record["metrics"])}


_Q14_REFUSAL_GATES: dict[str, tuple[tuple[str, str], ...]] = {
    "fixed-support": (
        ("refusal-status", "support_exceeded"),
        ("refusal-not-accepted", "not_accepted"),
        ("refusal-nan-value", "primal_nan"),
        ("refusal-nan-jvp", "jvp_nan"),
        ("refusal-nan-vjp", "vjp_nan"),
    ),
    "cutoff-crossing": (
        ("outsider-not-candidate", "outsider_not_candidate"),
        ("refusal-status", "support_exceeded"),
        ("refusal-nan-value", "value_nan"),
        ("refusal-nan-jvp", "jvp_nan"),
        ("refusal-nan-vjp", "vjp_nan"),
        ("inside-envelope-control", "inside_envelope_finite"),
    ),
    "strict-active": (
        ("baseline-regular", "baseline_regular"),
        ("refusal-status", "active_set_change_named"),
        ("derivative-withheld", "derivative_withheld"),
        ("tangent-unavailable", "tangent_unavailable"),
        ("refusal-nan-jvp", "tangent_nan"),
    ),
    "frozen-remap": (
        ("refusal-status", "remap_refused"),
        ("failed-history-named", "failed_histories_named"),
        ("derivative-withheld", "value_derivative_unavailable"),
        ("refusal-nan-value", "values_nan"),
        ("refusal-nan-jvp", "jvp_nan"),
        ("pullback-refusal", "pullback_refused"),
        ("adjoint-refusal", "adjoint_refused"),
    ),
    "knn-tie": (
        ("anchor-value-finite", "anchor_value_finite"),
        ("anchor-tangent-nan", "anchor_tangent_nan"),
        ("refusal-status", "crossing_support_exceeded"),
        ("refusal-nan-value", "crossing_value_nan"),
        ("refusal-nan-jvp", "crossing_jvp_nan"),
        ("refusal-nan-vjp", "crossing_vjp_nan"),
    ),
}


def _q14_refusal_row(
    route: str, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = _workload(_Q14_REFUSALS[route], capacity, seed, config)
    metrics = record["metrics"]
    return {
        **record,
        "gates": [
            boolean_gate(name, metrics, key) for name, key in _Q14_REFUSAL_GATES[route]
        ],
    }


def _q14_routes(routes: Sequence[str], config: MeshfreeConfig, /) -> tuple[str, ...]:
    # The cutoff crossing is declared only where an oblique crossing exists.
    return tuple(
        route
        for route in routes
        if not (route == "cutoff-crossing" and config.dimension == 1)
    )


def _plan_q14(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.sensitivities",
            _q14_support(route, config),
            size,
            seed,
            route,
            partial(_q14_row, route, size, seed, config),
        )
        for route in _q14_routes(tuple(_Q14_WORKLOADS), config)
        for size in config.sizes
        for seed in config.seeds
    )


def _plan_q14_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.sensitivities",
            _q14_support(route, config),
            size,
            seed,
            f"{route}-refusal",
            partial(_q14_refusal_row, route, size, seed, config),
        )
        for route in _q14_routes(tuple(_Q14_REFUSALS), config)
        for size in config.sizes
        for seed in config.seeds
    )


# ---------------------------------------------------------------------------
# Q15 adaptive/learned/calibrated/hybrid and Q17 long-run restart campaigns.

from benchmarks.meshfree_closure_adaptive_restart import (
    ADAPTIVE_RESTART_WORKLOADS,
)


@dataclass(frozen=True, slots=True)
class _Criterion:
    """One declared gate: threshold fixed before execution, never tuned."""

    name: str
    kind: Literal["upper", "lower", "boolean"]
    key: str
    target: float | bool
    evidence_kind: GateEvidenceKind = "scientific"
    # (Boolean key, capacity key): when the Boolean is False the comparison
    # cannot be measured within the declared capacity and the gate passes with
    # that reason recorded (Main's P16 decision for the Q15 equal-work gate).
    unreached: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _Q15Q17Row:
    capability: str
    method: str
    boundary: str
    geometry: str
    derivative: str
    support_capacity: int
    dimension: int
    workload: str
    capacity: int
    label: str
    owner: str
    criteria: tuple[_Criterion, ...]
    temporal: str = "none"
    device: MeshfreeDevice = "cpu"
    reproducibility: str = "deterministic-seeded"

    def support(self, config: MeshfreeConfig, /) -> dict[str, SupportValue]:
        return meshfree_support(
            dimension=self.dimension,
            method=self.method,
            boundary=self.boundary,
            geometry_authority=self.geometry,
            derivative_class=self.derivative,
            capacity=self.support_capacity,
            precision=config.precision,
            temporal_method=self.temporal,
            device=self.device,
            reproducibility=self.reproducibility,
        )


def _upper(
    name: str, key: str, tolerance: float, kind: GateEvidenceKind = "scientific"
) -> _Criterion:
    return _Criterion(name, "upper", key, tolerance, kind)


def _lower(
    name: str, key: str, minimum: float, kind: GateEvidenceKind = "scientific"
) -> _Criterion:
    return _Criterion(name, "lower", key, minimum, kind)


def _flag(
    name: str, key: str, expected: bool = True, kind: GateEvidenceKind = "scientific"
) -> _Criterion:
    return _Criterion(name, "boolean", key, expected, kind)


def _q15q17_unreached(item: _Criterion, metrics: dict[str, Any], /) -> GateRecord | None:
    """Uniform refinement never reached the adaptive error within its declared capacity.

    Main's P16 decision: the equal-work gate then passes only on a strict
    same-work envelope (a measured uniform row already spends at least the
    adaptive cumulative work with a worse error); otherwise it is unexecuted.
    No win is granted by capacity alone and nothing is extrapolated.
    """
    if item.unreached is None:
        return None
    flag_key, capacity_key = item.unreached
    if metrics.get(flag_key) is not False:
        return None
    dominated = metrics.get(_SAME_WORK_ENVELOPE) is True
    return gate(
        item.name,
        "passed" if dominated else "unexecuted",
        metric=item.key,
        comparison="greater-than-or-equal",
        target=float(item.target),
        observed=None,
        evidence_kind=item.evidence_kind,
        reason=(
            "uniform refinement did not reach the adaptive error within the declared "
            f"uniform capacity {metrics.get(capacity_key)} points; "
            + (
                "a measured uniform row spends at least the adaptive work with a worse "
                "error (same-work envelope)"
                if dominated
                else "no measured uniform row spends the adaptive work, so the "
                "equal-work comparison is unmeasured (no extrapolation)"
            )
        ),
    )


def _q15q17_gates(
    criteria: tuple[_Criterion, ...], metrics: dict[str, Any], /
) -> list[GateRecord]:
    gates: list[GateRecord] = []
    for item in criteria:
        unreached = _q15q17_unreached(item, metrics)
        if unreached is not None:
            gates.append(unreached)
            continue
        match item.kind:
            case "upper":
                gates.append(
                    upper_gate(
                        item.name,
                        metrics,
                        item.key,
                        float(item.target),
                        evidence_kind=item.evidence_kind,
                    )
                )
            case "lower":
                gates.append(
                    lower_gate(
                        item.name,
                        metrics,
                        item.key,
                        float(item.target),
                        evidence_kind=item.evidence_kind,
                    )
                )
            case "boolean":
                gates.append(
                    boolean_gate(
                        item.name,
                        metrics,
                        item.key,
                        expected=bool(item.target),
                        evidence_kind=item.evidence_kind,
                    )
                )
    return gates


_LAYER_METHOD = "phs-rbf-fd-degree3-probe-residual-dorfler-h-refinement"
_LAYER_GEOMETRY = "unit-square-jittered-lattice-projected-boundary-children"
_CORRECTION_BOUNDARY = "dirichlet-incomplete-axial-neighborhood"
_HYBRID_GEOMETRY = "unit-square-frame-cloud-and-interior-fv-grid"
_EPOCH_RESTART = (
    "gmls-degree2-ale-adr-production-run-support-epoch",
    "periodic",
    "periodic-unit-square-jittered-lattice",
)

# Q15 equal-work gate (Main's P16 decisions): the uniform sweep is extended over
# capacities declared before execution until its error is at most the adaptive
# final error; uniform CPU work at the adaptive error is then interpolated
# (log-log) inside the measured uniform range, never extrapolated, and the gain
# is that work over the adaptive cumulative work (>= 1). If the declared uniform
# capacity cannot reach the adaptive error, the gate passes only on a strict
# same-work envelope and is otherwise unexecuted.
_EQUAL_WORK_UNREACHED = ("uniform_reaches_adaptive_error", "uniform_points_max")
_SAME_WORK_ENVELOPE = "uniform_worse_at_or_above_adaptive_work"
_MIGRATION_RESTART = (
    "gmls-degree2-explicit-diffusion-distributed-production-run",
    "none",
    "unit-square-uniform-random-owner-blocked",
)
_MOVING_RESTART = (
    "surface-material-reaction-diffusion-production-run-live-history",
    "closed-surface",
    "prescribed-expanding-sphere",
)


def _layer_row(
    workload: str, capacity: int, label: str, criteria: tuple[_Criterion, ...]
) -> _Q15Q17Row:
    return _Q15Q17Row(
        "meshfree.adaptive-refinement",
        _LAYER_METHOD,
        "dirichlet",
        _LAYER_GEOMETRY,
        "none",
        2048,
        2,
        workload,
        capacity,
        label,
        "AdaptivityP14c",
        criteria,
    )


def _correction_row(
    method: str,
    geometry: str,
    derivative: str,
    support_capacity: int,
    workload: str,
    capacity: int,
    label: str,
    criteria: tuple[_Criterion, ...],
) -> _Q15Q17Row:
    return _Q15Q17Row(
        "meshfree.learned-correction",
        method,
        _CORRECTION_BOUNDARY,
        geometry,
        derivative,
        support_capacity,
        2,
        workload,
        capacity,
        label,
        "AdaptivityP14c",
        criteria,
    )


def _coupled_row(
    family: str, method: str, criteria: tuple[_Criterion, ...]
) -> _Q15Q17Row:
    return _Q15Q17Row(
        "meshfree.learned-coupled-law",
        method,
        "none",
        "random-gaussian-edge-frame-ring",
        "jacobian-blocks",
        1024,
        3,
        f"learned-coupled-{family}-invariance",
        64,
        f"o3-{family}-invariance",
        "ConstitutiveP14a",
        criteria,
    )


def _restart_row(
    route: tuple[str, str, str],
    dimension: int,
    support_capacity: int,
    temporal: str,
    workload: str,
    capacity: int,
    label: str,
    owner: str,
    criteria: tuple[_Criterion, ...],
    *,
    device: MeshfreeDevice = "cpu",
    reproducibility: str = "bitwise-replay",
) -> _Q15Q17Row:
    method, boundary, geometry = route
    return _Q15Q17Row(
        "meshfree.runtime-restart",
        method,
        boundary,
        geometry,
        "none",
        support_capacity,
        dimension,
        workload,
        capacity,
        label,
        owner,
        criteria,
        temporal=temporal,
        device=device,
        reproducibility=reproducibility,
    )


# Q15 thresholds, declared from the plan (P14/P16) before any run:
# - Adaptivity claims only an improvement over uniform refinement at equal
#   point count (error ratio >= 1) and at equal error (uniform CPU work over
#   adaptive cumulative CPU work >= 1, warm, interpolated inside the extended
#   uniform sweep; see _EQUAL_WORK_UNREACHED); its magnitude is reported, not
#   claimed. Indicators are never error bounds.
# - Bulk boundary layer requests 2 adaptive levels (AdaptivityP14c, approved by
#   Main): with the default grading 2.05 a third level needs 1219 > 1156 points.
# - The joint degree-0 transfer conserves content exactly through a native
#   constrained solve; O(1) content admits 1e-9 absolute round-off.
# - Correction moments are audited at the correction policy tolerance 1e-9; the
#   sign margin is the declared policy margin (ratio >= 1 up to 1e-9 round-off).
# - Implicit training gradient: central difference with h = 1e-5 has O(h^2)
#   truncation and 1e-11 solve tolerances, so a correct adjoint agrees to 1e-6.
# - Training identifies two coefficients of a model that contains the truth
#   with superlinear L-BFGS: six steps must reduce the loss by 100x.
# - Coupled laws: reversal parity is exact by construction (0); O(3) covariance
#   and independent reversal are float64 round-off of composed tensor
#   products (relative 1e-11 / 1e-12); certified bounds hold up to 1e-9.
# - Split-conformal held-out coverage lies within 3 standard deviations of the
#   Beta-binomial finite-sample band under exchangeability; shifted cases get
#   no guarantee and are only reported.
# - Hybrid overlap: PHS degree-3 Laplacian consistency and cell-centred FV are
#   both O(h^2); a measured order >= 1.5 over four levels allows pre-asymptotic
#   behavior. The composite error may exceed the monolithic meshfree error by
#   at most 2x (interpolating overlap transfers); the certified component
#   residual uses the owner's acceptance tolerance 1e-8; no overlap transfer
#   row may be refused.
_Q15_ROWS: tuple[_Q15Q17Row, ...] = (
    _layer_row(
        "adaptive-bulk-boundary-layer",
        1156,
        "bulk-boundary-layer",
        (
            _lower("equal-points-gain", "adaptive_gain_at_equal_points", 1.0),
            _Criterion(
                "equal-work-gain",
                "lower",
                "adaptive_work_gain_at_equal_error",
                1.0,
                "performance",
                unreached=_EQUAL_WORK_UNREACHED,
            ),
            _flag("epochs-published", "all_epochs_published"),
            _flag("epoch-stability", "all_epoch_stability_admitted"),
            _upper("epoch-conservation", "max_epoch_conservation_residual", 1e-9),
            _flag("solve-status", "all_solves_successful"),
        ),
    ),
    _Q15Q17Row(
        "meshfree.adaptive-refinement",
        "phs-rbf-fd-degree3-vs-degree4-indicator-h-refinement-laplace-beltrami",
        "closed-surface",
        "unit-sphere-implicit-level-set-fibonacci",
        "none",
        1024,
        3,
        "adaptive-sphere-zonal-peak",
        768,
        "sphere-zonal-peak",
        "AdaptivityP14c",
        (
            _lower("equal-points-gain", "adaptive_gain_at_equal_points", 1.0),
            _Criterion(
                "equal-work-gain",
                "lower",
                "adaptive_work_gain_at_equal_error",
                1.0,
                "performance",
                unreached=_EQUAL_WORK_UNREACHED,
            ),
            _flag("proposals-admitted", "all_proposals_admitted"),
            _flag("error-monotone", "adaptive_error_monotone"),
            _flag("solve-status", "all_solves_successful"),
        ),
    ),
    _correction_row(
        "edge-moment-metric-learned-candidate-nonnegative-conic-projection",
        "jittered-cartesian-nodal-control-volumes",
        "projection-tangent",
        256,
        "learned-correction-constrained",
        36,
        "constrained-conic-projection",
        (
            _flag("correction-admitted", "admitted"),
            _upper("moment-exactness", "max_moment_residual", 1e-9),
            _lower("sign-margin", "minimum_weight_over_margin", 1.0 - 1e-9),
            _flag("coercivity", "coercive"),
            _flag("derivative-availability", "derivative_available"),
        ),
    ),
    _correction_row(
        "edge-moment-metric-learned-log-linear-training",
        "cartesian-nodal-control-volumes",
        "implicit-adjoint",
        256,
        "learned-correction-training",
        36,
        "implicit-adjoint-training",
        (
            _upper("implicit-gradient", "gradient_fd_relative_error", 1e-6),
            _upper("training-loss-reduction", "loss_reduction_ratio", 1e-2),
            _lower("accepted-updates", "accepted_updates", 1, "operational"),
            _flag("trained-admitted", "trained_correction_admitted"),
            _upper("moment-exactness", "trained_max_moment_residual", 1e-9),
            _flag("primal-status", "trained_primal_successful"),
            _flag("adjoint-status", "trained_adjoint_successful"),
        ),
    ),
    _coupled_row(
        "monotone",
        "o3-invariant-convex-monotone-coupled-edge-flux",
        (
            _upper("reversal-parity", "reversal_parity_max_abs", 0.0),
            _upper("independent-reversal", "independent_reversal_relative_error", 1e-12),
            _upper("rotation-covariance", "rotation_covariance_relative_error", 1e-11),
            _upper(
                "rotoreflection-covariance",
                "rotoreflection_covariance_relative_error",
                1e-11,
            ),
            _lower("strong-monotonicity", "sampled_monotonicity_over_bound", 1.0 - 1e-9),
            _upper("jacobian-symmetry", "jacobian_symmetry_max_abs", 1e-10),
            _lower(
                "jacobian-coercivity", "jacobian_min_eigenvalue_over_bound", 1.0 - 1e-9
            ),
        ),
    ),
    _coupled_row(
        "lipschitz",
        "o3-equivariant-lipschitz-coupled-edge-flux",
        (
            _upper("reversal-parity", "reversal_parity_max_abs", 0.0),
            _upper("independent-reversal", "independent_reversal_relative_error", 1e-12),
            _upper("rotation-covariance", "rotation_covariance_relative_error", 1e-11),
            _upper(
                "rotoreflection-covariance",
                "rotoreflection_covariance_relative_error",
                1e-11,
            ),
            _upper("lipschitz-bound", "sampled_lipschitz_over_bound", 1.0 + 1e-9),
        ),
    ),
    _Q15Q17Row(
        "meshfree.calibrated-prediction",
        "split-conformal-probe-values-hybrid-overlap",
        "dirichlet",
        _HYBRID_GEOMETRY,
        "none",
        512,
        2,
        "calibrated-hybrid-held-out",
        384,
        "held-out-split-conformal",
        "AdaptivityP14c",
        (
            _flag("split-disjoint", "split_disjoint"),
            _flag("cases-accepted", "all_family_cases_accepted"),
            _upper("held-out-coverage", "held_out_coverage_z", 3.0),
            _flag(
                "shift-no-guarantee",
                "distribution_free_guarantee_claimed_under_shift",
                False,
                "operational",
            ),
        ),
    ),
    _Q15Q17Row(
        "meshfree.hybrid-schwarz",
        "phs-rbf-fd-degree3-cloud-fv-grid-overlap-dirichlet-gmres-schwarz",
        "dirichlet",
        _HYBRID_GEOMETRY,
        "none",
        4096,
        2,
        "hybrid-overlap-schwarz",
        2220,
        "overlap-refinement",
        "AdaptivityP14c",
        (
            _flag(
                "schwarz-additive-convergence", "additive_schwarz_converged_all_levels"
            ),
            _flag(
                "schwarz-multiplicative-convergence",
                "multiplicative_schwarz_converged_all_levels",
            ),
            _flag("schwarz-accepted", "multiplicative_schwarz_accepted_all_levels"),
            _upper("component-residual", "component_residual_max", 1e-8),
            _upper("hybrid-vs-monolithic", "hybrid_over_monolithic_error_max", 2.0),
            _upper("overlap-transfer-rows", "transfer_refused_rows", 0.0),
        ),
    ),
)

# Q15 expected refusals: each passes only on the documented typed status.
_Q15_REFUSAL_ROWS: tuple[_Q15Q17Row, ...] = (
    _layer_row(
        "adaptive-bulk-capacity-refusal",
        144,
        "proposal-capacity-refusal",
        (_flag("capacity-refusal", "capacity_refused"),),
    ),
    _layer_row(
        "adaptive-bulk-rejected-coarsening",
        144,
        "rejected-coarsening-rollback",
        (
            _flag("rejected-coarsening", "refused_indicator_not_reduced"),
            _flag("rejected-coarsening-rollback", "source_composition_unchanged"),
            _flag("epochs-published", "epoch_published", False),
        ),
    ),
    _correction_row(
        "edge-moment-metric-learned-candidate-signed-minimum-norm-projection",
        "jittered-cartesian-nodal-control-volumes",
        "projection-tangent",
        256,
        "learned-correction-signed-refusal",
        36,
        "signed-projection-audit-refusal",
        (
            _flag("positivity-conflict", "positivity_conflict"),
            _flag("correction-admitted", "weights_published", False),
            _flag("moment-exactness", "moments_exact"),
        ),
    ),
    _correction_row(
        "edge-moment-metric-learned-log-linear-training",
        "cartesian-nodal-control-volumes",
        "implicit-adjoint",
        256,
        "learned-correction-failed-step",
        36,
        "failed-primal-step-rejection",
        (
            _flag("sign-margin", "projection_admissible", False),
            _flag("failed-step-rejected", "step_rejected"),
        ),
    ),
    _correction_row(
        "edge-geometry-empirical-coverage-refuse",
        "cartesian-nodal-control-volumes-refined-query",
        "none",
        1024,
        "learned-correction-coverage-refusal",
        121,
        "out-of-coverage-refusal",
        (
            _flag("coverage-refusal", "refused_outside_marginal_support"),
            _flag("coverage-control-admitted", "control_admitted"),
        ),
    ),
)


# Q17 thresholds, declared from the plan (P15/P16) before any run: the declared
# replay class of same-topology and support-epoch restarts is bitwise equality
# of accepted state, live history, RNG, controller, schedules, moments and
# evidence cursors; a changed owner partition transports values bitwise and
# continues under a reduction-order tolerance of 1e-12 (20 explicit steps of
# O(1) values, as declared by the runtime owner).
_EPOCH_REPLAY = (
    _flag("refused-epoch", "refused_epoch_category_step_rejected"),
    _flag("refusal-evidence-retained", "refusal_evidence_survives_epoch"),
    _flag("replay-class", "replay_class_bitwise_match"),
    _flag("replay-classification", "replay_bitwise_interrupted"),
    _flag("rng-controller", "rng_state_equal"),
    _flag("completed", "both_completed"),
)
_Q17_ROWS: tuple[_Q15Q17Row, ...] = (
    _restart_row(
        _EPOCH_RESTART,
        2,
        256,
        "ssprk33",
        "restart-refused-epoch-interrupt",
        64,
        "interrupt-across-refused-epoch",
        "RuntimeP15b",
        _EPOCH_REPLAY,
    ),
    _restart_row(
        (
            "gmls-degree3-reaction-diffusion-production-cli",
            "periodic",
            "periodic-unit-square-jittered-lattice",
        ),
        2,
        100,
        "imex-ars-222",
        "restart-cli-sigkill",
        100,
        "cli-sigkill-resume",
        "RuntimeP15b",
        (
            _flag("sigkill-interrupt", "killed_before_completion", True, "operational"),
            _flag("resume-from-commit", "resumed_from_committed_step"),
            _flag("replay-class", "final_record_equal"),
            _flag("completed", "reference_completed"),
            _flag(
                "memory-evidence", "sampled_peak_resident_reported", True, "operational"
            ),
        ),
    ),
    _restart_row(
        _MIGRATION_RESTART,
        2,
        256,
        "explicit-euler",
        "restart-ownership-migration",
        48,
        "changed-ownership-restart",
        "RuntimeP15b",
        (
            _flag("migration-committed", "migration_committed"),
            _flag("partition-changed", "partition_changed"),
            _flag("placement-only-roles", "migrated_roles_placement_only"),
            _flag("replay-class", "restored_bitwise"),
            _flag("replay-classification", "replay_bitwise"),
            _upper("continuation", "final_maximum_difference", 1e-12),
            _flag("completed", "final_completed"),
        ),
        device="cpu-forced-multi-device",
        reproducibility="bitwise-transport-reduction-order-continuation",
    ),
    _restart_row(
        _MOVING_RESTART,
        3,
        256,
        "moving-surface-fixed-step",
        "restart-moving-surface-history",
        48,
        "moving-surface-live-history",
        "MotionP7",
        (
            _flag("live-history", "history_window_full"),
            _flag("history-window", "archive_cursor_after_all_steps"),
            _flag("replay-class", "replay_class_bitwise_match"),
            _flag("completed", "reference_completed"),
        ),
    ),
    _restart_row(
        (
            "langmuir-bulk-surface-newton-production-run",
            "bulk-surface-interface",
            "sphere-interface-in-box",
        ),
        3,
        256,
        "implicit-newton-fixed-step",
        "restart-coupled-bulk-surface",
        128,
        "coupled-refused-newton-restart",
        "CouplingFinish",
        (
            _flag("refused-newton-evidence", "refused_status_solve_failed"),
            _flag("replay-class", "replay_class_bitwise_match"),
            _flag("completed", "reference_completed"),
        ),
    ),
)


def _stale_row(role: str) -> _Q15Q17Row:
    return _restart_row(
        _EPOCH_RESTART,
        2,
        256,
        "ssprk33",
        f"restart-stale-{role}",
        64,
        f"stale-{role}",
        "RuntimeP15b",
        (
            _flag(f"stale-{role}", "changed_role_named"),
            _flag("store-unchanged", "store_unchanged"),
        ),
    )


_Q17_REFUSAL_ROWS: tuple[_Q15Q17Row, ...] = (
    *(
        _stale_row(role)
        for role in ("geometry", "capacity", "precision", "source", "program")
    ),
    _restart_row(
        _EPOCH_RESTART,
        2,
        256,
        "ssprk33",
        "restart-truncated-archive",
        64,
        "truncated-archive",
        "RuntimeP15b",
        (_flag("truncated-archive", "refused_as_archive_corruption"),),
    ),
    _restart_row(
        _EPOCH_RESTART,
        2,
        256,
        "ssprk33",
        "restart-foreign-case",
        64,
        "foreign-case",
        "RuntimeP15b",
        (_flag("foreign-case", "refused_foreign_store"),),
    ),
    _restart_row(
        _MIGRATION_RESTART,
        2,
        256,
        "explicit-euler",
        "restart-ownership-migration",
        48,
        "identity-restart-into-changed-partition",
        "RuntimeP15b",
        (
            _flag(
                "identity-restart-partition-refused", "identity_restart_refused_partition"
            ),
        ),
        device="cpu-forced-multi-device",
        reproducibility="bitwise-transport-reduction-order-continuation",
    ),
    _restart_row(
        _MOVING_RESTART,
        3,
        256,
        "moving-surface-fixed-step",
        "restart-moving-history-window-refusal",
        48,
        "history-window-refusal",
        "MotionP7",
        (_flag("history-window-refusal", "refused_history_window"),),
    ),
)


def _q15q17_hybrid_orders(
    row: _Q15Q17Row, support: dict[str, SupportValue], record: dict[str, Any], /
) -> list[GateRecord]:
    levels = record["series"]["levels"]
    support_id = SupportTuple(row.capability, support).support_tuple_id
    scales = [level["spacing"] for level in levels]
    return [
        rate_gate(
            f"{owner}-order",
            scales,
            [level[f"{owner}_max_error"] for level in levels],
            1.5,
            metric=f"{owner}_max_error_observed_order",
            provenance=record["series"]["spacing_provenance"],
            support_tuple_id=support_id,
            coordinates={"cells": [level["cells"] for level in levels]},
        )
        for owner in ("cloud", "grid")
    ]


def _q15q17_execute(
    row: _Q15Q17Row,
    support: dict[str, SupportValue],
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    # One measurement serves every row (accepted or refusal) observing it.
    record = shared(
        (
            "Q15-Q17",
            row.workload,
            row.capacity,
            seed,
            replace(config, sizes=(row.capacity,)),
        ),
        lambda: ADAPTIVE_RESTART_WORKLOADS[row.workload](
            row.capacity, seed, replace(config, sizes=(row.capacity,))
        ),
    )
    gates = _q15q17_gates(row.criteria, record["metrics"])
    if row.workload == "hybrid-overlap-schwarz":
        gates += _q15q17_hybrid_orders(row, support, record)
    return {**record, "owner": row.owner, "gates": gates}


def _q15q17_plan(
    name: str, rows: tuple[_Q15Q17Row, ...], config: MeshfreeConfig, /
) -> tuple[PlannedRow, ...]:
    """Rows of the requested dimension whose declared capacity is requested.

    Each row's capacity is fixed by its scenario (levels, lattice or example);
    ``--sizes`` selects rows by that capacity and an unmatched size refuses.
    """
    declared = {row.capacity for row in rows}
    unmatched = sorted(set(config.sizes) - declared)
    if unmatched:
        raise ValueError(
            f"{name}: requested capacities {unmatched} match no declared scenario; "
            f"declared capacities are {sorted(declared)}."
        )
    dimensions = sorted({row.dimension for row in rows})
    if config.dimension not in dimensions:
        raise ValueError(f"{name} declares rows for dimensions {dimensions} only.")
    planned: list[PlannedRow] = []
    for row in rows:
        if row.dimension != config.dimension or row.capacity not in config.sizes:
            continue
        support = row.support(config)
        for seed in config.seeds:
            planned.append(
                PlannedRow(
                    row.capability,
                    support,
                    row.capacity,
                    seed,
                    row.label,
                    partial(_q15q17_execute, row, support, seed, config),
                )
            )
    return tuple(planned)


def _q15q17_sizes(rows: tuple[_Q15Q17Row, ...], /) -> tuple[int, ...]:
    return tuple(sorted({row.capacity for row in rows}))


_plan_q15 = partial(_q15q17_plan, "Q15", _Q15_ROWS)
_plan_q15_refusal = partial(_q15q17_plan, "Q15-refusal", _Q15_REFUSAL_ROWS)
_plan_q17 = partial(_q15q17_plan, "Q17", _Q17_ROWS)
_plan_q17_refusal = partial(_q15q17_plan, "Q17-refusal", _Q17_REFUSAL_ROWS)


# ---------------------------------------------------------------------------
# Q16 distributed campaigns.
#
# Forced host CPU rows run in one dedicated subprocess per (workloads, config)
# with XLA_FLAGS forcing four devices before JAX initializes; they prove
# functional distributed parity only. GPU/multi-host rows stay declared and
# are blocked by the driver when that hardware is absent; on such hardware
# they execute in-process on the driver's own runtime.
#
# Declared rounding bounds (benchmarks/meshfree_closure_distributed.py): a sum
# of m terms in precision eps is within m * eps of exact relative to the sum of
# magnitudes; two evaluations differ by twice that; with a safety factor two
# the declared tolerance is 4 * m * eps, where m is the measured row width k,
# column in-degree, owner-local slot count or owner count of that sum. Each
# such error is reported in units of its m * eps so one gate keeps the constant
# target 4 across capacities. Exact copies (halo gather, migration
# payload/points) must be bit-identical.

_Q16_FORCED_DEVICES = 4
_Q16_DERIVATIVE = "linear-action-jvp-vjp"
_Q16_ROUTES: dict[str, tuple[str, str]] = {
    "distributed-gmls-open": ("gmls-laplacian-shell-knn-halo-bind", "open-box"),
    "distributed-gmls-open-mixed": ("gmls-laplacian-shell-knn-halo-bind", "open-box"),
    "distributed-smoother-periodic": (
        "inverse-quadratic-neighbor-row-smoother-shell-knn-halo",
        "periodic-all-axes",
    ),
    "distributed-refusal": (
        "declared-distributed-capacity-ownership-precision-refusals",
        "open-box",
    ),
}
# Minimum owners a row must observe; the plan requires kNN shells crossing
# at least three owners wherever three or more owners exist.
_Q16_OWNERS: dict[MeshfreeDevice, int] = {
    "cpu-forced-multi-device": _Q16_FORCED_DEVICES,
    "gpu": 1,
    "multi-device-gpu": 2,
    "multi-host-cpu": 2,
    "multi-host-gpu": 2,
}
_Q16_REFUSALS = (
    "halo_tight_successful",
    "halo_overflow_refused",
    "halo_overflow_refused_rows_invalid",
    "owner_overflow_refused",
    "owner_overflow_refused_rows_invalid",
    "row_overflow_refused",
    "row_overflow_refused_rows_invalid",
    "missing_owner_refused",
    "local_capacity_tight_admitted",
    "local_capacity_refused",
    "migration_receive_overflow_refused",
    "migration_receive_overflow_rolled_back",
    "migration_packet_overflow_refused",
    "migration_packet_overflow_rolled_back",
    "incomplete_halo_bind_refused",
    "unsupported_float16_refused",
    "unsupported_bfloat16_refused",
    "certification_narrowing_refused",
    "certification_downcast_refused",
)


def _q16_support(
    workload: str,
    device: MeshfreeDevice,
    precision: str,
    capacity: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, SupportValue]:
    method, boundary = _Q16_ROUTES[workload]
    return meshfree_support(
        dimension=config.dimension,
        method=method,
        boundary=boundary,
        geometry_authority="unit-cube-stratified-samples",
        derivative_class="none" if workload == "distributed-refusal" else _Q16_DERIVATIVE,
        capacity=capacity,
        precision=precision,
        device=device,
    )


def _q16_accepted_workloads(config: MeshfreeConfig, /) -> tuple[tuple[str, str], ...]:
    """(workload, support precision) pairs the requested runtime precision admits."""
    if config.precision == "float64":
        return (
            ("distributed-gmls-open", "float64"),
            ("distributed-gmls-open-mixed", "mixed"),
            ("distributed-smoother-periodic", "float64"),
        )
    return (
        ("distributed-gmls-open", "float32"),
        ("distributed-smoother-periodic", "float32"),
    )


def _q16_forced_record(
    workloads: tuple[str, ...], artifact: str, config: MeshfreeConfig, /
) -> dict[str, Any]:
    from benchmarks.meshfree_closure_distributed import run_forced_device_workloads

    suffix = "" if config.precision == "float64" else f"_{config.precision}"
    output = PROJECT_ROOT / "benchmarks" / f"{artifact}{suffix}.json"
    return shared(
        ("Q16", workloads, artifact, config),
        lambda: run_forced_device_workloads(
            workloads, config, devices=_Q16_FORCED_DEVICES, output=output
        ),
    )


def _q16_recorded_row(
    record: dict[str, Any], workload: str, capacity: int, seed: int, /
) -> dict[str, Any]:
    for row in record["rows"]:
        if (
            row["workload"] == workload
            and row["capacity"] == capacity
            and row["seed"] == seed
        ):
            return row
    raise ValueError(
        f"Forced-device process produced no {workload} row at {capacity}/{seed}."
    )


def _q16_bound_units(row: dict[str, Any], precision: str, /) -> dict[str, float]:
    """Rounding errors divided by their declared unit m * eps (see section header)."""
    metrics = row["metrics"]
    # Mixed rows store coefficients/fields/output in float32: its eps bounds them.
    eps = float(np.finfo(np.float64 if precision == "float64" else np.float32).eps)
    k = metrics["maximum_row_entries"]
    m = metrics["maximum_column_entries"]
    local = row["local_capacity"]
    owners = metrics["owner_count"]
    units = {
        "halo_duality_error": owners + 1,
        "apply_parity_error": k,
        "transpose_parity_error": m,
        "adjoint_parity_error": m + 2,
        "duality_error": k + m,
        "weighted_duality_error": k + m + local + owners + 2,
        "jvp_parity_error": k,
        "vjp_parity_error": m,
        "conservation_error": m + local + owners,
        "reduction_sum_error": local + owners,
        "reduction_norm_error": local + owners,
        "rebind_parity_error": k,
    }
    scaled = {
        f"{key}_in_bound_units": metrics[key] / (terms * eps)
        for key, terms in units.items()
    }
    if precision == "mixed":
        # Against the float64 fit only float32 coefficient storage and output
        # round (2 eps32 of |A||x|); float64 accumulation adds no k-term.
        scaled["mixed_accuracy_error_in_eps32"] = metrics["mixed_accuracy_error"] / float(
            np.finfo(np.float32).eps
        )
    return scaled


def _q16_hardware_observed(row: dict[str, Any], device: MeshfreeDevice, /) -> bool:
    """Whether the row ran on the declared device class (not a substitute)."""
    devices = row["devices"]
    accelerated = any(
        platform in ("gpu", "cuda", "rocm") for platform in devices["platforms"]
    )
    match device:
        case "cpu-forced-multi-device":
            return (
                devices["forced_host_devices"]
                and devices["device_count"] == _Q16_FORCED_DEVICES
            )
        case "gpu" | "multi-device-gpu":
            return accelerated and devices["process_count"] == 1
        case "multi-host-cpu":
            return devices["process_count"] > 1 and not accelerated
        case "multi-host-gpu":
            return devices["process_count"] > 1 and accelerated
        case "cpu":
            raise ValueError("Q16 declares no single-device CPU rows.")


def _q16_accepted_gates(
    row: dict[str, Any], precision: str, device: MeshfreeDevice, /
) -> tuple[dict[str, Any], list[GateRecord]]:
    metrics = {
        **row["metrics"],
        **_q16_bound_units(row, precision),
        "declared_hardware_observed": _q16_hardware_observed(row, device),
    }
    minimum_owners = _Q16_OWNERS[device]
    bound = "4 = safety 2 x two evaluations of m * eps"
    gates = [
        boolean_gate(
            "declared-hardware",
            metrics,
            "declared_hardware_observed",
            evidence_kind="operational",
        ),
        lower_gate(
            "owner-count",
            metrics,
            "owner_count",
            minimum_owners,
            evidence_kind="operational",
        ),
        upper_gate("knn-completeness", metrics, "mismatched_certified_rows", 0),
        upper_gate("knn-status", metrics, "incomplete_rows", 0),
        upper_gate(
            "knn-distance", metrics, "distance_error", metrics["certification_margin"]
        ),
        lower_gate(
            "knn-shell-crossing", metrics, "max_row_owners", min(3, minimum_owners)
        ),
        boolean_gate("knn-targeted-communication", metrics, "targets_not_replicated"),
        upper_gate("radius-pair-once-missing", metrics, "pair_missing", 0),
        upper_gate("radius-pair-once-extra", metrics, "pair_extra", 0),
        upper_gate("radius-pair-once-duplicates", metrics, "pair_duplicates", 0),
        upper_gate("radius-status", metrics, "pair_incomplete_rows", 0),
        boolean_gate("halo-complete", metrics, "halo_successful"),
        upper_gate("halo-gather-exact", metrics, "halo_gather_errors", 0),
        upper_gate("halo-exactly-once", metrics, "halo_exactly_once_errors", 0),
        *(
            upper_gate(name, metrics, f"{key}_in_bound_units", 4, unit=f"m*eps; {bound}")
            for name, key in (
                ("halo-duality", "halo_duality_error"),
                ("apply-parity", "apply_parity_error"),
                ("transpose-parity", "transpose_parity_error"),
                ("adjoint-parity", "adjoint_parity_error"),
                ("duality", "duality_error"),
                ("weighted-duality", "weighted_duality_error"),
                ("jvp-parity", "jvp_parity_error"),
                ("vjp-parity", "vjp_parity_error"),
                ("conservation", "conservation_error"),
                ("reduction-sum", "reduction_sum_error"),
                ("reduction-norm", "reduction_norm_error"),
                ("rebind-parity", "rebind_parity_error"),
            )
        ),
        boolean_gate("migration-committed", metrics, "migration_committed"),
        boolean_gate("migration-epoch", metrics, "migration_epoch_advanced"),
        boolean_gate("migration-stable-ids", metrics, "migration_ids_preserved"),
        upper_gate("migration-owners", metrics, "migration_owner_errors", 0),
        upper_gate("migration-payload", metrics, "migration_payload_error", 0),
        upper_gate("migration-id-payload", metrics, "migration_id_payload_errors", 0),
        upper_gate("migration-points", metrics, "migration_points_error", 0),
        upper_gate(
            "requery-completeness", metrics, "requery_mismatched_certified_rows", 0
        ),
        upper_gate("requery-status", metrics, "requery_incomplete_rows", 0),
        boolean_gate("precision-roles", metrics, "precision_roles_match"),
    ]
    if precision == "mixed":
        gates.append(
            upper_gate(
                "mixed-accuracy",
                metrics,
                "mixed_accuracy_error_in_eps32",
                4,
                unit="eps32",
            )
        )
    return metrics, gates


def _q16_producer(record: dict[str, Any], precision: str, /) -> dict[str, Any]:
    """Bind the child process that actually prepared and executed this row."""
    return {
        "environment": record["environment"],
        "identity": record["identity"],
        "driver_dependencies": record["driver_dependencies"],
        "backend": record["environment"]["jax"]["backend"],
        # The forced launcher starts one isolated Python process; device
        # inventory comes from that child's captured environment, even on failure.
        "topology": f"1-process-{len(record['environment']['jax']['devices'])}-device",
        "precision": precision,
    }


def _q16_originating_failure(
    record: dict[str, Any], row: RowRecord, precision: str, /
) -> RowRecord | None:
    if row["status"] == "measured":
        return None
    status: GateStatus = "blocked" if row["status"] == "declared-refusal" else "failed"
    return {
        **row,
        "producer": _q16_producer(record, precision),
        "subprocess": record["subprocess"],
        "gates": [
            _execution_gate(
                status,
                row.get("reason", row.get("failure", {}).get("message", row["status"])),
            )
        ],
    }


def _q16_forced_row(
    workload: str, precision: str, capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    workloads = tuple(name for name, _ in _q16_accepted_workloads(config))
    record = _q16_forced_record(workloads, "meshfree_closure_distributed", config)
    row = _q16_recorded_row(record, workload, capacity, seed)
    failure = _q16_originating_failure(record, row, precision)
    if failure is not None:
        return failure
    metrics, gates = _q16_accepted_gates(row, precision, "cpu-forced-multi-device")
    return {
        **row,
        "metrics": metrics,
        "producer": _q16_producer(record, precision),
        "subprocess": {**record["subprocess"], "shared_by": "every row of this process"},
        "gates": gates,
    }


def _q16_hardware_row(
    workload: str,
    precision: str,
    device: MeshfreeDevice,
    capacity: int,
    seed: int,
    config: MeshfreeConfig,
    /,
) -> dict[str, Any]:
    """Executed only where the driver itself runs on the declared hardware."""
    from benchmarks.meshfree_closure_distributed import DISTRIBUTED_WORKLOADS

    row = DISTRIBUTED_WORKLOADS[workload](capacity, seed, config)
    metrics, gates = _q16_accepted_gates(row, precision, device)
    return {**row, "metrics": metrics, "gates": gates}


def _plan_q16(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    forced = tuple(
        PlannedRow(
            "meshfree.distributed",
            _q16_support(workload, "cpu-forced-multi-device", precision, 4096, config),
            size,
            seed,
            f"forced-cpu-{precision}-{workload}",
            partial(_q16_forced_row, workload, precision, size, seed, config),
        )
        for workload, precision in _q16_accepted_workloads(config)
        for size in config.sizes
        for seed in config.seeds
    )
    hardware = tuple(
        PlannedRow(
            "meshfree.distributed",
            _q16_support(workload, device, precision, 1048576, config),
            size,
            seed,
            f"{device}-{precision}-{workload}",
            partial(_q16_hardware_row, workload, precision, device, size, seed, config),
        )
        for device in ("gpu", "multi-device-gpu", "multi-host-cpu", "multi-host-gpu")
        for workload, precision in _q16_accepted_workloads(config)
        if workload != "distributed-smoother-periodic"
        for size in config.sizes
        for seed in config.seeds
    )
    return forced + hardware


def _q16_sweep(rows: Sequence[RowRecord], config: MeshfreeConfig, /) -> list[GateRecord]:
    """Measured capacity sweep per executed support tuple; slopes are observed, not claims."""
    from benchmarks.meshfree_scaling import observed_slopes

    gates: list[GateRecord] = []
    executed = [
        {**row, "status": "measured"}
        for row in rows
        if row["status"] == "passed" and "phases" in row
    ]
    for support_id in sorted({row["support_tuple_id"] for row in executed}):
        scoped = [row for row in executed if row["support_tuple_id"] == support_id]
        sizes = sorted({row["capacity"] for row in scoped})
        gates.append(
            gate(
                "capacity-sweep",
                "passed" if len(sizes) >= 3 else "unexecuted",
                metric="measured_capacities",
                comparison="greater-than-or-equal",
                target=3,
                observed=len(sizes),
                evidence_kind="performance",
                reason=None
                if len(sizes) >= 3
                else "Fewer than three measured capacities",
                support_tuple_id=support_id,
                coordinates={
                    "capacities": sizes,
                    "local_capacities": [row["local_capacity"] for row in scoped],
                    "observed_slopes": {
                        phase: observed_slopes(scoped, phase)
                        for phase in (
                            "search",
                            "assembly",
                            "communication-migration",
                            "warm",
                            "jvp",
                            "vjp",
                        )
                    },
                },
            )
        )
    return gates


def _q16_refusal_row(
    capacity: int, seed: int, config: MeshfreeConfig, /
) -> dict[str, Any]:
    record = _q16_forced_record(
        ("distributed-refusal",), "meshfree_closure_distributed_refusal", config
    )
    row = _q16_recorded_row(record, "distributed-refusal", capacity, seed)
    failure = _q16_originating_failure(record, row, config.precision)
    if failure is not None:
        return failure
    metrics = row["metrics"]
    return {
        **row,
        "producer": _q16_producer(record, config.precision),
        "subprocess": record["subprocess"],
        "gates": [
            *(
                boolean_gate(key.removesuffix("_refused").replace("_", "-"), metrics, key)
                for key in _Q16_REFUSALS
            ),
            upper_gate(
                "halo-overflow-survivors-exact",
                metrics,
                "halo_overflow_surviving_mismatches",
                0,
            ),
        ],
    }


def _plan_q16_refusal(config: MeshfreeConfig, /) -> tuple[PlannedRow, ...]:
    return tuple(
        PlannedRow(
            "meshfree.distributed",
            _q16_support(
                "distributed-refusal",
                "cpu-forced-multi-device",
                config.precision,
                4096,
                config,
            ),
            size,
            seed,
            f"forced-cpu-{config.precision}-declared-refusals",
            partial(_q16_refusal_row, size, seed, config),
        )
        for size in config.sizes
        for seed in config.seeds
    )


CAMPAIGNS: dict[str, Campaign] = {
    "Q1": Campaign(
        "polynomial-derivatives-and-measured-order",
        "accepted",
        (512, 1024, 2048, 4096),
        2,
        _plan_q1,
        _q1_convergence,
    ),
    "Q1-refusal": Campaign(
        "degenerate-cloud-row-refusal", "expected-refusal", (64,), 2, _plan_q1_refusal
    ),
    "Q2": Campaign(
        "boundary-elliptic-error-and-order",
        "accepted",
        (256, 576, 1024, 2304),
        2,
        _plan_q2,
        _q2_convergence,
    ),
    "Q3": Campaign(
        "fine-system-preconditioner-comparison", "accepted", (64, 128), 2, _plan_q3
    ),
    "Q4": Campaign(
        "conservative-exterior-and-coordinate-derivatives",
        "accepted",
        (25, 49),
        2,
        _plan_q4,
    ),
    "Q4-refusal": Campaign(
        "nonnegative-metric-infeasibility-refusal",
        "expected-refusal",
        (25, 49),
        2,
        _plan_q4_refusal,
    ),
    "Q5": Campaign(
        "sphere-torus-laplace-beltrami-order",
        "accepted",
        (1024, 2048, 4096, 8192),
        3,
        _plan_q5,
        _q5_convergence,
    ),
    "Q6": Campaign(
        "expanding-surface-and-repeated-remap", "accepted", (48, 64), 3, _plan_q6
    ),
    # The bulk-surface example documents a 128-point minimum total capacity.
    "Q7": Campaign(
        "langmuir-exchange-and-moving-window", "accepted", (128, 256), 3, _plan_q7
    ),
    "Q8": Campaign(
        "learned-law-and-implicit-derivatives",
        "accepted",
        (12, 24, 256, 512),
        2,
        _plan_q8,
    ),
    "Q8-refusal": Campaign(
        "learned-law-coverage-and-failed-newton-refusal",
        "expected-refusal",
        (12, 24),
        2,
        _plan_q8_refusal,
    ),
    "Q12": Campaign(
        "geometry-authorized-higher-forms-and-abstract-clique-research",
        "accepted",
        (16, 64, 144, 256),
        2,
        _plan_q12,
        _q12_order,
    ),
    "Q12-refusal": Campaign(
        "abstract-clique-and-authority-audit-refusals",
        "expected-refusal",
        (64,),
        2,
        _plan_q12_refusal,
    ),
    "Q14": Campaign(
        "fixed-support-strict-active-frozen-remap-sensitivities",
        "accepted",
        (64, 256, 1024),
        2,
        _plan_q14,
    ),
    "Q14-refusal": Campaign(
        "envelope-tie-active-set-and-epoch-derivative-refusals",
        "expected-refusal",
        (64,),
        2,
        _plan_q14_refusal,
    ),
    "Q10": Campaign(
        "incompressible-and-lagrangian-particle-flow",
        "accepted",
        (144, 256, 576, 1024),
        2,
        _plan_q10,
        _q10_orders,
    ),
    "Q10-refusal": Campaign(
        "cfl-incompatible-source-coercivity-courant-and-uncovered-transfer-refusals",
        "expected-refusal",
        (144,),
        2,
        _plan_q10_refusal,
    ),
    "Q11": Campaign(
        "elasticity-finite-strain-mixed-and-fluid-structure",
        "accepted",
        (169, 289, 625, 1024),
        2,
        _plan_q11,
        _q11_orders,
    ),
    "Q11-fluid-structure": Campaign(
        "block-ghost-fluid-structure-equations-and-work",
        "accepted",
        (169, 289, 625, 1024),
        2,
        _plan_q11_fsi,
        _q11_orders,
    ),
    "Q11-fluid-structure-energy": Campaign(
        "asymptotic-block-ghost-fluid-structure-energy",
        "accepted",
        (578, 968, 1250, 2178),
        2,
        _plan_q11_fsi_energy,
        _q11_orders,
    ),
    "Q11-refusal": Campaign(
        "newton-rollback-floating-body-and-inf-sup-refusals",
        "expected-refusal",
        (81,),
        2,
        _plan_q11_refusal,
    ),
    "Q11-surface": Campaign(
        "closed-surface-strain-form-stokes",
        "accepted",
        (256, 512, 1024, 2048),
        3,
        _plan_q11_surface,
        _q11_surface_orders,
    ),
    "Q11-surface-refusal": Campaign(
        "surface-stokes-starved-krylov-refusal",
        "expected-refusal",
        (256,),
        3,
        _plan_q11_surface_refusal,
    ),
    "Q9": Campaign(
        "bulk-adr-ale-limited-transport-order-cfl-positivity-conservation",
        "accepted",
        (256, 576, 1024, 2304),
        2,
        _plan_q9,
        _q9_convergence,
    ),
    "Q9-refusal": Campaign(
        "cfl-positivity-support-and-topology-trust-refusals",
        "expected-refusal",
        (256,),
        2,
        _plan_q9_refusal,
    ),
    "Q13": Campaign(
        "joint-transfer-and-physical-split-merge-epochs",
        "accepted",
        # P16 (Main, TransferP8): 27 targets (3 per axis) is pre-asymptotic, so
        # the sweep is 125..1331 targets (3375 sources <= 4096 max points), with
        # k = 4 * comb(d+2, 2) routes and the Lebesgue constant gated <= 10; the
        # degree-2 order threshold 2.5 is kept.
        (125, 343, 729, 1331),
        3,
        _plan_q13,
        _q13_convergence,
    ),
    "Q13-joint-signed": Campaign(
        "joint-quadratic-transfer-admitted-support-replay",
        "accepted",
        (125, 343, 729, 1331),
        3,
        _plan_q13_signed,
        _q13_convergence,
    ),
    "Q13-refusal": Campaign(
        "uncovered-obstruction-infeasible-and-missing-history-refusals",
        "expected-refusal",
        (125,),
        3,
        _plan_q13_refusal,
    ),
    "Q16": Campaign(
        "distributed-relations-actions-migration-and-precision",
        "accepted",
        (1024, 2048, 4096),
        2,
        _plan_q16,
        _q16_sweep,
    ),
    "Q16-refusal": Campaign(
        "distributed-capacity-ownership-and-precision-refusals",
        "expected-refusal",
        (256,),
        2,
        _plan_q16_refusal,
    ),
    "Q15": Campaign(
        "adaptive-learned-calibrated-hybrid-workflows",
        "accepted",
        _q15q17_sizes(_Q15_ROWS),
        2,
        _plan_q15,
    ),
    "Q15-refusal": Campaign(
        "adaptive-capacity-rollback-learned-audit-and-coverage-refusals",
        "expected-refusal",
        _q15q17_sizes(_Q15_REFUSAL_ROWS),
        2,
        _plan_q15_refusal,
    ),
    "Q17": Campaign(
        "long-run-interrupt-restart-and-changed-ownership",
        "accepted",
        _q15q17_sizes(_Q17_ROWS),
        2,
        _plan_q17,
    ),
    "Q17-refusal": Campaign(
        "stale-identity-truncated-foreign-and-partition-restart-refusals",
        "expected-refusal",
        _q15q17_sizes(_Q17_REFUSAL_ROWS),
        2,
        _plan_q17_refusal,
    ),
    # Traction-face manufactured elasticity order on lattice sides 25/33/49/65,
    # finer than Q11's 13/17/25/32; threshold unchanged (1.5).
    "Q11-elasticity-order": Campaign(
        "traction-face-manufactured-elasticity-observed-order",
        "accepted",
        (625, 1089, 2401, 4225),
        2,
        _plan_q11_elasticity_order,
        _manufactured_elasticity_orders,
    ),
}


# ---------------------------------------------------------------------------
# Validation, execution and evidence


def _profiles() -> dict[str, CapabilityProfile]:
    return {profile.capability: profile for profile in meshfree_candidate_profiles()}


def _hardware_block(device: str, /) -> str | None:
    """Return why a declared device row cannot execute on this host, if so."""
    selected = parse(device, MeshfreeDevice, "device")
    platforms = [item.platform for item in jax.devices()]
    accelerators = [
        platform for platform in platforms if platform in ("gpu", "cuda", "rocm")
    ]
    match selected:
        case "cpu" | "cpu-forced-multi-device":
            return None
        case "gpu":
            return (
                None
                if accelerators
                else "blocked: hardware unavailable (no GPU device in jax.devices())"
            )
        case "multi-device-gpu":
            return (
                None
                if len(accelerators) > 1
                else "blocked: hardware unavailable (fewer than two GPU devices)"
            )
        case "multi-host-cpu":
            return (
                None
                if jax.process_count() > 1
                else "blocked: hardware unavailable (single-process host; no multi-host collective)"
            )
        case "multi-host-gpu":
            return (
                None
                if jax.process_count() > 1 and accelerators
                else "blocked: hardware unavailable (no multi-host GPU collective)"
            )


def _validate(
    name: str,
    rows: Sequence[PlannedRow],
    config: MeshfreeConfig,
    profiles: dict[str, CapabilityProfile],
    /,
) -> list[tuple[PlannedRow, SupportTuple, str | None]]:
    """Refuse undeclared support/capacity/dimension/precision before execution."""
    validated: list[tuple[PlannedRow, SupportTuple, str | None]] = []
    for row in rows:
        profile = profiles.get(row.capability)
        if profile is None:
            raise ValueError(f"{name}: {row.capability!r} is not a declared capability.")
        support = SupportTuple(row.capability, row.support)
        if not profile.supports(support):
            raise ValueError(
                f"{name}: row {row.label!r} requests undeclared support {row.support}."
            )
        if row.support["dimension"] != config.dimension:
            raise ValueError(
                f"{name}: requested dimension {config.dimension} is not the declared "
                f"dimension {row.support['dimension']} of row {row.label!r}."
            )
        precision = row.support["precision"]
        if precision != config.precision and not (
            precision == "mixed" and config.precision == "float64"
        ):
            raise ValueError(
                f"{name}: requested precision {config.precision} is not declared for "
                f"row {row.label!r} ({precision})."
            )
        declared = row.support["capacity"]
        if type(declared) is not int or row.capacity > declared:
            raise ValueError(
                f"{name}: requested capacity {row.capacity} exceeds the declared "
                f"support capacity {declared} of row {row.label!r}."
            )
        validated.append((row, support, _hardware_block(str(row.support["device"]))))
    return validated


def _row_status(gates: Sequence[GateRecord], /) -> GateStatus:
    statuses = {item["status"] for item in gates}
    if "failed" in statuses:
        return "failed"
    if "blocked" in statuses:
        return "blocked"
    if not gates or "unexecuted" in statuses:
        return "unexecuted"
    return "passed"


def _execution_gate(status: GateStatus, reason: str, /) -> GateRecord:
    return gate(
        "execution",
        status,
        metric="row-executed",
        comparison="equal",
        target=1,
        observed=0,
        evidence_kind="operational",
        reason=reason,
    )


def _execute(row: PlannedRow, support: SupportTuple, block: str | None, /) -> RowRecord:
    base: RowRecord = {
        "label": row.label,
        "capability": row.capability,
        "support": dict(row.support),
        "support_tuple_id": support.support_tuple_id,
        "capacity": row.capacity,
        "seed": row.seed,
    }
    if block is not None:
        return {**base, "status": "blocked", "gates": [_execution_gate("blocked", block)]}
    started = time.perf_counter()
    cpu = time.process_time()
    with BenchmarkExecutionScope() as execution:
        try:
            result = row.execute()
        except DeclaredCapacityRefusal as error:
            status: GateStatus = "failed" if execution.executed else "blocked"
            reason = (
                "post-execution-capacity-refusal"
                if execution.executed
                else "declared-capacity-refusal"
            ) + f": {error}"
            return {
                **base,
                **execution.record(),
                "status": status,
                "failure": failure_record(error),
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                "gates": [_execution_gate(status, reason)],
            }
        except Exception as error:  # Verbatim partial evidence; no retry.
            return {
                **base,
                **execution.record(),
                "status": "failed",
                "failure": failure_record(error),
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                "gates": [_execution_gate("failed", f"{type(error).__name__}: {error}")],
            }
    record = {**base, **result}
    record["wall_seconds"] = time.perf_counter() - started
    record["cpu_seconds"] = time.process_time() - cpu
    record["gates"] = list(result.get("gates", ()))
    record["status"] = _row_status(record["gates"])
    return record


def _evidence_label(statuses: Sequence[str], /) -> tuple[str, GateStatus]:
    if "failed" in statuses:
        return "failed", "failed"
    if "blocked" in statuses:
        return "inconclusive", "blocked"
    if "unexecuted" in statuses:
        return "inconclusive", "unexecuted"
    return "passed", "passed"


def _producer_context(row: RowRecord, context: dict[str, Any], /) -> dict[str, Any]:
    producer = row.get("producer")
    if producer is None:
        return {
            **context,
            "precision": row.get("support", {}).get("precision", context["precision"]),
        }
    return {
        **context,
        "source_identity": producer["identity"],
        "driver_dependencies": producer["driver_dependencies"],
        "build_id": canonical_fingerprint(
            {
                "identity": producer["identity"],
                "driver_dependencies": producer["driver_dependencies"],
            }
        ),
        "environment_id": canonical_fingerprint(producer["environment"]),
        "backend": producer["backend"],
        "topology": producer["topology"],
        "precision": producer["precision"],
        "run_spec_id": canonical_fingerprint(
            {"run_spec_id": context["run_spec_id"], "producer": producer}
        ),
    }


def _campaign_evidence(
    name: str,
    rows: Sequence[RowRecord],
    aggregate: Sequence[GateRecord],
    context: dict[str, Any],
    /,
) -> dict[str, Any]:
    """Typed criterion -> start -> observation -> evidence chain per tuple and gate."""
    groups: dict[tuple[str, str, str], list[tuple[GateRecord, dict[str, Any]]]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    row_contexts: dict[str, set[str]] = {}
    for row in rows:
        producer_context = _producer_context(row, context)
        producer_id = canonical_fingerprint(producer_context)
        contexts[producer_id] = producer_context
        row_contexts.setdefault(row["support_tuple_id"], set()).add(producer_id)
        for item in row["gates"]:
            groups.setdefault(
                (row["support_tuple_id"], item["gate"], producer_id), []
            ).append(
                (
                    item,
                    {
                        "capacity": row["capacity"],
                        "seed": row["seed"],
                        "label": row["label"],
                        **{
                            key: row[key]
                            for key in (
                                "failure",
                                "producer",
                                "phases",
                                "compiler",
                                "reservations",
                                "execution_stage",
                                "reserved_working_set_bytes",
                                "reservation_scope",
                                "consumer_source_fingerprint",
                                "oracle_provenance",
                            )
                            if key in row
                        },
                    },
                )
            )
    for item in aggregate:
        producer_ids = row_contexts.get(item["support_tuple_id"], set())
        if not producer_ids:
            producer_id = canonical_fingerprint(context)
            contexts[producer_id] = context
            producer_ids = {producer_id}
        if len(producer_ids) != 1:
            raise ValueError(
                f"{name}: aggregate gate {item['gate']!r} mixes executing producers."
            )
        producer_id = next(iter(producer_ids))
        groups.setdefault(
            (item["support_tuple_id"], item["gate"], producer_id), []
        ).append((item, item.get("coordinates", {})))
    records: list[dict[str, Any]] = []
    evidence: list[QualificationEvidence] = []
    predicates: dict[str, dict[str, str]] = {}
    for (support_id, gate_name, producer_id), observations in sorted(groups.items()):
        producer_context = contexts[producer_id]
        declared = {
            (
                item["metric"],
                item["comparison"],
                item["target"],
                item["unit"],
                item["evidence_kind"],
            )
            for item, _ in observations
        }
        if len(declared) != 1:
            raise ValueError(
                f"{name}: gate {gate_name!r} of one support tuple mixes declared criteria."
            )
        metric, comparison, target, unit, kind = next(iter(declared))
        outcome, label = _evidence_label([item["status"] for item, _ in observations])
        criterion = QualificationCriterion(
            support_tuple_id=support_id,
            metric=metric,
            unit=unit,
            comparison=comparison,
            target=target,
            aggregation="every-observation",
            uncertainty="deterministic-seeded-single-execution",
            applicability=f"{name}:{gate_name}",
            approval_id=APPROVAL_ID,
            issued_at=context["criteria_at"],
        )
        start = CampaignStartRecord(
            campaign_spec_id=producer_context["campaign_spec_id"],
            criterion_id=criterion.criterion_id,
            resolved_run_spec_id=producer_context["run_spec_id"],
            support_tuple_id=support_id,
            started_at=context["started_at"],
        )
        raw = {
            "kind": "meshfree-qualification-observation",
            "campaign": name,
            "gate": gate_name,
            "producer_context": producer_context,
            "observations": [
                {
                    **coordinates,
                    "status": item["status"],
                    "observed": item["observed"],
                    "reason": item["reason"],
                }
                for item, coordinates in observations
            ],
        }
        raw_id = canonical_fingerprint(_json_ready(raw))
        observation = CampaignObservationRecord(
            start_record_id=start.start_record_id,
            campaign_spec_id=producer_context["campaign_spec_id"],
            criterion_id=criterion.criterion_id,
            resolved_run_spec_id=producer_context["run_spec_id"],
            support_tuple_id=support_id,
            raw_artifact_ids=(raw_id,),
            observed_at=context["observed_at"],
        )
        reasons = sorted({item["reason"] for item, _ in observations if item["reason"]})
        item = QualificationEvidence(
            kind,
            outcome,
            (support_id,),
            build_id=producer_context["build_id"],
            environment_id=producer_context["environment_id"],
            backend=producer_context["backend"],
            topology=producer_context["topology"],
            precision=producer_context["precision"],
            reduction="every-observation",
            replay_id=producer_context["run_spec_id"],
            criteria_ids=(criterion.criterion_id,),
            raw_artifact_ids=(raw_id,),
            campaign_start_record_ids=(start.start_record_id,),
            campaign_observation_record_ids=(observation.observation_record_id,),
            reviewer_id=REVIEWER_ID,
            issued_at=context["issued_at"],
            expires_at=context["issued_at"] + EVIDENCE_VALIDITY_NS,
            # Evidence reasons are canonical one-line identifiers; the verbatim
            # multi-line failure stays in the raw observation and row record.
            reason=" ".join(
                (
                    f"{label}: "
                    + (
                        "; ".join(reasons)
                        if reasons
                        else "observed against declared criterion"
                    )
                ).split()
            ),
            requalification_triggers=(
                "environment-change",
                "source-change",
                "threshold-change",
            ),
        )
        validate_qualification_causality(criterion, start, observation, item)
        predicate = f"{name}:{support_id[:16]}:{gate_name}"
        predicates[predicate] = {
            "evidence_kind": kind,
            "criterion_id": criterion.criterion_id,
            "subject_id": support_id,
        }
        evidence.append(item)
        records.append(
            {
                "support_tuple_id": support_id,
                "gate": gate_name,
                "status": label,
                "criterion": criterion.to_record(),
                "campaign_start": start.to_record(),
                "raw_observation": {**_json_ready(raw), "raw_artifact_id": raw_id},
                "campaign_observation": observation.to_record(),
                "evidence": item.to_record(),
            }
        )
    if not predicates:
        return {"records": [], "matrix": None, "coverage": None}
    matrix = QualificationMatrix(predicates)
    coverage = matrix.evaluate(tuple(evidence), at_time=context["issued_at"])
    return {
        "records": records,
        "matrix": matrix.to_record(),
        "coverage": coverage.to_record(),
    }


def _json_ready(value: Any, /) -> Any:
    """Finite JSON: non-finite floats are retained as explicit strings, not NaN."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.ndarray, jax.Array)):
        return _json_ready(np.asarray(value).tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else f"nonfinite:{number}"
    return value


def _release_gates(name: str, /) -> list[GateRecord]:
    return [
        blocked_gate(
            "independent-reference",
            metric="independent-physical-reference",
            comparison="equal",
            target=1,
            reason="blocked: manufactured/analytic oracles prove numerical behavior only; "
            "an independent physical reference with provenance, tolerances and rights "
            "is an external prerequisite",
        ),
        blocked_gate(
            "independent-review",
            metric="independent-reviewer-decision",
            comparison="equal",
            target=1,
            evidence_kind="operational",
            reason="blocked: producer evidence is unreviewed; release needs an "
            "independent reviewer, trust record and promotion decision",
        ),
        gate(
            "resource-envelope",
            "unexecuted",
            metric="approved-resource-envelope",
            comparison="equal",
            target=1,
            observed=None,
            evidence_kind="performance",
            reason="Sampled phase peaks are not certified upper bounds; no approved "
            f"resource envelope exists for {name}",
        ),
    ]


def _summary(
    name: str,
    campaign: Campaign,
    rows: Sequence[RowRecord],
    aggregate: Sequence[GateRecord],
    /,
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for support_id in sorted({row["support_tuple_id"] for row in rows}):
        scoped = [row for row in rows if row["support_tuple_id"] == support_id]
        gates = [item for row in scoped for item in row["gates"]] + [
            item for item in aggregate if item["support_tuple_id"] == support_id
        ]
        per_gate: dict[str, list[str]] = {}
        for item in gates:
            per_gate.setdefault(item["gate"], []).append(item["status"])
        support = scoped[0]["support"]
        table.append(
            {
                "campaign": name,
                "scenario": campaign.scenario,
                "capability": scoped[0]["capability"],
                "support_tuple_id": support_id,
                "method": support["method"],
                "boundary": support["boundary"],
                "geometry_authority": support["geometry-authority"],
                "dimension": support["dimension"],
                "device": support["device"],
                "precision": support["precision"],
                "capacities": sorted({row["capacity"] for row in scoped}),
                "status": _row_status(gates),
                "gates": {
                    key: _evidence_label(value)[1]
                    for key, value in sorted(per_gate.items())
                },
                "wall_seconds": sum(row.get("wall_seconds", 0.0) for row in scoped),
                "cpu_seconds": sum(row.get("cpu_seconds", 0.0) for row in scoped),
                "compile_count": sum(
                    row.get("phases", {}).get(phase, {}).get("compile_count", 0)
                    for row in scoped
                    if row.get("shared_execution") != "reused"
                    for phase in row.get("phases", {})
                ),
            }
        )
    return table


def run(
    base: MeshfreeConfig,
    campaigns: tuple[str, ...],
    /,
    *,
    sizes: tuple[int, ...] | None = None,
    dimension: int | None = None,
    elliptic: EllipticSelection = EllipticSelection(),
) -> dict[str, Any]:
    """Plan and validate every selected campaign, then execute and record evidence."""
    unknown = sorted(set(campaigns) - set(CAMPAIGNS))
    if not campaigns or unknown or len(set(campaigns)) != len(campaigns):
        raise ValueError(
            "Select distinct registered campaigns; unknown: " + ", ".join(unknown)
        )
    # Morton codes, sort keys and stable IDs are 64-bit for every precision;
    # ``precision`` is the declared coordinate/field dtype of each row.
    jax.config.update("jax_enable_x64", True)
    profiles = _profiles()
    criteria_at = time.time_ns()
    planned: list[
        tuple[
            str,
            Campaign,
            MeshfreeConfig,
            list[tuple[PlannedRow, SupportTuple, str | None]],
        ]
    ] = []
    for name in campaigns:
        campaign = CAMPAIGNS[name]
        if name == "Q2":
            campaign = replace(
                campaign,
                plan=partial(_plan_q2, selection=elliptic),
                aggregate=partial(_q2_convergence, degree=elliptic.degree),
            )
        config = replace(
            base,
            sizes=campaign.sizes if sizes is None else sizes,
            dimension=campaign.dimension if dimension is None else dimension,
        )
        planned.append(
            (
                name,
                campaign,
                config,
                _validate(name, campaign.plan(config), config, profiles),
            )
        )
    environment = capture_environment().to_dict()
    identity = capture_benchmark_identity(
        PROJECT_ROOT,
        Path(__file__),
        ("campaigns", "config", "environment", "identity", "released", "summary"),
    ).to_dict()
    run_spec_id = canonical_fingerprint(
        {
            "selectors": list(campaigns),
            "config": asdict(base),
            "sizes": None if sizes is None else list(sizes),
            "dimension": dimension,
            "elliptic": asdict(elliptic),
            "source": identity,
        }
    )
    common = {
        "build_id": canonical_fingerprint(identity),
        "source_identity": identity,
        "environment_id": canonical_fingerprint(environment),
        "backend": jax.default_backend(),
        "topology": f"{jax.process_count()}-process-{jax.device_count()}-device",
        "precision": base.precision,
        "run_spec_id": run_spec_id,
        "criteria_at": criteria_at,
    }
    records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for name, campaign, config, validated in planned:
        started_at = max(time.time_ns(), criteria_at + 1)
        print(
            f"[meshfree-qualification] {name}: {len(validated)} planned rows",
            file=sys.stderr,
        )
        rows = [_execute(row, support, block) for row, support, block in validated]
        aggregate = [] if campaign.aggregate is None else campaign.aggregate(rows, config)
        observed_at = max(time.time_ns(), started_at)
        context = {
            **common,
            "campaign_spec_id": canonical_fingerprint(
                {
                    "campaign": name,
                    "description": campaign.description,
                    "source": identity,
                    "config": asdict(config),
                }
            ),
            "started_at": started_at,
            "observed_at": observed_at,
            "issued_at": observed_at + 1,
        }
        evidence = _campaign_evidence(name, rows, aggregate, context)
        table = _summary(name, campaign, rows, aggregate)
        summary.extend(table)
        records.append(
            {
                "campaign_id": name,
                "name": campaign.description,
                "scenario": campaign.scenario,
                "config": asdict(config),
                "status": _row_status(
                    [item for row in rows for item in row["gates"]] + aggregate
                ),
                "rows": rows,
                "aggregate_gates": aggregate,
                "evidence": evidence,
                "release_gates": _release_gates(name),
                "released": False,
            }
        )
    record: dict[str, Any] = {
        "kind": "meshfree-qualification-campaign",
        "config": asdict(base),
        "selected_campaigns": list(campaigns),
        "unexecuted_campaigns": [name for name in CAMPAIGNS if name not in campaigns],
        "campaigns": records,
        "summary": summary,
        "environment": environment,
        "identity": identity,
        "run_spec_id": run_spec_id,
        "approval_id": APPROVAL_ID,
        "reviewer_id": REVIEWER_ID,
        "released": False,
        "promotion": {
            "status": "not-requested",
            "reason": "Unreviewed producer evidence cannot advance a promotion channel",
        },
    }
    return _json_ready(record)


def _print_summary(summary: Sequence[dict[str, Any]], /) -> None:
    header = f"{'campaign':<14} {'scenario':<17} {'capability':<36} {'method':<46} {'geometry':<34} {'d':>1} {'device':<10} {'prec':<7} {'status':<10} {'wall_s':>8} {'cpu_s':>8} {'compiles':>8}"
    print(header, file=sys.stderr)
    for item in summary:
        print(
            f"{item['campaign']:<14} {item['scenario']:<17} {item['capability']:<36} "
            f"{str(item['method'])[:46]:<46} {str(item['geometry_authority'])[:34]:<34} "
            f"{item['dimension']:>1} {item['device']:<10} {item['precision']:<7} "
            f"{item['status']:<10} {item['wall_seconds']:>8.1f} {item['cpu_seconds']:>8.1f} "
            f"{item['compile_count']:>8}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.set_defaults(sizes=None, dimension=None)
    parser.add_argument(
        "--campaigns", nargs="+", choices=tuple(CAMPAIGNS), default=list(CAMPAIGNS)
    )
    parser.add_argument(
        "--elliptic-boundaries",
        nargs="+",
        choices=get_args(EllipticBoundary),
        default=list(get_args(EllipticBoundary)),
    )
    parser.add_argument(
        "--elliptic-forms",
        nargs="+",
        choices=get_args(PointDiffusionForm),
        default=["collocated"],
    )
    parser.add_argument(
        "--elliptic-stencil",
        choices=get_args(MeshfreeApproximation),
        default="phs-rbf-fd",
    )
    parser.add_argument("--elliptic-degree", type=int, default=3)
    parser.add_argument(
        "--elliptic-neighbors",
        type=int,
        help="Explicit Q2 width; default is native support count",
    )
    args = parser.parse_args()
    if args.baseline is not None:
        parser.error(
            "Qualification does not perform before/after timing comparisons; use benchmark drivers."
        )
    base = MeshfreeConfig(
        sizes=(64,),
        seeds=tuple(args.seeds),
        dimension=2,
        repeats=args.repeats,
        neighbors=args.neighbors,
        degree=args.degree,
        chunk_rows=args.chunk_rows,
        working_set_bytes=args.working_set_mib * 1024**2,
        resource_bytes=args.resource_mib * 1024**2,
        max_points=args.max_points,
        steps=args.steps,
        precision=args.precision,
        candidates=args.candidates,
    )
    record = run(
        base,
        tuple(args.campaigns),
        sizes=None if args.sizes is None else tuple(args.sizes),
        dimension=args.dimension,
        elliptic=EllipticSelection(
            boundaries=tuple(args.elliptic_boundaries),
            forms=tuple(args.elliptic_forms),
            stencil=args.elliptic_stencil,
            degree=args.elliptic_degree,
            neighbors=args.elliptic_neighbors,
        ),
    )
    if args.output is not None:
        write_json_atomic(args.output, record)
    else:
        print(json.dumps(record, indent=2, allow_nan=False))
    _print_summary(record["summary"])
    if any(campaign["status"] == "failed" for campaign in record["campaigns"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
