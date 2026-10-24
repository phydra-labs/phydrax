#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Repeatable globally constrained rigid-graph rank-audit and step benchmark.

Measures the joint-Jacobian rank audit alone (``--phase rank``) or the whole
constrained step (``--phase step``) for the default general dense-SVD audit, the
opt-in certified-forest audit, and the certified forest with its admitted
bounded dense fallback.

Each row separates pure host topology construction (NumPy), validated joint-plan
construction (synchronized, since the plans hold device copies), host-to-device
transfer of the numerical inputs, plan preparation (forest admission, block
qualification, fallback planning and the nonlinear template) and state
initialization from lowering, compilation, the first synchronized call and every
individually synchronized warm call, and reports the row's end-to-end wall time
from host topology through the last warm call (excluding the reference check).
Compiler argument/output/temporary/generated-code bytes are XLA estimates;
logical bytes count unique array payloads; vendor-provider workspace is
explicitly not measured. The prepared model, state and step arguments are
arguments of the compiled function, so their arrays are executable leaves rather
than embedded constants. Every row lowers a fresh function, so no trace or
executable is reused between rows.

Every executed evaluation (first, warmup and timed) contributes to the rates.
Certified, measured, inconclusive, failed, empty and fallback outcomes are
reported under the library's own status and route names, so an inconclusive
certificate is never counted as rank deficiency, and ``RANK_DEFICIENT_MEASURED``
means deficiency under the strict relative cutoff
``sigma > rank_tolerance * sigma_max`` only. Preparation refusals (forest
workspace or fallback budgets) are reported with their elapsed preparation time
and no execution timings. Rows whose evaluations were not all accepted are
marked as not timing comparable; this tool computes no speedups. Within an
explicit size bound, an independent host LAPACK float64 SVD of the authoritative
audited operator (the forward-mode Jacobian of its action, not the forest blocks
or the native SVD) adjudicates the last evaluation.

Without ``--suite`` the legacy row flags form their Cartesian product (default:
general dense-SVD whole step on nominal chains of 4, 8 and 16 bodies).
``--suite`` selects an explicit bounded diagnostic row list instead.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never, get_args, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    CompilationTiming,
    compiler_evidence,
    CompilerEvidence,
    DurationDistribution,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax._fingerprint import canonical_json
from phydrax._jax_config import configuration_values
from phydrax.discretization.particle._rigid_constraint_dynamics import (
    _constraint_operator,
    PreparedRigidConstraintDynamics,
    RigidConstraintRejectionReason,
    RigidConstraintSolverPlan,
    RigidConstraintState,
    RigidConstraintStepResult,
)
from phydrax.discretization.particle._rigid_constraint_rank import (
    RigidConstraintRankAudit,
    RigidConstraintRankEvidence,
    RigidConstraintRankRoute,
    RigidConstraintRankStatus,
)
from phydrax.discretization.particle._rigid_joints import _RigidMobileIncrement
from phydrax.linalg._forest_bounds import BlockForestRankStatus
from phydrax.linalg.svd import SVDResourcePolicy
from phydrax.typing import parse


AuditMode: TypeAlias = Literal[
    "general-dense-svd", "certified-forest", "certified-forest-fallback"
]
Topology: TypeAlias = Literal["chain", "binary-tree"]
Phase: TypeAlias = Literal["step", "rank"]
Profile: TypeAlias = Literal["nominal", "global-unit-pivot", "unequal-scales"]
Suite: TypeAlias = Literal["smoke", "scaling", "conditioning"]

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

_PROFILES: dict[str, JsonValue] = {
    "nominal": (
        "Unit spacing along x with a 0.25*floor(log2(i+1)) y offset; one component "
        "rooted at the fixed body 0; ball and hinge joints alternate over edges "
        "with anchors at link midpoints."
    ),
    "global-unit-pivot": (
        "Nominal geometry scaled by --lever-spacing: every child translation "
        "pivot keeps unit scaled norm while parent lever arms grow, so forest "
        "propagation coefficients can exceed one and the global bound can grow "
        "with depth; dense measurement and the reference expose the true "
        "conditioning."
    ),
    "unequal-scales": (
        "Two disconnected components: the first n//2 bodies rooted at the fixed "
        "body 0 with nominal geometry, the rest a free component with geometry "
        "scaled by --scale-ratio, so the strict relative cutoff compares one "
        "component against the other's largest singular value."
    ),
}
_STEP_SIZE = 1.0e-3
_GRAVITY = (0.0, -9.81, 0.0)
_HINGE_AXIS = (0.0, 0.0, 1.0)
_SECOND_COMPONENT_OFFSET = 2.0
_REFUSAL_BYTES = 1
_DEFAULT_BODIES = (4, 8, 16)
_PROVIDER_MEMORY: dict[str, JsonValue] = {
    "status": "not-measured",
    "reason": (
        "Vendor solver/SVD workspace and allocator peaks outside XLA buffer "
        "assignment are not measured; compiler bytes are XLA estimates only."
    ),
}
_ROW_FIELDS = (
    "audit_policy",
    "case",
    "compiled",
    "end_to_end_seconds",
    "identity",
    "logical_bytes",
    "preparation",
    "provider_memory",
    "refusal",
    "scientific",
    "status",
)
_CUSTOM_CALL_TARGETS = (
    re.compile(r"stablehlo\.custom_call\s+@([^\s(]+)"),
    re.compile(r"call_target_name\s*=\s*\"([^\"]+)\""),
)


@dataclass(frozen=True, slots=True)
class _Row:
    """One explicit benchmark row; budgets apply only to forest audits."""

    body_count: int
    topology: Topology
    profile: Profile
    audit: AuditMode
    phase: Phase
    rank_tolerance: float
    forest_workspace_bytes: int | None
    fallback_workspace_bytes: int | None
    fallback_preparation_bytes: int | None


@dataclass(frozen=True, slots=True)
class _Settings:
    warmup: int
    repeats: int
    reference_max_entries: int
    lever_spacing: float
    scale_ratio: float


class _Budgets(NamedTuple):
    forest_workspace_bytes: int | None
    fallback_workspace_bytes: int | None
    fallback_preparation_bytes: int | None


class _HostCase(NamedTuple):
    """Pure host topology: NumPy geometry, edges and structural statistics."""

    body_ids: np.ndarray
    fixed: np.ndarray
    position: np.ndarray
    parents: np.ndarray
    children: np.ndarray
    anchors: np.ndarray
    ball_edges: np.ndarray
    hinge_edges: np.ndarray
    component_count: int
    depth: int
    maximum_branching: int


class _DeviceCase(NamedTuple):
    """Numerical inputs on device, including every step time of the row."""

    body_ids: Array
    masses: Array
    material: Array
    inertia: Array
    fixed: Array
    position: Array
    velocity: Array
    orientation: Array
    angular_velocity: Array
    times: tuple[Array, ...]
    step_size: Array


class _Executed(NamedTuple):
    """Synchronized execution of one compiled phase."""

    compiled: dict[str, JsonValue]
    arguments: tuple[object, ...]
    first: object
    evidence: list[RigidConstraintRankEvidence]
    step: dict[str, JsonValue] | None
    successes: list[bool]
    reference_kinematics: phx.discretization.RigidBodyKinematics


def _fallback_policy(row: _Row) -> SVDResourcePolicy:
    defaults = SVDResourcePolicy()
    return SVDResourcePolicy(
        preparation_bytes=defaults.preparation_bytes
        if row.fallback_preparation_bytes is None
        else row.fallback_preparation_bytes,
        workspace_bytes=defaults.workspace_bytes
        if row.fallback_workspace_bytes is None
        else row.fallback_workspace_bytes,
        operator_matvecs=defaults.operator_matvecs,
    )


def _audit(row: _Row) -> RigidConstraintRankAudit:
    match row.audit:
        case "general-dense-svd":
            return RigidConstraintRankAudit()
        case "certified-forest":
            return RigidConstraintRankAudit(
                "certified-forest", workspace_bytes=row.forest_workspace_bytes
            )
        case "certified-forest-fallback":
            return RigidConstraintRankAudit(
                "certified-forest",
                dense_fallback=_fallback_policy(row),
                workspace_bytes=row.forest_workspace_bytes,
            )
        case unknown:
            assert_never(unknown)


def _audit_record(audit: RigidConstraintRankAudit) -> dict[str, JsonValue]:
    fallback = audit.dense_fallback
    return {
        "kind": audit.kind,
        "audit_id": audit.audit_id,
        "forest_workspace_bytes": audit.workspace_bytes,
        "dense_fallback": None
        if fallback is None
        else {
            "preparation_bytes": fallback.preparation_bytes,
            "workspace_bytes": fallback.workspace_bytes,
            "operator_matvecs": fallback.operator_matvecs,
        },
    }


def _parents(body_count: int, topology: Topology) -> np.ndarray:
    index = np.arange(1, body_count)
    match topology:
        case "chain":
            return index - 1
        case "binary-tree":
            return (index - 1) // 2
        case unknown:
            assert_never(unknown)


def _nominal_position(body_count: int) -> np.ndarray:
    position = np.zeros((body_count, 3))
    position[:, 0] = np.arange(body_count, dtype=np.float64)
    position[:, 1] = 0.25 * np.floor(np.log2(np.arange(body_count) + 1.0))
    return position


def _components(row: _Row) -> tuple[tuple[int, int], ...]:
    """``(first body, body count)`` of each joint component."""
    match row.profile:
        case "nominal" | "global-unit-pivot":
            return ((0, row.body_count),)
        case "unequal-scales":
            half = row.body_count // 2
            return ((0, half), (half, row.body_count - half))
        case unknown:
            assert_never(unknown)


def _component_position(
    row: _Row, component: int, body_count: int, settings: _Settings
) -> np.ndarray:
    nominal = _nominal_position(body_count)
    match row.profile:
        case "nominal":
            return nominal
        case "global-unit-pivot":
            return settings.lever_spacing * nominal
        case "unequal-scales":
            if component == 0:
                return nominal
            scaled = settings.scale_ratio * nominal
            scaled[:, 2] += _SECOND_COMPONENT_OFFSET
            return scaled
        case unknown:
            assert_never(unknown)


def _profile_parameters(row: _Row, settings: _Settings) -> dict[str, JsonValue]:
    match row.profile:
        case "nominal":
            return {}
        case "global-unit-pivot":
            return {"lever_spacing": settings.lever_spacing}
        case "unequal-scales":
            sizes: list[JsonValue] = [size for _, size in _components(row)]
            return {"scale_ratio": settings.scale_ratio, "component_bodies": sizes}
        case unknown:
            assert_never(unknown)


def _host_case(row: _Row, settings: _Settings) -> _HostCase:
    components = _components(row)
    parents = np.concatenate(
        [first + _parents(size, row.topology) for first, size in components]
    )
    children = np.concatenate([first + np.arange(1, size) for first, size in components])
    position = np.concatenate(
        [
            _component_position(row, index, size, settings)
            for index, (_, size) in enumerate(components)
        ]
    )
    edges = np.arange(children.size)
    depth = np.zeros(row.body_count, dtype=np.int64)
    # Every constructed parent precedes its child, so one ordered sweep suffices.
    for child, parent in zip(children.tolist(), parents.tolist(), strict=True):
        depth[child] = depth[parent] + 1
    branching = np.bincount(parents, minlength=row.body_count)
    return _HostCase(
        np.arange(1000, 1000 + row.body_count, dtype=np.int64),
        np.arange(row.body_count) == 0,
        position,
        parents,
        children,
        0.5 * (position[parents] + position[children]),
        edges[::2],
        edges[1::2],
        len(components),
        int(depth.max()),
        int(branching.max()),
    )


def _joint_graph(host: _HostCase) -> phx.discretization.RigidJointGraphPlan:
    """Validated joint plans; they hold device copies of their identity arrays."""
    ids, parents, children = host.body_ids, host.parents, host.children
    ball, hinge = host.ball_edges, host.hinge_edges
    return phx.discretization.RigidJointGraphPlan(
        ball=phx.discretization.BallJointSetPlan(
            2000 + ball, ids[parents[ball]], ids[children[ball]], host.anchors[ball]
        ),
        hinge=phx.discretization.HingeJointSetPlan(
            3000 + hinge,
            ids[parents[hinge]],
            ids[children[hinge]],
            host.anchors[hinge],
            np.broadcast_to(np.asarray(_HINGE_AXIS), (hinge.size, 3)),
        ),
    )


def _transfer(host: _HostCase, calls: int) -> _DeviceCase:
    count = host.body_ids.size
    return _DeviceCase(
        jnp.asarray(host.body_ids),
        jnp.asarray(np.ones((count,))),
        jnp.asarray(np.zeros((count,), dtype=np.int32)),
        jnp.asarray(np.broadcast_to(np.eye(3), (count, 3, 3))),
        jnp.asarray(host.fixed),
        jnp.asarray(host.position),
        jnp.asarray(np.zeros((count, 3))),
        jnp.asarray(np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (count, 1))),
        jnp.asarray(np.zeros((count, 3))),
        tuple(jnp.asarray(index * _STEP_SIZE) for index in range(calls)),
        jnp.asarray(_STEP_SIZE),
    )


def _gravity(
    time_: Array, kinematics: phx.discretization.RigidBodyKinematics, masses: Array
) -> phx.discretization.RigidBodyLoad:
    """Gravity from the per-body masses passed as the step ``args`` leaf."""
    del time_
    return phx.discretization.RigidBodyLoad(
        masses[:, None] * jnp.asarray(_GRAVITY, dtype=masses.dtype),
        jnp.zeros_like(kinematics.angular_velocity),
    )


def _prepare(
    graph: phx.discretization.RigidJointGraphPlan,
    device: _DeviceCase,
    audit: RigidConstraintRankAudit,
    rank_tolerance: float,
) -> tuple[PreparedRigidConstraintDynamics, phx.discretization.RigidBodyKinematics]:
    particles = phx.discretization.ParticleSetPlan(
        device.body_ids, device.masses, ambient_dimension=3
    ).prepare()
    bodies = phx.discretization.RigidBodySetPlan(
        device.material, device.inertia, fixed_mask=device.fixed
    ).prepare(particles)
    reference = bodies.kinematics(
        device.position, device.velocity, device.orientation, device.angular_velocity
    )
    solver = RigidConstraintSolverPlan(rank_audit=audit, rank_tolerance=rank_tolerance)
    dynamics = phx.discretization.RigidConstraintDynamicsPlan(
        graph, solver=solver
    ).prepare(
        bodies,
        reference,
        external_load=_gravity,
        external_load_id="benchmark-gravity-from-mass-args",
    )
    return dynamics, reference


def _timed_prepare(
    graph: phx.discretization.RigidJointGraphPlan,
    device: _DeviceCase,
    audit: RigidConstraintRankAudit,
    rank_tolerance: float,
) -> tuple[
    tuple[PreparedRigidConstraintDynamics, phx.discretization.RigidBodyKinematics]
    | ValueError,
    float,
]:
    """Synchronized preparation, or the refusal it raised and the time to refusal."""
    started = time.perf_counter_ns()
    outcome: (
        tuple[PreparedRigidConstraintDynamics, phx.discretization.RigidBodyKinematics]
        | ValueError
    )
    try:
        outcome = synchronize(_prepare(graph, device, audit, rank_tolerance))
    except ValueError as error:
        outcome = error
    return outcome, (time.perf_counter_ns() - started) / 1_000_000_000.0


def _finite(value: Array | float) -> float | None:
    """JSON has no NaN/infinity: ``null`` marks a non-finite scalar.

    NaN ``measured_condition`` means no dense SVD ran; an infinite
    ``condition_upper_bound`` means the singular-value lower bound is zero.
    """
    number = float(value)
    return number if np.isfinite(number) else None


def _rank_record(evidence: RigidConstraintRankEvidence) -> dict[str, JsonValue]:
    record: dict[str, JsonValue] = {
        "audit": evidence.audit,
        "row_count": evidence.row_count,
        "status": RigidConstraintRankStatus(int(evidence.status)).name,
        "executed_route": RigidConstraintRankRoute(int(evidence.executed_route)).name,
        "full_row_rank": bool(evidence.full_row_rank),
        "rank_lower_bound": int(evidence.rank_lower_bound),
        "rank_upper_bound": int(evidence.rank_upper_bound),
        "measured_condition": _finite(evidence.measured_condition),
        "certificate": None,
    }
    certificate = evidence.certificate
    if certificate is not None:
        record["certificate"] = {
            "status": BlockForestRankStatus(int(certificate.status)).name,
            "certified": bool(certificate.certified),
            "rank_lower_bound": int(certificate.rank_lower_bound),
            "minimum_singular_value_lower_bound": _finite(
                certificate.minimum_singular_value_lower_bound
            ),
            "maximum_singular_value_upper_bound": _finite(
                certificate.maximum_singular_value_upper_bound
            ),
            "condition_upper_bound": _finite(certificate.condition_upper_bound),
            "right_inverse_norm_upper_bound": _finite(
                certificate.right_inverse_norm_upper_bound
            ),
            "residual_norm_upper_bound": _finite(certificate.residual_norm_upper_bound),
            "residual_row_sum_upper_bound": _finite(
                certificate.residual_row_sum_upper_bound
            ),
            "residual_column_sum_upper_bound": _finite(
                certificate.residual_column_sum_upper_bound
            ),
            "threshold_upper_bound": _finite(certificate.threshold_upper_bound),
            "separation_lower_bound": _finite(certificate.separation_lower_bound),
            "unresolved_local_blocks": int(certificate.unresolved_local_blocks),
            "numeric_version": int(certificate.numeric_version),
            "relative_cutoff": certificate.relative_cutoff,
            "absolute_cutoff": certificate.absolute_cutoff,
        }
    return record


def _counts(names: Sequence[str]) -> dict[str, JsonValue]:
    counts: dict[str, JsonValue] = {}
    for name, count in sorted(Counter(names).items()):
        counts[name] = count
    return counts


def _rates(evidence: Sequence[RigidConstraintRankEvidence]) -> dict[str, JsonValue]:
    """Route/status rates over every executed evaluation, by library status name."""
    statuses = [RigidConstraintRankStatus(int(item.status)) for item in evidence]
    routes = [RigidConstraintRankRoute(int(item.executed_route)) for item in evidence]
    count = len(evidence)

    def rate(hits: int) -> float:
        return hits / count

    return {
        "evaluations": count,
        "status_counts": _counts([status.name for status in statuses]),
        "route_counts": _counts([route.name for route in routes]),
        "certificate_status_counts": _counts(
            [
                BlockForestRankStatus(int(item.certificate.status)).name
                for item in evidence
                if item.certificate is not None
            ]
        ),
        "full_row_rank_rate": rate(sum(bool(item.full_row_rank) for item in evidence)),
        "certified_rate": rate(
            statuses.count(RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED)
        ),
        "measured_full_row_rank_rate": rate(
            statuses.count(RigidConstraintRankStatus.FULL_ROW_RANK_MEASURED)
        ),
        "measured_rank_deficient_rate": rate(
            statuses.count(RigidConstraintRankStatus.RANK_DEFICIENT_MEASURED)
        ),
        "inconclusive_rate": rate(statuses.count(RigidConstraintRankStatus.INCONCLUSIVE)),
        "dense_svd_failed_rate": rate(
            statuses.count(RigidConstraintRankStatus.DENSE_SVD_FAILED)
        ),
        "empty_rate": rate(statuses.count(RigidConstraintRankStatus.EMPTY)),
        "fallback_rate": rate(
            routes.count(RigidConstraintRankRoute.FOREST_DENSE_FALLBACK)
        ),
    }


def _compiler_record(evidence: CompilerEvidence) -> dict[str, JsonValue]:
    return {
        "source": evidence.source,
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _compile[T](
    function: Callable[..., T], arguments: tuple[object, ...]
) -> tuple[Callable[..., T], dict[str, JsonValue]]:
    """Lower and compile separately; opaque custom calls void XLA cost totals."""
    jitted = eqx.filter_jit(function)
    (lowered, executable), timing = measure_lower_and_compile(
        # ty: ignore[unresolved-attribute]
        lambda: jitted.lower(*arguments),
        lambda value: (value, value.compile()),
    )
    text = lowered.as_text()
    targets: list[JsonValue] = sorted(
        {target for pattern in _CUSTOM_CALL_TARGETS for target in pattern.findall(text)}
    )
    compiled = executable.compiled
    cost = compiled.cost_analysis()
    memory = compiled.memory_analysis()
    evidence = compiler_evidence(
        cost or None,
        memory,
        source="jax-lowered-compiled-executable",
        unavailable_reason=(
            "Compiler cost and memory analysis were both unavailable on this backend."
            if not cost and memory is None
            else None
        ),
        opaque_custom_calls=bool(targets),
    )
    return executable, {
        **_timing_record(timing),
        "compiler": _compiler_record(evidence),
        "custom_call_targets": targets,
    }


def _timing_record(timing: CompilationTiming) -> dict[str, JsonValue]:
    return {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
    }


def _executed_record(
    compiled: dict[str, JsonValue],
    first_seconds: float,
    warm: DurationDistribution,
    settings: _Settings,
) -> dict[str, JsonValue]:
    return {
        **compiled,
        "first_call_seconds": first_seconds,
        "warmup_calls": settings.warmup,
        "warm": warm.to_seconds_dict(),
    }


def _rank_function() -> Callable[
    [PreparedRigidConstraintDynamics, phx.discretization.RigidBodyKinematics],
    RigidConstraintRankEvidence,
]:
    """A fresh function object, so no trace or executable is shared across rows."""

    def evaluate(
        model: PreparedRigidConstraintDynamics,
        kinematics: phx.discretization.RigidBodyKinematics,
    ) -> RigidConstraintRankEvidence:
        return model.constraint_rank_evidence(kinematics)

    return evaluate


def _step_function() -> Callable[
    [PreparedRigidConstraintDynamics, RigidConstraintState, Array, Array, Array],
    RigidConstraintStepResult,
]:
    """A fresh function object, so no trace or executable is shared across rows."""

    def advance(
        model: PreparedRigidConstraintDynamics,
        state: RigidConstraintState,
        time_: Array,
        step_size: Array,
        masses: Array,
    ) -> RigidConstraintStepResult:
        return model.step(state, time_, step_size, masses)

    return advance


def _run_rank(
    dynamics: PreparedRigidConstraintDynamics,
    state: RigidConstraintState,
    settings: _Settings,
) -> _Executed:
    arguments: tuple[object, ...] = (dynamics, state.kinematics)
    executable, compiled = _compile(_rank_function(), arguments)
    evidence: list[RigidConstraintRankEvidence] = []

    def call() -> RigidConstraintRankEvidence:
        value = executable(dynamics, state.kinematics)
        evidence.append(value)
        return value

    first, first_seconds = measure_synchronized(call)
    _, warm = measure_repeated(call, warmup=settings.warmup, repeats=settings.repeats)
    return _Executed(
        _executed_record(compiled, first_seconds, warm, settings),
        arguments,
        first,
        evidence,
        None,
        [bool(item.full_row_rank) for item in evidence],
        state.kinematics,
    )


def _step_record(
    last: RigidConstraintStepResult, successes: Sequence[bool], reasons: Sequence[int]
) -> dict[str, JsonValue]:
    position = last.evaluation.position_result
    velocity = last.evaluation.velocity_result
    diagnostics = last.evaluation.diagnostics
    flags = [flag for flag in RigidConstraintRejectionReason if int(flag) != 0]
    return {
        "steps": len(successes),
        "successful_steps": sum(successes),
        "rejection_reason_counts": _counts(
            [str(flag.name) for reason in reasons for flag in flags if reason & int(flag)]
        ),
        "last_step": {
            "successful": bool(last.successful),
            "rejection_reasons": [
                str(flag.name)
                for flag in flags
                if int(last.rejection_reasons) & int(flag)
            ],
            "position_iterations": None
            if position is None
            else int(position.diagnostics.iterations),
            "position_residual_evaluations": None
            if position is None
            else int(position.diagnostics.residual_evaluations),
            "position_jvp_evaluations": None
            if position is None
            else int(position.diagnostics.jvp_evaluations),
            "velocity_iterations": None
            if velocity is None
            else int(velocity.diagnostics.iterations),
            "velocity_matvecs": None
            if velocity is None
            else int(velocity.diagnostics.matvec_count),
            "maximum_position_residual": _finite(diagnostics.maximum_position_residual),
            "maximum_stationarity_residual": _finite(
                diagnostics.maximum_stationarity_residual
            ),
            "maximum_velocity_residual": _finite(diagnostics.maximum_velocity_residual),
            "quaternion_defect": _finite(diagnostics.quaternion_defect),
            "velocity_projection_energy_increase": _finite(
                diagnostics.velocity_projection_energy_increase
            ),
        },
    }


def _run_step(
    dynamics: PreparedRigidConstraintDynamics,
    state: RigidConstraintState,
    device: _DeviceCase,
    settings: _Settings,
) -> _Executed:
    """Advance a rollout: each call starts from the previous accepted state."""
    arguments: tuple[object, ...] = (
        dynamics,
        state,
        device.times[0],
        device.step_size,
        device.masses,
    )
    executable, compiled = _compile(_step_function(), arguments)
    current = [state]
    calls = [0]
    evidence: list[RigidConstraintRankEvidence] = []
    successful: list[Array] = []
    reasons: list[Array] = []

    def call() -> RigidConstraintStepResult:
        index = calls[0]
        calls[0] = index + 1
        result = executable(
            dynamics, current[0], device.times[index], device.step_size, device.masses
        )
        current[0] = result.accepted_state
        evidence.append(result.evaluation.diagnostics.rank)
        successful.append(result.successful)
        reasons.append(result.rejection_reasons)
        return result

    first, first_seconds = measure_synchronized(call)
    last, warm = measure_repeated(call, warmup=settings.warmup, repeats=settings.repeats)
    successes = [bool(value) for value in successful]
    return _Executed(
        _executed_record(compiled, first_seconds, warm, settings),
        arguments,
        first,
        evidence,
        _step_record(last, successes, [int(value) for value in reasons]),
        successes,
        last.candidate_state.kinematics,
    )


def _reference(
    dynamics: PreparedRigidConstraintDynamics,
    kinematics: phx.discretization.RigidBodyKinematics,
    evidence: RigidConstraintRankEvidence,
    rank_tolerance: float,
    maximum_entries: int,
) -> dict[str, JsonValue]:
    """Independent strict-cutoff rank of the audited operator at ``kinematics``.

    The matrix is the forward-mode Jacobian of the authoritative scaled
    constraint action in packed row order (not the forest blocks); its singular
    values come from host LAPACK in float64 (not the native SVD). The rounding
    band ``max(rows, columns) * eps(dtype) * sigma_max`` uses the evaluated
    dtype; a strict-cutoff margin inside it is reported as ambiguous.
    """
    joints = dynamics.joints
    rows = joints.constraint_count
    dimension = dynamics.bodies.ambient_dimension
    width = dimension + dynamics.bodies.angular_dimension
    columns = joints.mobile_count * width
    record: dict[str, JsonValue] = {
        "method": "host-lapack-float64-svd-of-forward-jacobian-of-audited-operator",
        "rows": rows,
        "columns": columns,
        "maximum_entries": maximum_entries,
    }
    if rows == 0:
        return {**record, "status": "not-applicable-empty-graph"}
    if rows * columns > maximum_entries:
        return {**record, "status": "not-computed-exceeds-maximum-entries"}
    operator = _constraint_operator(
        joints, dynamics.solver.characteristic_length, kinematics, dynamics.prepared_id
    )

    def packed(flat: Array) -> Array:
        coordinates = flat.reshape(joints.mobile_count, width)
        increment = _RigidMobileIncrement(
            coordinates[:, :dimension], coordinates[:, dimension:]
        )
        return joints.pack_residuals(operator.mv(increment))

    dtype = kinematics.position.dtype
    matrix = np.asarray(
        jax.jacfwd(packed)(jnp.zeros((columns,), dtype=dtype)), dtype=np.float64
    )
    if not np.all(np.isfinite(matrix)):
        return {**record, "status": "nonfinite-operator"}
    values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(values[0])
    smallest = float(values[-1]) if rows <= columns else 0.0
    threshold = rank_tolerance * largest
    strict_rank = int(np.count_nonzero(values > threshold))
    band = max(rows, columns) * float(np.finfo(np.dtype(dtype)).eps) * largest
    if band > 0.0 and abs(smallest - threshold) <= band:
        decision = "within-rounding-band"
    elif strict_rank == rows:
        decision = "full-row-rank"
    else:
        decision = "rank-deficient-under-strict-cutoff"
    status = RigidConstraintRankStatus(int(evidence.status))
    match status:
        case (
            RigidConstraintRankStatus.FULL_ROW_RANK_CERTIFIED
            | RigidConstraintRankStatus.FULL_ROW_RANK_MEASURED
        ):
            claim: str | None = "full-row-rank"
        case RigidConstraintRankStatus.RANK_DEFICIENT_MEASURED:
            claim = "rank-deficient-under-strict-cutoff"
        case (
            RigidConstraintRankStatus.INCONCLUSIVE
            | RigidConstraintRankStatus.DENSE_SVD_FAILED
            | RigidConstraintRankStatus.EMPTY
        ):
            claim = None
        case unknown:
            assert_never(unknown)
    if claim is None:
        agreement = "no-route-decision"
    elif decision == "within-rounding-band":
        agreement = "reference-ambiguous"
    else:
        agreement = "agrees" if claim == decision else "disagrees"
    certificate = evidence.certificate
    return {
        **record,
        "status": "computed",
        "evaluated_dtype": str(dtype),
        "minimum_singular_value": smallest,
        "maximum_singular_value": largest,
        "condition": largest / smallest if smallest > 0.0 else None,
        "strict_threshold": threshold,
        "strict_rank": strict_rank,
        "rounding_band": band,
        "decision": decision,
        "route_status": status.name,
        "agreement": agreement,
        "certificate_lower_bound_at_most_reference_minimum": None
        if certificate is None
        else bool(float(certificate.minimum_singular_value_lower_bound) <= smallest),
        "certificate_upper_bound_at_least_reference_maximum": None
        if certificate is None
        else bool(float(certificate.maximum_singular_value_upper_bound) >= largest),
    }


def _case_record(
    row: _Row,
    host: _HostCase,
    graph: phx.discretization.RigidJointGraphPlan,
    device: _DeviceCase,
    settings: _Settings,
) -> dict[str, JsonValue]:
    return {
        "body_count": row.body_count,
        "topology": row.topology,
        "profile": row.profile,
        "profile_parameters": _profile_parameters(row, settings),
        "rank_audit": row.audit,
        "phase": row.phase,
        "rank_tolerance": row.rank_tolerance,
        "dtype": str(device.position.dtype),
        "mobile_coordinate_count": 6 * int(np.count_nonzero(~host.fixed)),
        "ball_joint_count": int(host.ball_edges.size),
        "hinge_joint_count": int(host.hinge_edges.size),
        "constraint_row_count": graph.constraint_count,
        "component_count": host.component_count,
        "depth": host.depth,
        "maximum_branching": host.maximum_branching,
    }


def _measure(row: _Row, settings: _Settings) -> dict[str, JsonValue]:
    started = time.perf_counter_ns()
    host, topology_seconds = measure_host(lambda: _host_case(row, settings))
    graph, joint_plan_seconds = measure_synchronized(lambda: _joint_graph(host))
    calls = 1 + settings.warmup + settings.repeats if row.phase == "step" else 0
    device, transfer_seconds = measure_synchronized(lambda: _transfer(host, calls))
    audit = _audit(row)
    prepared, prepare_seconds = _timed_prepare(graph, device, audit, row.rank_tolerance)
    preparation: dict[str, JsonValue] = {
        "host_topology_seconds": topology_seconds,
        "joint_plan_seconds": joint_plan_seconds,
        "transfer_seconds": transfer_seconds,
        "plan_and_prepare_seconds": prepare_seconds,
        "initialize_state_seconds": None,
    }
    record: dict[str, JsonValue] = {
        "case": _case_record(row, host, graph, device, settings),
        "audit_policy": _audit_record(audit),
        "preparation": preparation,
        "provider_memory": _PROVIDER_MEMORY,
    }
    if isinstance(prepared, ValueError):
        return {
            **record,
            "status": "refused-at-preparation",
            "refusal": {
                "stage": "prepare",
                "error_type": type(prepared).__name__,
                "message": str(prepared),
            },
            "compiled": None,
            "logical_bytes": None,
            "scientific": None,
            "identity": None,
            "end_to_end_seconds": (time.perf_counter_ns() - started) / 1_000_000_000.0,
        }
    dynamics, reference = prepared
    state, preparation["initialize_state_seconds"] = measure_synchronized(
        lambda: dynamics.initialize_state(
            reference.position,
            reference.velocity,
            reference.orientation,
            reference.angular_velocity,
        )
    )
    match row.phase:
        case "rank":
            executed = _run_rank(dynamics, state, settings)
        case "step":
            executed = _run_step(dynamics, state, device, settings)
        case unknown:
            assert_never(unknown)
    end_to_end = (time.perf_counter_ns() - started) / 1_000_000_000.0
    accepted = all(executed.successes)
    last = executed.evidence[-1]
    forest = dynamics.rank.forest
    unsuccessful: list[JsonValue] = [
        name
        for name in sorted(
            {
                RigidConstraintRankStatus(int(item.status)).name
                for item in executed.evidence
                if not bool(item.full_row_rank)
            }
        )
    ]
    return {
        **record,
        "status": "completed",
        "refusal": None,
        "compiled": executed.compiled,
        "logical_bytes": {
            "prepared_dynamics": logical_array_bytes(dynamics),
            "state": logical_array_bytes(state),
            "compiled_arguments": logical_array_bytes(executed.arguments),
            "first_result": logical_array_bytes(executed.first),
            "forest_template_retained": None
            if forest is None
            else forest.template.retained_bytes,
            "forest_workspace_admitted": None
            if forest is None
            else forest.template.workspace_bytes,
        },
        "scientific": {
            "outcome": "accepted" if accepted else "unsuccessful",
            "timing_comparable": accepted,
            "unsuccessful_rank_statuses": unsuccessful,
            "rank_rates": _rates(executed.evidence),
            "last_rank": _rank_record(last),
            "step": executed.step,
            "reference": _reference(
                dynamics,
                executed.reference_kinematics,
                last,
                row.rank_tolerance,
                settings.reference_max_entries,
            ),
        },
        "identity": {
            "prepared_id": dynamics.prepared_id,
            "solver_id": dynamics.solver.plan_id,
            "audit_id": dynamics.solver.rank_audit.audit_id,
            "rank_problem_id": dynamics.rank.problem_id,
        },
        "end_to_end_seconds": end_to_end,
    }


def _row(
    body_count: int,
    topology: Topology,
    profile: Profile,
    audit: AuditMode,
    phase: Phase,
    rank_tolerance: float,
    budgets: _Budgets,
) -> _Row:
    return _Row(body_count, topology, profile, audit, phase, rank_tolerance, *budgets)


def _suite_rows(suite: Suite, budgets: _Budgets) -> list[_Row]:
    """Explicit bounded diagnostic rows, never an unbounded Cartesian product."""
    audits: tuple[AuditMode, ...] = get_args(AuditMode)
    topologies: tuple[Topology, ...] = get_args(Topology)
    phases: tuple[Phase, ...] = ("rank", "step")
    stepped: tuple[AuditMode, ...] = ("general-dense-svd", "certified-forest")
    match suite:
        case "smoke":
            refused = _Budgets(_REFUSAL_BYTES, None, None)
            unaffordable = _Budgets(None, _REFUSAL_BYTES, _REFUSAL_BYTES)
            return [
                *(
                    _row(4, "chain", "nominal", audit, phase, 1.0e-10, budgets)
                    for phase in phases
                    for audit in audits
                ),
                # A strict cutoff the nominal operator cannot meet: measured
                # deficiency, inconclusive certificate and fallback stay distinct.
                *(
                    _row(4, "chain", "nominal", audit, phase, 0.9, budgets)
                    for phase in phases
                    for audit in audits
                ),
                _row(
                    8,
                    "chain",
                    "global-unit-pivot",
                    "certified-forest-fallback",
                    "rank",
                    1.0e-10,
                    budgets,
                ),
                _row(
                    8,
                    "chain",
                    "unequal-scales",
                    "certified-forest-fallback",
                    "rank",
                    1.0e-10,
                    budgets,
                ),
                _row(4, "chain", "nominal", "certified-forest", "rank", 1.0e-10, refused),
                _row(
                    4,
                    "chain",
                    "nominal",
                    "certified-forest-fallback",
                    "rank",
                    1.0e-10,
                    unaffordable,
                ),
            ]
        case "scaling":
            return [
                *(
                    _row(count, topology, "nominal", audit, "rank", 1.0e-10, budgets)
                    for topology in topologies
                    for audit in audits
                    for count in (8, 32, 128, 512)
                ),
                *(
                    _row(count, topology, "nominal", audit, "step", 1.0e-10, budgets)
                    for topology in topologies
                    for audit in stepped
                    for count in (8, 32, 128)
                ),
            ]
        case "conditioning":
            profiles: tuple[Profile, ...] = get_args(Profile)
            return [
                _row(16, topology, profile, audit, "rank", tolerance, budgets)
                for profile in profiles
                for topology in topologies
                for audit in audits
                for tolerance in (1.0e-10, 1.0e-3)
            ]
        case unknown:
            assert_never(unknown)


def _cache_policy() -> dict[str, JsonValue]:
    settings = configuration_values()
    directory = settings["jax_compilation_cache_dir"]
    persistent = bool(settings["jax_enable_compilation_cache"]) and bool(directory)
    return {
        "in_process": "fresh-function-per-row; no trace or executable reuse between rows",
        "persistent_compilation_cache": "configured" if persistent else "disabled",
        "persistent_compilation_cache_dir": str(directory) if persistent else None,
        "compilation_temperature": "persistent-cache-hits-possible"
        if persistent
        else "cold-in-process",
        "phases": "separate host topology, transfer, prepare, initialize, lowering, "
        "compilation, first synchronized call and synchronized warm calls",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=get_args(Suite),
        default=None,
        help="explicit bounded diagnostic rows; excludes the row flags below",
    )
    parser.add_argument("--bodies", type=int, nargs="+", default=None)
    parser.add_argument("--topology", choices=get_args(Topology), nargs="+")
    parser.add_argument("--profile", choices=get_args(Profile), nargs="+")
    parser.add_argument("--rank-audit", choices=get_args(AuditMode), nargs="+")
    parser.add_argument("--phase", choices=get_args(Phase), nargs="+")
    parser.add_argument("--rank-tolerance", type=float, nargs="+")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reference-max-entries", type=int, default=8_000_000)
    parser.add_argument("--lever-spacing", type=float, default=8.0)
    parser.add_argument("--scale-ratio", type=float, default=1.0e3)
    parser.add_argument("--forest-workspace-bytes", type=int, default=None)
    parser.add_argument("--fallback-workspace-bytes", type=int, default=None)
    parser.add_argument("--fallback-preparation-bytes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    row_flags = (
        arguments.bodies,
        arguments.topology,
        arguments.profile,
        arguments.rank_audit,
        arguments.phase,
        arguments.rank_tolerance,
    )
    if arguments.repeats < 1 or arguments.warmup < 0:
        parser.error("--repeats must be positive and --warmup nonnegative.")
    if arguments.reference_max_entries < 0:
        parser.error("--reference-max-entries must be nonnegative.")
    if not (
        np.isfinite(arguments.lever_spacing)
        and arguments.lever_spacing > 0.0
        and np.isfinite(arguments.scale_ratio)
        and arguments.scale_ratio > 0.0
    ):
        parser.error("--lever-spacing and --scale-ratio must be finite and positive.")
    budget_values = (
        arguments.forest_workspace_bytes,
        arguments.fallback_workspace_bytes,
        arguments.fallback_preparation_bytes,
    )
    if any(value is not None and value < 0 for value in budget_values):
        parser.error("Byte budgets must be nonnegative.")
    budgets = _Budgets(*budget_values)
    suite: Suite | None = (
        None if arguments.suite is None else parse(arguments.suite, Suite, "suite")
    )
    if suite is not None:
        if any(value is not None for value in row_flags):
            parser.error("--suite rows are explicit; omit the individual row flags.")
        rows = _suite_rows(suite, budgets)
    else:
        rows = [
            _row(
                count,
                parse(topology, Topology, "topology"),
                parse(profile, Profile, "profile"),
                parse(mode, AuditMode, "rank_audit"),
                parse(phase, Phase, "phase"),
                tolerance,
                budgets,
            )
            for topology in arguments.topology or ["chain"]
            for profile in arguments.profile or ["nominal"]
            for mode in arguments.rank_audit or ["general-dense-svd"]
            for phase in arguments.phase or ["step"]
            for tolerance in arguments.rank_tolerance or [1.0e-10]
            for count in arguments.bodies or _DEFAULT_BODIES
        ]
    for row in rows:
        minimum = 4 if row.profile == "unequal-scales" else 2
        if row.body_count < minimum:
            parser.error(f"Profile {row.profile!r} needs at least {minimum} bodies.")
    settings = _Settings(
        arguments.warmup,
        arguments.repeats,
        arguments.reference_max_entries,
        arguments.lever_spacing,
        arguments.scale_ratio,
    )
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parent.parent, driver, _ROW_FIELDS)
    environment = capture_environment()
    results: list[JsonValue] = [_measure(row, settings) for row in rows]
    payload: dict[str, JsonValue] = {
        "benchmark": "rigid-constraint-dynamics",
        "command": [sys.executable, *sys.argv],
        "suite": suite,
        "device": str(jax.devices()[0]),
        "dtype": str(jnp.ones(()).dtype),
        "settings": {
            "warmup": settings.warmup,
            "repeats": settings.repeats,
            "reference_max_entries": settings.reference_max_entries,
            "lever_spacing": settings.lever_spacing,
            "scale_ratio": settings.scale_ratio,
            "step_size": _STEP_SIZE,
        },
        "profiles": _PROFILES,
        "cache_policy": _cache_policy(),
        "environment": environment.to_dict(),
        "identity": identity.to_dict(),
        "results": results,
    }
    canonical_json(payload)
    if arguments.output is None:
        print(
            json.dumps(
                payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
            )
        )
    else:
        print(write_json_atomic(arguments.output, payload))


if __name__ == "__main__":
    main()
