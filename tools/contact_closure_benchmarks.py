#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated contact closure, operator cone, derivative and rough-contact campaign.

Run, for example:
    PYTHONPATH=. python tools/contact_closure_benchmarks.py
    PYTHONPATH=. python tools/contact_closure_benchmarks.py --profile smoke
    PYTHONPATH=. python tools/contact_closure_benchmarks.py --profile scaling \
        --output contact-closure-scaling.json

Sections:

- ``closure``: barrier/Coulomb closure over canonical host search and contact
  kinematics of a moving polyline above a static segment.
- ``kinematic-cone``: the cone program lowered from those kinematics with a
  native local-block response, its explicit dense authority, SAP, and prepared
  fixed-branch JVP/VJP.
- ``operator-cone``: diagnostic rows of a matrix-free point-body response
  ``W = G M^-1 G^T`` (block-diagonal, depth-limited stacks, clustered and
  redundant coupling) varying capacity, active count, mass-ratio conditioning,
  compliance and dtype, with a position-level ``gap / h`` normal bias.
- ``rough``: periodic rough contact against the analytic uniform-gap pressure.

Every executable is lowered, compiled, first-executed and warm-sampled as
separate phases over numerical leaves passed as arguments. Solve, branch and
derivative statuses accompany every timing; refused and uncertified work is
reported and never counted as accepted work. Dense authorities are assembled
independently on the host in float64 from the same physical factors under an
explicit resource policy and are refused, not attempted, beyond it.
Derivatives are admitted only with a native rank certificate of the fixed-branch
Jacobian under an explicit bounded ``SVDSolvePolicy`` (``--regularity-*``);
beyond its limits the certificate is unavailable and derivatives are refused,
and redundant support is expected to report a nullspace refusal.
Compiler buffer estimates and logical array bytes are not allocator peaks;
provider/allocator memory is not measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, assert_never, get_args, Literal, TypeAlias

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
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import config as pxconfig, linalg as la
from phydrax.applications.contact import (
    certify_contact_cone_impulse,
    contact_cone_solution_jvp,
    contact_cone_solution_vjp,
    contact_cone_space,
    ContactBlockLayout,
    ContactCertificatePolicy,
    ContactClosureEvaluation,
    ContactClosurePlan,
    ContactConeBranch,
    ContactConeJVP,
    ContactConeProgram,
    ContactConeResult,
    ContactConeSolverPlan,
    ContactConeTangent,
    ContactConeVJP,
    ContactDerivativeEvidence,
    ContactMaterialPairTable,
    ContactRouteState,
    dense_contact_response,
    local_block_contact_response,
    PeriodicRoughContactPlan,
    PeriodicRoughContactResult,
    prepare_contact_cone_sensitivity,
    PreparedContactConeSensitivity,
    SAPContactSolverPlan,
    solve_contact_cone,
    solve_contact_sap,
)
from phydrax.applications.contact._cone import contact_cone_result_is_certified
from phydrax.discretization import ContactKinematicsEpoch
from phydrax.linalg.svd import DenseSVD, SVDResourcePolicy, SVDSolvePolicy


Coupling: TypeAlias = Literal["block-diagonal", "stack", "clustered", "redundant"]
DTypeName: TypeAlias = Literal["float32", "float64"]
Profile: TypeAlias = Literal["legacy", "smoke", "scaling"]
Section: TypeAlias = Literal["closure", "kinematic-cone", "operator-cone", "rough"]

_ROOT = Path(__file__).resolve().parents[1]
_WIDTH = 3
_STACK_DEPTH = 8
_CLUSTER_SIZE = 8
_REDUNDANCY = 4
_STEP_SIZE = 1.0e-2
_REFERENCE_VELOCITY = 1.0
_ROUTE_KEY_OFFSET = 1_000_000_007
_MAXIMUM_CAPACITY = 65_536
_MAXIMUM_ROWS = 64
_MAXIMUM_SEGMENTS = 16_384
_MAXIMUM_ROUGH_GRID = 1_024
_MAXIMUM_REPEATS = 1_000
_MAXIMUM_WARMUP = 100
_MAXIMUM_SOLVER_ITERATIONS = 100_000
_MAXIMUM_LINEAR_ITERATIONS = 10_000
_MAXIMUM_MASS_RATIO = 1.0e8
_ROUGH_GAP = -0.01
_CERTIFICATE_FLAGS = (
    "normal_impulse_valid",
    "law_normal_valid",
    "complementarity_valid",
    "cone_valid",
    "friction_law_valid",
    "padding_valid",
    "friction_work_valid",
    "numeric_inputs_valid",
    "material_law_complete",
    "finite",
    "successful",
)
_UNAVAILABLE_COMPILER = "The active backend did not report every compiler estimate."
_PROVIDER_MEMORY = {
    "measured": False,
    "reason": (
        "No allocator or provider memory instrumentation is used; compiler "
        "argument/output/temporary/code bytes and logical array bytes are "
        "reported separately and neither is a peak-allocation bound."
    ),
}


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Dtype-dependent certificate, solver and derivative-check controls."""

    certificate_relative_tolerance: float
    sap_tolerance: float
    linear_tolerance: float
    finite_difference_relative_step: float
    finite_difference_agreement: float
    adjoint_agreement: float
    action_agreement: float


_PRECISION: dict[DTypeName, PrecisionPolicy] = {
    "float64": PrecisionPolicy(1.0e-7, 1.0e-9, 1.0e-12, 1.0e-4, 1.0e-3, 1.0e-8, 1.0e-12),
    "float32": PrecisionPolicy(1.0e-3, 1.0e-5, 1.0e-4, 3.0e-2, 5.0e-2, 1.0e-3, 1.0e-4),
}


@dataclass(frozen=True, slots=True)
class CampaignSettings:
    """Explicit bounded execution controls shared by every row."""

    warmup: int
    repeats: int
    solver_iterations: int
    linear_iterations: int
    margin_factor: float
    seed: int
    dense_policy: la.MaterializationPolicy
    regularity: SVDSolvePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "repeats": self.repeats,
            "solver_iterations": self.solver_iterations,
            "linear_iterations": self.linear_iterations,
            "margin_factor": self.margin_factor,
            "seed": self.seed,
            "dense_authority_policy": {
                "max_entries": self.dense_policy.max_entries,
                "max_bytes": self.dense_policy.max_bytes,
                "max_workspace_bytes": self.dense_policy.max_workspace_bytes,
                "assembly_dtype": "float64",
                "scope": (
                    "Host float64 response, host Jacobian/spectrum temporaries; "
                    "rows beyond either limit are resource-refused before assembly."
                ),
            },
            "jacobian_regularity_policy": _regularity_policy_record(self.regularity),
            "precision": {name: asdict(policy) for name, policy in _PRECISION.items()},
        }


@dataclass(frozen=True, slots=True)
class OperatorRowSpec:
    """One diagnostic operator-cone row; never an implicit Cartesian product."""

    coupling: Coupling
    capacity: int
    active: int
    mass_ratio: float
    compliance: float
    dtype: DTypeName

    def __post_init__(self) -> None:
        if self.coupling not in get_args(Coupling):
            raise ValueError(f"Unknown coupling {self.coupling!r}.")
        if self.dtype not in get_args(DTypeName):
            raise ValueError(f"Unknown dtype {self.dtype!r}.")
        if not 1 <= self.capacity <= _MAXIMUM_CAPACITY:
            raise ValueError(f"capacity must lie in [1, {_MAXIMUM_CAPACITY}].")
        if not 0 <= self.active <= self.capacity:
            raise ValueError("active count must lie in [0, capacity].")
        if not (
            math.isfinite(self.mass_ratio)
            and 1.0 <= self.mass_ratio <= _MAXIMUM_MASS_RATIO
        ):
            raise ValueError(f"mass_ratio must lie in [1, {_MAXIMUM_MASS_RATIO:g}].")
        if not (math.isfinite(self.compliance) and self.compliance >= 0.0):
            raise ValueError("compliance must be finite and nonnegative.")

    @property
    def label(self) -> str:
        return (
            f"{self.coupling}-c{self.capacity}-a{self.active}-m{self.mass_ratio:g}"
            f"-k{self.compliance:g}-{self.dtype}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "coupling": self.coupling,
            "capacity": self.capacity,
            "active": self.active,
            "mass_ratio": self.mass_ratio,
            "compliance": self.compliance,
            "dtype": self.dtype,
        }

    @classmethod
    def parse(cls, text: str) -> OperatorRowSpec:
        """Parse ``COUPLING:CAPACITY:ACTIVE:MASS_RATIO:COMPLIANCE:DTYPE``."""

        parts = text.split(":")
        if len(parts) != 6:
            raise argparse.ArgumentTypeError(
                "Rows use COUPLING:CAPACITY:ACTIVE:MASS_RATIO:COMPLIANCE:DTYPE."
            )
        coupling, capacity, active, mass_ratio, compliance, dtype = parts
        couplings: dict[str, Coupling] = {name: name for name in get_args(Coupling)}
        dtypes: dict[str, DTypeName] = {name: name for name in get_args(DTypeName)}
        if coupling not in couplings or dtype not in dtypes:
            raise argparse.ArgumentTypeError(
                f"coupling must be one of {sorted(couplings)}; dtype one of {sorted(dtypes)}."
            )
        try:
            return cls(
                couplings[coupling],
                int(capacity),
                int(active),
                float(mass_ratio),
                float(compliance),
                dtypes[dtype],
            )
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error


@dataclass(frozen=True, slots=True)
class ConeControls:
    """Solver plans and derivative controls of one cone row."""

    solver: ContactConeSolverPlan
    sap: SAPContactSolverPlan
    precision: PrecisionPolicy
    margin_factor: float
    linear_iterations: int
    regularity: SVDSolvePolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "projected_gradient": {
                "maximum_iterations": self.solver.maximum_iterations,
                "absolute_tolerance": self.solver.absolute_tolerance,
                "relative_tolerance": self.solver.relative_tolerance,
                "maximum_step_reductions": self.solver.maximum_step_reductions,
            },
            "sap": {
                "maximum_iterations": self.sap.maximum_iterations,
                "tolerance": self.sap.tolerance,
                "acceleration": self.sap.acceleration,
            },
            "derivative": {
                "margin_factor": self.margin_factor,
                "linear_tolerance": self.precision.linear_tolerance,
                "linear_iterations": self.linear_iterations,
                "linear_method": "restarted GMRES on the prepared fixed-branch Jacobian",
                "jacobian_regularity": _regularity_policy_record(self.regularity),
            },
        }


@dataclass(frozen=True, slots=True)
class DenseAuthoritySource:
    """Independent host float64 assembly of a response ``W`` over block coordinates."""

    description: str
    rows: int
    workspace_bytes: int
    assemble: Callable[[], np.ndarray]


@dataclass(frozen=True, slots=True)
class BodyCouplingFactors:
    """Host physical factors of the point-body response ``W = G M^-1 G^T``.

    Frames are orthonormal local contact frames (row 0 is the normal); body
    index ``body_count`` is the static ground with zero inverse mass.
    """

    frames: np.ndarray
    left: np.ndarray
    right: np.ndarray
    inverse_mass: np.ndarray
    free_velocity: np.ndarray
    normal_bias: np.ndarray
    static_friction: np.ndarray
    dynamic_friction: np.ndarray
    valid: np.ndarray
    route_keys: np.ndarray
    body_count: int
    effective_mass_reference: float


@dataclass(frozen=True, slots=True)
class ClosureScene:
    """Canonical kinematics, materials, closure plan and route state of one scene."""

    segment_count: int
    capacity: int
    kinematics: ContactKinematicsEpoch
    materials: ContactMaterialPairTable
    closure: ContactClosurePlan
    state: ContactRouteState
    preparation: dict[str, float]


@dataclass(frozen=True, slots=True)
class CompiledPhase:
    """A compiled executable over dynamic leaves and its fixed output statics."""

    compiled: jax.stages.Compiled
    output_static: Any

    def numeric(self, *arguments: Any) -> Any:
        dynamic, _ = eqx.partition(arguments, eqx.is_array)
        return self.compiled(dynamic)

    def __call__(self, *arguments: Any) -> Any:
        return eqx.combine(self.numeric(*arguments), self.output_static)


def _number(value: Any) -> float | None:
    """JSON has no NaN/infinity: ``null`` marks a non-finite scalar."""

    number = float(value)
    return number if math.isfinite(number) else None


def _flag(value: Any) -> bool:
    return bool(np.all(np.asarray(value)))


def _host64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _maximum_abs(value: Any) -> float:
    return float(np.max(np.abs(_host64(value)), initial=0.0))


def _difference(left: Any, right: Any) -> float:
    return _maximum_abs(_host64(left) - _host64(right))


def _relative(error: float, scale: float) -> float:
    return error / max(scale, np.finfo(np.float64).tiny)


def _optimized_hlo(compiled: jax.stages.Compiled) -> dict[str, Any]:
    try:
        text = compiled.as_text()
    except NotImplementedError as error:
        return {"available": False, "reason": f"HLO text unavailable: {error}"}
    if not text:
        return {"available": False, "reason": "The executable exposed no HLO text."}
    return {
        "available": True,
        "while_loops": text.count(" while("),
        "conditionals": text.count(" conditional("),
        "fusions": text.count(" fusion("),
        "custom_calls": text.count(" custom-call("),
        "custom_call_targets": sorted(
            set(re.findall(r'custom_call_target="([^"]+)"', text))
        ),
        "scope": (
            "Static instruction occurrences in the optimized HLO module text, "
            "including nested loop bodies; not measured kernel launches."
        ),
    }


def _measure_phase(
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    settings: CampaignSettings,
) -> tuple[Any, CompiledPhase, dict[str, Any]]:
    """Lower, compile, first-execute and warm-sample one executable.

    Every array leaf of ``arguments`` (including closure-converted operator
    constants) is a dynamic executable argument; only static metadata is
    closed over. Output statics are recorded during the single trace.
    """

    dynamic, static = eqx.partition(arguments, eqx.is_array)
    output_statics: list[Any] = []

    def numeric(values: Any) -> Any:
        output = function(*eqx.combine(values, static))
        output_dynamic, output_static = eqx.partition(output, eqx.is_array)
        output_statics.append(output_static)
        return output_dynamic

    jitted = jax.jit(numeric)
    compiled, compilation = measure_lower_and_compile(
        lambda: jitted.lower(dynamic), lambda lowered: lowered.compile()
    )
    _, first_seconds = measure_synchronized(lambda: compiled(dynamic))
    output, warm = measure_repeated(
        lambda: compiled(dynamic), warmup=settings.warmup, repeats=settings.repeats
    )
    phase = CompiledPhase(compiled, output_statics[-1])
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason=_UNAVAILABLE_COMPILER,
    )
    record = {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_execution_synchronized_seconds": first_seconds,
        "warm_synchronized": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "compiler_estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "optimized_hlo": _optimized_hlo(compiled),
        "logical_bytes": {
            "dynamic_arguments": logical_array_bytes(dynamic),
            "outputs": logical_array_bytes(output),
        },
    }
    return eqx.combine(output, phase.output_static), phase, record


def _cold_path(preparation: dict[str, float], phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "seconds": sum(preparation.values())
        + phase["lowering_seconds"]
        + phase["compilation_seconds"]
        + phase["first_execution_synchronized_seconds"],
        "scope": (
            "Sum of separately measured host preparation, transfer, binding, "
            "lowering, compilation and first synchronized execution."
        ),
    }


def _branch_counts(branch: Any, valid: np.ndarray) -> dict[str, int]:
    labels = np.asarray(branch)[valid]
    return {
        member.name.lower(): int(np.count_nonzero(labels == int(member)))
        for member in ContactConeBranch
    }


def _solve_outcome(result: ContactConeResult, valid: np.ndarray) -> dict[str, Any]:
    evidence = result.evidence
    certificate = evidence.certificate
    if _flag(evidence.successful):
        status = "certified"
    elif _flag(evidence.step_budget_exhausted):
        status = "step-budget-exhausted"
    elif not _flag(evidence.converged):
        status = "iteration-budget-exhausted"
    elif not _flag(evidence.finite):
        status = "nonfinite"
    else:
        status = "certificate-failed"
    return {
        "status": status,
        "converged": _flag(evidence.converged),
        "iterations": int(evidence.iterations),
        "step_reductions": int(evidence.step_reductions),
        "step_budget_exhausted": _flag(evidence.step_budget_exhausted),
        "fixed_point_residual": _number(evidence.fixed_point_residual),
        "iteration_tolerance": _number(evidence.iteration_tolerance),
        "certificate": {
            name: _flag(getattr(certificate, name)) for name in _CERTIFICATE_FLAGS
        },
        "maximum_residuals": {
            "scope": "Physical certificate of the final candidate impulse, in family units.",
            "normal_impulse_defect": _number(
                _maximum_abs(certificate.normal_impulse_defect)
            ),
            "law_normal_defect": _number(_maximum_abs(certificate.law_normal_defect)),
            "complementarity_work": _number(
                _maximum_abs(certificate.complementarity_work)
            ),
            "cone_defect": _number(_maximum_abs(certificate.cone_defect)),
            "physical_tangential_work": _number(certificate.physical_tangential_work),
            "impulse_tolerance": _maximum_abs(certificate.impulse_tolerance),
            "law_velocity_tolerance": _maximum_abs(certificate.law_velocity_tolerance),
        },
        "branch_counts": _branch_counts(result.branch, valid),
    }


def _regularity_policy_record(policy: SVDSolvePolicy) -> dict[str, Any]:
    method = policy.method
    return {
        "method": method.name,
        "algorithm": method.algorithm if isinstance(method, DenseSVD) else None,
        "rank_relative_cutoff": policy.rank.relative_cutoff,
        "rank_absolute_cutoff": policy.rank.absolute_cutoff,
        "materialization": {
            "max_entries": policy.materialization.max_entries,
            "max_bytes": policy.materialization.max_bytes,
            "max_workspace_bytes": policy.materialization.max_workspace_bytes,
        },
        "resources": {
            "preparation_bytes": policy.resources.preparation_bytes,
            "workspace_bytes": policy.resources.workspace_bytes,
            "operator_matvecs": policy.resources.operator_matvecs,
        },
        "scope": (
            "Native rank certificate of the matrix-free fixed-branch Jacobian via "
            "certify_rectangular_rank; it is formed only within these explicit limits. "
            "Beyond them the certificate is unavailable and every derivative is refused."
        ),
    }


def _derivative_outcome(
    evidence: ContactDerivativeEvidence, valid: np.ndarray
) -> dict[str, Any]:
    def optional(value: Array | None) -> bool | None:
        return None if value is None else _flag(value)

    primal = _flag(evidence.primal_successful)
    qualified = _flag(evidence.branch_qualified)
    finite = _flag(evidence.derivative_finite)
    linear = optional(evidence.linear_successful)
    regular = optional(evidence.jacobian_regular)
    friction_boundary = optional(evidence.friction_boundary_refused)
    if _flag(evidence.successful):
        status = "accepted"
    elif not primal:
        status = "refused-primal-uncertified"
    elif not qualified:
        status = "refused-branch-not-qualified"
    elif regular is False:
        status = "refused-jacobian-regularity-unproven"
    elif friction_boundary:
        status = "refused-frictionless-coulomb-boundary"
    elif linear is False:
        status = "refused-linear-solve-unsuccessful"
    elif not finite:
        status = "refused-nonfinite"
    else:
        status = "refused"
    return {
        "status": status,
        "timed_work": (
            "accepted fixed-branch derivative"
            if status == "accepted"
            else "refused derivative; the returned tangent/cotangent is exactly zero"
        ),
        "primal_successful": primal,
        "branch_qualified": qualified,
        "jacobian_regular": regular,
        "regularity_margin": (
            None
            if evidence.regularity_margin is None
            else _number(evidence.regularity_margin)
        ),
        "friction_boundary_refused": friction_boundary,
        "linear_successful": linear,
        "derivative_finite": finite,
        "cone_margin_tolerance_units": _number(evidence.branch_margin),
        "branch_counts": (
            None
            if evidence.branch_classification is None
            else _branch_counts(evidence.branch_classification, valid)
        ),
    }


def _masked_direction(
    rng: np.random.Generator, valid: np.ndarray, width: int, dtype: np.dtype[Any]
) -> Array:
    value = rng.normal(size=(valid.size, width)) * valid[:, None]
    scale = float(np.max(np.abs(value), initial=0.0))
    return jnp.asarray(value / scale if scale > 0.0 else value, dtype=dtype)


def _with_response(
    program: ContactConeProgram,
    response: la.AbstractLinearOperator,
    program_id: str,
) -> ContactConeProgram:
    """Same law, routes and certificate bound to another response operator."""

    return ContactConeProgram(
        response,
        program.free_velocity,
        program.compliance,
        program.dynamic_friction,
        program.route_keys,
        program.valid,
        program.layout,
        program.certificate,
        program_id,
        static_friction=program.static_friction,
        normal_bias=program.normal_bias,
        restitution=program.restitution,
        mechanical_available=program.mechanical_available,
        response_binding=program.response_binding,
    )


def _spectrum(
    matrix: np.ndarray, valid: np.ndarray, compliance: np.ndarray
) -> dict[str, Any]:
    """Host float64 spectrum of the response restricted to valid block rows."""

    rows = np.repeat(valid, compliance.shape[1])
    restricted = matrix[np.ix_(rows, rows)]
    diagonal = compliance.reshape(-1)[rows]
    size = int(restricted.shape[0])
    if size == 0:
        return {
            "rows": 0,
            "response_rank": 0,
            "minimum_eigenvalue": None,
            "maximum_eigenvalue": None,
            "regularized_minimum_eigenvalue": None,
            "regularized_condition": None,
            "regularized_spectrum_separated_from_zero": True,
            "rank_threshold": None,
            "scope": "Valid rows of W and W + C; host float64 eigvalsh.",
            "uniqueness_claim": "none: this record reports the response spectrum only",
        }
    eigenvalues = np.linalg.eigvalsh(restricted)
    maximum = float(eigenvalues[-1])
    threshold = size * np.finfo(np.float64).eps * max(abs(maximum), 1.0)
    if np.all(diagonal == diagonal[0]):
        regularized = eigenvalues + diagonal[0]
    else:
        regularized = np.linalg.eigvalsh(restricted + np.diag(diagonal))
    minimum_regularized = float(regularized[0])
    separated = minimum_regularized > threshold
    return {
        "rows": size,
        "response_rank": int(np.count_nonzero(eigenvalues > threshold)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": maximum,
        "regularized_minimum_eigenvalue": minimum_regularized,
        "regularized_condition": (
            float(regularized[-1]) / minimum_regularized if separated else None
        ),
        "regularized_spectrum_separated_from_zero": bool(separated),
        "rank_threshold": threshold,
        "scope": "Valid rows of W and W + C; host float64 eigvalsh.",
        "uniqueness_claim": "none: positive response spectrum alone does not prove global Coulomb solution uniqueness",
    }


def _jvp_function(controls: ConeControls) -> Callable[..., ContactConeJVP]:
    def jvp(
        program: ContactConeProgram, solved: ContactConeResult, tangent: Array
    ) -> ContactConeJVP:
        return contact_cone_solution_jvp(
            program,
            solved,
            ContactConeTangent(program, free_velocity=tangent),
            regularity=controls.regularity,
            margin_factor=controls.margin_factor,
            linear_tolerance=controls.precision.linear_tolerance,
            linear_iterations=controls.linear_iterations,
        )

    return jvp


def _derivative_campaign(
    native: ContactConeProgram,
    result: ContactConeResult,
    solve_phase: CompiledPhase,
    controls: ConeControls,
    settings: CampaignSettings,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], ContactConeJVP, Array]:
    """Prepared fixed-branch linearization reused by separately compiled JVP/VJP."""

    valid = np.asarray(native.valid)
    count, width = native.free_velocity.shape
    dtype = native.free_velocity.dtype
    direction = _masked_direction(rng, valid, width, dtype)
    cotangent = _masked_direction(rng, valid, width, dtype)

    regularity_statics: list[dict[str, Any]] = []

    def prepared(
        program: ContactConeProgram, solved: ContactConeResult
    ) -> PreparedContactConeSensitivity:
        return prepare_contact_cone_sensitivity(
            program,
            solved,
            regularity=controls.regularity,
            linear_tolerance=controls.precision.linear_tolerance,
            linear_iterations=controls.linear_iterations,
        )

    def prepare(
        program: ContactConeProgram, solved: ContactConeResult
    ) -> tuple[list[Array], dict[str, Array]]:
        value = prepared(program, solved)
        leaves = jax.tree.leaves(eqx.filter(value, eqx.is_array))
        certificate = value.regularity
        summary = {
            "primal_certified": value.fixed_branch.certified,
            "jacobian_regular": value.jacobian_regular,
            "regularity_margin": value.regularity_margin,
            "cone_margin": value.fixed_branch.cone_margin,
        }
        if certificate is None:
            regularity_statics.append({"route": None, "reason": "no regularity policy"})
            return leaves, summary
        cost = certificate.cost
        # Route, reason and admitted/refused cost are static planning facts.
        regularity_statics.append(
            {
                "route": certificate.route,
                "reason": certificate.reason,
                "rows": certificate.rows,
                "columns": certificate.columns,
                "cost": {
                    "accepted": cost.accepted,
                    "reason": cost.reason,
                    "storage_bytes": cost.storage_bytes,
                    "preparation_workspace_bytes": cost.preparation_workspace_bytes,
                    "operator_matvec_count": cost.operator_matvec_count,
                    "adjoint_matvec_count": cost.adjoint_matvec_count,
                    "provider_workspace_exact": cost.provider_workspace_exact,
                },
            }
        )
        summary.update(
            {
                "rank": certificate.rank,
                "right_nullity": certificate.right_nullity,
                "fixed_rank": certificate.fixed_rank,
                "converged": certificate.converged,
                "svd_status": certificate.svd_status,
                "gap_margin": certificate.gap_margin,
                "smallest_retained_singular_value": certificate.smallest_retained_singular_value,
                "largest_discarded_singular_value": certificate.largest_discarded_singular_value,
                "singular_value_error_bound": certificate.singular_value_error_bound,
            }
        )
        return leaves, summary

    def reuse(
        program: ContactConeProgram, solved: ContactConeResult, leaves: list[Array]
    ) -> PreparedContactConeSensitivity:
        # Prepared leaves replace the template's numerical state. The recorded
        # optimized-HLO targets expose whether template preparation survives;
        # source-level construction alone is not evidence of compiler removal.
        numeric, static = eqx.partition(prepared(program, solved), eqx.is_array)
        return eqx.combine(
            jax.tree.unflatten(jax.tree.structure(numeric), leaves), static
        )

    def jvp(
        program: ContactConeProgram,
        solved: ContactConeResult,
        leaves: list[Array],
        tangent: Array,
    ) -> ContactConeJVP:
        return contact_cone_solution_jvp(
            program,
            solved,
            ContactConeTangent(program, free_velocity=tangent),
            sensitivity=reuse(program, solved, leaves),
            margin_factor=controls.margin_factor,
        )

    def vjp(
        program: ContactConeProgram,
        solved: ContactConeResult,
        leaves: list[Array],
        impulse_cotangent: Array,
    ) -> ContactConeVJP:
        return contact_cone_solution_vjp(
            program,
            solved,
            impulse_cotangent,
            sensitivity=reuse(program, solved, leaves),
            margin_factor=controls.margin_factor,
            friction_cotangent=True,
        )

    preparation, _, prepare_record = _measure_phase(prepare, (native, result), settings)
    leaves, summary = preparation
    regular = _flag(summary["jacobian_regular"])
    certificate_record: dict[str, Any] = {
        **regularity_statics[0],
        **{
            name: (
                _flag(value)
                if np.asarray(value).dtype == np.bool_
                else int(value)
                if np.issubdtype(np.asarray(value).dtype, np.integer)
                else _number(value)
            )
            for name, value in summary.items()
            if name not in ("primal_certified", "jacobian_regular", "cone_margin")
        },
    }
    if regularity_statics[0]["route"] == "unavailable":
        regularity_status = "resource-refused"
    elif regular:
        regularity_status = "proven-regular"
    elif "right_nullity" in summary and int(summary["right_nullity"]) > 0:
        regularity_status = "rank-deficient"
    else:
        regularity_status = "inconclusive"
    prepare_record["outcome"] = {
        "status": (
            "prepared-admissible"
            if _flag(summary["primal_certified"]) and regular
            else "prepared-refusing"
        ),
        "primal_certified": _flag(summary["primal_certified"]),
        "jacobian_regularity": {"status": regularity_status, **certificate_record},
        "cone_margin_tolerance_units": _number(summary["cone_margin"]),
        "prepared_logical_bytes": logical_array_bytes(leaves),
        "branch_qualification": (
            f"applied per JVP/VJP call with margin factor {controls.margin_factor}"
        ),
    }
    tangent, _, jvp_record = _measure_phase(
        jvp, (native, result, leaves, direction), settings
    )
    jvp_record["outcome"] = _derivative_outcome(tangent.evidence, valid)
    adjoint, _, vjp_record = _measure_phase(
        vjp, (native, result, leaves, cotangent), settings
    )
    vjp_record["outcome"] = _derivative_outcome(adjoint.evidence, valid)
    accepted_jvp = jvp_record["outcome"]["status"] == "accepted"
    accepted_vjp = vjp_record["outcome"]["status"] == "accepted"
    precision = controls.precision
    if accepted_jvp and accepted_vjp:
        left = float(jnp.vdot(cotangent, tangent.impulse_tangent))
        right = float(jnp.vdot(adjoint.cotangent.free_velocity, direction))
        error = _relative(abs(left - right), max(abs(left), abs(right)))
        adjoint_check: dict[str, Any] = {
            "status": "agrees" if error <= precision.adjoint_agreement else "disagrees",
            "cotangent_dot_jvp": left,
            "vjp_dot_tangent": right,
            "relative_error": _number(error),
            "agreement_threshold": precision.adjoint_agreement,
        }
    else:
        adjoint_check = {"status": "not-evaluated-derivative-refused"}
    if accepted_jvp:
        free = np.asarray(native.free_velocity)[valid]
        step = precision.finite_difference_relative_step * max(1.0, _maximum_abs(free))

        def perturbed(sign: float) -> ContactConeResult:
            return solve_phase(
                eqx.tree_at(
                    lambda program: program.free_velocity,
                    native,
                    native.free_velocity + sign * step * direction,
                )
            )

        plus = perturbed(1.0)
        minus = perturbed(-1.0)
        if not (_flag(plus.evidence.successful) and _flag(minus.evidence.successful)):
            finite_difference: dict[str, Any] = {"status": "perturbed-solve-uncertified"}
        elif not (
            np.array_equal(np.asarray(plus.branch), np.asarray(result.branch))
            and np.array_equal(np.asarray(minus.branch), np.asarray(result.branch))
        ):
            finite_difference = {"status": "branch-changed"}
        else:
            estimate = (_host64(plus.impulse) - _host64(minus.impulse)) / (2.0 * step)
            error = _relative(
                _difference(estimate, tangent.impulse_tangent),
                max(_maximum_abs(tangent.impulse_tangent), _maximum_abs(estimate)),
            )
            finite_difference = {
                "status": (
                    "agrees"
                    if error <= precision.finite_difference_agreement
                    else "disagrees"
                ),
                "relative_error": _number(error),
                "agreement_threshold": precision.finite_difference_agreement,
            }
        finite_difference = {
            **finite_difference,
            "step_velocity": step,
            "reference": (
                "Central difference of independent certified primal re-solves with "
                "perturbed free velocity; noise is bounded by solver tolerance / step."
            ),
        }
    else:
        finite_difference = {"status": "not-evaluated-derivative-refused"}
    return (
        {
            "direction": "free-velocity tangent and impulse cotangent, unit max-norm on valid rows",
            "sensitivity_preparation": prepare_record,
            "jvp": jvp_record,
            "vjp": vjp_record,
            "adjoint_consistency": adjoint_check,
            "finite_difference": finite_difference,
            "block_count": int(count),
        },
        tangent,
        direction,
    )


def _dense_authority(
    native: ContactConeProgram,
    native_result: ContactConeResult,
    native_solve: dict[str, Any],
    native_jvp: ContactConeJVP,
    direction: Array,
    source: DenseAuthoritySource,
    controls: ConeControls,
    settings: CampaignSettings,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Explicit dense authority under the declared policy; never a native fallback."""

    policy = settings.dense_policy
    dtype = native.free_velocity.dtype
    entries = source.rows * source.rows
    record: dict[str, Any] = {
        "route": "explicit-dense-authority",
        "assembly": source.description,
        "required_entries": entries,
        "required_host_bytes": entries * np.dtype(np.float64).itemsize,
        "required_device_bytes": entries * np.dtype(dtype).itemsize,
        "required_workspace_bytes": source.workspace_bytes,
    }
    try:
        policy.require(entries, np.float64)
        policy.require_workspace(source.workspace_bytes)
    except la.LinearCapabilityError as error:
        record["status"] = "resource-refused"
        record["reason"] = str(error)
        return record
    valid = np.asarray(native.valid)
    matrix, assembly_seconds = measure_host(source.assemble)
    spectrum, spectrum_seconds = measure_host(
        lambda: _spectrum(matrix, valid, np.asarray(native.compliance, dtype=np.float64))
    )
    device_matrix, transfer_seconds = measure_synchronized(
        lambda: jax.device_put(matrix.astype(dtype))
    )
    dense, binding_seconds = measure_synchronized(
        lambda: _with_response(
            native,
            dense_contact_response(
                device_matrix,
                native.contact_space,
                operator_id=f"{native.program_id}/dense-authority-response",
            ),
            f"{native.program_id}/dense-authority",
        )
    )
    preparation = {
        "host_assembly_seconds": assembly_seconds,
        "input_transfer_synchronized_seconds": transfer_seconds,
        "program_binding_synchronized_seconds": binding_seconds,
    }
    solver = controls.solver

    def solve(program: ContactConeProgram) -> ContactConeResult:
        return solve_contact_cone(program, solver=solver)

    result, _, solve_record = _measure_phase(solve, (dense,), settings)
    solve_record["outcome"] = _solve_outcome(result, valid)
    tangent, _, jvp_record = _measure_phase(
        _jvp_function(controls), (dense, result, direction), settings
    )
    jvp_record["outcome"] = _derivative_outcome(tangent.evidence, valid)
    probe = jnp.asarray(rng.normal(size=native.free_velocity.shape), dtype=dtype)
    native_action = eqx.filter_jit(lambda program, value: program.response_action(value))(
        native, probe
    )
    reference_action = (matrix @ _host64(probe).reshape(-1)).reshape(probe.shape)
    action_error = _relative(
        _difference(native_action, reference_action), _maximum_abs(reference_action)
    )
    certify = eqx.filter_jit(certify_contact_cone_impulse)
    native_certified = _flag(native_result.evidence.successful)
    dense_certified = _flag(result.evidence.successful)
    native_by_dense = _flag(certify(dense, native_result.impulse).successful)
    dense_by_native = _flag(certify(native, result.impulse).successful)
    if not native_certified:
        parity_status = "not-comparable-native-uncertified"
    elif not dense_certified:
        parity_status = "not-comparable-dense-uncertified"
    elif native_by_dense and dense_by_native:
        parity_status = "cross-certified"
    else:
        parity_status = "cross-certification-failed"
    native_jvp_accepted = _flag(native_jvp.evidence.successful)
    dense_jvp_accepted = jvp_record["outcome"]["status"] == "accepted"
    derivative_parity: dict[str, Any] = (
        {
            "status": "compared",
            "regularized_spectrum_separated_from_zero": spectrum[
                "regularized_spectrum_separated_from_zero"
            ],
            "max_abs_impulse_tangent_difference": _difference(
                tangent.impulse_tangent, native_jvp.impulse_tangent
            ),
            "relative_impulse_tangent_difference": _relative(
                _difference(tangent.impulse_tangent, native_jvp.impulse_tangent),
                _maximum_abs(tangent.impulse_tangent),
            ),
        }
        if native_jvp_accepted and dense_jvp_accepted
        else {
            "status": "not-comparable-derivative-refused",
            "native_accepted": native_jvp_accepted,
            "dense_accepted": dense_jvp_accepted,
        }
    )
    matched = parity_status == "cross-certified"
    native_median = native_solve["warm_synchronized"]["median_seconds"]
    dense_median = solve_record["warm_synchronized"]["median_seconds"]
    record.update(
        {
            "status": "admitted",
            "preparation": preparation,
            "host_spectrum_seconds": spectrum_seconds,
            "spectrum": spectrum,
            "solve": solve_record,
            "jvp_unprepared": jvp_record,
            "cold_path": _cold_path(preparation, solve_record),
            "action_parity": {
                "status": (
                    "agrees"
                    if action_error <= controls.precision.action_agreement
                    else "disagrees"
                ),
                "relative_error": _number(action_error),
                "agreement_threshold": controls.precision.action_agreement,
                "reference": "host float64 dense W applied to the same probe",
            },
            "primal_parity": {
                "status": parity_status,
                "regularized_spectrum_separated_from_zero": spectrum[
                    "regularized_spectrum_separated_from_zero"
                ],
                "native_impulse_certified_by_dense_response": native_by_dense,
                "dense_impulse_certified_by_native_response": dense_by_native,
                "max_abs_impulse_difference": _difference(
                    result.impulse, native_result.impulse
                ),
                "max_abs_post_velocity_difference": _difference(
                    result.post_relative_velocity, native_result.post_relative_velocity
                ),
                "max_abs_law_velocity_difference": _difference(
                    result.contact_law_velocity, native_result.contact_law_velocity
                ),
            },
            "derivative_parity": derivative_parity,
            "matched_work_warm_median_ratio_dense_over_native": (
                dense_median / native_median if matched and native_median else None
            ),
            "matched_work_reason": (
                "Both solves certified and each impulse is certified by the other response."
                if matched
                else f"No ratio: primal parity status is {parity_status}."
            ),
        }
    )
    return record


def _cone_campaign(
    native: ContactConeProgram,
    source: DenseAuthoritySource,
    controls: ConeControls,
    settings: CampaignSettings,
    preparation: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    valid = np.asarray(native.valid)
    solver = controls.solver
    sap = controls.sap

    def solve(program: ContactConeProgram) -> ContactConeResult:
        return solve_contact_cone(program, solver=solver)

    def solve_sap(program: ContactConeProgram) -> ContactConeResult:
        return solve_contact_sap(program, solver=sap)

    result, solve_phase, solve_record = _measure_phase(solve, (native,), settings)
    solve_record["outcome"] = _solve_outcome(result, valid)
    solve_record["outcome"]["recertified_with_current_actions"] = _flag(
        eqx.filter_jit(contact_cone_result_is_certified)(native, result)
    )
    sap_result, _, sap_record = _measure_phase(solve_sap, (native,), settings)
    sap_record["outcome"] = _solve_outcome(sap_result, valid)
    both = _flag(result.evidence.successful) and _flag(sap_result.evidence.successful)
    sap_record["post_velocity_difference_from_projected_gradient"] = (
        _difference(sap_result.post_relative_velocity, result.post_relative_velocity)
        if both
        else None
    )
    derivatives, tangent, direction = _derivative_campaign(
        native, result, solve_phase, controls, settings, rng
    )
    dense = _dense_authority(
        native, result, solve_record, tangent, direction, source, controls, settings, rng
    )
    return {
        "native_route": {
            "response": "matrix-free native operator actions; no dense response is formed",
            "response_operator_id": native.response.operator_id,
            "preparation": preparation,
            "program_logical_bytes": logical_array_bytes(native),
            "solve": solve_record,
            "sap": sap_record,
            "cold_path": _cold_path(preparation, solve_record),
            "derivatives": derivatives,
        },
        "dense_authority": dense,
        "controls": controls.to_dict(),
        "active_blocks": int(np.count_nonzero(valid)),
        "block_capacity": int(valid.size),
        "block_width": int(native.local_dimension),
    }


def _collision_scene(
    segment_count: int,
) -> tuple[phx.discretization.PreparedCollisionScene, Array, Array]:
    left = np.arange(segment_count, dtype=np.float64)
    x = np.stack((left, left + 0.2), axis=-1).reshape((-1,))
    moving_positions = jnp.asarray(np.stack((x, np.full_like(x, 0.05)), axis=-1))
    source = la.ArraySpace(moving_positions.shape, dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        np.arange(2 * segment_count, dtype=np.int64),
        ambient_dimension=2,
        edges=np.arange(2 * segment_count, dtype=np.int32).reshape((-1, 2)),
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        moving_positions,
        phx.discretization.selection_collision_operator(
            source, jnp.arange(2 * segment_count, dtype=jnp.int32)
        ),
    )
    static_plan = phx.discretization.CollisionSurfacePlan(
        np.asarray((10_000, 10_001), dtype=np.int64),
        ambient_dimension=2,
        edges=np.asarray(((0, 1),), dtype=np.int32),
        body_ids=1,
        material_ids=0,
        static_mask=True,
    )
    static = phx.discretization.PreparedCollisionSurface(
        static_plan,
        jnp.asarray(((-1.0, 0.0), (segment_count + 1.0, 0.0))),
        phx.discretization.static_collision_operator(source, 2, 2, dtype=np.float64),
    )
    scene = phx.discretization.PreparedCollisionScene((moving, static))
    positions = scene.positions(source.zeros())
    velocities = scene.map_values(
        jnp.broadcast_to(jnp.asarray((0.2, -0.1)), source.shape)
    )
    return scene, positions, velocities


def _closure_scene(segment_count: int) -> ClosureScene:
    capacity = 4 * segment_count
    (scene, positions, velocities), scene_seconds = measure_synchronized(
        lambda: _collision_scene(segment_count)
    )
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=capacity,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    epoch, search_seconds = measure_synchronized(lambda: search.build(scene, positions))
    kinematics, kinematics_seconds = measure_synchronized(
        lambda: phx.discretization.evaluate_contact_kinematics(
            scene,
            epoch,
            positions,
            velocities,
            0.01,
            activation_distance=0.1,
        )
    )
    materials = ContactMaterialPairTable.uniform(
        normal_stiffness=100.0,
        static_friction=0.5,
        dynamic_friction=0.4,
    )
    closure = ContactClosurePlan(
        phx.applications.contact.BarrierNormalContactLaw(0.1),
        materials,
        tangential=phx.applications.contact.RegularizedCoulombContactLaw(1.0e-3),
    )
    state, state_seconds = measure_synchronized(
        lambda: (
            phx.applications.contact.remap_contact_route_state(
                ContactRouteState.empty(0, 1, closure.closure_id), kinematics
            ).candidate
        )
    )
    return ClosureScene(
        segment_count,
        capacity,
        kinematics,
        materials,
        closure,
        state,
        {
            "scene_preparation_synchronized_seconds": scene_seconds,
            "host_search_build_synchronized_seconds": search_seconds,
            "kinematics_eager_synchronized_seconds": kinematics_seconds,
            "route_state_remap_synchronized_seconds": state_seconds,
        },
    )


def _closure_row(scene: ClosureScene, settings: CampaignSettings) -> dict[str, Any]:
    def evaluate(
        plan: ContactClosurePlan,
        kinematics: ContactKinematicsEpoch,
        state: ContactRouteState,
    ) -> ContactClosureEvaluation:
        return phx.applications.contact.evaluate_contact_closure(plan, kinematics, state)

    evaluation, _, phase = _measure_phase(
        evaluate, (scene.closure, scene.kinematics, scene.state), settings
    )
    evidence = evaluation.evidence
    kinematics = scene.kinematics.evidence
    phase["outcome"] = {
        "status": "successful" if _flag(evidence.successful) else "unsuccessful",
        "total_potential": _number(evidence.total_potential),
        "total_dissipated_power": _number(evidence.total_dissipated_power),
        "maximum_cone_defect": _number(evidence.maximum_cone_defect),
        "active_contacts": int(evidence.active_contacts),
        "finite": _flag(evidence.finite),
        "material_complete": _flag(evidence.material_complete),
    }
    return {
        "segment_count": scene.segment_count,
        "route_capacity": scene.capacity,
        "laws": {
            "normal": "barrier, activation distance 0.1",
            "tangential": "regularized Coulomb, velocity threshold 1e-3",
        },
        "kinematics": {
            "status": "successful" if _flag(kinematics.successful) else "unsuccessful",
            "active_contacts": int(kinematics.active_contacts),
            "failed_contacts": int(kinematics.failed_contacts),
            "search_complete": _flag(kinematics.search_complete),
            "minimum_gap": _number(kinematics.minimum_gap),
        },
        "preparation": scene.preparation,
        "evaluation": phase,
        "cold_path": _cold_path(scene.preparation, phase),
        "reference": {
            "status": "not-available",
            "reason": "No independent closed-form barrier reference is evaluated here.",
        },
    }


def _host_block_diagonal(blocks: np.ndarray) -> np.ndarray:
    count, width, _ = blocks.shape
    matrix = np.zeros((count, width, count, width))
    index = np.arange(count)
    matrix[index, :, index, :] = blocks
    return matrix.reshape((count * width, count * width))


def _kinematic_cone_row(
    scene: ClosureScene, settings: CampaignSettings
) -> dict[str, Any]:
    count = scene.capacity
    width = 1 + int(scene.kinematics.batches[0].tangential_velocity.shape[-1])
    host_blocks, host_seconds = measure_host(
        lambda: np.ascontiguousarray(
            np.broadcast_to(np.eye(width), (count, width, width))
        )
    )
    blocks, transfer_seconds = measure_synchronized(lambda: jax.device_put(host_blocks))
    certificate = ContactCertificatePolicy(
        ("contact",),
        reference_velocity=1.0,
        impulse_tolerance=1.0e-8,
        law_velocity_tolerance=1.0e-8,
        slip_speed_tolerance=1.0e-8,
        complementarity_work_tolerance=1.0e-8,
        friction_work_tolerance=1.0e-8,
    )
    native, binding_seconds = measure_synchronized(
        lambda: phx.applications.contact.build_contact_cone_program(
            scene.kinematics,
            scene.materials,
            local_block_contact_response(blocks),
            certificate=certificate,
            compliance=1.0e-6,
        )
    )
    precision = _PRECISION["float64"]
    controls = ConeControls(
        ContactConeSolverPlan(maximum_iterations=settings.solver_iterations),
        SAPContactSolverPlan(
            maximum_iterations=settings.solver_iterations,
            tolerance=precision.sap_tolerance,
        ),
        precision,
        settings.margin_factor,
        settings.linear_iterations,
        settings.regularity,
    )
    rows = count * width
    source = DenseAuthoritySource(
        "host float64 block-diagonal assembly of the same local response blocks",
        rows,
        2 * rows * rows * np.dtype(np.float64).itemsize,
        lambda: _host_block_diagonal(host_blocks),
    )
    campaign = _cone_campaign(
        native,
        source,
        controls,
        settings,
        {
            "host_block_preparation_seconds": host_seconds,
            "input_transfer_synchronized_seconds": transfer_seconds,
            "program_binding_synchronized_seconds": binding_seconds,
        },
        settings.seed,
    )
    campaign.update(
        {
            "segment_count": scene.segment_count,
            "dtype": str(native.free_velocity.dtype),
            "native_response": "local identity blocks over canonical kinematic routes",
            "certificate_units": {
                "family": "contact",
                "reference_velocity": 1.0,
                "tolerances": "1e-8 in the declared family impulse/velocity/work units",
            },
            "compliance": 1.0e-6,
            "structural_rank_upper_bound": width * campaign["active_blocks"],
        }
    )
    return campaign


def _frames(rng: np.random.Generator, count: int) -> np.ndarray:
    normal = np.asarray((0.0, 0.0, 1.0)) + 0.35 * rng.normal(size=(count, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    first = np.cross(normal, np.asarray((1.0, 0.0, 0.0)))
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(normal, first)
    return np.stack((normal, first, second), axis=1)


def _coupling_bodies(
    coupling: Coupling, capacity: int
) -> tuple[int, np.ndarray, np.ndarray]:
    index = np.arange(capacity)
    match coupling:
        case "block-diagonal":
            bodies = capacity
            left = index
            right = np.full((capacity,), bodies)
        case "stack":
            bodies = capacity
            left = index
            right = np.where(index % _STACK_DEPTH == 0, bodies, index - 1)
        case "clustered":
            bodies = max(1, -(-capacity // _CLUSTER_SIZE))
            left = index % bodies
            right = np.where(
                (index % 2 == 0) | (bodies == 1), bodies, (left + 1) % bodies
            )
        case "redundant":
            bodies = max(1, -(-capacity // _REDUNDANCY))
            left = index // _REDUNDANCY
            right = np.full((capacity,), bodies)
        case unknown:
            assert_never(unknown)
    return bodies, left.astype(np.int32), right.astype(np.int32)


def _operator_factors(spec: OperatorRowSpec, seed: int) -> BodyCouplingFactors:
    rng = np.random.default_rng(seed)
    bodies, left, right = _coupling_bodies(spec.coupling, spec.capacity)
    frames = (
        _frames(rng, bodies)[left]
        if spec.coupling == "redundant"
        else _frames(rng, spec.capacity)
    )
    exponents = np.linspace(0.0, 1.0, bodies) if bodies > 1 else np.zeros((1,))
    masses = rng.permutation(spec.mass_ratio**exponents)
    inverse_mass = np.concatenate((1.0 / masses, np.zeros((1,))))
    body_velocity = rng.normal(scale=0.5, size=(bodies + 1, 3))
    body_velocity[:, 2] -= 1.0
    body_velocity[bodies] = 0.0
    free = np.einsum("cij,cj->ci", frames, body_velocity[left] - body_velocity[right])
    gap = rng.uniform(0.0, 1.0e-3, size=(spec.capacity,))
    static = rng.uniform(0.3, 0.8, size=(spec.capacity,))
    valid = np.arange(spec.capacity) < spec.active
    effective = 1.0 / (inverse_mass[left] + inverse_mass[right])
    reference = float(np.median(effective[valid] if spec.active else effective))
    return BodyCouplingFactors(
        frames,
        left,
        right,
        inverse_mass,
        free,
        gap / _STEP_SIZE,
        static,
        0.8 * static,
        valid,
        (_ROUTE_KEY_OFFSET + np.arange(spec.capacity)).astype(np.int64),
        bodies,
        reference,
    )


def _host_body_response(factors: BodyCouplingFactors) -> np.ndarray:
    """Independent host float64 ``W = G M^-1 G^T`` from the same physical factors."""

    count = factors.frames.shape[0]
    bodies = factors.body_count
    jacobian = np.zeros((count, _WIDTH, bodies + 1, _WIDTH))
    rows = np.arange(count)
    jacobian[rows, :, factors.left, :] += factors.frames
    jacobian[rows, :, factors.right, :] -= factors.frames
    jacobian = jacobian[:, :, :bodies, :].reshape((count * _WIDTH, bodies * _WIDTH))
    inverse = np.repeat(factors.inverse_mass[:bodies], _WIDTH)
    return (jacobian * inverse) @ jacobian.T


def _body_response(
    frames: Array,
    left: Array,
    right: Array,
    inverse_mass: Array,
    space: la.ArraySpace,
    operator_id: str,
) -> la.FunctionLinearOperator:
    """Matrix-free gather/scatter action of ``G M^-1 G^T``; symmetric by construction."""

    bodies = int(inverse_mass.shape[0])

    def action(impulse: Array) -> Array:
        world = jnp.einsum("cij,ci->cj", frames, impulse)
        force = (
            jnp.zeros((bodies, _WIDTH), dtype=impulse.dtype)
            .at[left]
            .add(world)
            .at[right]
            .add(-world)
        )
        velocity = inverse_mass[:, None] * force
        return jnp.einsum("cij,cj->ci", frames, velocity[left] - velocity[right])

    return la.FunctionLinearOperator(
        action,
        source=la.DualSpace(space),
        target=space,
        transpose_action=action,
        operator_id=operator_id,
    )


def _translational_certificate(relative: float, mass: float) -> ContactCertificatePolicy:
    velocity = _REFERENCE_VELOCITY
    return ContactCertificatePolicy(
        ("translational-contact",),
        reference_velocity=velocity,
        impulse_tolerance=relative * mass * velocity,
        law_velocity_tolerance=relative * velocity,
        slip_speed_tolerance=relative * velocity,
        complementarity_work_tolerance=relative * mass * velocity**2,
        friction_work_tolerance=relative * mass * velocity**2,
    )


def _operator_row(
    spec: OperatorRowSpec, settings: CampaignSettings, index: int
) -> dict[str, Any]:
    seed = settings.seed + 7919 * (index + 1)
    precision = _PRECISION[spec.dtype]
    dtype = np.dtype(spec.dtype)
    factors, host_seconds = measure_host(lambda: _operator_factors(spec, seed))
    device, transfer_seconds = measure_synchronized(
        lambda: jax.device_put(
            (
                factors.frames.astype(dtype),
                factors.left,
                factors.right,
                factors.inverse_mass.astype(dtype),
                factors.free_velocity.astype(dtype),
                factors.normal_bias.astype(dtype),
                factors.static_friction.astype(dtype),
                factors.dynamic_friction.astype(dtype),
                factors.valid,
                factors.route_keys,
            )
        )
    )
    frames, left, right, inverse_mass, free, bias, static, dynamic, valid, keys = device
    name = f"contact-closure-benchmark/{spec.label}"
    certificate = _translational_certificate(
        precision.certificate_relative_tolerance, factors.effective_mass_reference
    )

    def bind() -> ContactConeProgram:
        space = contact_cone_space(spec.capacity, _WIDTH, dtype)
        return ContactConeProgram(
            _body_response(
                frames, left, right, inverse_mass, space, f"{name}/native-body-response"
            ),
            free,
            spec.compliance,
            dynamic,
            keys,
            valid,
            ContactBlockLayout.coulomb(spec.capacity, _WIDTH),
            certificate,
            f"{name}/native",
            static_friction=static,
            normal_bias=bias,
        )

    native, binding_seconds = measure_synchronized(bind)
    controls = ConeControls(
        ContactConeSolverPlan(maximum_iterations=settings.solver_iterations),
        SAPContactSolverPlan(
            maximum_iterations=settings.solver_iterations,
            tolerance=precision.sap_tolerance,
        ),
        precision,
        settings.margin_factor,
        settings.linear_iterations,
        settings.regularity,
    )
    rows = spec.capacity * _WIDTH
    float_bytes = np.dtype(np.float64).itemsize
    jacobian_bytes = rows * (factors.body_count + 1) * _WIDTH * float_bytes
    source = DenseAuthoritySource(
        "host float64 W = G M^-1 G^T from the same frames, body incidences and masses",
        rows,
        2 * jacobian_bytes + 2 * rows * rows * float_bytes,
        lambda: _host_body_response(factors),
    )
    campaign = _cone_campaign(
        native,
        source,
        controls,
        settings,
        {
            "host_factor_preparation_seconds": host_seconds,
            "input_transfer_synchronized_seconds": transfer_seconds,
            "program_binding_synchronized_seconds": binding_seconds,
        },
        seed,
    )
    touched = np.unique(
        np.concatenate((factors.left[factors.valid], factors.right[factors.valid]))
    )
    campaign.update(
        {
            "row": spec.to_dict(),
            "native_response": (
                "point-body gather/scatter G M^-1 G^T over local frames; "
                f"stack depth {_STACK_DEPTH}, cluster size {_CLUSTER_SIZE}, "
                f"redundancy {_REDUNDANCY}"
            ),
            "body_count": factors.body_count,
            "structural_rank_upper_bound": _WIDTH
            * min(spec.active, int(np.count_nonzero(touched < factors.body_count))),
            "normal_bias": f"position-level gap / h with gap in [0, 1e-3] m and h = {_STEP_SIZE} s",
            "certificate_units": {
                "family": "translational-contact",
                "reference_velocity_m_per_s": _REFERENCE_VELOCITY,
                "effective_mass_reference_kg": factors.effective_mass_reference,
                "relative_tolerance": precision.certificate_relative_tolerance,
                "impulse_tolerance_n_s": precision.certificate_relative_tolerance
                * factors.effective_mass_reference
                * _REFERENCE_VELOCITY,
                "velocity_tolerance_m_per_s": precision.certificate_relative_tolerance
                * _REFERENCE_VELOCITY,
                "work_tolerance_j": precision.certificate_relative_tolerance
                * factors.effective_mass_reference
                * _REFERENCE_VELOCITY**2,
                "compliance_units": "m s^-1 per N s",
            },
        }
    )
    return campaign


def _rough_row(grid: int, settings: CampaignSettings) -> dict[str, Any]:
    (spectrum, gap_host), host_seconds = measure_host(
        lambda: (np.ones((grid, grid)), np.full((grid, grid), _ROUGH_GAP))
    )
    plan, binding_seconds = measure_synchronized(
        lambda: PeriodicRoughContactPlan(
            spectrum, maximum_iterations=200, tolerance=1.0e-8
        )
    )
    gap, transfer_seconds = measure_synchronized(lambda: jax.device_put(gap_host))

    def solve(
        rough: PeriodicRoughContactPlan, undeformed: Array
    ) -> PeriodicRoughContactResult:
        return phx.applications.contact.solve_periodic_rough_contact(rough, undeformed)

    result, _, phase = _measure_phase(solve, (plan, gap), settings)
    reference = -_ROUGH_GAP / float(spectrum[0, 0])
    error = _relative(_difference(result.pressure, reference), reference)
    load = float(result.evidence.total_load)
    reference_load = reference * grid * grid
    evidence = result.evidence
    phase["outcome"] = {
        "status": "successful" if _flag(evidence.successful) else "unsuccessful",
        "converged": _flag(evidence.converged),
        "iterations": int(evidence.iterations),
        "projected_residual": _number(evidence.projected_residual),
        "complementarity_defect": _number(evidence.complementarity_defect),
        "total_load": _number(load),
        "contact_area_fraction": _number(evidence.contact_area_fraction),
    }
    preparation = {
        "host_input_preparation_seconds": host_seconds,
        "plan_binding_synchronized_seconds": binding_seconds,
        "input_transfer_synchronized_seconds": transfer_seconds,
    }
    return {
        "grid": [grid, grid],
        "preparation": preparation,
        "solve": phase,
        "cold_path": _cold_path(preparation, phase),
        "reference": {
            "kind": "analytic uniform pressure -g0 / S[0, 0] for a uniform gap",
            "pressure": reference,
            "relative_pressure_error": _number(error),
            "relative_load_error": _number(
                _relative(abs(load - reference_load), reference_load)
            ),
            "status": "agrees" if error <= 1.0e-6 else "disagrees",
        },
    }


def _profile_rows(profile: Profile) -> tuple[OperatorRowSpec, ...]:
    match profile:
        case "legacy":
            return ()
        case "smoke":
            return (
                OperatorRowSpec("block-diagonal", 24, 24, 10.0, 1.0e-6, "float64"),
                OperatorRowSpec("stack", 24, 18, 10.0, 1.0e-6, "float64"),
                OperatorRowSpec("clustered", 48, 48, 10.0, 1.0e-6, "float64"),
                OperatorRowSpec("redundant", 24, 24, 1.0, 0.0, "float64"),
                OperatorRowSpec("stack", 24, 24, 10.0, 1.0e-6, "float32"),
            )
        case "scaling":
            capacity = tuple(
                OperatorRowSpec(coupling, size, size, 10.0, 1.0e-6, "float64")
                for coupling in ("stack", "clustered")
                for size in (64, 256, 1024, 4096, 16384)
            )
            return (
                *capacity,
                OperatorRowSpec("block-diagonal", 1024, 1024, 10.0, 1.0e-6, "float64"),
                OperatorRowSpec("block-diagonal", 16384, 16384, 10.0, 1.0e-6, "float64"),
                *(
                    OperatorRowSpec("stack", 4096, active, 10.0, 1.0e-6, "float64")
                    for active in (0, 256, 2048)
                ),
                *(
                    OperatorRowSpec("stack", 1024, 1024, ratio, 1.0e-6, "float64")
                    for ratio in (1.0, 1.0e2, 1.0e4)
                ),
                OperatorRowSpec("redundant", 1024, 1024, 10.0, 0.0, "float64"),
                OperatorRowSpec("redundant", 1024, 1024, 10.0, 1.0e-6, "float64"),
                OperatorRowSpec("clustered", 1024, 1024, 10.0, 0.0, "float64"),
                OperatorRowSpec("stack", 1024, 1024, 10.0, 1.0e-6, "float32"),
                OperatorRowSpec("clustered", 4096, 4096, 10.0, 1.0e-6, "float32"),
                OperatorRowSpec("redundant", 1024, 1024, 10.0, 1.0e-6, "float32"),
                OperatorRowSpec("block-diagonal", 16384, 16384, 10.0, 1.0e-6, "float32"),
            )
        case unknown:
            assert_never(unknown)


_PROFILE_DEFAULTS: dict[Profile, tuple[int, int, int, int]] = {
    # segment count, rough grid, repeats, warmup
    "legacy": (128, 64, 20, 1),
    "smoke": (8, 16, 2, 1),
    "scaling": (512, 256, 10, 2),
}


def _reset() -> None:
    """Every row lowers and compiles from cleared in-process JAX caches."""

    jax.clear_caches()
    gc.collect()


def _summary(records: dict[str, Any]) -> dict[str, Any]:
    cones = [records["kinematic_cone"]] if records.get("kinematic_cone") else []
    cones += records.get("operator_cone", [])
    native = Counter(cone["native_route"]["solve"]["outcome"]["status"] for cone in cones)
    jvp = Counter(
        cone["native_route"]["derivatives"]["jvp"]["outcome"]["status"] for cone in cones
    )
    vjp = Counter(
        cone["native_route"]["derivatives"]["vjp"]["outcome"]["status"] for cone in cones
    )
    dense = Counter(cone["dense_authority"]["status"] for cone in cones)
    regularity = Counter(
        cone["native_route"]["derivatives"]["sensitivity_preparation"]["outcome"][
            "jacobian_regularity"
        ]["status"]
        for cone in cones
    )
    parity = Counter(
        cone["dense_authority"]["primal_parity"]["status"]
        for cone in cones
        if cone["dense_authority"]["status"] == "admitted"
    )
    return {
        "cone_rows": len(cones),
        "native_solve_status_counts": dict(sorted(native.items())),
        "jvp_status_counts": dict(sorted(jvp.items())),
        "vjp_status_counts": dict(sorted(vjp.items())),
        "jacobian_regularity_status_counts": dict(sorted(regularity.items())),
        "dense_authority_status_counts": dict(sorted(dense.items())),
        "primal_parity_status_counts": dict(sorted(parity.items())),
    }


def run(
    *,
    profile: Profile,
    sections: Sequence[Section],
    rows: Sequence[OperatorRowSpec],
    segment_count: int,
    rough_grid: int,
    settings: CampaignSettings,
) -> dict[str, Any]:
    if not 1 <= segment_count <= _MAXIMUM_SEGMENTS:
        raise ValueError(f"segment count must lie in [1, {_MAXIMUM_SEGMENTS}].")
    if not 1 <= rough_grid <= _MAXIMUM_ROUGH_GRID:
        raise ValueError(f"rough grid must lie in [1, {_MAXIMUM_ROUGH_GRID}].")
    if len(rows) > _MAXIMUM_ROWS:
        raise ValueError(f"At most {_MAXIMUM_ROWS} operator-cone rows are allowed.")
    if not 1 <= settings.repeats <= _MAXIMUM_REPEATS:
        raise ValueError(f"repeats must lie in [1, {_MAXIMUM_REPEATS}].")
    if not 0 <= settings.warmup <= _MAXIMUM_WARMUP:
        raise ValueError(f"warmup must lie in [0, {_MAXIMUM_WARMUP}].")
    if not 1 <= settings.solver_iterations <= _MAXIMUM_SOLVER_ITERATIONS:
        raise ValueError(
            f"solver iterations must lie in [1, {_MAXIMUM_SOLVER_ITERATIONS}]."
        )
    if not 1 <= settings.linear_iterations <= _MAXIMUM_LINEAR_ITERATIONS:
        raise ValueError(
            f"linear iterations must lie in [1, {_MAXIMUM_LINEAR_ITERATIONS}]."
        )
    if not (math.isfinite(settings.margin_factor) and settings.margin_factor >= 0.0):
        raise ValueError("margin factor must be finite and nonnegative.")
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    records: dict[str, Any] = {}
    scene: ClosureScene | None = None
    if "closure" in sections or "kinematic-cone" in sections:
        _reset()
        scene = _closure_scene(segment_count)
    if "closure" in sections and scene is not None:
        _reset()
        records["closure"] = _closure_row(scene, settings)
    if "kinematic-cone" in sections and scene is not None:
        _reset()
        records["kinematic_cone"] = _kinematic_cone_row(scene, settings)
    del scene
    if "operator-cone" in sections:
        operator: list[dict[str, Any]] = []
        for index, spec in enumerate(rows):
            _reset()
            operator.append(_operator_row(spec, settings, index))
        records["operator_cone"] = operator
    if "rough" in sections:
        _reset()
        records["rough"] = _rough_row(rough_grid, settings)
    payload: dict[str, Any] = {
        "benchmark": "contact-closure",
        "environment": capture_environment().to_dict(),
        "settings": {
            "profile": profile,
            "sections": list(sections),
            "segment_count": segment_count,
            "rough_grid": rough_grid,
            "operator_rows": [spec.to_dict() for spec in rows],
            **settings.to_dict(),
        },
        "cache_policy": {
            "in_process": (
                "jax.clear_caches() before every row; each row lowers and compiles "
                "its own executables, and first execution is reported apart from "
                "warm samples."
            ),
            "persistent_compilation_cache_dir": os.environ.get(
                "JAX_COMPILATION_CACHE_DIR"
            ),
        },
        "provider_memory": _PROVIDER_MEMORY,
        "records": records,
        "summary": _summary(records),
    }
    identity = capture_benchmark_identity(_ROOT, Path(__file__).resolve(), payload.keys())
    payload["identity"] = identity.to_dict()
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase-separated contact closure, operator cone and derivative campaign."
    )
    parser.add_argument("--profile", choices=get_args(Profile), default="legacy")
    parser.add_argument("--sections", nargs="+", choices=get_args(Section))
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        type=OperatorRowSpec.parse,
        metavar="COUPLING:CAPACITY:ACTIVE:MASS_RATIO:COMPLIANCE:DTYPE",
        help="Append one explicit operator-cone row to the profile rows.",
    )
    parser.add_argument("--segment-count", type=int)
    parser.add_argument("--rough-grid", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--solver-iterations", type=int, default=2000)
    parser.add_argument("--linear-iterations", type=int, default=400)
    parser.add_argument("--margin-factor", type=float, default=1.0)
    parser.add_argument("--dense-max-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument(
        "--dense-max-workspace-bytes", type=int, default=1024 * 1024 * 1024
    )
    parser.add_argument(
        "--regularity-max-bytes",
        type=int,
        default=128 * 1024 * 1024,
        help="Fixed-branch Jacobian materialization limit for the native rank proof.",
    )
    parser.add_argument(
        "--regularity-workspace-bytes", type=int, default=512 * 1024 * 1024
    )
    parser.add_argument("--regularity-matvecs", type=int, default=8192)
    parser.add_argument(
        "--regularity-relative-cutoff",
        type=float,
        help="Explicit RankPolicy relative cutoff; omitted uses the native default.",
    )
    parser.add_argument("--seed", type=int, default=20_261_022)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    profiles: dict[str, Profile] = {name: name for name in get_args(Profile)}
    section_names: dict[str, Section] = {name: name for name in get_args(Section)}
    profile = profiles[arguments.profile]
    rows = (*_profile_rows(profile), *arguments.row)
    if arguments.sections is not None:
        sections = tuple(section_names[name] for name in arguments.sections)
    elif profile == "legacy" and not rows:
        sections = ("closure", "kinematic-cone", "rough")
    else:
        sections = get_args(Section)
    segment_count, rough_grid, repeats, warmup = _PROFILE_DEFAULTS[profile]
    policy = la.MaterializationPolicy(
        max_entries=max(1, arguments.dense_max_bytes // np.dtype(np.float64).itemsize),
        max_bytes=arguments.dense_max_bytes,
        max_workspace_bytes=arguments.dense_max_workspace_bytes,
    )
    settings = CampaignSettings(
        warmup=warmup if arguments.warmup is None else arguments.warmup,
        repeats=repeats if arguments.repeats is None else arguments.repeats,
        solver_iterations=arguments.solver_iterations,
        linear_iterations=arguments.linear_iterations,
        margin_factor=arguments.margin_factor,
        seed=arguments.seed,
        dense_policy=policy,
        regularity=SVDSolvePolicy(
            DenseSVD(algorithm="qr"),
            rank=la.RankPolicy(relative_cutoff=arguments.regularity_relative_cutoff),
            materialization=la.MaterializationPolicy(
                max_entries=max(
                    1, arguments.regularity_max_bytes // np.dtype(np.float64).itemsize
                ),
                max_bytes=arguments.regularity_max_bytes,
                max_workspace_bytes=arguments.regularity_workspace_bytes,
            ),
            resources=SVDResourcePolicy(
                preparation_bytes=arguments.regularity_workspace_bytes,
                workspace_bytes=arguments.regularity_workspace_bytes,
                operator_matvecs=arguments.regularity_matvecs,
            ),
        ),
    )
    payload = run(
        profile=profile,
        sections=sections,
        rows=rows,
        segment_count=(
            segment_count if arguments.segment_count is None else arguments.segment_count
        ),
        rough_grid=rough_grid if arguments.rough_grid is None else arguments.rough_grid,
        settings=settings,
    )
    if arguments.output is not None:
        print(write_json_atomic(arguments.output, payload))
    else:
        print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
