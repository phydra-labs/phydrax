#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import uuid
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from ..._bounds import Bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._cones import NonnegativeCone, ProductCone, ZeroCone
from ._policy import (
    ClarabelInteriorPoint,
    ConvexSolvePolicy,
    DensePrimalDualQP,
    MPAXr2HPDHG,
    MPAXraPDHG,
    QPaxInteriorPoint,
)
from ._problem import ConicProgram, LinearProgram
from ._quadratic import (
    _apply_failure_policy,
    _validate_quadratic_resources,
    ConvexProgramResult,
    QuadraticProgram,
    solve_quadratic_program,
)
from ._types import ConvexProgramProvenance, ConvexWarmStart


CanonicalProgram: TypeAlias = LinearProgram | QuadraticProgram | ConicProgram


def _program_kind(program: CanonicalProgram, /) -> str:
    if isinstance(program, LinearProgram):
        return "linear-program"
    if isinstance(program, QuadraticProgram):
        return "quadratic-program"
    if program.quadratic is None:
        return "linear-conic-program"
    return "quadratic-conic-program"


def _program_signature(program: CanonicalProgram, /) -> str:
    if isinstance(program, LinearProgram):
        return program.structure_id
    if isinstance(program, ConicProgram):
        return program.structure_id
    return program.structure_id


def _program_arrays(program: CanonicalProgram, /) -> tuple[Array, ...]:
    if isinstance(program, LinearProgram):
        return (
            program.linear,
            program.equality_matrix,
            program.equality_rhs,
            program.inequality_matrix,
            program.inequality_rhs,
            program.lower_bounds,
            program.upper_bounds,
        )
    if isinstance(program, QuadraticProgram):
        return (
            program.quadratic,
            program.linear,
            program.equality_matrix,
            program.equality_rhs,
            program.inequality_matrix,
            program.inequality_rhs,
            program.lower_bounds,
            program.upper_bounds,
        )
    arrays = (
        program.linear,
        program.constraint_matrix,
        program.constraint_rhs,
        program.lower_bounds,
        program.upper_bounds,
    )
    return arrays if program.quadratic is None else (program.quadratic, *arrays)


def _program_numeric_fingerprint(program: CanonicalProgram, /) -> str:
    arrays = _program_arrays(program)
    if any(isinstance(value, jax.core.Tracer) for value in arrays):
        return "traced-numeric-program"
    return array_tree_fingerprint(arrays)


def _validate_program_resources(
    program: CanonicalProgram,
    policy: ConvexSolvePolicy,
    /,
) -> None:
    arrays = _program_arrays(program)
    input_entries = sum(int(array.size) for array in arrays)
    input_bytes = sum(int(array.size) * int(array.dtype.itemsize) for array in arrays)
    materialization = policy.materialization
    if input_entries > materialization.max_entries:
        raise ValueError(
            f"Canonical program requires {input_entries} dense entries, exceeding "
            f"the materialization limit {materialization.max_entries}."
        )
    if input_bytes > materialization.max_bytes:
        raise ValueError(
            f"Canonical program requires {input_bytes} dense bytes, exceeding "
            f"the materialization limit {materialization.max_bytes}."
        )
    method = policy.method
    if isinstance(method, (DensePrimalDualQP, QPaxInteriorPoint)):
        _validate_quadratic_resources(
            _quadratic_program(program),
            policy,
            max_dense_dimension=method.max_kkt_dimension,
        )


class ConvexProgramPlan(StrictModule):
    """Backend selection and immutable structure for one canonical program."""

    policy: ConvexSolvePolicy
    problem_kind: str = eqx.field(static=True)
    problem_signature: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, program: CanonicalProgram, policy: ConvexSolvePolicy, /):
        if not isinstance(program, (LinearProgram, QuadraticProgram, ConicProgram)):
            raise TypeError(
                "program must be a LinearProgram, QuadraticProgram, or ConicProgram."
            )
        if not isinstance(policy, ConvexSolvePolicy):
            raise TypeError("policy must be a ConvexSolvePolicy.")
        kind = _program_kind(program)
        capabilities = policy.method.capabilities
        if kind == "linear-program" and not capabilities.linear_program:
            raise ValueError(f"Method {policy.method.method_id!r} does not support LPs.")
        if kind == "quadratic-program" and not capabilities.quadratic_program:
            raise ValueError(f"Method {policy.method.method_id!r} does not support QPs.")
        if "conic" in kind and not capabilities.conic_program:
            raise ValueError(
                f"Method {policy.method.method_id!r} does not support general conic programs."
            )
        _validate_program_resources(program, policy)
        signature = _program_signature(program)
        self.policy = policy
        self.problem_kind = kind
        self.problem_signature = signature
        self.plan_id = canonical_fingerprint(
            {
                "kind": "convex-program-plan",
                "program": signature,
                "policy": policy.policy_id,
            }
        )


class ConvexProgramTemplate(StrictModule):
    """Coefficient-independent program structure and selected backend plan."""

    plan: ConvexProgramPlan
    symbolic_state: Any
    template_id: str = eqx.field(static=True)

    def __init__(self, plan: ConvexProgramPlan, symbolic_state: Any = None, /):
        if not isinstance(plan, ConvexProgramPlan):
            raise TypeError("plan must be a ConvexProgramPlan.")
        self.plan = plan
        self.symbolic_state = symbolic_state
        self.template_id = canonical_fingerprint(
            {"kind": "convex-program-template", "plan": plan.plan_id}
        )


class PreparedConvexProgram(StrictModule):
    """Numeric program state bound to one reusable symbolic template."""

    program: CanonicalProgram
    template: ConvexProgramTemplate
    state: Any
    numeric_version: Array
    numeric_binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        program: CanonicalProgram,
        template: ConvexProgramTemplate,
        state: Any = None,
        /,
        *,
        numeric_version: Any = 0,
        numeric_binding_id: str | None = None,
    ):
        if not isinstance(program, (LinearProgram, QuadraticProgram, ConicProgram)):
            raise TypeError("program must be a canonical convex program.")
        if not isinstance(template, ConvexProgramTemplate):
            raise TypeError("template must be a ConvexProgramTemplate.")
        if _program_signature(program) != template.plan.problem_signature:
            raise ValueError("Program structure does not match the prepared template.")
        raw_version = jnp.asarray(numeric_version)
        if raw_version.shape != () or not jnp.issubdtype(
            raw_version.dtype, jnp.signedinteger
        ):
            raise TypeError("numeric_version must be one signed integer scalar.")
        version = raw_version.astype(jnp.int32)
        binding = (
            canonical_fingerprint(
                {
                    "kind": "convex-program-numeric-binding",
                    "structure": _program_signature(program),
                    "arrays": _program_numeric_fingerprint(program),
                    "instance": uuid.uuid4().hex,
                }
            )
            if numeric_binding_id is None
            else str(numeric_binding_id)
        )
        if not binding:
            raise ValueError("numeric_binding_id must be non-empty.")
        self.program = program
        self.template = template
        self.state = state
        self.numeric_version = eqx.error_if(
            version,
            version < 0,
            "numeric_version must be non-negative.",
        )
        self.numeric_binding_id = binding

    @property
    def plan(self) -> ConvexProgramPlan:
        return self.template.plan


class ConvexProgramExecution(StrictModule):
    """Audited result paired with the prepared numeric version that produced it."""

    result: ConvexProgramResult
    numeric_version: Array
    plan_id: str = eqx.field(static=True)
    numeric_binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        result: ConvexProgramResult,
        /,
        *,
        numeric_version: Any,
        plan_id: str,
        numeric_binding_id: str,
    ):
        if not isinstance(result, ConvexProgramResult):
            raise TypeError("result must be a ConvexProgramResult.")
        identifier = str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        binding = str(numeric_binding_id)
        if not binding:
            raise ValueError("numeric_binding_id must be non-empty.")
        if result.provenance.numeric_binding_id != binding:
            raise ValueError(
                "Execution result provenance does not match its numeric binding."
            )
        raw_version = jnp.asarray(numeric_version)
        if raw_version.shape != () or not jnp.issubdtype(
            raw_version.dtype, jnp.signedinteger
        ):
            raise TypeError("numeric_version must be one signed integer scalar.")
        iterations = jnp.asarray(result.iterations)
        converged = jnp.asarray(result.backend_converged)
        if (
            iterations.shape != result.batch_shape
            or not jnp.issubdtype(iterations.dtype, jnp.signedinteger)
            or converged.shape != result.batch_shape
            or converged.dtype != jnp.bool_
        ):
            raise TypeError(
                "Result iterations/convergence must match the batch with integer/Boolean dtypes."
            )
        checked_iterations = eqx.error_if(
            iterations,
            jnp.any((iterations < 0) | (iterations > result.max_iterations)),
            "Result iterations must lie within the declared solver limit.",
        )
        self.result = eqx.tree_at(
            lambda value: value.iterations,
            result,
            checked_iterations,
        )
        self.numeric_version = eqx.error_if(
            raw_version.astype(jnp.int32),
            raw_version < 0,
            "numeric_version must be non-negative.",
        )
        self.plan_id = identifier
        self.numeric_binding_id = binding


def plan_convex_program(
    program: CanonicalProgram,
    policy: ConvexSolvePolicy | None = None,
    /,
) -> ConvexProgramPlan:
    """Validate method capabilities and create an immutable program plan."""

    selected = ConvexSolvePolicy() if policy is None else policy
    return ConvexProgramPlan(program, selected)


def prepare_convex_template(
    program: CanonicalProgram,
    policy: ConvexSolvePolicy | ConvexProgramPlan | None = None,
    /,
) -> ConvexProgramTemplate:
    """Prepare coefficient-independent structure for repeated numeric binding."""

    plan = (
        policy
        if isinstance(policy, ConvexProgramPlan)
        else plan_convex_program(program, policy)
    )
    if plan.problem_signature != _program_signature(program):
        raise ValueError("Supplied plan does not match the program structure.")
    symbolic_state = None
    if isinstance(plan.policy.method, (MPAXraPDHG, MPAXr2HPDHG)):
        from ._mpax import prepare_mpax_policy

        symbolic_state = prepare_mpax_policy(plan.policy)
    elif isinstance(plan.policy.method, ClarabelInteriorPoint):
        from ._clarabel import prepare_clarabel_policy

        symbolic_state = prepare_clarabel_policy(
            _conic_program(program),
            plan.policy,
        )
    return ConvexProgramTemplate(plan, symbolic_state)


def bind_convex_numeric(
    template: ConvexProgramTemplate,
    program: CanonicalProgram,
    /,
    *,
    numeric_version: Any = 0,
) -> PreparedConvexProgram:
    """Bind one numeric program to a reusable symbolic template."""

    return PreparedConvexProgram(
        program,
        template,
        numeric_version=numeric_version,
    )


def prepare_convex_program(
    program: CanonicalProgram,
    policy: ConvexSolvePolicy | ConvexProgramPlan | None = None,
    /,
) -> PreparedConvexProgram:
    """Compose planning, symbolic preparation, and numeric binding."""

    template = prepare_convex_template(program, policy)
    return bind_convex_numeric(template, program)


def refresh_convex_program(
    prepared: PreparedConvexProgram,
    program: CanonicalProgram,
    /,
) -> PreparedConvexProgram:
    """Refresh coefficients while preserving exact symbolic topology."""

    if not isinstance(prepared, PreparedConvexProgram):
        raise TypeError("prepared must be a PreparedConvexProgram.")
    if _program_signature(program) != prepared.plan.problem_signature:
        raise ValueError("Numeric refresh must preserve the convex-program structure.")
    return bind_convex_numeric(
        prepared.template,
        program,
        numeric_version=prepared.numeric_version + 1,
    )


def _quadratic_program(program: CanonicalProgram, /) -> QuadraticProgram:
    if isinstance(program, QuadraticProgram):
        return program
    if isinstance(program, LinearProgram):
        return program.as_quadratic_program()
    raise TypeError("General conic programs require a conic-capable backend.")


def _conic_program(program: CanonicalProgram, /) -> ConicProgram:
    if isinstance(program, LinearProgram):
        return program.canonical
    if isinstance(program, ConicProgram):
        return program
    return ConicProgram(
        program.quadratic,
        program.linear,
        jnp.concatenate(
            (
                program.equality_matrix[..., : program.num_user_equalities, :],
                program.inequality_matrix[..., : program.num_user_inequalities, :],
            ),
            axis=-2,
        ),
        jnp.concatenate(
            (
                program.equality_rhs[..., : program.num_user_equalities],
                program.inequality_rhs[..., : program.num_user_inequalities],
            ),
            axis=-1,
        ),
        ProductCone(
            (
                ZeroCone(program.num_user_equalities),
                NonnegativeCone(program.num_user_inequalities),
            )
        ),
        bounds=Bounds(program.lower_bounds, program.upper_bounds),
        problem_id=program.problem_id,
        convexity_evidence=program.convexity_evidence,
    )


def _restore_linear_constraint_fields(
    result: ConvexProgramResult,
    program: LinearProgram | QuadraticProgram,
    /,
) -> ConvexProgramResult:
    if isinstance(program, LinearProgram):
        equalities = program.num_equalities
        inequalities = program.num_inequalities
        equality_matrix = program.equality_matrix
        equality_rhs = program.equality_rhs
        inequality_matrix = program.inequality_matrix
        inequality_rhs = program.inequality_rhs
    else:
        equalities = program.num_user_equalities
        inequalities = program.num_user_inequalities
        equality_matrix = program.equality_matrix[..., :equalities, :]
        equality_rhs = program.equality_rhs[..., :equalities]
        inequality_matrix = program.inequality_matrix[..., :inequalities, :]
        inequality_rhs = program.inequality_rhs[..., :inequalities]
    equality_dual = result.cone_dual[..., :equalities]
    inequality_dual = result.cone_dual[..., equalities : equalities + inequalities]
    inequality_slack = result.cone_slack[..., equalities : equalities + inequalities]
    equality_residual = (
        oe.contract("...ij,...j->...i", equality_matrix, result.primal) - equality_rhs
    )
    inequality_action = oe.contract(
        "...ij,...j->...i",
        inequality_matrix,
        result.primal,
    )
    inequality_residual = inequality_action + inequality_slack - inequality_rhs
    inequality_violation = jnp.maximum(inequality_action - inequality_rhs, 0.0)
    complementarity = inequality_slack * inequality_dual
    return eqx.tree_at(
        lambda candidate: (
            candidate.equality_dual,
            candidate.inequality_dual,
            candidate.inequality_slack,
            candidate.equality_residual,
            candidate.inequality_residual,
            candidate.inequality_violation,
            candidate.complementarity_residual,
        ),
        result,
        (
            equality_dual,
            inequality_dual,
            inequality_slack,
            equality_residual,
            inequality_residual,
            inequality_violation,
            complementarity,
        ),
    )


def solve_prepared_convex_program(
    prepared: PreparedConvexProgram,
    /,
    *,
    warm_start: ConvexWarmStart | None = None,
) -> ConvexProgramExecution:
    """Execute one prepared LP/QP with explicit policy and numeric provenance."""

    if not isinstance(prepared, PreparedConvexProgram):
        raise TypeError("prepared must be a PreparedConvexProgram.")
    policy = prepared.plan.policy
    method = policy.method
    if warm_start is not None:
        if not isinstance(warm_start, ConvexWarmStart):
            raise TypeError("warm_start must be a ConvexWarmStart or None.")
        if not method.capabilities.warm_start:
            raise ValueError(f"Method {method.method_id!r} does not support warm starts.")
        if warm_start.structure_id != prepared.plan.problem_signature:
            raise ValueError("Warm start does not match the prepared program structure.")
    program = prepared.program
    if isinstance(method, DensePrimalDualQP):
        problem = _quadratic_program(program)
        lowered_warm = warm_start
        if (
            warm_start is not None
            and isinstance(program, LinearProgram)
            and warm_start.structure_id == program.structure_id
        ):
            lowered_warm = ConvexWarmStart(
                primal=warm_start.primal,
                equality_dual=warm_start.equality_dual,
                inequality_dual=warm_start.inequality_dual,
                inequality_slack=warm_start.inequality_slack,
                lower_bound_dual=warm_start.lower_bound_dual,
                upper_bound_dual=warm_start.upper_bound_dual,
                structure_id=problem.structure_id,
            )
        result = solve_quadratic_program(
            problem,
            policy=policy,
            warm_start=lowered_warm,
        )
    elif isinstance(method, QPaxInteriorPoint):
        if warm_start is not None:
            raise ValueError("QPaxInteriorPoint does not accept warm starts.")
        result = solve_quadratic_program(
            _quadratic_program(program),
            policy=policy,
        )
    elif isinstance(method, ClarabelInteriorPoint):
        if warm_start is not None:
            raise ValueError("ClarabelInteriorPoint does not accept warm starts.")
        from ._clarabel import solve_clarabel_program

        result = solve_clarabel_program(
            _conic_program(program),
            policy,
            prepared_backend=prepared.template.symbolic_state,
        )
        if isinstance(program, (LinearProgram, QuadraticProgram)):
            result = _restore_linear_constraint_fields(result, program)
    elif isinstance(method, (MPAXraPDHG, MPAXr2HPDHG)):
        from ._mpax import solve_mpax_program

        if not isinstance(program, (LinearProgram, QuadraticProgram)):
            raise TypeError("MPAX methods require a LinearProgram or QuadraticProgram.")
        result = solve_mpax_program(
            program,
            policy,
            warm_start=warm_start,
            prepared_backend=prepared.template.symbolic_state,
        )
    else:
        raise TypeError(f"Unsupported convex-program method {type(method).__name__!r}.")
    result = _apply_failure_policy(result, policy)
    backend_provenance = result.provenance
    public_provenance = ConvexProgramProvenance(
        numeric_version=prepared.numeric_version,
        problem_id=program.problem_id,
        structure_id=program.structure_id,
        policy_id=policy.policy_id,
        method_id=method.method_id,
        backend=backend_provenance.backend,
        backend_version=backend_provenance.backend_version,
        convexity_evidence=backend_provenance.convexity_evidence,
        regularization=policy.regularization,
        numeric_binding_id=prepared.numeric_binding_id,
    )
    result = eqx.tree_at(
        lambda candidate: candidate.provenance,
        result,
        public_provenance,
    )
    return ConvexProgramExecution(
        result,
        numeric_version=prepared.numeric_version,
        plan_id=prepared.plan.plan_id,
        numeric_binding_id=prepared.numeric_binding_id,
    )


def solve_convex_program(
    program_or_prepared: CanonicalProgram | PreparedConvexProgram,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
    warm_start: ConvexWarmStart | None = None,
) -> ConvexProgramExecution:
    """Solve one canonical program or execute an already prepared program."""

    if isinstance(program_or_prepared, PreparedConvexProgram):
        if policy is not None:
            raise ValueError("policy must be omitted when solving prepared state.")
        prepared = program_or_prepared
    elif isinstance(program_or_prepared, (LinearProgram, QuadraticProgram, ConicProgram)):
        prepared = prepare_convex_program(program_or_prepared, policy)
    else:
        raise TypeError("Expected a canonical program or PreparedConvexProgram.")
    return solve_prepared_convex_program(prepared, warm_start=warm_start)


def solve_conic_program(
    problem: ConicProgram,
    /,
    *,
    policy: ConvexSolvePolicy,
) -> ConvexProgramResult:
    """Solve one canonical conic program through an explicitly selected method."""

    if not isinstance(problem, ConicProgram):
        raise TypeError("problem must be a ConicProgram.")
    if not isinstance(policy, ConvexSolvePolicy):
        raise TypeError("policy must be a ConvexSolvePolicy.")
    return solve_convex_program(problem, policy=policy).result


def solve_linear_program(
    problem: LinearProgram,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
) -> ConvexProgramResult:
    """Solve one LP without storing a quadratic objective in its public representation."""

    if not isinstance(problem, LinearProgram):
        raise TypeError("problem must be a LinearProgram.")
    return solve_convex_program(problem, policy=policy).result


__all__ = [
    "CanonicalProgram",
    "ConvexProgramExecution",
    "ConvexProgramPlan",
    "ConvexProgramTemplate",
    "ConvexWarmStart",
    "PreparedConvexProgram",
    "bind_convex_numeric",
    "plan_convex_program",
    "prepare_convex_program",
    "prepare_convex_template",
    "refresh_convex_program",
    "solve_convex_program",
    "solve_conic_program",
    "solve_prepared_convex_program",
    "solve_linear_program",
]
