#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._mixed_integer import (
    MixedIntegerCandidate,
    MixedIntegerProgram,
    MixedIntegerResult,
)
from ._mixed_integer_policy import (
    MixedIntegerSolvePolicy,
    NativeMixedIntegerBranchAndBound,
)
from ._problem import ConicProgram, LinearProgram
from ._quadratic import QuadraticProgram


def _problem_signature(program: MixedIntegerProgram, /) -> tuple[str, str]:
    return type(program.relaxation).__name__, program.structure_id


def _binding_id(
    program: MixedIntegerProgram,
    policy: MixedIntegerSolvePolicy,
    numeric_version: int,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "mixed-integer-numeric-binding",
            "structure": program.structure_id,
            "policy": policy.policy_id,
            "numeric_version": int(numeric_version),
            "arrays": array_tree_fingerprint(program.relaxation),
        }
    )


def _validate_method(
    program: MixedIntegerProgram, policy: MixedIntegerSolvePolicy, /
) -> None:
    capabilities = policy.method.capabilities
    relaxation = program.relaxation
    supported = (
        capabilities.linear_program
        if isinstance(relaxation, LinearProgram)
        else capabilities.quadratic_program
        if isinstance(relaxation, QuadraticProgram)
        else capabilities.conic_program
        if isinstance(relaxation, ConicProgram)
        else False
    )
    if not supported:
        raise ValueError(
            f"Method {policy.method.method_id!r} does not support "
            f"{type(relaxation).__name__}."
        )


class MixedIntegerProgramPlan(StrictModule, NonTrainableState):
    """Validated immutable mixed-integer problem/method contract."""

    program: MixedIntegerProgram
    policy: MixedIntegerSolvePolicy
    problem_signature: tuple[str, str] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        program: MixedIntegerProgram,
        policy: MixedIntegerSolvePolicy | None = None,
        /,
    ):
        if not isinstance(program, MixedIntegerProgram):
            raise TypeError("program must be a MixedIntegerProgram.")
        policy_ = MixedIntegerSolvePolicy() if policy is None else policy
        if not isinstance(policy_, MixedIntegerSolvePolicy):
            raise TypeError("policy must be a MixedIntegerSolvePolicy.")
        _validate_method(program, policy_)
        signature = _problem_signature(program)
        self.program = program
        self.policy = policy_
        self.problem_signature = signature
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-program-plan",
                "problem": list(signature),
                "policy": policy_.policy_id,
            }
        )


class MixedIntegerProgramTemplate(StrictModule, NonTrainableState):
    """Reusable method-specific symbolic state."""

    plan: MixedIntegerProgramPlan
    method_state: Any
    template_id: str = eqx.field(static=True)


class PreparedMixedIntegerProgram(StrictModule, NonTrainableState):
    """One numeric binding of a reusable mixed-integer template."""

    template: MixedIntegerProgramTemplate
    program: MixedIntegerProgram
    numeric_version: int = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    @property
    def plan(self) -> MixedIntegerProgramPlan:
        return self.template.plan


class MixedIntegerProgramExecution(StrictModule, NonTrainableState):
    prepared: PreparedMixedIntegerProgram
    result: MixedIntegerResult


def plan_mixed_integer_program(
    program: MixedIntegerProgram,
    policy: MixedIntegerSolvePolicy | None = None,
    /,
) -> MixedIntegerProgramPlan:
    return MixedIntegerProgramPlan(program, policy)


def prepare_mixed_integer_template(
    program: MixedIntegerProgram,
    policy: MixedIntegerSolvePolicy | MixedIntegerProgramPlan | None = None,
    /,
) -> MixedIntegerProgramTemplate:
    plan = (
        policy
        if isinstance(policy, MixedIntegerProgramPlan)
        else MixedIntegerProgramPlan(program, policy)
    )
    if _problem_signature(program) != plan.problem_signature:
        raise ValueError("Program does not match the supplied mixed-integer plan.")
    if isinstance(plan.policy.method, NativeMixedIntegerBranchAndBound):
        from ._mixed_integer_native import prepare_native_mixed_integer

        state = prepare_native_mixed_integer(program, plan.policy.method)
    else:
        from ._mixed_integer_oa import (
            ConicOuterApproximation,
            prepare_conic_outer_approximation,
        )

        if isinstance(plan.policy.method, ConicOuterApproximation):
            state = prepare_conic_outer_approximation(program, plan.policy.method)
        else:
            from ._scip import prepare_scip_mixed_integer, SCIPMixedInteger

            if not isinstance(plan.policy.method, SCIPMixedInteger):
                raise TypeError("Unsupported mixed-integer method.")
            state = prepare_scip_mixed_integer(program, plan.policy.method)
    return MixedIntegerProgramTemplate(
        plan,
        state,
        canonical_fingerprint(
            {
                "kind": "mixed-integer-template",
                "plan": plan.plan_id,
                "method": plan.policy.method.method_id,
            }
        ),
    )


def bind_mixed_integer_numeric(
    template: MixedIntegerProgramTemplate,
    program: MixedIntegerProgram,
    /,
    *,
    numeric_version: int = 0,
) -> PreparedMixedIntegerProgram:
    if not isinstance(template, MixedIntegerProgramTemplate):
        raise TypeError("template must be a MixedIntegerProgramTemplate.")
    if not isinstance(program, MixedIntegerProgram):
        raise TypeError("program must be a MixedIntegerProgram.")
    if _problem_signature(program) != template.plan.problem_signature:
        raise ValueError("Numeric binding must preserve mixed-integer structure.")
    version = int(numeric_version)
    if version < 0:
        raise ValueError("numeric_version must be nonnegative.")
    return PreparedMixedIntegerProgram(
        template,
        program,
        version,
        _binding_id(program, template.plan.policy, version),
    )


def prepare_mixed_integer_program(
    program: MixedIntegerProgram,
    policy: MixedIntegerSolvePolicy | MixedIntegerProgramPlan | None = None,
    /,
) -> PreparedMixedIntegerProgram:
    template = prepare_mixed_integer_template(program, policy)
    return bind_mixed_integer_numeric(template, program)


def refresh_mixed_integer_program(
    prepared: PreparedMixedIntegerProgram,
    program: MixedIntegerProgram,
    /,
) -> PreparedMixedIntegerProgram:
    if not isinstance(prepared, PreparedMixedIntegerProgram):
        raise TypeError("prepared must be a PreparedMixedIntegerProgram.")
    if not prepared.plan.policy.method.capabilities.prepared_refresh:
        raise ValueError("The selected mixed-integer method does not support refresh.")
    return bind_mixed_integer_numeric(
        prepared.template,
        program,
        numeric_version=prepared.numeric_version + 1,
    )


def solve_prepared_mixed_integer_program(
    prepared: PreparedMixedIntegerProgram,
    /,
    *,
    candidates: tuple[MixedIntegerCandidate, ...] = (),
) -> MixedIntegerProgramExecution:
    if not isinstance(prepared, PreparedMixedIntegerProgram):
        raise TypeError("prepared must be a PreparedMixedIntegerProgram.")
    if not isinstance(candidates, tuple) or not all(
        isinstance(candidate, MixedIntegerCandidate) for candidate in candidates
    ):
        raise TypeError("candidates must be a tuple of MixedIntegerCandidate values.")
    method = prepared.plan.policy.method
    if isinstance(method, NativeMixedIntegerBranchAndBound):
        from ._mixed_integer_native import solve_native_mixed_integer

        result = solve_native_mixed_integer(prepared, candidates)
    else:
        from ._mixed_integer_oa import (
            ConicOuterApproximation,
            solve_conic_outer_approximation,
        )

        if isinstance(method, ConicOuterApproximation):
            result = solve_conic_outer_approximation(prepared, candidates)
        else:
            from ._scip import SCIPMixedInteger, solve_scip_mixed_integer

            if not isinstance(method, SCIPMixedInteger):
                raise TypeError("Unsupported mixed-integer method.")
            result = solve_scip_mixed_integer(prepared, candidates)
    return MixedIntegerProgramExecution(prepared, result)


__all__ = [
    "MixedIntegerProgramExecution",
    "MixedIntegerProgramPlan",
    "MixedIntegerProgramTemplate",
    "PreparedMixedIntegerProgram",
    "bind_mixed_integer_numeric",
    "plan_mixed_integer_program",
    "prepare_mixed_integer_program",
    "prepare_mixed_integer_template",
    "refresh_mixed_integer_program",
    "solve_prepared_mixed_integer_program",
]
