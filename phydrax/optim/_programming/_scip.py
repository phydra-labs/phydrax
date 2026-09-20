#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ...backends.scip import prepare_scip, PreparedSCIP, SCIPPlan
from ._mixed_integer import (
    MixedIntegerCandidate,
    MixedIntegerCertificate,
    MixedIntegerProgram,
    MixedIntegerProvenance,
    MixedIntegerResult,
    MixedIntegerStatus,
    MixedIntegerWork,
)
from ._mixed_integer_audit import audit_mixed_integer_candidate
from ._mixed_integer_policy import (
    AbstractMixedIntegerMethod,
    MixedIntegerMethodCapabilities,
)
from ._problem import LinearProgram


class SCIPMixedInteger(AbstractMixedIntegerMethod):
    """Optional PySCIPOpt MILP execution with provider-qualified global bounds."""

    plan: SCIPPlan

    def __init__(self, plan: SCIPPlan | None = None, /):
        selected = SCIPPlan() if plan is None else plan
        if not isinstance(selected, SCIPPlan):
            raise TypeError("plan must be a SCIPPlan or None.")
        self.plan = selected

    @property
    def method_id(self) -> str:
        return "scip-mixed-integer-linear"

    @property
    def backend(self) -> str:
        return "scip"

    @property
    def capabilities(self) -> MixedIntegerMethodCapabilities:
        return MixedIntegerMethodCapabilities(
            linear_program=True,
            quadratic_program=False,
            conic_program=False,
            warm_start=False,
            incumbent_start=True,
            partial_start=False,
            solution_pool=False,
            global_cuts=False,
            lazy_constraints=False,
            incremental_rows=False,
            prepared_refresh=False,
            infeasibility_certificates=False,
            independent_global_bound=False,
            exact_arithmetic=False,
            deterministic=self.plan.threads == 1,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (("scip_plan", self.plan.plan_id),)


@dataclass(slots=True)
class _SCIPMixedIntegerState:
    prepared: PreparedSCIP


def prepare_scip_mixed_integer(
    program: MixedIntegerProgram,
    method: SCIPMixedInteger,
    /,
) -> _SCIPMixedIntegerState:
    if not isinstance(program.relaxation, LinearProgram):
        raise TypeError("SCIPMixedInteger initially supports LinearProgram only.")
    return _SCIPMixedIntegerState(prepare_scip(method.plan))


def _configure(model, plan: SCIPPlan, /) -> None:
    model.setParam("limits/nodes", plan.maximum_nodes)
    if plan.time_limit is not None:
        model.setParam("limits/time", plan.time_limit)
    model.setParam("limits/absgap", plan.absolute_gap)
    model.setParam("limits/gap", plan.relative_gap)
    model.setParam("parallel/maxnthreads", plan.threads)
    model.setParam("randomization/randomseedshift", plan.random_seed)
    model.setParam("display/verblevel", 4 if plan.verbose else 0)
    if not plan.presolve:
        model.setParam("presolving/maxrounds", 0)


def _provider_status(status: str, /) -> MixedIntegerStatus:
    normalized = str(status).strip().lower()
    if normalized == "optimal":
        return MixedIntegerStatus.OPTIMAL
    if normalized == "infeasible":
        return MixedIntegerStatus.INFEASIBLE
    if normalized in ("unbounded", "inforunbd"):
        return MixedIntegerStatus.UNBOUNDED
    if normalized == "gaplimit":
        return MixedIntegerStatus.GAP_REACHED
    if normalized in (
        "timelimit",
        "nodelimit",
        "stallnodelimit",
        "totalnodelimit",
        "solutionlimit",
        "bestsollimit",
        "memlimit",
        "restartlimit",
    ):
        return MixedIntegerStatus.WORK_LIMIT
    return MixedIntegerStatus.BACKEND_FAILURE


def solve_scip_mixed_integer(prepared, candidates, /) -> MixedIntegerResult:
    from ._mixed_integer_lifecycle import PreparedMixedIntegerProgram

    if not isinstance(prepared, PreparedMixedIntegerProgram):
        raise TypeError("prepared must be a PreparedMixedIntegerProgram.")
    method = prepared.plan.policy.method
    if not isinstance(method, SCIPMixedInteger):
        raise TypeError("Prepared policy is not SCIPMixedInteger.")
    state = prepared.template.method_state
    if not isinstance(state, _SCIPMixedIntegerState):
        raise TypeError("Prepared SCIP state is invalid.")
    program = prepared.program
    relaxation = program.relaxation
    if not isinstance(relaxation, LinearProgram):
        raise TypeError("SCIPMixedInteger initially supports LinearProgram only.")

    module = state.prepared.module
    model = module.Model(program.program_id)
    _configure(model, method.plan)
    infinity = float(model.infinity())
    integer = set(program.integer_indices)
    binary = set(program.binary_indices)
    lower = np.asarray(relaxation.lower_bounds)
    upper = np.asarray(relaxation.upper_bounds)
    linear = np.asarray(relaxation.linear)
    variables = []
    for index in range(relaxation.num_variables):
        variable_type = "B" if index in binary else "I" if index in integer else "C"
        lower_value = float(lower[index]) if np.isfinite(lower[index]) else -infinity
        upper_value = float(upper[index]) if np.isfinite(upper[index]) else infinity
        variables.append(
            model.addVar(
                name=f"x{index}",
                vtype=variable_type,
                lb=lower_value,
                ub=upper_value,
                obj=float(linear[index]),
            )
        )
    model.setMinimize()
    equality = np.asarray(relaxation.equality_matrix)
    equality_rhs = np.asarray(relaxation.equality_rhs)
    inequality = np.asarray(relaxation.inequality_matrix)
    inequality_rhs = np.asarray(relaxation.inequality_rhs)
    for row in range(relaxation.num_equalities):
        expression = module.quicksum(
            float(equality[row, column]) * variables[column]
            for column in np.flatnonzero(equality[row])
        )
        model.addCons(expression == float(equality_rhs[row]), name=f"eq{row}")
    for row in range(relaxation.num_inequalities):
        expression = module.quicksum(
            float(inequality[row, column]) * variables[column]
            for column in np.flatnonzero(inequality[row])
        )
        model.addCons(expression <= float(inequality_rhs[row]), name=f"ineq{row}")

    audited_count = 0
    accepted_count = 0
    for candidate in candidates:
        audit = audit_mixed_integer_candidate(
            program,
            candidate,
            prepared.plan.policy.certification,
        )
        audited_count += 1
        if not bool(np.asarray(audit.valid)):
            continue
        accepted_count += 1
        solution = model.createSol()
        for variable, value in zip(variables, np.asarray(audit.primal), strict=True):
            model.setSolVal(solution, variable, float(value))
        model.addSol(solution, free=True)

    model.optimize()
    status = _provider_status(str(model.getStatus()))
    solution = model.getBestSol()
    audit = None
    if solution is not None:
        primal = jnp.asarray(
            [model.getSolVal(solution, variable) for variable in variables],
            dtype=relaxation.linear.dtype,
        )
        provider_objective = model.getSolObjVal(solution)
        candidate = MixedIntegerCandidate(
            primal,
            reported_objective=provider_objective,
            source_kind="scip-solution",
            source_id=prepared.binding_id,
        )
        audit = audit_mixed_integer_candidate(
            program,
            candidate,
            prepared.plan.policy.certification,
        )
        audited_count += 1
        accepted_count += int(bool(np.asarray(audit.valid)))
        if not bool(np.asarray(audit.valid)):
            audit = None
            if status == MixedIntegerStatus.OPTIMAL:
                status = MixedIntegerStatus.CERTIFICATION_FAILURE
    lower_bound = float(model.getDualbound())
    explored_nodes = int(model.getNNodes())
    frontier_size = 0
    model.freeProb()

    objective = float("inf") if audit is None else float(np.asarray(audit.objective))
    if np.isfinite(objective) and np.isfinite(lower_bound):
        lower_bound = min(lower_bound, objective)
        absolute_gap = objective - lower_bound
        relative_gap = absolute_gap / max(abs(objective), 1.0)
    else:
        absolute_gap = float("inf")
        relative_gap = float("inf")
    provider_complete = status in (
        MixedIntegerStatus.OPTIMAL,
        MixedIntegerStatus.INFEASIBLE,
        MixedIntegerStatus.UNBOUNDED,
    )
    certificate = MixedIntegerCertificate(
        candidate_audit=audit,
        infeasibility_certified=jnp.asarray(False),
        global_bound_certified=jnp.asarray(False),
        search_complete=jnp.asarray(provider_complete),
        optimality_certified=jnp.asarray(False),
        provider_solved=jnp.asarray(status == MixedIntegerStatus.OPTIMAL),
        proof_kind="provider-reported",
    )
    work = MixedIntegerWork(
        jnp.asarray(explored_nodes, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(frontier_size, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(audited_count, dtype=jnp.int32),
        jnp.asarray(accepted_count, dtype=jnp.int32),
    )
    provenance = MixedIntegerProvenance(
        program.program_id,
        program.structure_id,
        prepared.plan.policy.policy_id,
        method.method_id,
        method.backend,
        prepared.binding_id,
        prepared.numeric_version,
    )
    primal = (
        jnp.full(
            (relaxation.num_variables,),
            jnp.nan,
            dtype=relaxation.linear.dtype,
        )
        if audit is None
        else audit.primal
    )
    return MixedIntegerResult(
        primal,
        jnp.asarray(objective),
        jnp.asarray(lower_bound),
        jnp.asarray(absolute_gap),
        jnp.asarray(relative_gap),
        jnp.asarray(int(status), dtype=jnp.int32),
        certificate,
        work,
        provenance,
        None,
        None,
    )


__all__ = [
    "SCIPMixedInteger",
    "prepare_scip_mixed_integer",
    "solve_scip_mixed_integer",
]
