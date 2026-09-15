#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._bounds import Bounds
from ._conic_cuts import (
    audit_conic_cut,
    conic_cut_from_dual,
    ConicCut,
    polyhedral_conic_rows,
    separate_conic_point,
)
from ._lifecycle import (
    bind_convex_numeric,
    ConvexProgramTemplate,
    prepare_convex_template,
    solve_prepared_convex_program,
)
from ._mixed_integer import (
    MixedIntegerCandidate,
    MixedIntegerCandidateAudit,
    MixedIntegerCertificate,
    MixedIntegerProgram,
    MixedIntegerProvenance,
    MixedIntegerResult,
    MixedIntegerStatus,
    MixedIntegerWork,
)
from ._mixed_integer_audit import audit_mixed_integer_candidate
from ._mixed_integer_native import _replace_bounds
from ._mixed_integer_policy import (
    AbstractMixedIntegerMethod,
    MixedIntegerMethodCapabilities,
    MixedIntegerSolvePolicy,
    NativeMixedIntegerBranchAndBound,
)
from ._policy import ConvexSolvePolicy, NativeHomogeneousConic
from ._problem import ConicProgram, LinearProgram
from ._types import ConvexProgramStatus


class ConicOuterApproximation(AbstractMixedIntegerMethod):
    """Iterative mixed-integer outer approximation from audited cone cuts."""

    master: AbstractMixedIntegerMethod
    conic: ConvexSolvePolicy
    maximum_rounds: int = eqx.field(static=True)
    maximum_cuts: int = eqx.field(static=True)
    separation_tolerance: float = eqx.field(static=True)
    duplicate_tolerance: float = eqx.field(static=True)
    absolute_gap: float = eqx.field(static=True)
    relative_gap: float = eqx.field(static=True)

    def __init__(
        self,
        master: AbstractMixedIntegerMethod | None = None,
        /,
        *,
        conic: ConvexSolvePolicy | None = None,
        maximum_rounds: int = 100,
        maximum_cuts: int = 10_000,
        separation_tolerance: float = 1e-7,
        duplicate_tolerance: float = 1e-9,
        absolute_gap: float = 1e-7,
        relative_gap: float = 1e-7,
    ):
        master_ = NativeMixedIntegerBranchAndBound() if master is None else master
        conic_ = ConvexSolvePolicy(NativeHomogeneousConic()) if conic is None else conic
        if not isinstance(master_, AbstractMixedIntegerMethod):
            raise TypeError("master must be an AbstractMixedIntegerMethod.")
        if isinstance(master_, ConicOuterApproximation):
            raise TypeError("Conic outer approximation cannot recursively be its master.")
        if not master_.capabilities.linear_program:
            raise ValueError("The outer-approximation master must support MILP.")
        if not isinstance(conic_, ConvexSolvePolicy):
            raise TypeError("conic must be a ConvexSolvePolicy.")
        if not conic_.method.capabilities.conic_program:
            raise ValueError("The selected conic policy does not support ConicProgram.")
        if conic_.failure.mode != "status":
            raise ValueError("Outer-approximation conic solves require status mode.")
        rounds = int(maximum_rounds)
        cuts = int(maximum_cuts)
        values = tuple(
            float(value)
            for value in (
                separation_tolerance,
                duplicate_tolerance,
                absolute_gap,
                relative_gap,
            )
        )
        if rounds < 1 or cuts < 1:
            raise ValueError("maximum_rounds and maximum_cuts must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Outer-approximation tolerances must be finite and nonnegative."
            )
        if values[2] == 0.0 and values[3] == 0.0:
            raise ValueError("At least one outer optimality tolerance must be positive.")
        self.master = master_
        self.conic = conic_
        self.maximum_rounds = rounds
        self.maximum_cuts = cuts
        (
            self.separation_tolerance,
            self.duplicate_tolerance,
            self.absolute_gap,
            self.relative_gap,
        ) = values

    @property
    def method_id(self) -> str:
        return "native-conic-outer-approximation-v1"

    @property
    def backend(self) -> str:
        return "phydrax-conic-outer-approximation"

    @property
    def capabilities(self) -> MixedIntegerMethodCapabilities:
        return MixedIntegerMethodCapabilities(
            linear_program=False,
            quadratic_program=False,
            conic_program=True,
            warm_start=False,
            incumbent_start=True,
            partial_start=False,
            solution_pool=False,
            global_cuts=True,
            lazy_constraints=False,
            incremental_rows=False,
            prepared_refresh=self.conic.method.capabilities.prepared_refresh,
            infeasibility_certificates=True,
            independent_global_bound=self.master.capabilities.independent_global_bound,
            exact_arithmetic=False,
            deterministic=self.master.capabilities.deterministic,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("master", self.master.method_id),
            ("conic_policy", self.conic.policy_id),
            ("maximum_rounds", str(self.maximum_rounds)),
            ("maximum_cuts", str(self.maximum_cuts)),
            ("separation_tolerance", repr(self.separation_tolerance)),
            ("duplicate_tolerance", repr(self.duplicate_tolerance)),
            ("absolute_gap", repr(self.absolute_gap)),
            ("relative_gap", repr(self.relative_gap)),
        )


@dataclass(slots=True)
class _ConicOuterApproximationState:
    continuous_template: ConvexProgramTemplate
    fixed_template: ConvexProgramTemplate
    retained_duals: list[np.ndarray]


def prepare_conic_outer_approximation(
    program: MixedIntegerProgram,
    method: ConicOuterApproximation,
    /,
) -> _ConicOuterApproximationState:
    relaxation = program.relaxation
    if not isinstance(relaxation, ConicProgram):
        raise TypeError("ConicOuterApproximation requires a ConicProgram relaxation.")
    if relaxation.quadratic is not None:
        raise ValueError(
            "Initial conic outer approximation supports linear objectives only."
        )
    if not np.all(np.isfinite(np.asarray(relaxation.lower_bounds))) or not np.all(
        np.isfinite(np.asarray(relaxation.upper_bounds))
    ):
        raise ValueError("Conic outer approximation currently requires finite bounds.")
    lower = np.asarray(relaxation.lower_bounds).copy()
    upper = np.asarray(relaxation.upper_bounds).copy()
    discrete = np.asarray(program.discrete_indices, dtype=np.int64)
    upper[discrete] = lower[discrete]
    fixed = _replace_bounds(relaxation, lower, upper)
    return _ConicOuterApproximationState(
        prepare_convex_template(relaxation, method.conic),
        prepare_convex_template(fixed, method.conic),
        [],
    )


def _master_program(
    program: MixedIntegerProgram,
    cuts: tuple[ConicCut, ...],
    /,
) -> MixedIntegerProgram:
    conic = program.relaxation
    if not isinstance(conic, ConicProgram):
        raise TypeError("Outer master requires a ConicProgram.")
    equality, equality_rhs, inequality, inequality_rhs = polyhedral_conic_rows(conic)
    if cuts:
        cut_matrix = jnp.stack(tuple(cut.row for cut in cuts))
        cut_rhs = jnp.stack(tuple(cut.rhs for cut in cuts))
        inequality = jnp.concatenate((inequality, cut_matrix), axis=0)
        inequality_rhs = jnp.concatenate((inequality_rhs, cut_rhs), axis=0)
    relaxation = LinearProgram(
        conic.linear,
        equality_matrix=equality,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality,
        inequality_rhs=inequality_rhs,
        bounds=Bounds(conic.lower_bounds, conic.upper_bounds),
        problem_id=f"{program.program_id}:outer-master",
    )
    return MixedIntegerProgram(
        relaxation,
        integer_indices=program.integer_indices,
        binary_indices=program.binary_indices,
        program_id=f"{program.program_id}:outer-master",
    )


def _fixed_program(
    program: MixedIntegerProgram,
    assignment,
    /,
) -> ConicProgram:
    conic = program.relaxation
    if not isinstance(conic, ConicProgram):
        raise TypeError("Fixed-discrete solve requires a ConicProgram.")
    lower = np.asarray(conic.lower_bounds).copy()
    upper = np.asarray(conic.upper_bounds).copy()
    indices = np.asarray(program.discrete_indices, dtype=np.int64)
    values = np.rint(np.asarray(assignment)[indices])
    lower[indices] = values
    upper[indices] = values
    fixed = _replace_bounds(conic, lower, upper)
    if not isinstance(fixed, ConicProgram):
        raise TypeError("Fixed-discrete conic lowering changed program type.")
    return fixed


def _gap(objective: float, lower_bound: float, /) -> tuple[float, float]:
    if not np.isfinite(objective) or not np.isfinite(lower_bound):
        return float("inf"), float("inf")
    safe_lower = min(lower_bound, objective)
    absolute = objective - safe_lower
    return absolute, absolute / max(abs(objective), 1.0)


def _oa_result(
    prepared,
    method: ConicOuterApproximation,
    /,
    *,
    status: MixedIntegerStatus,
    audit: MixedIntegerCandidateAudit | None,
    lower_bound: float,
    global_bound_certified: bool,
    search_complete: bool,
    incumbent_relaxation,
    last_master,
    explored_nodes: int,
    pruned_nodes: int,
    frontier_size: int,
    relaxation_solves: int,
    master_solves: int,
    fixed_discrete_solves: int,
    cuts_proposed: int,
    cuts_accepted: int,
    candidates_audited: int,
    candidates_accepted: int,
) -> MixedIntegerResult:
    objective = float("inf") if audit is None else float(np.asarray(audit.objective))
    absolute_gap, relative_gap = _gap(objective, lower_bound)
    optimality = (
        status == MixedIntegerStatus.OPTIMAL
        and audit is not None
        and bool(np.asarray(audit.valid))
        and global_bound_certified
        and search_complete
        and (absolute_gap <= method.absolute_gap or relative_gap <= method.relative_gap)
    )
    proof_kind = (
        "native-audited"
        if method.master.capabilities.independent_global_bound
        else "provider-reported"
    )
    certificate = MixedIntegerCertificate(
        candidate_audit=audit,
        infeasibility_certified=jnp.asarray(
            status == MixedIntegerStatus.INFEASIBLE
            and global_bound_certified
            and search_complete
        ),
        global_bound_certified=jnp.asarray(global_bound_certified),
        search_complete=jnp.asarray(search_complete),
        optimality_certified=jnp.asarray(optimality),
        provider_solved=jnp.asarray(
            status == MixedIntegerStatus.OPTIMAL and proof_kind == "provider-reported"
        ),
        proof_kind=proof_kind,
    )
    work = MixedIntegerWork(
        jnp.asarray(explored_nodes, dtype=jnp.int32),
        jnp.asarray(pruned_nodes, dtype=jnp.int32),
        jnp.asarray(frontier_size, dtype=jnp.int32),
        jnp.asarray(True),
        jnp.asarray(relaxation_solves, dtype=jnp.int32),
        jnp.asarray(master_solves, dtype=jnp.int32),
        jnp.asarray(fixed_discrete_solves, dtype=jnp.int32),
        jnp.asarray(cuts_proposed, dtype=jnp.int32),
        jnp.asarray(cuts_accepted, dtype=jnp.int32),
        jnp.asarray(candidates_audited, dtype=jnp.int32),
        jnp.asarray(candidates_accepted, dtype=jnp.int32),
    )
    provenance = MixedIntegerProvenance(
        prepared.program.program_id,
        prepared.program.structure_id,
        prepared.plan.policy.policy_id,
        method.method_id,
        method.backend,
        prepared.binding_id,
        prepared.numeric_version,
    )
    primal = (
        jnp.full(
            (prepared.program.relaxation.num_variables,),
            jnp.nan,
            dtype=prepared.program.relaxation.linear.dtype,
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
        incumbent_relaxation,
        None if last_master is None else last_master.search,
    )


def solve_conic_outer_approximation(prepared, candidates, /) -> MixedIntegerResult:
    from ._mixed_integer_lifecycle import PreparedMixedIntegerProgram

    if not isinstance(prepared, PreparedMixedIntegerProgram):
        raise TypeError("prepared must be a PreparedMixedIntegerProgram.")
    method = prepared.plan.policy.method
    if not isinstance(method, ConicOuterApproximation):
        raise TypeError("Prepared policy is not conic outer approximation.")
    state = prepared.template.method_state
    if not isinstance(state, _ConicOuterApproximationState):
        raise TypeError("Prepared conic outer-approximation state is invalid.")
    program = prepared.program
    conic = program.relaxation
    if not isinstance(conic, ConicProgram):
        raise TypeError("Conic outer approximation requires ConicProgram.")
    certification = prepared.plan.policy.certification

    accepted_cuts: list[ConicCut] = []
    incumbent_audit = None
    incumbent_relaxation = None
    global_lower_bound = -float("inf")
    global_bound_certified = True
    explored_nodes = 0
    pruned_nodes = 0
    frontier_size = 0
    relaxation_solves = 0
    master_solves = 0
    fixed_discrete_solves = 0
    cuts_proposed = 0
    cuts_accepted = 0
    candidates_audited = 0
    candidates_accepted = 0
    last_master = None

    def consider(candidate, relaxation_result=None):
        nonlocal incumbent_audit, incumbent_relaxation
        nonlocal candidates_audited, candidates_accepted
        audit = audit_mixed_integer_candidate(program, candidate, certification)
        candidates_audited += 1
        if not bool(np.asarray(audit.valid)):
            return False
        candidates_accepted += 1
        if incumbent_audit is None or bool(
            np.asarray(audit.objective < incumbent_audit.objective)
        ):
            incumbent_audit = audit
            incumbent_relaxation = relaxation_result
        return True

    def accept_cut(cut, *, require_violation):
        nonlocal cuts_proposed, cuts_accepted
        cuts_proposed += 1
        audit = audit_conic_cut(
            conic,
            cut,
            tolerance=method.separation_tolerance,
        )
        if not bool(np.asarray(audit.global_valid)):
            return False
        if require_violation and not bool(np.asarray(audit.source_violated)):
            return False
        row = np.asarray(cut.row)
        rhs = float(np.asarray(cut.rhs))
        for existing in accepted_cuts:
            if np.allclose(
                row,
                np.asarray(existing.row),
                atol=method.duplicate_tolerance,
                rtol=method.duplicate_tolerance,
            ) and np.isclose(
                rhs,
                float(np.asarray(existing.rhs)),
                atol=method.duplicate_tolerance,
                rtol=method.duplicate_tolerance,
            ):
                return False
        if len(accepted_cuts) >= method.maximum_cuts:
            return False
        accepted_cuts.append(cut)
        dual = np.asarray(cut.dual)
        if not any(
            np.allclose(
                dual,
                existing,
                atol=method.duplicate_tolerance,
                rtol=method.duplicate_tolerance,
            )
            for existing in state.retained_duals
        ):
            state.retained_duals.append(dual.copy())
        cuts_accepted += 1
        return True

    for candidate in candidates:
        consider(candidate)

    continuous = solve_prepared_convex_program(
        bind_convex_numeric(
            state.continuous_template,
            conic,
            numeric_version=prepared.numeric_version,
        )
    ).result
    relaxation_solves += 1
    continuous_status = ConvexProgramStatus(int(np.asarray(continuous.status)))
    if continuous_status == ConvexProgramStatus.PRIMAL_INFEASIBLE and bool(
        np.asarray(continuous.certificate.dual_ray_valid)
    ):
        return _oa_result(
            prepared,
            method,
            status=MixedIntegerStatus.INFEASIBLE,
            audit=None,
            lower_bound=float("inf"),
            global_bound_certified=True,
            search_complete=True,
            incumbent_relaxation=None,
            last_master=None,
            explored_nodes=0,
            pruned_nodes=0,
            frontier_size=0,
            relaxation_solves=relaxation_solves,
            master_solves=0,
            fixed_discrete_solves=0,
            cuts_proposed=0,
            cuts_accepted=0,
            candidates_audited=candidates_audited,
            candidates_accepted=candidates_accepted,
        )
    if not bool(np.asarray(continuous.successful)):
        return _oa_result(
            prepared,
            method,
            status=MixedIntegerStatus.RELAXATION_FAILURE,
            audit=incumbent_audit,
            lower_bound=-float("inf"),
            global_bound_certified=False,
            search_complete=False,
            incumbent_relaxation=incumbent_relaxation,
            last_master=None,
            explored_nodes=0,
            pruned_nodes=0,
            frontier_size=0,
            relaxation_solves=relaxation_solves,
            master_solves=0,
            fixed_discrete_solves=0,
            cuts_proposed=0,
            cuts_accepted=0,
            candidates_audited=candidates_audited,
            candidates_accepted=candidates_accepted,
        )
    global_lower_bound = float(np.asarray(continuous.objective))
    continuous_candidate = MixedIntegerCandidate(
        continuous.primal,
        reported_objective=continuous.objective,
        source_kind="continuous-conic-relaxation",
        source_id=continuous.provenance.structure_id,
    )
    consider(continuous_candidate, continuous)

    for dual in tuple(state.retained_duals):
        cut = conic_cut_from_dual(
            conic,
            jnp.asarray(dual, dtype=conic.linear.dtype),
            source_kind="continuous-dual",
            cone_block=-1,
            cone_id=conic.cone.cone_id,
            binding_id=prepared.binding_id,
        )
        accept_cut(cut, require_violation=False)
    root_cut = conic_cut_from_dual(
        conic,
        continuous.cone_dual,
        source_kind="continuous-dual",
        cone_block=-1,
        cone_id=conic.cone.cone_id,
        binding_id=prepared.binding_id,
    )
    accept_cut(root_cut, require_violation=False)

    for _ in range(method.maximum_rounds):
        master_program = _master_program(program, tuple(accepted_cuts))
        master_policy = MixedIntegerSolvePolicy(
            method.master,
            certification=certification,
        )
        master_candidates = ()
        if incumbent_audit is not None:
            master_candidates = (
                MixedIntegerCandidate(
                    incumbent_audit.primal,
                    reported_objective=incumbent_audit.objective,
                    source_kind="outer-incumbent",
                    source_id=prepared.binding_id,
                ),
            )
        from ._mixed_integer import solve_mixed_integer_program

        last_master = solve_mixed_integer_program(
            master_program,
            master_policy,
            candidates=master_candidates,
        )
        master_solves += 1
        explored_nodes += int(np.asarray(last_master.explored_nodes))
        pruned_nodes += int(np.asarray(last_master.pruned_nodes))
        frontier_size = int(np.asarray(last_master.frontier_size))
        global_bound_certified &= bool(
            np.asarray(last_master.certificate.global_bound_certified)
        )
        if bool(np.asarray(last_master.certificate.global_bound_certified)):
            global_lower_bound = max(
                global_lower_bound,
                float(np.asarray(last_master.global_lower_bound)),
            )
        master_status = MixedIntegerStatus(int(np.asarray(last_master.status)))
        if master_status == MixedIntegerStatus.INFEASIBLE:
            if incumbent_audit is None and bool(
                np.asarray(last_master.certificate.infeasibility_certified)
            ):
                return _oa_result(
                    prepared,
                    method,
                    status=MixedIntegerStatus.INFEASIBLE,
                    audit=None,
                    lower_bound=float("inf"),
                    global_bound_certified=True,
                    search_complete=True,
                    incumbent_relaxation=None,
                    last_master=last_master,
                    explored_nodes=explored_nodes,
                    pruned_nodes=pruned_nodes,
                    frontier_size=frontier_size,
                    relaxation_solves=relaxation_solves,
                    master_solves=master_solves,
                    fixed_discrete_solves=fixed_discrete_solves,
                    cuts_proposed=cuts_proposed,
                    cuts_accepted=cuts_accepted,
                    candidates_audited=candidates_audited,
                    candidates_accepted=candidates_accepted,
                )
            break
        if not bool(np.asarray(last_master.feasible)):
            status = (
                MixedIntegerStatus.WORK_LIMIT
                if master_status == MixedIntegerStatus.WORK_LIMIT
                else MixedIntegerStatus.RELAXATION_FAILURE
            )
            return _oa_result(
                prepared,
                method,
                status=status,
                audit=incumbent_audit,
                lower_bound=global_lower_bound,
                global_bound_certified=global_bound_certified,
                search_complete=False,
                incumbent_relaxation=incumbent_relaxation,
                last_master=last_master,
                explored_nodes=explored_nodes,
                pruned_nodes=pruned_nodes,
                frontier_size=frontier_size,
                relaxation_solves=relaxation_solves,
                master_solves=master_solves,
                fixed_discrete_solves=fixed_discrete_solves,
                cuts_proposed=cuts_proposed,
                cuts_accepted=cuts_accepted,
                candidates_audited=candidates_audited,
                candidates_accepted=candidates_accepted,
            )

        accepted_before = cuts_accepted
        master_candidate = MixedIntegerCandidate(
            last_master.primal,
            reported_objective=last_master.objective,
            source_kind="outer-master",
            source_id=last_master.provenance.binding_id,
        )
        original_feasible = consider(master_candidate)
        separation = separate_conic_point(
            conic,
            last_master.primal,
            tolerance=method.separation_tolerance,
            binding_id=prepared.binding_id,
        )
        if not bool(np.asarray(separation.valid)):
            return _oa_result(
                prepared,
                method,
                status=MixedIntegerStatus.CERTIFICATION_FAILURE,
                audit=incumbent_audit,
                lower_bound=global_lower_bound,
                global_bound_certified=global_bound_certified,
                search_complete=False,
                incumbent_relaxation=incumbent_relaxation,
                last_master=last_master,
                explored_nodes=explored_nodes,
                pruned_nodes=pruned_nodes,
                frontier_size=frontier_size,
                relaxation_solves=relaxation_solves,
                master_solves=master_solves,
                fixed_discrete_solves=fixed_discrete_solves,
                cuts_proposed=cuts_proposed,
                cuts_accepted=cuts_accepted,
                candidates_audited=candidates_audited,
                candidates_accepted=candidates_accepted,
            )
        for cut in separation.cuts:
            accept_cut(cut, require_violation=True)

        if not original_feasible:
            fixed = _fixed_program(program, last_master.primal)
            fixed_result = solve_prepared_convex_program(
                bind_convex_numeric(
                    state.fixed_template,
                    fixed,
                    numeric_version=prepared.numeric_version,
                )
            ).result
            fixed_discrete_solves += 1
            relaxation_solves += 1
            fixed_status = ConvexProgramStatus(int(np.asarray(fixed_result.status)))
            if bool(np.asarray(fixed_result.successful)):
                fixed_candidate = MixedIntegerCandidate(
                    fixed_result.primal,
                    reported_objective=fixed_result.objective,
                    source_kind="fixed-discrete-conic",
                    source_id=fixed_result.provenance.structure_id,
                )
                consider(fixed_candidate, fixed_result)
                cut = conic_cut_from_dual(
                    conic,
                    fixed_result.cone_dual,
                    source_kind="fixed-discrete-dual",
                    cone_block=-1,
                    cone_id=conic.cone.cone_id,
                    binding_id=prepared.binding_id,
                    source_primal=last_master.primal,
                )
                accept_cut(cut, require_violation=False)
            elif fixed_status == ConvexProgramStatus.PRIMAL_INFEASIBLE and bool(
                np.asarray(fixed_result.certificate.dual_ray_valid)
            ):
                cut = conic_cut_from_dual(
                    conic,
                    fixed_result.certificate.inequality_dual_ray,
                    source_kind="infeasibility-ray",
                    cone_block=-1,
                    cone_id=conic.cone.cone_id,
                    binding_id=prepared.binding_id,
                    source_primal=last_master.primal,
                )
                accept_cut(cut, require_violation=True)
            else:
                return _oa_result(
                    prepared,
                    method,
                    status=MixedIntegerStatus.RELAXATION_FAILURE,
                    audit=incumbent_audit,
                    lower_bound=global_lower_bound,
                    global_bound_certified=global_bound_certified,
                    search_complete=False,
                    incumbent_relaxation=incumbent_relaxation,
                    last_master=last_master,
                    explored_nodes=explored_nodes,
                    pruned_nodes=pruned_nodes,
                    frontier_size=frontier_size,
                    relaxation_solves=relaxation_solves,
                    master_solves=master_solves,
                    fixed_discrete_solves=fixed_discrete_solves,
                    cuts_proposed=cuts_proposed,
                    cuts_accepted=cuts_accepted,
                    candidates_audited=candidates_audited,
                    candidates_accepted=candidates_accepted,
                )

        objective = (
            float("inf")
            if incumbent_audit is None
            else float(np.asarray(incumbent_audit.objective))
        )
        absolute_gap, relative_gap = _gap(objective, global_lower_bound)
        if incumbent_audit is not None and (
            absolute_gap <= method.absolute_gap or relative_gap <= method.relative_gap
        ):
            status = (
                MixedIntegerStatus.OPTIMAL
                if master_status == MixedIntegerStatus.OPTIMAL
                else MixedIntegerStatus.GAP_REACHED
            )
            return _oa_result(
                prepared,
                method,
                status=status,
                audit=incumbent_audit,
                lower_bound=global_lower_bound,
                global_bound_certified=global_bound_certified,
                search_complete=status == MixedIntegerStatus.OPTIMAL,
                incumbent_relaxation=incumbent_relaxation,
                last_master=last_master,
                explored_nodes=explored_nodes,
                pruned_nodes=pruned_nodes,
                frontier_size=frontier_size,
                relaxation_solves=relaxation_solves,
                master_solves=master_solves,
                fixed_discrete_solves=fixed_discrete_solves,
                cuts_proposed=cuts_proposed,
                cuts_accepted=cuts_accepted,
                candidates_audited=candidates_audited,
                candidates_accepted=candidates_accepted,
            )
        if cuts_accepted == accepted_before:
            return _oa_result(
                prepared,
                method,
                status=MixedIntegerStatus.CERTIFICATION_FAILURE,
                audit=incumbent_audit,
                lower_bound=global_lower_bound,
                global_bound_certified=global_bound_certified,
                search_complete=False,
                incumbent_relaxation=incumbent_relaxation,
                last_master=last_master,
                explored_nodes=explored_nodes,
                pruned_nodes=pruned_nodes,
                frontier_size=frontier_size,
                relaxation_solves=relaxation_solves,
                master_solves=master_solves,
                fixed_discrete_solves=fixed_discrete_solves,
                cuts_proposed=cuts_proposed,
                cuts_accepted=cuts_accepted,
                candidates_audited=candidates_audited,
                candidates_accepted=candidates_accepted,
            )

    return _oa_result(
        prepared,
        method,
        status=MixedIntegerStatus.WORK_LIMIT,
        audit=incumbent_audit,
        lower_bound=global_lower_bound,
        global_bound_certified=global_bound_certified,
        search_complete=False,
        incumbent_relaxation=incumbent_relaxation,
        last_master=last_master,
        explored_nodes=explored_nodes,
        pruned_nodes=pruned_nodes,
        frontier_size=frontier_size,
        relaxation_solves=relaxation_solves,
        master_solves=master_solves,
        fixed_discrete_solves=fixed_discrete_solves,
        cuts_proposed=cuts_proposed,
        cuts_accepted=cuts_accepted,
        candidates_audited=candidates_audited,
        candidates_accepted=candidates_accepted,
    )


__all__ = [
    "ConicOuterApproximation",
    "prepare_conic_outer_approximation",
    "solve_conic_outer_approximation",
]
