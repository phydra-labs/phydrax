#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .._bounds import Bounds
from .._branch_and_bound import (
    AbstractBranchAndBoundProblem,
    branch_and_bound,
    BranchAndBoundStatus,
    BranchBoundEvidence,
    BranchCandidate,
    BranchNodeEvaluation,
)
from ._audit import audit_dual_infeasibility_ray, DualRayAudit
from ._lifecycle import (
    bind_convex_numeric,
    ConvexProgramExecution,
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
from ._mixed_integer_policy import (
    MixedIntegerSolvePolicy,
    NativeMixedIntegerBranchAndBound,
)
from ._problem import ConicProgram, LinearProgram
from ._quadratic import ConvexProgramResult, QuadraticProgram
from ._types import ConvexProgramStatus, ConvexWarmStart


@dataclass(slots=True)
class _NativeTemplateState:
    templates: dict[str, ConvexProgramTemplate]


@dataclass(slots=True)
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    path: str
    parent_result: ConvexProgramResult | None = None
    execution: ConvexProgramExecution | None = None
    infeasibility_certificate: DualRayAudit | None = None


@dataclass(slots=True)
class _NativeIncumbent:
    primal: jnp.ndarray
    audit: MixedIntegerCandidateAudit
    relaxation: ConvexProgramResult | None
    source_id: str


def prepare_native_mixed_integer(
    program: MixedIntegerProgram,
    method: NativeMixedIntegerBranchAndBound,
    /,
) -> _NativeTemplateState:
    if method.inherit_relaxation_start and isinstance(program.relaxation, ConicProgram):
        raise ValueError(
            "Parent relaxation warm starts are not yet defined for general conic nodes."
        )
    return _NativeTemplateState(
        {
            program.relaxation.structure_id: prepare_convex_template(
                program.relaxation,
                method.relaxation,
            )
        }
    )


def _child_warm_start(
    result: ConvexProgramResult,
    program: LinearProgram | QuadraticProgram,
    /,
) -> ConvexWarmStart:
    dtype = program.linear.dtype
    lower = np.asarray(program.lower_bounds)
    upper = np.asarray(program.upper_bounds)
    primal = np.asarray(result.primal, dtype=np.dtype(dtype)).copy()
    scale = np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    finite_scale = np.where(np.isfinite(scale), scale, 1.0)
    margin = 1e-8 * finite_scale
    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    two_sided = np.isfinite(lower) & np.isfinite(upper) & ~fixed
    margin[two_sided] = np.minimum(
        margin[two_sided],
        0.25 * (upper[two_sided] - lower[two_sided]),
    )
    primal[fixed] = lower[fixed]
    primal[two_sided] = np.clip(
        primal[two_sided],
        lower[two_sided] + margin[two_sided],
        upper[two_sided] - margin[two_sided],
    )
    lower_only = np.isfinite(lower) & ~np.isfinite(upper)
    upper_only = ~np.isfinite(lower) & np.isfinite(upper)
    primal[lower_only] = np.maximum(
        primal[lower_only], lower[lower_only] + margin[lower_only]
    )
    primal[upper_only] = np.minimum(
        primal[upper_only], upper[upper_only] - margin[upper_only]
    )
    scalar_margin = jnp.asarray(1e-8, dtype=dtype)
    return ConvexWarmStart(
        primal=jnp.asarray(primal, dtype=dtype),
        equality_dual=jnp.asarray(result.equality_dual, dtype=dtype),
        inequality_dual=jnp.maximum(
            jnp.asarray(result.inequality_dual, dtype=dtype), scalar_margin
        ),
        inequality_slack=jnp.maximum(
            jnp.asarray(result.inequality_slack, dtype=dtype), scalar_margin
        ),
        lower_bound_dual=jnp.maximum(
            jnp.asarray(result.lower_bound_dual, dtype=dtype), scalar_margin
        ),
        upper_bound_dual=jnp.maximum(
            jnp.asarray(result.upper_bound_dual, dtype=dtype), scalar_margin
        ),
        structure_id=program.structure_id,
    )


class _MixedIntegerBranchProblem(AbstractBranchAndBoundProblem):
    mixed: MixedIntegerProgram
    policy: MixedIntegerSolvePolicy
    method: NativeMixedIntegerBranchAndBound
    templates: dict[str, ConvexProgramTemplate]
    relaxation_solves: list[int]
    candidates_audited: list[int]
    candidates_accepted: list[int]

    def __init__(
        self,
        mixed: MixedIntegerProgram,
        policy: MixedIntegerSolvePolicy,
        state: _NativeTemplateState,
        /,
    ):
        method = policy.method
        if not isinstance(method, NativeMixedIntegerBranchAndBound):
            raise TypeError("Native mixed-integer execution requires its native method.")
        self.mixed = mixed
        self.policy = policy
        self.method = method
        self.templates = state.templates
        self.relaxation_solves = [0]
        self.candidates_audited = [0]
        self.candidates_accepted = [0]
        self.problem_id = mixed.structure_id

    def root(self, /) -> _Node:
        return _Node(
            np.asarray(self.mixed.relaxation.lower_bounds).copy(),
            np.asarray(self.mixed.relaxation.upper_bounds).copy(),
            "root",
        )

    def node_id(self, node: _Node, /) -> str:
        return node.path

    def _solve(self, node: _Node, /) -> ConvexProgramResult | None:
        if node.infeasibility_certificate is not None:
            return None
        if node.execution is None:
            numeric = _replace_bounds(
                self.mixed.relaxation,
                node.lower,
                node.upper,
            )
            quadratic = (
                numeric.as_quadratic_program()
                if isinstance(numeric, LinearProgram)
                else numeric
            )
            if isinstance(quadratic, QuadraticProgram):
                node.infeasibility_certificate = _linear_bound_certificate(
                    quadratic,
                    self.method.relaxation.termination.primal_infeasible,
                )
                if node.infeasibility_certificate is not None:
                    return None
            if numeric.structure_id not in self.templates:
                self.templates[numeric.structure_id] = prepare_convex_template(
                    numeric,
                    self.method.relaxation,
                )
            warm_start = None
            if (
                self.method.inherit_relaxation_start
                and node.parent_result is not None
                and isinstance(numeric, (LinearProgram, QuadraticProgram))
            ):
                warm_start = _child_warm_start(node.parent_result, numeric)
            node.execution = solve_prepared_convex_program(
                bind_convex_numeric(self.templates[numeric.structure_id], numeric),
                warm_start=warm_start,
            )
            self.relaxation_solves[0] += 1
        return node.execution.result

    def evaluate(self, node: _Node, /) -> BranchNodeEvaluation:
        if np.any(node.lower > node.upper):
            return BranchNodeEvaluation.proven_infeasible(
                f"{self.problem_id}:{node.path}:bound-contradiction"
            )
        result = self._solve(node)
        if node.infeasibility_certificate is not None:
            return BranchNodeEvaluation.proven_infeasible(
                f"{self.problem_id}:{node.path}:audited-farkas-ray",
                state=node.infeasibility_certificate,
            )
        if result is None:
            return BranchNodeEvaluation.failed(
                "missing-relaxation-result",
                "A mixed-integer node produced neither a relaxation nor a certificate.",
            )
        status = int(np.asarray(result.status))
        if status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE):
            if bool(np.asarray(result.certificate.dual_ray_valid)):
                return BranchNodeEvaluation.proven_infeasible(
                    f"{self.problem_id}:{node.path}:audited-primal-infeasibility",
                    state=result,
                )
            return BranchNodeEvaluation.failed(
                "uncertified-primal-infeasibility",
                "The relaxation reported infeasibility without a valid dual ray.",
                state=result,
            )
        if not bool(np.asarray(result.successful)):
            return BranchNodeEvaluation.failed(
                "relaxation-failure",
                "The convex node relaxation did not return an audited optimum.",
                state=result,
            )

        objective = float(np.asarray(result.objective))
        lower_bound = BranchBoundEvidence(
            objective,
            certified=True,
            certificate_id=f"{self.problem_id}:{node.path}:convex-optimum",
        )
        primal = np.asarray(result.primal)
        indices = np.asarray(self.mixed.discrete_indices, dtype=np.int64)
        values = primal[indices]
        integral = bool(
            np.all(np.isfinite(values))
            and np.all(
                np.abs(values - np.rint(values)) <= self.policy.certification.integrality
            )
        )
        candidate = None
        if integral:
            proposal = MixedIntegerCandidate(
                result.primal,
                reported_objective=result.objective,
                source_kind="node-relaxation",
                source_id=f"{self.problem_id}:{node.path}",
            )
            audit = audit_mixed_integer_candidate(
                self.mixed,
                proposal,
                self.policy.certification,
            )
            self.candidates_audited[0] += 1
            if not bool(np.asarray(audit.valid)):
                return BranchNodeEvaluation.failed(
                    "integral-candidate-audit-failed",
                    "An integral relaxation point failed canonical replay.",
                    state=audit,
                )
            self.candidates_accepted[0] += 1
            incumbent = _NativeIncumbent(
                audit.primal,
                audit,
                result,
                proposal.source_id,
            )
            candidate = BranchCandidate(
                incumbent,
                float(np.asarray(audit.objective)),
                certificate_id=proposal.candidate_id,
            )
        return BranchNodeEvaluation(
            lower_bound=lower_bound,
            candidate=candidate,
            terminal=integral,
            state=result,
        )

    def branch(
        self,
        node: _Node,
        evaluation: BranchNodeEvaluation,
        /,
    ) -> tuple[_Node, _Node]:
        result = evaluation.state
        if not isinstance(result, ConvexProgramResult):
            raise TypeError("A mixed-integer branch requires a convex relaxation.")
        primal = np.asarray(result.primal)
        indices = np.asarray(self.mixed.discrete_indices, dtype=np.int64)
        values = primal[indices]
        position = int(np.argmax(np.abs(values - np.rint(values))))
        variable, value = int(indices[position]), float(values[position])
        floor, ceil = np.floor(value), np.ceil(value)
        llo, lhi, rlo, rhi = (
            node.lower.copy(),
            node.upper.copy(),
            node.lower.copy(),
            node.upper.copy(),
        )
        lhi[variable] = min(lhi[variable], floor)
        rlo[variable] = max(rlo[variable], ceil)
        return (
            _Node(
                llo,
                lhi,
                f"{node.path}/x{variable}<={floor:g}",
                parent_result=result,
            ),
            _Node(
                rlo,
                rhi,
                f"{node.path}/x{variable}>={ceil:g}",
                parent_result=result,
            ),
        )


def solve_native_mixed_integer(prepared, candidates, /) -> MixedIntegerResult:
    from ._mixed_integer_lifecycle import PreparedMixedIntegerProgram

    if not isinstance(prepared, PreparedMixedIntegerProgram):
        raise TypeError("prepared must be a PreparedMixedIntegerProgram.")
    method = prepared.plan.policy.method
    if not isinstance(method, NativeMixedIntegerBranchAndBound):
        raise TypeError("Prepared policy is not the native mixed-integer method.")
    state = prepared.template.method_state
    if not isinstance(state, _NativeTemplateState):
        raise TypeError("Prepared native mixed-integer state is invalid.")

    audited_count = 0
    accepted_count = 0
    initial = None
    for candidate in candidates:
        audit = audit_mixed_integer_candidate(
            prepared.program,
            candidate,
            prepared.plan.policy.certification,
        )
        audited_count += 1
        if bool(np.asarray(audit.valid)):
            accepted_count += 1
            incumbent = _NativeIncumbent(
                audit.primal,
                audit,
                None,
                candidate.source_id,
            )
            branch_candidate = BranchCandidate(
                incumbent,
                float(np.asarray(audit.objective)),
                certificate_id=candidate.candidate_id,
            )
            if initial is None or branch_candidate.objective < initial.objective:
                initial = branch_candidate

    problem = _MixedIntegerBranchProblem(
        prepared.program,
        prepared.plan.policy,
        state,
    )
    search = branch_and_bound(
        problem,
        policy=method.tree,
        initial_candidate=initial,
    )
    incumbent = search.incumbent
    audit = None if incumbent is None else incumbent.audit
    relaxation = None if incumbent is None else incumbent.relaxation
    primal = (
        jnp.full(
            (prepared.program.relaxation.num_variables,),
            jnp.nan,
            dtype=prepared.program.relaxation.linear.dtype,
        )
        if audit is None
        else audit.primal
    )
    branch_status = BranchAndBoundStatus(int(np.asarray(search.status)))
    status = {
        BranchAndBoundStatus.OPTIMAL: MixedIntegerStatus.OPTIMAL,
        BranchAndBoundStatus.GAP_REACHED: MixedIntegerStatus.GAP_REACHED,
        BranchAndBoundStatus.WORK_LIMIT: MixedIntegerStatus.WORK_LIMIT,
        BranchAndBoundStatus.INFEASIBLE: MixedIntegerStatus.INFEASIBLE,
        BranchAndBoundStatus.UNBOUNDED: MixedIntegerStatus.RELAXATION_FAILURE,
        BranchAndBoundStatus.EVALUATION_FAILURE: MixedIntegerStatus.RELAXATION_FAILURE,
    }[branch_status]
    optimality = (
        status == MixedIntegerStatus.OPTIMAL
        and audit is not None
        and bool(np.asarray(audit.valid))
        and bool(np.asarray(search.global_lower_bound_certified))
        and bool(np.asarray(search.search_complete))
    )
    certificate = MixedIntegerCertificate(
        candidate_audit=audit,
        infeasibility_certified=jnp.asarray(
            status == MixedIntegerStatus.INFEASIBLE
            and bool(np.asarray(search.search_complete))
        ),
        global_bound_certified=search.global_lower_bound_certified,
        search_complete=search.search_complete,
        optimality_certified=jnp.asarray(optimality),
        provider_solved=jnp.asarray(False),
        proof_kind="native-audited",
    )
    work = MixedIntegerWork(
        search.explored_nodes,
        search.pruned_nodes,
        search.frontier_size,
        jnp.asarray(True),
        jnp.asarray(problem.relaxation_solves[0], dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(audited_count + problem.candidates_audited[0], dtype=jnp.int32),
        jnp.asarray(accepted_count + problem.candidates_accepted[0], dtype=jnp.int32),
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
    return MixedIntegerResult(
        primal,
        search.objective,
        search.global_lower_bound,
        search.absolute_gap,
        search.relative_gap,
        jnp.asarray(int(status), dtype=jnp.int32),
        certificate,
        work,
        provenance,
        relaxation,
        search,
    )


def _linear_bound_certificate(
    problem: QuadraticProgram, tolerance: float, /
) -> DualRayAudit | None:
    """Derive affine implications; prune only with an independently audited ray."""
    equality = np.asarray(problem.equality_matrix)
    inequality = np.asarray(problem.inequality_matrix)
    matrix = np.concatenate((equality, -equality, inequality))
    rhs = np.concatenate(
        (
            np.asarray(problem.equality_rhs),
            -np.asarray(problem.equality_rhs),
            np.asarray(problem.inequality_rhs),
        )
    )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        return None
    n, m = problem.num_variables, problem.num_equalities
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    lower_proof, upper_proof = [None] * n, [None] * n
    supports = tuple(np.flatnonzero(row) for row in matrix)

    def add(target, proof, scale):
        for index, value in proof.items():
            target[index] = target.get(index, 0.0) + scale * value

    def audit(proof):
        multipliers = np.zeros(len(rhs))
        for index, value in proof.items():
            multipliers[index] = value
        equality_multiplier = multipliers[:m] - multipliers[m : 2 * m]
        inequality_multiplier = multipliers[2 * m :]
        lower_multiplier, upper_multiplier = np.zeros(n), np.zeros(n)
        fixed = np.asarray(problem.fixed_bound_indices, dtype=int)
        signed = equality_multiplier[problem.num_user_equalities :]
        lower_multiplier[fixed] = np.maximum(-signed, 0)
        upper_multiplier[fixed] = np.maximum(signed, 0)
        start = problem.num_user_inequalities
        middle = start + len(problem.lower_bound_indices)
        lower_multiplier[np.asarray(problem.lower_bound_indices, dtype=int)] = (
            inequality_multiplier[start:middle]
        )
        upper_multiplier[np.asarray(problem.upper_bound_indices, dtype=int)] = (
            inequality_multiplier[middle:]
        )
        result = audit_dual_infeasibility_ray(
            problem,
            jnp.asarray(equality_multiplier[: problem.num_user_equalities]),
            jnp.asarray(inequality_multiplier[: problem.num_user_inequalities]),
            jnp.asarray(lower_multiplier),
            jnp.asarray(upper_multiplier),
            tolerance=tolerance,
        )
        return result if bool(np.asarray(result.valid)) else None

    for _ in range(n + len(rhs) + 1):
        changed = False
        for row, indices in enumerate(supports):
            coefficients = matrix[row, indices]
            selected = np.where(
                coefficients > 0,
                lower[indices],
                upper[indices],
            )
            contributions = coefficients * selected
            if np.any(np.isnan(contributions)) or np.any(np.isposinf(contributions)):
                continue
            known = np.isfinite(contributions)
            unknown = int(np.count_nonzero(~known))
            minimum = float(np.sum(contributions[known]))
            proofs = tuple(
                lower_proof[index] if coefficient > 0 else upper_proof[index]
                for index, coefficient in zip(
                    indices,
                    coefficients,
                    strict=True,
                )
            )
            if unknown == 0 and minimum > rhs[row] + tolerance:
                proof = {row: 1.0}
                for coefficient, bound_proof in zip(
                    coefficients,
                    proofs,
                    strict=True,
                ):
                    add(proof, bound_proof, abs(coefficient))
                certificate = audit(proof)
                if certificate is not None:
                    return certificate
            for position, index in enumerate(indices):
                if unknown > int(not known[position]):
                    continue
                other = minimum - (contributions[position] if known[position] else 0.0)
                coefficient = coefficients[position]
                candidate = (rhs[row] - other) / coefficient
                if not np.isfinite(candidate):
                    continue
                improves = (
                    candidate < upper[index] - tolerance
                    if coefficient > 0
                    else candidate > lower[index] + tolerance
                )
                if not improves:
                    continue
                proof = {row: 1.0 / abs(coefficient)}
                for other_position, bound_proof in enumerate(proofs):
                    if other_position != position:
                        add(
                            proof,
                            bound_proof,
                            abs(coefficients[other_position] / coefficient),
                        )
                if coefficient > 0:
                    upper[index], upper_proof[index] = candidate, proof
                else:
                    lower[index], lower_proof[index] = candidate, proof
                changed = True
                if lower[index] > upper[index] + tolerance:
                    contradiction = dict(lower_proof[index])
                    add(contradiction, upper_proof[index], 1.0)
                    certificate = audit(contradiction)
                    if certificate is not None:
                        return certificate
        if not changed:
            break
    return None


def _replace_bounds(program, lower, upper, /):
    if np.array_equal(lower, np.asarray(program.lower_bounds)) and np.array_equal(
        upper, np.asarray(program.upper_bounds)
    ):
        return program
    dtype = program.linear.dtype
    bounds = Bounds(
        jnp.asarray(lower, dtype=dtype),
        jnp.asarray(upper, dtype=dtype),
    )
    if isinstance(program, ConicProgram):
        return ConicProgram(
            program.quadratic,
            program.linear,
            program.constraint_matrix,
            program.constraint_rhs,
            program.cone,
            bounds=bounds,
            problem_id=program.problem_id,
            convexity_evidence=program.convexity_evidence,
        )
    if isinstance(program, LinearProgram):
        return LinearProgram(
            program.linear,
            equality_matrix=program.equality_matrix,
            equality_rhs=program.equality_rhs,
            inequality_matrix=program.inequality_matrix,
            inequality_rhs=program.inequality_rhs,
            bounds=bounds,
            problem_id=program.problem_id,
        )
    return QuadraticProgram(
        program.quadratic,
        program.linear,
        equality_matrix=program.equality_matrix[..., : program.num_user_equalities, :],
        equality_rhs=program.equality_rhs[..., : program.num_user_equalities],
        inequality_matrix=program.inequality_matrix[
            ..., : program.num_user_inequalities, :
        ],
        inequality_rhs=program.inequality_rhs[..., : program.num_user_inequalities],
        bounds=bounds,
        problem_id=program.problem_id,
        convexity_evidence=program.convexity_evidence,
    )


__all__ = ["prepare_native_mixed_integer", "solve_native_mixed_integer"]
