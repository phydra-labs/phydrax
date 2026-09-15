#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._bounds import Bounds
from ._mixed_integer_nonlinear import (
    ConvexMINLPCandidateAudit,
    ConvexMINLPEvaluation,
    ConvexMixedIntegerNonlinearProgram,
)
from ._programming._mixed_integer import (
    MixedIntegerProgram,
    MixedIntegerResult,
    MixedIntegerSolvePolicy,
    MixedIntegerStatus,
    solve_mixed_integer_program,
)
from ._programming._mixed_integer_policy import (
    AbstractMixedIntegerMethod,
    NativeMixedIntegerBranchAndBound,
)
from ._programming._problem import LinearProgram


ConvexMINLPCutKind: TypeAlias = Literal[
    "objective",
    "convex-upper",
    "concave-lower",
    "affine-equality",
]


class ConvexMINLPStatus(IntEnum):
    OPTIMAL = 0
    GAP_REACHED = 1
    WORK_LIMIT = 2
    INFEASIBLE = 3
    MASTER_FAILURE = 4
    CERTIFICATION_FAILURE = 5
    NONFINITE_EVALUATION = 6


class ConvexMINLPCut(StrictModule, NonTrainableState):
    row: Array
    rhs: Array
    equality: bool = eqx.field(static=True)
    kind: ConvexMINLPCutKind = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    cut_id: str = eqx.field(static=True)

    def __init__(
        self,
        row: Array,
        rhs: Array,
        /,
        *,
        equality: bool,
        kind: ConvexMINLPCutKind,
        source_index: int,
        evidence_id: str,
    ):
        row_ = jnp.asarray(row)
        rhs_ = jnp.asarray(rhs, dtype=row_.dtype)
        if row_.ndim != 1 or rhs_.shape != ():
            raise ValueError("Convex MINLP cuts require one row and scalar RHS.")
        if not bool(np.asarray(jnp.all(jnp.isfinite(row_)) & jnp.isfinite(rhs_))):
            raise ValueError("Convex MINLP cuts must be finite.")
        identifier = str(evidence_id)
        if not identifier:
            raise ValueError("Cut evidence_id must be nonempty.")
        self.row = row_
        self.rhs = rhs_
        self.equality = bool(equality)
        self.kind = kind
        self.source_index = int(source_index)
        self.evidence_id = identifier
        self.cut_id = canonical_fingerprint(
            {
                "kind": "convex-minlp-cut",
                "cut_kind": kind,
                "source_index": int(source_index),
                "evidence": identifier,
                "geometry": array_tree_fingerprint((row_, rhs_)),
            }
        )


class ConvexMINLPOuterApproximation(StrictModule, NonTrainableState):
    """Extended cutting-plane OA for compact convex MINLPs."""

    master: AbstractMixedIntegerMethod
    maximum_rounds: int = eqx.field(static=True)
    maximum_cuts: int = eqx.field(static=True)
    feasibility_tolerance: float = eqx.field(static=True)
    integrality_tolerance: float = eqx.field(static=True)
    duplicate_tolerance: float = eqx.field(static=True)
    absolute_gap: float = eqx.field(static=True)
    relative_gap: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        master: AbstractMixedIntegerMethod | None = None,
        /,
        *,
        maximum_rounds: int = 100,
        maximum_cuts: int = 10_000,
        feasibility_tolerance: float = 1e-7,
        integrality_tolerance: float = 1e-7,
        duplicate_tolerance: float = 1e-9,
        absolute_gap: float = 1e-7,
        relative_gap: float = 1e-7,
    ):
        master_ = NativeMixedIntegerBranchAndBound() if master is None else master
        if not isinstance(master_, AbstractMixedIntegerMethod):
            raise TypeError("master must be an AbstractMixedIntegerMethod.")
        if not master_.capabilities.linear_program:
            raise ValueError("Convex MINLP OA requires a MILP-capable master.")
        rounds, cut_limit = int(maximum_rounds), int(maximum_cuts)
        values = tuple(
            float(value)
            for value in (
                feasibility_tolerance,
                integrality_tolerance,
                duplicate_tolerance,
                absolute_gap,
                relative_gap,
            )
        )
        if rounds < 1 or cut_limit < 1:
            raise ValueError("maximum_rounds and maximum_cuts must be positive.")
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Convex MINLP tolerances must be finite and nonnegative.")
        if not 0.0 < values[1] < 0.5:
            raise ValueError("integrality_tolerance must lie in (0, 0.5).")
        if values[3] == 0.0 and values[4] == 0.0:
            raise ValueError("At least one optimality tolerance must be positive.")
        self.master = master_
        self.maximum_rounds = rounds
        self.maximum_cuts = cut_limit
        (
            self.feasibility_tolerance,
            self.integrality_tolerance,
            self.duplicate_tolerance,
            self.absolute_gap,
            self.relative_gap,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "convex-minlp-outer-approximation",
                "master": master_.method_id,
                "master_configuration": list(master_.configuration),
                "maximum_rounds": rounds,
                "maximum_cuts": cut_limit,
                "tolerances": list(values),
            }
        )


class ConvexMINLPWork(StrictModule, NonTrainableState):
    master_solves: Array
    explored_nodes: Array
    function_evaluations: Array
    derivative_evaluations: Array
    cuts_proposed: Array
    cuts_accepted: Array
    candidates_audited: Array
    candidates_accepted: Array


class ConvexMINLPCertificate(StrictModule, NonTrainableState):
    candidate_audit: ConvexMINLPCandidateAudit | None
    global_bound_certified: Array
    search_complete: Array
    optimality_certified: Array
    convexity_evidence: tuple[str, ...] = eqx.field(static=True)


class ConvexMINLPResult(StrictModule):
    primal: Array
    objective: Array
    global_lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    status: Array
    certificate: ConvexMINLPCertificate
    work: ConvexMINLPWork
    last_master: MixedIntegerResult | None
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            self.status == int(ConvexMINLPStatus.OPTIMAL)
        ) & self.certificate.optimality_certified


def _linearization_cuts(
    program: ConvexMixedIntegerNonlinearProgram,
    evaluation: ConvexMINLPEvaluation,
    /,
    *,
    include_satisfied: bool,
    tolerance: float,
) -> tuple[ConvexMINLPCut, ...]:
    point = evaluation.primal
    objective_row = jnp.concatenate(
        (evaluation.gradient, jnp.asarray([-1.0], dtype=point.dtype))
    )
    objective_rhs = (
        ein.contract("i,i->", evaluation.gradient, point) - evaluation.objective
    )
    cuts = [
        ConvexMINLPCut(
            objective_row,
            objective_rhs,
            equality=False,
            kind="objective",
            source_index=-1,
            evidence_id=program.objective_evidence_id,
        )
    ]
    for index, evidence in enumerate(program.constraint_evidence):
        value = evaluation.constraints[index]
        gradient = evaluation.jacobian[index]
        row = jnp.concatenate((gradient, jnp.asarray([0.0], dtype=point.dtype)))
        if evidence.kind == "affine-equality":
            rhs = (
                program.constraint_lower[index]
                - value
                + ein.contract("i,i->", gradient, point)
            )
            cuts.append(
                ConvexMINLPCut(
                    row,
                    rhs,
                    equality=True,
                    kind="affine-equality",
                    source_index=index,
                    evidence_id=evidence.evidence_id,
                )
            )
        elif evidence.kind == "convex-upper":
            violation = value - program.constraint_upper[index]
            if include_satisfied or bool(np.asarray(violation > tolerance)):
                rhs = (
                    program.constraint_upper[index]
                    - value
                    + ein.contract("i,i->", gradient, point)
                )
                cuts.append(
                    ConvexMINLPCut(
                        row,
                        rhs,
                        equality=False,
                        kind="convex-upper",
                        source_index=index,
                        evidence_id=evidence.evidence_id,
                    )
                )
        else:
            violation = program.constraint_lower[index] - value
            if include_satisfied or bool(np.asarray(violation > tolerance)):
                concave_row = -row
                rhs = (
                    value
                    - ein.contract("i,i->", gradient, point)
                    - program.constraint_lower[index]
                )
                cuts.append(
                    ConvexMINLPCut(
                        concave_row,
                        rhs,
                        equality=False,
                        kind="concave-lower",
                        source_index=index,
                        evidence_id=evidence.evidence_id,
                    )
                )
    return tuple(cuts)


def _master_program(
    program: ConvexMixedIntegerNonlinearProgram,
    cuts: tuple[ConvexMINLPCut, ...],
    /,
) -> MixedIntegerProgram:
    n = program.num_variables
    equalities = [cut.row for cut in cuts if cut.equality]
    equality_rhs = [cut.rhs for cut in cuts if cut.equality]
    inequalities = [cut.row for cut in cuts if not cut.equality]
    inequality_rhs = [cut.rhs for cut in cuts if not cut.equality]
    dtype = program.variable_lower.dtype
    linear = jnp.concatenate((jnp.zeros((n,), dtype=dtype), jnp.ones((1,), dtype=dtype)))
    equality_matrix = (
        jnp.stack(equalities) if equalities else jnp.empty((0, n + 1), dtype=dtype)
    )
    equality_vector = (
        jnp.stack(equality_rhs) if equality_rhs else jnp.empty((0,), dtype=dtype)
    )
    inequality_matrix = (
        jnp.stack(inequalities) if inequalities else jnp.empty((0, n + 1), dtype=dtype)
    )
    inequality_vector = (
        jnp.stack(inequality_rhs) if inequality_rhs else jnp.empty((0,), dtype=dtype)
    )
    relaxation = LinearProgram(
        linear,
        equality_matrix=equality_matrix,
        equality_rhs=equality_vector,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_vector,
        bounds=Bounds(
            jnp.concatenate(
                (program.variable_lower, jnp.asarray([-jnp.inf], dtype=dtype))
            ),
            jnp.concatenate(
                (program.variable_upper, jnp.asarray([jnp.inf], dtype=dtype))
            ),
        ),
        problem_id=f"{program.program_id}:ecp-master",
    )
    return MixedIntegerProgram(
        relaxation,
        integer_indices=program.integer_indices,
        binary_indices=program.binary_indices,
        program_id=f"{program.program_id}:ecp-master",
    )


def _finalize(
    program,
    policy,
    status,
    audit,
    lower_bound,
    global_bound_certified,
    search_complete,
    work,
    last_master,
):
    objective = float("inf") if audit is None else float(np.asarray(audit.objective))
    if np.isfinite(objective) and np.isfinite(lower_bound):
        lower_bound = min(lower_bound, objective)
        absolute_gap = objective - lower_bound
        relative_gap = absolute_gap / max(abs(objective), 1.0)
    else:
        absolute_gap = float("inf")
        relative_gap = float("inf")
    optimality = (
        status == ConvexMINLPStatus.OPTIMAL
        and audit is not None
        and bool(np.asarray(audit.valid))
        and global_bound_certified
        and search_complete
        and (absolute_gap <= policy.absolute_gap or relative_gap <= policy.relative_gap)
    )
    evidence = (
        program.objective_evidence_id,
        *(value.evidence_id for value in program.constraint_evidence),
    )
    certificate = ConvexMINLPCertificate(
        audit,
        jnp.asarray(global_bound_certified),
        jnp.asarray(search_complete),
        jnp.asarray(optimality),
        evidence,
    )
    primal = (
        jnp.full((program.num_variables,), jnp.nan, dtype=program.variable_lower.dtype)
        if audit is None
        else audit.primal
    )
    result = ConvexMINLPResult(
        primal,
        jnp.asarray(objective),
        jnp.asarray(lower_bound),
        jnp.asarray(absolute_gap),
        jnp.asarray(relative_gap),
        jnp.asarray(int(status), dtype=jnp.int32),
        certificate,
        work,
        last_master,
        program.program_id,
        program.structure_id,
        policy.policy_id,
    )
    return result


def solve_convex_minlp(
    program: ConvexMixedIntegerNonlinearProgram,
    policy: ConvexMINLPOuterApproximation | None = None,
    /,
) -> ConvexMINLPResult:
    """Solve a compact declared-convex MINLP through ECP outer approximation."""
    if not isinstance(program, ConvexMixedIntegerNonlinearProgram):
        raise TypeError("program must be ConvexMixedIntegerNonlinearProgram.")
    selected = ConvexMINLPOuterApproximation() if policy is None else policy
    if not isinstance(selected, ConvexMINLPOuterApproximation):
        raise TypeError("policy must be ConvexMINLPOuterApproximation or None.")
    cuts: list[ConvexMINLPCut] = []
    incumbent = None
    last_master = None
    global_lower_bound = -float("inf")
    master_solves = 0
    explored_nodes = 0
    function_evaluations = 0
    derivative_evaluations = 0
    cuts_proposed = 0
    cuts_accepted = 0
    candidates_audited = 0
    candidates_accepted = 0

    def accept(cut):
        nonlocal cuts_proposed, cuts_accepted
        cuts_proposed += 1
        for existing in cuts:
            if existing.equality != cut.equality:
                continue
            if np.allclose(
                np.asarray(existing.row),
                np.asarray(cut.row),
                atol=selected.duplicate_tolerance,
                rtol=selected.duplicate_tolerance,
            ) and np.isclose(
                float(np.asarray(existing.rhs)),
                float(np.asarray(cut.rhs)),
                atol=selected.duplicate_tolerance,
                rtol=selected.duplicate_tolerance,
            ):
                return False
        if len(cuts) >= selected.maximum_cuts:
            return False
        cuts.append(cut)
        cuts_accepted += 1
        return True

    midpoint = 0.5 * (program.variable_lower + program.variable_upper)
    evaluation = program.evaluate(midpoint)
    function_evaluations += 1
    derivative_evaluations += 1
    if not bool(np.asarray(evaluation.finite)):
        work = ConvexMINLPWork(
            *(
                jnp.asarray(value, dtype=jnp.int32)
                for value in (
                    master_solves,
                    explored_nodes,
                    function_evaluations,
                    derivative_evaluations,
                    cuts_proposed,
                    cuts_accepted,
                    candidates_audited,
                    candidates_accepted,
                )
            )
        )
        return _finalize(
            program,
            selected,
            ConvexMINLPStatus.NONFINITE_EVALUATION,
            None,
            global_lower_bound,
            False,
            False,
            work,
            None,
        )
    for cut in _linearization_cuts(
        program,
        evaluation,
        include_satisfied=True,
        tolerance=selected.feasibility_tolerance,
    ):
        accept(cut)

    for _ in range(selected.maximum_rounds):
        master = _master_program(program, tuple(cuts))
        last_master = solve_mixed_integer_program(
            master,
            MixedIntegerSolvePolicy(selected.master),
        )
        master_solves += 1
        explored_nodes += int(np.asarray(last_master.explored_nodes))
        if bool(np.asarray(last_master.certificate.global_bound_certified)):
            global_lower_bound = max(
                global_lower_bound,
                float(np.asarray(last_master.global_lower_bound)),
            )
        master_status = MixedIntegerStatus(int(np.asarray(last_master.status)))
        if master_status == MixedIntegerStatus.INFEASIBLE:
            work = ConvexMINLPWork(
                *(
                    jnp.asarray(value, dtype=jnp.int32)
                    for value in (
                        master_solves,
                        explored_nodes,
                        function_evaluations,
                        derivative_evaluations,
                        cuts_proposed,
                        cuts_accepted,
                        candidates_audited,
                        candidates_accepted,
                    )
                )
            )
            return _finalize(
                program,
                selected,
                ConvexMINLPStatus.INFEASIBLE,
                None,
                float("inf"),
                bool(np.asarray(last_master.certificate.infeasibility_certified)),
                True,
                work,
                last_master,
            )
        if not bool(np.asarray(last_master.feasible)):
            status = (
                ConvexMINLPStatus.WORK_LIMIT
                if master_status == MixedIntegerStatus.WORK_LIMIT
                else ConvexMINLPStatus.MASTER_FAILURE
            )
            work = ConvexMINLPWork(
                *(
                    jnp.asarray(value, dtype=jnp.int32)
                    for value in (
                        master_solves,
                        explored_nodes,
                        function_evaluations,
                        derivative_evaluations,
                        cuts_proposed,
                        cuts_accepted,
                        candidates_audited,
                        candidates_accepted,
                    )
                )
            )
            return _finalize(
                program,
                selected,
                status,
                incumbent,
                global_lower_bound,
                False,
                False,
                work,
                last_master,
            )

        point = last_master.primal[: program.num_variables]
        audit = program.audit(
            point,
            feasibility_tolerance=selected.feasibility_tolerance,
            integrality_tolerance=selected.integrality_tolerance,
        )
        candidates_audited += 1
        if bool(np.asarray(audit.valid)):
            candidates_accepted += 1
            if incumbent is None or bool(
                np.asarray(audit.objective < incumbent.objective)
            ):
                incumbent = audit
        evaluation = program.evaluate(point)
        function_evaluations += 1
        derivative_evaluations += 1
        if not bool(np.asarray(evaluation.finite)):
            status = ConvexMINLPStatus.NONFINITE_EVALUATION
            break
        before = cuts_accepted
        for cut in _linearization_cuts(
            program,
            evaluation,
            include_satisfied=False,
            tolerance=selected.feasibility_tolerance,
        ):
            accept(cut)
        objective = (
            float("inf") if incumbent is None else float(np.asarray(incumbent.objective))
        )
        if np.isfinite(objective) and np.isfinite(global_lower_bound):
            lower = min(global_lower_bound, objective)
            absolute_gap = objective - lower
            relative_gap = absolute_gap / max(abs(objective), 1.0)
            if (
                absolute_gap <= selected.absolute_gap
                or relative_gap <= selected.relative_gap
            ):
                status = (
                    ConvexMINLPStatus.OPTIMAL
                    if master_status == MixedIntegerStatus.OPTIMAL
                    else ConvexMINLPStatus.GAP_REACHED
                )
                work = ConvexMINLPWork(
                    *(
                        jnp.asarray(value, dtype=jnp.int32)
                        for value in (
                            master_solves,
                            explored_nodes,
                            function_evaluations,
                            derivative_evaluations,
                            cuts_proposed,
                            cuts_accepted,
                            candidates_audited,
                            candidates_accepted,
                        )
                    )
                )
                return _finalize(
                    program,
                    selected,
                    status,
                    incumbent,
                    global_lower_bound,
                    bool(np.asarray(last_master.certificate.global_bound_certified)),
                    status == ConvexMINLPStatus.OPTIMAL,
                    work,
                    last_master,
                )
        if cuts_accepted == before:
            status = ConvexMINLPStatus.CERTIFICATION_FAILURE
            break
    else:
        status = ConvexMINLPStatus.WORK_LIMIT

    work = ConvexMINLPWork(
        *(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                master_solves,
                explored_nodes,
                function_evaluations,
                derivative_evaluations,
                cuts_proposed,
                cuts_accepted,
                candidates_audited,
                candidates_accepted,
            )
        )
    )
    return _finalize(
        program,
        selected,
        status,
        incumbent,
        global_lower_bound,
        False,
        False,
        work,
        last_master,
    )


__all__ = [
    "ConvexMINLPCertificate",
    "ConvexMINLPCut",
    "ConvexMINLPCutKind",
    "ConvexMINLPOuterApproximation",
    "ConvexMINLPResult",
    "ConvexMINLPStatus",
    "ConvexMINLPWork",
    "solve_convex_minlp",
]
