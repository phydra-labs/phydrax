#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ...._strict import StrictModule
from ....optim import (
    AbstractMinimizationMethod,
    Bounds,
    MinimizationProblem,
    MinimizationResult,
    minimize,
    NonlinearConstraint,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ._buckling import LocalMemberBucklingResult
from ._construction import ConstructionSequenceResult
from ._equilibrium import MemberNetworkResult
from ._prestress import PrestressRealizabilityResult, StructuralEvidenceVerdict


class MemberSizingEvaluation(StrictModule):
    """Mass, utilization, serviceability, and governing source evidence."""

    mass: Array
    cost: Array
    carbon: Array
    axial_stress: Array
    tension_utilization: Array
    compression_utilization: Array
    displacement_norm: Array
    buckling: LocalMemberBucklingResult | None
    maximum_utilization: Array
    governing_member: Array
    valid: Array


class MemberSizingConstraint(StrictModule):
    function: Callable = eqx.field(static=True)
    lower: Any
    upper: Any
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable,
        /,
        *,
        lower: Any = -jnp.inf,
        upper: Any = jnp.inf,
        constraint_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        self.function = function
        self.lower = lower
        self.upper = upper
        self.constraint_id = str(constraint_id)


class ContinuousMemberSizingProblem(StrictModule):
    """Design evaluator plus explicit physical objective and constraints."""

    evaluate_design: Callable = eqx.field(static=True)
    objective: Callable = eqx.field(static=True)
    bounds: Bounds | None
    constraints: tuple[MemberSizingConstraint, ...]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluate_design: Callable,
        /,
        *,
        objective: Callable | None = None,
        bounds: Bounds | None = None,
        constraints: Sequence[MemberSizingConstraint] = (),
        problem_id: str = "continuous-member-sizing",
    ):
        if not callable(evaluate_design):
            raise TypeError("evaluate_design must be callable.")
        objective_ = (
            (lambda evaluation, design, args: evaluation.mass)
            if objective is None
            else objective
        )
        if not callable(objective_):
            raise TypeError("objective must be callable.")
        constraints_ = tuple(constraints)
        if any(not isinstance(value, MemberSizingConstraint) for value in constraints_):
            raise TypeError("constraints must contain MemberSizingConstraint values.")
        self.evaluate_design = evaluate_design
        self.objective = objective_
        self.bounds = bounds
        self.constraints = constraints_
        self.problem_id = str(problem_id)


class ContinuousMemberSizingResult(StrictModule):
    optimization: MinimizationResult
    evaluation: MemberSizingEvaluation
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.optimization.successful & self.evaluation.valid


class CatalogMemberSizingResult(StrictModule):
    selected_index: int = eqx.field(static=True)
    selected_label: str = eqx.field(static=True)
    evaluation: MemberSizingEvaluation
    scores: Array
    valid_candidates: Array
    successful: Array


class StructuralVerificationResult(StrictModule):
    """Aggregate required structural evidence without treating absence as success."""

    verdict: Array
    equilibrium: MemberNetworkResult | None
    prestress: PrestressRealizabilityResult | None
    construction: ConstructionSequenceResult | None
    sizing: MemberSizingEvaluation | None
    local_buckling: LocalMemberBucklingResult | None
    required: tuple[str, ...] = eqx.field(static=True)
    failed: tuple[str, ...] = eqx.field(static=True)
    missing: tuple[str, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.verdict == int(StructuralEvidenceVerdict.CERTIFIED)


def evaluate_member_sizing(
    definition,
    result: MemberNetworkResult,
    /,
    *,
    local_buckling: LocalMemberBucklingResult | None = None,
) -> MemberSizingEvaluation:
    """Evaluate physical section demand for one accepted member-network state."""
    if not isinstance(result, MemberNetworkResult):
        raise TypeError("result must be a MemberNetworkResult.")
    properties = definition.properties.structural_arrays()
    structure = definition.structure
    vectors = (
        result.state.kinematics.positions[structure.receivers]
        - result.state.kinematics.positions[structure.senders]
    )
    lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
    member_mass = properties["density"] * properties["area"] * lengths
    section_values = definition.properties.sections
    cost_per_mass = jnp.asarray([value.cost_per_mass for value in section_values])[
        definition.properties.member_section
    ]
    carbon_per_mass = jnp.asarray([value.carbon_per_mass for value in section_values])[
        definition.properties.member_section
    ]
    axial_stress = result.state.assembly.axial_force / properties["area"]
    tension = jnp.maximum(axial_stress, 0.0) / properties["tension_allowable"]
    compression = jnp.maximum(-axial_stress, 0.0) / properties["compression_allowable"]
    utilization = jnp.maximum(tension, compression)
    if local_buckling is not None:
        utilization = jnp.maximum(utilization, local_buckling.utilization)
    displacement = result.state.kinematics.positions - definition.reference.positions
    displacement_norm = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
    maximum = jnp.max(utilization)
    governing = jnp.argmax(utilization).astype(jnp.int32)
    valid = (
        result.successful
        & jnp.all(jnp.isfinite(utilization))
        & jnp.all(properties["area"] > 0.0)
    )
    return MemberSizingEvaluation(
        jnp.sum(member_mass),
        jnp.sum(member_mass * cost_per_mass),
        jnp.sum(member_mass * carbon_per_mass),
        axial_stress,
        tension,
        compression,
        displacement_norm,
        local_buckling,
        maximum,
        governing,
        valid,
    )


def solve_continuous_member_sizing(
    problem: ContinuousMemberSizingProblem,
    initial_design: PyTree[Any],
    /,
    *,
    method: AbstractMinimizationMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> ContinuousMemberSizingResult:
    """Solve one continuous sizing problem through ordinary PhydraX optimization."""

    def objective(design, arguments):
        evaluation = problem.evaluate_design(design, arguments)
        if not isinstance(evaluation, MemberSizingEvaluation):
            raise TypeError("evaluate_design must return MemberSizingEvaluation.")
        return jnp.asarray(problem.objective(evaluation, design, arguments))

    constraints = []
    for constraint in problem.constraints:

        def value(design, arguments, constraint=constraint):
            evaluation = problem.evaluate_design(design, arguments)
            return constraint.function(evaluation, design, arguments)

        constraints.append(
            NonlinearConstraint(
                value,
                lower=constraint.lower,
                upper=constraint.upper,
                constraint_id=constraint.constraint_id,
            )
        )
    minimization = MinimizationProblem(
        objective,
        bounds=problem.bounds,
        constraints=tuple(constraints),
        problem_id=problem.problem_id,
    )
    solved = minimize(
        minimization,
        initial_design,
        method=ProjectedLBFGS() if method is None else method,
        termination=(
            OptimizationTermination(maximum_steps=300)
            if termination is None
            else termination
        ),
        args=args,
    )
    evaluation = problem.evaluate_design(solved.parameters, args)
    return ContinuousMemberSizingResult(solved, evaluation, problem.problem_id)


def select_catalog_member_sizing(
    labels: Sequence[str],
    evaluator: Callable[[int], MemberSizingEvaluation],
    /,
) -> CatalogMemberSizingResult:
    """Exact bounded section-catalog enumeration with deterministic tie-breaking."""
    labels_ = tuple(str(value) for value in labels)
    if not labels_ or not callable(evaluator):
        raise TypeError("A nonempty label catalog and evaluator are required.")
    evaluations = tuple(evaluator(index) for index in range(len(labels_)))
    if any(not isinstance(value, MemberSizingEvaluation) for value in evaluations):
        raise TypeError("Catalog evaluator must return MemberSizingEvaluation.")
    scores = jnp.stack(tuple(value.mass for value in evaluations))
    valid = jnp.stack(tuple(value.valid for value in evaluations))
    selected = int(jnp.argmin(jnp.where(valid, scores, jnp.inf)))
    successful = jnp.any(valid)
    return CatalogMemberSizingResult(
        selected,
        labels_[selected],
        evaluations[selected],
        scores,
        valid,
        successful,
    )


def verify_member_structure(
    *,
    equilibrium: MemberNetworkResult | None = None,
    prestress: PrestressRealizabilityResult | None = None,
    construction: ConstructionSequenceResult | None = None,
    sizing: MemberSizingEvaluation | None = None,
    local_buckling: LocalMemberBucklingResult | None = None,
    required: Sequence[str] = ("equilibrium",),
) -> StructuralVerificationResult:
    """Return CERTIFIED, FAILED, or INCOMPLETE over explicitly required evidence."""
    evidence = {
        "equilibrium": None if equilibrium is None else equilibrium.successful,
        "prestress": None if prestress is None else prestress.successful,
        "construction": None if construction is None else construction.successful,
        "sizing": None if sizing is None else sizing.valid,
        "local_buckling": (
            None
            if local_buckling is None
            else jnp.all(local_buckling.valid & (local_buckling.margin >= 0.0))
        ),
    }
    required_ = tuple(str(value) for value in required)
    if any(name not in evidence for name in required_):
        raise ValueError("Unknown required structural evidence name.")
    missing = tuple(name for name in required_ if evidence[name] is None)
    failed = tuple(
        name
        for name in required_
        if evidence[name] is not None and not bool(evidence[name])
    )
    verdict = (
        StructuralEvidenceVerdict.FAILED
        if failed
        else (
            StructuralEvidenceVerdict.INCOMPLETE
            if missing
            else StructuralEvidenceVerdict.CERTIFIED
        )
    )
    return StructuralVerificationResult(
        jnp.asarray(int(verdict), dtype=jnp.int32),
        equilibrium,
        prestress,
        construction,
        sizing,
        local_buckling,
        required_,
        failed,
        missing,
    )


__all__ = [
    "CatalogMemberSizingResult",
    "ContinuousMemberSizingProblem",
    "ContinuousMemberSizingResult",
    "MemberSizingConstraint",
    "MemberSizingEvaluation",
    "StructuralVerificationResult",
    "evaluate_member_sizing",
    "select_catalog_member_sizing",
    "solve_continuous_member_sizing",
    "verify_member_structure",
]
