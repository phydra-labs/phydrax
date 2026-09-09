#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax.random as jr

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain import (
    cover_integration_ownership,
    PointSampling,
    prepare_field_routing,
    PreparedFieldRouting,
    SubdomainCoverEvidence,
)
from .._functional_solver import FunctionalSolver
from ._problem import FunctionalDecompositionProblem, GlobalScope, PairScope, PatchScope
from ._strategy import (
    BlockDecompositionTraining,
    DecompositionTraining,
    JointDecompositionTraining,
    SchwarzDecompositionTraining,
)


def _strategy_payload(strategy: DecompositionTraining, /) -> dict[str, Any]:
    if isinstance(strategy, JointDecompositionTraining):
        return {"kind": "joint", "num_iterations": strategy.num_iterations}
    if isinstance(strategy, BlockDecompositionTraining):
        return {
            "kind": "block",
            "sweeps": strategy.sweeps,
            "inner_iterations": strategy.inner_iterations,
            "sweep": strategy.sweep,
            "active_patch_ids": strategy.active_patch_ids,
            "fixed_patch_ids": strategy.fixed_patch_ids,
        }
    if isinstance(strategy, SchwarzDecompositionTraining):
        return {
            "kind": "schwarz",
            "sweeps": strategy.sweeps,
            "inner_iterations": strategy.inner_iterations,
            "sweep": strategy.sweep,
            "relaxation": strategy.relaxation,
            "interface_tolerance": strategy.interface_tolerance,
        }
    raise TypeError("Unknown decomposition training strategy.")


class FunctionalDecompositionPlan(StrictModule, NonTrainableState):
    """Validated execution and certification policy for one decomposition problem."""

    training: DecompositionTraining
    required_window_regularity: int = eqx.field(static=True)
    certification_tolerance: float | None = eqx.field(static=True)
    trace_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        training: DecompositionTraining,
        /,
        *,
        required_window_regularity: int = 0,
        certification_tolerance: float | None = None,
        trace_points: int = 64,
    ):
        if not isinstance(
            training,
            (
                JointDecompositionTraining,
                BlockDecompositionTraining,
                SchwarzDecompositionTraining,
            ),
        ):
            raise TypeError("training has an invalid decomposition strategy type.")
        regularity = int(required_window_regularity)
        if regularity < 0:
            raise ValueError("required_window_regularity must be non-negative.")
        trace_points_ = int(trace_points)
        if trace_points_ <= 0:
            raise ValueError("trace_points must be positive.")
        if certification_tolerance is None:
            tolerance = None
        else:
            tolerance = float(certification_tolerance)
            if not math.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError(
                    "certification_tolerance must be finite and non-negative."
                )
        payload = {
            "kind": "functional-domain-decomposition-plan",
            "training": _strategy_payload(training),
            "required_window_regularity": regularity,
            "certification_tolerance": tolerance,
            "trace_points": trace_points_,
        }
        self.training = training
        self.required_window_regularity = regularity
        self.certification_tolerance = tolerance
        self.trace_points = trace_points_
        self.plan_id = canonical_fingerprint(payload)


class PreparedFunctionalDecomposition(StrictModule):
    """Prepared solver, topology evidence, and immutable execution identity."""

    problem: FunctionalDecompositionProblem
    plan: FunctionalDecompositionPlan
    solver: FunctionalSolver
    cover_evidence: SubdomainCoverEvidence
    routing: PreparedFieldRouting
    trace_batches: tuple[Any, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: FunctionalDecompositionProblem,
        plan: FunctionalDecompositionPlan,
        solver: FunctionalSolver,
        cover_evidence: SubdomainCoverEvidence,
        routing: PreparedFieldRouting,
        trace_batches: tuple[Any, ...],
        /,
    ):
        self.problem = problem
        self.plan = plan
        self.solver = solver
        self.cover_evidence = cover_evidence
        self.routing = routing
        self.trace_batches = tuple(trace_batches)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-functional-domain-decomposition",
                "cover_id": problem.cover.cover_id,
                "plan_id": plan.plan_id,
                "discretization_bundle_id": solver.discretization_bundle.bundle_id,
            }
        )


def _validate_scopes(problem: FunctionalDecompositionProblem, /) -> None:
    patch_ids = set(problem.cover.patch_ids)
    pairing_ids = set(problem.cover.pairing_ids)
    for scoped in (*problem.terms, *problem.evaluation_terms):
        scope = scoped.scope
        if isinstance(scope, PatchScope) and scope.patch_id not in patch_ids:
            raise ValueError(f"Term scope references unknown patch {scope.patch_id!r}.")
        if isinstance(scope, PairScope) and scope.pairing_id not in pairing_ids:
            raise ValueError(
                f"Term scope references unknown pairing {scope.pairing_id!r}."
            )
        if not isinstance(scope, (PatchScope, PairScope, GlobalScope)):
            raise TypeError("Unknown functional decomposition term scope.")


def _validate_windows(
    problem: FunctionalDecompositionProblem,
    plan: FunctionalDecompositionPlan,
    /,
) -> None:
    if problem.assembly != "partition-of-unity":
        return
    for patch in problem.cover.patches:
        window = patch.window
        if window is None:
            raise ValueError(
                "Partition-of-unity problems require a window on every patch."
            )
        regularity = int(window.metadata.get("regularity_order", 0))
        if regularity < plan.required_window_regularity:
            raise ValueError(
                f"Patch {patch.patch_id!r} window regularity {regularity} is below "
                f"the required order {plan.required_window_regularity}."
            )


def prepare_functional_decomposition(
    problem: FunctionalDecompositionProblem,
    plan: FunctionalDecompositionPlan,
    /,
    *,
    audit_points: Any = None,
) -> PreparedFunctionalDecomposition:
    """Validate and lower one native functional decomposition problem."""
    if not isinstance(problem, FunctionalDecompositionProblem):
        raise TypeError("problem must be a FunctionalDecompositionProblem.")
    if not isinstance(plan, FunctionalDecompositionPlan):
        raise TypeError("plan must be a FunctionalDecompositionPlan.")
    _validate_scopes(problem)
    _validate_windows(problem, plan)
    strategy = plan.training
    if isinstance(strategy, BlockDecompositionTraining):
        known = set(problem.cover.patch_ids)
        active = (
            known if strategy.active_patch_ids is None else set(strategy.active_patch_ids)
        )
        fixed = set(strategy.fixed_patch_ids)
        unknown = (active | fixed) - known
        if unknown:
            raise ValueError(f"Block schedule references unknown patches {unknown!r}.")
        if not (active - fixed):
            raise ValueError("Block schedule has no active non-fixed patches.")

    if isinstance(strategy, SchwarzDecompositionTraining):
        if problem.assembly != "broken":
            raise ValueError("Schwarz strategies require broken local fields.")
    if isinstance(strategy, SchwarzDecompositionTraining):
        if not problem.cover.pairings:
            raise ValueError("Schwarz training requires at least one paired support.")
        if not any(isinstance(value.scope, PairScope) for value in problem.terms):
            raise ValueError("Schwarz training requires at least one pair-scoped term.")

    if problem.cover.exact_coverage:
        evidence = problem.cover.structural_evidence()
    else:
        if audit_points is None:
            raise ValueError(
                "A non-exact cover requires audit_points during preparation."
            )
        evidence = problem.cover.audit(audit_points)
        if not evidence.verified:
            raise ValueError("Sampled subdomain coverage audit failed.")

    ownership_kind = "window" if problem.assembly == "partition-of-unity" else "support"
    routing = prepare_field_routing(
        problem.cover,
        ownership=cover_integration_ownership(
            problem.cover,
            kind=ownership_kind,
        ),
    )
    trace_keys = jr.split(problem.collocation_key, len(problem.cover.pairings) + 1)
    trace_batches = tuple(
        pairing.component.sample(
            PointSampling(plan.trace_points),
            key=trace_keys[index + 1],
        )
        for index, pairing in enumerate(problem.cover.pairings)
    )
    solver = FunctionalSolver(
        functions=problem.functions,
        terms=problem.training_terms,
        evaluation_terms=problem.diagnostic_terms,
        enforcement=problem.enforcement,
        collocation_key=problem.collocation_key,
    )
    return PreparedFunctionalDecomposition(
        problem,
        plan,
        solver,
        evidence,
        routing,
        trace_batches,
    )


__all__ = [
    "FunctionalDecompositionPlan",
    "PreparedFunctionalDecomposition",
    "prepare_functional_decomposition",
]
