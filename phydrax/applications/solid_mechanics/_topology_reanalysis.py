#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import LinearSolveStatus
from ...optim import OptimizationStatus
from ._topology import TopologyMechanicsProblem, TopologyOptimizationResult
from ._topology_state import certify_state_adjoint, StateAdjointEvidence


class DensityTransferCandidate(StrictModule):
    """Reference-mesh raw density proposed by an explicit transfer operation."""

    raw_density: Array

    def __init__(self, raw_density: ArrayLike, /):
        density = jnp.asarray(raw_density)
        if density.ndim != 1 or not jnp.issubdtype(density.dtype, jnp.floating):
            raise TypeError("Transferred raw density must be one real inexact vector.")
        self.raw_density = density


class DensityTransferEvidence(StrictModule):
    """Independent material-measure audit of a reference-mesh transfer."""

    raw_density: Array
    physical_density: Array
    source_material_measure: Array
    reference_material_measure: Array
    relative_measure_error: Array
    tolerance: Array
    finite: Array
    accepted: Array


class FiniteElementReanalysisCandidate(StrictModule):
    """Executed reference FE primal and transpose roots before recertification."""

    state: PyTree[Array]
    adjoint: PyTree[Array]
    state_status: Array
    adjoint_status: Array
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: PyTree[Any],
        adjoint: PyTree[Any],
        /,
        *,
        state_status: Any = OptimizationStatus.SUCCESS,
        adjoint_status: Any = LinearSolveStatus.SUCCESS,
        solver_id: str,
    ):
        state_ = jax.tree.map(jnp.asarray, state)
        adjoint_ = jax.tree.map(jnp.asarray, adjoint)
        if not jax.tree.leaves(state_) or not jax.tree.leaves(adjoint_):
            raise ValueError("Reference state and adjoint must contain array leaves.")
        if any(
            not jnp.issubdtype(value.dtype, jnp.floating)
            for value in (*jax.tree.leaves(state_), *jax.tree.leaves(adjoint_))
        ):
            raise TypeError(
                "Reference state and adjoint leaves must be real inexact arrays."
            )
        state_status_ = jnp.asarray(state_status, dtype=jnp.int32)
        adjoint_status_ = jnp.asarray(adjoint_status, dtype=jnp.int32)
        identifier = str(solver_id)
        if state_status_.shape != () or adjoint_status_.shape != ():
            raise ValueError("Reference primal and adjoint statuses must be scalar.")
        if not identifier:
            raise ValueError("solver_id must be non-empty.")
        self.state = state_
        self.adjoint = adjoint_
        self.state_status = state_status_
        self.adjoint_status = adjoint_status_
        self.solver_id = identifier


class TopologyReanalysisPlan(StrictModule, NonTrainableState):
    """Independent transfer and mandatory final reference-FE reanalysis route."""

    reference_problem: TopologyMechanicsProblem
    transfer_function: Callable = eqx.field(static=True)
    finite_element_solve: Callable = eqx.field(static=True)
    proposal_function: Callable | None = eqx.field(static=True)
    beta: Array
    penalty: Array
    transfer_tolerance: float = eqx.field(static=True)
    uniform_source_objective: Array | None
    uniform_reference_objective: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_problem: TopologyMechanicsProblem,
        transfer: Callable,
        finite_element_solve: Callable,
        /,
        *,
        proposal: Callable | None = None,
        beta: ArrayLike | None = None,
        penalty: ArrayLike | None = None,
        transfer_tolerance: float = 1.0e-6,
        uniform_source_objective: ArrayLike | None = None,
        uniform_reference_objective: ArrayLike | None = None,
        plan_id: str = "topology-fe-reanalysis",
    ):
        if not isinstance(reference_problem, TopologyMechanicsProblem):
            raise TypeError("reference_problem must be TopologyMechanicsProblem.")
        if not callable(transfer) or not callable(finite_element_solve):
            raise TypeError("transfer and finite_element_solve must be callable.")
        if proposal is not None and not callable(proposal):
            raise TypeError("proposal must be callable or None.")
        beta_ = (
            reference_problem.density_transform.beta
            if beta is None
            else jnp.asarray(beta)
        )
        penalty_ = (
            reference_problem.material_interpolation.penalty
            if penalty is None
            else jnp.asarray(penalty)
        )
        if (
            not jnp.issubdtype(beta_.dtype, jnp.number)
            or jnp.issubdtype(beta_.dtype, jnp.complexfloating)
            or not jnp.issubdtype(penalty_.dtype, jnp.number)
            or jnp.issubdtype(penalty_.dtype, jnp.complexfloating)
        ):
            raise TypeError("Reanalysis beta and penalty must be real-valued.")
        if (
            beta_.shape != ()
            or penalty_.shape != ()
            or not isfinite(float(beta_))
            or float(beta_) <= 0.0
            or not isfinite(float(penalty_))
            or float(penalty_) <= 0.0
        ):
            raise ValueError(
                "Reanalysis beta and penalty must be finite positive scalar arrays."
            )
        tolerance = float(transfer_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("transfer_tolerance must be finite and non-negative.")
        source_uniform = (
            None
            if uniform_source_objective is None
            else jnp.asarray(uniform_source_objective)
        )
        reference_uniform = (
            None
            if uniform_reference_objective is None
            else jnp.asarray(uniform_reference_objective)
        )
        if (source_uniform is None) != (reference_uniform is None):
            raise ValueError("Uniform discretization controls must be supplied together.")
        if source_uniform is not None and (
            source_uniform.shape != ()
            or reference_uniform.shape != ()
            or not jnp.issubdtype(source_uniform.dtype, jnp.floating)
            or not jnp.issubdtype(reference_uniform.dtype, jnp.floating)
        ):
            raise TypeError("Uniform discretization controls must be real scalar arrays.")
        if source_uniform is not None and (
            not isfinite(float(source_uniform))
            or float(source_uniform) <= 0.0
            or not isfinite(float(reference_uniform))
            or float(reference_uniform) <= 0.0
        ):
            raise ValueError(
                "Uniform discretization controls must be finite and strictly positive."
            )
        identifier = str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.reference_problem = reference_problem
        self.transfer_function = transfer
        self.finite_element_solve = finite_element_solve
        self.proposal_function = proposal
        self.beta = beta_
        self.penalty = penalty_
        self.transfer_tolerance = tolerance
        self.uniform_source_objective = source_uniform
        self.uniform_reference_objective = reference_uniform
        self.plan_id = identifier

    def transfer(
        self,
        source_physical_density: Array,
        source_material_measure: Array,
        /,
    ) -> DensityTransferEvidence:
        candidate = self.transfer_function(source_physical_density)
        if not isinstance(candidate, DensityTransferCandidate):
            raise TypeError("transfer must return DensityTransferCandidate.")
        physical = self.reference_problem.physical_density(
            candidate.raw_density, self.beta
        )
        measures = self.reference_problem.density_transform.measures
        reference_measure = jnp.sum(physical * measures)
        source_measure = jnp.asarray(source_material_measure)
        tiny = jnp.asarray(jnp.finfo(reference_measure.dtype).tiny)
        error = jnp.abs(reference_measure - source_measure) / jnp.maximum(
            jnp.abs(source_measure), tiny
        )
        finite = (
            jnp.all(jnp.isfinite(candidate.raw_density))
            & jnp.all((candidate.raw_density >= 0.0) & (candidate.raw_density <= 1.0))
            & jnp.all(jnp.isfinite(physical))
            & jnp.isfinite(reference_measure)
            & jnp.isfinite(source_measure)
            & jnp.isfinite(error)
        )
        accepted = finite & (error <= self.transfer_tolerance)
        return DensityTransferEvidence(
            candidate.raw_density,
            physical,
            source_measure,
            reference_measure,
            error,
            jnp.asarray(self.transfer_tolerance, dtype=error.dtype),
            finite,
            accepted,
        )


class TopologyReanalysisEvidence(StrictModule):
    """Transfer, primal, transpose, and final-FE authority evidence."""

    transfer: DensityTransferEvidence
    mechanics: StateAdjointEvidence
    source_optimization_accepted: Array
    learned_proposal_used: Array
    final_fe_reanalysis: Array
    accepted: Array
    solver_id: str = eqx.field(static=True)


class TopologyReanalysisReport(StrictModule):
    """Independent reference-mesh objective and discretization-control report."""

    optimized_objective: Array
    reference_objective: Array
    reference_load_values: Array
    objective_ratio: Array
    uniform_source_objective: Array | None
    uniform_reference_objective: Array | None
    discretization_ratio: Array | None
    excess_stiffness_overreport: Array | None
    evidence: TopologyReanalysisEvidence
    reference_state: PyTree[Array]
    reference_adjoint: PyTree[Array]
    plan_id: str = eqx.field(static=True)

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted


def reanalyse_topology_design(
    result: TopologyOptimizationResult,
    plan: TopologyReanalysisPlan,
    initial_reference_state: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> TopologyReanalysisReport:
    """Transfer and recertify a design using mandatory final FE primal/adjoint roots."""

    if not isinstance(result, TopologyOptimizationResult):
        raise TypeError("result must be TopologyOptimizationResult.")
    if not isinstance(plan, TopologyReanalysisPlan):
        raise TypeError("plan must be TopologyReanalysisPlan.")
    transfer = plan.transfer(result.physical_density, result.material_measure)
    selected_initial = (
        initial_reference_state
        if plan.proposal_function is None
        else plan.proposal_function(
            plan.reference_problem,
            transfer.raw_density,
            initial_reference_state,
            args,
        )
    )
    selected_initial = jax.tree.map(jnp.asarray, selected_initial)
    state_problem = plan.reference_problem.as_state_design_problem(
        beta=plan.beta,
        penalty=plan.penalty,
    )
    candidate = plan.finite_element_solve(
        state_problem,
        transfer.raw_density,
        selected_initial,
        args,
    )
    if not isinstance(candidate, FiniteElementReanalysisCandidate):
        raise TypeError(
            "finite_element_solve must return FiniteElementReanalysisCandidate."
        )
    mechanics = certify_state_adjoint(
        state_problem,
        candidate.state,
        transfer.raw_density,
        candidate.adjoint,
        reference_state=selected_initial,
        state_status=candidate.state_status,
        adjoint_status=candidate.adjoint_status,
        args=args,
    )
    reference_objective, reference_values = state_problem.value(
        candidate.state,
        transfer.raw_density,
        args,
    )
    optimized = jnp.asarray(result.state_design.objective)
    objective_ratio = reference_objective / optimized
    if plan.uniform_source_objective is None:
        discretization_ratio = None
        excess = None
    else:
        discretization_ratio = (
            plan.uniform_reference_objective / plan.uniform_source_objective
        )
        excess = objective_ratio / discretization_ratio - 1.0
    final_fe = jnp.asarray(True)
    source_accepted = result.successful
    evidence = TopologyReanalysisEvidence(
        transfer,
        mechanics,
        source_accepted,
        jnp.asarray(plan.proposal_function is not None),
        final_fe,
        source_accepted & transfer.accepted & mechanics.accepted & final_fe,
        candidate.solver_id,
    )
    return TopologyReanalysisReport(
        optimized,
        reference_objective,
        jnp.asarray(reference_values),
        objective_ratio,
        plan.uniform_source_objective,
        plan.uniform_reference_objective,
        discretization_ratio,
        excess,
        evidence,
        candidate.state,
        candidate.adjoint,
        plan.plan_id,
    )


__all__ = [
    "DensityTransferCandidate",
    "DensityTransferEvidence",
    "FiniteElementReanalysisCandidate",
    "TopologyReanalysisEvidence",
    "TopologyReanalysisPlan",
    "TopologyReanalysisReport",
    "reanalyse_topology_design",
]
