#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import (
    AbstractStateDesignMethod,
    Bounds,
    OptimizationTermination,
    ReducedMMA,
    solve_state_design,
    StateAcceptancePolicy,
    StateDesignConstraint,
    StateDesignProblem,
    StateDesignResult,
)
from ._topology_design import (
    Aggregation,
    DensityTransform,
    LoadCase,
    MaterialInterpolation,
)
from ._topology_state import (
    BranchGateEvidence,
    FiniteElementStateSolver,
    NeuralVariationalStateSolver,
)


class TopologyMechanicsProblem(StrictModule, NonTrainableState):
    """FE-authoritative density topology problem over one or more mechanics loads."""

    state_residual: Callable = eqx.field(static=True)
    load_cases: tuple[LoadCase, ...]
    density_transform: DensityTransform
    material_interpolation: MaterialInterpolation
    aggregation: Aggregation
    volume_fraction: float = eqx.field(static=True)
    state_solver: FiniteElementStateSolver | NeuralVariationalStateSolver
    acceptance_policy: StateAcceptancePolicy
    branch_evaluator: Callable | None = eqx.field(static=True)
    state_realization: Callable | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_residual: Callable,
        load_cases: Sequence[LoadCase],
        density_transform: DensityTransform,
        material_interpolation: MaterialInterpolation,
        volume_fraction: float,
        state_solver: FiniteElementStateSolver | NeuralVariationalStateSolver,
        /,
        *,
        aggregation: Aggregation | None = None,
        acceptance_policy: StateAcceptancePolicy | None = None,
        branch_evaluator: Callable | None = None,
        state_realization: Callable | None = None,
        problem_id: str = "topology-mechanics",
    ):
        if not callable(state_residual):
            raise TypeError("state_residual must be callable.")
        cases = tuple(load_cases)
        if not cases or any(not isinstance(case, LoadCase) for case in cases):
            raise ValueError("load_cases must contain at least one LoadCase.")
        case_ids = tuple(case.case_id for case in cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Load-case identifiers must be unique.")
        if not isinstance(density_transform, DensityTransform):
            raise TypeError("density_transform must be a DensityTransform.")
        if not isinstance(material_interpolation, MaterialInterpolation):
            raise TypeError("material_interpolation must be MaterialInterpolation.")
        if not isinstance(
            state_solver, (FiniteElementStateSolver, NeuralVariationalStateSolver)
        ):
            raise TypeError(
                "state_solver must end in FiniteElementStateSolver, directly or through "
                "NeuralVariationalStateSolver."
            )
        aggregation_ = Aggregation() if aggregation is None else aggregation
        acceptance = (
            StateAcceptancePolicy() if acceptance_policy is None else acceptance_policy
        )
        if not isinstance(aggregation_, Aggregation):
            raise TypeError("aggregation must be Aggregation or None.")
        if not isinstance(acceptance, StateAcceptancePolicy):
            raise TypeError("acceptance_policy must be StateAcceptancePolicy or None.")
        if branch_evaluator is not None and not callable(branch_evaluator):
            raise TypeError("branch_evaluator must be callable or None.")
        if state_realization is not None and not callable(state_realization):
            raise TypeError("state_realization must be callable or None.")
        fraction = float(volume_fraction)
        identifier = str(problem_id)
        if not isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("volume_fraction must lie strictly between zero and one.")
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.state_residual = state_residual
        self.load_cases = cases
        self.density_transform = density_transform
        self.material_interpolation = material_interpolation
        self.aggregation = aggregation_
        self.volume_fraction = fraction
        self.state_solver = state_solver
        self.acceptance_policy = acceptance
        self.branch_evaluator = branch_evaluator
        self.state_realization = state_realization
        self.problem_id = identifier

    def physical_density(
        self,
        raw_density: ArrayLike,
        beta: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.density_transform.apply(raw_density, beta)

    def material_parameters(
        self,
        raw_density: ArrayLike,
        beta: ArrayLike | None = None,
        penalty: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.material_interpolation(
            self.physical_density(raw_density, beta),
            penalty,
        )

    def volume_ratio(
        self,
        raw_density: ArrayLike,
        beta: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.density_transform.volume_ratio(raw_density, beta)

    def load_values(
        self,
        states: PyTree[Any],
        raw_density: ArrayLike,
        /,
        *,
        beta: ArrayLike | None = None,
        penalty: ArrayLike | None = None,
        args: Any = None,
    ) -> Array:
        if not isinstance(states, tuple) or len(states) != len(self.load_cases):
            raise ValueError("states must contain one tuple entry per load case.")
        material = self.material_parameters(raw_density, beta, penalty)
        return jnp.stack(
            tuple(
                case.value(state, material, args)
                for state, case in zip(states, self.load_cases, strict=True)
            )
        )

    def as_state_design_problem(
        self,
        /,
        *,
        beta: ArrayLike | None = None,
        penalty: ArrayLike | None = None,
    ) -> StateDesignProblem:
        """Lower the mechanics problem to the canonical state/adjoint runtime."""

        selected_beta = self.density_transform.beta if beta is None else jnp.asarray(beta)
        selected_penalty = (
            self.material_interpolation.penalty
            if penalty is None
            else jnp.asarray(penalty)
        )

        def residual(states, density, dynamic_args):
            if not isinstance(states, tuple) or len(states) != len(self.load_cases):
                raise ValueError("states must contain one tuple entry per load case.")
            material = self.material_parameters(
                density,
                selected_beta,
                selected_penalty,
            )
            return tuple(
                self.state_residual(state, material, case, dynamic_args)
                for state, case in zip(states, self.load_cases, strict=True)
            )

        def objective(states, density, dynamic_args):
            values = self.load_values(
                states,
                density,
                beta=selected_beta,
                penalty=selected_penalty,
                args=dynamic_args,
            )
            weights = jnp.asarray(
                tuple(case.weight for case in self.load_cases),
                dtype=values.dtype,
            )
            return self.aggregation(values, weights), values

        def admissibility(states, density, dynamic_args):
            if self.branch_evaluator is None:
                return jnp.asarray(True)
            physical = self.physical_density(density, selected_beta)
            evidence = tuple(
                self.branch_evaluator(state, physical, case, dynamic_args)
                for state, case in zip(states, self.load_cases, strict=True)
            )
            if any(not isinstance(item, BranchGateEvidence) for item in evidence):
                raise TypeError("branch_evaluator must return BranchGateEvidence.")
            return jnp.all(jnp.stack(tuple(item.accepted for item in evidence)))

        def realization(states, density, dynamic_args):
            if self.state_realization is None:
                return jnp.asarray(True)
            physical = self.physical_density(density, selected_beta)
            return self.state_realization(states, physical, dynamic_args)

        volume = StateDesignConstraint(
            lambda states, density, dynamic_args: self.volume_ratio(
                density,
                selected_beta,
            ),
            upper=self.volume_fraction,
            constraint_id=f"{self.problem_id}/volume",
            depends_on_state=False,
        )
        return StateDesignProblem(
            residual,
            objective,
            state_solver=self.state_solver,
            acceptance_policy=self.acceptance_policy,
            state_admissibility=admissibility,
            state_realization=realization,
            design_bounds=Bounds(0.0, 1.0),
            constraints=(volume,),
            has_aux=True,
            problem_id=self.problem_id,
        )


class TopologyContinuationStage(StrictModule, NonTrainableState):
    """One finite projection/material-penalty stage in topology continuation."""

    beta: Array
    penalty: Array | None
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: ArrayLike,
        /,
        *,
        penalty: ArrayLike | None = None,
        stage_id: str,
    ):
        beta_ = np.asarray(beta)
        penalty_ = None if penalty is None else np.asarray(penalty)
        identifier = str(stage_id)
        if (
            beta_.shape != ()
            or not np.issubdtype(beta_.dtype, np.number)
            or np.issubdtype(beta_.dtype, np.complexfloating)
        ):
            raise TypeError("beta must be one real scalar array.")
        if not isfinite(float(beta_)) or float(beta_) <= 0.0:
            raise ValueError("beta must be finite and strictly positive.")
        if penalty_ is not None and (
            penalty_.shape != ()
            or not np.issubdtype(penalty_.dtype, np.number)
            or np.issubdtype(penalty_.dtype, np.complexfloating)
        ):
            raise TypeError("penalty must be one real scalar array or None.")
        if penalty_ is not None and (
            not isfinite(float(penalty_)) or float(penalty_) <= 0.0
        ):
            raise ValueError("penalty must be finite and positive.")
        if not identifier:
            raise ValueError("stage_id must be non-empty.")
        self.beta = jnp.asarray(float(beta_))
        self.penalty = None if penalty_ is None else jnp.asarray(float(penalty_))
        self.stage_id = identifier


class TopologyContinuationSchedule(StrictModule, NonTrainableState):
    """Monotone finite continuation schedule with stable stage identities."""

    stages: tuple[TopologyContinuationStage, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        stages: Sequence[TopologyContinuationStage],
        /,
        *,
        schedule_id: str = "topology-continuation",
    ):
        stages_ = tuple(stages)
        if not stages_ or any(
            not isinstance(stage, TopologyContinuationStage) for stage in stages_
        ):
            raise ValueError("stages must contain TopologyContinuationStage values.")
        identifiers = tuple(stage.stage_id for stage in stages_)
        betas = tuple(float(stage.beta) for stage in stages_)
        penalties = tuple(
            None if stage.penalty is None else float(stage.penalty) for stage in stages_
        )
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Topology continuation stage identifiers must be unique.")
        if any(right < left for left, right in zip(betas, betas[1:])):
            raise ValueError("Topology continuation beta values must be nondecreasing.")
        explicit_penalties = tuple(value for value in penalties if value is not None)
        if any(
            right < left
            for left, right in zip(
                explicit_penalties,
                explicit_penalties[1:],
            )
        ):
            raise ValueError("Explicit material penalties must be nondecreasing.")
        identifier = str(schedule_id)
        if not identifier:
            raise ValueError("schedule_id must be non-empty.")
        self.stages = stages_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "topology-continuation-schedule",
                "identity": identifier,
                "stages": [
                    {
                        "id": stage.stage_id,
                        "beta": float(stage.beta),
                        "penalty": (
                            None if stage.penalty is None else float(stage.penalty)
                        ),
                    }
                    for stage in stages_
                ],
            }
        )


class TopologyContinuationStageEvidence(StrictModule):
    """Acceptance and exact warm-state rollback decision for one stage."""

    accepted: Array
    rollback_applied: Array
    status: Array
    state_accepted: Array
    adjoint_accepted: Array
    stage_id: str = eqx.field(static=True)


class TopologyOptimizationResult(StrictModule):
    """Accepted topology design with density views and continuation evidence."""

    state_design: StateDesignResult
    stage_results: tuple[StateDesignResult, ...]
    continuation_evidence: tuple[TopologyContinuationStageEvidence, ...]
    raw_density: Array
    filtered_density: Array
    physical_density: Array
    material_parameters: Array
    load_values: Array
    volume_ratio: Array
    material_measure: Array
    domain_measure: Array
    beta: Array
    penalty: Array
    continuation_completed: Array
    problem_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.state_design.successful & self.continuation_completed


def solve_topology_optimization(
    problem: TopologyMechanicsProblem,
    initial_states: PyTree[Any],
    initial_density: ArrayLike,
    /,
    *,
    schedule: TopologyContinuationSchedule | None = None,
    method: AbstractStateDesignMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> TopologyOptimizationResult:
    """Solve finite continuation stages with exact accepted-stage warm rollback."""

    if not isinstance(problem, TopologyMechanicsProblem):
        raise TypeError("problem must be a TopologyMechanicsProblem.")
    schedule_ = (
        TopologyContinuationSchedule(
            (
                TopologyContinuationStage(
                    problem.density_transform.beta,
                    penalty=problem.material_interpolation.penalty,
                    stage_id="target",
                ),
            )
        )
        if schedule is None
        else schedule
    )
    if not isinstance(schedule_, TopologyContinuationSchedule):
        raise TypeError("schedule must be TopologyContinuationSchedule or None.")
    method_ = ReducedMMA() if method is None else method
    if not isinstance(method_, AbstractStateDesignMethod):
        raise TypeError("method must be an AbstractStateDesignMethod or None.")
    termination_ = (
        OptimizationTermination(maximum_steps=200) if termination is None else termination
    )
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")

    warm_states = initial_states
    warm_density = jnp.asarray(initial_density)
    stage_results: list[StateDesignResult] = []
    evidence: list[TopologyContinuationStageEvidence] = []
    accepted_result: StateDesignResult | None = None
    accepted_stage: TopologyContinuationStage | None = None

    for stage in schedule_.stages:
        stage_problem = problem.as_state_design_problem(
            beta=stage.beta,
            penalty=stage.penalty,
        )
        candidate = solve_state_design(
            stage_problem,
            warm_states,
            warm_density,
            method=method_,
            termination=termination_,
            args=args,
        )
        stage_results.append(candidate)
        adjoint_accepted = (
            jnp.asarray(False)
            if candidate.adjoint_acceptance is None
            else candidate.adjoint_acceptance.accepted
        )
        stage_accepted = (
            candidate.successful & candidate.state_acceptance.accepted & adjoint_accepted
        )
        accepted = bool(np.asarray(stage_accepted))
        rollback = not accepted and accepted_result is not None
        evidence.append(
            TopologyContinuationStageEvidence(
                stage_accepted,
                jnp.asarray(rollback),
                candidate.status,
                candidate.state_acceptance.accepted,
                adjoint_accepted,
                stage.stage_id,
            )
        )
        if not accepted:
            if accepted_result is None:
                accepted_result = candidate
                accepted_stage = stage
            break
        accepted_result = candidate
        accepted_stage = stage
        warm_states = candidate.state
        warm_density = candidate.design

    if accepted_result is None or accepted_stage is None:
        raise RuntimeError("A non-empty topology continuation produced no stage result.")
    completed = len(stage_results) == len(schedule_.stages) and all(
        bool(np.asarray(item.accepted)) for item in evidence
    )
    penalty = (
        problem.material_interpolation.penalty
        if accepted_stage.penalty is None
        else accepted_stage.penalty
    )
    raw = jnp.asarray(accepted_result.design)
    filtered = problem.density_transform.filtered(raw)
    physical = problem.physical_density(raw, accepted_stage.beta)
    material = problem.material_interpolation(physical, penalty)
    values = problem.load_values(
        accepted_result.state,
        raw,
        beta=accepted_stage.beta,
        penalty=penalty,
        args=args,
    )
    return TopologyOptimizationResult(
        accepted_result,
        tuple(stage_results),
        tuple(evidence),
        raw,
        filtered,
        physical,
        material,
        values,
        problem.volume_ratio(raw, accepted_stage.beta),
        jnp.sum(physical * problem.density_transform.measures),
        jnp.sum(problem.density_transform.measures),
        accepted_stage.beta,
        penalty,
        jnp.asarray(completed),
        problem.problem_id,
    )


__all__ = [
    "TopologyContinuationSchedule",
    "TopologyContinuationStage",
    "TopologyContinuationStageEvidence",
    "TopologyMechanicsProblem",
    "TopologyOptimizationResult",
    "solve_topology_optimization",
]
