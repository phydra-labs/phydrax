#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Deterministic subsystem-specific cardiovascular inverse adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....optim import (
    AbstractStateDesignMethod,
    AbstractStateSolver,
    OptimizationTermination,
    solve_state_design,
    StateAcceptancePolicy,
    StateDesignProblem,
    StateDesignResult,
)
from ._likelihood import (
    MultimodalLikelihoodResult,
    PreparedMultimodalLikelihood,
)
from ._parameters import CardiacParameterSchema, CardiacSubsystem


class ElectrophysiologyInverseRoute(StrictModule, NonTrainableState):
    """Activation/repolarization parameter route on a fixed topology."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = "electrophysiology-inverse"

    @property
    def accepted_subsystems(self) -> frozenset[CardiacSubsystem]:
        return frozenset((CardiacSubsystem.ELECTROPHYSIOLOGY,))


class MechanicsInverseRoute(StrictModule, NonTrainableState):
    """Passive/active constitutive parameter route on an accepted mechanics solve."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = "mechanics-inverse"

    @property
    def accepted_subsystems(self) -> frozenset[CardiacSubsystem]:
        return frozenset(
            (CardiacSubsystem.PASSIVE_MECHANICS, CardiacSubsystem.ACTIVE_MECHANICS)
        )


class LoadingInverseRoute(StrictModule, NonTrainableState):
    """Loading/circulation parameter route with geometry and material held fixed."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = "loading-inverse"

    @property
    def accepted_subsystems(self) -> frozenset[CardiacSubsystem]:
        return frozenset((CardiacSubsystem.LOADING, CardiacSubsystem.CIRCULATION))


class UnloadedGeometryInverseRoute(StrictModule, NonTrainableState):
    """Reference-configuration route separated from constitutive calibration."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = "unloaded-geometry-inverse"

    @property
    def accepted_subsystems(self) -> frozenset[CardiacSubsystem]:
        return frozenset((CardiacSubsystem.UNLOADED_GEOMETRY,))


InverseRoute = (
    ElectrophysiologyInverseRoute
    | MechanicsInverseRoute
    | LoadingInverseRoute
    | UnloadedGeometryInverseRoute
)


class InverseObjectiveEvaluation(StrictModule):
    """MAP objective components at one realized state/design pair."""

    physical_parameters: tuple[Array, ...]
    likelihood: MultimodalLikelihoodResult
    log_parameter_prior: Array
    negative_log_posterior: Array
    fixed_topology: Array
    finite: Array
    successful: Array


class InverseAcceptanceEvidence(StrictModule):
    """Fail-closed state, adjoint, likelihood, and topology evidence."""

    state_accepted: Array
    adjoint_accepted: Array
    likelihood_accepted: Array
    fixed_topology: Array
    finite: Array
    successful: Array


class CardiovascularInverseResult(StrictModule):
    """One accepted candidate retaining the native state-design result."""

    state_design: StateDesignResult
    physical_parameters: tuple[Array, ...]
    objective_evaluation: InverseObjectiveEvaluation
    evidence: InverseAcceptanceEvidence
    problem_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class CardiovascularMultiStartResult(StrictModule):
    """Deterministic explicit-start ensemble with no hidden start generation."""

    results: tuple[CardiovascularInverseResult, ...]
    objectives: Array
    accepted: Array
    best_index: Array
    best: CardiovascularInverseResult
    successful: Array
    problem_id: str = eqx.field(static=True)


class _DeterministicInverseProblem(StrictModule, NonTrainableState):
    __strict_abstract__ = True

    schema: CardiacParameterSchema
    likelihood: PreparedMultimodalLikelihood
    route: InverseRoute
    state_residual: Any
    forward_observables: Any
    nuisance_values: Any | None
    fixed_topology: Any
    state_solver: AbstractStateSolver | None
    acceptance_policy: StateAcceptancePolicy | None
    state_admissibility: Any | None
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: CardiacParameterSchema,
        likelihood: PreparedMultimodalLikelihood,
        route: InverseRoute,
        state_residual: Callable[
            [PyTree[Any], tuple[Array, ...], Any], PyTree[ArrayLike]
        ],
        forward_observables: Callable[
            [PyTree[Any], tuple[Array, ...], Any], Sequence[ArrayLike]
        ],
        /,
        *,
        fixed_topology: Callable[[PyTree[Any], tuple[Array, ...], Any], ArrayLike],
        nuisance_values: Callable[
            [PyTree[Any], tuple[Array, ...], Any], Sequence[ArrayLike | None]
        ]
        | None = None,
        state_solver: AbstractStateSolver | None = None,
        acceptance_policy: StateAcceptancePolicy | None = None,
        state_admissibility: Callable[[PyTree[Any], tuple[Array, ...], Any], ArrayLike]
        | None = None,
        problem_id: str | None = None,
    ):
        if not isinstance(schema, CardiacParameterSchema):
            raise TypeError("schema must be a CardiacParameterSchema.")
        if not isinstance(likelihood, PreparedMultimodalLikelihood):
            raise TypeError("likelihood must be a PreparedMultimodalLikelihood.")
        if not isinstance(
            route,
            (
                ElectrophysiologyInverseRoute,
                MechanicsInverseRoute,
                LoadingInverseRoute,
                UnloadedGeometryInverseRoute,
            ),
        ):
            raise TypeError("route must be a concrete cardiovascular inverse route.")
        if not callable(state_residual) or not callable(forward_observables):
            raise TypeError("state_residual and forward_observables must be callable.")
        if not callable(fixed_topology):
            raise TypeError("fixed_topology evidence must be callable.")
        if nuisance_values is not None and not callable(nuisance_values):
            raise TypeError("nuisance_values must be callable or None.")
        if state_solver is not None and not isinstance(state_solver, AbstractStateSolver):
            raise TypeError("state_solver must be an AbstractStateSolver or None.")
        if acceptance_policy is not None and not isinstance(
            acceptance_policy, StateAcceptancePolicy
        ):
            raise TypeError("acceptance_policy must be a StateAcceptancePolicy or None.")
        if state_admissibility is not None and not callable(state_admissibility):
            raise TypeError("state_admissibility must be callable or None.")
        disallowed = schema.subsystems - route.accepted_subsystems
        if disallowed:
            names = sorted(subsystem.value for subsystem in disallowed)
            raise ValueError(
                f"{route.route_id} cannot own parameters from subsystems {names}."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-deterministic-inverse",
                    "route": route.route_id,
                    "schema": schema.schema_id,
                    "likelihood": likelihood.runtime_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.schema = schema
        self.likelihood = likelihood
        self.route = route
        self.state_residual = state_residual
        self.forward_observables = forward_observables
        self.nuisance_values = nuisance_values
        self.fixed_topology = fixed_topology
        self.state_solver = state_solver
        self.acceptance_policy = acceptance_policy
        self.state_admissibility = state_admissibility
        self.problem_id = identifier

    def _physical(
        self,
        raw_parameters: Sequence[ArrayLike],
        fixed_physical: Sequence[ArrayLike],
        /,
    ) -> tuple[Array, ...]:
        return self.schema.constrain_optimization(raw_parameters, fixed_physical)

    def objective_evaluation(
        self,
        state: PyTree[Any],
        raw_parameters: Sequence[ArrayLike],
        args: Any = None,
        /,
        *,
        fixed_physical: Sequence[ArrayLike],
    ) -> InverseObjectiveEvaluation:
        physical = self._physical(raw_parameters, fixed_physical)
        predictions = tuple(self.forward_observables(state, physical, args))
        nuisance = (
            None
            if self.nuisance_values is None
            else tuple(self.nuisance_values(state, physical, args))
        )
        likelihood = self.likelihood.evaluate(predictions, nuisance_values=nuisance)
        parameter_prior = self.schema.log_prior(physical)
        fixed = jnp.all(
            jnp.asarray(self.fixed_topology(state, physical, args), dtype=bool)
        )
        supported = self.schema.contains(physical)
        finite = (
            likelihood.finite
            & jnp.isfinite(parameter_prior)
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in physical))
            )
        )
        successful = likelihood.successful & supported & fixed & finite
        negative_log_posterior = jnp.where(
            successful,
            -(likelihood.log_density + parameter_prior),
            jnp.inf,
        )
        return InverseObjectiveEvaluation(
            physical_parameters=physical,
            likelihood=likelihood,
            log_parameter_prior=parameter_prior,
            negative_log_posterior=negative_log_posterior,
            fixed_topology=fixed,
            finite=finite,
            successful=successful,
        )

    def as_state_design_problem(
        self,
        initial_physical: Sequence[ArrayLike],
        /,
    ) -> tuple[StateDesignProblem, tuple[Array, ...]]:
        """Lower to the authoritative native state/adjoint design contract."""

        physical_reference = self.schema.validate_physical(initial_physical)
        parameter_space = self.schema.parameter_space(physical_reference)
        initial_raw = tuple(parameter_space.initial)

        def residual(state, raw_parameters, dynamic_args):
            physical = self._physical(raw_parameters, physical_reference)
            return self.state_residual(state, physical, dynamic_args)

        def objective(state, raw_parameters, dynamic_args):
            evaluation = self.objective_evaluation(
                state,
                raw_parameters,
                dynamic_args,
                fixed_physical=physical_reference,
            )
            return evaluation.negative_log_posterior, evaluation

        def realization(state, raw_parameters, dynamic_args):
            physical = self._physical(raw_parameters, physical_reference)
            return self.fixed_topology(state, physical, dynamic_args)

        admissibility = None
        if self.state_admissibility is not None:

            def admissibility(state, raw_parameters, dynamic_args):
                physical = self._physical(raw_parameters, physical_reference)
                return self.state_admissibility(state, physical, dynamic_args)

        problem = StateDesignProblem(
            residual,
            objective,
            state_solver=self.state_solver,
            acceptance_policy=self.acceptance_policy,
            state_admissibility=admissibility,
            state_realization=realization,
            has_aux=True,
            problem_id=self.problem_id,
        )
        return problem, initial_raw

    def solve(
        self,
        initial_state: PyTree[Any],
        initial_physical: Sequence[ArrayLike],
        /,
        *,
        method: AbstractStateDesignMethod,
        termination: OptimizationTermination | None = None,
        args: Any = None,
    ) -> CardiovascularInverseResult:
        problem, initial_raw = self.as_state_design_problem(initial_physical)
        result = solve_state_design(
            problem,
            initial_state,
            initial_raw,
            method=method,
            termination=termination,
            args=args,
        )
        physical_reference = self.schema.validate_physical(initial_physical)
        evaluation = self.objective_evaluation(
            result.state,
            result.design,
            args,
            fixed_physical=physical_reference,
        )
        adjoint_accepted = (
            jnp.asarray(False)
            if result.adjoint_acceptance is None
            else result.adjoint_acceptance.accepted
        )
        finite = (
            evaluation.finite
            & jnp.isfinite(result.objective)
            & jnp.all(
                jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in result.design))
            )
        )
        successful = (
            result.successful
            & result.state_acceptance.accepted
            & adjoint_accepted
            & evaluation.successful
            & finite
        )
        evidence = InverseAcceptanceEvidence(
            state_accepted=result.state_acceptance.accepted,
            adjoint_accepted=adjoint_accepted,
            likelihood_accepted=evaluation.likelihood.successful,
            fixed_topology=evaluation.fixed_topology,
            finite=finite,
            successful=successful,
        )
        return CardiovascularInverseResult(
            state_design=result,
            physical_parameters=evaluation.physical_parameters,
            objective_evaluation=evaluation,
            evidence=evidence,
            problem_id=self.problem_id,
            route_id=self.route.route_id,
        )

    def solve_multistart(
        self,
        initial_state: PyTree[Any],
        initial_physical_starts: Sequence[Sequence[ArrayLike]],
        /,
        *,
        method: AbstractStateDesignMethod,
        termination: OptimizationTermination | None = None,
        args: Any = None,
    ) -> CardiovascularMultiStartResult:
        """Run stable explicit starts through the same state/adjoint acceptance path."""

        raw_starts = tuple(tuple(start) for start in initial_physical_starts)
        if not raw_starts:
            raise ValueError("At least one physical multi-start point is required.")
        starts = tuple(self.schema.validate_physical(start) for start in raw_starts)
        reference = starts[0]
        for start in starts[1:]:
            if any(
                not bool(jnp.array_equal(start[index], reference[index]))
                for index in self.schema.fixed_indices
            ):
                raise ValueError(
                    "FIXED parameter values must be identical across every multi-start."
                )
        results = tuple(
            self.solve(
                initial_state,
                start,
                method=method,
                termination=termination,
                args=args,
            )
            for start in starts
        )
        objectives = jnp.stack(tuple(result.state_design.objective for result in results))
        accepted = jnp.stack(tuple(result.successful for result in results))
        finite_objectives = jnp.where(jnp.isfinite(objectives), objectives, jnp.inf)
        score = jnp.where(accepted, finite_objectives, finite_objectives + 1.0e12)
        best_index = jnp.argmin(score).astype(jnp.int32)
        best = results[int(best_index)]
        return CardiovascularMultiStartResult(
            results=results,
            objectives=objectives,
            accepted=accepted,
            best_index=best_index,
            best=best,
            successful=jnp.any(accepted),
            problem_id=self.problem_id,
        )


class ElectrophysiologyInverseProblem(_DeterministicInverseProblem):
    def __init__(
        self,
        schema: CardiacParameterSchema,
        likelihood: PreparedMultimodalLikelihood,
        state_residual: Callable,
        forward_observables: Callable,
        /,
        **kwargs: Any,
    ):
        super().__init__(
            schema,
            likelihood,
            ElectrophysiologyInverseRoute(),
            state_residual,
            forward_observables,
            **kwargs,
        )


class MechanicsInverseProblem(_DeterministicInverseProblem):
    def __init__(
        self,
        schema: CardiacParameterSchema,
        likelihood: PreparedMultimodalLikelihood,
        state_residual: Callable,
        forward_observables: Callable,
        /,
        **kwargs: Any,
    ):
        super().__init__(
            schema,
            likelihood,
            MechanicsInverseRoute(),
            state_residual,
            forward_observables,
            **kwargs,
        )


class LoadingInverseProblem(_DeterministicInverseProblem):
    def __init__(
        self,
        schema: CardiacParameterSchema,
        likelihood: PreparedMultimodalLikelihood,
        state_residual: Callable,
        forward_observables: Callable,
        /,
        **kwargs: Any,
    ):
        super().__init__(
            schema,
            likelihood,
            LoadingInverseRoute(),
            state_residual,
            forward_observables,
            **kwargs,
        )


class UnloadedGeometryInverseProblem(_DeterministicInverseProblem):
    def __init__(
        self,
        schema: CardiacParameterSchema,
        likelihood: PreparedMultimodalLikelihood,
        state_residual: Callable,
        forward_observables: Callable,
        /,
        **kwargs: Any,
    ):
        super().__init__(
            schema,
            likelihood,
            UnloadedGeometryInverseRoute(),
            state_residual,
            forward_observables,
            **kwargs,
        )


__all__ = [
    "CardiovascularInverseResult",
    "CardiovascularMultiStartResult",
    "ElectrophysiologyInverseProblem",
    "ElectrophysiologyInverseRoute",
    "InverseAcceptanceEvidence",
    "InverseObjectiveEvaluation",
    "LoadingInverseProblem",
    "LoadingInverseRoute",
    "MechanicsInverseProblem",
    "MechanicsInverseRoute",
    "UnloadedGeometryInverseProblem",
    "UnloadedGeometryInverseRoute",
]
