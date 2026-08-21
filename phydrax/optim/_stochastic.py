#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import core as jax_core
from jaxtyping import Array, Key, PyTree

from .._strict import AbstractAttribute, StrictModule
from ._bounds import ProjectedLBFGS
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._types import (
    _tree_allfinite,
    _tree_inner,
    _tree_norm,
    _validate_real_inexact_tree,
    Bounds,
    MinimizationProblem,
    NonlinearConstraint,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._newton_krylov import NewtonKrylov


class SampleBatch(StrictModule):
    """Scenario PyTree and normalized non-negative quadrature weights."""

    scenarios: PyTree[Array]
    weights: Array
    size: int = eqx.field(static=True)

    def __init__(self, scenarios: PyTree[Any], weights: Any | None = None, /):
        scenarios_ = jax.tree.map(jnp.asarray, scenarios)
        leaves = jax.tree.leaves(scenarios_)
        if any(
            jnp.issubdtype(leaf.dtype, jnp.complexfloating)
            or not (
                jnp.issubdtype(leaf.dtype, jnp.number)
                or jnp.issubdtype(leaf.dtype, jnp.bool_)
            )
            for leaf in leaves
        ):
            raise TypeError("Scenario leaves must be real numeric or boolean arrays.")
        if not leaves or any(leaf.ndim < 1 for leaf in leaves):
            raise ValueError("Every scenario leaf must have a leading sample axis.")
        size = int(leaves[0].shape[0])
        if size < 1 or any(int(leaf.shape[0]) != size for leaf in leaves):
            raise ValueError("Scenario leaves must share a non-empty leading axis.")
        weight_dtype = jnp.result_type(leaves[0], jnp.float32)
        weights_ = (
            jnp.full((size,), 1.0 / size, dtype=weight_dtype)
            if weights is None
            else jnp.asarray(weights, dtype=weight_dtype)
        )
        if weights_.shape != (size,):
            raise ValueError(f"weights must have shape ({size},).")
        invalid_weights = (
            ~jnp.all(jnp.isfinite(weights_))
            | ~jnp.all(weights_ >= 0.0)
            | ~(jnp.sum(weights_) > 0.0)
        )
        if isinstance(invalid_weights, jax_core.Tracer):
            weights_ = eqx.error_if(
                weights_,
                invalid_weights,
                "weights must be finite, non-negative, and have positive mass.",
            )
        elif bool(invalid_weights):
            raise ValueError(
                "weights must be finite, non-negative, and have positive mass."
            )
        self.scenarios = scenarios_
        self.weights = weights_ / jnp.sum(weights_)
        self.size = size

    def scenario(self, index: int, /) -> PyTree[Array]:
        index_ = int(index)
        if not 0 <= index_ < self.size:
            raise IndexError("scenario index out of range.")
        return jax.tree.map(lambda leaf: leaf[index_], self.scenarios)


class AbstractSamplingPolicy(StrictModule):
    """Explicit scenario sampling and refresh semantics."""

    policy_id: AbstractAttribute[str]
    refresh: AbstractAttribute[str]

    @abc.abstractmethod
    def sample(self, key: Key[Array, ""], iteration: int, /) -> SampleBatch:
        raise NotImplementedError


class FixedSampling(AbstractSamplingPolicy):
    """Immutable empirical or quadrature scenario batch."""

    batch: SampleBatch

    def __init__(self, scenarios: PyTree[Any], weights: Any | None = None, /):
        self.batch = SampleBatch(scenarios, weights)

    @property
    def policy_id(self) -> str:
        return "fixed-sampling"

    @property
    def refresh(self) -> str:
        return "fixed"

    def sample(self, key: Key[Array, ""], iteration: int, /) -> SampleBatch:
        del key, iteration
        return self.batch


class MonteCarloSampling(AbstractSamplingPolicy):
    """Seed-reproducible Monte Carlo scenarios with declared refresh policy."""

    sampler: Callable[[Key[Array, ""], int], PyTree[Any]]
    sample_size: int = eqx.field(static=True)
    refresh: Literal["fixed", "per_iteration"] = eqx.field(static=True)

    def __init__(
        self,
        sampler: Callable[[Key[Array, ""], int], PyTree[Any]],
        sample_size: int,
        /,
        *,
        refresh: Literal["fixed", "per_iteration"] = "per_iteration",
    ):
        if not callable(sampler):
            raise TypeError("sampler must be callable.")
        size = int(sample_size)
        if size < 1:
            raise ValueError("sample_size must be positive.")
        if refresh not in ("fixed", "per_iteration"):
            raise ValueError("refresh must be 'fixed' or 'per_iteration'.")
        self.sampler = sampler
        self.sample_size = size
        self.refresh = refresh

    @property
    def policy_id(self) -> str:
        return "monte-carlo"

    def sample(self, key: Key[Array, ""], iteration: int, /) -> SampleBatch:
        fold = 0 if self.refresh == "fixed" else iteration
        return SampleBatch(self.sampler(jr.fold_in(key, fold), self.sample_size))


class AbstractRiskMeasure(StrictModule):
    """Scalar law-invariant risk over weighted scenario losses."""

    risk_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(self, losses: Array, weights: Array, /) -> Array:
        raise NotImplementedError


class ExpectationRisk(AbstractRiskMeasure):
    """Weighted expected loss."""

    def __init__(self):
        pass

    @property
    def risk_id(self) -> str:
        return "expectation"

    def evaluate(self, losses: Array, weights: Array, /) -> Array:
        return jnp.vdot(weights, losses).real


class MeanVarianceRisk(AbstractRiskMeasure):
    """Mean plus a non-negative multiple of variance."""

    coefficient: float = eqx.field(static=True)

    def __init__(self, coefficient: float = 1.0, /):
        value = float(coefficient)
        if not isfinite(value) or value < 0.0:
            raise ValueError("coefficient must be finite and non-negative.")
        self.coefficient = value

    @property
    def risk_id(self) -> str:
        return "mean-variance"

    def evaluate(self, losses: Array, weights: Array, /) -> Array:
        mean = jnp.vdot(weights, losses).real
        variance = jnp.vdot(weights, jnp.square(losses - mean)).real
        return mean + self.coefficient * variance


class CVaRRisk(AbstractRiskMeasure):
    """Upper-tail conditional value at risk at confidence ``alpha``."""

    alpha: float = eqx.field(static=True)

    def __init__(self, alpha: float = 0.95, /):
        value = float(alpha)
        if not isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("alpha must be finite and lie in [0, 1).")
        self.alpha = value

    @property
    def risk_id(self) -> str:
        return "cvar"

    def evaluate(self, losses: Array, weights: Array, /) -> Array:
        order = jnp.argsort(losses)
        ordered_losses = losses[order]
        cumulative = jnp.cumsum(weights[order])
        index = jnp.minimum(
            jnp.sum(cumulative < self.alpha, dtype=jnp.int32),
            losses.size - 1,
        )
        value_at_risk = jax.lax.stop_gradient(ordered_losses[index])
        return value_at_risk + jnp.vdot(
            weights,
            jnp.maximum(losses - value_at_risk, 0.0),
        ).real / (1.0 - self.alpha)


class EntropicRisk(AbstractRiskMeasure):
    """Stable entropic risk ``log E[exp(aversion * loss)] / aversion``."""

    aversion: float = eqx.field(static=True)

    def __init__(self, aversion: float = 1.0, /):
        value = float(aversion)
        if not isfinite(value) or value <= 0.0:
            raise ValueError("aversion must be positive and finite.")
        self.aversion = value

    @property
    def risk_id(self) -> str:
        return "entropic"

    def evaluate(self, losses: Array, weights: Array, /) -> Array:
        scaled = self.aversion * losses
        maximum = jnp.max(scaled)
        return (
            maximum + jnp.log(jnp.sum(weights * jnp.exp(scaled - maximum)))
        ) / self.aversion


class ChanceConstraint(StrictModule):
    """Sample approximation of ``P(event(parameters, scenario) > 0) <= epsilon``."""

    event: Callable[[PyTree[Any], PyTree[Any], Any], Any]
    maximum_probability: float = eqx.field(static=True)
    smoothing_temperature: float = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        event: Callable[[PyTree[Any], PyTree[Any], Any], Any],
        /,
        *,
        maximum_probability: float,
        smoothing_temperature: float = 0.05,
        constraint_id: str = "chance-constraint",
    ):
        probability = float(maximum_probability)
        temperature = float(smoothing_temperature)
        if not callable(event):
            raise TypeError("event must be callable.")
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("maximum_probability must lie in [0, 1].")
        if not isfinite(temperature) or temperature <= 0.0:
            raise ValueError("smoothing_temperature must be positive and finite.")
        identifier = str(constraint_id)
        if not identifier:
            raise ValueError("constraint_id must be non-empty.")
        self.event = event
        self.maximum_probability = probability
        self.smoothing_temperature = temperature
        self.constraint_id = identifier

    def probabilities(
        self,
        parameters: PyTree[Any],
        batch: SampleBatch,
        args: Any = None,
        /,
    ) -> tuple[Array, Array]:
        events = jax.vmap(lambda scenario: self.event(parameters, scenario, args))(
            batch.scenarios
        )
        if events.shape != (batch.size,):
            raise ValueError("A chance event must return one scalar per scenario.")
        empirical = jnp.vdot(batch.weights, events > 0.0).real
        smooth = jnp.vdot(
            batch.weights,
            jax.nn.sigmoid(events / self.smoothing_temperature),
        ).real
        return empirical, smooth

    def bind(self, batch: SampleBatch, args: Any = None, /) -> NonlinearConstraint:
        if not isinstance(batch, SampleBatch):
            raise TypeError("batch must be a SampleBatch.")
        return NonlinearConstraint(
            lambda parameters, dynamic_args: self.probabilities(
                parameters,
                batch,
                args if dynamic_args is None else dynamic_args,
            )[1],
            upper=self.maximum_probability,
            constraint_id=self.constraint_id,
        )


class StochasticProblem(StrictModule):
    """Scenario loss, sampling policy, risk measure, bounds, and chance constraints."""

    scenario_loss: Callable[[PyTree[Any], PyTree[Any], Any], Any]
    sampling: AbstractSamplingPolicy
    risk: AbstractRiskMeasure
    bounds: Bounds | None
    chance_constraints: tuple[ChanceConstraint, ...]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        scenario_loss: Callable[[PyTree[Any], PyTree[Any], Any], Any],
        sampling: AbstractSamplingPolicy,
        /,
        *,
        risk: AbstractRiskMeasure | None = None,
        bounds: Bounds | None = None,
        chance_constraints: Sequence[ChanceConstraint] = (),
        problem_id: str = "stochastic-program",
    ):
        if not callable(scenario_loss):
            raise TypeError("scenario_loss must be callable.")
        if not isinstance(sampling, AbstractSamplingPolicy):
            raise TypeError("sampling must be an AbstractSamplingPolicy.")
        risk_ = ExpectationRisk() if risk is None else risk
        if not isinstance(risk_, AbstractRiskMeasure):
            raise TypeError("risk must be an AbstractRiskMeasure or None.")
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError("bounds must be a Bounds or None.")
        constraints = tuple(chance_constraints)
        if any(not isinstance(item, ChanceConstraint) for item in constraints):
            raise TypeError("chance_constraints must contain ChanceConstraint values.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.scenario_loss = scenario_loss
        self.sampling = sampling
        self.risk = risk_
        self.bounds = bounds
        self.chance_constraints = constraints
        self.problem_id = identifier

    def losses(
        self,
        parameters: PyTree[Any],
        batch: SampleBatch,
        args: Any = None,
        /,
    ) -> Array:
        losses = jax.vmap(
            lambda scenario: self.scenario_loss(parameters, scenario, args)
        )(batch.scenarios)
        losses = jnp.asarray(losses)
        if losses.shape != (batch.size,) or not jnp.issubdtype(
            losses.dtype,
            jnp.floating,
        ):
            raise TypeError("scenario_loss must return one real scalar per scenario.")
        return losses

    def value(
        self,
        parameters: PyTree[Any],
        batch: SampleBatch,
        args: Any = None,
        /,
    ) -> Array:
        return self.risk.evaluate(self.losses(parameters, batch, args), batch.weights)

    def frozen(
        self,
        key: Key[Array, ""],
        iteration: int = 0,
        /,
        *,
        args: Any = None,
    ) -> tuple[MinimizationProblem, SampleBatch]:
        batch = self.sampling.sample(key, iteration)
        constraints = tuple(
            constraint.bind(batch, args) for constraint in self.chance_constraints
        )
        frozen_problem = MinimizationProblem(
            lambda parameters, dynamic_args: self.value(
                parameters,
                batch,
                args if dynamic_args is None else dynamic_args,
            ),
            bounds=self.bounds,
            constraints=constraints,
            problem_id=f"{self.problem_id}/sample-{iteration}",
        )
        return frozen_problem, batch


class StochasticResult(StrictModule):
    """Consensus stochastic optimizer result with scenario decomposition evidence."""

    parameters: PyTree[Array]
    scenario_parameters: PyTree[Array] | None
    duals: PyTree[Array] | None
    objective: Array
    status: Array
    diagnostics: OptimizationDiagnostics
    provenance: OptimizationProvenance
    key: Key[Array, ""]

    def __init__(
        self,
        parameters: PyTree[Any],
        scenario_parameters: PyTree[Any] | None,
        duals: PyTree[Any] | None,
        objective: Any,
        status: Any,
        diagnostics: OptimizationDiagnostics,
        provenance: OptimizationProvenance,
        key: Key[Array, ""],
        /,
    ):
        self.parameters = _validate_real_inexact_tree(parameters, name="parameters")
        self.scenario_parameters = scenario_parameters
        self.duals = duals
        self.objective = jnp.asarray(objective)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        if not isinstance(diagnostics, OptimizationDiagnostics):
            raise TypeError("diagnostics must be OptimizationDiagnostics.")
        if not isinstance(provenance, OptimizationProvenance):
            raise TypeError("provenance must be OptimizationProvenance.")
        self.diagnostics = diagnostics
        self.provenance = provenance
        self.key = key

    @property
    def successful(self) -> Array:
        return self.status == int(OptimizationStatus.SUCCESS)


class AbstractStochasticMethod(StrictModule):
    """Complete stochastic optimization method."""

    method_id: AbstractAttribute[str]

    @abc.abstractmethod
    def solve(
        self,
        problem: StochasticProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        key: Key[Array, ""],
        args: Any,
    ) -> StochasticResult:
        raise NotImplementedError


class StochasticAdam(AbstractStochasticMethod):
    """Adam baseline over explicitly refreshed risk batches."""

    learning_rate: float = eqx.field(static=True)
    beta1: float = eqx.field(static=True)
    beta2: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)

    def __init__(
        self,
        learning_rate: float = 1e-2,
        /,
        *,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        values = tuple(float(value) for value in (learning_rate, beta1, beta2, epsilon))
        if any(not isfinite(value) for value in values):
            raise ValueError("Adam parameters must be finite.")
        if values[0] <= 0.0 or not 0.0 <= values[1] < 1.0 or not 0.0 <= values[2] < 1.0:
            raise ValueError("Adam requires positive rate and beta values in [0, 1).")
        if values[3] <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.learning_rate, self.beta1, self.beta2, self.epsilon = values

    @property
    def method_id(self) -> str:
        return "stochastic-adam"

    def solve(
        self,
        problem: StochasticProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        key: Key[Array, ""],
        args: Any,
    ) -> StochasticResult:
        if problem.chance_constraints:
            raise ValueError(
                "StochasticAdam does not silently penalize chance constraints; "
                "solve problem.frozen(...) with a constrained method."
            )
        parameters = _validate_real_inexact_tree(initial_parameters, name="parameters")
        if problem.bounds is not None:
            parameters = problem.bounds.project(parameters)
        optimizer = optax.adam(
            self.learning_rate,
            b1=self.beta1,
            b2=self.beta2,
            eps=self.epsilon,
        )
        optimizer_state = optimizer.init(parameters)
        initial_batch = problem.sampling.sample(key, 0)
        scalar_template = _tree_norm(jax.tree.map(jnp.zeros_like, parameters))
        initial_optimality = jnp.full_like(scalar_template, jnp.nan)
        final_optimality = jnp.full_like(scalar_template, jnp.nan)
        final_step_norm = jnp.zeros_like(scalar_template)
        status = jnp.where(
            _tree_allfinite(parameters),
            int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        accepted_steps = jnp.asarray(0, dtype=jnp.int32)
        rejected_steps = jnp.asarray(0, dtype=jnp.int32)
        loop_evaluations = jnp.asarray(0, dtype=jnp.int32)

        def condition(carry):
            (
                _,
                _,
                _,
                _,
                _,
                current_status,
                _,
                _,
                evaluations,
                _,
            ) = carry
            within_evaluations = (
                jnp.asarray(True)
                if termination.maximum_evaluations is None
                else evaluations < termination.maximum_evaluations
            )
            return (
                (current_status == int(OptimizationStatus.ITERATING))
                & (evaluations < termination.maximum_steps)
                & within_evaluations
            )

        def body(carry):
            (
                current_parameters,
                current_optimizer_state,
                current_initial_optimality,
                current_final_optimality,
                current_step_norm,
                _,
                current_accepted_steps,
                current_rejected_steps,
                iteration,
                current_last_batch,
            ) = carry
            if problem.sampling.refresh == "fixed":
                batch = initial_batch
            else:
                batch = jax.lax.cond(
                    iteration == 0,
                    lambda _: initial_batch,
                    lambda index: problem.sampling.sample(key, index),
                    iteration,
                )
            value, gradient = jax.value_and_grad(
                lambda candidate: problem.value(candidate, batch, args)
            )(current_parameters)
            projected_gradient = (
                gradient
                if problem.bounds is None
                else problem.bounds.projected_gradient(current_parameters, gradient)
            )
            optimality = _tree_norm(projected_gradient)
            next_initial_optimality = jnp.where(
                iteration == 0,
                optimality,
                current_initial_optimality,
            )
            finite = (
                _tree_allfinite(current_parameters)
                & jnp.isfinite(value)
                & jnp.isfinite(optimality)
                & _tree_allfinite(gradient)
            )
            converged = finite & (
                optimality <= termination.optimality_threshold(next_initial_optimality)
            )
            update_parameters = finite & ~converged
            next_status = jnp.where(
                ~finite,
                int(OptimizationStatus.NONFINITE_EVALUATION),
                jnp.where(
                    converged,
                    int(OptimizationStatus.SUCCESS),
                    int(OptimizationStatus.ITERATING),
                ),
            ).astype(jnp.int32)

            def perform_update(operand):
                values, state = operand
                updates, next_optimizer_state = optimizer.update(
                    gradient,
                    state,
                    values,
                )
                candidate = optax.apply_updates(values, updates)
                if problem.bounds is not None:
                    candidate = problem.bounds.project(candidate)
                step_norm = _tree_norm(
                    jax.tree.map(
                        lambda new, old: new - old,
                        candidate,
                        values,
                    )
                )
                return candidate, next_optimizer_state, step_norm

            def retain_parameters(operand):
                values, state = operand
                return values, state, current_step_norm

            (
                candidate_parameters,
                candidate_optimizer_state,
                candidate_step_norm,
            ) = jax.lax.cond(
                update_parameters,
                perform_update,
                retain_parameters,
                (current_parameters, current_optimizer_state),
            )
            candidate_finite = (
                _tree_allfinite(candidate_parameters)
                & _tree_allfinite(candidate_optimizer_state)
                & jnp.isfinite(candidate_step_norm)
            )
            accept_candidate = update_parameters & candidate_finite
            next_parameters, next_optimizer_state, next_step_norm = jax.lax.cond(
                accept_candidate,
                lambda _: (
                    candidate_parameters,
                    candidate_optimizer_state,
                    candidate_step_norm,
                ),
                lambda _: (
                    current_parameters,
                    current_optimizer_state,
                    current_step_norm,
                ),
                None,
            )
            invalid_candidate = update_parameters & ~candidate_finite
            next_status = jnp.where(
                invalid_candidate,
                int(OptimizationStatus.NONFINITE_EVALUATION),
                next_status,
            ).astype(jnp.int32)
            retain_evaluation_batch = converged | accept_candidate
            next_last_batch = jax.lax.cond(
                retain_evaluation_batch,
                lambda _: batch,
                lambda _: current_last_batch,
                None,
            )
            return (
                next_parameters,
                next_optimizer_state,
                next_initial_optimality,
                jnp.where(
                    finite,
                    optimality,
                    current_final_optimality,
                ),
                next_step_norm,
                next_status,
                current_accepted_steps + accept_candidate.astype(jnp.int32),
                current_rejected_steps + invalid_candidate.astype(jnp.int32),
                iteration + 1,
                next_last_batch,
            )

        (
            parameters,
            optimizer_state,
            initial_optimality,
            final_optimality,
            final_step_norm,
            status,
            accepted_steps,
            rejected_steps,
            loop_evaluations,
            last_batch,
        ) = jax.lax.while_loop(
            condition,
            body,
            (
                parameters,
                optimizer_state,
                initial_optimality,
                final_optimality,
                final_step_norm,
                status,
                accepted_steps,
                rejected_steps,
                loop_evaluations,
                initial_batch,
            ),
        )
        del optimizer_state
        if termination.maximum_evaluations is not None:
            status = jnp.where(
                (status == int(OptimizationStatus.ITERATING))
                & (loop_evaluations >= termination.maximum_evaluations),
                int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED),
                status,
            )
        status = jnp.where(
            status == int(OptimizationStatus.ITERATING),
            int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
            status,
        ).astype(jnp.int32)

        final_value, final_gradient = jax.value_and_grad(
            lambda candidate: problem.value(candidate, last_batch, args)
        )(parameters)
        projected_final_gradient = (
            final_gradient
            if problem.bounds is None
            else problem.bounds.projected_gradient(parameters, final_gradient)
        )
        final_optimality = _tree_norm(projected_final_gradient)
        terminal_budget_status = (
            status == int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
        ) | (status == int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED))
        status = jnp.where(
            terminal_budget_status
            & (final_optimality <= termination.optimality_threshold(initial_optimality)),
            int(OptimizationStatus.SUCCESS),
            status,
        ).astype(jnp.int32)
        total_value_and_gradient_evaluations = loop_evaluations + 1
        diagnostics = OptimizationDiagnostics(
            iterations=accepted_steps,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            objective_evaluations=total_value_and_gradient_evaluations,
            gradient_evaluations=total_value_and_gradient_evaluations,
            initial_optimality_norm=initial_optimality,
            final_optimality_norm=final_optimality,
            final_step_norm=final_step_norm,
            primal_feasibility=(
                0.0 if problem.bounds is None else problem.bounds.violation(parameters)
            ),
            dual_feasibility=final_optimality,
            counts_complete=True,
        )
        provenance = OptimizationProvenance(
            problem_id=problem.problem_id,
            method=self.method_id,
            backend="optax",
            backend_method="adam",
            globalization=problem.sampling.refresh,
            matrix_free=True,
            notes=f"Risk measure: {problem.risk.risk_id}.",
        )
        return StochasticResult(
            parameters,
            None,
            None,
            final_value,
            status,
            diagnostics,
            provenance,
            key,
        )


class _AbstractConsensusMethod(AbstractStochasticMethod):
    penalty: float = eqx.field(static=True)
    maximum_outer_steps: int = eqx.field(static=True)
    inner_maximum_steps: int = eqx.field(static=True)
    inner_method: AbstractMinimizationMethod | None

    def __init__(
        self,
        *,
        penalty: float = 1.0,
        maximum_outer_steps: int = 50,
        inner_maximum_steps: int = 50,
        inner_method: AbstractMinimizationMethod | None = None,
    ):
        penalty_ = float(penalty)
        outer = int(maximum_outer_steps)
        inner = int(inner_maximum_steps)
        if not isfinite(penalty_) or penalty_ <= 0.0:
            raise ValueError("penalty must be positive and finite.")
        if outer < 1 or inner < 1:
            raise ValueError("Outer and inner step limits must be positive.")
        if inner_method is not None and not isinstance(
            inner_method,
            AbstractMinimizationMethod,
        ):
            raise TypeError("inner_method must be an AbstractMinimizationMethod or None.")
        self.penalty = penalty_
        self.maximum_outer_steps = outer
        self.inner_maximum_steps = inner
        self.inner_method = inner_method

    @property
    @abc.abstractmethod
    def mode(self) -> Literal["progressive-hedging", "consensus-admm"]:
        raise NotImplementedError

    def solve(
        self,
        problem: StochasticProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        key: Key[Array, ""],
        args: Any,
    ) -> StochasticResult:
        return _solve_consensus(
            self,
            problem,
            initial_parameters,
            termination=termination,
            key=key,
            args=args,
        )


class ProgressiveHedging(_AbstractConsensusMethod):
    """Scenario-decomposed progressive hedging with weighted consensus."""

    @property
    def method_id(self) -> str:
        return "progressive-hedging"

    @property
    def mode(self) -> Literal["progressive-hedging", "consensus-admm"]:
        return "progressive-hedging"


class ConsensusADMM(_AbstractConsensusMethod):
    """Consensus ADMM over independent scenario subproblems."""

    @property
    def method_id(self) -> str:
        return "consensus-admm"

    @property
    def mode(self) -> Literal["progressive-hedging", "consensus-admm"]:
        return "consensus-admm"


def _weighted_consensus(
    parameters: PyTree[Any],
    weights: Array,
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda values: jnp.tensordot(weights, values, axes=1),
        parameters,
    )


def _stack_pytrees(values: list[PyTree[Any]], /) -> PyTree[Array]:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *values)


@eqx.filter_jit
def _consensus_numeric_update(
    mode: Literal["progressive-hedging", "consensus-admm"],
    penalty: float,
    weights: Array,
    scenario_parameters: PyTree[Any],
    duals: PyTree[Any],
    previous_consensus: PyTree[Any],
    /,
) -> tuple[
    PyTree[Array],
    PyTree[Array],
    Array,
    Array,
    Array,
]:
    """Stage the batched consensus and dual algebra for one outer step."""
    if mode == "progressive-hedging":
        consensus = _weighted_consensus(scenario_parameters, weights)
        next_duals = jax.tree.map(
            lambda dual, value, center: dual + penalty * (value - center),
            duals,
            scenario_parameters,
            consensus,
        )
    else:
        shifted_parameters = jax.tree.map(
            lambda value, dual: value + dual,
            scenario_parameters,
            duals,
        )
        consensus = _weighted_consensus(shifted_parameters, weights)
        next_duals = jax.tree.map(
            lambda dual, value, center: dual + value - center,
            duals,
            scenario_parameters,
            consensus,
        )
    displacements = jax.tree.map(
        lambda value, center: value - center,
        scenario_parameters,
        consensus,
    )
    scenario_squared_norms = jax.vmap(_tree_inner)(
        displacements,
        displacements,
    )
    primal_squared = jnp.vdot(weights, scenario_squared_norms).real
    primal = jnp.sqrt(jnp.maximum(primal_squared, 0.0))
    final_step_norm = _tree_norm(
        jax.tree.map(
            lambda current, previous: current - previous,
            consensus,
            previous_consensus,
        )
    )
    return (
        consensus,
        next_duals,
        primal,
        penalty * final_step_norm,
        final_step_norm,
    )


def _solve_consensus(
    method: _AbstractConsensusMethod,
    problem: StochasticProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    termination: OptimizationTermination,
    key: Key[Array, ""],
    args: Any,
) -> StochasticResult:
    if not isinstance(problem.risk, ExpectationRisk):
        raise ValueError(
            "ProgressiveHedging and ConsensusADMM require separable ExpectationRisk."
        )
    if problem.chance_constraints:
        raise ValueError("Scenario consensus methods do not absorb chance constraints.")
    # Consensus duals are indexed by scenario, so decomposition freezes one
    # realization at the workflow boundary even for a refreshing policy. Replacing
    # scenarios between outer steps would change the optimization problem attached
    # to each dual rather than implement per-iteration stochastic sampling.
    batch = problem.sampling.sample(key, 0)
    initial = _validate_real_inexact_tree(initial_parameters, name="parameters")
    if problem.bounds is not None:
        initial = problem.bounds.project(initial)
    scenario_parameters = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (batch.size, *value.shape)),
        initial,
    )
    duals = jax.tree.map(jnp.zeros_like, scenario_parameters)
    consensus = initial
    inner_method = method.inner_method
    if inner_method is None:
        inner_method = ProjectedLBFGS() if problem.bounds is not None else NewtonKrylov()
    maximum_outer = min(method.maximum_outer_steps, termination.maximum_steps)
    initial_optimality = jnp.asarray(jnp.nan)
    primal = jnp.asarray(jnp.inf)
    dual_residual = jnp.asarray(jnp.inf)
    final_step_norm = jnp.asarray(0.0)
    status = (
        OptimizationStatus.ITERATING
        if bool(_tree_allfinite(initial))
        else OptimizationStatus.NONFINITE_INPUT
    )
    accepted_steps = 0
    rejected_steps = 0
    objective_evaluations = 0
    gradient_evaluations = 0
    residual_evaluations = 0
    jvp_evaluations = 0
    vjp_evaluations = 0
    hvp_evaluations = 0
    jacobian_evaluations = 0
    constraint_evaluations = 0
    linear_solves = 0
    setup_refreshes = 0
    numeric_refreshes = 0
    linear_iterations = 0
    globalization_evaluations = 0
    direction_fallbacks = 0
    iterations = 0

    # Workflow boundary: every local solve constructs scenario-specific Python
    # provenance and may call an arbitrary externally supplied method. Its status,
    # diagnostic record, and outer-dependent static termination policy are consumed
    # before the next scenario is built. The scenario/outer orchestration therefore
    # intentionally remains on the host; staging it would falsely promise transform
    # support across that open method boundary. Each inner method stages its own
    # numerical iteration/globalization, while `_consensus_numeric_update` stages
    # the fixed-shape consensus algebra.
    for outer in range(maximum_outer):
        if status != OptimizationStatus.ITERATING:
            break
        if (
            termination.maximum_evaluations is not None
            and objective_evaluations >= termination.maximum_evaluations
        ):
            status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
            break
        updated_parameters = []
        for index in range(batch.size):
            scenario = batch.scenario(index)
            dual_value = jax.tree.map(lambda value: value[index], duals)

            def local_objective(candidate, dynamic_args):
                scenario_value = problem.scenario_loss(
                    candidate,
                    scenario,
                    dynamic_args,
                )
                if method.mode == "progressive-hedging":
                    displacement = jax.tree.map(
                        lambda value, center: value - center,
                        candidate,
                        consensus,
                    )
                    return (
                        scenario_value
                        + _tree_inner(dual_value, candidate)
                        + 0.5 * method.penalty * _tree_inner(displacement, displacement)
                    )
                shifted = jax.tree.map(
                    lambda value, center, dual: value - center + dual,
                    candidate,
                    consensus,
                    dual_value,
                )
                return scenario_value + 0.5 * method.penalty * _tree_inner(
                    shifted,
                    shifted,
                )

            local_problem = MinimizationProblem(
                local_objective,
                bounds=problem.bounds,
                problem_id=f"{problem.problem_id}/scenario-{index}",
            )
            inner_result = inner_method.solve(
                local_problem,
                jax.tree.map(lambda value: value[index], scenario_parameters),
                termination=OptimizationTermination(
                    absolute_optimality=max(
                        termination.absolute_optimality,
                        min(1e-3, 0.1 / (outer + 1)),
                    ),
                    relative_optimality=0.0,
                    absolute_step=termination.absolute_step,
                    relative_step=termination.relative_step,
                    maximum_steps=method.inner_maximum_steps,
                ),
                args=args,
            )
            diagnostics = inner_result.diagnostics
            if (
                termination.maximum_evaluations is not None
                and not diagnostics.counts_complete
            ):
                raise ValueError(
                    "Scenario consensus cannot enforce maximum_evaluations with "
                    "an inner method whose diagnostic counts are incomplete."
                )
            objective_evaluations += max(int(diagnostics.objective_evaluations), 0)
            gradient_evaluations += max(int(diagnostics.gradient_evaluations), 0)
            residual_evaluations += max(int(diagnostics.residual_evaluations), 0)
            jvp_evaluations += max(int(diagnostics.jvp_evaluations), 0)
            vjp_evaluations += max(int(diagnostics.vjp_evaluations), 0)
            hvp_evaluations += max(int(diagnostics.hvp_evaluations), 0)
            jacobian_evaluations += max(int(diagnostics.jacobian_evaluations), 0)
            constraint_evaluations += max(int(diagnostics.constraint_evaluations), 0)
            linear_solves += max(int(diagnostics.linear_solves), 0)
            setup_refreshes += max(int(diagnostics.setup_refreshes), 0)
            numeric_refreshes += max(int(diagnostics.numeric_refreshes), 0)
            linear_iterations += max(int(diagnostics.linear_iterations), 0)
            globalization_evaluations += max(
                int(diagnostics.globalization_evaluations), 0
            )
            direction_fallbacks += max(int(diagnostics.direction_fallbacks), 0)
            if int(inner_result.status) in (
                int(OptimizationStatus.NONFINITE_INPUT),
                int(OptimizationStatus.NONFINITE_EVALUATION),
                int(OptimizationStatus.BACKEND_FAILED),
                int(OptimizationStatus.DIVERGENCE),
            ):
                status = OptimizationStatus.BACKEND_FAILED
                rejected_steps += 1
                break
            updated_parameters.append(inner_result.parameters)
        if status != OptimizationStatus.ITERATING:
            break

        previous_consensus = consensus
        scenario_parameters = _stack_pytrees(updated_parameters)
        (
            consensus,
            duals,
            primal,
            dual_residual,
            final_step_norm,
        ) = _consensus_numeric_update(
            method.mode,
            method.penalty,
            batch.weights,
            scenario_parameters,
            duals,
            previous_consensus,
        )
        optimality = jnp.maximum(primal, dual_residual)
        if outer == 0:
            initial_optimality = optimality
        iterations = outer + 1
        accepted_steps += 1
        if bool(optimality <= termination.optimality_threshold(initial_optimality)):
            status = OptimizationStatus.SUCCESS
            break
        if (
            termination.maximum_evaluations is not None
            and objective_evaluations >= termination.maximum_evaluations
        ):
            status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
            break
    else:
        status = OptimizationStatus.MAXIMUM_STEPS_REACHED

    objective = problem.value(consensus, batch, args)
    objective_evaluations += 1
    diagnostics = OptimizationDiagnostics(
        iterations=iterations,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        residual_evaluations=residual_evaluations,
        jvp_evaluations=jvp_evaluations,
        vjp_evaluations=vjp_evaluations,
        hvp_evaluations=hvp_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        constraint_evaluations=constraint_evaluations,
        linear_solves=linear_solves,
        setup_refreshes=setup_refreshes,
        numeric_refreshes=numeric_refreshes,
        linear_iterations=linear_iterations,
        globalization_evaluations=globalization_evaluations,
        direction_fallbacks=direction_fallbacks,
        initial_optimality_norm=initial_optimality,
        final_optimality_norm=jnp.maximum(primal, dual_residual),
        final_step_norm=final_step_norm,
        damping=method.penalty,
        primal_feasibility=primal,
        dual_feasibility=dual_residual,
        counts_complete=False,
    )
    provenance = OptimizationProvenance(
        problem_id=problem.problem_id,
        method=method.method_id,
        backend="phydrax",
        backend_method=inner_method.method_id,
        globalization="scenario-consensus",
        matrix_free=inner_method.capabilities.matrix_free,
        notes=f"{batch.size} fixed weighted scenarios; separable expectation objective.",
    )
    return StochasticResult(
        consensus,
        scenario_parameters,
        duals,
        objective,
        status,
        diagnostics,
        provenance,
        key,
    )


def minimize_stochastic(
    problem: StochasticProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractStochasticMethod,
    termination: OptimizationTermination | None = None,
    key: Key[Array, ""] | None = None,
    seed: int = 0,
    args: Any = None,
) -> StochasticResult:
    """Minimize a stochastic program with explicit PRNG and sampling semantics."""

    if not isinstance(problem, StochasticProblem):
        raise TypeError("problem must be a StochasticProblem.")
    if not isinstance(method, AbstractStochasticMethod):
        raise TypeError("method must be an AbstractStochasticMethod.")
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    key_ = jr.key(int(seed)) if key is None else key
    return method.solve(
        problem,
        initial_parameters,
        termination=termination_,
        key=key_,
        args=args,
    )


__all__ = [
    "AbstractRiskMeasure",
    "AbstractSamplingPolicy",
    "AbstractStochasticMethod",
    "CVaRRisk",
    "ChanceConstraint",
    "ConsensusADMM",
    "EntropicRisk",
    "ExpectationRisk",
    "FixedSampling",
    "MeanVarianceRisk",
    "MonteCarloSampling",
    "ProgressiveHedging",
    "SampleBatch",
    "StochasticAdam",
    "StochasticProblem",
    "StochasticResult",
    "minimize_stochastic",
]
