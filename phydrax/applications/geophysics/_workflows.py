#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

import phydrax.linalg as la

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import DiagonalCovarianceAction


class MAPResult(StrictModule):
    parameters: Array
    objective: Array
    gradient_norm: Array
    iterations: Array
    converged: Array


class MatrixFreeMAPPlan(StrictModule, NonTrainableState):
    """Damped matrix-free Newton/Gauss-Newton MAP with monotone line search."""

    objective_fn: Callable[[Array], Array] = eqx.field(static=True)
    hessian_action_fn: Callable[[Array, Array], Array] = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)
    hessian_action_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    gradient_tolerance: float = eqx.field(static=True)
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array], Array],
        hessian_action: Callable[[Array, Array], Array],
        dimension: int,
        /,
        *,
        objective_id: str,
        hessian_action_id: str,
        damping: float = 1e-6,
        maximum_iterations: int = 30,
        gradient_tolerance: float = 1e-6,
    ):
        if not callable(objective) or not callable(hessian_action):
            raise TypeError("MAP objective and Hessian action must be callable.")
        objective_identity = str(objective_id).strip()
        hessian_identity = str(hessian_action_id).strip()
        if not objective_identity or not hessian_identity:
            raise ValueError("MAP callable identities must be nonempty.")
        dimension_, damping_, iterations, tolerance = (
            int(dimension),
            float(damping),
            int(maximum_iterations),
            float(gradient_tolerance),
        )
        if (
            dimension_ <= 0
            or not np.isfinite(damping_)
            or damping_ < 0
            or iterations <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("MAP dimension/damping/iteration/tolerance are invalid.")
        self.objective_fn, self.hessian_action_fn = objective, hessian_action
        self.objective_id = objective_identity
        self.hessian_action_id = hessian_identity
        self.dimension, self.damping = dimension_, damping_
        self.maximum_iterations, self.gradient_tolerance = iterations, tolerance
        self.policy = la.LinearSolvePolicy(
            la.ConjugateGradient(),
            tolerance=la.TolerancePolicy(relative=1e-5, absolute=1e-10, max_steps=500),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "matrix-free-map",
                "objective": objective_identity,
                "hessian_action": hessian_identity,
                "dimension": dimension_,
                "damping": damping_,
                "maximum_iterations": iterations,
                "gradient_tolerance": tolerance,
            }
        )

    def solve(self, initial: ArrayLike, /) -> MAPResult:
        parameters = jnp.asarray(initial)
        if parameters.shape != (self.dimension,):
            raise ValueError("MAP initial parameters have wrong shape.")
        if jnp.iscomplexobj(parameters):
            raise TypeError("MAP parameters must be real.")
        parameters = eqx.error_if(
            parameters,
            jnp.any(~jnp.isfinite(parameters)),
            "MAP parameters must be finite.",
        )
        objective = jnp.asarray(self.objective_fn(parameters))
        if objective.shape != () or jnp.iscomplexobj(objective):
            raise ValueError("MAP objective must return one real scalar.")
        objective = eqx.error_if(
            objective, ~jnp.isfinite(objective), "MAP objective must be finite."
        )
        converged = jnp.asarray(False)
        completed = jnp.asarray(0, dtype=jnp.int32)
        space = la.ArraySpace((self.dimension,), dtype=parameters.dtype)
        for iteration in range(self.maximum_iterations):
            gradient = jax.grad(self.objective_fn)(parameters)
            norm = jnp.sqrt(jnp.real(jnp.vdot(gradient, gradient)))
            gradient = eqx.error_if(
                gradient,
                jnp.any(~jnp.isfinite(gradient)),
                "MAP objective gradient must be finite.",
            )
            active = ~converged

            def action(direction):
                return (
                    self.hessian_action_fn(parameters, direction)
                    + self.damping * direction
                )

            operator = la.FunctionLinearOperator(
                action,
                source=space,
                target=space,
                properties=la.OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={
                        "self_adjoint": "asserted",
                        "positive_definite": "asserted",
                    },
                ),
            )
            step = la.solve(la.LinearSystem(operator), -gradient, policy=self.policy)
            direction = eqx.error_if(
                step.value, ~step.successful, "MAP matrix-free Newton solve failed."
            )
            directional = jnp.real(jnp.vdot(gradient, direction))
            directional = eqx.error_if(
                directional,
                active & (norm > self.gradient_tolerance) & (directional >= 0),
                "MAP Hessian action did not produce a descent direction.",
            )
            scale = jnp.asarray(1.0)
            candidate = parameters + scale * direction
            candidate_objective = self.objective_fn(candidate)
            for _ in range(20):
                accept = candidate_objective <= objective + 1e-4 * scale * directional
                scale = jnp.where(accept, scale, 0.5 * scale)
                candidate = parameters + scale * direction
                candidate_objective = self.objective_fn(candidate)
            update = active & (norm > self.gradient_tolerance)
            accepted = candidate_objective <= (objective + 1e-4 * scale * directional)
            candidate = eqx.error_if(
                candidate,
                update & (~jnp.isfinite(candidate_objective) | ~accepted),
                "MAP monotone line search failed to accept a finite step.",
            )
            parameters = jnp.where(update, candidate, parameters)
            objective = jnp.where(update, candidate_objective, objective)
            converged = converged | (norm <= self.gradient_tolerance)
            completed = jnp.where(
                active & (norm > self.gradient_tolerance), iteration + 1, completed
            )
        gradient = jax.grad(self.objective_fn)(parameters)
        norm = jnp.sqrt(jnp.real(jnp.vdot(gradient, gradient)))
        converged = converged | (norm <= self.gradient_tolerance)
        return MAPResult(parameters, objective, norm, completed, converged)


class EnsembleInversionResult(StrictModule):
    ensemble: Array
    prediction_mean: Array
    data_misfit: Array
    spread: Array
    finite: Array


class EnsembleKalmanInversionPlan(StrictModule, NonTrainableState):
    plan_id: str = eqx.field(static=True)
    observation: Array
    covariance: DiagonalCovarianceAction
    inflation: float = eqx.field(static=True)

    def __init__(
        self,
        observation: ArrayLike,
        covariance: DiagonalCovarianceAction,
        /,
        *,
        inflation: float = 1.0,
    ):
        if not isinstance(covariance, DiagonalCovarianceAction):
            raise TypeError(
                "Scalable ensemble inversion currently requires diagonal covariance."
            )
        observed = jnp.asarray(observation)
        inflation_ = float(inflation)
        if (
            observed.shape != (covariance.layout.size,)
            or not np.isfinite(inflation_)
            or inflation_ <= 0
        ):
            raise ValueError("Ensemble observation shape or inflation is invalid.")
        if jnp.iscomplexobj(observed):
            raise TypeError("Ensemble inversion observations must be real.")
        observed = eqx.error_if(
            observed,
            jnp.any(~jnp.isfinite(observed)),
            "Ensemble inversion observations must be finite.",
        )
        self.observation, self.covariance, self.inflation = (
            observed,
            covariance,
            inflation_,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "ensemble-kalman-inversion",
                "observation": observed,
                "covariance": covariance.action_id,
                "inflation": inflation_,
            }
        )

    def update(
        self,
        ensemble: ArrayLike,
        predict: Callable[[Array], Array],
        /,
    ) -> EnsembleInversionResult:
        members = jnp.asarray(ensemble)
        if members.ndim != 2 or members.shape[0] < 2 or not callable(predict):
            raise ValueError(
                "Ensemble inversion requires at least two parameter vectors and prediction callable."
            )
        if jnp.iscomplexobj(members):
            raise TypeError("Ensemble inversion members must be real.")
        members = eqx.error_if(
            members,
            jnp.any(~jnp.isfinite(members)),
            "Ensemble inversion members must be finite.",
        )
        predictions = jax.vmap(predict)(members)
        if predictions.shape != (members.shape[0], self.observation.size):
            raise ValueError("Ensemble prediction shape does not match observation.")
        if jnp.iscomplexobj(predictions):
            raise TypeError("Ensemble predictions must be real.")
        predictions = eqx.error_if(
            predictions,
            jnp.any(~jnp.isfinite(predictions)),
            "Ensemble predictions must be finite.",
        )
        parameter_mean, prediction_mean = (
            jnp.mean(members, axis=0),
            jnp.mean(predictions, axis=0),
        )
        x = (members - parameter_mean) / jnp.sqrt(members.shape[0] - 1.0)
        y = (predictions - prediction_mean) / jnp.sqrt(members.shape[0] - 1.0)
        whitened_y = y / jnp.sqrt(self.covariance.variance)[None, :]
        innovation = (self.observation[None, :] - predictions) / jnp.sqrt(
            self.covariance.variance
        )[None, :]
        ensemble_matrix = jnp.eye(members.shape[0]) + whitened_y @ whitened_y.T
        right = whitened_y @ innovation.T
        space = la.ArraySpace((members.shape[0],), dtype=ensemble_matrix.dtype)
        operator = la.DenseLinearOperator(
            ensemble_matrix,
            source=space,
            target=space,
            properties=la.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        )
        policy = la.LinearSolvePolicy(
            la.DenseCholesky(), failure=la.FailurePolicy("status")
        )

        def solve_column(column: Array) -> Array:
            result = la.solve(la.LinearSystem(operator), column, policy=policy)
            return eqx.error_if(
                result.value,
                ~result.successful,
                "Ensemble inversion covariance solve failed.",
            )

        coefficients = jax.vmap(solve_column, in_axes=1, out_axes=1)(right)
        updated = members + self.inflation * (x.T @ coefficients).T
        misfit = jnp.mean(jnp.sum(innovation**2, axis=1))
        spread = jnp.mean(jnp.var(updated, axis=0, ddof=1))
        finite = jnp.all(jnp.isfinite(updated)) & jnp.isfinite(misfit)
        return EnsembleInversionResult(updated, prediction_mean, misfit, spread, finite)


class PCNSamplingResult(StrictModule):
    samples: Array
    log_likelihood: Array
    acceptance_rate: Array
    final_key: Array


class PCNSampler(StrictModule, NonTrainableState):
    """Preconditioned Crank-Nicolson for standard-normal whitened parameters."""

    log_likelihood_fn: Callable[[Array], Array] = eqx.field(static=True)
    log_likelihood_id: str = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)

    def __init__(
        self,
        log_likelihood: Callable[[Array], Array],
        step_size: float,
        sample_count: int,
        /,
        *,
        log_likelihood_id: str,
    ):
        likelihood_identity = str(log_likelihood_id).strip()
        step, count = float(step_size), int(sample_count)
        if (
            not callable(log_likelihood)
            or not likelihood_identity
            or not 0 < step < 1
            or count <= 0
        ):
            raise ValueError("pCN likelihood/step/count are invalid.")
        self.log_likelihood_fn = log_likelihood
        self.log_likelihood_id = likelihood_identity
        self.step_size, self.sample_count = step, count
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "pcn-sampler",
                "log_likelihood": likelihood_identity,
                "step_size": step,
                "sample_count": count,
            }
        )

    def sample(
        self, key: PRNGKeyArray, initial_whitened: ArrayLike, /
    ) -> PCNSamplingResult:
        initial = jnp.asarray(initial_whitened)
        if jnp.iscomplexobj(initial):
            raise TypeError("pCN whitened parameters must be real.")
        initial = eqx.error_if(
            initial,
            jnp.any(~jnp.isfinite(initial)),
            "pCN initial whitened parameters must be finite.",
        )
        initial_log = jnp.asarray(self.log_likelihood_fn(initial))
        if initial_log.shape != () or jnp.iscomplexobj(initial_log):
            raise ValueError("pCN log likelihood must be one real scalar.")
        initial_log = eqx.error_if(
            initial_log,
            ~jnp.isfinite(initial_log),
            "pCN initial log likelihood must be finite.",
        )

        def step(carry, _):
            state, value, random_key, accepted = carry
            proposal_key, uniform_key, next_key = jax.random.split(random_key, 3)
            noise = jax.random.normal(proposal_key, state.shape, dtype=state.dtype)
            proposal = jnp.sqrt(1.0 - self.step_size**2) * state + self.step_size * noise
            proposal_value = jnp.asarray(self.log_likelihood_fn(proposal))
            if proposal_value.shape != () or jnp.iscomplexobj(proposal_value):
                raise ValueError("pCN log likelihood must remain one real scalar.")
            proposal_value = eqx.error_if(
                proposal_value,
                ~jnp.isfinite(proposal_value),
                "pCN proposal log likelihood must be finite.",
            )
            accept = jnp.log(jax.random.uniform(uniform_key)) < proposal_value - value
            state = jnp.where(accept, proposal, state)
            value = jnp.where(accept, proposal_value, value)
            return (state, value, next_key, accepted + accept), (state, value)

        final, (samples, values) = jax.lax.scan(
            step,
            (initial, initial_log, key, jnp.asarray(0)),
            None,
            length=self.sample_count,
        )
        return PCNSamplingResult(
            samples,
            values,
            final[3] / self.sample_count,
            final[2],
        )


__all__ = [
    "EnsembleInversionResult",
    "EnsembleKalmanInversionPlan",
    "MAPResult",
    "MatrixFreeMAPPlan",
    "PCNSampler",
    "PCNSamplingResult",
]
