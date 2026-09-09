#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import sqrt
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    EuclideanPairing,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


StochasticDesignCriterion = Literal["a-optimal", "d-optimal", "e-optimal"]


class StochasticDesignResult(StrictModule):
    objective: Array
    estimate: Array
    standard_error: Array
    probe_values: Array
    successful: Array
    lanczos_breakdown: Array


class StochasticExperimentDesignPlan(StrictModule, NonTrainableState):
    """Fixed-probe matrix-free A/D/E design objective.

    ``precision_factory(design)`` returns the positive-definite parameter-space
    posterior precision for that design. Rademacher probes are generated once at
    construction; objective gradients therefore differentiate a deterministic
    approximation rather than resampling noise. No identity basis is materialized.
    ``objective`` is a maximization score: negative inverse trace for A-optimal,
    log-determinant for D-optimal, and minimum Ritz value for E-optimal design.
    """

    precision_factory: Callable[[Array], AbstractLinearOperator] = eqx.field(static=True)
    precision_factory_id: str = eqx.field(static=True)
    probes: Array
    criterion: StochasticDesignCriterion = eqx.field(static=True)
    lanczos_steps: int = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    linear_policy_id: str | None = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        precision_factory: Callable[[Array], AbstractLinearOperator],
        design_template: ArrayLike,
        key: PRNGKeyArray,
        /,
        *,
        precision_factory_id: str,
        parameter_dimension: int,
        probe_count: int,
        criterion: StochasticDesignCriterion,
        lanczos_steps: int = 32,
        linear_policy: LinearSolvePolicy | None = None,
        linear_policy_id: str | None = None,
    ):
        if not callable(precision_factory):
            raise TypeError("precision_factory must be callable.")
        factory_id = str(precision_factory_id).strip()
        if not factory_id:
            raise ValueError("Precision factory identity must be nonempty.")
        dimension, probes, steps = (
            int(parameter_dimension),
            int(probe_count),
            int(lanczos_steps),
        )
        if dimension <= 0 or probes <= 1 or steps <= 0:
            raise ValueError(
                "Design dimension/steps must be positive and probes greater than one."
            )
        if criterion not in ("a-optimal", "d-optimal", "e-optimal"):
            raise ValueError("Unknown stochastic design criterion.")
        policy_id = None if linear_policy_id is None else str(linear_policy_id).strip()
        if criterion == "a-optimal":
            if not isinstance(linear_policy, LinearSolvePolicy) or not policy_id:
                raise TypeError(
                    "Stochastic A-optimality requires a native solve policy and identity."
                )
        elif linear_policy is not None or linear_policy_id is not None:
            raise ValueError("D/E-optimal Lanczos designs do not consume a solve policy.")
        template = jnp.asarray(design_template)
        if bool(jnp.any(~jnp.isfinite(template))):
            raise ValueError("Design template must be finite.")
        sample = precision_factory(template)
        if not isinstance(sample, AbstractLinearOperator):
            raise TypeError("precision_factory must return AbstractLinearOperator.")
        if (
            not isinstance(sample.source, ArraySpace)
            or not isinstance(sample.source.pairing, EuclideanPairing)
            or sample.source.size != dimension
            or not sample.source.compatible(sample.target)
        ):
            raise ValueError(
                "Design precision must be a Euclidean ArraySpace endomorphism with declared dimension."
            )
        if not sample.properties.certifies("positive_definite"):
            raise ValueError(
                "Design precision must carry positive-definite operator evidence."
            )
        bits = jax.random.bernoulli(key, 0.5, shape=(probes, dimension))
        self.precision_factory = precision_factory
        self.precision_factory_id = factory_id
        self.probes = jnp.where(bits, 1.0, -1.0)
        self.criterion, self.lanczos_steps = criterion, min(steps, dimension)
        self.linear_policy, self.linear_policy_id = linear_policy, policy_id
        self.dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "criterion": criterion,
                "kind": "stochastic-experiment-design",
                "precision_factory": factory_id,
                "sample_operator": sample.operator_id,
                "dimension": dimension,
                "lanczos_steps": self.lanczos_steps,
                "probes": array_tree_fingerprint(self.probes),
                "design_template": array_tree_fingerprint(template),
                "linear_policy": policy_id,
            }
        )

    def _flat_action(self, operator, vector):
        structured = operator.source.unflatten(vector)
        return operator.target.flatten(operator.mv(structured))

    def _a_value(self, operator, probe):
        if self.linear_policy is None:
            raise RuntimeError("A-optimal design lost its required linear solve policy.")
        result = solve(
            LinearSystem(operator),
            operator.target.unflatten(probe),
            policy=self.linear_policy,
        )
        solution = operator.source.flatten(result.value)
        return jnp.real(jnp.vdot(probe, solution)), result.successful

    def _lanczos(self, operator, probe):
        dtype = jnp.result_type(probe.dtype, float)
        q = probe.astype(dtype) / jnp.sqrt(jnp.asarray(self.dimension, dtype=dtype))
        previous = jnp.zeros_like(q)
        beta = jnp.asarray(0.0, dtype=dtype)
        alphas: list[Array] = []
        betas: list[Array] = []
        active = jnp.asarray(True)
        breakdown = jnp.asarray(False)
        tolerance = 100 * jnp.finfo(dtype).eps
        for step in range(self.lanczos_steps):
            action = self._flat_action(operator, q)
            residual = action - beta * previous
            alpha = jnp.real(jnp.vdot(q, residual))
            residual = residual - alpha * q
            next_beta = jnp.sqrt(jnp.real(jnp.vdot(residual, residual)))
            alphas.append(jnp.where(active, alpha, -jnp.asarray(step + 1, dtype=dtype)))
            continues = active & (next_beta > tolerance)
            if step + 1 < self.lanczos_steps:
                betas.append(jnp.where(continues, next_beta, 0.0))
            safe = jnp.where(continues, next_beta, 1.0)
            next_q = residual / safe
            previous = jnp.where(continues, q, previous)
            q = jnp.where(continues, next_q, q)
            breakdown = breakdown | (active & ~continues)
            active = continues
            beta = jnp.where(continues, next_beta, 0.0)
        diagonal = jnp.stack(alphas)
        tridiagonal = jnp.diag(diagonal)
        if betas:
            off_diagonal = jnp.stack(betas)
            tridiagonal = (
                tridiagonal + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
            )
        eigenvalues, eigenvectors = jnp.linalg.eigh(tridiagonal)
        weights = eigenvectors[0] ** 2
        supported = weights > tolerance
        positive = jnp.all(jnp.where(supported, eigenvalues > 0, True))
        safe_eigenvalues = jnp.where(supported & (eigenvalues > 0), eigenvalues, 1.0)
        log_quadrature = self.dimension * jnp.sum(
            jnp.where(supported, weights * jnp.log(safe_eigenvalues), 0.0)
        )
        minimum_ritz = jnp.min(jnp.where(supported, eigenvalues, jnp.inf))
        return log_quadrature, minimum_ritz, breakdown, ~positive

    def evaluate(self, design: ArrayLike, /) -> StochasticDesignResult:
        design_ = jnp.asarray(design)
        if bool(jnp.any(~jnp.isfinite(design_))):
            raise ValueError("Experiment design must be finite.")
        operator = self.precision_factory(design_)
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("precision_factory must return AbstractLinearOperator.")
        if (
            not isinstance(operator.source, ArraySpace)
            or not isinstance(operator.source.pairing, EuclideanPairing)
            or operator.source.size != self.dimension
            or not operator.source.compatible(operator.target)
        ):
            raise ValueError("Design changed the declared Euclidean precision space.")
        if not operator.properties.certifies("positive_definite"):
            raise ValueError("Design precision lost positive-definite evidence.")
        if self.criterion == "a-optimal":
            rows = [self._a_value(operator, probe) for probe in self.probes]
            values = jnp.stack([row[0] for row in rows])
            successful = jnp.all(jnp.stack([row[1] for row in rows]))
            breakdown = jnp.asarray(False)
            estimate = jnp.mean(values)
            objective = -estimate
        else:
            rows = [self._lanczos(operator, probe) for probe in self.probes]
            log_values = jnp.stack([row[0] for row in rows])
            minimum = jnp.stack([row[1] for row in rows])
            breakdown = jnp.any(jnp.stack([row[2] for row in rows]))
            invalid = jnp.any(jnp.stack([row[3] for row in rows]))
            if self.criterion == "d-optimal":
                values = log_values
                estimate = jnp.mean(values)
            else:
                values = minimum
                estimate = jnp.min(values)
            objective = estimate
            successful = ~invalid
        standard_error = jnp.std(values, ddof=1) / sqrt(values.size)
        successful = successful & jnp.isfinite(estimate) & jnp.isfinite(standard_error)
        return StochasticDesignResult(
            objective,
            estimate,
            standard_error,
            values,
            successful,
            breakdown,
        )

    def objective(self, design: ArrayLike, /) -> Array:
        return self.evaluate(design).objective


__all__ = [
    "StochasticDesignCriterion",
    "StochasticDesignResult",
    "StochasticExperimentDesignPlan",
]
