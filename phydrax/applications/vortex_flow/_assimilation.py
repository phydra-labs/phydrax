#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    LinearSolveResult,
    solve as solve_linear,
)


class VortexObservationSet(StrictModule, NonTrainableState):
    operator: Array
    values: Array
    standard_deviation: Array
    kind: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: ArrayLike,
        values: ArrayLike,
        standard_deviation: ArrayLike,
        /,
        *,
        kind: str,
    ):
        matrix, value, deviation = (
            jnp.asarray(operator),
            jnp.asarray(values),
            jnp.asarray(standard_deviation),
        )
        if (
            matrix.ndim != 2
            or value.shape != (matrix.shape[0],)
            or deviation.shape != value.shape
            or jnp.any(deviation <= 0.0)
            or kind not in ("velocity", "vorticity", "load", "particle-track", "pressure")
        ):
            raise ValueError("Vortex observation set is invalid.")
        self.operator, self.values, self.standard_deviation, self.kind = (
            matrix,
            value,
            deviation,
            kind,
        )
        self.observation_id = canonical_fingerprint(
            {
                "kind": "vortex-observation-set",
                "observation_kind": kind,
                "operator": array_tree_fingerprint(matrix),
                "values": array_tree_fingerprint(value),
                "standard_deviation": array_tree_fingerprint(deviation),
            }
        )


class VortexAssimilationResult(StrictModule):
    state: Array
    innovation: Array
    weighted_residual_norm: Array
    prior_residual_norm: Array
    linear_result: LinearSolveResult
    successful: Array
    assimilation_id: str = eqx.field(static=True)


class VortexDataAssimilationPlan(StrictModule, NonTrainableState):
    observations: tuple[VortexObservationSet, ...]
    prior_precision: Array
    policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        observations: tuple[VortexObservationSet, ...],
        prior_precision: ArrayLike,
        /,
        *,
        policy: LinearSolvePolicy | None = None,
    ):
        if not observations or any(
            not isinstance(observation, VortexObservationSet)
            for observation in observations
        ):
            raise ValueError("Assimilation requires observation sets.")
        state_size = observations[0].operator.shape[1]
        if any(
            observation.operator.shape[1] != state_size for observation in observations
        ):
            raise ValueError("Observation operators must share one state size.")
        precision = jnp.asarray(prior_precision)
        if precision.shape == ():
            precision = jnp.full((state_size,), precision)
        if precision.shape != (state_size,) or jnp.any(precision < 0.0):
            raise ValueError("Prior precision must be nonnegative scalar/vector.")
        self.observations, self.prior_precision = observations, precision
        self.policy = LinearSolvePolicy(DenseSVD()) if policy is None else policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "vortex-data-assimilation",
                "observations": [
                    observation.observation_id for observation in observations
                ],
                "state_size": state_size,
            }
        )

    def assimilate(self, prior_state: ArrayLike, /) -> VortexAssimilationResult:
        prior = jnp.asarray(prior_state)
        if prior.shape != self.prior_precision.shape:
            raise ValueError("Prior state shape is incompatible.")
        rows, values = [], []
        innovations = []
        for observation in self.observations:
            inverse_sigma = 1.0 / observation.standard_deviation
            rows.append(inverse_sigma[:, None] * observation.operator)
            values.append(inverse_sigma * observation.values)
            innovations.append(observation.values - observation.operator @ prior)
        prior_root = jnp.sqrt(self.prior_precision)
        rows.append(jnp.diag(prior_root))
        values.append(prior_root * prior)
        matrix, rhs = (
            jnp.concatenate(tuple(rows), axis=0),
            jnp.concatenate(tuple(values), axis=0),
        )
        linear = solve_linear(
            LeastSquaresProblem(
                DenseLinearOperator(matrix), problem_id=f"{self.plan_id}:analysis"
            ),
            rhs,
            policy=self.policy,
        )
        state = jnp.asarray(linear.value)
        weighted = jnp.linalg.norm(matrix @ state - rhs)
        prior_residual = jnp.linalg.norm(prior_root * (state - prior))
        successful = linear.successful & jnp.all(jnp.isfinite(state))
        return VortexAssimilationResult(
            state,
            jnp.concatenate(tuple(innovations)),
            weighted,
            prior_residual,
            linear,
            successful,
            self.plan_id,
        )


__all__ = [
    "VortexAssimilationResult",
    "VortexDataAssimilationPlan",
    "VortexObservationSet",
]
