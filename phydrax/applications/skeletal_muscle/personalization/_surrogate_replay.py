#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Causal exact-model replay acceptance for skeletal surrogate decisions."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....control import (
    AbstractControlParameterization,
    ControlProblem,
    ControlResult,
    ControlTrajectory,
)


class SkeletalReplayObservationOperator(StrictModule, NonTrainableState):
    """Immutable identified projection from an exact control trajectory."""

    observation: Callable[[ControlTrajectory], ArrayLike] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        observation: Callable[[ControlTrajectory], ArrayLike],
        operator_id: str,
        /,
    ):
        if not callable(observation):
            raise TypeError("observation must be callable.")
        identifier = str(operator_id).strip()
        if not identifier:
            raise ValueError("operator_id must be nonempty.")
        self.observation = observation
        self.operator_id = identifier

    def evaluate(self, trajectory: ControlTrajectory, /) -> ArrayLike:
        if not isinstance(trajectory, ControlTrajectory):
            raise TypeError("trajectory must be ControlTrajectory.")
        return self.observation(trajectory)


class SkeletalSurrogateReplayEvidence(StrictModule, NonTrainableState):
    exact_result: ControlResult
    exact_values: Array
    surrogate_values: Array
    finite: Array
    active_sample_count: Array
    maximum_absolute_error: Array
    maximum_relative_error: Array
    absolute_tolerance: Array
    relative_tolerance: Array
    accepted: Array
    source_problem_id: str = eqx.field(static=True)
    source_dynamics_id: str = eqx.field(static=True)
    source_control_id: str = eqx.field(static=True)
    surrogate_id: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)
    observation_operator_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)


class SkeletalSurrogateReplayPlan(StrictModule):
    """Replay candidate controls through one identified exact ControlProblem."""

    source_problem: ControlProblem
    parameterization: AbstractControlParameterization
    observation_operator: SkeletalReplayObservationOperator
    valid_mask: Array
    surrogate_id: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_problem: ControlProblem,
        parameterization: AbstractControlParameterization,
        observation_operator: SkeletalReplayObservationOperator,
        valid_mask: ArrayLike,
        surrogate_id: str,
        quantity_id: str,
        /,
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
    ):
        if not isinstance(source_problem, ControlProblem):
            raise TypeError("source_problem must be ControlProblem.")
        if not isinstance(parameterization, AbstractControlParameterization):
            raise TypeError("parameterization must implement AbstractControlParameterization.")
        if parameterization.control_shape != source_problem.control_shape:
            raise ValueError("Parameterization and source problem control shapes differ.")
        if not isinstance(observation_operator, SkeletalReplayObservationOperator):
            raise TypeError(
                "observation_operator must be SkeletalReplayObservationOperator."
            )
        identifiers = tuple(str(value).strip() for value in (surrogate_id, quantity_id))
        if any(not value for value in identifiers):
            raise ValueError("Surrogate and quantity IDs must be nonempty.")
        mask = jnp.asarray(valid_mask, dtype=bool)
        if mask.ndim == 0 or not bool(np.any(np.asarray(mask))):
            raise ValueError("valid_mask must contain at least one active sample.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not isfinite(absolute)
            or absolute < 0.0
            or not isfinite(relative)
            or relative < 0.0
            or (absolute == 0.0 and relative == 0.0)
        ):
            raise ValueError("At least one finite nonnegative replay tolerance is required.")
        self.source_problem = source_problem
        self.parameterization = parameterization
        self.observation_operator = observation_operator
        self.valid_mask = mask
        self.surrogate_id, self.quantity_id = identifiers
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "skeletal-surrogate-causal-exact-replay",
                "source_problem_id": source_problem.problem_id,
                "source_dynamics_id": source_problem.dynamics.dynamics_id,
                "parameterization_id": parameterization.parameterization_id,
                "surrogate_id": identifiers[0],
                "quantity_id": identifiers[1],
                "observation_operator_id": observation_operator.operator_id,
                "valid_mask": array_tree_fingerprint(mask),
                "absolute_tolerance": absolute.hex(),
                "relative_tolerance": relative.hex(),
            }
        )

    def evaluate(
        self,
        control_coefficients: ArrayLike,
        surrogate_values: ArrayLike,
        /,
    ) -> SkeletalSurrogateReplayEvidence:
        coefficients = jnp.asarray(control_coefficients)
        if coefficients.shape != self.parameterization.parameter_shape:
            raise ValueError(
                "control_coefficients must match the prepared parameterization shape."
            )
        result = self.source_problem.evaluate(
            self.parameterization,
            coefficients,
        )
        trajectory = result.trajectory
        exact_input = jnp.asarray(self.observation_operator.evaluate(trajectory))
        surrogate_input = jnp.asarray(surrogate_values)
        if jnp.issubdtype(exact_input.dtype, jnp.complexfloating) or jnp.issubdtype(
            surrogate_input.dtype, jnp.complexfloating
        ):
            raise TypeError("Exact replay and surrogate predictions must be real-valued.")
        comparison_dtype = jnp.result_type(exact_input, surrogate_input, float)
        exact = exact_input.astype(comparison_dtype)
        surrogate = surrogate_input.astype(comparison_dtype)
        if exact.shape != self.valid_mask.shape or surrogate.shape != exact.shape:
            raise ValueError(
                "Exact replay, surrogate prediction, and valid-mask shapes must agree."
            )
        mask = self.valid_mask
        active_count = jnp.sum(mask)
        active_exact = jnp.where(mask, exact, 0.0)
        active_surrogate = jnp.where(mask, surrogate, 0.0)
        difference = jnp.abs(active_surrogate - active_exact)
        denominator = jnp.maximum(jnp.abs(active_exact), self.absolute_tolerance)
        denominator_is_positive = denominator > 0.0
        safe_denominator = jnp.where(denominator_is_positive, denominator, 1.0)
        relative = jnp.where(
            mask,
            jnp.where(
                denominator_is_positive,
                difference / safe_denominator,
                jnp.where(difference == 0.0, 0.0, jnp.inf),
            ),
            0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(active_exact) & jnp.isfinite(active_surrogate))
            & jnp.all(jnp.isfinite(relative))
        )
        maximum_absolute = jnp.max(difference)
        maximum_relative = jnp.max(relative)
        accepted = (
            result.successful
            & finite
            & (active_count > 0)
            & jnp.all(
                ~mask
                | (difference <= self.absolute_tolerance)
                | (relative <= self.relative_tolerance)
            )
        )
        return SkeletalSurrogateReplayEvidence(
            result,
            exact,
            surrogate,
            finite,
            active_count,
            maximum_absolute,
            maximum_relative,
            jnp.asarray(self.absolute_tolerance, dtype=exact.dtype),
            jnp.asarray(self.relative_tolerance, dtype=exact.dtype),
            accepted,
            self.source_problem.problem_id,
            self.source_problem.dynamics.dynamics_id,
            self.parameterization.parameterization_id,
            self.surrogate_id,
            self.quantity_id,
            self.observation_operator.operator_id,
            canonical_fingerprint(
                {
                    "kind": "skeletal-surrogate-replay-result",
                    "plan": self.plan_id,
                    "source_problem": self.source_problem.problem_id,
                    "source_control": self.parameterization.parameterization_id,
                    "observation_operator": self.observation_operator.operator_id,
                }
            ),
        )


__all__ = [
    "SkeletalReplayObservationOperator",
    "SkeletalSurrogateReplayEvidence",
    "SkeletalSurrogateReplayPlan",
]
