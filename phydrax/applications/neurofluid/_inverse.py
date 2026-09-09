#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Image-space likelihoods and identifiable PDE-constrained neurofluid inversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bounds import Bounds
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, pseudoinverse, svd as svd_api
from ...observation import ObservationRecord
from ...optim import StateDesignProblem
from ...spatial_sampling import PreparedObservationOperator


class NeurofluidForward(Protocol):
    def __call__(self, design: Array, /) -> Array: ...


class ImageSpaceLikelihoodEvidence(StrictModule):
    valid_count: Array
    finite: Array
    coverage_complete: Array
    successful: Array


class ImageSpaceLikelihoodResult(StrictModule):
    negative_log_likelihood: Array
    prediction: Array
    standardized_residual: Array
    evidence: ImageSpaceLikelihoodEvidence
    likelihood_id: str = eqx.field(static=True)


class ImageSpaceObservation(StrictModule):
    operator: PreparedObservationOperator
    observed: Array
    valid: Array
    standard_deviation: Array
    likelihood_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: PreparedObservationOperator,
        observed: ArrayLike,
        valid: ArrayLike,
        standard_deviation: ArrayLike,
        /,
        *,
        observation_id: str,
    ):
        if not isinstance(operator, PreparedObservationOperator):
            raise TypeError("operator must be PreparedObservationOperator.")
        observed_ = jnp.asarray(observed)
        valid_ = jnp.asarray(valid, dtype=bool)
        deviation = jnp.asarray(standard_deviation)
        if observed_.shape != operator.query_shape or valid_.shape != observed_.shape:
            raise ValueError(
                "Observed values and validity must match the operator query shape."
            )
        if deviation.shape == ():
            deviation = jnp.broadcast_to(deviation, observed_.shape)
        if deviation.shape != observed_.shape:
            raise ValueError("standard_deviation must be scalar or match observations.")
        deviation = eqx.error_if(
            deviation,
            jnp.any(~jnp.isfinite(deviation)) | jnp.any(deviation <= 0.0),
            "Observation standard deviations must be finite and positive.",
        )
        if not bool(np.any(np.asarray(valid_))):
            raise ValueError(
                "Image-space observations require at least one valid sample."
            )
        identifier = str(observation_id).strip()
        if not identifier:
            raise ValueError("observation_id must be non-empty.")
        self.operator = operator
        self.observed = observed_
        self.valid = valid_
        self.standard_deviation = deviation
        self.likelihood_id = canonical_fingerprint(
            {
                "kind": "image-space-observation",
                "observation": identifier,
                "operator": operator.prepared_id,
                "observed": array_tree_fingerprint(observed_),
                "valid": array_tree_fingerprint(valid_),
                "standard_deviation": array_tree_fingerprint(deviation),
            }
        )

    @classmethod
    def from_record(
        cls,
        operator: PreparedObservationOperator,
        record: ObservationRecord,
        standard_deviation: ArrayLike,
        /,
    ) -> ImageSpaceObservation:
        if not isinstance(record, ObservationRecord):
            raise TypeError("record must be ObservationRecord.")
        return cls(
            operator,
            record.values,
            record.valid_mask,
            standard_deviation,
            observation_id=record.record_id,
        )

    def evaluate(self, source_state: ArrayLike, /) -> ImageSpaceLikelihoodResult:
        candidate = self.operator.apply(source_state)
        prediction = candidate.values
        residual = jnp.where(
            self.valid, (prediction - self.observed) / self.standard_deviation, 0.0
        )
        likelihood = 0.5 * jnp.sum(residual * residual)
        finite = jnp.all(jnp.where(self.valid, jnp.isfinite(residual), True))
        evidence = ImageSpaceLikelihoodEvidence(
            jnp.sum(self.valid, dtype=jnp.int32),
            finite,
            candidate.evidence.complete_coverage,
            finite & candidate.evidence.successful,
        )
        return ImageSpaceLikelihoodResult(
            likelihood, prediction, residual, evidence, self.likelihood_id
        )


@dataclass(frozen=True, slots=True)
class NeurofluidParameterSchema:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    initial: np.ndarray
    schema_id: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(value).strip() for value in self.names)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Parameter names must be unique and non-empty.")
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        initial = np.asarray(self.initial, dtype=float)
        shape = (len(names),)
        if lower.shape != shape or upper.shape != shape or initial.shape != shape:
            raise ValueError("Parameter bounds and initial values must match names.")
        if (
            np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(~np.isfinite(initial))
        ):
            raise ValueError("Parameter arrays must be finite.")
        if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
            raise ValueError(
                "Parameter bounds must be ordered and contain the initial value."
            )
        for name, value in (("lower", lower), ("upper", upper), ("initial", initial)):
            value = np.array(value, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "schema_id",
            canonical_fingerprint(
                {
                    "kind": "neurofluid-parameter-schema",
                    "names": list(names),
                    "lower": array_tree_fingerprint(lower),
                    "upper": array_tree_fingerprint(upper),
                    "initial": array_tree_fingerprint(initial),
                }
            ),
        )

    @property
    def bounds(self) -> Bounds:
        return Bounds(jnp.asarray(self.lower), jnp.asarray(self.upper))


class IdentifiabilityReport(StrictModule):
    singular_values: Array
    numerical_rank: Array
    condition_number: Array
    posterior_covariance: Array
    full_rank: Array
    finite: Array
    report_id: str = eqx.field(static=True)


class NeurofluidInverseProblem(StrictModule):
    schema: NeurofluidParameterSchema
    observation: ImageSpaceObservation
    forward: NeurofluidForward
    regularization: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: NeurofluidParameterSchema,
        observation: ImageSpaceObservation,
        forward: NeurofluidForward,
        /,
        *,
        regularization: float = 0.0,
    ):
        if not isinstance(schema, NeurofluidParameterSchema):
            raise TypeError("schema must be NeurofluidParameterSchema.")
        if not isinstance(observation, ImageSpaceObservation):
            raise TypeError("observation must be ImageSpaceObservation.")
        if not callable(forward):
            raise TypeError("forward must be callable.")
        penalty = float(regularization)
        if not np.isfinite(penalty) or penalty < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        self.schema = schema
        self.observation = observation
        self.forward = forward
        self.regularization = penalty
        self.problem_id = canonical_fingerprint(
            {
                "kind": "neurofluid-inverse-problem",
                "schema": schema.schema_id,
                "observation": observation.likelihood_id,
                "regularization": penalty.hex(),
            }
        )

    def state_design_problem(self) -> StateDesignProblem:
        initial = jnp.asarray(self.schema.initial)

        def residual(state, design, args=None):
            return state - self.forward(design)

        def objective(state, design, args=None):
            likelihood = self.observation.evaluate(state).negative_log_likelihood
            difference = design - initial
            return (
                likelihood
                + 0.5 * self.regularization * jnp.vdot(difference, difference).real
            )

        return StateDesignProblem(
            residual,
            objective,
            design_bounds=self.schema.bounds,
            state_admissibility=lambda state, design, args: jnp.all(jnp.isfinite(state)),
            problem_id=self.problem_id,
        )

    def identifiability(
        self, design: ArrayLike, /, *, rank_tolerance: float = 1.0e-8
    ) -> IdentifiabilityReport:
        parameters = jnp.asarray(design)
        if parameters.shape != (len(self.schema.names),):
            raise ValueError("design must match the parameter schema.")
        tolerance = float(rank_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive.")

        def prediction(value):
            return self.observation.operator.apply(self.forward(value)).values.reshape(
                (-1,)
            )

        jacobian = jax.jacrev(prediction)(parameters)
        valid = self.observation.valid.reshape((-1,))
        weighted = (
            jacobian[valid]
            / self.observation.standard_deviation.reshape((-1,))[valid, None]
        )
        decomposition = svd_api.svd(
            svd_api.SVDProblem(DenseLinearOperator(weighted)),
            policy=svd_api.SVDSolvePolicy(count=min(weighted.shape)),
        )
        singular = decomposition.singular_values
        threshold = tolerance * jnp.maximum(singular[0], 1.0)
        rank = jnp.sum(singular > threshold, dtype=jnp.int32)
        condition = singular[0] / jnp.where(
            singular[-1] > threshold, singular[-1], jnp.inf
        )
        fisher = weighted.T @ weighted + self.regularization * jnp.eye(parameters.size)
        covariance_result = pseudoinverse(fisher)
        covariance = covariance_result.value
        finite = (
            jnp.all(jnp.isfinite(singular))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.all(decomposition.successful)
            & jnp.all(covariance_result.successful)
        )
        full_rank = rank == parameters.size
        return IdentifiabilityReport(
            singular,
            rank,
            condition,
            covariance,
            full_rank,
            finite,
            canonical_fingerprint(
                {
                    "kind": "neurofluid-identifiability",
                    "problem": self.problem_id,
                    "tolerance": tolerance.hex(),
                }
            ),
        )


__all__ = [
    "IdentifiabilityReport",
    "NeurofluidForward",
    "ImageSpaceLikelihoodEvidence",
    "ImageSpaceLikelihoodResult",
    "ImageSpaceObservation",
    "NeurofluidInverseProblem",
    "NeurofluidParameterSchema",
]
