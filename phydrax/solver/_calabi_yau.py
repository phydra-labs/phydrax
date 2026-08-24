#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..geometry.complex import (
    HypersurfaceKahlerEvaluation,
    HypersurfaceKahlerGeometry,
    ProjectiveHypersurface,
    ProjectiveLineSamples,
)


class CalabiYauMetricProblem(StrictModule):
    hypersurface: ProjectiveHypersurface
    samples: ProjectiveLineSamples
    potential_model: Any
    weights: Array
    normalization: Array
    positivity_floor: float
    precision: GeometryPrecisionPolicy

    def __init__(
        self,
        hypersurface: ProjectiveHypersurface,
        samples: ProjectiveLineSamples,
        potential_model: Any,
        /,
        *,
        weights: ArrayLike | None = None,
        normalization: ArrayLike = 0.0,
        positivity_floor: float = 1e-7,
        precision: GeometryPrecisionPolicy | None = None,
    ):
        if not isinstance(hypersurface, ProjectiveHypersurface):
            raise TypeError("hypersurface must be a ProjectiveHypersurface.")
        if not isinstance(samples, ProjectiveLineSamples):
            raise TypeError("samples must be ProjectiveLineSamples.")
        if not callable(potential_model):
            raise TypeError("potential_model must be callable.")
        count = samples.homogeneous_points.shape[0]
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        weights_ = precision_.accumulation(
            jnp.ones((count,)) / float(count) if weights is None else weights
        )
        if weights_.shape != (count,):
            raise ValueError("weights must match the sample axis.")
        weights_ = precision_.accumulation(weights_ / jnp.sum(weights_))
        self.hypersurface = hypersurface
        self.samples = samples
        self.potential_model = potential_model
        self.weights = weights_
        self.normalization = precision_.compute(normalization).reshape(())
        self.positivity_floor = float(positivity_floor)
        self.precision = precision_


class CalabiYauSolvePolicy(StrictModule):
    iterations: int
    learning_rate: float
    maximum_backtracks: int
    contraction: float
    gradient_tolerance: float

    def __init__(
        self,
        *,
        iterations: int = 50,
        learning_rate: float = 1e-2,
        maximum_backtracks: int = 8,
        contraction: float = 0.5,
        gradient_tolerance: float = 1e-8,
    ):
        if int(iterations) < 0 or int(maximum_backtracks) < 0:
            raise ValueError("Iteration counts must be non-negative.")
        if learning_rate <= 0.0 or not 0.0 < contraction < 1.0:
            raise ValueError("Learning-rate and contraction policies are invalid.")
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.maximum_backtracks = int(maximum_backtracks)
        self.contraction = float(contraction)
        self.gradient_tolerance = float(gradient_tolerance)


class CalabiYauMetricResult(StrictModule):
    potential_model: Any
    normalization: Array
    objective_history: Array
    residual_history: Array
    positivity_history: Array
    accepted_history: Array
    valid: Array
    converged: Array
    iteration_count: int
    hypersurface_id: str
    precision_evidence: PrecisionEvidenceEnvelope
    precision: GeometryPrecisionPolicy

    def __init__(
        self,
        potential_model: Any,
        normalization: ArrayLike,
        objective_history: ArrayLike,
        residual_history: ArrayLike,
        positivity_history: ArrayLike,
        accepted_history: ArrayLike,
        /,
        *,
        converged: ArrayLike,
        hypersurface_id: str,
        precision_evidence: PrecisionEvidenceEnvelope,
        precision: GeometryPrecisionPolicy,
    ):
        self.potential_model = potential_model
        self.normalization = jnp.asarray(normalization)
        self.objective_history = jnp.asarray(objective_history)
        self.residual_history = jnp.asarray(residual_history)
        self.positivity_history = jnp.asarray(positivity_history)
        self.accepted_history = jnp.asarray(accepted_history, dtype=bool)
        self.valid = (
            jnp.all(jnp.isfinite(self.objective_history))
            & jnp.all(jnp.isfinite(self.residual_history))
            & jnp.all(jnp.isfinite(self.positivity_history))
        )
        self.converged = jnp.asarray(converged, dtype=bool)
        self.iteration_count = int(self.objective_history.shape[0])
        self.hypersurface_id = str(hypersurface_id)
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        self.precision_evidence = precision_evidence
        if not isinstance(precision, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy.")
        self.precision = precision

    def evaluate(
        self,
        hypersurface: ProjectiveHypersurface,
        homogeneous_point: ArrayLike,
        /,
    ) -> HypersurfaceKahlerEvaluation:
        if hypersurface.hypersurface_id != self.hypersurface_id:
            raise ValueError("Artifact hypersurface identity does not match.")
        geometry = HypersurfaceKahlerGeometry(
            hypersurface,
            self.potential_model,
            normalization=self.normalization,
        )
        return geometry.evaluate(homogeneous_point)


def _objective(
    hypersurface: ProjectiveHypersurface,
    samples: ProjectiveLineSamples,
    weights: Array,
    normalization: Array,
    positivity_floor: float,
    potential_model: Any,
    precision: GeometryPrecisionPolicy,
):
    geometry = HypersurfaceKahlerGeometry(
        hypersurface,
        potential_model,
        normalization=normalization,
        positivity_floor=positivity_floor,
    )
    residuals = []
    margins = []
    potentials = []
    valid = []
    for index in range(samples.homogeneous_points.shape[0]):
        evaluation = geometry.evaluate(
            samples.homogeneous_points[index],
            chart_index=int(samples.chart_indices[index]),
            pivot_index=int(samples.pivot_indices[index]),
        )
        residuals.append(evaluation.monge_ampere_residual)
        margins.append(evaluation.positivity_margin)
        potentials.append(evaluation.potential)
        valid.append(evaluation.valid)
    residual = precision.compute(jnp.stack(residuals))
    margin = precision.decision(jnp.stack(margins))
    potential = precision.compute(jnp.stack(potentials))
    validity = jnp.stack(valid)
    accumulated_weights = precision.accumulation(weights)
    mean_potential = jnp.sum(accumulated_weights * precision.accumulation(potential))
    equation = jnp.sum(accumulated_weights * precision.accumulation(residual**2))
    gauge = mean_potential**2
    objective = precision.decision(equation + gauge)
    return objective, (jnp.sqrt(equation), jnp.min(margin), jnp.all(validity))


def solve_calabi_yau_metric(
    problem: CalabiYauMetricProblem,
    /,
    *,
    policy: CalabiYauSolvePolicy | None = None,
) -> CalabiYauMetricResult:
    if not isinstance(problem, CalabiYauMetricProblem):
        raise TypeError("problem must be a CalabiYauMetricProblem.")
    policy_ = CalabiYauSolvePolicy() if policy is None else policy
    if not isinstance(policy_, CalabiYauSolvePolicy):
        raise TypeError("policy must be a CalabiYauSolvePolicy.")
    model = problem.potential_model
    objectives = []
    residuals = []
    margins = []
    accepted = []
    converged = False

    def objective(candidate):
        return _objective(
            problem.hypersurface,
            problem.samples,
            problem.weights,
            problem.normalization,
            problem.positivity_floor,
            candidate,
            problem.precision,
        )

    value_and_grad = eqx.filter_value_and_grad(objective, has_aux=True)
    for _ in range(policy_.iterations):
        (value, auxiliary), gradient = value_and_grad(model)
        residual, margin, validity = auxiliary
        gradient_norm = problem.precision.decision(
            jnp.sqrt(
                sum(
                    jnp.real(
                        jnp.vdot(
                            problem.precision.accumulation(leaf),
                            problem.precision.accumulation(leaf),
                        )
                    )
                    for leaf in jax.tree.leaves(eqx.filter(gradient, eqx.is_array))
                )
            )
        )
        step = policy_.learning_rate
        candidate = model
        candidate_value = value
        candidate_auxiliary = auxiliary
        did_accept = False
        for _ in range(policy_.maximum_backtracks + 1):
            candidate = eqx.apply_updates(
                model,
                jax.tree.map(
                    lambda leaf: None if leaf is None else -step * leaf,
                    gradient,
                    is_leaf=lambda leaf: leaf is None,
                ),
            )
            candidate_value, candidate_auxiliary = objective(candidate)
            if bool(
                jax.device_get(
                    (
                        problem.precision.decision(candidate_value)
                        <= problem.precision.decision(value)
                    )
                    & candidate_auxiliary[2]
                    & (
                        candidate_auxiliary[1]
                        > problem.precision.decision(problem.positivity_floor)
                    )
                )
            ):
                did_accept = True
                break
            step *= policy_.contraction
        if did_accept:
            model = candidate
            value = candidate_value
            residual, margin, _ = candidate_auxiliary
        objectives.append(value)
        residuals.append(residual)
        margins.append(margin)
        accepted.append(did_accept)
        if bool(
            jax.device_get(
                gradient_norm <= problem.precision.decision(policy_.gradient_tolerance)
            )
        ):
            converged = True
            break
    return CalabiYauMetricResult(
        model,
        problem.normalization,
        problem.precision.output(
            jnp.stack(objectives) if objectives else jnp.zeros((0,))
        ),
        problem.precision.output(jnp.stack(residuals) if residuals else jnp.zeros((0,))),
        problem.precision.output(jnp.stack(margins) if margins else jnp.zeros((0,))),
        jnp.asarray(accepted),
        converged=converged,
        hypersurface_id=problem.hypersurface.hypersurface_id,
        precision_evidence=problem.precision.evidence_for(problem.weights),
        precision=problem.precision,
    )


__all__ = [
    "CalabiYauMetricProblem",
    "CalabiYauMetricResult",
    "CalabiYauSolvePolicy",
    "solve_calabi_yau_metric",
]
