#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._differentiation import ComponentAuthority, DerivativeRoute, ObjectiveKind
from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import combine_parameters
from .._training_kernel import (
    AbstractKernelUpdateRule,
    KernelObjective,
    KernelUpdateContext,
    prepare_training_kernel,
    run_training_attempt,
    TrainingAttemptOutcome,
    TrainingKernelSpec,
)
from .._training_objective import _ObjectiveContribution
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
        self.accepted_history = jnp.asarray(accepted_history, dtype=jnp.bool_)
        self.valid = (
            jnp.all(jnp.isfinite(self.objective_history))
            & jnp.all(jnp.isfinite(self.residual_history))
            & jnp.all(jnp.isfinite(self.positivity_history))
        )
        self.converged = jnp.asarray(converged, dtype=jnp.bool_)
        self.iteration_count = self.objective_history.shape[0]
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


class _CalabiYauPayload(StrictModule):
    """Attempt payload: the problem data the objective and the step guard read."""

    hypersurface: ProjectiveHypersurface
    homogeneous_points: Array
    weights: Array
    normalization: Array
    precision: GeometryPrecisionPolicy
    chart_indices: tuple[int, ...] = eqx.field(static=True)
    pivot_indices: tuple[int, ...] = eqx.field(static=True)
    positivity_floor: float = eqx.field(static=True)

    def __init__(self, problem: CalabiYauMetricProblem, /):
        self.hypersurface = problem.hypersurface
        self.homogeneous_points = problem.samples.homogeneous_points
        self.weights = problem.weights
        self.normalization = problem.normalization
        self.precision = problem.precision
        self.chart_indices = tuple(
            int(index) for index in jax.device_get(problem.samples.chart_indices)
        )
        self.pivot_indices = tuple(
            int(index) for index in jax.device_get(problem.samples.pivot_indices)
        )
        self.positivity_floor = problem.positivity_floor


class _CalabiYauDiagnostics(StrictModule):
    """Weighted Monge-Ampere residual norm, minimum positivity margin, validity."""

    residual: Array
    margin: Array
    valid: Array


def _calabi_yau_terms(
    potential_model: Any, payload: _CalabiYauPayload, /
) -> tuple[Array, _CalabiYauDiagnostics]:
    precision = payload.precision
    geometry = HypersurfaceKahlerGeometry(
        payload.hypersurface,
        potential_model,
        normalization=payload.normalization,
        positivity_floor=payload.positivity_floor,
    )
    residuals = []
    margins = []
    potentials = []
    valid = []
    for index, (chart, pivot) in enumerate(
        zip(payload.chart_indices, payload.pivot_indices, strict=True)
    ):
        evaluation = geometry.evaluate(
            payload.homogeneous_points[index],
            chart_index=chart,
            pivot_index=pivot,
        )
        residuals.append(evaluation.monge_ampere_residual)
        margins.append(evaluation.positivity_margin)
        potentials.append(evaluation.potential)
        valid.append(evaluation.valid)
    residual = precision.compute(jnp.stack(residuals))
    margin = precision.decision(jnp.stack(margins))
    potential = precision.compute(jnp.stack(potentials))
    validity = jnp.stack(valid)
    accumulated_weights = precision.accumulation(payload.weights)
    mean_potential = jnp.sum(accumulated_weights * precision.accumulation(potential))
    equation = jnp.sum(accumulated_weights * precision.accumulation(residual**2))
    gauge = mean_potential**2
    objective = precision.decision(equation + gauge)
    return objective, _CalabiYauDiagnostics(
        jnp.sqrt(equation), jnp.min(margin), jnp.all(validity)
    )


def _calabi_yau_objective(parameters, model_state, fixed, payload, keys, /):
    """Kernel objective: squared Monge-Ampere residual plus the potential gauge."""
    del keys
    objective, diagnostics = _calabi_yau_terms(
        combine_parameters(parameters, model_state, fixed), payload
    )
    contribution = _ObjectiveContribution(
        objective,
        jnp.ones((), objective.dtype),
        jnp.zeros((), objective.dtype),
    )
    return contribution, model_state, diagnostics


class _CalabiYauStepState(StrictModule):
    """Objective, residual norm, and margin measured at the last accepted trial."""

    value: Array
    residual: Array
    margin: Array


def _step_state(value: Array, diagnostics: _CalabiYauDiagnostics, /):
    return _CalabiYauStepState(
        jnp.asarray(value, jnp.float64),
        jnp.asarray(diagnostics.residual, jnp.float64),
        jnp.asarray(diagnostics.margin, jnp.float64),
    )


@final
class _CalabiYauBacktrackingRule(AbstractKernelUpdateRule):
    """Monotone gradient step with a geometry-validity guarded backtracking search.

    Each attempt tries `learning_rate * contraction**k` for
    `k <= maximum_backtracks` along the negative gradient and accepts the first
    trial whose objective does not exceed the current value, whose every sample
    evaluates to a valid chart point, and whose minimum positivity margin exceeds
    the positivity floor. The trial value comes from the kernel objective; the
    validity and margin come from the rule's guard evaluation of the payload.
    Every attempt restarts at the learning rate, so nothing commits on a finite
    rejection. The accepted state records the accepted trial's measurements.
    """

    rejection_commit_policy: ClassVar[tuple[str, ...]] = ()
    learning_rate: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    maximum_backtracks: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(self, policy: CalabiYauSolvePolicy, /):
        self.learning_rate = policy.learning_rate
        self.contraction = policy.contraction
        self.maximum_backtracks = policy.maximum_backtracks
        self.rule_id = canonical_fingerprint(
            {
                "kind": "calabi-yau-guarded-backtracking",
                "learning_rate": self.learning_rate,
                "contraction": self.contraction,
                "maximum_backtracks": self.maximum_backtracks,
            }
        )

    @property
    def reads_payload(self) -> bool:
        return True

    def init(self, parameters: Any, /) -> _CalabiYauStepState:
        del parameters
        zero = jnp.zeros((), jnp.float64)
        return _CalabiYauStepState(zero, zero, zero)

    def propose(
        self,
        parameters: Any,
        gradients: Any,
        value: Array,
        rule_state: _CalabiYauStepState,
        context: KernelUpdateContext,
        /,
    ) -> tuple[Any, _CalabiYauStepState, _CalabiYauStepState, Array]:
        payload = context.payload
        floor = payload.precision.decision(payload.positivity_floor)

        def descend(step: Array) -> Any:
            return jax.tree.map(
                lambda parameter, gradient: (
                    parameter - (step * gradient).astype(parameter.dtype)
                ),
                parameters,
                gradients,
            )

        def search(carry):
            index, _, accepted, _ = carry
            return (index <= self.maximum_backtracks) & ~accepted

        def trial(carry):
            index, step, _, _ = carry
            candidate = descend(step)
            trial_value = context.objective_value(candidate)
            _, diagnostics = _calabi_yau_terms(
                combine_parameters(candidate, context.model_state, context.fixed),
                payload,
            )
            accepted = (
                (trial_value <= value) & diagnostics.valid & (diagnostics.margin > floor)
            )
            return (
                index + 1,
                jnp.where(accepted, step, step * self.contraction),
                accepted,
                _step_state(trial_value, diagnostics),
            )

        _, step, accepted, measured = jax.lax.while_loop(
            search,
            trial,
            (
                jnp.zeros((), jnp.int32),
                jnp.asarray(self.learning_rate, jnp.float64),
                jnp.asarray(False),
                rule_state,
            ),
        )
        return descend(step), measured, rule_state, accepted


def solve_calabi_yau_metric(
    problem: CalabiYauMetricProblem,
    /,
    *,
    policy: CalabiYauSolvePolicy | None = None,
) -> CalabiYauMetricResult:
    """Fit the Kahler potential by guarded gradient descent on the training kernel.

    Each iteration is one kernel attempt (SURROGATE potential trained on its
    physical residual). An accepted attempt records the accepted trial's objective,
    residual, and margin; a rejected attempt (no admissible backtrack, or a
    nonfinite evaluation that the kernel rolls back) keeps the parameters and
    records the current ones. Rejections never stop the run: the budget equals the
    iteration count. The run stops early once the gradient norm at the attempt's
    parameters is within `gradient_tolerance`.
    """
    if not isinstance(problem, CalabiYauMetricProblem):
        raise TypeError("problem must be a CalabiYauMetricProblem.")
    policy_ = CalabiYauSolvePolicy() if policy is None else policy
    if not isinstance(policy_, CalabiYauSolvePolicy):
        raise TypeError("policy must be a CalabiYauSolvePolicy.")
    kernel = prepare_training_kernel(
        problem.potential_model,
        (
            KernelObjective(
                objective_id="calabi-yau-metric",
                kind=ObjectiveKind.PHYSICAL_RESIDUAL,
                route=DerivativeRoute.DIRECT,
                fn=_calabi_yau_objective,
            ),
        ),
        TrainingKernelSpec(
            _CalabiYauBacktrackingRule(policy_),
            context="solve_calabi_yau_metric",
            rejection_budget=policy_.iterations,
        ),
        root_authority=ComponentAuthority.SURROGATE,
    )
    # The Calabi-Yau objective draws no training randomness; the root key only
    # completes the kernel state.
    state = kernel.init(problem.potential_model, jax.random.key(0))
    payload = _CalabiYauPayload(problem)
    tolerance = problem.precision.decision(policy_.gradient_tolerance)
    objectives = []
    residuals = []
    margins = []
    accepted = []
    converged = False
    for _ in range(policy_.iterations):
        state, evidence = run_training_attempt(kernel, state, payload)
        (diagnostics,) = evidence.diagnostics
        did_accept = evidence.outcome == TrainingAttemptOutcome.ACCEPTED
        measured = state.rule_state
        objectives.append(jnp.where(did_accept, measured.value, evidence.value))
        residuals.append(jnp.where(did_accept, measured.residual, diagnostics.residual))
        margins.append(jnp.where(did_accept, measured.margin, diagnostics.margin))
        accepted.append(did_accept)
        if bool(
            jax.device_get(
                problem.precision.decision(evidence.gradient_norm) <= tolerance
            )
        ):
            converged = True
            break
    return CalabiYauMetricResult(
        kernel.tree(state),
        problem.normalization,
        problem.precision.output(
            jnp.stack(objectives) if objectives else jnp.zeros((0,))
        ),
        problem.precision.output(jnp.stack(residuals) if residuals else jnp.zeros((0,))),
        problem.precision.output(jnp.stack(margins) if margins else jnp.zeros((0,))),
        jnp.asarray(accepted, dtype=jnp.bool_),
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
