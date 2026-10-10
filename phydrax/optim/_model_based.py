#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._tree_math import validate_real_inexact_tree
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ..typing import checked
from ._interpolation_model import (
    coordinate_interpolation_points,
    fit_quadratic_scalar_model,
    QuadraticScalarModel,
)
from ._iterative import (
    AbstractMinimizationMethod,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._nonlinear_constraints import (
    _canonical_constraint_values,
    _constraint_layout,
    _constraint_violation,
    _ConstraintLayout,
)


def _coordinate_norm(value: Array, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.linalg.norm(precision.accumulation(value)))


ModelBasedKind: TypeAlias = Literal["bobyqa", "cobyqa"]


@dataclass(frozen=True, slots=True)
class _Evaluation:
    coordinates: Array
    objective: Array
    feasibility: Array
    merit: Array
    auxiliary: object
    finite: bool


@dataclass(slots=True)
class _ValueEvaluator:
    """Own all concrete user calls and their hard objective allowance."""

    problem: MinimizationProblem
    unflatten: Callable[[Array], PyTree[Array]]
    template: Array
    args: object
    precision: NonlinearPrecisionPolicy
    penalty: float
    maximum_evaluations: int | None
    layout: _ConstraintLayout | None = None
    value_shapes: tuple[PyTree[jax.ShapeDtypeStruct], ...] | None = None
    objective_evaluations: int = 0
    constraint_evaluations: int = 0
    nonfinite_evaluations: int = 0
    status: OptimizationStatus = OptimizationStatus.ITERATING

    def allows(self, count: int = 1) -> bool:
        return self.maximum_evaluations is None or (
            self.objective_evaluations + count <= self.maximum_evaluations
        )

    def evaluate(self, coordinates: Array) -> _Evaluation | None:
        if self.status != OptimizationStatus.ITERATING:
            return None
        if not self.allows():
            self.status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
            return None
        parameters = self.unflatten(jnp.asarray(coordinates, dtype=self.template.dtype))
        if self.problem.bounds is not None:
            parameters = self.problem.bounds.project(parameters)
        coordinates, _ = ravel_pytree(parameters)
        self.objective_evaluations += 1
        objective, auxiliary = self.problem.value(parameters, self.args)
        values: list[Array] = []
        shapes: list[PyTree[jax.ShapeDtypeStruct]] = []
        for constraint in self.problem.constraints:
            self.constraint_evaluations += 1
            value = constraint.value(parameters, self.args)
            shapes.append(
                jax.tree.map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), value
                )
            )
            flat, _ = ravel_pytree(value)
            values.append(flat)
        observed_shapes = tuple(shapes)
        if self.value_shapes is None:
            self.value_shapes = observed_shapes
            self.layout = _constraint_layout(
                self.problem, parameters, self.args, value_shapes=observed_shapes
            )
        elif observed_shapes != self.value_shapes:
            raise ValueError(
                "Constraint value structure, shape, and dtype must stay fixed."
            )
        if self.problem.bounds is not None:
            values.append(coordinates)
        raw = (
            jnp.concatenate(values)
            if values
            else jnp.empty((0,), dtype=coordinates.dtype)
        )
        if self.layout is None:
            raise RuntimeError("Constraint layout was not prepared.")
        equality, inequality = _canonical_constraint_values(self.layout, raw)
        feasibility = _constraint_violation(equality, inequality)
        merit = self.precision.decision(
            self.precision.accumulation(objective)
            + self.penalty * self.precision.accumulation(feasibility) ** 2
        )
        finite = bool(
            jnp.all(jnp.isfinite(coordinates))
            & jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(raw))
            & jnp.isfinite(feasibility)
            & jnp.isfinite(merit)
        )
        if not finite:
            self.nonfinite_evaluations += 1
            self.status = OptimizationStatus.NONFINITE_EVALUATION
        return _Evaluation(coordinates, objective, feasibility, merit, auxiliary, finite)


@dataclass(slots=True)
class _ModelState:
    current: _Evaluation
    radius: float
    samples: list[_Evaluation] = field(default_factory=list)
    model: QuadraticScalarModel | None = None
    best_feasible: _Evaluation | None = None
    iterations: int = 0
    accepted: int = 0
    rejected: int = 0
    linear_solves: int = 0
    globalization_evaluations: int = 0
    step_norm: float = 0.0
    ratio: Array = field(default_factory=lambda: jnp.asarray(jnp.nan))
    initial_optimality: Array | None = None

    def remember_feasible(self, value: _Evaluation, tolerance: float) -> None:
        if (
            value.finite
            and float(value.feasibility) <= tolerance
            and (
                self.best_feasible is None
                or float(value.objective) < float(self.best_feasible.objective)
            )
        ):
            self.best_feasible = value

    def fit(self, precision: NonlinearPrecisionPolicy) -> None:
        self.model = fit_quadratic_scalar_model(
            jnp.stack([sample.coordinates for sample in self.samples]),
            jnp.stack([sample.merit for sample in self.samples]),
            self.current.coordinates,
            self.radius,
            precision=precision,
        )


def _initialize_model(
    evaluator: _ValueEvaluator,
    radius: float,
    termination: OptimizationTermination,
) -> _ModelState:
    initial = evaluator.evaluate(evaluator.template)
    if initial is None:
        raise RuntimeError(
            "A positive evaluation allowance must admit the initial point."
        )
    state = _ModelState(initial, radius)
    state.remember_feasible(initial, termination.absolute_optimality)
    if not initial.finite:
        return state
    state.samples.append(initial)
    points = coordinate_interpolation_points(initial.coordinates, radius)
    for point in points[1:]:
        value = evaluator.evaluate(point)
        if value is None or not value.finite:
            return state
        state.samples.append(value)
        state.remember_feasible(value, termination.absolute_optimality)
    if not evaluator.allows():
        evaluator.status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
        return state
    state.fit(evaluator.precision)
    return state


def _model_step(
    model: QuadraticScalarModel,
    center: Array,
    radius: float,
    gradient: Array,
    method: AbstractModelBasedTrustRegion,
) -> Array:
    hessian = model.hessian(center)
    regularized = hessian + 1e-8 * jnp.eye(center.size, dtype=hessian.dtype)
    linear_result = solve_linear(
        LinearSystem(DenseLinearOperator(regularized)),
        -gradient,
        policy=method.precision.bind_linear(method.linear),
    )
    newton = method.precision.direction(linear_result.value)
    newton = (
        jnp.minimum(
            1.0, radius / jnp.maximum(_coordinate_norm(newton, method.precision), 1e-30)
        )
        * newton
    )
    cauchy = (
        -radius
        * gradient
        / jnp.maximum(_coordinate_norm(gradient, method.precision), 1e-30)
    )
    descent = method.precision.decision(
        jnp.real(
            jnp.sum(
                jnp.conj(method.precision.accumulation(gradient))
                * method.precision.accumulation(newton)
            )
        )
    )
    return jnp.where(
        jnp.all(jnp.isfinite(newton))
        & (descent < 0.0)
        & (model.condition_estimate < 1e12),
        newton,
        cauchy,
    )


def _poll_directions(center: Array, radius: float) -> Iterator[Array]:
    identity = jnp.eye(center.size, dtype=center.dtype)
    for index in range(center.size):
        for sign in (-1.0, 1.0):
            yield sign * radius * identity[index]
    for left in range(center.size):
        for right in range(left + 1, center.size):
            for sign in (-1.0, 1.0):
                yield sign * radius * (identity[left] - identity[right]) / jnp.sqrt(2.0)


def _trial_iteration(
    state: _ModelState,
    evaluator: _ValueEvaluator,
    method: AbstractModelBasedTrustRegion,
    termination: OptimizationTermination,
    gradient: Array,
) -> OptimizationStatus:
    model = state.model
    if model is None:
        raise RuntimeError("A trial requires a complete interpolation model.")
    center = state.current.coordinates
    step = _model_step(model, center, state.radius, gradient, method)
    state.linear_solves += 1
    before = evaluator.objective_evaluations
    candidate = evaluator.evaluate(center + step)
    if candidate is None or not candidate.finite:
        state.globalization_evaluations += evaluator.objective_evaluations - before
        return evaluator.status
    predicted = state.current.merit - model.value(candidate.coordinates)
    actual = state.current.merit - candidate.merit
    ratio = actual / jnp.maximum(predicted, 1e-30)
    accept = bool((predicted > 0.0) & (ratio >= 1e-4))
    if not accept:
        best_poll: _Evaluation | None = None
        for direction in _poll_directions(center, state.radius):
            value = evaluator.evaluate(center + direction)
            if value is None or not value.finite:
                break
            if best_poll is None or float(value.merit) < float(best_poll.merit):
                best_poll = value
        if best_poll is not None and float(best_poll.merit) < float(state.current.merit):
            candidate = best_poll
            ratio = jnp.asarray(1.0, dtype=candidate.merit.dtype)
            accept = True
    state.globalization_evaluations += evaluator.objective_evaluations - before
    step = candidate.coordinates - center
    if accept:
        state.current = candidate
        state.remember_feasible(candidate, termination.absolute_optimality)
        state.accepted += 1
    else:
        state.rejected += 1
    state.iterations += 1
    state.step_norm = float(_coordinate_norm(step, method.precision))
    state.ratio = ratio
    if evaluator.status != OptimizationStatus.ITERATING:
        return evaluator.status
    if not evaluator.allows():
        evaluator.status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
        return evaluator.status
    points = jnp.stack([sample.coordinates for sample in state.samples])
    replace = int(
        jnp.argmax(
            jnp.linalg.norm(
                method.precision.accumulation(
                    points - state.current.coordinates[None, :]
                ),
                axis=1,
            )
        )
    )
    state.samples[replace] = candidate
    if float(ratio) < 0.25:
        state.radius = max(method.minimum_radius, 0.25 * state.radius)
    elif float(ratio) > 0.75 and state.step_norm >= 0.9 * state.radius:
        state.radius = min(method.maximum_radius, 2.0 * state.radius)
    state.fit(method.precision)
    if accept and state.step_norm <= float(
        termination.step_threshold(
            _coordinate_norm(state.current.coordinates, method.precision)
        )
    ):
        return OptimizationStatus.STAGNATION
    if not accept and state.radius <= method.minimum_radius:
        return OptimizationStatus.TRUST_REGION_FAILED
    return OptimizationStatus.ITERATING


def _terminal_gradient(
    state: _ModelState,
    evaluator: _ValueEvaluator,
) -> Array | None:
    center = state.current.coordinates
    if evaluator.status != OptimizationStatus.ITERATING:
        return None
    if not evaluator.allows(2 * center.size):
        evaluator.status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
        return None
    step = max(state.radius, 1e-6)
    columns: list[Array] = []
    for index in range(center.size):
        direction = jnp.zeros_like(center).at[index].set(step)
        plus = evaluator.evaluate(center + direction)
        if plus is None or not plus.finite:
            return None
        minus = evaluator.evaluate(center - direction)
        if minus is None or not minus.finite:
            return None
        columns.append((plus.merit - minus.merit) / (2.0 * step))
    gradient = jnp.stack(columns)
    if not bool(jnp.all(jnp.isfinite(gradient))):
        evaluator.status = OptimizationStatus.NONFINITE_EVALUATION
        return None
    return gradient


def _model_result(
    state: _ModelState,
    evaluator: _ValueEvaluator,
    method: AbstractModelBasedTrustRegion,
    status: OptimizationStatus,
    gradient: Array | None,
) -> MinimizationResult:
    unavailable = jnp.asarray(jnp.nan, dtype=state.current.objective.dtype)
    final_optimality = (
        unavailable
        if gradient is None
        else method.precision.decision(
            jnp.linalg.norm(method.precision.accumulation(gradient), ord=jnp.inf)
        )
    )
    parameters = evaluator.unflatten(state.current.coordinates)
    output = jax.tree.map(method.precision.output, parameters)
    diagnostics = OptimizationDiagnostics(
        iterations=state.iterations,
        accepted_steps=state.accepted,
        rejected_steps=state.rejected,
        objective_evaluations=evaluator.objective_evaluations,
        gradient_evaluations=(
            evaluator.objective_evaluations
            if evaluator.problem.explicit_value_and_gradient is not None
            else 0
        ),
        constraint_evaluations=evaluator.constraint_evaluations,
        linear_solves=state.linear_solves,
        globalization_evaluations=state.globalization_evaluations,
        initial_optimality_norm=(
            unavailable if state.initial_optimality is None else state.initial_optimality
        ),
        final_optimality_norm=final_optimality,
        final_step_norm=state.step_norm,
        accepted_step_size=1.0 if state.accepted else 0.0,
        damping=state.radius,
        reduction_ratio=state.ratio,
        primal_feasibility=state.current.feasibility,
    )
    model = state.model
    model_notes = (
        "interpolation-model=unavailable"
        if model is None
        else f"poisedness-condition={float(model.condition_estimate):.6g};linear-plan={model.linear_plan_id}"
    )
    provenance = OptimizationProvenance(
        problem_id=evaluator.problem.problem_id,
        method=method.method_id,
        backend="phydrax-native",
        globalization="quadratic-interpolation-trust-region",
        matrix_free=False,
        implicit_differentiation=False,
        precision_policy_id=method.precision.policy_id,
        notes=f"{model_notes};value-only-host;terminal-optimality={'unavailable' if gradient is None else 'finite-difference'}",
    )
    # Observed scalar merit is the residual evidence when no independent gradient
    # is available; never invent a gradient to fill the precision envelope.
    residual = (
        method.precision.residual(state.current.merit)
        if gradient is None
        else evaluator.unflatten(method.precision.residual(gradient))
    )
    evidence = method.precision.evidence_for(
        parameters,
        residual,
        children={}
        if model is None
        else {"interpolation-model": model.precision_evidence},
        output_value=output,
    )
    return MinimizationResult(
        output,
        method.precision.output(state.current.objective),
        state.current.auxiliary,
        status,
        diagnostics,
        provenance,
        precision_evidence=evidence,
        method_evidence={
            "value_only": True,
            "finite": state.current.finite,
            "nonfinite_evaluations": evaluator.nonfinite_evaluations,
            "maximum_evaluations": evaluator.maximum_evaluations,
            "terminal_optimality_available": gradient is not None,
        },
    )


class AbstractModelBasedTrustRegion(AbstractMinimizationMethod):
    """Shared scalar quadratic interpolation trust-region optimizer."""

    initial_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    penalty: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        initial_radius: float = 0.25,
        minimum_radius: float = 1e-8,
        maximum_radius: float = 1e3,
        penalty: float = 100.0,
        maximum_dimension: int = 64,
        linear: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ) -> None:
        values = tuple(
            float(value)
            for value in (initial_radius, minimum_radius, maximum_radius, penalty)
        )
        dimension = int(maximum_dimension)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Model-based controls must be finite and positive.")
        if not values[1] <= values[0] <= values[2] or dimension < 1:
            raise ValueError("Model-based radius ordering or dimension is invalid.")
        linear_ = LinearSolvePolicy(DenseLU()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.initial_radius, self.minimum_radius, self.maximum_radius, self.penalty = (
            values
        )
        self.maximum_dimension = dimension
        self.linear = linear_
        self.precision = precision_

    @property
    @abc.abstractmethod
    def kind(self) -> ModelBasedKind:
        raise NotImplementedError

    @property
    def method_id(self) -> str:
        return self.kind

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    @checked
    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        self.precision.validate_tolerance(termination.absolute_optimality)
        if self.kind == "bobyqa" and problem.bounds is None:
            raise ValueError("BOBYQA requires parameter bounds.")
        if self.kind == "bobyqa" and problem.constraints:
            raise ValueError("BOBYQA supports bounds only; use COBYQA for constraints.")
        parameters = self.precision.state(
            validate_real_inexact_tree(initial_parameters, name="parameters")
        )
        if problem.bounds is not None:
            parameters = problem.bounds.project(parameters)
        center, unflatten = ravel_pytree(parameters)
        if center.size < 1 or center.size > self.maximum_dimension:
            raise ValueError(
                "Model-based dimension must be positive and within maximum_dimension."
            )
        evaluator = _ValueEvaluator(
            problem,
            unflatten,
            center,
            args,
            self.precision,
            self.penalty,
            termination.maximum_evaluations,
        )
        state = _initialize_model(evaluator, self.initial_radius, termination)
        status = evaluator.status
        while (
            status == OptimizationStatus.ITERATING
            and state.iterations < termination.maximum_steps
        ):
            if not evaluator.allows():
                evaluator.status = OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
                status = evaluator.status
                break
            model = state.model
            if model is None:
                raise RuntimeError(
                    "Initialization did not produce an interpolation model."
                )
            gradient = model.gradient(state.current.coordinates)
            optimality = self.precision.decision(
                jnp.linalg.norm(self.precision.accumulation(gradient), ord=jnp.inf)
            )
            if not bool(jnp.all(jnp.isfinite(gradient)) & jnp.isfinite(optimality)):
                evaluator.status = OptimizationStatus.NONFINITE_EVALUATION
                status = evaluator.status
                break
            if state.initial_optimality is None:
                state.initial_optimality = optimality
            if (
                float(optimality)
                <= float(termination.optimality_threshold(state.initial_optimality))
                and float(state.current.feasibility) <= termination.absolute_optimality
            ):
                status = OptimizationStatus.SUCCESS
                break
            status = _trial_iteration(state, evaluator, self, termination, gradient)
        if status == OptimizationStatus.ITERATING:
            status = OptimizationStatus.MAXIMUM_STEPS_REACHED
        if evaluator.status != OptimizationStatus.ITERATING:
            status = evaluator.status
        gradient = _terminal_gradient(state, evaluator)
        if evaluator.status != OptimizationStatus.ITERATING:
            status = evaluator.status
        elif gradient is not None:
            optimality = self.precision.decision(
                jnp.linalg.norm(self.precision.accumulation(gradient), ord=jnp.inf)
            )
            initial = (
                optimality
                if state.initial_optimality is None
                else state.initial_optimality
            )
            if (
                float(optimality) <= float(termination.optimality_threshold(initial))
                and float(state.current.feasibility) <= termination.absolute_optimality
            ):
                status = OptimizationStatus.SUCCESS
            elif status == OptimizationStatus.SUCCESS:
                status = OptimizationStatus.CERTIFICATION_FAILED
        if (
            status
            in (
                OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED,
                OptimizationStatus.NONFINITE_EVALUATION,
            )
            and state.best_feasible is not None
        ):
            state.current = state.best_feasible
            gradient = None
        return _model_result(state, evaluator, self, status, gradient)


class BOBYQA(AbstractModelBasedTrustRegion):
    if TYPE_CHECKING:
        __init__ = AbstractModelBasedTrustRegion.__init__

    @property
    def kind(self) -> ModelBasedKind:
        return "bobyqa"


class COBYQA(AbstractModelBasedTrustRegion):
    if TYPE_CHECKING:
        __init__ = AbstractModelBasedTrustRegion.__init__

    @property
    def kind(self) -> ModelBasedKind:
        return "cobyqa"


__all__ = ["BOBYQA", "COBYQA", "ModelBasedKind"]
