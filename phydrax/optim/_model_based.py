#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

from .._tree_math import validate_real_inexact_tree
from ._constrained_model import prepare_constrained_model
from ._interpolation_model import (
    coordinate_interpolation_points,
    fit_quadratic_scalar_model,
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


ModelBasedKind: TypeAlias = Literal["bobyqa", "cobyqa"]


class AbstractModelBasedTrustRegion(AbstractMinimizationMethod):
    """Shared scalar quadratic interpolation trust-region optimizer."""

    initial_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    penalty: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_radius: float = 0.25,
        minimum_radius: float = 1e-8,
        maximum_radius: float = 1e3,
        penalty: float = 100.0,
        maximum_dimension: int = 64,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_radius,
                minimum_radius,
                maximum_radius,
                penalty,
            )
        )
        dimension = int(maximum_dimension)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Model-based controls must be finite and positive.")
        if not values[1] <= values[0] <= values[2] or dimension < 1:
            raise ValueError("Model-based radius ordering or dimension is invalid.")
        (
            self.initial_radius,
            self.minimum_radius,
            self.maximum_radius,
            self.penalty,
        ) = values
        self.maximum_dimension = dimension

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

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        if not isinstance(problem, MinimizationProblem):
            raise TypeError("problem must be MinimizationProblem.")
        if self.kind == "bobyqa" and problem.bounds is None:
            raise ValueError("BOBYQA requires parameter bounds.")
        if self.kind == "bobyqa" and problem.constraints:
            raise ValueError("BOBYQA supports bounds only; use COBYQA for constraints.")
        parameters = validate_real_inexact_tree(initial_parameters, name="parameters")
        if problem.bounds is not None:
            parameters = problem.bounds.project(parameters)
        center, unflatten = ravel_pytree(parameters)
        if center.size > self.maximum_dimension:
            raise ValueError("Model-based dimension exceeds maximum_dimension.")
        constrained = (
            prepare_constrained_model(problem, parameters, args=args)
            if problem.constraints or problem.bounds is not None
            else None
        )

        def evaluate(coordinates):
            value = unflatten(coordinates)
            if problem.bounds is not None:
                value = problem.bounds.project(value)
                coordinates, _ = ravel_pytree(value)
            objective, auxiliary = problem.value(value, args)
            feasibility = (
                jnp.asarray(0.0, dtype=objective.dtype)
                if constrained is None
                else constrained.evaluate(value, args).primal_feasibility
            )
            merit = objective + self.penalty * feasibility * feasibility
            return coordinates, objective, feasibility, merit, auxiliary

        radius = self.initial_radius
        raw_points = coordinate_interpolation_points(center, radius)
        evaluated = [evaluate(point) for point in raw_points]
        points = jnp.stack([value[0] for value in evaluated])
        values = jnp.stack([value[3] for value in evaluated])
        center_eval = evaluate(center)
        objective = center_eval[1]
        feasibility = center_eval[2]
        merit = center_eval[3]
        auxiliary = center_eval[4]
        evaluations = len(evaluated) + 1
        model = fit_quadratic_scalar_model(points, values, center, radius)
        accepted = rejected = iterations = 0
        step_norm = 0.0
        ratio = jnp.asarray(jnp.nan, dtype=objective.dtype)
        status = int(OptimizationStatus.ITERATING)
        initial_optimality = None
        while (
            status == int(OptimizationStatus.ITERATING)
            and iterations < termination.maximum_steps
        ):
            gradient = model.gradient(center)
            hessian = model.hessian(center)
            optimality = jnp.linalg.norm(gradient, ord=jnp.inf)
            if initial_optimality is None:
                initial_optimality = optimality
            if float(optimality) <= float(
                termination.optimality_threshold(initial_optimality)
            ) and float(feasibility) <= float(termination.absolute_optimality):
                status = int(OptimizationStatus.SUCCESS)
                break
            regularized = hessian + 1e-8 * jnp.eye(center.size, dtype=hessian.dtype)
            newton_step = jnp.linalg.solve(regularized, -gradient)
            newton_step = (
                jnp.minimum(
                    1.0,
                    radius / jnp.maximum(jnp.linalg.norm(newton_step), 1e-30),
                )
                * newton_step
            )
            cauchy_step = (
                -radius
                * gradient
                / jnp.maximum(
                    jnp.linalg.norm(gradient),
                    1e-30,
                )
            )
            step = jnp.where(
                jnp.all(jnp.isfinite(newton_step))
                & (jnp.real(jnp.vdot(gradient, newton_step)) < 0.0)
                & (model.condition_estimate < 1e12),
                newton_step,
                cauchy_step,
            )
            candidate = evaluate(center + step)
            predicted = merit - model.value(candidate[0])
            actual = merit - candidate[3]
            ratio = actual / jnp.maximum(predicted, 1e-30)
            finite = bool(jnp.isfinite(candidate[3]))
            accept = finite and predicted > 0.0 and ratio >= 1e-4
            if not accept:
                identity = jnp.eye(center.size, dtype=center.dtype)
                poll_directions = [
                    sign * radius * identity[index]
                    for index in range(center.size)
                    for sign in (-1.0, 1.0)
                ]
                poll_directions.extend(
                    sign * radius * (identity[left] - identity[right]) / jnp.sqrt(2.0)
                    for left in range(center.size)
                    for right in range(left + 1, center.size)
                    for sign in (-1.0, 1.0)
                )
                poll = [evaluate(center + direction) for direction in poll_directions]
                evaluations += len(poll)
                best_poll = min(poll, key=lambda value: float(value[3]))
                if float(best_poll[3]) < float(merit):
                    candidate = best_poll
                    step = candidate[0] - center
                    actual = merit - candidate[3]
                    predicted = jnp.maximum(actual, 1e-30)
                    ratio = jnp.asarray(1.0, dtype=merit.dtype)
                    finite = bool(jnp.isfinite(candidate[3]))
                    accept = finite
            if accept:
                center = candidate[0]
                objective = candidate[1]
                feasibility = candidate[2]
                merit = candidate[3]
                auxiliary = candidate[4]
                accepted += 1
            else:
                rejected += 1
            replace = int(jnp.argmax(jnp.linalg.norm(points - center[None, :], axis=1)))
            points = points.at[replace].set(candidate[0])
            values = values.at[replace].set(candidate[3])
            if ratio < 0.25 or not finite:
                radius = max(self.minimum_radius, 0.25 * radius)
            elif ratio > 0.75 and jnp.linalg.norm(step) >= 0.9 * radius:
                radius = min(self.maximum_radius, 2.0 * radius)
            model = fit_quadratic_scalar_model(points, values, center, radius)
            evaluations += 1
            iterations += 1
            step_norm = float(jnp.linalg.norm(step))
            if accept and step_norm <= float(
                termination.step_threshold(jnp.linalg.norm(center))
            ):
                status = int(OptimizationStatus.STAGNATION)
            elif not accept and radius <= self.minimum_radius:
                status = int(OptimizationStatus.TRUST_REGION_FAILED)
            elif (
                termination.maximum_evaluations is not None
                and evaluations >= termination.maximum_evaluations
            ):
                status = int(OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED)
        if status == int(OptimizationStatus.ITERATING):
            status = int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
        finite_difference_step = max(radius, 1e-6)
        gradient_columns = []
        for index in range(center.size):
            direction = jnp.zeros_like(center).at[index].set(finite_difference_step)
            plus = evaluate(center + direction)[3]
            minus = evaluate(center - direction)[3]
            gradient_columns.append((plus - minus) / (2.0 * finite_difference_step))
        independent_gradient = jnp.stack(gradient_columns)
        final_optimality = jnp.linalg.norm(independent_gradient, ord=jnp.inf)
        if (
            float(final_optimality)
            <= float(
                termination.optimality_threshold(
                    final_optimality if initial_optimality is None else initial_optimality
                )
            )
            and float(feasibility) <= termination.absolute_optimality
        ):
            status = int(OptimizationStatus.SUCCESS)
        evaluations += 2 * center.size
        parameters = unflatten(center)
        diagnostics = OptimizationDiagnostics(
            iterations=iterations,
            accepted_steps=accepted,
            rejected_steps=rejected,
            objective_evaluations=evaluations,
            initial_optimality_norm=(
                final_optimality if initial_optimality is None else initial_optimality
            ),
            final_optimality_norm=final_optimality,
            final_step_norm=step_norm,
            accepted_step_size=1.0 if accepted else 0.0,
            damping=radius,
            reduction_ratio=ratio,
            primal_feasibility=feasibility,
        )
        provenance = OptimizationProvenance(
            problem_id=problem.problem_id,
            method=self.method_id,
            backend="phydrax-native",
            globalization="quadratic-interpolation-trust-region",
            matrix_free=False,
            implicit_differentiation=False,
            notes=f"poisedness-condition={float(model.condition_estimate):.6g}",
        )
        return MinimizationResult(
            parameters,
            objective,
            auxiliary,
            status,
            diagnostics,
            provenance,
        )


class BOBYQA(AbstractModelBasedTrustRegion):
    @property
    def kind(self) -> ModelBasedKind:
        return "bobyqa"


class COBYQA(AbstractModelBasedTrustRegion):
    @property
    def kind(self) -> ModelBasedKind:
        return "cobyqa"


__all__ = ["BOBYQA", "COBYQA", "ModelBasedKind"]
