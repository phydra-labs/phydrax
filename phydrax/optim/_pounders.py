#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp

from .._tree_math import validate_real_inexact_tree
from ..linalg import PyTreeSpace
from ._certificates import (
    certify_least_squares_physical,
    reconcile_optimization_status,
)
from ._interpolation_model import (
    coordinate_interpolation_points,
    fit_quadratic_residual_model,
    InterpolationSet,
)
from ._iterative import (
    AbstractLeastSquaresMethod,
    IterativeStepMetrics,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._least_squares import LeastSquaresState


class POUNDERSEvidence(eqx.Module):
    interpolation_rank: jax.Array
    poisedness_condition: jax.Array
    final_radius: jax.Array
    model_gradient_norm: jax.Array
    independent_stationarity: jax.Array
    certificate_evaluations: jax.Array


class POUNDERS(AbstractLeastSquaresMethod):
    """Residual-wise derivative-free quadratic trust-region method."""

    initial_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        initial_radius: float = 0.25,
        minimum_radius: float = 1e-8,
        maximum_radius: float = 1e3,
        maximum_dimension: int = 64,
        regularization: float = 1e-10,
    ):
        values = tuple(
            float(value)
            for value in (
                initial_radius,
                minimum_radius,
                maximum_radius,
                regularization,
            )
        )
        dimension = int(maximum_dimension)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("POUNDERS controls must be finite and positive.")
        if not values[1] <= values[0] <= values[2] or dimension < 1:
            raise ValueError("POUNDERS radius ordering or maximum_dimension is invalid.")
        self.initial_radius, self.minimum_radius, self.maximum_radius = values[:3]
        self.regularization = values[3]
        self.maximum_dimension = dimension

    @property
    def method_id(self):
        return "pounders"

    @property
    def capabilities(self):
        return OptimizationCapabilities(
            scalar_objective=False,
            residual_objective=True,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    def init(self, parameters, /):
        parameters_ = validate_real_inexact_tree(parameters, name="parameters")
        dtype = jax.tree.leaves(parameters_)[0].dtype
        nan = jnp.asarray(jnp.nan, dtype=dtype)
        return LeastSquaresState(
            initial_optimality_norm=nan,
            metrics=IterativeStepMetrics(objective=nan, damping=self.initial_radius),
        )

    def prepare_state(self, residual_function, parameters, /):
        if not callable(residual_function):
            raise TypeError("residual_function must be callable.")
        return self.init(parameters)

    def step(self, residual_function, parameters, state, /, *, termination):
        result = self.solve(
            NonlinearLeastSquaresProblem(lambda value, args: residual_function(value)),
            parameters,
            termination=OptimizationTermination(
                absolute_optimality=termination.absolute_optimality,
                relative_optimality=termination.relative_optimality,
                absolute_step=termination.absolute_step,
                relative_step=termination.relative_step,
                maximum_steps=1,
                maximum_evaluations=termination.maximum_evaluations,
            ),
            args=None,
        )
        next_state = self.init(result.parameters)
        next_state = eqx.tree_at(
            lambda value: (value.iteration, value.metrics),
            next_state,
            (
                state.iteration + 1,
                IterativeStepMetrics(
                    objective=result.objective,
                    optimality_norm=result.diagnostics.final_optimality_norm,
                    step_norm=result.diagnostics.final_step_norm,
                    accepted_step_size=result.diagnostics.accepted_step_size,
                    accepted=result.diagnostics.accepted_steps > 0,
                    damping=result.diagnostics.damping,
                    reduction_ratio=result.diagnostics.reduction_ratio,
                    status=result.status,
                ),
            ),
        )
        return result.parameters, next_state, result.objective

    def step_metrics(self, state, /):
        return state.metrics

    def solve(self, problem, initial_parameters, /, *, termination, args):
        if not isinstance(problem, NonlinearLeastSquaresProblem):
            raise TypeError("problem must be NonlinearLeastSquaresProblem.")
        parameters = validate_real_inexact_tree(
            initial_parameters, name="initial_parameters"
        )
        if problem.bounds is not None:
            parameters = problem.bounds.project(parameters)
        space = PyTreeSpace(parameters)
        if space.size > self.maximum_dimension:
            raise ValueError("POUNDERS dimension exceeds maximum_dimension.")
        center = space.flatten(parameters)

        def evaluate(coordinates):
            value = space.unflatten(coordinates)
            if problem.bounds is not None:
                value = problem.bounds.project(value)
                coordinates = space.flatten(value)
            residual, auxiliary = problem.value(value, args)
            return coordinates, PyTreeSpace(residual).flatten(residual), auxiliary

        raw_points = coordinate_interpolation_points(center, self.initial_radius)
        evaluated = [evaluate(point) for point in raw_points]
        points = jnp.stack([value[0] for value in evaluated])
        residuals = jnp.stack([value[1] for value in evaluated])
        evaluations = len(evaluated)
        radius = self.initial_radius
        interpolation = InterpolationSet(
            points,
            residuals,
            center,
            radius,
            evaluations=evaluations,
        )
        model = fit_quadratic_residual_model(
            interpolation, regularization=self.regularization
        )
        center_residual = evaluate(center)[1]
        evaluations += 1
        objective = 0.5 * jnp.real(jnp.vdot(center_residual, center_residual))
        initial_objective = objective
        accepted = rejected = iterations = 0
        step_norm = 0.0
        ratio = jnp.asarray(jnp.nan, dtype=objective.dtype)
        status = int(OptimizationStatus.ITERATING)
        initial_optimality = None
        last_auxiliary = evaluated[0][2]
        while (
            status == int(OptimizationStatus.ITERATING)
            and iterations < termination.maximum_steps
        ):
            jacobian = model.jacobian(center)
            model_residual = model.residual(center)
            gradient = jnp.conj(jacobian.T) @ model_residual
            normal = jnp.conj(jacobian.T) @ jacobian
            optimality = jnp.linalg.norm(gradient, ord=jnp.inf)
            if initial_optimality is None:
                initial_optimality = optimality
            if float(optimality) <= float(
                termination.optimality_threshold(initial_optimality)
            ):
                status = int(OptimizationStatus.SUCCESS)
                break
            regularized = normal + 1e-10 * jnp.eye(space.size, dtype=normal.dtype)
            step = jnp.linalg.solve(regularized, -gradient)
            step = (
                jnp.minimum(1.0, radius / jnp.maximum(jnp.linalg.norm(step), 1e-30))
                * step
            )
            candidate, candidate_residual, candidate_auxiliary = evaluate(center + step)
            candidate_objective = 0.5 * jnp.real(
                jnp.vdot(candidate_residual, candidate_residual)
            )
            predicted = objective - model.objective(candidate)
            actual = objective - candidate_objective
            ratio = actual / jnp.maximum(predicted, 1e-30)
            finite = bool(jnp.all(jnp.isfinite(candidate_residual)))
            accept = finite and predicted > 0.0 and ratio >= 1e-4
            if accept:
                center = candidate
                center_residual = candidate_residual
                objective = candidate_objective
                last_auxiliary = candidate_auxiliary
                accepted += 1
            else:
                rejected += 1
            distances = jnp.linalg.norm(points - center[None, :], axis=1)
            replace = int(jnp.argmax(distances))
            points = points.at[replace].set(candidate)
            residuals = residuals.at[replace].set(candidate_residual)
            if ratio < 0.25 or not finite:
                radius = max(self.minimum_radius, 0.25 * radius)
            elif ratio > 0.75 and jnp.linalg.norm(step) >= 0.9 * radius:
                radius = min(self.maximum_radius, 2.0 * radius)
            interpolation = InterpolationSet(
                points,
                residuals,
                center,
                radius,
                evaluations=evaluations + 1,
            )
            model = fit_quadratic_residual_model(
                interpolation, regularization=self.regularization
            )
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
        parameters = space.unflatten(center)
        certificate = certify_least_squares_physical(
            problem,
            parameters,
            args,
            termination,
            certificate_step=min(max(radius, 1e-8), 1e-4),
            kind="derivative-free-stationarity",
        )
        status_evidence = reconcile_optimization_status(
            status,
            certificate,
            allow_certificate_promotion=True,
        )
        primal_feasibility = certificate.primal_feasibility
        model_gradient = model.jacobian(center).T @ model.residual(center)
        evidence = POUNDERSEvidence(
            model.interpolation_rank,
            model.condition_estimate,
            jnp.asarray(radius),
            jnp.linalg.norm(model_gradient, ord=jnp.inf),
            certificate.projected_stationarity,
            certificate.evaluation_work,
        )
        diagnostics = OptimizationDiagnostics(
            iterations=iterations,
            accepted_steps=accepted,
            rejected_steps=rejected,
            residual_evaluations=evaluations + certificate.evaluation_work,
            initial_optimality_norm=(
                certificate.optimality_norm
                if initial_optimality is None
                else initial_optimality
            ),
            final_optimality_norm=certificate.optimality_norm,
            final_step_norm=step_norm,
            accepted_step_size=1.0 if accepted else 0.0,
            damping=radius,
            reduction_ratio=ratio,
            primal_feasibility=primal_feasibility,
        )
        provenance = OptimizationProvenance(
            problem_id=problem.problem_id,
            method=self.method_id,
            backend="phydrax-native",
            globalization="residual-interpolation-trust-region",
            matrix_free=False,
            implicit_differentiation=False,
            notes=(
                f"poisedness-condition={float(model.condition_estimate):.6g};"
                f"internal-status={status}"
            ),
        )
        return LeastSquaresResult(
            parameters,
            center_residual,
            objective,
            last_auxiliary,
            status_evidence.public_status,
            diagnostics,
            provenance,
            optimality_certificate=certificate,
            status_evidence=status_evidence,
            method_evidence=evidence,
        )


__all__ = ["POUNDERS", "POUNDERSEvidence"]
