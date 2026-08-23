#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp

from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._tree_math import validate_real_inexact_tree
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    PyTreeSpace,
    solve as solve_linear,
)
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


def _coordinate_norm(value, precision: NonlinearPrecisionPolicy, /):
    return precision.decision(jnp.linalg.norm(precision.accumulation(value)))


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
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        initial_radius: float = 0.25,
        minimum_radius: float = 1e-8,
        maximum_radius: float = 1e3,
        maximum_dimension: int = 64,
        regularization: float = 1e-10,
        linear: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
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
        linear_ = LinearSolvePolicy(DenseLU()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.initial_radius, self.minimum_radius, self.maximum_radius = values[:3]
        self.regularization = values[3]
        self.maximum_dimension = dimension
        self.linear = linear_
        self.precision = precision_

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
        self.precision.validate_tolerance(termination.absolute_optimality)
        parameters = self.precision.state(
            validate_real_inexact_tree(
                initial_parameters,
                name="initial_parameters",
            )
        )
        if problem.bounds is not None:
            parameters = problem.bounds.project(parameters)
        space = PyTreeSpace(parameters)
        if space.size > self.maximum_dimension:
            raise ValueError("POUNDERS dimension exceeds maximum_dimension.")
        center = space.flatten(parameters)
        coordinate_dtype = center.dtype

        def evaluate(coordinates):
            coordinates = jnp.asarray(coordinates, dtype=coordinate_dtype)
            value = space.unflatten(coordinates)
            if problem.bounds is not None:
                value = problem.bounds.project(value)
                coordinates = space.flatten(value)
            residual, auxiliary = problem.value(value, args)
            residual = self.precision.residual(residual)
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
            interpolation,
            regularization=self.regularization,
            precision=self.precision,
        )
        center_residual = evaluate(center)[1]
        evaluations += 1
        self.precision.validate_trees(parameters, center_residual)
        objective = self.precision.decision(
            0.5
            * jnp.real(
                jnp.sum(
                    jnp.conj(self.precision.accumulation(center_residual))
                    * self.precision.accumulation(center_residual)
                )
            )
        )
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
            jacobian_ = self.precision.accumulation(jacobian)
            model_residual_ = self.precision.accumulation(model_residual)
            gradient = jnp.conj(jacobian_.T) @ model_residual_
            normal = jnp.conj(jacobian_.T) @ jacobian_
            optimality = self.precision.decision(jnp.linalg.norm(gradient, ord=jnp.inf))
            if initial_optimality is None:
                initial_optimality = optimality
            if float(optimality) <= float(
                termination.optimality_threshold(initial_optimality)
            ):
                status = int(OptimizationStatus.SUCCESS)
                break
            regularized = normal + 1e-10 * jnp.eye(space.size, dtype=normal.dtype)
            linear_result = solve_linear(
                LinearSystem(DenseLinearOperator(regularized)),
                -gradient,
                policy=self.precision.bind_linear(self.linear),
            )
            step = self.precision.direction(linear_result.value)
            step = (
                jnp.minimum(
                    1.0,
                    radius / jnp.maximum(_coordinate_norm(step, self.precision), 1e-30),
                )
                * step
            )
            candidate, candidate_residual, candidate_auxiliary = evaluate(center + step)
            candidate_residual_ = self.precision.accumulation(candidate_residual)
            candidate_objective = self.precision.decision(
                0.5
                * jnp.real(jnp.sum(jnp.conj(candidate_residual_) * candidate_residual_))
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
            distances = jnp.linalg.norm(
                self.precision.accumulation(points - center[None, :]),
                axis=1,
            )
            replace = int(jnp.argmax(distances))
            points = points.at[replace].set(candidate)
            residuals = residuals.at[replace].set(candidate_residual)
            if ratio < 0.25 or not finite:
                radius = max(self.minimum_radius, 0.25 * radius)
            elif (
                ratio > 0.75
                and _coordinate_norm(
                    step,
                    self.precision,
                )
                >= 0.9 * radius
            ):
                radius = min(self.maximum_radius, 2.0 * radius)
            interpolation = InterpolationSet(
                points,
                residuals,
                center,
                radius,
                evaluations=evaluations + 1,
            )
            model = fit_quadratic_residual_model(
                interpolation,
                regularization=self.regularization,
                precision=self.precision,
            )
            evaluations += 1
            iterations += 1
            step_norm = float(_coordinate_norm(step, self.precision))
            if accept and step_norm <= float(
                termination.step_threshold(_coordinate_norm(center, self.precision))
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
            precision=self.precision,
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
            _coordinate_norm(model_gradient, self.precision),
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
            precision_policy_id=self.precision.policy_id,
            notes=(
                f"poisedness-condition={float(model.condition_estimate):.6g};"
                f"internal-status={status};"
                f"linear-plan={model.linear_plan_id}"
            ),
        )
        output_parameters = jax.tree.map(self.precision.output, parameters)
        precision_evidence = self.precision.evidence_for(
            parameters,
            space.unflatten(center_residual),
            children={
                "certificate": certificate.precision_evidence,
                "interpolation-model": model.precision_evidence,
            },
            output_value=output_parameters,
        )
        return LeastSquaresResult(
            output_parameters,
            center_residual,
            self.precision.output(objective),
            last_auxiliary,
            status_evidence.public_status,
            diagnostics,
            provenance,
            optimality_certificate=certificate,
            status_evidence=status_evidence,
            method_evidence=evidence,
            precision_evidence=precision_evidence,
        )


__all__ = ["POUNDERS", "POUNDERSEvidence"]
