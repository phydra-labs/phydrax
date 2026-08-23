#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp

from .._tree_math import validate_real_inexact_tree
from ..linalg import PyTreeSpace
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
from ._least_squares import BoundedLevenbergMarquardt, LeastSquaresState


DoglegMode: TypeAlias = Literal["traditional", "subspace", "dogbox"]


def _boundary_rate(point, direction, radius):
    a = jnp.real(jnp.vdot(direction, direction))
    b = 2.0 * jnp.real(jnp.vdot(point, direction))
    c = jnp.real(jnp.vdot(point, point)) - radius * radius
    discriminant = jnp.maximum(b * b - 4.0 * a * c, 0.0)
    return (-b + jnp.sqrt(discriminant)) / jnp.maximum(2.0 * a, 1e-30)


def _dogleg_step(gradient, normal, radius, mode):
    dimension = gradient.size
    regularized = normal + 1e-12 * jnp.eye(dimension, dtype=normal.dtype)
    gauss_newton = jnp.linalg.solve(regularized, -gradient)
    gradient_image = normal @ gradient
    denominator = jnp.real(jnp.vdot(gradient, gradient_image))
    cauchy_scale = jnp.real(jnp.vdot(gradient, gradient)) / jnp.maximum(
        denominator, 1e-30
    )
    cauchy = -cauchy_scale * gradient
    if mode == "dogbox":
        return jnp.clip(gauss_newton, -radius, radius)
    gauss_norm = jnp.linalg.norm(gauss_newton)
    cauchy_norm = jnp.linalg.norm(cauchy)
    boundary_cauchy = radius * cauchy / jnp.maximum(cauchy_norm, 1e-30)
    segment = gauss_newton - cauchy
    rate = _boundary_rate(cauchy, segment, radius)
    dogleg = cauchy + rate * segment
    if mode == "subspace":
        basis = jnp.stack(
            [
                -gradient / jnp.maximum(jnp.linalg.norm(gradient), 1e-30),
                gauss_newton / jnp.maximum(gauss_norm, 1e-30),
            ],
            axis=1,
        )
        reduced_normal = jnp.conj(basis.T) @ normal @ basis
        reduced_gradient = jnp.conj(basis.T) @ gradient
        reduced = jnp.linalg.solve(
            reduced_normal + 1e-12 * jnp.eye(2, dtype=normal.dtype),
            -reduced_gradient,
        )
        reduced_norm = jnp.linalg.norm(reduced)
        subspace = basis @ (
            jnp.minimum(1.0, radius / jnp.maximum(reduced_norm, 1e-30)) * reduced
        )
        dogleg = subspace
    return jnp.where(
        gauss_norm <= radius,
        gauss_newton,
        jnp.where(cauchy_norm >= radius, boundary_cauchy, dogleg),
    )


class DoglegLeastSquares(AbstractLeastSquaresMethod):
    """Dense traditional/subspace dogleg or rectangular dogbox residual solve."""

    mode: DoglegMode = eqx.field(static=True)
    initial_radius: float = eqx.field(static=True)
    minimum_radius: float = eqx.field(static=True)
    maximum_radius: float = eqx.field(static=True)
    maximum_dimension: int = eqx.field(static=True)
    nonmonotone_window: int = eqx.field(static=True)

    def __init__(
        self,
        mode: DoglegMode = "traditional",
        /,
        *,
        initial_radius: float = 1.0,
        minimum_radius: float = 1e-12,
        maximum_radius: float = 1e6,
        maximum_dimension: int = 512,
        nonmonotone_window: int = 1,
    ):
        if mode not in ("traditional", "subspace", "dogbox"):
            raise ValueError("Unknown dogleg mode.")
        values = tuple(
            float(value) for value in (initial_radius, minimum_radius, maximum_radius)
        )
        dimension = int(maximum_dimension)
        window = int(nonmonotone_window)
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Dogleg radii must be finite and positive.")
        if not values[1] <= values[0] <= values[2]:
            raise ValueError("Dogleg radii must satisfy minimum <= initial <= maximum.")
        if dimension < 1 or window < 1:
            raise ValueError("Dogleg dimension and window must be positive.")
        self.mode = mode
        self.initial_radius, self.minimum_radius, self.maximum_radius = values
        self.maximum_dimension = dimension
        self.nonmonotone_window = window

    @property
    def method_id(self):
        return f"least-squares-{self.mode}"

    @property
    def capabilities(self):
        return OptimizationCapabilities(
            scalar_objective=False,
            residual_objective=True,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=True,
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
        problem = NonlinearLeastSquaresProblem(
            lambda value, args: residual_function(value)
        )
        result = self.solve(
            problem,
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
        bounds = problem.bounds
        if bounds is not None:
            parameters = bounds.project(parameters)
        space = PyTreeSpace(parameters)
        if space.size > self.maximum_dimension:
            raise ValueError("Dogleg dimension exceeds maximum_dimension.")

        def residual_coordinates(coordinates):
            value = space.unflatten(coordinates)
            residual, _ = problem.value(value, args)
            return PyTreeSpace(residual).flatten(residual)

        def optimality_norm(gradient_coordinates, point):
            if bounds is None:
                return jnp.linalg.norm(
                    gradient_coordinates,
                    ord=jnp.inf,
                )
            projected = bounds.projected_gradient(
                point,
                space.unflatten(gradient_coordinates),
            )
            return jnp.linalg.norm(
                space.flatten(projected),
                ord=jnp.inf,
            )

        coordinates = space.flatten(parameters)
        residual, auxiliary = problem.value(parameters, args)
        residual_space = PyTreeSpace(residual)
        residual_vector = residual_space.flatten(residual)
        objective = 0.5 * jnp.real(jnp.vdot(residual_vector, residual_vector))
        radius = self.initial_radius
        history = [float(objective)]
        iterations = evaluations = accepted = rejected = 0
        jvp = vjp = 0
        step_norm = 0.0
        ratio = jnp.asarray(jnp.nan, dtype=objective.dtype)
        status = int(OptimizationStatus.ITERATING)
        initial_optimality = None
        while (
            status == int(OptimizationStatus.ITERATING)
            and iterations < termination.maximum_steps
        ):
            jacobian = jax.jacfwd(residual_coordinates)(coordinates)
            gradient = jnp.conj(jacobian.T) @ residual_vector
            normal = jnp.conj(jacobian.T) @ jacobian
            optimality = optimality_norm(gradient, parameters)
            if initial_optimality is None:
                initial_optimality = optimality
            if float(optimality) <= float(
                termination.optimality_threshold(initial_optimality)
            ):
                status = int(OptimizationStatus.SUCCESS)
                break
            step = _dogleg_step(gradient, normal, radius, self.mode)
            candidate_coordinates = coordinates + step
            candidate = space.unflatten(candidate_coordinates)
            if bounds is not None:
                candidate = bounds.project(candidate)
                candidate_coordinates = space.flatten(candidate)
                step = candidate_coordinates - coordinates
            candidate_residual, candidate_auxiliary = problem.value(candidate, args)
            candidate_vector = residual_space.flatten(candidate_residual)
            candidate_objective = 0.5 * jnp.real(
                jnp.vdot(candidate_vector, candidate_vector)
            )
            predicted = -(
                jnp.real(jnp.vdot(gradient, step))
                + 0.5 * jnp.real(jnp.vdot(step, normal @ step))
            )
            reference = max(history[-self.nonmonotone_window :])
            actual = reference - float(candidate_objective)
            ratio = actual / max(float(predicted), 1e-30)
            finite = bool(jnp.all(jnp.isfinite(candidate_vector)))
            accept = finite and predicted > 0.0 and ratio >= 1e-4
            if accept:
                coordinates = candidate_coordinates
                parameters = candidate
                residual = candidate_residual
                auxiliary = candidate_auxiliary
                residual_vector = candidate_vector
                objective = candidate_objective
                history.append(float(candidate_objective))
                accepted += 1
            else:
                rejected += 1
            step_norm = float(jnp.linalg.norm(step))
            if ratio < 0.25 or not finite:
                radius = max(self.minimum_radius, 0.25 * radius)
            elif ratio > 0.75 and step_norm >= 0.9 * radius:
                radius = min(self.maximum_radius, 2.0 * radius)
            iterations += 1
            evaluations += 2
            jvp += space.size
            vjp += 1
            if accept and step_norm <= float(
                termination.step_threshold(jnp.linalg.norm(coordinates))
            ):
                status = int(OptimizationStatus.STAGNATION)
            elif not accept and radius <= self.minimum_radius:
                status = int(OptimizationStatus.TRUST_REGION_FAILED)
        if status == int(OptimizationStatus.ITERATING):
            status = int(OptimizationStatus.MAXIMUM_STEPS_REACHED)
        final_jacobian = jax.jacfwd(residual_coordinates)(coordinates)
        final_gradient = jnp.conj(final_jacobian.T) @ residual_vector
        final_optimality = optimality_norm(final_gradient, parameters)
        primal_feasibility = 0.0 if bounds is None else bounds.violation(parameters)
        if (
            float(final_optimality)
            <= float(
                termination.optimality_threshold(
                    final_optimality if initial_optimality is None else initial_optimality
                )
            )
            and float(primal_feasibility) <= termination.absolute_optimality
        ):
            status = int(OptimizationStatus.SUCCESS)
        diagnostics = OptimizationDiagnostics(
            initial_optimality_norm=(
                final_optimality if initial_optimality is None else initial_optimality
            ),
            final_optimality_norm=final_optimality,
            objective_evaluations=evaluations + 2,
            final_step_norm=step_norm,
            iterations=iterations,
            residual_evaluations=evaluations + 2,
            jvp_evaluations=jvp + space.size,
            vjp_evaluations=vjp + 1,
            accepted_steps=accepted,
            rejected_steps=rejected,
            accepted_step_size=1.0 if accepted else 0.0,
            damping=radius,
            reduction_ratio=ratio,
            primal_feasibility=primal_feasibility,
        )
        provenance = OptimizationProvenance(
            problem_id=problem.problem_id,
            method=self.method_id,
            backend="phydrax-native",
            globalization="nonmonotone-trust-region",
            matrix_free=False,
            implicit_differentiation=True,
        )
        return LeastSquaresResult(
            parameters,
            residual,
            objective,
            auxiliary,
            status,
            diagnostics,
            provenance,
        )


class TrustRegionReflective(AbstractLeastSquaresMethod):
    """Bound-reflective facade over the matrix-free active-set LM kernel."""

    method: BoundedLevenbergMarquardt

    def __init__(self, **kwargs):
        self.method = BoundedLevenbergMarquardt(**kwargs)

    @property
    def method_id(self):
        return "trust-region-reflective"

    @property
    def capabilities(self):
        return self.method.capabilities

    def init(self, parameters, /):
        return self.method.init(parameters)

    def prepare_state(self, residual_function, parameters, /):
        return self.method.prepare_state(residual_function, parameters)

    def step(self, residual_function, parameters, state, /, *, termination):
        return self.method.step(
            residual_function, parameters, state, termination=termination
        )

    def step_metrics(self, state, /):
        return self.method.step_metrics(state)

    def solve(self, problem, initial_parameters, /, *, termination, args):
        result = self.method.solve(
            problem,
            initial_parameters,
            termination=termination,
            args=args,
        )
        provenance = OptimizationProvenance(
            problem_id=result.provenance.problem_id,
            method=self.method_id,
            backend=result.provenance.backend,
            globalization=result.provenance.globalization,
            matrix_free=result.provenance.matrix_free,
            implicit_differentiation=result.provenance.implicit_differentiation,
            notes=result.provenance.notes,
        )
        return LeastSquaresResult(
            result.parameters,
            result.residual,
            result.objective,
            result.auxiliary,
            result.status,
            result.diagnostics,
            provenance,
        )


__all__ = [
    "DoglegLeastSquares",
    "DoglegMode",
    "TrustRegionReflective",
]
