#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._tree_math import validate_real_inexact_tree
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve as solve_linear,
)
from ._constrained_model import prepare_constrained_model
from ._iterative import (
    AbstractMinimizationMethod,
    LeastSquaresResult,
    MinimizationResult,
    NonlinearLeastSquaresProblem,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._structured_nonlinear import (
    StructuredNonlinearProgram,
    StructuredNonlinearWarmStart,
)


def _module(name: str):
    if importlib.util.find_spec(name) is None:
        raise ImportError(f"Optional nonlinear backend {name!r} is not installed.")
    return importlib.import_module(name)


def _certify_minimization(problem, parameters, args, termination, backend_success):
    objective, auxiliary = problem.value(parameters, args)
    gradient = jax.grad(lambda value: problem.value(value, args)[0])(parameters)
    flat_gradient, _ = ravel_pytree(gradient)
    if problem.constraints or problem.bounds is not None:
        constrained = prepare_constrained_model(problem, parameters, args=args)
        evaluation = constrained.evaluate(parameters, args)
        feasibility = evaluation.primal_feasibility
        raw_jacobian = evaluation.constraint_jacobian
        equality_jacobian = raw_jacobian[constrained.equality_indices]
        inequality_jacobian = jnp.concatenate(
            [
                raw_jacobian[constrained.lower_indices],
                -raw_jacobian[constrained.upper_indices],
            ],
            axis=0,
        )
        active = evaluation.inequality_slacks <= jnp.sqrt(termination.absolute_optimality)
        active_jacobian = inequality_jacobian[active]
        multiplier_matrix = jnp.concatenate(
            [
                jnp.conj(equality_jacobian.T),
                -jnp.conj(active_jacobian.T),
            ],
            axis=1,
        )
        if multiplier_matrix.shape[1]:
            multipliers = solve_linear(
                LeastSquaresProblem(DenseLinearOperator(multiplier_matrix)),
                -flat_gradient,
                policy=LinearSolvePolicy(DenseSVD()),
            ).value
            stationarity = flat_gradient + multiplier_matrix @ multipliers
        else:
            stationarity = flat_gradient
        optimality = jnp.maximum(
            jnp.linalg.norm(stationarity, ord=jnp.inf),
            feasibility,
        )
    else:
        optimality = jnp.linalg.norm(flat_gradient, ord=jnp.inf)
        feasibility = jnp.asarray(0.0, dtype=objective.dtype)
    certified = (
        jnp.asarray(backend_success)
        & jnp.isfinite(objective)
        & jnp.isfinite(optimality)
        & (optimality <= termination.absolute_optimality)
        & (feasibility <= termination.absolute_optimality)
    )
    return objective, auxiliary, optimality, feasibility, certified


class SciPyMinimize(AbstractMinimizationMethod):
    """Host SciPy minimization with independent Phydrax recertification."""

    method: str = eqx.field(static=True)
    options: dict[str, Any] = eqx.field(static=True)

    def __init__(
        self, method: str = "L-BFGS-B", /, *, options: dict[str, Any] | None = None
    ):
        identifier = str(method)
        if not identifier:
            raise ValueError("SciPy method must be non-empty.")
        self.method = identifier
        self.options = {} if options is None else dict(options)

    @property
    def method_id(self):
        return f"scipy/{self.method.lower()}"

    @property
    def capabilities(self):
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    def solve(self, problem, initial_parameters, /, *, termination, args):
        scipy_optimize = _module("scipy.optimize")
        parameters = validate_real_inexact_tree(initial_parameters, name="parameters")
        coordinates, unflatten = ravel_pytree(parameters)

        def objective(value):
            return float(problem.value(unflatten(jnp.asarray(value)), args)[0])

        def gradient(value):
            point = unflatten(jnp.asarray(value))
            result = jax.grad(lambda candidate: problem.value(candidate, args)[0])(point)
            return np.asarray(ravel_pytree(result)[0], dtype=float)

        bounds = None
        if problem.bounds is not None:
            lower, upper = problem.bounds.materialize(parameters)
            lower_coordinates, _ = ravel_pytree(lower)
            upper_coordinates, _ = ravel_pytree(upper)
            bounds = list(
                zip(
                    np.asarray(lower_coordinates),
                    np.asarray(upper_coordinates),
                    strict=True,
                )
            )
        constraints = []
        for constraint in problem.constraints:
            value = constraint.value(parameters, args)
            lower, upper = constraint.bounds(value)
            constraints.append(
                scipy_optimize.NonlinearConstraint(
                    lambda candidate, constraint=constraint: np.asarray(
                        ravel_pytree(
                            constraint.value(unflatten(jnp.asarray(candidate)), args)
                        )[0]
                    ),
                    np.asarray(ravel_pytree(lower)[0]),
                    np.asarray(ravel_pytree(upper)[0]),
                )
            )
        options = dict(self.options)
        options.setdefault("maxiter", termination.maximum_steps)
        result = scipy_optimize.minimize(
            objective,
            np.asarray(coordinates),
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method=self.method,
            options=options,
        )
        final_parameters = unflatten(jnp.asarray(result.x, dtype=coordinates.dtype))
        objective_value, auxiliary, optimality, feasibility, certified = (
            _certify_minimization(
                problem,
                final_parameters,
                args,
                termination,
                result.success,
            )
        )
        status = jnp.where(
            certified,
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.BACKEND_FAILED),
        ).astype(jnp.int32)
        diagnostics = OptimizationDiagnostics(
            iterations=int(result.nit),
            objective_evaluations=int(result.nfev),
            gradient_evaluations=int(result.njev),
            final_optimality_norm=optimality,
            primal_feasibility=feasibility,
        )
        return MinimizationResult(
            final_parameters,
            objective_value,
            auxiliary,
            status,
            diagnostics,
            OptimizationProvenance(
                problem_id=problem.problem_id,
                method=self.method_id,
                backend="scipy",
                globalization=self.method,
                matrix_free=False,
                implicit_differentiation=False,
                notes=str(result.message),
            ),
        )


class NLoptMinimize(AbstractMinimizationMethod):
    """Optional NLopt scalar minimization with a caller-selected algorithm ID."""

    algorithm: int = eqx.field(static=True)

    def __init__(self, algorithm: int, /):
        self.algorithm = int(algorithm)

    @property
    def method_id(self):
        return f"nlopt/{self.algorithm}"

    @property
    def capabilities(self):
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    def solve(self, problem, initial_parameters, /, *, termination, args):
        if problem.constraints:
            raise ValueError(
                "NLoptMinimize currently supports bounds only; nonlinear "
                "constraints require an explicit backend-specific method."
            )
        nlopt = _module("nlopt")
        parameters = validate_real_inexact_tree(initial_parameters, name="parameters")
        coordinates, unflatten = ravel_pytree(parameters)
        optimizer = nlopt.opt(self.algorithm, coordinates.size)
        optimizer.set_maxeval(
            termination.maximum_evaluations
            if termination.maximum_evaluations is not None
            else termination.maximum_steps * 10
        )
        optimizer.set_ftol_abs(termination.absolute_optimality)
        if problem.bounds is not None:
            lower, upper = problem.bounds.materialize(parameters)
            optimizer.set_lower_bounds(np.asarray(ravel_pytree(lower)[0]))
            optimizer.set_upper_bounds(np.asarray(ravel_pytree(upper)[0]))

        def objective(value, gradient_buffer):
            point = unflatten(jnp.asarray(value))
            if gradient_buffer.size:
                derivative = jax.grad(
                    lambda candidate: problem.value(candidate, args)[0]
                )(point)
                gradient_buffer[:] = np.asarray(ravel_pytree(derivative)[0])
            return float(problem.value(point, args)[0])

        optimizer.set_min_objective(objective)
        final_coordinates = optimizer.optimize(np.asarray(coordinates))
        final_parameters = unflatten(jnp.asarray(final_coordinates))
        objective_value, auxiliary, optimality, feasibility, certified = (
            _certify_minimization(
                problem,
                final_parameters,
                args,
                termination,
                optimizer.last_optimize_result() > 0,
            )
        )
        status = jnp.where(
            certified,
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.BACKEND_FAILED),
        ).astype(jnp.int32)
        return MinimizationResult(
            final_parameters,
            objective_value,
            auxiliary,
            status,
            OptimizationDiagnostics(
                objective_evaluations=optimizer.get_numevals(),
                final_optimality_norm=optimality,
                primal_feasibility=feasibility,
            ),
            OptimizationProvenance(
                problem_id=problem.problem_id,
                method=self.method_id,
                backend="nlopt",
                globalization="backend-selected",
                matrix_free=False,
                implicit_differentiation=False,
            ),
        )


class _StructuredIpoptCallbacks:
    def __init__(self, program: StructuredNonlinearProgram, args: Any, /):
        self.program = program
        self.args = args
        self.jacobian_rows = np.asarray(program.jacobian_plan.pattern.rows, dtype=np.int32)
        self.jacobian_cols = np.asarray(program.jacobian_plan.pattern.cols, dtype=np.int32)
        self._objective = jax.jit(lambda value: program.objective(value, args))
        self._gradient = jax.jit(
            jax.grad(lambda value: jnp.asarray(program.objective(value, args)))
        )
        self._constraints = jax.jit(lambda value: program.constraints(value, args))
        self._jacobian = jax.jit(
            lambda value: program.jacobian_plan.coefficients(value, args)
        )
        if program.hessian_plan is None:
            self.hessian_rows = None
            self.hessian_cols = None
            self.hessian_positions = None
            self._hessian = None
        else:
            hessian_plan = program.hessian_plan
            rows = np.asarray(hessian_plan.pattern.rows, dtype=np.int32)
            cols = np.asarray(hessian_plan.pattern.cols, dtype=np.int32)
            positions = np.flatnonzero(rows >= cols)
            self.hessian_rows = rows[positions]
            self.hessian_cols = cols[positions]
            self.hessian_positions = positions
            self._hessian = jax.jit(
                lambda value, objective_factor, multipliers: (
                    hessian_plan.coefficients(
                        value,
                        (args, objective_factor, multipliers),
                    )
                )
            )

    @staticmethod
    def _point(value: Any, /) -> Array:
        return jnp.asarray(value)

    def objective(self, value):
        return float(np.asarray(self._objective(self._point(value))))

    def gradient(self, value):
        return np.asarray(self._gradient(self._point(value)), dtype=float)

    def constraints(self, value):
        return np.asarray(self._constraints(self._point(value)), dtype=float)

    def jacobian(self, value):
        return np.asarray(self._jacobian(self._point(value)), dtype=float)

    def jacobianstructure(self):
        return self.jacobian_rows, self.jacobian_cols

    def hessian(self, value, multipliers, objective_factor):
        if self._hessian is None or self.hessian_positions is None:
            raise RuntimeError("Ipopt requested an unavailable exact Hessian.")
        coefficients = np.asarray(
            self._hessian(
                self._point(value),
                jnp.asarray(objective_factor),
                jnp.asarray(multipliers),
            ),
            dtype=float,
        )
        return coefficients[self.hessian_positions]

    def hessianstructure(self):
        if self.hessian_rows is None or self.hessian_cols is None:
            raise RuntimeError("Ipopt requested an unavailable exact Hessian structure.")
        return self.hessian_rows, self.hessian_cols


class IpoptMinimize(AbstractMinimizationMethod):
    """Optional cyipopt minimization boundary with Phydrax KKT recertification."""

    options: dict[str, Any] = eqx.field(static=True)

    def __init__(self, *, options: dict[str, Any] | None = None):
        self.options = {} if options is None else dict(options)

    @property
    def method_id(self):
        return "ipopt"

    @property
    def capabilities(self):
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=False,
        )

    def solve(self, problem, initial_parameters, /, *, termination, args):
        cyipopt = _module("cyipopt")
        scipy_optimize = _module("scipy.optimize")
        parameters = validate_real_inexact_tree(initial_parameters, name="parameters")
        coordinates, unflatten = ravel_pytree(parameters)
        options = dict(self.options)
        options.setdefault("max_iter", termination.maximum_steps)
        bounds = None
        if problem.bounds is not None:
            lower, upper = problem.bounds.materialize(parameters)
            bounds = list(
                zip(
                    np.asarray(ravel_pytree(lower)[0]),
                    np.asarray(ravel_pytree(upper)[0]),
                    strict=True,
                )
            )
        constraints = []
        for constraint in problem.constraints:
            value = constraint.value(parameters, args)
            lower, upper = constraint.bounds(value)
            constraints.append(
                scipy_optimize.NonlinearConstraint(
                    lambda candidate, constraint=constraint: np.asarray(
                        ravel_pytree(
                            constraint.value(
                                unflatten(jnp.asarray(candidate)),
                                args,
                            )
                        )[0]
                    ),
                    np.asarray(ravel_pytree(lower)[0]),
                    np.asarray(ravel_pytree(upper)[0]),
                )
            )
        result = cyipopt.minimize_ipopt(
            lambda value: float(problem.value(unflatten(jnp.asarray(value)), args)[0]),
            np.asarray(coordinates),
            jac=lambda value: np.asarray(
                ravel_pytree(
                    jax.grad(lambda candidate: problem.value(candidate, args)[0])(
                        unflatten(jnp.asarray(value))
                    )
                )[0]
            ),
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        final_parameters = unflatten(jnp.asarray(result.x))
        objective_value, auxiliary, optimality, feasibility, certified = (
            _certify_minimization(
                problem, final_parameters, args, termination, result.success
            )
        )
        return MinimizationResult(
            final_parameters,
            objective_value,
            auxiliary,
            jnp.where(
                certified,
                int(OptimizationStatus.SUCCESS),
                int(OptimizationStatus.BACKEND_FAILED),
            ),
            OptimizationDiagnostics(
                iterations=int(result.nit),
                objective_evaluations=int(result.nfev),
                gradient_evaluations=int(result.njev),
                final_optimality_norm=optimality,
                primal_feasibility=feasibility,
            ),
            OptimizationProvenance(
                problem_id=problem.problem_id,
                method=self.method_id,
                backend="ipopt",
                globalization="filter-interior-point",
                matrix_free=False,
                implicit_differentiation=False,
                notes=str(result.message),
            ),
        )

    def solve_structured(
        self,
        program: StructuredNonlinearProgram,
        initial_coordinates: Any,
        /,
        *,
        termination: OptimizationTermination,
        args: Any = None,
        warm_start: StructuredNonlinearWarmStart | None = None,
    ) -> MinimizationResult:
        """Solve an exact sparse bound-form NLP through low-level cyipopt callbacks."""
        if not isinstance(program, StructuredNonlinearProgram):
            raise TypeError("program must be a StructuredNonlinearProgram.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination.")
        coordinates = program.validate_coordinates(initial_coordinates)
        if warm_start is not None:
            if not isinstance(warm_start, StructuredNonlinearWarmStart):
                raise TypeError("warm_start must be StructuredNonlinearWarmStart or None.")
            if warm_start.structure_id != program.structure_id:
                raise ValueError("warm_start structure does not match the program.")
            coordinates = program.validate_coordinates(warm_start.primal)
            if warm_start.constraint_multipliers.shape != (program.num_constraints,):
                raise ValueError("warm_start constraint multipliers have the wrong shape.")

        cyipopt = _module("cyipopt")
        callbacks = _StructuredIpoptCallbacks(program, args)
        options = dict(self.options)
        options.setdefault("max_iter", termination.maximum_steps)
        options.setdefault("tol", termination.absolute_optimality)
        if program.hessian_plan is None:
            requested_hessian = options.get("hessian_approximation", "limited-memory")
            if requested_hessian != "limited-memory":
                raise ValueError(
                    "A structured program without an exact Hessian requires "
                    "hessian_approximation='limited-memory'."
                )
            options["hessian_approximation"] = "limited-memory"
        problem = cyipopt.Problem(
            n=program.num_variables,
            m=program.num_constraints,
            problem_obj=callbacks,
            lb=np.asarray(program.variable_lower, dtype=float),
            ub=np.asarray(program.variable_upper, dtype=float),
            cl=np.asarray(program.constraint_lower, dtype=float),
            cu=np.asarray(program.constraint_upper, dtype=float),
        )
        for name, value in options.items():
            problem.add_option(name, value)
        if warm_start is None:
            final_coordinates, info = problem.solve(np.asarray(coordinates, dtype=float))
        else:
            problem.add_option("warm_start_init_point", "yes")
            final_coordinates, info = problem.solve(
                np.asarray(coordinates, dtype=float),
                lagrange=np.asarray(warm_start.constraint_multipliers, dtype=float),
                zl=np.asarray(warm_start.lower_bound_multipliers, dtype=float),
                zu=np.asarray(warm_start.upper_bound_multipliers, dtype=float),
            )
        final = jnp.asarray(final_coordinates, dtype=coordinates.dtype)
        constraint_multipliers = jnp.asarray(
            info["mult_g"], dtype=coordinates.dtype
        )
        lower_multipliers = jnp.asarray(
            info["mult_x_L"], dtype=coordinates.dtype
        )
        upper_multipliers = jnp.asarray(
            info["mult_x_U"], dtype=coordinates.dtype
        )
        active_tolerance = float(np.sqrt(termination.absolute_optimality))
        certificate = program.certificate(
            final,
            constraint_multipliers,
            lower_multipliers,
            upper_multipliers,
            args,
            active_tolerance=active_tolerance,
        )
        evaluation = program.evaluate(final, args)
        stationarity = jnp.max(
            jnp.abs(jnp.asarray(certificate.stationarity_residual)),
            initial=0.0,
        )
        optimality = jnp.maximum(
            stationarity,
            jnp.maximum(
                certificate.primal_feasibility,
                jnp.maximum(
                    certificate.dual_feasibility,
                    certificate.complementarity,
                ),
            ),
        )
        backend_status = int(info["status"])
        backend_success = backend_status in (0, 1)
        certified = (
            evaluation.finite
            & jnp.asarray(backend_success)
            & (optimality <= termination.absolute_optimality)
        )
        public_status = jnp.where(
            certified,
            int(OptimizationStatus.SUCCESS),
            (
                int(OptimizationStatus.CERTIFICATION_FAILED)
                if backend_success
                else int(OptimizationStatus.BACKEND_FAILED)
            ),
        ).astype(jnp.int32)
        iterations = int(info.get("iter_count", 0))
        message = str(info["status_msg"])
        return MinimizationResult(
            final,
            evaluation.objective,
            None,
            public_status,
            OptimizationDiagnostics(
                iterations=iterations,
                final_optimality_norm=optimality,
                primal_feasibility=certificate.primal_feasibility,
                dual_feasibility=certificate.dual_feasibility,
                complementarity=certificate.complementarity,
                active_constraints=jnp.sum(certificate.active_mask),
                counts_complete=False,
            ),
            OptimizationProvenance(
                problem_id=program.program_id,
                method=self.method_id,
                backend="ipopt",
                globalization="filter-interior-point",
                matrix_free=False,
                implicit_differentiation=False,
                notes=message,
            ),
            certificate=certificate,
            method_evidence=info,
        )


def ceres_least_squares(
    problem: NonlinearLeastSquaresProblem,
    initial_parameters: PyTree[Any],
    solver: Callable[[NonlinearLeastSquaresProblem, PyTree[Any], Any], Any],
    /,
    *,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> LeastSquaresResult:
    if not callable(solver):
        raise TypeError("solver must be callable.")
    termination_ = OptimizationTermination() if termination is None else termination
    backend = solver(problem, initial_parameters, args)
    if not isinstance(backend, tuple) or len(backend) != 3:
        raise TypeError("Ceres boundary must return (parameters, success, summary).")
    parameters, backend_success, summary = backend
    residual, auxiliary = problem.value(parameters, args)
    residual_vector, _ = ravel_pytree(residual)
    gradient = jax.grad(
        lambda value: (
            0.5
            * jnp.real(
                jnp.vdot(
                    ravel_pytree(problem.value(value, args)[0])[0],
                    ravel_pytree(problem.value(value, args)[0])[0],
                )
            )
        )
    )(parameters)
    gradient_vector, _ = ravel_pytree(gradient)
    optimality = jnp.linalg.norm(gradient_vector, ord=jnp.inf)
    certified = (
        jnp.asarray(backend_success)
        & jnp.all(jnp.isfinite(residual_vector))
        & (optimality <= termination_.absolute_optimality)
    )
    return LeastSquaresResult(
        parameters,
        residual,
        0.5 * jnp.real(jnp.vdot(residual_vector, residual_vector)),
        auxiliary,
        jnp.where(
            certified,
            int(OptimizationStatus.SUCCESS),
            int(OptimizationStatus.BACKEND_FAILED),
        ),
        OptimizationDiagnostics(
            residual_evaluations=1,
            vjp_evaluations=1,
            final_optimality_norm=optimality,
        ),
        OptimizationProvenance(
            problem_id=problem.problem_id,
            method="ceres",
            backend="ceres-callback",
            globalization="backend-selected",
            matrix_free=False,
            implicit_differentiation=False,
            notes=str(summary),
        ),
    )


__all__ = [
    "IpoptMinimize",
    "NLoptMinimize",
    "SciPyMinimize",
    "ceres_least_squares",
]
