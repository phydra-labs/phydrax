#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import validate_real_inexact_tree
from ._external_backends import _certify_minimization, _module
from ._iterative import (
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)
from ._structured_method import (
    AbstractStructuredNonlinearMethod,
    StructuredNonlinearCapabilities,
)
from ._structured_nonlinear import (
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearProgram,
    StructuredNonlinearResult,
    StructuredNonlinearWarmStart,
    StructuredOptimizationWork,
)


_IPOPT_STATUS_NAMES = {
    0: "solve-succeeded",
    1: "acceptable-level",
    2: "infeasible-problem",
    3: "search-direction-too-small",
    4: "diverging-iterates",
    5: "user-requested-stop",
    6: "feasible-point-found",
    -1: "maximum-iterations",
    -2: "restoration-failed",
    -3: "step-computation-error",
    -4: "maximum-cpu-time",
    -5: "maximum-wall-time",
    -10: "insufficient-degrees-of-freedom",
    -11: "invalid-problem-definition",
    -12: "invalid-option",
    -13: "invalid-number-detected",
    -100: "unrecoverable-exception",
    -101: "non-ipopt-exception",
    -102: "insufficient-memory",
    -199: "internal-error",
}


def _mapped_status(status: int, /) -> OptimizationStatus:
    if status in (0, 1):
        return OptimizationStatus.SUCCESS
    if status == 2:
        return OptimizationStatus.INFEASIBLE
    if status == 3:
        return OptimizationStatus.STAGNATION
    if status == 4:
        return OptimizationStatus.DIVERGENCE
    if status == 5:
        return OptimizationStatus.BACKEND_FAILED
    if status == 6:
        return OptimizationStatus.CERTIFICATION_FAILED
    if status == -1:
        return OptimizationStatus.MAXIMUM_STEPS_REACHED
    if status == -2:
        return OptimizationStatus.RESTORATION_FAILED
    if status == -3:
        return OptimizationStatus.LINE_SEARCH_FAILED
    if status in (-4, -5):
        return OptimizationStatus.MAXIMUM_EVALUATIONS_REACHED
    if status == -10:
        return OptimizationStatus.CONSTRAINT_QUALIFICATION_FAILED
    if status == -13:
        return OptimizationStatus.NONFINITE_EVALUATION
    return OptimizationStatus.BACKEND_FAILED


def _canonical_structure(
    rows: Any,
    cols: Any,
    /,
    *,
    shape: tuple[int, int],
    owner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_array = np.asarray(rows, dtype=np.int32)
    col_array = np.asarray(cols, dtype=np.int32)
    if row_array.ndim != 1 or col_array.shape != row_array.shape:
        raise ValueError(f"{owner} row and column arrays must be equal rank-one arrays.")
    if np.any(row_array < 0) or np.any(row_array >= shape[0]):
        raise ValueError(f"{owner} row indices lie outside shape {shape}.")
    if np.any(col_array < 0) or np.any(col_array >= shape[1]):
        raise ValueError(f"{owner} column indices lie outside shape {shape}.")
    pairs = list(zip(row_array.tolist(), col_array.tolist(), strict=True))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"{owner} structure contains duplicate coordinates.")
    order = np.lexsort((col_array, row_array))
    return row_array[order], col_array[order], order.astype(np.int32)


def _canonical_hessian_structure(
    rows: Any,
    cols: Any,
    /,
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_array = np.asarray(rows, dtype=np.int32)
    col_array = np.asarray(cols, dtype=np.int32)
    if row_array.ndim != 1 or col_array.shape != row_array.shape:
        raise ValueError("Hessian row and column arrays must be equal rank-one arrays.")
    if (
        np.any(row_array < 0)
        or np.any(row_array >= size)
        or np.any(col_array < 0)
        or np.any(col_array >= size)
    ):
        raise ValueError("Hessian structure indices lie outside the variable space.")
    oriented = list(zip(row_array.tolist(), col_array.tolist(), strict=True))
    if len(set(oriented)) != len(oriented):
        raise ValueError("Hessian structure contains duplicate oriented coordinates.")
    representatives: dict[tuple[int, int], int] = {}
    for position, (row, col) in enumerate(oriented):
        pair = (max(row, col), min(row, col))
        selected = representatives.get(pair)
        if selected is None or row >= col:
            representatives[pair] = position
    pairs = sorted(representatives)
    positions = np.asarray([representatives[pair] for pair in pairs], dtype=np.int32)
    rows_lower = np.asarray([pair[0] for pair in pairs], dtype=np.int32)
    cols_lower = np.asarray([pair[1] for pair in pairs], dtype=np.int32)
    return rows_lower, cols_lower, positions


class IpoptCallbackCounts(StrictModule):
    """Complete host callback and conversion counts for one structured Ipopt solve."""

    objective: int = eqx.field(static=True)
    gradient: int = eqx.field(static=True)
    constraints: int = eqx.field(static=True)
    jacobian: int = eqx.field(static=True)
    hessian: int = eqx.field(static=True)
    intermediate: int = eqx.field(static=True)
    host_to_device: int = eqx.field(static=True)
    device_to_host: int = eqx.field(static=True)


class IpoptStatusEvidence(StrictModule):
    """Raw Ipopt termination plus its explicit Phydrax interpretation."""

    status: int = eqx.field(static=True)
    status_name: str = eqx.field(static=True)
    message: str = eqx.field(static=True)
    mapped_status: int = eqx.field(static=True)
    backend_success: bool = eqx.field(static=True)
    certified: bool = eqx.field(static=True)


class StructuredIpoptEvidence(StrictModule):
    """Typed sparse topology, callback, status, and warm-start evidence."""

    status: IpoptStatusEvidence
    counts: IpoptCallbackCounts
    final_warm_start: StructuredNonlinearWarmStart
    jacobian_nonzeros: int = eqx.field(static=True)
    hessian_nonzeros: int = eqx.field(static=True)
    exact_hessian: bool = eqx.field(static=True)
    warm_started: bool = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    jacobian_plan_id: str = eqx.field(static=True)
    hessian_plan_id: str | None = eqx.field(static=True)
    options_id: str = eqx.field(static=True)
    coordinate_dtype: str = eqx.field(static=True)
    host_dtype: str = eqx.field(static=True)


class _MutableCallbackCounts:
    def __init__(self):
        self.objective = 0
        self.gradient = 0
        self.constraints = 0
        self.jacobian = 0
        self.hessian = 0
        self.intermediate = 0
        self.host_to_device = 0
        self.device_to_host = 0

    def freeze(self) -> IpoptCallbackCounts:
        return IpoptCallbackCounts(
            objective=self.objective,
            gradient=self.gradient,
            constraints=self.constraints,
            jacobian=self.jacobian,
            hessian=self.hessian,
            intermediate=self.intermediate,
            host_to_device=self.host_to_device,
            device_to_host=self.device_to_host,
        )


class _StructuredIpoptCallbacks:
    def __init__(self, program: StructuredNonlinearProgram, args: Any, /):
        self.program = program
        self.args = args
        self.counts = _MutableCallbackCounts()
        (
            self.jacobian_rows,
            self.jacobian_cols,
            self.jacobian_positions,
        ) = _canonical_structure(
            program.jacobian_plan.pattern.rows,
            program.jacobian_plan.pattern.cols,
            shape=(program.num_constraints, program.num_variables),
            owner="Jacobian",
        )
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
            (
                self.hessian_rows,
                self.hessian_cols,
                self.hessian_positions,
            ) = _canonical_hessian_structure(
                hessian_plan.pattern.rows,
                hessian_plan.pattern.cols,
                size=program.num_variables,
            )
            self._hessian = jax.jit(
                lambda value, objective_factor, multipliers: hessian_plan.coefficients(
                    value,
                    (args, objective_factor, multipliers),
                )
            )

    def _point(self, value: Any, /) -> Array:
        self.counts.host_to_device += 1
        return jnp.asarray(value)

    def _numpy(self, value: Any, owner: str, /) -> np.ndarray:
        self.counts.device_to_host += 1
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(
                f"Structured Ipopt {owner} returned nonfinite values."
            )
        return array

    def objective(self, value):
        self.counts.objective += 1
        return float(self._numpy(self._objective(self._point(value)), "objective"))

    def gradient(self, value):
        self.counts.gradient += 1
        return self._numpy(self._gradient(self._point(value)), "gradient")

    def constraints(self, value):
        self.counts.constraints += 1
        return self._numpy(self._constraints(self._point(value)), "constraints")

    def jacobian(self, value):
        self.counts.jacobian += 1
        coefficients = self._numpy(self._jacobian(self._point(value)), "Jacobian")
        if coefficients.shape != (self.program.jacobian_plan.nnz,):
            raise ValueError("Structured Ipopt Jacobian returned the wrong value count.")
        return coefficients[self.jacobian_positions]

    def jacobianstructure(self):
        return self.jacobian_rows, self.jacobian_cols

    def hessian(self, value, multipliers, objective_factor):
        if self._hessian is None or self.hessian_positions is None:
            return np.empty((0,), dtype=float)
        self.counts.hessian += 1
        self.counts.host_to_device += 2
        coefficients = self._numpy(
            self._hessian(
                self._point(value),
                jnp.asarray(objective_factor),
                jnp.asarray(multipliers),
            ),
            "Hessian",
        )
        assert self.program.hessian_plan is not None
        if coefficients.shape != (self.program.hessian_plan.nnz,):
            raise ValueError("Structured Ipopt Hessian returned the wrong value count.")
        return coefficients[self.hessian_positions]

    def hessianstructure(self):
        if self.hessian_rows is None or self.hessian_cols is None:
            empty = np.empty((0,), dtype=np.int32)
            return empty, empty
        return self.hessian_rows, self.hessian_cols

    def intermediate(
        self,
        algorithm_mode,
        iteration,
        objective,
        primal_infeasibility,
        dual_infeasibility,
        barrier_parameter,
        step_norm,
        regularization_size,
        primal_step,
        dual_step,
        line_search_trials,
    ):
        del (
            algorithm_mode,
            iteration,
            objective,
            primal_infeasibility,
            dual_infeasibility,
            barrier_parameter,
            step_norm,
            regularization_size,
            primal_step,
            dual_step,
            line_search_trials,
        )
        self.counts.intermediate += 1
        return True


class IpoptMinimize(AbstractStructuredNonlinearMethod):
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

    @property
    def structured_capabilities(self):
        return StructuredNonlinearCapabilities(
            exact_sparse_jacobian=True,
            exact_sparse_hessian=True,
            limited_memory_hessian=True,
            portable_warm_start=True,
            numeric_refresh=True,
            jit=False,
            ordinary_batch=False,
            pooled_batch=False,
            implicit_differentiation=False,
            device_execution=False,
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

    def _structured_options(
        self,
        program: StructuredNonlinearProgram,
        termination: OptimizationTermination,
        /,
    ) -> dict[str, Any]:
        reserved = {"max_iter", "tol", "warm_start_init_point"}
        conflicts = sorted(reserved.intersection(self.options))
        if conflicts:
            raise ValueError(
                "Structured Ipopt options are owned by Phydrax termination/warm-start "
                f"semantics: {conflicts}."
            )
        options = dict(self.options)
        options["max_iter"] = termination.maximum_steps
        options["tol"] = termination.absolute_optimality
        requested_hessian = options.get("hessian_approximation")
        if program.hessian_plan is None:
            if requested_hessian not in (None, "limited-memory"):
                raise ValueError(
                    "A structured program without an exact Hessian requires "
                    "hessian_approximation='limited-memory'."
                )
            options["hessian_approximation"] = "limited-memory"
        elif requested_hessian is not None:
            raise ValueError(
                "An exact structured Hessian cannot be combined with an explicit "
                "hessian_approximation option."
            )
        return options

    def solve_structured(
        self,
        prepared: PreparedStructuredNonlinearProgram,
        initial_coordinates: Any,
        /,
        *,
        termination: OptimizationTermination,
        warm_start: StructuredNonlinearWarmStart | None = None,
    ) -> StructuredNonlinearResult:
        """Solve an exact sparse bound-form NLP through low-level cyipopt callbacks."""
        if not isinstance(prepared, PreparedStructuredNonlinearProgram):
            raise TypeError("prepared must be a PreparedStructuredNonlinearProgram.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination.")
        program = prepared.program
        coordinates = prepared.validate_coordinates(initial_coordinates)
        if warm_start is not None:
            if not isinstance(warm_start, StructuredNonlinearWarmStart):
                raise TypeError(
                    "warm_start must be StructuredNonlinearWarmStart or None."
                )
            if warm_start.structure_id != program.structure_id:
                raise ValueError("warm_start structure does not match the program.")
            if (
                warm_start.source_program_id is not None
                and warm_start.source_program_id != program.program_id
            ):
                raise ValueError("warm_start program does not match the program.")
            coordinates = program.validate_coordinates(warm_start.primal)
            if warm_start.constraint_multipliers.shape != (program.num_constraints,):
                raise ValueError(
                    "warm_start constraint multipliers have the wrong shape."
                )

        cyipopt = _module("cyipopt")
        callbacks = _StructuredIpoptCallbacks(program, prepared.args)
        options = self._structured_options(program, termination)
        problem = cyipopt.Problem(
            n=program.num_variables,
            m=program.num_constraints,
            problem_obj=callbacks,
            lb=np.asarray(prepared.variable_lower, dtype=float),
            ub=np.asarray(prepared.variable_upper, dtype=float),
            cl=np.asarray(prepared.constraint_lower, dtype=float),
            cu=np.asarray(prepared.constraint_upper, dtype=float),
        )
        if warm_start is not None:
            options["warm_start_init_point"] = "yes"
        for name, value in options.items():
            problem.add_option(name, value)
        if warm_start is None:
            final_coordinates, info = problem.solve(np.asarray(coordinates, dtype=float))
        else:
            final_coordinates, info = problem.solve(
                np.asarray(coordinates, dtype=float),
                lagrange=np.asarray(warm_start.constraint_multipliers, dtype=float),
                zl=np.asarray(warm_start.lower_bound_multipliers, dtype=float),
                zu=np.asarray(warm_start.upper_bound_multipliers, dtype=float),
            )
        final = jnp.asarray(final_coordinates, dtype=coordinates.dtype)
        constraint_multipliers = jnp.asarray(info["mult_g"], dtype=coordinates.dtype)
        lower_multipliers = jnp.asarray(info["mult_x_L"], dtype=coordinates.dtype)
        upper_multipliers = jnp.asarray(info["mult_x_U"], dtype=coordinates.dtype)
        active_tolerance = float(np.sqrt(termination.absolute_optimality))
        certificate = prepared.certificate(
            final,
            constraint_multipliers,
            lower_multipliers,
            upper_multipliers,
            active_tolerance=active_tolerance,
        )
        evaluation = prepared.evaluate(final)
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
        certified_array = (
            evaluation.finite
            & jnp.asarray(backend_success)
            & (optimality <= termination.absolute_optimality)
        )
        certified = bool(np.asarray(certified_array))
        mapped = _mapped_status(backend_status)
        public_status = (
            OptimizationStatus.SUCCESS
            if certified
            else (OptimizationStatus.CERTIFICATION_FAILED if backend_success else mapped)
        )
        message = str(info["status_msg"])
        status_evidence = IpoptStatusEvidence(
            status=backend_status,
            status_name=_IPOPT_STATUS_NAMES.get(backend_status, "unknown"),
            message=message,
            mapped_status=int(mapped),
            backend_success=backend_success,
            certified=certified,
        )
        final_warm_start = StructuredNonlinearWarmStart(
            final,
            constraint_multipliers,
            lower_multipliers,
            upper_multipliers,
            structure_id=program.structure_id,
            numeric_version=prepared.numeric_version,
            source_program_id=program.program_id,
            source_backend="ipopt",
        )
        hessian_plan_id = (
            None if program.hessian_plan is None else program.hessian_plan.plan_id
        )
        evidence = StructuredIpoptEvidence(
            status_evidence,
            callbacks.counts.freeze(),
            final_warm_start,
            jacobian_nonzeros=int(callbacks.jacobian_rows.size),
            hessian_nonzeros=(
                0 if callbacks.hessian_rows is None else int(callbacks.hessian_rows.size)
            ),
            exact_hessian=program.hessian_plan is not None,
            warm_started=warm_start is not None,
            program_id=program.program_id,
            structure_id=program.structure_id,
            jacobian_plan_id=program.jacobian_plan.plan_id,
            hessian_plan_id=hessian_plan_id,
            options_id=canonical_fingerprint(options),
            coordinate_dtype=str(coordinates.dtype),
            host_dtype=str(np.dtype(float)),
        )
        counts = evidence.counts
        optimization = MinimizationResult(
            final,
            evaluation.objective,
            None,
            jnp.asarray(int(public_status), dtype=jnp.int32),
            OptimizationDiagnostics(
                iterations=counts.intermediate,
                objective_evaluations=counts.objective,
                constraint_evaluations=counts.constraints,
                gradient_evaluations=counts.gradient,
                jacobian_evaluations=counts.jacobian,
                hvp_evaluations=counts.hessian,
                final_optimality_norm=optimality,
                primal_feasibility=certificate.primal_feasibility,
                dual_feasibility=certificate.dual_feasibility,
                complementarity=certificate.complementarity,
                active_constraints=jnp.sum(certificate.active_mask),
                counts_complete=True,
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
            method_evidence=evidence,
        )
        return StructuredNonlinearResult(
            optimization,
            final_warm_start,
            StructuredOptimizationWork(
                objective_evaluations=counts.objective,
                constraint_evaluations=counts.constraints,
                gradient_evaluations=counts.gradient,
                jacobian_evaluations=counts.jacobian,
                hessian_evaluations=counts.hessian,
                certificate_evaluations=1,
                complete=True,
            ),
            numeric_version=prepared.numeric_version,
            structure_id=prepared.structure_id,
            method_id=self.method_id,
        )


__all__ = [
    "IpoptCallbackCounts",
    "IpoptMinimize",
    "IpoptStatusEvidence",
    "StructuredIpoptEvidence",
]
