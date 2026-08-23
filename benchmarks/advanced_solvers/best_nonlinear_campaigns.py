#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import phydrax as phx
from benchmarks.advanced_solvers.nonlinear_peer_runners import (
    load_peer_specs,
    make_runner_request,
    python_runtime_identity,
    run_external_peer,
    run_python_peer,
    stable_fingerprint,
)


Family = Literal[
    "root",
    "least-squares",
    "constrained",
    "global",
    "differentiation",
    "adversarial",
]


@dataclass(frozen=True, slots=True)
class CampaignObservation:
    family: str
    case_id: str
    implementation: str
    available: bool
    availability_reason: str
    availability_detail: str | None
    expected_identity: str | None
    observed_identity: str | None
    source_revision: str | None
    initial_fingerprint: str
    result_fingerprint: str | None
    backend_success: bool | None
    backend_scope: str
    backend_status: str | None
    certified: bool | None
    certificate_kind: str
    certificate_scope: str
    certificate_value: float | None
    certificate_tolerance: float | None
    certificate_components: dict[str, float | None]
    work: float | None
    work_unit: str | None
    work_counts: dict[str, float]
    cold_seconds: float | None
    warmup_seconds: tuple[float, ...]
    steady_seconds: tuple[float, ...]
    objective: float | None = None
    feasibility: float | None = None
    derivative_error: float | None = None

    @property
    def steady_median(self) -> float | None:
        return None if not self.steady_seconds else statistics.median(self.steady_seconds)


@dataclass(frozen=True, slots=True)
class PerformanceProfilePoint:
    family: str
    metric: Literal["primary-work", "steady-solve"]
    work_unit: str | None
    tau: float
    implementation: str
    fraction: float
    eligible_cases: int
    certified_cases: int


_PEER_MANIFEST_PATH = Path(__file__).with_name("nonlinear_peer_manifest.json")
_PEER_SPECS = load_peer_specs(_PEER_MANIFEST_PATH)
_IMPLEMENTATION_PEER = {
    "nonlinearsolve-jl": "nonlinearsolve-jl",
    "optimistix-newton": "optimistix",
    "scipy-root": "scipy",
    "scipy-least-squares": "scipy",
    "scipy-trust-constr": "scipy",
    "scipy-differential-evolution": "scipy",
    "ceres": "ceres",
    "ipopt": "ipopt",
    "nlopt": "nlopt",
    "theseus": "theseus",
    "gtsam": "gtsam",
}


def _peer_spec(implementation: str):
    peer_id = _IMPLEMENTATION_PEER.get(implementation)
    return None if peer_id is None else _PEER_SPECS[peer_id]


@dataclass(frozen=True, slots=True)
class _RawObservation:
    family: str
    case_id: str
    implementation: str
    available: bool
    certified: bool | None
    status: str
    work: float | None
    steady_seconds: float | None
    certificate: float | None
    objective: float | None = None
    feasibility: float | None = None
    derivative_error: float | None = None
    backend_claimed_success: bool | None = None
    availability_reason: str | None = None
    availability_detail: str | None = None
    expected_identity: str | None = None
    observed_identity: str | None = None
    source_revision: str | None = None
    work_counts: dict[str, float] | None = None
    certificate_components: dict[str, float | None] | None = None
    solution: Any = None


def _unavailable(
    family: str,
    case_id: str,
    implementation: str,
    *,
    reason: str = "runtime-missing",
    detail: str | None = None,
):
    spec = _peer_spec(implementation)
    observed_identity = (
        python_runtime_identity(spec)
        if spec is not None and spec.runner_kind == "python-distribution"
        else None
    )
    return _RawObservation(
        family,
        case_id,
        implementation,
        False,
        None,
        "runtime-unavailable",
        None,
        None,
        None,
        availability_reason=reason,
        availability_detail=(
            detail
            if detail is not None
            else f"No canonical runner is available for {implementation!r}."
        ),
        expected_identity=None if spec is None else spec.expected_identity,
        observed_identity=observed_identity,
        source_revision=None if spec is None else spec.source_revision,
    )


def _root_cases():
    return {
        "diagonal-polynomial": (
            lambda x, a: x * x - a,
            jnp.ones(8),
            jnp.arange(1.0, 9.0),
        ),
        "trigonometric": (
            lambda x, a: jnp.sin(x) - a,
            jnp.full((8,), 0.5),
            jnp.linspace(-0.75, 0.75, 8),
        ),
        "brown-almost-linear": (
            lambda x, a: jnp.concatenate(
                [
                    x[:-1] + jnp.sum(x) - (x.size + 1.0),
                    jnp.asarray([jnp.prod(x) - 1.0]),
                ]
            ),
            jnp.full((8,), 0.5),
            None,
        ),
        "domain-restricted": (
            lambda x, a: jnp.where(x > 0.0, jnp.log(x) - a, jnp.nan),
            jnp.full((8,), 0.5),
            jnp.linspace(-1.0, 1.0, 8),
        ),
    }


def _run_root(case_id, implementation):
    function, initial, args = _root_cases()[case_id]
    problem = phx.nonlinear.NonlinearSystemProblem(function, problem_id=case_id)
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-8,
        relative_residual=0.0,
        maximum_steps=200,
        maximum_evaluations=4000,
        maximum_linear_iterations=20000,
    )
    if implementation == "nonlinearsolve-jl":
        return _unavailable("root", case_id, implementation)
    if implementation == "scipy-root":
        scipy_optimize = __import__(
            "scipy.optimize",
            fromlist=["root"],
        )
        start = time.perf_counter()
        result = scipy_optimize.root(
            lambda value: np.asarray(
                function(jnp.asarray(value), args),
                dtype=float,
            ),
            np.asarray(initial),
            method="hybr",
            options={"maxfev": termination.maximum_evaluations},
        )
        elapsed = time.perf_counter() - start
        state = jnp.asarray(result.x)
        residual = function(state, args)
        certificate = float(jnp.linalg.norm(residual) / (1.0 + jnp.linalg.norm(initial)))
        return _RawObservation(
            "root",
            case_id,
            implementation,
            True,
            bool(jnp.all(jnp.isfinite(residual)) and certificate <= 1e-8),
            str(result.status),
            float(result.nfev),
            elapsed,
            certificate,
            backend_claimed_success=bool(result.success),
            solution=state,
            work_counts={"residual_evaluations": float(result.nfev)},
        )
    if implementation == "optimistix-newton":
        if importlib.util.find_spec("optimistix") is None:
            return _unavailable("root", case_id, implementation)
        optx = __import__("optimistix")
        solver = optx.Newton(rtol=0.0, atol=1e-8)
        start = time.perf_counter()
        result = optx.root_find(
            function,
            solver,
            initial,
            args=args,
            max_steps=termination.maximum_steps,
            throw=False,
        )
        jax.block_until_ready(result.value)
        elapsed = time.perf_counter() - start
        residual = function(result.value, args)
        certificate = float(jnp.linalg.norm(residual) / (1.0 + jnp.linalg.norm(initial)))
        return _RawObservation(
            "root",
            case_id,
            implementation,
            True,
            bool(jnp.all(jnp.isfinite(residual)) and certificate <= 1e-8),
            str(result.result),
            float(result.stats["num_steps"]),
            elapsed,
            certificate,
            backend_claimed_success=bool(result.result == optx.RESULTS.successful),
            solution=result.value,
            work_counts={"iterations": float(result.stats["num_steps"])},
        )
    methods = {
        "phydrax-newton": phx.nonlinear.NewtonKrylov(),
        "phydrax-robust": phx.nonlinear.RobustRoot(),
        "phydrax-broyden": phx.nonlinear.Broyden("good"),
        "phydrax-dfsane": phx.nonlinear.DFSANE(),
    }
    if implementation not in methods:
        return None
    start = time.perf_counter()
    result = methods[implementation].solve(
        problem,
        initial,
        termination=termination,
        args=args,
    )
    jax.block_until_ready(result.state)
    elapsed = time.perf_counter() - start
    residual = function(result.state, args)
    certificate = float(jnp.linalg.norm(residual) / (1.0 + jnp.linalg.norm(initial)))
    work = float(result.diagnostics.residual_evaluations)
    return _RawObservation(
        "root",
        case_id,
        implementation,
        True,
        bool(jnp.all(jnp.isfinite(residual)) & (certificate <= 1e-8)),
        str(int(result.status)),
        work,
        elapsed,
        certificate,
        backend_claimed_success=bool(result.successful),
        solution=result.state,
        work_counts={
            "residual_evaluations": float(result.diagnostics.residual_evaluations),
            "jvp_evaluations": float(result.diagnostics.jvp_evaluations),
            "vjp_evaluations": float(result.diagnostics.vjp_evaluations),
            "linear_iterations": float(result.diagnostics.linear_iterations),
        },
    )


def _least_squares_cases():
    time_axis = jnp.linspace(0.0, 1.0, 32)
    observations = 3.0 * jnp.exp(-2.0 * time_axis)
    return {
        "exponential-fit": (
            lambda x, a: x[0] * jnp.exp(-x[1] * time_axis) - observations,
            jnp.asarray([1.0, 1.0]),
            None,
            None,
        ),
        "rank-deficient": (
            lambda x, a: jnp.asarray([x[0] + x[1] - 1.0, 2.0 * x[0] + 2.0 * x[1] - 2.0]),
            jnp.zeros(2),
            None,
            None,
        ),
        "active-bounds": (
            lambda x, a: x - a,
            jnp.asarray([0.2, 0.2]),
            jnp.asarray([2.0, -0.5]),
            phx.optim.Bounds(0.0, 1.0),
        ),
        "robust-outlier": (
            lambda x, a: x[0] - a,
            jnp.asarray([0.0]),
            jnp.concatenate([jnp.ones(31), jnp.asarray([100.0])]),
            None,
        ),
    }


def _run_least_squares(case_id, implementation):
    if case_id == "robust-outlier" and implementation == "phydrax-pounders":
        return _unavailable(
            "least-squares",
            case_id,
            implementation,
            reason="unsupported-case",
            detail="POUNDERS does not implement robust-loss residual models.",
        )
    residual, initial, args, bounds = _least_squares_cases()[case_id]
    if case_id == "robust-outlier":
        graph = phx.optim.ResidualGraphProblem.from_residual(residual, problem_id=case_id)
        block = phx.optim.ResidualBlock(
            lambda values, current_args: residual(values[0], current_args),
            ("parameters",),
            loss=phx.optim.HuberLoss(1.0),
            block_id="robust",
        )
        graph = phx.optim.ResidualGraphProblem(
            graph.parameter_blocks, (block,), problem_id=case_id
        )
        problem = graph.as_least_squares_problem()
    else:
        problem = phx.optim.NonlinearLeastSquaresProblem(
            residual,
            bounds=bounds,
            problem_id=case_id,
        )
    if implementation in (
        "nonlinearsolve-jl",
        "ceres",
        "theseus",
    ):
        return _unavailable("least-squares", case_id, implementation)
    if implementation == "scipy-least-squares":
        if case_id == "robust-outlier":
            return _unavailable(
                "least-squares",
                case_id,
                implementation,
                reason="unsupported-case",
                detail="The canonical SciPy runner excludes graph robust losses.",
            )
        scipy_optimize = __import__(
            "scipy.optimize",
            fromlist=["least_squares"],
        )
        lower = -jnp.inf * jnp.ones_like(initial)
        upper = jnp.inf * jnp.ones_like(initial)
        if bounds is not None:
            lower, upper = bounds.materialize(initial)
        start = time.perf_counter()
        scipy_result = scipy_optimize.least_squares(
            lambda value: np.asarray(
                residual(jnp.asarray(value), args),
                dtype=float,
            ),
            np.asarray(initial),
            bounds=(np.asarray(lower), np.asarray(upper)),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=5000,
        )
        elapsed = time.perf_counter() - start
        parameters = jnp.asarray(scipy_result.x)
        residual_value = residual(parameters, args)
        gradient = jnp.asarray(scipy_result.grad)
        stationarity = (
            gradient
            if bounds is None
            else bounds.projected_gradient(parameters, gradient)
        )
        certificate = float(jnp.linalg.norm(stationarity, ord=jnp.inf))
        return _RawObservation(
            "least-squares",
            case_id,
            implementation,
            True,
            bool(jnp.all(jnp.isfinite(residual_value)) & (certificate <= 1e-6)),
            str(scipy_result.status),
            float(scipy_result.nfev),
            elapsed,
            certificate,
            objective=float(0.5 * jnp.real(jnp.vdot(residual_value, residual_value))),
            feasibility=float(0.0 if bounds is None else bounds.violation(parameters)),
            backend_claimed_success=bool(scipy_result.success),
            solution=parameters,
            work_counts={
                "residual_evaluations": float(scipy_result.nfev),
                "jacobian_evaluations": float(scipy_result.njev),
            },
        )
    methods = {
        "phydrax-lm": (
            phx.optim.BoundedLevenbergMarquardt()
            if bounds is not None
            else phx.optim.LevenbergMarquardt()
        ),
        "phydrax-dogleg": phx.optim.DoglegLeastSquares(
            "dogbox" if bounds is not None else "traditional"
        ),
        "phydrax-pounders": phx.optim.POUNDERS(initial_radius=0.25),
    }
    method = methods.get(implementation)
    if method is None:
        return None
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=200,
        maximum_evaluations=5000,
    )
    start = time.perf_counter()
    result = phx.optim.least_squares(
        problem,
        initial,
        method=method,
        termination=termination,
        args=args,
    )
    jax.block_until_ready(result.parameters)
    elapsed = time.perf_counter() - start
    residual_value = problem.value(result.parameters, args)[0]
    flat, _ = ravel_pytree(residual_value)
    flat_parameters, unflatten = ravel_pytree(result.parameters)

    def flat_residual(coordinates):
        value = problem.value(unflatten(coordinates), args)[0]
        return ravel_pytree(value)[0]

    jacobian = jax.jacfwd(flat_residual)(flat_parameters)
    gradient = jnp.conj(jacobian.T) @ flat
    if bounds is not None:
        gradient = ravel_pytree(
            bounds.projected_gradient(
                result.parameters,
                unflatten(gradient),
            )
        )[0]
    certificate = float(jnp.linalg.norm(gradient, ord=jnp.inf))
    feasibility = float(0.0 if bounds is None else bounds.violation(result.parameters))
    independently_certified = (
        jnp.all(jnp.isfinite(flat))
        & jnp.isfinite(certificate)
        & (certificate <= termination.absolute_optimality)
        & (feasibility <= termination.absolute_optimality)
    )
    return _RawObservation(
        "least-squares",
        case_id,
        implementation,
        True,
        bool(independently_certified),
        str(int(result.status)),
        float(result.diagnostics.residual_evaluations),
        elapsed,
        certificate,
        objective=float(0.5 * jnp.vdot(flat, flat).real),
        feasibility=feasibility,
        backend_claimed_success=bool(result.successful),
        solution=result.parameters,
        work_counts={
            "residual_evaluations": float(result.diagnostics.residual_evaluations),
            "jvp_evaluations": float(result.diagnostics.jvp_evaluations),
            "vjp_evaluations": float(result.diagnostics.vjp_evaluations),
            "linear_iterations": float(result.diagnostics.linear_iterations),
        },
    )


def _constrained_cases():
    equality = phx.optim.NonlinearConstraint(
        lambda x, a: jnp.asarray([jnp.sum(x)]),
        lower=1.0,
        upper=1.0,
        constraint_id="sum",
    )
    circle = phx.optim.NonlinearConstraint(
        lambda x, a: jnp.asarray([jnp.sum(x * x)]),
        lower=1.0,
        upper=1.0,
        constraint_id="circle",
    )
    return {
        "equality-quadratic": phx.optim.MinimizationProblem(
            lambda x, a: jnp.sum((x - jnp.asarray([0.2, 0.8])) ** 2),
            constraints=(equality,),
            problem_id="equality-quadratic",
        ),
        "circle": phx.optim.MinimizationProblem(
            lambda x, a: jnp.sum((x - jnp.asarray([0.0, 1.0])) ** 2),
            constraints=(circle,),
            problem_id="circle",
        ),
        "bound-equality": phx.optim.MinimizationProblem(
            lambda x, a: jnp.sum((x - jnp.asarray([0.2, 0.8])) ** 2),
            bounds=phx.optim.Bounds(0.0, jnp.inf),
            constraints=(equality,),
            problem_id="bound-equality",
        ),
    }


def _run_constrained(case_id, implementation):
    problem = _constrained_cases()[case_id]
    methods = {
        "phydrax-sqp": phx.optim.SQP(
            filter_globalization=phx.optim.FilterGlobalization(),
            hessian_update="exact",
        ),
        "phydrax-ipm": phx.optim.FilterInteriorPoint(),
        "scipy-trust-constr": phx.optim.SciPyMinimize(
            "trust-constr",
            options={"gtol": 1e-9},
        ),
        "ipopt": None,
    }
    method = methods.get(implementation)
    if method is None:
        return (
            _unavailable("constrained", case_id, implementation)
            if implementation == "ipopt"
            else None
        )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=200,
    )
    start = time.perf_counter()
    result = phx.optim.minimize(
        problem,
        jnp.asarray([0.5, 0.5]),
        method=method,
        termination=termination,
    )
    jax.block_until_ready(result.parameters)
    elapsed = time.perf_counter() - start
    prepared = phx.optim.prepare_constrained_model(
        problem,
        result.parameters,
    )
    evaluation = prepared.evaluate(result.parameters)
    raw_jacobian = evaluation.constraint_jacobian
    equality_jacobian = raw_jacobian[prepared.equality_indices]
    inequality_jacobian = jnp.concatenate(
        [
            raw_jacobian[prepared.lower_indices],
            -raw_jacobian[prepared.upper_indices],
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
        multipliers = jnp.linalg.lstsq(
            multiplier_matrix,
            -evaluation.gradient,
            rcond=None,
        )[0]
        stationarity = evaluation.gradient + multiplier_matrix @ multipliers
    else:
        stationarity = evaluation.gradient
    dual = jnp.linalg.norm(stationarity, ord=jnp.inf)
    certificate = float(jnp.maximum(evaluation.primal_feasibility, dual))
    independently_certified = evaluation.finite & (
        certificate <= termination.absolute_optimality
    )
    return _RawObservation(
        "constrained",
        case_id,
        implementation,
        True,
        bool(independently_certified),
        str(int(result.status)),
        float(result.diagnostics.objective_evaluations),
        elapsed,
        certificate,
        objective=float(result.objective),
        feasibility=float(evaluation.primal_feasibility),
        backend_claimed_success=bool(result.successful),
        solution=result.parameters,
        work_counts={
            "objective_evaluations": float(result.diagnostics.objective_evaluations),
            "gradient_evaluations": float(result.diagnostics.gradient_evaluations),
            "constraint_evaluations": float(result.diagnostics.constraint_evaluations),
            "linear_solves": float(result.diagnostics.linear_solves),
            "linear_iterations": float(result.diagnostics.linear_iterations),
        },
    )


def _global_cases():
    return {
        "rastrigin": lambda x: (
            10.0 * x.size + jnp.sum(x * x - 10.0 * jnp.cos(2.0 * jnp.pi * x))
        ),
        "double-well": lambda x: jnp.sum((x * x - 1.0) ** 2),
        "rosenbrock": lambda x: jnp.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2
        ),
    }


def _run_global(case_id, implementation):
    objective = _global_cases()[case_id]
    problem = phx.optim.MinimizationProblem(
        lambda x, a: objective(x),
        bounds=phx.optim.Bounds(-5.0, 5.0),
        problem_id=case_id,
    )
    termination = phx.optim.OptimizationTermination(
        absolute_optimality=1e-5,
        relative_optimality=0.0,
        maximum_steps=1000,
        maximum_evaluations=20000,
    )
    if implementation == "nlopt":
        return _unavailable("global", case_id, implementation)
    if implementation == "scipy-differential-evolution":
        scipy_optimize = __import__(
            "scipy.optimize",
            fromlist=["differential_evolution"],
        )
        start = time.perf_counter()
        scipy_result = scipy_optimize.differential_evolution(
            lambda value: float(objective(jnp.asarray(value))),
            [(-5.0, 5.0)] * 4,
            seed=11,
            maxiter=250,
            popsize=8,
            polish=True,
        )
        elapsed = time.perf_counter() - start
        parameters = jnp.asarray(scipy_result.x)
        gradient = jax.grad(objective)(parameters)
        objective_value = float(scipy_result.fun)
        feasibility = 0.0
        certificate = abs(objective_value)
        return _RawObservation(
            "global",
            case_id,
            implementation,
            True,
            bool(
                jnp.isfinite(scipy_result.fun)
                and certificate <= termination.absolute_optimality
            ),
            "0" if scipy_result.success else "1",
            float(scipy_result.nfev),
            elapsed,
            certificate,
            objective=objective_value,
            feasibility=feasibility,
            backend_claimed_success=bool(scipy_result.success),
            solution=parameters,
            work_counts={
                "objective_evaluations": float(scipy_result.nfev),
            },
            certificate_components={
                "objective_gap": certificate,
                "projected_stationarity": float(jnp.linalg.norm(gradient, ord=jnp.inf)),
                "feasibility": feasibility,
            },
        )
    start = time.perf_counter()
    if implementation == "phydrax-multistart":
        result = phx.optim.multistart_minimize(
            problem,
            jnp.full((4,), 0.25),
            policy=phx.optim.MultiStartPolicy(count=16, seed=11),
            termination=termination,
        ).best
    elif implementation == "phydrax-bobyqa":
        result = phx.optim.minimize(
            problem,
            jnp.full((4,), 0.25),
            method=phx.optim.BOBYQA(initial_radius=1.0),
            termination=termination,
        )
    else:
        return None
    jax.block_until_ready(result.parameters)
    elapsed = time.perf_counter() - start
    gradient = jax.grad(objective)(result.parameters)
    projected = problem.bounds.projected_gradient(
        result.parameters,
        gradient,
    )
    stationarity = float(jnp.linalg.norm(projected, ord=jnp.inf))
    feasibility = float(problem.bounds.violation(result.parameters))
    objective_value = float(result.objective)
    certificate = abs(objective_value)
    independently_certified = (
        jnp.isfinite(result.objective)
        & (certificate <= termination.absolute_optimality)
        & (feasibility <= termination.absolute_optimality)
    )
    return _RawObservation(
        "global",
        case_id,
        implementation,
        True,
        bool(independently_certified),
        str(int(result.status)),
        float(result.diagnostics.objective_evaluations),
        elapsed,
        certificate,
        objective=objective_value,
        feasibility=feasibility,
        backend_claimed_success=bool(result.successful),
        solution=result.parameters,
        work_counts={
            "objective_evaluations": float(result.diagnostics.objective_evaluations),
        },
        certificate_components={
            "objective_gap": certificate,
            "projected_stationarity": stationarity,
            "feasibility": feasibility,
        },
    )


def _run_differentiation(case_id, implementation):
    if implementation != "phydrax-implicit":
        return None
    problem = phx.nonlinear.NonlinearSystemProblem(lambda x, a: x * x - a)
    state = jnp.asarray([2.0])
    start = time.perf_counter()
    if case_id == "root-first-order":
        result = phx.nonlinear.root_solution_jvp(
            problem, state, jnp.asarray([4.0]), jnp.asarray([1.0])
        )
        expected = jnp.asarray([0.25])
    elif case_id == "root-second-order":
        result = phx.nonlinear.root_solution_second_jvp(
            problem, state, jnp.asarray([4.0]), jnp.asarray([1.0])
        )
        expected = jnp.asarray([-0.03125])
    else:
        return None
    jax.block_until_ready(result.value)
    elapsed = time.perf_counter() - start
    error = float(jnp.linalg.norm(result.value - expected))
    return _RawObservation(
        "differentiation",
        case_id,
        implementation,
        True,
        bool(jnp.isfinite(error) and error <= 1e-8),
        str(int(result.evidence.status)),
        1.0,
        elapsed,
        error,
        derivative_error=error,
        backend_claimed_success=bool(result.evidence.successful),
        solution=result.value,
        work_counts={"derivative_evaluations": 1.0},
        certificate_components={
            "derivative_error": error,
            "implicit_residual": float(result.evidence.residual_norm),
        },
    )


def _run_adversarial(case_id, implementation):
    if implementation != "phydrax-robust":
        return None
    start = time.perf_counter()
    if case_id == "permutation":
        problem = phx.nonlinear.NonlinearSystemProblem(lambda x, a: x * x - a)
        permutation = jnp.asarray([2, 0, 1])
        initial = jnp.ones(3)
        args = jnp.asarray([1.0, 4.0, 9.0])
        first = phx.nonlinear.RobustRoot().solve(
            problem,
            initial,
            args=args,
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1e-8,
                relative_residual=0.0,
                maximum_steps=100,
                maximum_evaluations=1000,
                maximum_linear_iterations=5000,
            ),
        )
        second = phx.nonlinear.RobustRoot().solve(
            problem,
            initial[permutation],
            args=args[permutation],
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1e-8,
                relative_residual=0.0,
                maximum_steps=100,
                maximum_evaluations=1000,
                maximum_linear_iterations=5000,
            ),
        )
        inverse = jnp.argsort(permutation)
        error = float(jnp.linalg.norm(first.state - second.state[inverse]))
        backend_success = bool(first.successful & second.successful)
        certified = bool(jnp.isfinite(error) and error <= 1e-8)
        work_counts = {
            "residual_evaluations": float(
                first.diagnostics.residual_evaluations
                + second.diagnostics.residual_evaluations
            ),
        }
    elif case_id == "domain":
        vi = phx.nonlinear.VariationalInequalityProblem(
            lambda x, a: jnp.where(x >= 0.0, x - 1.0, jnp.nan),
            phx.nonlinear.Bounds(0.0, jnp.inf),
        )
        result = phx.nonlinear.SemismoothNewton(feasibility="preserve-box").solve(
            vi, jnp.asarray([-1.0])
        )
        error = float(result.certificate.natural_residual_norm)
        backend_success = bool(result.successful)
        certified = bool(jnp.isfinite(error) and error <= 1e-8)
        work_counts = {"iterations": float(result.diagnostics.iterations)}
    else:
        return None
    elapsed = time.perf_counter() - start
    return _RawObservation(
        "adversarial",
        case_id,
        implementation,
        True,
        certified,
        "0" if backend_success else "1",
        next(iter(work_counts.values())),
        elapsed,
        error,
        backend_claimed_success=backend_success,
        solution=(
            {"error": error}
            if case_id == "domain"
            else {"first": first.state, "second": second.state}
        ),
        work_counts=work_counts,
        certificate_components={"behavioral_error": error},
    )


def _json_solution(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_solution(value.item())
    if isinstance(value, dict):
        return {str(key): _json_solution(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_solution(item) for item in value]
    array = np.asarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": _json_solution(array.tolist()),
    }


def _case_initial_evidence(family: Family, case_id: str):
    if family == "root":
        _, initial, args = _root_cases()[case_id]
    elif family == "least-squares":
        _, initial, args, _ = _least_squares_cases()[case_id]
    elif family == "constrained":
        initial = jnp.asarray([0.5, 0.5])
        args = None
    elif family == "global":
        initial = jnp.full((4,), 0.25)
        args = {
            "bounds": [-5.0, 5.0],
            "dimension": 4,
            "seed": 11,
        }
    elif family == "differentiation":
        initial = jnp.asarray([2.0])
        args = {
            "parameters": [4.0],
            "direction": [1.0],
            "order": 1 if case_id == "root-first-order" else 2,
        }
    else:
        initial = jnp.ones(3) if case_id == "permutation" else jnp.asarray([-1.0])
        args = (
            jnp.asarray([1.0, 4.0, 9.0])
            if case_id == "permutation"
            else {"bounds": [0.0, None]}
        )
    fingerprint = stable_fingerprint(
        {
            "family": family,
            "case_id": case_id,
            "initial": _json_solution(initial),
            "args": _json_solution(args),
        }
    )
    return initial, args, fingerprint


def _runner_payload(family: Family, case_id: str):
    initial, args, fingerprint = _case_initial_evidence(family, case_id)
    return {
        "family": family,
        "case_id": case_id,
        "initial": _json_solution(initial),
        "args": _json_solution(args),
        "initial_fingerprint": fingerprint,
    }


def _primary_work_key(family: Family) -> tuple[str, str]:
    return {
        "root": ("residual_evaluations", "residual-evaluations"),
        "least-squares": ("residual_evaluations", "residual-evaluations"),
        "constrained": ("objective_evaluations", "objective-evaluations"),
        "global": ("objective_evaluations", "objective-evaluations"),
        "differentiation": (
            "derivative_evaluations",
            "derivative-evaluations",
        ),
        "adversarial": ("behavior_evaluations", "behavior-evaluations"),
    }[family]


def _external_raw_observation(
    family: Family,
    case_id: str,
    implementation: str,
    response,
    elapsed: float,
):
    if not response["available"]:
        return _unavailable(
            family,
            case_id,
            implementation,
            reason=response["availability_reason"],
            detail="External runner reported itself unavailable.",
        )
    if not isinstance(response["solution"], list):
        return _unavailable(
            family,
            case_id,
            implementation,
            reason="runner-error",
            detail="External runner solution must be one numeric JSON array.",
        )
    parameters = jnp.asarray(response["solution"], dtype=jnp.float64)
    objective_value = None
    feasibility = 0.0
    components: dict[str, float | None]
    if family == "root":
        function, initial, args = _root_cases()[case_id]
        residual = function(parameters, args)
        certificate = float(jnp.linalg.norm(residual) / (1.0 + jnp.linalg.norm(initial)))
        certified = bool(jnp.all(jnp.isfinite(residual)) & (certificate <= 1e-8))
        components = {"relative_residual": certificate}
    elif family == "least-squares":
        residual_function, _, args, bounds = _least_squares_cases()[case_id]

        def physical_objective(value):
            residual = residual_function(value, args)
            if case_id == "robust-outlier":
                absolute = jnp.abs(residual)
                return jnp.sum(
                    jnp.where(
                        absolute <= 1.0,
                        0.5 * residual * residual,
                        absolute - 0.5,
                    )
                )
            return 0.5 * jnp.real(jnp.vdot(residual, residual))

        objective_array, gradient = jax.value_and_grad(physical_objective)(parameters)
        projected = (
            gradient
            if bounds is None
            else bounds.projected_gradient(parameters, gradient)
        )
        certificate = float(jnp.linalg.norm(projected, ord=jnp.inf))
        feasibility = float(0.0 if bounds is None else bounds.violation(parameters))
        objective_value = float(objective_array)
        certified = bool(
            jnp.isfinite(objective_array)
            & jnp.all(jnp.isfinite(gradient))
            & (certificate <= 1e-6)
            & (feasibility <= 1e-6)
        )
        components = {
            "projected_stationarity": certificate,
            "feasibility": feasibility,
        }
    elif family == "constrained":
        problem = _constrained_cases()[case_id]
        prepared = phx.optim.prepare_constrained_model(problem, parameters)
        evaluation = prepared.evaluate(parameters)
        raw_jacobian = evaluation.constraint_jacobian
        equality_jacobian = raw_jacobian[prepared.equality_indices]
        inequality_jacobian = jnp.concatenate(
            [
                raw_jacobian[prepared.lower_indices],
                -raw_jacobian[prepared.upper_indices],
            ],
            axis=0,
        )
        active = evaluation.inequality_slacks <= 1e-3
        active_jacobian = inequality_jacobian[active]
        multiplier_matrix = jnp.concatenate(
            [
                jnp.conj(equality_jacobian.T),
                -jnp.conj(active_jacobian.T),
            ],
            axis=1,
        )
        if multiplier_matrix.shape[1]:
            multipliers = jnp.linalg.lstsq(
                multiplier_matrix,
                -evaluation.gradient,
                rcond=None,
            )[0]
            stationarity = evaluation.gradient + multiplier_matrix @ multipliers
        else:
            stationarity = evaluation.gradient
        dual = float(jnp.linalg.norm(stationarity, ord=jnp.inf))
        feasibility = float(evaluation.primal_feasibility)
        certificate = max(feasibility, dual)
        objective_value = float(evaluation.objective)
        certified = bool(evaluation.finite & (certificate <= 1e-6))
        components = {
            "stationarity": dual,
            "feasibility": feasibility,
        }
    elif family == "global":
        objective = _global_cases()[case_id]
        objective_value = float(objective(parameters))
        gradient = jax.grad(objective)(parameters)
        certificate = abs(objective_value)
        certified = bool(jnp.isfinite(objective_value) & (certificate <= 1e-5))
        components = {
            "objective_gap": certificate,
            "projected_stationarity": float(jnp.linalg.norm(gradient, ord=jnp.inf)),
            "feasibility": 0.0,
        }
    else:
        return _unavailable(
            family,
            case_id,
            implementation,
            reason="unsupported-case",
            detail="No external certificate evaluator exists for this family.",
        )
    counts = {str(key): float(value) for key, value in response["work_counts"].items()}
    primary_key, _ = _primary_work_key(family)
    backend = response["backend"]
    return _RawObservation(
        family,
        case_id,
        implementation,
        True,
        certified,
        str(backend.get("status_code", "unknown")),
        counts.get(primary_key),
        elapsed,
        certificate,
        objective=objective_value,
        feasibility=feasibility,
        backend_claimed_success=backend.get("claimed_success"),
        expected_identity=response["observed_identity"],
        observed_identity=response["observed_identity"],
        source_revision=response["source_revision"],
        work_counts=counts,
        certificate_components=components,
        solution=parameters,
    )


def _execute_raw_once(
    family: Family,
    case_id: str,
    implementation: str,
    runner: Callable[[str, str], _RawObservation | None],
) -> _RawObservation:
    spec = _peer_spec(implementation)
    payload = _runner_payload(family, case_id)
    fingerprint = payload["initial_fingerprint"]
    if spec is None:
        raw = runner(case_id, implementation)
        if raw is None:
            return _unavailable(
                family,
                case_id,
                implementation,
                reason="unsupported-case",
                detail="Implementation does not support this canonical case.",
            )
        return replace(
            raw,
            expected_identity="phydrax-native",
            observed_identity="phydrax-native",
            source_revision="working-tree",
        )
    request = make_runner_request(
        spec,
        family,
        case_id,
        implementation,
        fingerprint,
        payload,
    )
    if spec.runner_kind == "external-process":
        start = time.perf_counter()
        invocation = run_external_peer(request, spec)
        elapsed = time.perf_counter() - start
        if invocation.response is None:
            return _unavailable(
                family,
                case_id,
                implementation,
                reason=invocation.reason or "runner-error",
                detail=invocation.detail,
            )
        return _external_raw_observation(
            family,
            case_id,
            implementation,
            invocation.response,
            elapsed,
        )
    holder: list[_RawObservation | None] = []

    def callback():
        raw = runner(case_id, implementation)
        holder.append(raw)
        if raw is None:
            return {
                "backend": {
                    "attempted": False,
                    "completed": False,
                    "claimed_success": None,
                    "status_code": None,
                },
                "solution": None,
                "work_counts": {},
            }
        return {
            "backend": {
                "attempted": raw.available,
                "completed": raw.available,
                "claimed_success": raw.backend_claimed_success,
                "status_code": raw.status,
            },
            "solution": _json_solution(raw.solution),
            "work_counts": raw.work_counts or {},
        }

    invocation = run_python_peer(request, spec, callback)
    if invocation.response is None:
        return _unavailable(
            family,
            case_id,
            implementation,
            reason=invocation.reason or "runner-error",
            detail=invocation.detail,
        )
    raw = holder[0]
    if raw is None:
        return _unavailable(
            family,
            case_id,
            implementation,
            reason="unsupported-case",
            detail="Python peer does not support this canonical case.",
        )
    return replace(
        raw,
        expected_identity=spec.expected_identity,
        observed_identity=invocation.observed_identity,
        source_revision=spec.source_revision,
    )


def _finite_or_none(value: float | None):
    if value is None:
        return None
    value_ = float(value)
    return value_ if math.isfinite(value_) else None


def _certificate_contract(family: Family):
    return {
        "root": ("scaled-root-residual", "equation", 1e-8),
        "least-squares": (
            "projected-normal-stationarity",
            "local",
            1e-6,
        ),
        "constrained": ("active-kkt-residual", "local", 1e-6),
        "global": ("known-global-target-gap", "global", 1e-5),
        "differentiation": (
            "analytic-derivative-error",
            "derivative",
            1e-8,
        ),
        "adversarial": (
            "metamorphic-behavior-error",
            "behavioral",
            1e-8,
        ),
    }[family]


def _backend_claim_scope(family: Family):
    return {
        "root": "equation",
        "least-squares": "local",
        "constrained": "local",
        "global": "algorithm",
        "differentiation": "derivative",
        "adversarial": "behavioral",
    }[family]


def _to_observation(
    raw: _RawObservation,
    initial_fingerprint: str,
    *,
    cold_seconds: float | None,
    warmup_seconds: tuple[float, ...],
    steady_seconds: tuple[float, ...],
) -> CampaignObservation:
    spec = _peer_spec(raw.implementation)
    source_revision = (
        raw.source_revision
        if raw.source_revision is not None
        else (None if spec is None else spec.source_revision)
    )
    if not raw.available:
        return CampaignObservation(
            family=raw.family,
            case_id=raw.case_id,
            implementation=raw.implementation,
            available=False,
            availability_reason=raw.availability_reason or "runtime-missing",
            availability_detail=raw.availability_detail,
            expected_identity=raw.expected_identity,
            observed_identity=raw.observed_identity,
            source_revision=source_revision,
            initial_fingerprint=initial_fingerprint,
            result_fingerprint=None,
            backend_success=None,
            backend_scope="unavailable",
            backend_status=None,
            certified=None,
            certificate_kind="unavailable",
            certificate_scope="unavailable",
            certificate_value=None,
            certificate_tolerance=None,
            certificate_components={},
            work=None,
            work_unit=None,
            work_counts={},
            cold_seconds=None,
            warmup_seconds=(),
            steady_seconds=(),
        )
    serialized_solution = _json_solution(raw.solution)
    result_id = stable_fingerprint({"result": serialized_solution})
    certificate_kind, certificate_scope, tolerance = _certificate_contract(raw.family)
    components = {
        str(key): _finite_or_none(value)
        for key, value in (raw.certificate_components or {}).items()
    }
    if not components:
        components = {"certificate_value": _finite_or_none(raw.certificate)}
        if raw.feasibility is not None:
            components["feasibility"] = _finite_or_none(raw.feasibility)
    primary_key, primary_unit = _primary_work_key(raw.family)
    counts = {
        str(key): float(value)
        for key, value in (raw.work_counts or {}).items()
        if math.isfinite(float(value))
    }
    return CampaignObservation(
        family=raw.family,
        case_id=raw.case_id,
        implementation=raw.implementation,
        available=True,
        availability_reason="available",
        availability_detail=None,
        expected_identity=raw.expected_identity or "phydrax-native",
        observed_identity=(
            raw.observed_identity or raw.expected_identity or "phydrax-native"
        ),
        source_revision=source_revision,
        initial_fingerprint=initial_fingerprint,
        result_fingerprint=result_id,
        backend_success=raw.backend_claimed_success,
        backend_scope=_backend_claim_scope(raw.family),
        backend_status=raw.status,
        certified=bool(raw.certified),
        certificate_kind=certificate_kind,
        certificate_scope=certificate_scope,
        certificate_value=_finite_or_none(raw.certificate),
        certificate_tolerance=tolerance,
        certificate_components=components,
        work=counts.get(primary_key),
        work_unit=primary_unit,
        work_counts=counts,
        cold_seconds=cold_seconds,
        warmup_seconds=warmup_seconds,
        steady_seconds=steady_seconds,
        objective=_finite_or_none(raw.objective),
        feasibility=_finite_or_none(raw.feasibility),
        derivative_error=_finite_or_none(raw.derivative_error),
    )


def _timed_observation(
    family: Family,
    case_id: str,
    implementation: str,
    runner: Callable[[str, str], _RawObservation | None],
    *,
    warmup: int,
    repeats: int,
) -> CampaignObservation:
    _, _, initial_fingerprint = _case_initial_evidence(family, case_id)
    cold = _execute_raw_once(family, case_id, implementation, runner)
    if not cold.available:
        return _to_observation(
            cold,
            initial_fingerprint,
            cold_seconds=None,
            warmup_seconds=(),
            steady_seconds=(),
        )
    warm_rows = [
        _execute_raw_once(family, case_id, implementation, runner) for _ in range(warmup)
    ]
    steady_rows = [
        _execute_raw_once(family, case_id, implementation, runner) for _ in range(repeats)
    ]
    failed = next(
        (row for row in (*warm_rows, *steady_rows) if not row.available),
        None,
    )
    if failed is not None:
        return _to_observation(
            failed,
            initial_fingerprint,
            cold_seconds=None,
            warmup_seconds=(),
            steady_seconds=(),
        )
    return _to_observation(
        steady_rows[-1],
        initial_fingerprint,
        cold_seconds=cold.steady_seconds,
        warmup_seconds=tuple(float(row.steady_seconds) for row in warm_rows),
        steady_seconds=tuple(float(row.steady_seconds) for row in steady_rows),
    )


def run_campaign(
    family: Family,
    /,
    *,
    warmup: int = 1,
    repeats: int = 3,
) -> list[CampaignObservation]:
    warmup_ = int(warmup)
    repeats_ = int(repeats)
    if warmup_ < 0 or repeats_ < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive.")
    runners: dict[
        str,
        tuple[
            list[str],
            list[str],
            Callable[[str, str], _RawObservation | None],
        ],
    ] = {
        "root": (
            list(_root_cases()),
            [
                "phydrax-newton",
                "phydrax-robust",
                "phydrax-broyden",
                "phydrax-dfsane",
                "optimistix-newton",
                "scipy-root",
                "nonlinearsolve-jl",
            ],
            _run_root,
        ),
        "least-squares": (
            list(_least_squares_cases()),
            [
                "phydrax-lm",
                "phydrax-dogleg",
                "phydrax-pounders",
                "scipy-least-squares",
                "nonlinearsolve-jl",
                "ceres",
                "theseus",
            ],
            _run_least_squares,
        ),
        "constrained": (
            list(_constrained_cases()),
            [
                "phydrax-sqp",
                "phydrax-ipm",
                "scipy-trust-constr",
                "ipopt",
            ],
            _run_constrained,
        ),
        "global": (
            list(_global_cases()),
            [
                "phydrax-multistart",
                "phydrax-bobyqa",
                "scipy-differential-evolution",
                "nlopt",
            ],
            _run_global,
        ),
        "differentiation": (
            ["root-first-order", "root-second-order"],
            ["phydrax-implicit"],
            _run_differentiation,
        ),
        "adversarial": (
            ["permutation", "domain"],
            ["phydrax-robust"],
            _run_adversarial,
        ),
    }
    cases, implementations, runner = runners[family]
    return [
        _timed_observation(
            family,
            case_id,
            implementation,
            runner,
            warmup=warmup_,
            repeats=repeats_,
        )
        for case_id in cases
        for implementation in implementations
    ]


def performance_profile(
    observations: list[CampaignObservation],
    /,
    *,
    taus: tuple[float, ...] = (1.0, 1.5, 2.0, 4.0, 10.0),
    metric: Literal["primary-work", "steady-solve"] = "primary-work",
) -> list[PerformanceProfilePoint]:
    if metric not in ("primary-work", "steady-solve"):
        raise ValueError("metric must be 'primary-work' or 'steady-solve'.")
    tau_values = tuple(float(tau) for tau in taus)
    if any(not math.isfinite(tau) or tau < 1.0 for tau in tau_values):
        raise ValueError("Performance-profile taus must be finite and at least one.")
    profile: list[PerformanceProfilePoint] = []
    families = sorted({row.family for row in observations})
    for family in families:
        family_rows = [row for row in observations if row.family == family]
        units: tuple[str | None, ...]
        if metric == "primary-work":
            units = tuple(
                sorted(
                    {
                        row.work_unit
                        for row in family_rows
                        if row.available
                        and row.work is not None
                        and row.work_unit is not None
                    }
                )
            )
        else:
            units = (None,)
        for unit in units:
            comparable = [
                row
                for row in family_rows
                if row.available and (metric == "steady-solve" or row.work_unit == unit)
            ]
            implementations = sorted({row.implementation for row in comparable})
            ratios: dict[tuple[str, str], float] = {}
            eligible_cases: list[str] = []
            for case_id in sorted({row.case_id for row in family_rows}):
                rows = [row for row in comparable if row.case_id == case_id]
                values = {
                    row.implementation: (
                        row.work if metric == "primary-work" else row.steady_median
                    )
                    for row in rows
                }
                certified_values = [
                    float(values[row.implementation])
                    for row in rows
                    if row.certified is True and values[row.implementation] is not None
                ]
                if len(rows) < 2 or not certified_values:
                    continue
                eligible_cases.append(case_id)
                best = min(certified_values)
                for row in rows:
                    value = values[row.implementation]
                    if row.certified is True and value is not None:
                        ratios[(case_id, row.implementation)] = float(value) / max(
                            best, 1e-30
                        )
            if not eligible_cases:
                continue
            for implementation in implementations:
                certified_cases = sum(
                    (case_id, implementation) in ratios for case_id in eligible_cases
                )
                for tau in tau_values:
                    fraction = sum(
                        ratios.get((case_id, implementation), math.inf) <= tau
                        for case_id in eligible_cases
                    ) / len(eligible_cases)
                    profile.append(
                        PerformanceProfilePoint(
                            family,
                            metric,
                            unit,
                            tau,
                            implementation,
                            fraction,
                            len(eligible_cases),
                            certified_cases,
                        )
                    )
    return profile


def superiority_audit(
    observations: list[CampaignObservation],
    /,
) -> dict[str, Any]:
    families = sorted({row.family for row in observations})
    portfolio_coverage = {}
    for family in families:
        cases = {row.case_id for row in observations if row.family == family}
        covered = {
            row.case_id
            for row in observations
            if row.family == family
            and row.implementation.startswith("phydrax-")
            and row.certified is True
            and (family != "global" or row.certificate_scope == "global")
        }
        portfolio_coverage[family] = {
            "certified_cases": len(covered),
            "total_cases": len(cases),
            "complete": covered == cases,
        }
    false_successes = [
        {
            "family": row.family,
            "case_id": row.case_id,
            "implementation": row.implementation,
            "backend_status": row.backend_status,
            "backend_scope": row.backend_scope,
            "certificate_kind": row.certificate_kind,
            "certificate_value": row.certificate_value,
        }
        for row in observations
        if row.available
        and row.backend_success is True
        and row.backend_scope == row.certificate_scope
        and row.certified is False
    ]
    backend_false_negatives = [
        {
            "family": row.family,
            "case_id": row.case_id,
            "implementation": row.implementation,
            "backend_status": row.backend_status,
            "backend_scope": row.backend_scope,
        }
        for row in observations
        if row.available
        and row.backend_success is False
        and row.backend_scope == row.certificate_scope
        and row.certified is True
    ]
    peer_ids = sorted(
        {
            row.implementation
            for row in observations
            if not row.implementation.startswith("phydrax-")
        }
    )
    unavailable_peers = [
        {
            "implementation": implementation,
            "reasons": sorted(
                {
                    row.availability_reason
                    for row in observations
                    if row.implementation == implementation and not row.available
                }
            ),
        }
        for implementation in peer_ids
        if not any(
            row.implementation == implementation and row.available for row in observations
        )
    ]
    coverage_complete = all(value["complete"] for value in portfolio_coverage.values())
    return {
        "portfolio_coverage": portfolio_coverage,
        "false_successes": false_successes,
        "backend_false_negatives": backend_false_negatives,
        "unavailable_peers": unavailable_peers,
        "claim_ready": (
            coverage_complete and not false_successes and not unavailable_peers
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family",
        choices=(
            "root",
            "least-squares",
            "constrained",
            "global",
            "differentiation",
            "adversarial",
            "all",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args(argv)
    families = (
        (
            "root",
            "least-squares",
            "constrained",
            "global",
            "differentiation",
            "adversarial",
        )
        if arguments.family == "all"
        else (arguments.family,)
    )
    observations = [
        observation
        for family in families
        for observation in run_campaign(
            family,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
    ]
    profiles = [
        *performance_profile(observations, metric="primary-work"),
        *performance_profile(observations, metric="steady-solve"),
    ]
    manifest = json.loads(_PEER_MANIFEST_PATH.read_text())
    payload = {
        "campaign": {
            "families": list(families),
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "certificate_source": "independent-physical",
            "timing_policy": "cold-then-warmup-then-steady",
        },
        "observations": [asdict(value) for value in observations],
        "performance_profiles": [asdict(value) for value in profiles],
        "superiority_audit": superiority_audit(observations),
        "peer_manifest": {
            "path": str(_PEER_MANIFEST_PATH),
            "fingerprint": stable_fingerprint(manifest),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
