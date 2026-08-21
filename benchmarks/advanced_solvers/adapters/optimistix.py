#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

from ..problems import NonlinearProblem, OptimizationProblem
from ._availability import import_module, probe_modules
from .base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    SolveResult,
    TransferEvidence,
)


_CAPABILITIES = frozenset({"nonlinear.root", "optimization.unconstrained"})


@dataclass
class _OptimistixState:
    spec: CaseSpec
    initial: Any
    target: Any
    executable: Any = None
    differentiation_executable: Any = None
    host_to_device_bytes: int = 0


class OptimistixAdapter(BenchmarkAdapter):
    """JIT-compiled Optimistix Newton and BFGS baselines."""

    name = "optimistix"
    dependency = "optimistix+lineax+jax"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("jax", "lineax", "optimistix"),
            distribution="optimistix",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        supported = spec.capability in self.capabilities
        if spec.capability == "optimization.unconstrained":
            method = "optimistix-bfgs"
            preconditioner = "inverse-bfgs-update"
        elif supported:
            methods = {
                "default": ("optimistix-newton", "lineax-auto-linear-solver"),
                "dense": ("optimistix-newton+dense-lu", "none"),
                "matrix-free": (
                    "optimistix-newton+matrix-free-gmres",
                    "identity",
                ),
                "sparse": (
                    "optimistix-newton+matrix-free-gmres-reference",
                    "identity",
                ),
            }
            method, preconditioner = methods[spec.solver_mode]
        else:
            method, preconditioner = "unsupported", "none"
        return Implementation(
            adapter=self.name,
            backend="optimistix-jax-default-device",
            method=method,
            preconditioner=preconditioner,
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _OptimistixState:
        jax = import_module("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = import_module("jax.numpy")
        problem = spec.problem
        if isinstance(problem, OptimizationProblem):
            if problem.variant != "unconstrained":
                raise TypeError(
                    f"Optimistix adapter does not implement {spec.capability!r}"
                )
            return _OptimistixState(
                spec=spec,
                initial=jnp.asarray(problem.initial),
                target=None,
                host_to_device_bytes=int(problem.initial.nbytes),
            )
        if not isinstance(problem, NonlinearProblem) or problem.variant != "root":
            raise TypeError(
                f"Optimistix adapter does not implement {spec.capability!r}; "
                "VI bounds are not equivalent to an unconstrained root contract"
            )
        return _OptimistixState(
            spec=spec,
            initial=jnp.asarray(problem.initial),
            target=jnp.asarray(problem.target),
            host_to_device_bytes=int(problem.initial.nbytes + problem.target.nbytes),
        )

    def compilation_applicable(self, setup_state: _OptimistixState, /) -> bool:
        return True

    def compile(self, setup_state: _OptimistixState, /) -> _OptimistixState:
        jax = import_module("jax")
        optx = import_module("optimistix")
        jnp = import_module("jax.numpy")
        tolerance = setup_state.spec.tolerances
        if isinstance(setup_state.spec.problem, OptimizationProblem):
            solver = optx.BFGS(
                rtol=tolerance.relative,
                atol=tolerance.absolute,
            )

            def solve_minimum(initial: Any) -> tuple[Any, Any, Any]:
                solution = optx.minimise(
                    lambda value, args: jnp.sum(
                        100.0 * (value[1:] - value[:-1] ** 2) ** 2
                        + (1.0 - value[:-1]) ** 2
                    ),
                    solver,
                    initial,
                    max_steps=tolerance.max_steps,
                    throw=False,
                )
                return (
                    solution.value,
                    solution.result == optx.RESULTS.successful,
                    solution.stats["num_steps"],
                )

            setup_state.executable = (
                jax.jit(solve_minimum).lower(setup_state.initial).compile()
            )
        else:
            lx = import_module("lineax")
            solver = _root_solver(setup_state, optx, lx)
            problem = setup_state.spec.problem

            def solve_root(initial: Any, target: Any) -> tuple[Any, Any, Any]:
                solution = optx.root_find(
                    lambda value, expected: _jax_root_residual(
                        problem, value, expected, jnp
                    ),
                    solver,
                    initial,
                    args=target,
                    max_steps=tolerance.max_steps,
                    throw=False,
                )
                return (
                    solution.value,
                    solution.result == optx.RESULTS.successful,
                    solution.stats["num_steps"],
                )

            setup_state.executable = (
                jax.jit(solve_root)
                .lower(
                    setup_state.initial,
                    setup_state.target,
                )
                .compile()
            )
        return setup_state

    def solve(self, prepared_state: _OptimistixState, /) -> SolveResult:
        if isinstance(prepared_state.spec.problem, OptimizationProblem):
            solution, successful, steps = prepared_state.executable(
                prepared_state.initial
            )
        else:
            solution, successful, steps = prepared_state.executable(
                prepared_state.initial,
                prepared_state.target,
            )
        return SolveResult(
            solution=solution,
            auxiliary={"successful": successful},
            converged=successful,
            message="Optimistix solve completed; status is in auxiliary evidence",
            operations={
                "iterations": steps,
                "matvecs": None,
                "preconditioner_applications": None,
                "linear_solves": steps,
                "nonlinear_evaluations": None,
                "jacobian_evaluations": None,
            },
        )

    def differentiation_applicable(self, prepared_state: _OptimistixState, /) -> bool:
        problem = prepared_state.spec.problem
        return (
            isinstance(problem, NonlinearProblem)
            and problem.variant == "root"
            and problem.root_kind == "separable"
        )

    def compile_differentiation(
        self,
        prepared_state: _OptimistixState,
        /,
    ) -> _OptimistixState:
        jax = import_module("jax")
        jnp = import_module("jax.numpy")
        lx = import_module("lineax")
        optx = import_module("optimistix")
        tolerance = prepared_state.spec.tolerances
        solver = _root_solver(prepared_state, optx, lx)
        problem = prepared_state.spec.problem

        def solve_for_target(target):
            solution = optx.root_find(
                lambda value, expected: _jax_root_residual(problem, value, expected, jnp),
                solver,
                prepared_state.initial,
                args=target,
                max_steps=tolerance.max_steps,
                throw=False,
            )
            return jnp.asarray(solution.value)

        derivative = jax.jacrev(solve_for_target)
        prepared_state.differentiation_executable = (
            jax.jit(derivative).lower(prepared_state.target).compile()
        )
        return prepared_state

    def differentiate(self, prepared_state: _OptimistixState, /) -> Any:
        if prepared_state.differentiation_executable is None:
            raise ValueError(
                "Optimistix differentiation must be compiled before execution."
            )
        return prepared_state.differentiation_executable(prepared_state.target)

    def memory(
        self,
        prepared_state: _OptimistixState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        del result
        input_bytes = int(prepared_state.initial.nbytes)
        if prepared_state.target is not None:
            input_bytes += int(prepared_state.target.nbytes)
        return {
            "matrix_bytes": input_bytes,
            "setup_bytes": 0,
            "peak_estimate_bytes": None,
            "evidence": (
                "exact retained device input bytes; Optimistix/Lineax/XLA transient "
                "workspace peak is unavailable"
            ),
        }

    def transfers(
        self,
        prepared_state: _OptimistixState,
        result: SolveResult,
        /,
        *,
        device_to_host_bytes: int,
    ) -> TransferEvidence:
        del result
        return TransferEvidence(
            input_origin="numpy-host",
            host_to_device_bytes=prepared_state.host_to_device_bytes,
            host_to_device_timing_phase="setup",
            device_to_host_bytes=device_to_host_bytes,
            device_to_host_timing_phase="verification",
            evidence=(
                "canonical NumPy initial state and target were converted during setup; "
                "Optimistix solution, result, and iteration arrays were materialized "
                "by jax.device_get during verification"
            ),
        )


def _root_solver(state: _OptimistixState, optx, lx):
    tolerance = state.spec.tolerances
    if state.spec.solver_mode == "default":
        return optx.Newton(
            rtol=tolerance.relative,
            atol=tolerance.absolute,
        )
    if state.spec.solver_mode == "dense":
        linear_solver = lx.LU()
    else:
        restart = min(16, int(state.initial.size))
        linear_max_steps = tolerance.max_steps
        if state.spec.solver_mode == "sparse":
            linear_max_steps = max(linear_max_steps, int(state.initial.size))
        linear_solver = lx.GMRES(
            rtol=tolerance.relative,
            atol=tolerance.absolute,
            norm=optx.two_norm,
            max_steps=max(1, (linear_max_steps + restart - 1) // restart),
            restart=restart,
        )
    return optx.Newton(
        rtol=tolerance.relative,
        atol=tolerance.absolute,
        norm=optx.two_norm,
        linear_solver=linear_solver,
    )


def _jax_root_residual(problem, value, target, jnp):
    if problem.root_kind == "separable":
        return value * value - target
    if problem.grid_spacing is None:
        raise ValueError("semilinear Poisson problem is missing grid spacing")
    extended = jnp.pad(value, (1, 1))
    laplacian = (2.0 * value - extended[:-2] - extended[2:]) / problem.grid_spacing**2
    return laplacian + problem.nonlinearity * value**3 - target


def _version_evidence() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("optimistix", "lineax", "jax"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


__all__ = ["OptimistixAdapter"]
