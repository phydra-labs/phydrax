#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..problems import (
    ContinuationProblem,
    GeneralEigenProblem,
    MathematicalProgramProblem,
    NonlinearProblem,
    OptimizationProblem,
    SparseLinearProblem,
)
from ._availability import import_module, unsupported
from .base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    RefreshEvidence,
    SolveResult,
    TransferEvidence,
)


_CAPABILITIES = frozenset(
    {
        "linear.scalar",
        "linear.block",
        "nonlinear.root",
        "nonlinear.vi",
        "eigen.general",
        "continuation.fold",
        "optimization.unconstrained",
        "optimization.constrained",
        "optimization.proximal",
        "optimization.bounded-least-squares",
        "optimization.linear-program",
        "optimization.quadratic-program",
    }
)


@dataclass
class _PhydraxState:
    spec: CaseSpec
    phx: Any
    native_problem: Any
    policy: Any = None
    method: Any = None
    rhs: Any = None
    target: Any = None
    plan: Any = None
    prepared: Any = None
    executable: Any = None
    differentiation_executable: Any = None
    initial_coordinate: Any = None
    refreshed_certificate_problem: Any = None
    host_to_device_bytes: int = 0


class PhydraxAdapter(BenchmarkAdapter):
    """Public Phydrax paths for every common advanced-solver problem family."""

    name = "phydrax"
    dependency = "phydrax+jax"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        if capability not in self.capabilities:
            return unsupported(
                adapter=self.name,
                dependency=self.dependency,
                capability=capability,
            )
        if (
            capability == "optimization.conic-program"
            and importlib.util.find_spec("clarabel") is None
        ):
            return Availability(
                available=False,
                capability=capability,
                dependency="phydrax[clarabel]",
                dependency_version=None,
                reason="Clarabel is required for the selected Phydrax conic method",
            )
        required_module, required_names = _required_public_api(capability)
        try:
            module = importlib.import_module(required_module)
        except ModuleNotFoundError as error:
            missing = error.name or required_module
            return Availability(
                available=False,
                capability=capability,
                dependency=self.dependency,
                dependency_version=None,
                reason=f"required public module {missing!r} is not installed",
            )
        except ImportError as error:
            return Availability(
                available=False,
                capability=capability,
                dependency=self.dependency,
                dependency_version=None,
                reason=(
                    f"required public module {required_module!r} could not be imported: "
                    f"{type(error).__name__}: {error}"
                ),
            )
        missing_names = sorted(required_names - module.__dict__.keys())
        if missing_names:
            return Availability(
                available=False,
                capability=capability,
                dependency=self.dependency,
                dependency_version=_phydrax_version(),
                reason=(
                    f"public module {required_module!r} lacks required capability symbols: "
                    f"{', '.join(missing_names)}"
                ),
            )
        return Availability(
            available=True,
            capability=capability,
            dependency=self.dependency,
            dependency_version=_phydrax_version(),
            reason=None,
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        capability = spec.capability
        if capability == "linear.scalar":
            method, preconditioner = "conjugate-gradient", "jacobi"
        elif capability == "linear.block":
            method, preconditioner = "block-conjugate-gradient", "jacobi"
        elif capability == "nonlinear.root":
            methods = {
                "default": ("phydrax-root-auto", "policy-selected-linear-solver"),
                "dense": ("newton+dense-lu", "none"),
                "matrix-free": ("newton+matrix-free-gmres", "identity"),
                "sparse": ("newton+sparse-pcg", "jacobi"),
            }
            method, preconditioner = methods[spec.solver_mode]
        elif capability == "nonlinear.vi":
            method, preconditioner = (
                "semismooth-newton-fischer-burmeister-preserve-box",
                "policy-selected-linear-solver",
            )
        elif capability == "eigen.general":
            method, preconditioner = "restarted-arnoldi-largest-magnitude", "none"
        elif capability == "optimization.unconstrained":
            method, preconditioner = "newton-trust-region", "dense-hessian"
        elif capability == "optimization.constrained":
            method, preconditioner = "sqp-merit", "dense-bfgs-qp"
        elif capability == "optimization.proximal":
            method, preconditioner = "proximal-gradient", "exact-l1-proximal-map"
        elif capability == "optimization.bounded-least-squares":
            method, preconditioner = (
                "bounded-levenberg-marquardt",
                "active-set-trust-region",
            )
        elif capability == "optimization.linear-program":
            method, preconditioner = "dense-primal-dual-lp", "none"
        elif capability == "optimization.quadratic-program":
            method, preconditioner = "dense-primal-dual-qp", "none"
        elif capability == "optimization.conic-program":
            method, preconditioner = "unsupported-native-conic", "none"
        else:
            method, preconditioner = "pseudo-arclength", "newton-krylov-corrector"
        return Implementation(
            adapter=self.name,
            backend="phydrax-public-jax",
            method=method,
            preconditioner=preconditioner,
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _PhydraxState:
        phx = import_module("phydrax")
        jnp = import_module("jax.numpy")
        problem = spec.problem
        if isinstance(problem, SparseLinearProblem):
            properties = phx.linalg.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                    "positive_semidefinite": "construction",
                },
            )
            relation = phx.sparse.EdgeRelation(
                jnp.asarray(problem.columns),
                jnp.asarray(problem.rows),
                source_size=problem.dimension,
                target_size=problem.dimension,
            )
            space = phx.linalg.ArraySpace((problem.dimension,), dtype=jnp.float64)
            operator = phx.sparse.SparseCoordinateOperator(
                relation,
                jnp.asarray(problem.coefficients),
                source=space,
                target=space,
                properties=properties,
                operator_id=f"benchmark:{problem.identity()['fingerprint']}",
            )
            native_problem = phx.linalg.LinearSystem(
                operator,
                problem_id=f"benchmark-system:{problem.identity()['fingerprint']}",
            )
            method = (
                phx.linalg.ConjugateGradient()
                if problem.block_size == 1
                else phx.linalg.BlockCG()
            )
            policy = phx.linalg.LinearSolvePolicy(
                method,
                tolerance=phx.linalg.TolerancePolicy(
                    relative=spec.tolerances.relative,
                    absolute=spec.tolerances.absolute,
                    max_steps=spec.tolerances.max_steps,
                ),
                preconditioning=phx.linalg.PreconditioningPolicy(
                    phx.linalg.JacobiPreconditionerBuilder(relaxation=1.0)
                ),
                differentiation=phx.linalg.DifferentiationPolicy("none"),
                failure=phx.linalg.FailurePolicy("status"),
            )
            return _PhydraxState(
                spec=spec,
                phx=phx,
                native_problem=native_problem,
                policy=policy,
                rhs=jnp.asarray(problem.rhs),
                host_to_device_bytes=int(
                    problem.rows.nbytes
                    + problem.columns.nbytes
                    + problem.coefficients.nbytes
                    + problem.rhs.nbytes
                ),
            )
        if isinstance(problem, GeneralEigenProblem):
            operator = phx.linalg.DenseLinearOperator(
                jnp.asarray(problem.matrix),
                operator_id=f"benchmark:{problem.identity()['fingerprint']}",
            )
            native_problem = phx.linalg.eigen.GeneralEigenproblem(
                operator,
                problem_id=f"benchmark-general-eigen:{problem.identity()['fingerprint']}",
            )
            subspace_dimension = min(
                problem.matrix.shape[0],
                max(problem.eigenpairs + 2, 2 * problem.eigenpairs),
            )
            policy = phx.linalg.eigen.GeneralEigenSolvePolicy(
                phx.linalg.eigen.RestartedArnoldi(
                    subspace_dimension=subspace_dimension,
                ),
                transform=phx.linalg.eigen.StandardTransform(),
                selection=phx.linalg.eigen.GeneralEigenSelection(
                    "largest-magnitude",
                    count=problem.eigenpairs,
                ),
                max_steps=spec.tolerances.max_steps,
                tolerance=phx.linalg.eigen.GeneralEigenTolerancePolicy(
                    relative=spec.tolerances.relative,
                    absolute=spec.tolerances.absolute,
                ),
                initial_vector=jnp.linspace(
                    1.0,
                    2.0,
                    problem.matrix.shape[0],
                    dtype=jnp.float64,
                ),
                failure=phx.linalg.FailurePolicy("status"),
            )
            return _PhydraxState(
                spec=spec,
                phx=phx,
                native_problem=native_problem,
                policy=policy,
                host_to_device_bytes=int(problem.matrix.nbytes),
            )
        if isinstance(problem, NonlinearProblem):
            nonlinear = import_module("phydrax.nonlinear")
            target = jnp.asarray(problem.target)
            initial = jnp.asarray(problem.initial)
            transferred_bytes = int(problem.initial.nbytes + problem.target.nbytes)
            termination = nonlinear.NonlinearTermination(
                absolute_residual=spec.tolerances.absolute,
                relative_residual=spec.tolerances.relative,
                maximum_steps=spec.tolerances.max_steps,
                maximum_evaluations=max(2 * spec.tolerances.max_steps, 2),
            )
            method = None
            if problem.variant == "root":
                residual = lambda value, expected: _jax_root_residual(
                    problem, value, expected, jnp
                )
                native_problem = nonlinear.NonlinearSystemProblem(
                    residual,
                    problem_id=f"benchmark-root:{problem.identity()['fingerprint']}",
                )
                policy = termination
                if spec.solver_mode != "default":
                    linear_max_steps = spec.tolerances.max_steps
                    linear_method = phx.linalg.DenseLU()
                    preconditioning = None
                    jacobian_policy = nonlinear.JacobianPolicy()
                    if spec.solver_mode == "matrix-free":
                        linear_method = phx.linalg.GMRES(restart=16)
                    elif spec.solver_mode == "sparse":
                        if problem.grid_spacing is None:
                            raise ValueError(
                                "sparse root benchmark requires grid spacing"
                            )
                        dimension = problem.initial.size
                        indices = np.arange(dimension, dtype=np.int64)
                        pattern = phx.sparse.SparsePattern.from_coo(
                            np.concatenate((indices, indices[:-1], indices[1:])),
                            np.concatenate((indices, indices[1:], indices[:-1])),
                            (dimension, dimension),
                            symmetric=True,
                        )
                        properties = phx.linalg.OperatorProperties(
                            self_adjoint=True,
                            positive_definite=True,
                            evidence={
                                "self_adjoint": "construction",
                                "positive_definite": "construction",
                                "positive_semidefinite": "construction",
                            },
                        )
                        space = phx.linalg.PyTreeSpace(initial)
                        sparse_plan = phx.sparse.compile_sparse_jacobian(
                            residual,
                            initial,
                            source=space,
                            target=space,
                            sample_args=target,
                            structure=pattern,
                            compiler="native",
                            symmetric=True,
                            properties=properties,
                            plan_id=(
                                "benchmark-sparse-root:"
                                f"{problem.identity()['fingerprint']}"
                            ),
                        )
                        jacobian_policy = nonlinear.JacobianPolicy(
                            "sparse", sparse_plan=sparse_plan
                        )
                        linear_method = phx.linalg.PCG()
                        linear_max_steps = max(linear_max_steps, dimension)
                        preconditioning = phx.linalg.PreconditioningPolicy(
                            phx.linalg.JacobiPreconditionerBuilder(relaxation=1.0)
                        )
                    linear_policy = phx.linalg.LinearSolvePolicy(
                        linear_method,
                        tolerance=phx.linalg.TolerancePolicy(
                            relative=spec.tolerances.relative,
                            absolute=spec.tolerances.absolute,
                            max_steps=linear_max_steps,
                        ),
                        preconditioning=preconditioning,
                        failure=phx.linalg.FailurePolicy("status"),
                    )
                    forcing_value = min(
                        max(spec.tolerances.relative, 1e-12),
                        0.5,
                    )
                    forcing_policy = (
                        nonlinear.NewtonForcingPolicy()
                        if spec.solver_mode == "sparse"
                        else nonlinear.NewtonForcingPolicy(
                            "constant",
                            initial=forcing_value,
                            minimum=forcing_value,
                            maximum=forcing_value,
                        )
                    )
                    method = nonlinear.NewtonKrylov(
                        jacobian_policy=jacobian_policy,
                        linear_policy=linear_policy,
                        forcing_policy=forcing_policy,
                    )
            else:
                if (
                    problem.lower is None
                    or problem.upper is None
                    or problem.diagonal is None
                ):
                    raise ValueError("VI problem lacks bounds or its diagonal operator")
                lower = jnp.asarray(problem.lower)
                upper = jnp.asarray(problem.upper)
                diagonal = jnp.asarray(problem.diagonal)
                transferred_bytes += int(
                    problem.lower.nbytes + problem.upper.nbytes + problem.diagonal.nbytes
                )
                bounds = nonlinear.Bounds(lower, upper)
                native_problem = nonlinear.VariationalInequalityProblem(
                    lambda value, args: diagonal * value - target,
                    bounds,
                    problem_id=f"benchmark-vi:{problem.identity()['fingerprint']}",
                )
                policy = (
                    nonlinear.SemismoothNewton(
                        feasibility="preserve-box",
                        certification_tolerance=max(
                            spec.tolerances.relative,
                            spec.tolerances.absolute,
                        ),
                    ),
                    termination,
                )
            return _PhydraxState(
                spec=spec,
                phx=phx,
                native_problem=native_problem,
                policy=policy,
                method=method,
                rhs=initial,
                target=target,
                host_to_device_bytes=transferred_bytes,
            )
        if isinstance(problem, OptimizationProblem):
            optim = import_module("phydrax.optim")
            initial = jnp.asarray(problem.initial)
            termination = optim.OptimizationTermination(
                absolute_optimality=spec.tolerances.absolute,
                relative_optimality=spec.tolerances.relative,
                absolute_step=spec.tolerances.absolute,
                relative_step=spec.tolerances.relative,
                maximum_steps=spec.tolerances.max_steps,
                maximum_evaluations=max(8 * spec.tolerances.max_steps, 8),
            )
            if problem.variant == "unconstrained":
                native_problem = optim.MinimizationProblem(
                    lambda value, args: jnp.sum(
                        100.0 * (value[1:] - value[:-1] ** 2) ** 2
                        + (1.0 - value[:-1]) ** 2
                    ),
                    problem_id=f"benchmark-rosenbrock:{problem.identity()['fingerprint']}",
                )
                policy = (optim.NewtonTrustRegion(), termination)
                transferred_bytes = int(problem.initial.nbytes)
            elif problem.variant == "constrained":
                equality = optim.NonlinearConstraint(
                    lambda value, args: jnp.sum(value * value) - 1.0,
                    lower=0.0,
                    upper=0.0,
                    constraint_id="unit-circle",
                )
                inequality = optim.NonlinearConstraint(
                    lambda value, args: value[0] + value[1] - 2.0,
                    upper=0.0,
                    constraint_id="affine-upper",
                )
                native_problem = optim.MinimizationProblem(
                    lambda value, args: 2.0 * (jnp.sum(value * value) - 1.0) - value[0],
                    constraints=(equality, inequality),
                    problem_id=f"benchmark-maratos:{problem.identity()['fingerprint']}",
                )
                policy = (optim.SQP(), termination)
                transferred_bytes = int(problem.initial.nbytes)
            elif problem.variant == "bounded-least-squares":
                if problem.target is None:
                    raise ValueError("bounded least-squares benchmark lacks its target")
                target = jnp.asarray(problem.target)
                native_problem = optim.NonlinearLeastSquaresProblem(
                    lambda value, args: value - target,
                    bounds=optim.Bounds(0.0, 1.0),
                    problem_id=(
                        "benchmark-bounded-least-squares:"
                        f"{problem.identity()['fingerprint']}"
                    ),
                )
                policy = (
                    optim.BoundedLevenbergMarquardt(),
                    termination,
                )
                transferred_bytes = int(problem.initial.nbytes + problem.target.nbytes)
            else:
                if problem.target is None:
                    raise ValueError("proximal benchmark lacks its target")
                target = jnp.asarray(problem.target)
                native_problem = optim.ProximalProblem(
                    optim.MinimizationProblem(
                        lambda value, args: 0.5 * jnp.sum((value - target) ** 2),
                        problem_id=f"benchmark-l1:{problem.identity()['fingerprint']}",
                    ),
                    optim.L1Functional(problem.l1_weight),
                )
                policy = (optim.ProximalGradient(), termination)
                transferred_bytes = int(problem.initial.nbytes + problem.target.nbytes)
            return _PhydraxState(
                spec=spec,
                phx=phx,
                native_problem=native_problem,
                policy=policy,
                rhs=initial,
                host_to_device_bytes=transferred_bytes,
            )
        if isinstance(problem, MathematicalProgramProblem):
            optim = import_module("phydrax.optim")
            bounds = optim.Bounds(
                jnp.asarray(problem.lower),
                jnp.asarray(problem.upper),
            )
            if problem.variant == "lp":
                native_problem = optim.LinearProgram(
                    jnp.asarray(problem.linear),
                    equality_matrix=jnp.asarray(problem.equality_matrix),
                    equality_rhs=jnp.asarray(problem.equality_rhs),
                    inequality_matrix=jnp.asarray(problem.inequality_matrix),
                    inequality_rhs=jnp.asarray(problem.inequality_rhs),
                    bounds=bounds,
                    problem_id=f"benchmark-lp:{problem.name}:{problem.seed}",
                )
                method = optim.DensePrimalDualQP()
            elif problem.variant == "qp":
                native_problem = optim.QuadraticProgram(
                    jnp.asarray(problem.quadratic),
                    jnp.asarray(problem.linear),
                    equality_matrix=jnp.asarray(problem.equality_matrix),
                    equality_rhs=jnp.asarray(problem.equality_rhs),
                    inequality_matrix=jnp.asarray(problem.inequality_matrix),
                    inequality_rhs=jnp.asarray(problem.inequality_rhs),
                    bounds=bounds,
                    problem_id=f"benchmark-qp:{problem.name}:{problem.seed}",
                )
                method = optim.DensePrimalDualQP()
            else:
                if problem.conic_matrix is None or problem.conic_rhs is None:
                    raise ValueError("SOCP benchmark lacks conic data")
                native_problem = optim.ConicProgram(
                    jnp.asarray(problem.quadratic),
                    jnp.asarray(problem.linear),
                    jnp.asarray(problem.conic_matrix),
                    jnp.asarray(problem.conic_rhs),
                    optim.SecondOrderCone(problem.conic_matrix.shape[0]),
                    bounds=bounds,
                    problem_id=f"benchmark-socp:{problem.name}:{problem.seed}",
                )
                method = optim.ClarabelInteriorPoint(presolve=False)
            policy = optim.ConvexSolvePolicy(
                method,
                termination=optim.ConvexTermination(
                    absolute=spec.tolerances.absolute,
                    relative=spec.tolerances.relative,
                    maximum_steps=spec.tolerances.max_steps,
                ),
            )
            transferred_bytes = int(
                sum(
                    array.nbytes
                    for array in (
                        problem.linear,
                        problem.equality_matrix,
                        problem.equality_rhs,
                        problem.inequality_matrix,
                        problem.inequality_rhs,
                        problem.lower,
                        problem.upper,
                    )
                )
            )
            return _PhydraxState(
                spec=spec,
                phx=phx,
                native_problem=native_problem,
                policy=policy,
                host_to_device_bytes=transferred_bytes,
            )
        if isinstance(problem, ContinuationProblem):
            continuation = import_module("phydrax.continuation")
            nonlinear = import_module("phydrax.nonlinear")
            native_problem = continuation.ParameterContinuationProblem(
                lambda state, parameter, args: state * state - parameter,
                problem_id=f"benchmark-fold:{problem.identity()['fingerprint']}",
            )
            maximum_corrector_steps = min(20, spec.tolerances.max_steps)
            method = continuation.PseudoArclengthContinuation(
                termination=nonlinear.NonlinearTermination(
                    absolute_residual=max(
                        spec.tolerances.absolute,
                        spec.tolerances.relative,
                    ),
                    relative_residual=0.0,
                    absolute_step=0.0,
                    relative_step=0.0,
                    maximum_steps=maximum_corrector_steps,
                ),
                initial_step=problem.initial_step,
                minimum_step=problem.min_step,
                maximum_step=problem.max_step,
                target_corrector_steps=min(4, maximum_corrector_steps),
                direction=problem.direction,
            )
            initial_coordinate = jnp.asarray(
                problem.initial_coordinate,
                dtype=jnp.asarray(problem.initial_state).dtype,
            )
            return _PhydraxState(
                spec=spec,
                phx=phx,
                native_problem=native_problem,
                policy=method,
                rhs=jnp.asarray(problem.initial_state),
                initial_coordinate=initial_coordinate,
                host_to_device_bytes=int(
                    problem.initial_state.nbytes + initial_coordinate.nbytes
                ),
            )
        raise TypeError(f"unsupported Phydrax problem type {type(problem).__name__!r}")

    def compilation_applicable(self, setup_state: _PhydraxState, /) -> bool:
        problem = setup_state.spec.problem
        return isinstance(
            problem,
            (
                SparseLinearProblem,
                GeneralEigenProblem,
                MathematicalProgramProblem,
                ContinuationProblem,
                OptimizationProblem,
            ),
        ) or (isinstance(problem, NonlinearProblem) and problem.variant == "root")

    def compilation_after_preparation(self, setup_state: _PhydraxState, /) -> bool:
        problem = setup_state.spec.problem
        return isinstance(problem, NonlinearProblem) and problem.variant == "root"

    def compile(self, setup_state: _PhydraxState, /) -> _PhydraxState:
        problem = setup_state.spec.problem
        phx = setup_state.phx
        if isinstance(problem, SparseLinearProblem):
            layout = (
                phx.linalg.RHSLayout((problem.rhs.shape[1],))
                if problem.block_size > 1
                else None
            )
            setup_state.plan = phx.linalg.plan(
                setup_state.native_problem,
                setup_state.policy,
                rhs_layout=layout,
            )
        elif isinstance(problem, GeneralEigenProblem):
            setup_state.plan = phx.linalg.eigen.plan_general_eigensolve(
                setup_state.native_problem,
                setup_state.policy,
            )
        elif isinstance(problem, MathematicalProgramProblem):
            setup_state.plan = phx.optim.plan_convex_program(
                setup_state.native_problem,
                setup_state.policy,
            )
        elif isinstance(problem, ContinuationProblem):
            setup_state.plan = phx.continuation.plan_continuation(
                setup_state.native_problem,
                num_steps=problem.max_points - 1,
                method=setup_state.policy,
                branch_id=f"benchmark:{problem.identity()['fingerprint']}",
            )
        elif isinstance(problem, NonlinearProblem) and problem.variant == "root":
            eqx = import_module("equinox")
            nonlinear = import_module("phydrax.nonlinear")

            def operation(prepared):
                result = nonlinear.solve_prepared_nonlinear(prepared)
                diagnostics = result.diagnostics
                return (
                    result.state,
                    result.status,
                    result.successful,
                    diagnostics.iterations,
                    diagnostics.jvp_evaluations,
                    diagnostics.linear_solves,
                    diagnostics.residual_evaluations,
                    diagnostics.jacobian_preparations,
                )

            setup_state.executable = (
                eqx.filter_jit(operation).lower(setup_state.prepared).compile()
            )
        elif isinstance(problem, OptimizationProblem):
            jax = import_module("jax")
            method, termination = setup_state.policy
            if problem.variant == "bounded-least-squares":

                def operation(initial):
                    return phx.optim.least_squares(
                        setup_state.native_problem,
                        initial,
                        method=method,
                        termination=termination,
                    )
            elif problem.variant == "proximal":

                def operation(initial):
                    return phx.optim.proximal_minimize(
                        setup_state.native_problem,
                        initial,
                        method=method,
                        termination=termination,
                    )
            else:

                def operation(initial):
                    return phx.optim.minimize(
                        setup_state.native_problem,
                        initial,
                        method=method,
                        termination=termination,
                    )

            setup_state.prepared = jax.jit(operation).lower(setup_state.rhs).compile()
        else:
            raise TypeError("Phydrax compilation does not apply to this problem type")
        return setup_state

    def preparation_applicable(self, compiled_state: _PhydraxState, /) -> bool:
        problem = compiled_state.spec.problem
        return isinstance(
            problem,
            (
                SparseLinearProblem,
                GeneralEigenProblem,
                ContinuationProblem,
                MathematicalProgramProblem,
            ),
        ) or (isinstance(problem, NonlinearProblem) and problem.variant == "root")

    def prepare(self, compiled_state: _PhydraxState, /) -> _PhydraxState:
        problem = compiled_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            compiled_state.prepared = compiled_state.phx.linalg.prepare(
                compiled_state.native_problem,
                compiled_state.plan,
            )
        elif isinstance(problem, GeneralEigenProblem):
            compiled_state.prepared = (
                compiled_state.phx.linalg.eigen.prepare_general_eigensolve(
                    compiled_state.native_problem,
                    compiled_state.plan,
                )
            )
        elif isinstance(problem, MathematicalProgramProblem):
            compiled_state.prepared = compiled_state.phx.optim.prepare_convex_program(
                compiled_state.native_problem,
                compiled_state.plan,
            )
        elif isinstance(problem, ContinuationProblem):
            compiled_state.prepared = (
                compiled_state.phx.continuation.prepare_continuation(
                    compiled_state.native_problem,
                    compiled_state.rhs,
                    compiled_state.initial_coordinate,
                    compiled_state.plan,
                )
            )
        elif isinstance(problem, NonlinearProblem) and problem.variant == "root":
            nonlinear = import_module("phydrax.nonlinear")
            compiled_state.prepared = nonlinear.prepare_nonlinear(
                compiled_state.native_problem,
                compiled_state.rhs,
                method=compiled_state.method,
                termination=compiled_state.policy,
                args=compiled_state.target,
            )
        else:
            raise TypeError(
                "Phydrax preparation applies only to linear, eigen, continuation, "
                "and nonlinear-root lifecycle cases"
            )
        return compiled_state

    def solve(self, prepared_state: _PhydraxState, /) -> SolveResult:
        problem = prepared_state.spec.problem
        phx = prepared_state.phx
        jnp = import_module("jax.numpy")
        if isinstance(problem, SparseLinearProblem):
            result = phx.linalg.solve(prepared_state.prepared, prepared_state.rhs)
            statuses = result.status
            iterations = result.diagnostics.iterations
            success_code = int(phx.linalg.LinearSolveStatus.SUCCESS)
            return SolveResult(
                solution=result.value,
                auxiliary={"status_codes": statuses},
                converged=jnp.all(statuses == success_code),
                message="Phydrax linear solve completed; status is in auxiliary evidence",
                operations={
                    "iterations": jnp.max(iterations),
                    "matvecs": jnp.sum(iterations),
                    "preconditioner_applications": jnp.sum(iterations),
                    "linear_solves": statuses.size,
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if isinstance(problem, GeneralEigenProblem):
            result = phx.linalg.eigen.general_eigensolve(prepared_state.prepared)
            diagnostics = result.diagnostics
            return SolveResult(
                solution=result.right_eigenvector_coordinates,
                auxiliary={
                    "eigenvalues": result.eigenvalues,
                    "status_code": result.status,
                },
                converged=result.successful,
                message=(
                    "Phydrax general eigensolve completed; status is in "
                    "auxiliary evidence"
                ),
                operations={
                    "iterations": diagnostics.decomposition_count,
                    "matvecs": diagnostics.arnoldi_action_count,
                    "preconditioner_applications": 0,
                    "linear_solves": diagnostics.transform_solve_count,
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if isinstance(problem, NonlinearProblem):
            if problem.variant == "root":
                (
                    state,
                    status,
                    successful,
                    iterations,
                    jvp_evaluations,
                    linear_solves,
                    residual_evaluations,
                    jacobian_preparations,
                ) = prepared_state.executable(prepared_state.prepared)
                return SolveResult(
                    solution=state,
                    auxiliary={"status_code": status},
                    converged=successful,
                    message=(
                        "Phydrax nonlinear solve completed; status is in "
                        "auxiliary evidence"
                    ),
                    operations={
                        "iterations": iterations,
                        "matvecs": jvp_evaluations,
                        "preconditioner_applications": None,
                        "linear_solves": linear_solves,
                        "nonlinear_evaluations": residual_evaluations,
                        "jacobian_evaluations": jacobian_preparations,
                    },
                )
            method, termination = prepared_state.policy
            result = method.solve(
                prepared_state.native_problem,
                prepared_state.rhs,
                termination=termination,
            )
            diagnostics = result.diagnostics
            return SolveResult(
                solution=result.state,
                auxiliary={"status_code": result.status},
                converged=result.successful,
                message=(
                    "Phydrax nonlinear solve completed; status is in auxiliary evidence"
                ),
                operations={
                    "iterations": diagnostics.iterations,
                    "matvecs": diagnostics.jvp_evaluations,
                    "preconditioner_applications": None,
                    "linear_solves": diagnostics.linear_solves,
                    "nonlinear_evaluations": diagnostics.residual_evaluations,
                    "jacobian_evaluations": diagnostics.jacobian_preparations,
                },
            )
        if isinstance(problem, MathematicalProgramProblem):
            execution = phx.optim.solve_convex_program(prepared_state.prepared)
            result = execution.result
            return SolveResult(
                solution=result.primal,
                auxiliary={
                    "status_code": result.status,
                    "objective": result.objective,
                    "certificate": result.certificate,
                    "equality_dual": result.equality_dual,
                    "inequality_dual": result.inequality_dual,
                    "lower_bound_dual": result.lower_bound_dual,
                    "upper_bound_dual": result.upper_bound_dual,
                    "cone_dual": result.cone_dual,
                },
                converged=result.successful,
                message="Phydrax mathematical program completed with audited evidence",
                operations={
                    "iterations": result.iterations,
                    "matvecs": None,
                    "preconditioner_applications": None,
                    "linear_solves": None,
                    "nonlinear_evaluations": None,
                    "jacobian_evaluations": None,
                },
            )
        if isinstance(problem, OptimizationProblem):
            result = prepared_state.prepared(prepared_state.rhs)
            diagnostics = result.diagnostics
            return SolveResult(
                solution=result.parameters,
                auxiliary={
                    "status_code": result.status,
                    "objective": result.objective,
                },
                converged=result.successful,
                message=(
                    "Phydrax optimization completed; status and objective are "
                    "in auxiliary evidence"
                ),
                operations={
                    "iterations": diagnostics.iterations,
                    "matvecs": diagnostics.hvp_evaluations,
                    "preconditioner_applications": None,
                    "linear_solves": diagnostics.linear_solves,
                    "nonlinear_evaluations": (
                        diagnostics.objective_evaluations
                        + diagnostics.gradient_evaluations
                        + diagnostics.constraint_evaluations
                    ),
                    "jacobian_evaluations": diagnostics.jacobian_evaluations,
                },
            )
        if isinstance(problem, ContinuationProblem):
            result = phx.continuation.run_continuation(prepared_state.prepared)
            points = result.points
            states = jnp.stack([point.state for point in points])
            coordinates = jnp.asarray([point.coordinate for point in points])
            iterations = result.diagnostics.corrector_iterations
            return SolveResult(
                solution=states,
                auxiliary={
                    "coordinates": coordinates,
                    "branch_successful": result.successful,
                    "residual_tolerance": max(
                        prepared_state.spec.tolerances.absolute,
                        prepared_state.spec.tolerances.relative,
                    ),
                    "termination_status": result.status,
                },
                converged=result.successful & (len(points) > 1),
                message=(
                    "Phydrax continuation completed; status is in auxiliary evidence"
                ),
                operations={
                    "iterations": iterations,
                    "matvecs": None,
                    "preconditioner_applications": None,
                    "linear_solves": iterations,
                    "nonlinear_evaluations": None,
                    "jacobian_evaluations": None,
                },
            )
        raise TypeError(f"unsupported Phydrax problem type {type(problem).__name__!r}")

    def differentiation_applicable(self, prepared_state: _PhydraxState, /) -> bool:
        problem = prepared_state.spec.problem
        return (
            isinstance(problem, NonlinearProblem)
            and problem.variant == "root"
            and problem.root_kind == "separable"
        )

    def compile_differentiation(
        self,
        prepared_state: _PhydraxState,
        /,
    ) -> _PhydraxState:
        nonlinear = import_module("phydrax.nonlinear")
        jax = import_module("jax")

        def solve_for_target(target):
            return nonlinear.implicit_root(
                prepared_state.native_problem,
                prepared_state.rhs,
                method=prepared_state.prepared.method,
                termination=prepared_state.policy,
                args=target,
            )

        derivative = jax.jacrev(solve_for_target)
        prepared_state.differentiation_executable = (
            jax.jit(derivative).lower(prepared_state.target).compile()
        )
        return prepared_state

    def differentiate(self, prepared_state: _PhydraxState, /) -> Any:
        if prepared_state.differentiation_executable is None:
            raise ValueError("Phydrax differentiation must be compiled before execution.")
        return prepared_state.differentiation_executable(prepared_state.target)

    def refresh_applicable(self, prepared_state: _PhydraxState, /) -> bool:
        problem = prepared_state.spec.problem
        return isinstance(
            problem,
            (
                SparseLinearProblem,
                GeneralEigenProblem,
                ContinuationProblem,
                MathematicalProgramProblem,
            ),
        ) or (isinstance(problem, NonlinearProblem) and problem.variant == "root")

    def refresh(
        self,
        prepared_state: _PhydraxState,
        /,
    ) -> tuple[_PhydraxState, RefreshEvidence]:
        problem = prepared_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            operator = prepared_state.native_problem.operator
            refreshed_operator = prepared_state.phx.sparse.SparseCoordinateOperator(
                operator.relation,
                operator.coefficients * 1.01,
                source=operator.source,
                target=operator.target,
                properties=operator.properties,
                operator_id=operator.operator_id,
            )
            prepared_state.native_problem = prepared_state.phx.linalg.LinearSystem(
                refreshed_operator,
                problem_id=prepared_state.native_problem.problem_id,
            )
            prepared_state.refreshed_certificate_problem = replace(
                problem,
                coefficients=problem.coefficients * 1.01,
            )
            prepared_state.prepared = prepared_state.phx.linalg.refresh(
                prepared_state.prepared,
                prepared_state.native_problem,
            )
        elif isinstance(problem, GeneralEigenProblem):
            operator = prepared_state.native_problem.operator
            refreshed_operator = prepared_state.phx.linalg.DenseLinearOperator(
                operator.matrix * 1.01,
                source=operator.source,
                target=operator.target,
                properties=operator.properties,
                operator_id=operator.operator_id,
            )
            prepared_state.native_problem = (
                prepared_state.phx.linalg.eigen.GeneralEigenproblem(
                    refreshed_operator,
                    problem_id=prepared_state.native_problem.problem_id,
                )
            )
            prepared_state.refreshed_certificate_problem = replace(
                problem,
                matrix=problem.matrix * 1.01,
            )
            prepared_state.prepared = (
                prepared_state.phx.linalg.eigen.refresh_general_eigensolve(
                    prepared_state.prepared,
                    prepared_state.native_problem,
                )
            )
        elif isinstance(problem, NonlinearProblem) and problem.variant == "root":
            nonlinear = import_module("phydrax.nonlinear")
            prepared_state.target = prepared_state.target * 1.01
            prepared_state.refreshed_certificate_problem = replace(
                problem,
                target=problem.target * 1.01,
            )
            prepared_state.prepared = nonlinear.refresh_nonlinear(
                prepared_state.prepared,
                prepared_state.native_problem,
                prepared_state.rhs,
                args=prepared_state.target,
            )
        elif isinstance(problem, MathematicalProgramProblem):
            refreshed_problem = replace(
                problem,
                quadratic=(
                    None if problem.quadratic is None else problem.quadratic * 1.01
                ),
                linear=problem.linear * 1.01,
            )
            refreshed_spec = replace(prepared_state.spec, problem=refreshed_problem)
            refreshed_setup = self.setup(refreshed_spec)
            prepared_state.native_problem = refreshed_setup.native_problem
            prepared_state.refreshed_certificate_problem = refreshed_problem
            prepared_state.prepared = prepared_state.phx.optim.refresh_convex_program(
                prepared_state.prepared,
                prepared_state.native_problem,
            )
        elif isinstance(problem, ContinuationProblem):
            prepared_state.initial_coordinate = prepared_state.initial_coordinate * 0.99
            prepared_state.refreshed_certificate_problem = replace(
                problem,
                initial_coordinate=problem.initial_coordinate * 0.99,
            )
            prepared_state.prepared = (
                prepared_state.phx.continuation.refresh_continuation(
                    prepared_state.prepared,
                    prepared_state.rhs,
                    prepared_state.initial_coordinate,
                )
            )
        else:
            raise TypeError(
                "Phydrax refresh applies only to linear, eigen, continuation, "
                "mathematical-program, and nonlinear-root lifecycle cases"
            )
        return prepared_state, RefreshEvidence(
            applicable=True,
            symbolic_reused=True,
            numeric_refreshed=True,
            symbolic_refresh_count=0,
            numeric_refresh_count=1,
            evidence=(
                "public Phydrax refresh reused the compiled symbolic plan after a "
                "deterministic structure-preserving 1% numeric perturbation; the "
                "refreshed state is re-solved and independently certified"
            ),
        )

    def certificate_problem(self, prepared_state: _PhydraxState, /) -> Any:
        return (
            prepared_state.spec.problem
            if prepared_state.refreshed_certificate_problem is None
            else prepared_state.refreshed_certificate_problem
        )

    def memory(
        self,
        prepared_state: _PhydraxState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        problem = prepared_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            matrix_bytes = int(
                problem.coefficients.nbytes + problem.rows.nbytes + problem.columns.nbytes
            )
            selected = prepared_state.plan.candidates[-1]
            setup_bytes = int(
                selected.existing_storage_bytes
                + selected.additional_matrix_bytes
                + selected.factorization_bytes
                + selected.preconditioner_storage_bytes
                + selected.recycling_state_bytes
            )
            preparation_workspace = int(
                selected.preparation_workspace_bytes
                + selected.preconditioner_preparation_workspace_bytes
            )
            right_hand_sides = 1 if problem.rhs.ndim == 1 else int(problem.rhs.shape[1])
            solve_workspace = right_hand_sides * int(
                selected.solve_workspace_bytes_per_rhs
                + selected.krylov_basis_bytes_per_rhs
                + selected.preconditioner_apply_workspace_bytes_per_rhs
            )
            return {
                "matrix_bytes": matrix_bytes,
                "setup_bytes": setup_bytes,
                "peak_estimate_bytes": setup_bytes
                + max(preparation_workspace, solve_workspace),
                "evidence": "public Phydrax selected-plan conservative storage and solve-workspace estimate",
            }
        if isinstance(problem, GeneralEigenProblem):
            cost = prepared_state.plan.cost
            return {
                "matrix_bytes": int(cost.input_matrix_bytes),
                "setup_bytes": int(cost.preparation_bytes),
                "peak_estimate_bytes": int(
                    cost.preparation_bytes
                    + cost.workspace_bytes
                    + cost.krylov_basis_bytes
                ),
                "evidence": "public Phydrax general-eigen plan cost estimate",
            }
        if isinstance(problem, NonlinearProblem):
            arrays = [problem.initial, problem.target]
            if problem.lower is not None:
                arrays.append(problem.lower)
            if problem.upper is not None:
                arrays.append(problem.upper)
            if problem.diagonal is not None:
                arrays.append(problem.diagonal)
            matrix_bytes = int(sum(array.nbytes for array in arrays))
        elif isinstance(problem, MathematicalProgramProblem):
            arrays = [
                problem.linear,
                problem.equality_matrix,
                problem.equality_rhs,
                problem.inequality_matrix,
                problem.inequality_rhs,
                problem.lower,
                problem.upper,
            ]
            if problem.quadratic is not None:
                arrays.append(problem.quadratic)
            if problem.conic_matrix is not None and problem.conic_rhs is not None:
                arrays.extend((problem.conic_matrix, problem.conic_rhs))
            matrix_bytes = int(sum(array.nbytes for array in arrays))
        elif isinstance(problem, OptimizationProblem):
            arrays = [problem.initial, problem.optimum]
            if problem.target is not None:
                arrays.append(problem.target)
            matrix_bytes = int(sum(array.nbytes for array in arrays))
        else:
            matrix_bytes = int(
                problem.initial_state.nbytes
                + np.asarray(problem.initial_coordinate).nbytes
            )
        return {
            "matrix_bytes": matrix_bytes,
            "setup_bytes": 0,
            "peak_estimate_bytes": None,
            "evidence": (
                "exact benchmark input bytes; optimization/nonlinear/continuation "
                "transient workspace peak is unavailable"
            ),
        }

    def transfers(
        self,
        prepared_state: _PhydraxState,
        result: SolveResult,
        /,
        *,
        device_to_host_bytes: int,
    ) -> TransferEvidence:
        del result
        refresh_measured = self.refresh_applicable(prepared_state)
        return TransferEvidence(
            input_origin="numpy-host",
            host_to_device_bytes=prepared_state.host_to_device_bytes,
            host_to_device_timing_phase="setup",
            device_to_host_bytes=device_to_host_bytes,
            device_to_host_timing_phase=(
                "verification+refreshed_verification"
                if refresh_measured
                else "verification"
            ),
            evidence=(
                "canonical NumPy problem arrays were converted to JAX arrays during "
                "setup; all returned JAX solution, status, diagnostics, and certificate "
                "inputs were materialized by jax.device_get during "
                + (
                    "verification and refreshed verification"
                    if refresh_measured
                    else "verification"
                )
            ),
        )


def _jax_root_residual(problem, value, target, jnp):
    if problem.root_kind == "separable":
        return value * value - target
    if problem.grid_spacing is None:
        raise ValueError("semilinear Poisson problem is missing grid spacing")
    extended = jnp.pad(value, (1, 1))
    laplacian = (2.0 * value - extended[:-2] - extended[2:]) / problem.grid_spacing**2
    return laplacian + problem.nonlinearity * value**3 - target


def _required_public_api(capability: str) -> tuple[str, frozenset[str]]:
    if capability.startswith("linear."):
        return "phydrax.linalg", frozenset({"plan", "prepare", "refresh", "solve"})
    if capability == "eigen.general":
        return "phydrax.linalg.eigen", frozenset(
            {
                "GeneralEigenproblem",
                "GeneralEigenSelection",
                "GeneralEigenSolvePolicy",
                "GeneralEigenTolerancePolicy",
                "RestartedArnoldi",
                "StandardTransform",
                "plan_general_eigensolve",
                "prepare_general_eigensolve",
                "refresh_general_eigensolve",
                "general_eigensolve",
            }
        )
    if capability == "nonlinear.root":
        return "phydrax.nonlinear", frozenset(
            {
                "NonlinearSystemProblem",
                "NonlinearTermination",
                "prepare_nonlinear",
                "refresh_nonlinear",
                "solve_prepared_nonlinear",
            }
        )
    if capability == "nonlinear.vi":
        return "phydrax.nonlinear", frozenset(
            {
                "Bounds",
                "NonlinearTermination",
                "SemismoothNewton",
                "VariationalInequalityProblem",
            }
        )
    if capability.startswith("optimization."):
        if capability == "optimization.linear-program":
            return "phydrax.optim", frozenset(
                {
                    "LinearProgram",
                    "ConvexSolvePolicy",
                    "DensePrimalDualQP",
                    "plan_convex_program",
                    "prepare_convex_program",
                    "solve_convex_program",
                }
            )
        if capability == "optimization.quadratic-program":
            return "phydrax.optim", frozenset(
                {
                    "QuadraticProgram",
                    "ConvexSolvePolicy",
                    "DensePrimalDualQP",
                    "plan_convex_program",
                    "prepare_convex_program",
                    "solve_convex_program",
                }
            )
        if capability == "optimization.conic-program":
            return "phydrax.optim", frozenset(
                {
                    "ConicProgram",
                    "ConvexSolvePolicy",
                    "ClarabelInteriorPoint",
                    "SecondOrderCone",
                    "plan_convex_program",
                    "prepare_convex_program",
                    "solve_convex_program",
                }
            )
        if capability == "optimization.bounded-least-squares":
            return "phydrax.optim", frozenset(
                {
                    "BoundedLevenbergMarquardt",
                    "Bounds",
                    "NonlinearLeastSquaresProblem",
                    "OptimizationTermination",
                    "least_squares",
                }
            )
        common = {
            "MinimizationProblem",
            "OptimizationTermination",
        }
        if capability == "optimization.unconstrained":
            names = common | {"NewtonTrustRegion", "minimize"}
        elif capability == "optimization.constrained":
            names = common | {"NonlinearConstraint", "SQP", "minimize"}
        else:
            names = common | {
                "L1Functional",
                "ProximalGradient",
                "ProximalProblem",
                "proximal_minimize",
            }
        return "phydrax.optim", frozenset(names)
    return "phydrax.continuation", frozenset(
        {
            "ParameterContinuationProblem",
            "PseudoArclengthContinuation",
            "plan_continuation",
            "prepare_continuation",
            "refresh_continuation",
            "run_continuation",
        }
    )


def _phydrax_version() -> str:
    try:
        return importlib.metadata.version("phydrax")
    except importlib.metadata.PackageNotFoundError:
        return "source checkout"


def _version_evidence() -> dict[str, str]:
    versions = {"phydrax": _phydrax_version()}
    versions["phydrax_source_sha256"] = _source_fingerprint()
    try:
        versions["jax"] = importlib.metadata.version("jax")
    except importlib.metadata.PackageNotFoundError:
        pass
    return versions


def _source_fingerprint() -> str:
    module_spec = importlib.util.find_spec("phydrax")
    if module_spec is None or module_spec.origin is None:
        return "unavailable"
    package_root = Path(module_spec.origin).parent
    source_paths = sorted(package_root.rglob("*.py"))
    digest = hashlib.sha256()
    for source_path in source_paths:
        relative_path = source_path.relative_to(package_root)
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(source_path.read_bytes())
    return digest.hexdigest() if source_paths else "unavailable"


__all__ = ["PhydraxAdapter"]
