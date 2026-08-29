#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from ..linalg import (
    DifferentiationPolicy,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    MINRES,
    OperatorProperties,
    PyTreeSpace,
    solve as solve_linear,
    TolerancePolicy,
)
from ._interior_point import PrimalDualInteriorPoint
from ._iterative._base import (
    AbstractLeastSquaresMethod,
    AbstractMinimizationMethod,
    AbstractScalarIterativeMethod,
)
from ._iterative._types import (
    _tree_allfinite,
    _tree_norm,
    _validate_real_inexact_tree,
    MinimizationProblem,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)
from ._least_squares import (
    _prepare_residual_model,
    _run_least_squares_iterations,
    GaussNewton,
)
from ._newton_krylov import NewtonKrylov
from ._nonlinear_constraints import _canonical_constraints, _constraint_layout
from ._scalar import _run_scalar_iterations


def _default_implicit_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(
            relative=1e-9,
            absolute=1e-11,
            max_steps=512,
        ),
        differentiation=DifferentiationPolicy("none"),
    )


def _algorithmic_linear_policy(policy: LinearSolvePolicy, /) -> LinearSolvePolicy:
    return eqx.tree_at(
        lambda selected: selected.differentiation,
        policy,
        DifferentiationPolicy("none"),
    )


def _error_if_tree(
    value: PyTree[Array],
    predicate: Any,
    message: str,
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda leaf: eqx.error_if(leaf, predicate, message),
        value,
    )


def _implicit_tangent_solve(
    root: PyTree[Any],
    policy: LinearSolvePolicy,
    /,
):
    space = PyTreeSpace(root)
    algorithmic_policy = _algorithmic_linear_policy(policy)

    def tangent_solve(linearized, right_hand_side):
        def solve_action(action, rhs):
            operator = FunctionLinearOperator(
                action,
                source=space,
                target=space,
                transpose_action=action,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "asserted"},
                ),
                operator_id="implicit-stationarity-jacobian",
                closure_convert=False,
            )
            result = solve_linear(
                LinearSystem(operator),
                rhs,
                policy=algorithmic_policy,
            )
            failed = (
                ~jnp.all(result.diagnostics.converged)
                | ~jnp.all(result.diagnostics.finite)
                | ~_tree_allfinite(result.value)
            )
            return _error_if_tree(
                result.value,
                failed,
                "Implicit stationarity solve is singular or did not converge.",
            )

        return jax.lax.custom_linear_solve(
            linearized,
            right_hand_side,
            solve=solve_action,
            symmetric=True,
        )

    return tangent_solve


def _regularity_anchor(
    stationarity,
    root: PyTree[Any],
    policy: LinearSolvePolicy,
    /,
) -> Array:
    leaves, structure = jax.tree.flatten(root)
    keys = jax.random.split(jax.random.key(0), len(leaves))
    probe = jax.tree.unflatten(
        structure,
        [
            jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
            for key, leaf in zip(keys, leaves, strict=True)
        ],
    )
    _, linearized = jax.linearize(stationarity, root)
    solution = _implicit_tangent_solve(root, policy)(linearized, probe)
    return sum(
        (jnp.sum(leaf) for leaf in jax.tree.leaves(solution)),
        start=jnp.asarray(0.0),
    )


def implicit_minimize(
    problem_or_objective: MinimizationProblem | Callable[[PyTree[Any], Any], Any],
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractScalarIterativeMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    has_aux: bool = False,
    linear_policy: LinearSolvePolicy | None = None,
) -> PyTree[Array]:
    """Return a regular unconstrained minimizer with implicit derivatives.

    The primal solve uses the selected native method. Derivatives solve the
    stationarity system and therefore do not differentiate through iterations.
    """

    problem = (
        problem_or_objective
        if isinstance(problem_or_objective, MinimizationProblem)
        else MinimizationProblem(problem_or_objective, has_aux=has_aux)
    )
    method_ = NewtonKrylov() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    policy = _default_implicit_linear_policy() if linear_policy is None else linear_policy
    if not isinstance(method_, AbstractScalarIterativeMethod):
        raise TypeError("method must be an AbstractScalarIterativeMethod or None.")
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
    if problem.bounds is not None or problem.constraints:
        raise ValueError("implicit_minimize supports only unconstrained smooth problems.")
    initial = _validate_real_inexact_tree(
        initial_parameters,
        name="initial_parameters",
    )

    def value_function(parameters):
        value, _ = problem.value(parameters, args)
        return value

    def stationarity(parameters):
        _, gradient = problem.value_and_gradient(parameters, args)
        return gradient

    def primal_solve(_, guess):
        run = _run_scalar_iterations(
            method_,
            value_function,
            guess,
            termination_,
        )
        gradient = stationarity(run.parameters)
        gradient_norm = _tree_norm(gradient)
        successful = (
            _tree_allfinite(run.parameters)
            & _tree_allfinite(gradient)
            & jnp.isfinite(gradient_norm)
            & (
                gradient_norm
                <= termination_.optimality_threshold(run.state.initial_optimality_norm)
            )
        )
        regularity_anchor = jax.lax.cond(
            successful,
            lambda _: _regularity_anchor(stationarity, run.parameters, policy),
            lambda _: jnp.asarray(0.0),
            None,
        )
        guarded = _error_if_tree(
            run.parameters,
            ~successful,
            "Implicit minimization requires a successful regular stationary point.",
        )
        return jax.tree.map(
            lambda leaf: leaf + jnp.asarray(0, dtype=leaf.dtype) * regularity_anchor,
            guarded,
        )

    return jax.lax.custom_root(
        stationarity,
        initial,
        primal_solve,
        _implicit_tangent_solve(initial, policy),
    )


def implicit_least_squares(
    problem_or_residual: NonlinearLeastSquaresProblem | Callable[[PyTree[Any], Any], Any],
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractLeastSquaresMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    has_aux: bool = False,
    linear_policy: LinearSolvePolicy | None = None,
) -> PyTree[Array]:
    """Return a regular nonlinear least-squares solution with implicit derivatives."""

    problem = (
        problem_or_residual
        if isinstance(problem_or_residual, NonlinearLeastSquaresProblem)
        else NonlinearLeastSquaresProblem(problem_or_residual, has_aux=has_aux)
    )
    method_ = GaussNewton() if method is None else method
    termination_ = OptimizationTermination() if termination is None else termination
    policy = _default_implicit_linear_policy() if linear_policy is None else linear_policy
    if not isinstance(method_, AbstractLeastSquaresMethod):
        raise TypeError("method must be an AbstractLeastSquaresMethod or None.")
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
    initial = _validate_real_inexact_tree(
        initial_parameters,
        name="initial_parameters",
    )

    def residual_function(parameters):
        residual, _ = problem.value(parameters, args)
        return residual

    def stationarity(parameters):
        return _prepare_residual_model(
            residual_function,
            parameters,
        ).gradient

    def primal_solve(_, guess):
        run = _run_least_squares_iterations(
            method_,
            residual_function,
            guess,
            termination_,
        )
        gradient = stationarity(run.parameters)
        gradient_norm = _tree_norm(gradient)
        successful = (
            _tree_allfinite(run.parameters)
            & _tree_allfinite(gradient)
            & jnp.isfinite(gradient_norm)
            & (
                gradient_norm
                <= termination_.optimality_threshold(run.state.initial_optimality_norm)
            )
        )
        regularity_anchor = jax.lax.cond(
            successful,
            lambda _: _regularity_anchor(stationarity, run.parameters, policy),
            lambda _: jnp.asarray(0.0),
            None,
        )
        guarded = _error_if_tree(
            run.parameters,
            ~successful,
            "Implicit least squares requires a successful regular stationary point.",
        )
        return jax.tree.map(
            lambda leaf: leaf + jnp.asarray(0, dtype=leaf.dtype) * regularity_anchor,
            guarded,
        )

    return jax.lax.custom_root(
        stationarity,
        initial,
        primal_solve,
        _implicit_tangent_solve(initial, policy),
    )


def _default_constrained_implicit_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        MINRES(),
        tolerance=TolerancePolicy(
            relative=1e-9,
            absolute=1e-11,
            max_steps=512,
        ),
        differentiation=DifferentiationPolicy("none"),
    )


def _implicit_kkt_tangent_solve(
    root: PyTree[Any],
    policy: LinearSolvePolicy,
    /,
):
    space = PyTreeSpace(root)
    algorithmic_policy = _algorithmic_linear_policy(policy)

    def tangent_solve(linearized, right_hand_side):
        def solve_action(action, rhs):
            operator = FunctionLinearOperator(
                action,
                source=space,
                target=space,
                transpose_action=action,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "asserted"},
                ),
                operator_id="implicit-active-kkt-jacobian",
                closure_convert=False,
            )
            result = solve_linear(
                LinearSystem(operator),
                rhs,
                policy=algorithmic_policy,
            )
            failed = (
                ~jnp.all(result.diagnostics.converged)
                | ~jnp.all(result.diagnostics.finite)
                | ~_tree_allfinite(result.value)
            )
            return _error_if_tree(
                result.value,
                failed,
                "Implicit active-set KKT solve is singular or did not converge.",
            )

        return jax.lax.custom_linear_solve(
            linearized,
            right_hand_side,
            solve=solve_action,
            symmetric=True,
        )

    return tangent_solve


def _kkt_regularity_anchor(
    equation,
    root: PyTree[Any],
    policy: LinearSolvePolicy,
    /,
) -> Array:
    leaves, structure = jax.tree.flatten(root)
    keys = jax.random.split(jax.random.key(1), len(leaves))
    probe = jax.tree.unflatten(
        structure,
        [
            jax.random.normal(key, leaf.shape, dtype=leaf.dtype)
            for key, leaf in zip(keys, leaves, strict=True)
        ],
    )
    _, linearized = jax.linearize(equation, root)
    solution = _implicit_kkt_tangent_solve(root, policy)(linearized, probe)
    return sum(
        (jnp.sum(leaf) for leaf in jax.tree.leaves(solution)),
        start=jnp.asarray(0.0),
    )


def implicit_constrained_minimize(
    problem: MinimizationProblem,
    initial_parameters: PyTree[Any],
    /,
    *,
    method: AbstractMinimizationMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    linear_policy: LinearSolvePolicy | None = None,
    active_tolerance: float = 1e-6,
    strict_complementarity_tolerance: float = 1e-5,
) -> PyTree[Array]:
    """Return a regular constrained minimizer with implicit KKT derivatives.

    Equalities, nonlinear inequalities, and parameter bounds share one fixed
    canonical layout. Derivatives freeze the strictly complementary active set;
    ambiguous active sets, failed primal solves, and singular KKT systems fail
    explicitly instead of returning a selected generalized derivative.
    """

    method_ = (
        PrimalDualInteriorPoint(mode="matrix-free-centered") if method is None else method
    )
    termination_ = OptimizationTermination() if termination is None else termination
    policy = (
        _default_constrained_implicit_linear_policy()
        if linear_policy is None
        else linear_policy
    )
    active_tolerance_ = float(active_tolerance)
    strict_tolerance = float(strict_complementarity_tolerance)
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be a MinimizationProblem.")
    if not problem.constraints and problem.bounds is None:
        raise ValueError(
            "implicit_constrained_minimize requires constraints or parameter bounds; "
            "use implicit_minimize for unconstrained problems."
        )
    if not isinstance(method_, AbstractMinimizationMethod):
        raise TypeError("method must be an AbstractMinimizationMethod or None.")
    if not method_.capabilities.implicit_differentiation:
        raise ValueError(
            f"{method_.method_id} does not declare implicit-differentiation support."
        )
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be an OptimizationTermination or None.")
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
    if (
        not np.isfinite(active_tolerance_)
        or not np.isfinite(strict_tolerance)
        or active_tolerance_ <= 0.0
        or strict_tolerance <= active_tolerance_
    ):
        raise ValueError(
            "Strict-complementarity tolerance must exceed a positive active tolerance."
        )
    initial = _validate_real_inexact_tree(
        initial_parameters,
        name="initial_parameters",
    )
    flat_initial, unravel = ravel_pytree(initial)
    layout = _constraint_layout(problem, initial, args)
    parameter_size = int(flat_initial.size)
    equality_size = int(layout.equality_indices.size)
    inequality_size = int(layout.lower_indices.size + layout.upper_indices.size)
    root_size = parameter_size + equality_size + inequality_size
    root_spec = jax.ShapeDtypeStruct((root_size,), flat_initial.dtype)
    active_spec = jax.ShapeDtypeStruct((inequality_size,), flat_initial.dtype)
    valid_spec = jax.ShapeDtypeStruct((), flat_initial.dtype)
    output_spec = root_spec, active_spec, valid_spec
    has_dynamic_args = args is not None
    callback_argument = args if has_dynamic_args else jnp.asarray(0, dtype=jnp.int32)

    def actual_args(callback_args):
        return callback_args if has_dynamic_args else None

    def host_primal_solve(callback_args, callback_initial):
        dynamic_args = actual_args(callback_args)
        primal_initial = unravel(callback_initial)
        result = method_.solve(
            problem,
            primal_initial,
            termination=termination_,
            args=dynamic_args,
        )
        certificate = result.certificate
        if certificate is None:
            equality_multipliers = np.zeros(
                (equality_size,),
                dtype=np.dtype(flat_initial.dtype),
            )
            inequality_multipliers = np.zeros(
                (inequality_size,),
                dtype=np.dtype(flat_initial.dtype),
            )
            active_mask = np.zeros((inequality_size,), dtype=np.bool_)
            strictly_complementary = False
        else:
            equality_multipliers = np.asarray(certificate.equality_multipliers)
            inequality_multipliers = np.asarray(certificate.inequality_multipliers)
            slacks = np.asarray(certificate.slacks)
            active_mask = (slacks <= active_tolerance_) & (
                inequality_multipliers >= strict_tolerance
            )
            inactive = (slacks >= strict_tolerance) & (
                inequality_multipliers <= active_tolerance_
            )
            strictly_complementary = bool(np.all(active_mask | inactive))
        valid = bool(result.successful) and strictly_complementary
        flat_parameters, _ = ravel_pytree(result.parameters)
        root = np.concatenate(
            (
                np.asarray(flat_parameters),
                equality_multipliers,
                inequality_multipliers,
            )
        )
        dtype = np.dtype(flat_initial.dtype)
        return (
            np.asarray(root, dtype=dtype),
            np.asarray(active_mask, dtype=dtype),
            np.asarray(valid, dtype=dtype),
        )

    @jax.custom_jvp
    def primal_callback(dynamic_callback_args, callback_initial):
        return jax.pure_callback(
            host_primal_solve,
            output_spec,
            dynamic_callback_args,
            callback_initial,
            vmap_method="sequential",
        )

    @primal_callback.defjvp
    def primal_callback_jvp(primals, tangents):
        del tangents
        values = primal_callback(*primals)
        return values, jax.tree.map(jnp.zeros_like, values)

    def root_equation(root, callback_args, active_mask):
        flat_parameters = root[:parameter_size]
        equality_multipliers = root[parameter_size : parameter_size + equality_size]
        inequality_multipliers = root[parameter_size + equality_size :]
        parameters = unravel(flat_parameters)
        dynamic_args = actual_args(callback_args)

        def constraints(candidate):
            return _canonical_constraints(
                problem,
                layout,
                candidate,
                dynamic_args,
            )

        _, objective_gradient = problem.value_and_gradient(parameters, dynamic_args)
        equality, inequality = constraints(parameters)
        _, pullback = jax.vjp(constraints, parameters)
        active_multipliers = jnp.where(
            active_mask,
            inequality_multipliers,
            0.0,
        )
        constraint_gradient = pullback((equality_multipliers, active_multipliers))[0]
        stationarity = jax.tree.map(
            lambda objective_part, constraint_part: objective_part + constraint_part,
            objective_gradient,
            constraint_gradient,
        )
        flat_stationarity, _ = ravel_pytree(stationarity)
        inequality_equation = jnp.where(
            active_mask,
            inequality,
            inequality_multipliers,
        )
        return jnp.concatenate((flat_stationarity, equality, inequality_equation))

    def primal_bundle(dynamic_callback_args):
        root, active_mask, valid = primal_callback(
            dynamic_callback_args,
            flat_initial,
        )
        guarded = eqx.error_if(
            root,
            valid <= 0.5,
            (
                "Constrained implicit differentiation requires a successful KKT "
                "point with an unambiguous strictly complementary active set."
            ),
        )
        return guarded, jax.lax.stop_gradient(active_mask > 0.5)

    initial_root = jnp.concatenate(
        (
            flat_initial,
            jnp.zeros((equality_size,), dtype=flat_initial.dtype),
            jnp.zeros((inequality_size,), dtype=flat_initial.dtype),
        )
    )

    def solve_dynamic(dynamic_callback_args):
        solved_root, active_mask = primal_bundle(dynamic_callback_args)
        solved_root = jax.lax.stop_gradient(solved_root)

        def equation(root):
            return root_equation(
                root,
                dynamic_callback_args,
                active_mask,
            )

        def primal_solve(_, guess):
            del guess
            regularity_anchor = _kkt_regularity_anchor(
                equation,
                solved_root,
                policy,
            )
            return (
                solved_root + jnp.asarray(0, dtype=solved_root.dtype) * regularity_anchor
            )

        root = jax.lax.custom_root(
            equation,
            initial_root,
            primal_solve,
            _implicit_kkt_tangent_solve(initial_root, policy),
        )
        return unravel(root[:parameter_size])

    return solve_dynamic(callback_argument)


__all__ = [
    "implicit_constrained_minimize",
    "implicit_least_squares",
    "implicit_minimize",
]
