#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..._trainable import partition_trainable
from ...domain import DomainFunction
from ...linalg import (
    ArraySpace,
    ConjugateGradient,
    DenseLinearOperator,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ...nn.parameters import ParameterSubspace
from .._functional_residual import prepare_functional_residual
from .._functional_solver import FunctionalSolver


class LocalCurvaturePlan(StrictModule):
    """Bounded exact dense local curvature policy."""

    damping: float = eqx.field(static=True)
    max_parameters: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        damping: float = 1.0e-4,
        max_parameters: int = 2048,
    ):
        damping_ = float(damping)
        maximum = int(max_parameters)
        if not math.isfinite(damping_) or damping_ <= 0.0:
            raise ValueError("damping must be finite and positive.")
        if maximum <= 0:
            raise ValueError("max_parameters must be positive.")
        self.damping = damping_
        self.max_parameters = maximum


class LocalCurvatureResult(StrictModule):
    functions: frozendict[str, DomainFunction]
    loss: Array
    gradient_norm: Array
    step_norm: Array
    accepted: bool = eqx.field(static=True)
    approximation: str = eqx.field(static=True)

    def __init__(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        loss: Array,
        gradient_norm: Array,
        step_norm: Array,
        accepted: bool,
    ):
        self.functions = frozendict(functions)
        self.loss = jnp.asarray(loss).reshape(())
        self.gradient_norm = jnp.asarray(gradient_norm).reshape(())
        self.step_norm = jnp.asarray(step_norm).reshape(())
        self.accepted = bool(accepted)
        self.approximation = "exact-local-dense-hessian"


def dense_local_curvature_step(
    solver: FunctionalSolver,
    subspace: ParameterSubspace,
    plan: LocalCurvaturePlan | None = None,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> LocalCurvatureResult:
    """Take one damped exact local Newton step through Phydrax linear algebra."""
    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solver must be a FunctionalSolver.")
    if not isinstance(subspace, ParameterSubspace):
        raise TypeError("subspace must be a ParameterSubspace.")
    plan_ = LocalCurvaturePlan() if plan is None else plan
    if not isinstance(plan_, LocalCurvaturePlan):
        raise TypeError("plan must be a LocalCurvaturePlan or None.")
    subspace.validate_root(solver.functions)
    if subspace.total_dimension > plan_.max_parameters:
        raise ValueError(
            f"Local curvature dimension {subspace.total_dimension} exceeds "
            f"max_parameters={plan_.max_parameters}."
        )
    position = subspace.pack()

    def objective(vector):
        functions = subspace.reconstruct_vector(vector)
        bound = eqx.tree_at(lambda value: value.functions, solver, functions)
        return bound.loss(key=key)

    loss, gradient = jax.value_and_grad(objective)(position)
    hessian = jax.hessian(objective)(position)
    hessian = 0.5 * (hessian + hessian.T)
    matrix = hessian + plan_.damping * jnp.eye(
        subspace.total_dimension,
        dtype=hessian.dtype,
    )
    space = ArraySpace((subspace.total_dimension,), dtype=matrix.dtype)
    operator = DenseLinearOperator(
        matrix,
        source=space,
        target=space,
        operator_id="functional-local-curvature",
    )
    linear = solve(
        LinearSystem(operator, problem_id="functional-local-curvature-step"),
        -gradient,
    )
    step = jnp.asarray(linear.value)
    candidate_vector = position + step
    candidate_functions = subspace.reconstruct_vector(candidate_vector)
    candidate_solver = eqx.tree_at(
        lambda value: value.functions,
        solver,
        candidate_functions,
    )
    candidate_loss = candidate_solver.loss(key=key)
    finite = bool(
        jax.device_get(
            jnp.isfinite(candidate_loss) & jnp.all(jnp.isfinite(candidate_vector))
        )
    )
    accepted = finite and float(candidate_loss) <= float(loss)
    functions = candidate_functions if accepted else solver.functions
    return LocalCurvatureResult(
        functions,
        loss=loss,
        gradient_norm=jnp.linalg.norm(gradient),
        step_norm=jnp.linalg.norm(step),
        accepted=accepted,
    )


class DecompositionKFACResult(StrictModule):
    functions: frozendict[str, DomainFunction]
    patch_ids: tuple[str, ...] = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    diagnostics: Any

    def __init__(
        self,
        functions: Mapping[str, DomainFunction],
        patch_ids: tuple[str, ...],
        diagnostics: Any,
        /,
    ):
        self.functions = frozendict(functions)
        self.patch_ids = tuple(patch_ids)
        self.approximation = (
            "local-block-kfac" if len(self.patch_ids) == 1 else "overlap-coupled-kfac"
        )
        self.diagnostics = diagnostics


def solve_decomposition_kfac(
    prepared,
    patch_ids,
    /,
    *,
    num_iter: int,
    optimizer=None,
    seed: int = 0,
    jit: bool = False,
) -> DecompositionKFACResult:
    """Train one local or overlap patch block with native functional KFAC."""
    from ...optim import kfac
    from .._functional_correction import freeze_domain_function
    from ._prepare import PreparedFunctionalDecomposition
    from ._problem import GlobalScope, PairScope, PatchScope

    if not isinstance(prepared, PreparedFunctionalDecomposition):
        raise TypeError("prepared must be a PreparedFunctionalDecomposition.")
    if prepared.problem.assembly != "broken":
        raise ValueError("Local KFAC currently requires explicit broken local fields.")
    selected = tuple(str(value) for value in patch_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("patch_ids must contain distinct local patches.")
    known = set(prepared.problem.cover.patch_ids)
    if not set(selected).issubset(known):
        raise ValueError("Local KFAC references an unknown patch.")
    functions = {}
    selected_names = {
        prepared.problem.family.field_name(patch_id) for patch_id in selected
    }
    for name, field in prepared.solver.functions.items():
        functions[name] = (
            field if name in selected_names else freeze_domain_function(field)
        )
    terms = []
    for scoped in prepared.problem.terms:
        if isinstance(scoped.scope, GlobalScope):
            terms.append(scoped.term)
        elif isinstance(scoped.scope, PatchScope):
            if scoped.scope.patch_id in selected:
                terms.append(scoped.term)
        elif isinstance(scoped.scope, PairScope):
            pairing = prepared.problem.cover.pairing(scoped.scope.pairing_id)
            if pairing.left_patch_id in selected or pairing.right_patch_id in selected:
                terms.append(scoped.term)
    if not terms:
        raise ValueError("Selected KFAC block has no incident residual terms.")
    local_solver = FunctionalSolver(functions=functions, terms=tuple(terms))
    trained = local_solver.solve(
        num_iter=int(num_iter),
        optim=kfac() if optimizer is None else optimizer,
        seed=seed,
        jit=jit,
        keep_best=False,
        log_every=0,
    )
    merged = dict(prepared.solver.functions)
    for name in selected_names:
        merged[name] = trained.functions[name]
    return DecompositionKFACResult(
        merged,
        selected,
        trained.training_diagnostics,
    )


def solve_local_kfac(prepared, patch_id: str, /, **kwargs) -> DecompositionKFACResult:
    return solve_decomposition_kfac(prepared, (patch_id,), **kwargs)


def solve_overlap_kfac(
    prepared,
    patch_ids: Sequence[str],
    /,
    **kwargs,
) -> DecompositionKFACResult:
    if len(tuple(patch_ids)) < 2:
        raise ValueError("Overlap KFAC requires at least two patch IDs.")
    return solve_decomposition_kfac(prepared, tuple(patch_ids), **kwargs)


class MatrixFreeGaussNewtonPlan(StrictModule):
    damping: float = eqx.field(static=True)

    def __init__(self, *, damping: float = 1.0e-4):
        damping_ = float(damping)
        if not math.isfinite(damping_) or damping_ <= 0.0:
            raise ValueError("damping must be finite and positive.")
        self.damping = damping_


class MatrixFreeGaussNewtonResult(StrictModule):
    functions: frozendict[str, DomainFunction]
    initial_loss: Array
    final_loss: Array
    accepted: bool = eqx.field(static=True)
    matrix_free: bool = eqx.field(static=True)
    linear_result: Any

    def __init__(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        initial_loss: Array,
        final_loss: Array,
        accepted: bool,
        linear_result: Any,
    ):
        self.functions = frozendict(functions)
        self.initial_loss = jnp.asarray(initial_loss).reshape(())
        self.final_loss = jnp.asarray(final_loss).reshape(())
        self.accepted = bool(accepted)
        self.matrix_free = True
        self.linear_result = linear_result


def matrix_free_gauss_newton_step(
    solver: FunctionalSolver,
    subspace: ParameterSubspace,
    plan: MatrixFreeGaussNewtonPlan | None = None,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> MatrixFreeGaussNewtonResult:
    """Take one damped Gauss-Newton step without assembling a Jacobian or Hessian."""
    if not isinstance(solver, FunctionalSolver):
        raise TypeError("solver must be a FunctionalSolver.")
    if not isinstance(subspace, ParameterSubspace):
        raise TypeError("subspace must be a ParameterSubspace.")
    plan_ = MatrixFreeGaussNewtonPlan() if plan is None else plan
    if not isinstance(plan_, MatrixFreeGaussNewtonPlan):
        raise TypeError("plan must be a MatrixFreeGaussNewtonPlan or None.")
    subspace.validate_root(solver.functions)
    prepared = solver.objective.prepare_training(
        range(len(solver.terms)),
        scale=1.0,
        evaluation_key=key,
        sampling_key=key,
        iteration=0,
        evaluation_kwargs={},
    )
    full_params, non_trainable = partition_trainable(solver.functions)
    residual = prepare_functional_residual(
        prepared,
        full_params,
        non_trainable,
        solver.enforcement,
        require_all=True,
    )
    position = subspace.pack()

    def roots(vector):
        functions = subspace.reconstruct_vector(vector)
        params, _ = partition_trainable(functions)
        return residual.roots(params)

    residual_value, transpose = jax.vjp(roots, position)
    right_hand_side = -transpose(residual_value)[0]

    def normal_action(direction):
        _, tangent = jax.jvp(roots, (position,), (direction,))
        _, pullback = jax.vjp(roots, position)
        return pullback(tangent)[0] + plan_.damping * direction

    space = ArraySpace((subspace.total_dimension,), dtype=position.dtype)
    properties = OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )
    operator = FunctionLinearOperator(
        normal_action,
        source=space,
        target=space,
        transpose_action=normal_action,
        properties=properties,
        operator_id="functional-matrix-free-gauss-newton",
        closure_convert=False,
    )
    linear_result = solve(
        LinearSystem(operator, problem_id="functional-matrix-free-gauss-newton-step"),
        right_hand_side,
        policy=LinearSolvePolicy(ConjugateGradient()),
    )
    candidate_vector = position + jnp.asarray(linear_result.value)
    initial_loss = jnp.real(jnp.vdot(residual_value, residual_value))
    candidate_residual = roots(candidate_vector)
    final_loss = jnp.real(jnp.vdot(candidate_residual, candidate_residual))
    accepted = bool(
        jax.device_get(
            jnp.isfinite(final_loss)
            & jnp.all(jnp.isfinite(candidate_vector))
            & (final_loss <= initial_loss)
        )
    )
    functions = (
        subspace.reconstruct_vector(candidate_vector) if accepted else solver.functions
    )
    return MatrixFreeGaussNewtonResult(
        functions,
        initial_loss=initial_loss,
        final_loss=final_loss if accepted else initial_loss,
        accepted=accepted,
        linear_result=linear_result,
    )


__all__ = [
    "MatrixFreeGaussNewtonPlan",
    "MatrixFreeGaussNewtonResult",
    "DecompositionKFACResult",
    "LocalCurvaturePlan",
    "LocalCurvatureResult",
    "dense_local_curvature_step",
    "solve_decomposition_kfac",
    "matrix_free_gauss_newton_step",
    "solve_local_kfac",
    "solve_overlap_kfac",
]
