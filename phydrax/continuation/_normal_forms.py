#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import AbstractAttribute, StrictModule
from .._tree_math import tree_norm
from ..linalg import AbstractVectorSpace
from ._bifurcation import HopfState, NullspaceEvidence
from ._core import _execution_residual, ContinuationCurveProblem
from ._geometry import ContinuationGeometry


class _ExecutionCurve:
    def __init__(
        self,
        problem: ContinuationCurveProblem,
        geometry: ContinuationGeometry,
        args: Any,
        /,
    ):
        self.problem = problem
        self.geometry = geometry
        self.args = args
        self.problem_id = problem.problem_id

    def residual(self, state, parameter, args=None, /):
        del args
        return _execution_residual(
            self.problem,
            self.geometry,
            state,
            parameter,
            self.args,
        )


class NormalFormStatus(IntEnum):
    """Terminal status of a local normal-form coefficient calculation."""

    SUCCESS = 0
    SPECTRAL_EVIDENCE_INVALID = 1
    LINEAR_SOLVE_FAILED = 2
    LINEAR_SOLVE_RESIDUAL_TOO_LARGE = 3
    ILL_CONDITIONED = 4
    NONFINITE = 5


class NormalFormPolicy(StrictModule):
    """Conditioning and residual thresholds for coefficient calculations."""

    linear_residual_tolerance: float = eqx.field(static=True)
    spectral_residual_tolerance: float = eqx.field(static=True)
    overlap_tolerance: float = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_residual_tolerance: float = 1e-7,
        spectral_residual_tolerance: float = 1e-7,
        overlap_tolerance: float = 1e-8,
        maximum_condition: float = 1e8,
    ):
        residual = float(linear_residual_tolerance)
        spectral_residual = float(spectral_residual_tolerance)
        overlap = float(overlap_tolerance)
        condition = float(maximum_condition)
        if not isfinite(residual) or residual < 0.0:
            raise ValueError("linear_residual_tolerance must be finite and non-negative.")
        if not isfinite(spectral_residual) or spectral_residual < 0.0:
            raise ValueError(
                "spectral_residual_tolerance must be finite and non-negative."
            )
        if not isfinite(overlap) or overlap < 0.0:
            raise ValueError("overlap_tolerance must be finite and non-negative.")
        if not isfinite(condition) or condition < 1.0:
            raise ValueError("maximum_condition must be finite and at least one.")
        self.linear_residual_tolerance = residual
        self.spectral_residual_tolerance = spectral_residual
        self.overlap_tolerance = overlap
        self.maximum_condition = condition


class NormalFormLinearSolveResult(StrictModule):
    """One correction solve supplied by an external linear solver hook."""

    solution: PyTree[Array]
    residual_norm: Array
    condition_estimate: Array
    iterations: Array
    successful: Array
    source_status: Array
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        solution: PyTree[Any],
        residual_norm: Any,
        condition_estimate: Any,
        iterations: Any,
        successful: Any,
        source_status: Any,
        solver_id: str,
    ):
        identifier = str(solver_id)
        if not identifier:
            raise ValueError("solver_id must be non-empty.")
        self.solution = solution
        self.residual_norm = jnp.asarray(residual_norm)
        self.condition_estimate = jnp.asarray(condition_estimate)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.successful = jnp.asarray(successful, dtype=bool)
        self.source_status = jnp.asarray(source_status, dtype=jnp.int32)
        self.solver_id = identifier


class AbstractNormalFormLinearSolver(StrictModule):
    """Hook for range and shifted solves used by normal-form formulas."""

    solver_id: AbstractAttribute[str]

    @abc.abstractmethod
    def solve(
        self,
        action: Callable[[PyTree[Any]], PyTree[Any]],
        right_hand_side: PyTree[Any],
        /,
        *,
        system_id: str,
    ) -> NormalFormLinearSolveResult:
        raise NotImplementedError


class CallableNormalFormLinearSolver(AbstractNormalFormLinearSolver):
    """Adapter around a project linear solve or external structured solver."""

    function: Callable[
        [Callable[[PyTree[Any]], PyTree[Any]], PyTree[Any], str],
        NormalFormLinearSolveResult,
    ]
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[
            [Callable[[PyTree[Any]], PyTree[Any]], PyTree[Any], str],
            NormalFormLinearSolveResult,
        ],
        /,
        *,
        solver_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        identifier = str(solver_id)
        if not identifier:
            raise ValueError("solver_id must be non-empty.")
        self.function = function
        self.solver_id = identifier

    def solve(
        self,
        action: Callable[[PyTree[Any]], PyTree[Any]],
        right_hand_side: PyTree[Any],
        /,
        *,
        system_id: str,
    ) -> NormalFormLinearSolveResult:
        result = self.function(action, right_hand_side, system_id)
        if not isinstance(result, NormalFormLinearSolveResult):
            raise TypeError(
                "A normal-form linear solver must return NormalFormLinearSolveResult."
            )
        return result


class NormalFormDiagnostics(StrictModule):
    """Conditioning, residual, and derivative-work evidence."""

    mode_overlap: Array
    eigenvalue_condition: Array
    spectral_residuals: Array
    linear_residuals: Array
    linear_condition_estimates: Array
    derivative_evaluations: Array
    finite: Array

    def __init__(
        self,
        *,
        mode_overlap: Any,
        eigenvalue_condition: Any,
        spectral_residuals: Any,
        linear_residuals: Any,
        linear_condition_estimates: Any,
        derivative_evaluations: Any,
        finite: Any,
    ):
        spectral_residuals_ = jnp.asarray(spectral_residuals)
        residuals = jnp.asarray(linear_residuals)
        conditions = jnp.asarray(linear_condition_estimates)
        if residuals.ndim != 1 or conditions.ndim != 1:
            raise ValueError(
                "linear residuals and condition estimates must be rank-one arrays."
            )
        if spectral_residuals_.ndim != 1:
            raise ValueError("spectral residuals must be a rank-one array.")
        if residuals.shape != conditions.shape:
            raise ValueError(
                "linear residuals and condition estimates must have matching shapes."
            )
        self.mode_overlap = jnp.asarray(mode_overlap)
        self.spectral_residuals = spectral_residuals_
        self.eigenvalue_condition = jnp.asarray(eigenvalue_condition)
        self.linear_residuals = residuals
        self.linear_condition_estimates = conditions
        self.derivative_evaluations = jnp.asarray(
            derivative_evaluations,
            dtype=jnp.int32,
        )
        self.finite = jnp.asarray(finite, dtype=bool)


class NormalFormProvenance(StrictModule):
    """Problem, formula, derivative, and delegated solver identities."""

    problem_id: str = eqx.field(static=True)
    formula_id: str = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)
    linear_solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        formula_id: str,
        derivative_id: str = "jax-nested-jvp",
        linear_solver_id: str = "",
    ):
        identifiers = tuple(
            str(value) for value in (problem_id, formula_id, derivative_id)
        )
        if any(not value for value in identifiers):
            raise ValueError("Normal-form provenance identifiers must be non-empty.")
        self.problem_id, self.formula_id, self.derivative_id = identifiers
        self.linear_solver_id = str(linear_solver_id)


class FoldNormalFormResult(StrictModule):
    """Quadratic fold coefficient in normalized right/left coordinates."""

    coefficient: Array
    diagnostics: NormalFormDiagnostics
    status: Array
    provenance: NormalFormProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(NormalFormStatus.SUCCESS)


class PitchforkNormalFormResult(StrictModule):
    """Lyapunov--Schmidt quadratic and cubic pitchfork coefficients."""

    quadratic_coefficient: Array
    cubic_coefficient: Array
    second_order_correction: PyTree[Array]
    range_solve: NormalFormLinearSolveResult
    diagnostics: NormalFormDiagnostics
    status: Array
    provenance: NormalFormProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(NormalFormStatus.SUCCESS)


class TranscriticalNormalFormResult(StrictModule):
    """Reduced transcritical coefficients and reference-branch tangent evidence."""

    quadratic_coefficient: Array
    mixed_coefficient: Array
    tangent_separation: Array
    reference_tangent: PyTree[Array]
    reference_solve: NormalFormLinearSolveResult
    diagnostics: NormalFormDiagnostics
    status: Array
    provenance: NormalFormProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(NormalFormStatus.SUCCESS)


class HopfNormalFormResult(StrictModule):
    """First Lyapunov coefficient and harmonic correction evidence."""

    first_lyapunov_coefficient: Array
    g21: Array
    zero_harmonic_correction: PyTree[Array]
    second_harmonic_correction: PyTree[Array]
    zero_harmonic_solve: NormalFormLinearSolveResult
    second_harmonic_solve: NormalFormLinearSolveResult
    diagnostics: NormalFormDiagnostics
    status: Array
    provenance: NormalFormProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(NormalFormStatus.SUCCESS)


def _tree_scale(scale: Any, tree: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(lambda value: scale * value, tree)


def _tree_add(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _tree_add_many(*trees: PyTree[Any]) -> PyTree[Array]:
    result = trees[0]
    for tree in trees[1:]:
        result = _tree_add(result, tree)
    return result


def _tree_real(tree: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(jnp.real, tree)


def _tree_imaginary(tree: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(jnp.imag, tree)


def _tree_real_like(
    tree: PyTree[Any],
    reference: PyTree[Any],
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda value, target: jnp.asarray(
            jnp.real(value),
            dtype=jnp.asarray(target).dtype,
        ),
        tree,
        reference,
    )


def _tree_imaginary_like(
    tree: PyTree[Any],
    reference: PyTree[Any],
    /,
) -> PyTree[Array]:
    return jax.tree.map(
        lambda value, target: jnp.asarray(
            jnp.imag(value),
            dtype=jnp.asarray(target).dtype,
        ),
        tree,
        reference,
    )


def _tree_complex(real: PyTree[Any], imaginary: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + 1j * y, real, imaginary)


def _tree_conjugate(tree: PyTree[Any], /) -> PyTree[Array]:
    return jax.tree.map(jnp.conj, tree)


def _complex_inner(
    space: AbstractVectorSpace,
    left: PyTree[Any],
    right: PyTree[Any],
    /,
) -> Array:
    left_real = _tree_real(left)
    left_imaginary = _tree_imaginary(left)
    right_real = _tree_real(right)
    right_imaginary = _tree_imaginary(right)
    real = space.inner(left_real, right_real) + space.inner(
        left_imaginary, right_imaginary
    )
    imaginary = space.inner(left_real, right_imaginary) - space.inner(
        left_imaginary, right_real
    )
    return real + 1j * imaginary


def _complex_norm(space: AbstractVectorSpace, vector: PyTree[Any], /) -> Array:
    return jnp.sqrt(jnp.maximum(jnp.real(_complex_inner(space, vector, vector)), 0.0))


def _linear_action(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    direction: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    direction_real = _tree_real_like(direction, state)
    direction_imaginary = _tree_imaginary_like(direction, state)
    residual = lambda value: problem.residual(value, parameter, args)
    real_action = jax.jvp(residual, (state,), (direction_real,))[1]
    imaginary_action = jax.jvp(residual, (state,), (direction_imaginary,))[1]
    return _tree_complex(real_action, imaginary_action)


def _adjoint_action_real(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    state_space: AbstractVectorSpace,
    direction: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    residual = lambda value: problem.residual(value, parameter, args)
    _, pullback = jax.vjp(residual, state)
    direction_ = _tree_real_like(direction, state)
    covector = state_space.riesz(direction_)
    return state_space.inverse_riesz(pullback(covector)[0])


def _bilinear_real(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    left: PyTree[Any],
    right: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    def first(value):
        return jax.jvp(
            lambda inner: problem.residual(inner, parameter, args),
            (value,),
            (left,),
        )[1]

    return jax.jvp(first, (state,), (right,))[1]


def _bilinear(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    left: PyTree[Any],
    right: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    left_real = _tree_real_like(left, state)
    left_imaginary = _tree_imaginary_like(left, state)
    right_real = _tree_real_like(right, state)
    right_imaginary = _tree_imaginary_like(right, state)
    real = _tree_add(
        _bilinear_real(problem, state, parameter, left_real, right_real, args),
        _tree_scale(
            -1.0,
            _bilinear_real(
                problem,
                state,
                parameter,
                left_imaginary,
                right_imaginary,
                args,
            ),
        ),
    )
    imaginary = _tree_add(
        _bilinear_real(problem, state, parameter, left_real, right_imaginary, args),
        _bilinear_real(problem, state, parameter, left_imaginary, right_real, args),
    )
    return _tree_complex(real, imaginary)


def _trilinear_real(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    first_direction: PyTree[Any],
    second_direction: PyTree[Any],
    third_direction: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    def first(value):
        def second(inner):
            return jax.jvp(
                lambda deepest: problem.residual(deepest, parameter, args),
                (inner,),
                (first_direction,),
            )[1]

        return jax.jvp(second, (value,), (second_direction,))[1]

    return jax.jvp(first, (state,), (third_direction,))[1]


def _trilinear(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Array,
    first: PyTree[Any],
    second: PyTree[Any],
    third: PyTree[Any],
    args: Any,
    /,
) -> PyTree[Array]:
    directions = (
        (_tree_real_like(first, state), _tree_imaginary_like(first, state)),
        (_tree_real_like(second, state), _tree_imaginary_like(second, state)),
        (_tree_real_like(third, state), _tree_imaginary_like(third, state)),
    )
    real_terms: list[PyTree[Array]] = []
    imaginary_terms: list[PyTree[Array]] = []
    for first_imaginary in (0, 1):
        for second_imaginary in (0, 1):
            for third_imaginary in (0, 1):
                imaginary_count = first_imaginary + second_imaginary + third_imaginary
                term = _trilinear_real(
                    problem,
                    state,
                    parameter,
                    directions[0][first_imaginary],
                    directions[1][second_imaginary],
                    directions[2][third_imaginary],
                    args,
                )
                coefficient = (1j) ** imaginary_count
                if float(jnp.real(coefficient)) != 0.0:
                    real_terms.append(_tree_scale(jnp.real(coefficient), term))
                if float(jnp.imag(coefficient)) != 0.0:
                    imaginary_terms.append(_tree_scale(jnp.imag(coefficient), term))
    real = real_terms[0]
    for term in real_terms[1:]:
        real = _tree_add(real, term)
    imaginary = imaginary_terms[0]
    for term in imaginary_terms[1:]:
        imaginary = _tree_add(imaginary, term)
    return _tree_complex(real, imaginary)


def _verified_nullspace_status(
    evidence: NullspaceEvidence,
    policy: NormalFormPolicy,
    coefficient: Array,
    /,
) -> tuple[Array, Array]:
    overlap = jnp.abs(evidence.left_right_pairing)
    finite = (
        evidence.source_success
        & jnp.isfinite(overlap)
        & jnp.isfinite(evidence.eigenvalue_condition)
        & jnp.isfinite(coefficient)
        & jnp.isfinite(evidence.right_residual_norm)
        & jnp.isfinite(evidence.left_residual_norm)
    )
    status = jnp.where(
        finite,
        int(NormalFormStatus.SUCCESS),
        int(NormalFormStatus.NONFINITE),
    )
    status = jnp.where(
        ~evidence.source_success,
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        (evidence.right_residual_norm > policy.spectral_residual_tolerance)
        | (evidence.left_residual_norm > policy.spectral_residual_tolerance),
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & (
            (overlap <= policy.overlap_tolerance)
            | (evidence.eigenvalue_condition > policy.maximum_condition)
        ),
        int(NormalFormStatus.ILL_CONDITIONED),
        status,
    )
    return status, finite


def fold_normal_form(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Any,
    geometry: ContinuationGeometry,
    nullspace: NullspaceEvidence,
    /,
    *,
    policy: NormalFormPolicy | None = None,
    args: Any = None,
) -> FoldNormalFormResult:
    """Compute the normalized quadratic coefficient of a fold."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(geometry, ContinuationGeometry):
        raise TypeError("geometry must be a ContinuationGeometry.")
    if not isinstance(nullspace, NullspaceEvidence):
        raise TypeError("nullspace must be NullspaceEvidence.")
    policy_ = NormalFormPolicy() if policy is None else policy
    if not isinstance(policy_, NormalFormPolicy):
        raise TypeError("policy must be NormalFormPolicy or None.")
    state_ = geometry.state_to_execution(state)
    parameter_ = jnp.asarray(parameter)
    right = geometry.state_tangent_to_execution(
        state,
        nullspace.right_nullvector,
    )
    left = geometry.residual_to_execution(nullspace.left_nullvector)
    pairing = jnp.vdot(
        geometry.execution_residual_space.flatten(left),
        geometry.execution_state_space.flatten(right),
    )
    normalized_left = _tree_scale(1.0 / jnp.conj(pairing), left)
    execution_problem = _ExecutionCurve(problem, geometry, args)
    curvature = _bilinear_real(
        execution_problem,
        state_,
        parameter_,
        right,
        right,
        None,
    )
    coefficient = 0.5 * geometry.execution_residual_space.inner(
        normalized_left,
        curvature,
    )
    status, finite = _verified_nullspace_status(nullspace, policy_, coefficient)
    diagnostics = NormalFormDiagnostics(
        mode_overlap=jnp.abs(pairing),
        eigenvalue_condition=nullspace.eigenvalue_condition,
        spectral_residuals=jnp.asarray(
            [nullspace.right_residual_norm, nullspace.left_residual_norm]
        ),
        linear_residuals=jnp.empty((0,), dtype=jnp.real(coefficient).dtype),
        linear_condition_estimates=jnp.empty(
            (0,),
            dtype=jnp.real(coefficient).dtype,
        ),
        derivative_evaluations=2,
        finite=finite,
    )
    return FoldNormalFormResult(
        coefficient=coefficient,
        diagnostics=diagnostics,
        status=status,
        provenance=NormalFormProvenance(
            problem_id=problem.problem_id,
            formula_id="fold-quadratic",
        ),
    )


def pitchfork_normal_form(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Any,
    geometry: ContinuationGeometry,
    nullspace: NullspaceEvidence,
    linear_solver: AbstractNormalFormLinearSolver,
    /,
    *,
    policy: NormalFormPolicy | None = None,
    args: Any = None,
) -> PitchforkNormalFormResult:
    """Compute Lyapunov--Schmidt coefficients using an external range solve."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(geometry, ContinuationGeometry):
        raise TypeError("geometry must be a ContinuationGeometry.")
    if not isinstance(nullspace, NullspaceEvidence):
        raise TypeError("nullspace must be NullspaceEvidence.")
    if not isinstance(linear_solver, AbstractNormalFormLinearSolver):
        raise TypeError("linear_solver must be an AbstractNormalFormLinearSolver.")
    policy_ = NormalFormPolicy() if policy is None else policy
    if not isinstance(policy_, NormalFormPolicy):
        raise TypeError("policy must be NormalFormPolicy or None.")
    state_ = geometry.state_to_execution(state)
    parameter_ = jnp.asarray(parameter)
    right = geometry.state_tangent_to_execution(
        state,
        nullspace.right_nullvector,
    )
    left = geometry.residual_to_execution(nullspace.left_nullvector)
    pairing = jnp.vdot(
        geometry.execution_residual_space.flatten(left),
        geometry.execution_state_space.flatten(right),
    )
    normalized_left = _tree_scale(1.0 / jnp.conj(pairing), left)
    execution_problem = _ExecutionCurve(problem, geometry, args)
    quadratic_vector = _bilinear_real(
        execution_problem,
        state_,
        parameter_,
        right,
        right,
        None,
    )
    quadratic = 0.5 * geometry.execution_residual_space.inner(
        normalized_left,
        quadratic_vector,
    )
    action = lambda direction: _tree_real(
        _linear_action(execution_problem, state_, parameter_, direction, None)
    )
    range_result = linear_solver.solve(
        action,
        _tree_scale(-1.0, quadratic_vector),
        system_id=f"{problem.problem_id}/pitchfork-range",
    )
    actual_residual = geometry.residual_norm(
        _tree_add(action(range_result.solution), quadratic_vector)
    )
    solve_residual = jnp.maximum(range_result.residual_norm, actual_residual)
    cubic_vector = _trilinear_real(
        execution_problem,
        state_,
        parameter_,
        right,
        right,
        right,
        None,
    )
    interaction = _bilinear_real(
        execution_problem,
        state_,
        parameter_,
        right,
        range_result.solution,
        None,
    )
    cubic = geometry.execution_residual_space.inner(
        normalized_left,
        _tree_add(
            _tree_scale(1.0 / 6.0, cubic_vector),
            _tree_scale(0.5, interaction),
        ),
    )
    finite = (
        jnp.isfinite(quadratic)
        & jnp.isfinite(cubic)
        & jnp.isfinite(solve_residual)
        & jnp.isfinite(range_result.condition_estimate)
    )
    status = jnp.where(
        finite,
        int(NormalFormStatus.SUCCESS),
        int(NormalFormStatus.NONFINITE),
    )
    status = jnp.where(
        ~nullspace.source_success,
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        (nullspace.right_residual_norm > policy_.spectral_residual_tolerance)
        | (nullspace.left_residual_norm > policy_.spectral_residual_tolerance),
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        ~range_result.successful,
        int(NormalFormStatus.LINEAR_SOLVE_FAILED),
        status,
    )
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & (solve_residual > policy_.linear_residual_tolerance),
        int(NormalFormStatus.LINEAR_SOLVE_RESIDUAL_TOO_LARGE),
        status,
    )
    maximum_condition = jnp.maximum(
        nullspace.eigenvalue_condition,
        range_result.condition_estimate,
    )
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & (
            (jnp.abs(pairing) <= policy_.overlap_tolerance)
            | (maximum_condition > policy_.maximum_condition)
        ),
        int(NormalFormStatus.ILL_CONDITIONED),
        status,
    )
    diagnostics = NormalFormDiagnostics(
        mode_overlap=jnp.abs(pairing),
        spectral_residuals=jnp.asarray(
            [nullspace.right_residual_norm, nullspace.left_residual_norm]
        ),
        eigenvalue_condition=nullspace.eigenvalue_condition,
        linear_residuals=jnp.asarray([solve_residual]),
        linear_condition_estimates=jnp.asarray([range_result.condition_estimate]),
        derivative_evaluations=5,
        finite=finite,
    )
    return PitchforkNormalFormResult(
        quadratic_coefficient=quadratic,
        cubic_coefficient=cubic,
        second_order_correction=geometry.state_tangent_from_execution(
            state_,
            range_result.solution,
        ),
        range_solve=range_result,
        diagnostics=diagnostics,
        status=status,
        provenance=NormalFormProvenance(
            problem_id=problem.problem_id,
            formula_id="pitchfork-lyapunov-schmidt",
            linear_solver_id=linear_solver.solver_id,
        ),
    )


def transcritical_normal_form(
    problem: ContinuationCurveProblem,
    state: PyTree[Any],
    parameter: Any,
    geometry: ContinuationGeometry,
    nullspace: NullspaceEvidence,
    linear_solver: AbstractNormalFormLinearSolver,
    /,
    *,
    policy: NormalFormPolicy | None = None,
    args: Any = None,
) -> TranscriticalNormalFormResult:
    """Compute the two reduced coefficients of a transcritical intersection."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(geometry, ContinuationGeometry):
        raise TypeError("geometry must be a ContinuationGeometry.")
    if not isinstance(nullspace, NullspaceEvidence):
        raise TypeError("nullspace must be NullspaceEvidence.")
    if not isinstance(linear_solver, AbstractNormalFormLinearSolver):
        raise TypeError("linear_solver must be an AbstractNormalFormLinearSolver.")
    policy_ = NormalFormPolicy() if policy is None else policy
    if not isinstance(policy_, NormalFormPolicy):
        raise TypeError("policy must be NormalFormPolicy or None.")
    state_ = geometry.state_to_execution(state)
    parameter_ = jnp.asarray(parameter)
    right = geometry.state_tangent_to_execution(
        state,
        nullspace.right_nullvector,
    )
    left = geometry.residual_to_execution(nullspace.left_nullvector)
    pairing = jnp.vdot(
        geometry.execution_residual_space.flatten(left),
        geometry.execution_state_space.flatten(right),
    )
    normalized_left = _tree_scale(1.0 / jnp.conj(pairing), left)
    execution_problem = _ExecutionCurve(problem, geometry, args)
    action = lambda direction: _tree_real(
        _linear_action(execution_problem, state_, parameter_, direction, None)
    )
    parameter_derivative = jax.jvp(
        lambda value: execution_problem.residual(state_, value, None),
        (parameter_,),
        (jnp.ones_like(parameter_),),
    )[1]
    reference_result = linear_solver.solve(
        action,
        _tree_scale(-1.0, parameter_derivative),
        system_id=f"{problem.problem_id}/transcritical-reference-tangent",
    )
    actual_residual = geometry.residual_norm(
        _tree_add(action(reference_result.solution), parameter_derivative)
    )
    solve_residual = jnp.maximum(reference_result.residual_norm, actual_residual)
    quadratic_vector = _bilinear_real(
        execution_problem,
        state_,
        parameter_,
        right,
        right,
        None,
    )
    quadratic = 0.5 * geometry.execution_residual_space.inner(
        normalized_left,
        quadratic_vector,
    )

    def critical_action(parameter_value):
        return _tree_real(
            _linear_action(
                execution_problem,
                state_,
                parameter_value,
                right,
                None,
            )
        )

    direct_mixed = jax.jvp(
        critical_action,
        (parameter_,),
        (jnp.ones_like(parameter_),),
    )[1]
    branch_interaction = _bilinear_real(
        execution_problem,
        state_,
        parameter_,
        right,
        reference_result.solution,
        None,
    )
    mixed_vector = _tree_add(direct_mixed, branch_interaction)
    mixed = geometry.execution_residual_space.inner(
        normalized_left,
        mixed_vector,
    )
    separation = jnp.abs(mixed / quadratic)
    finite = (
        jnp.isfinite(quadratic)
        & jnp.isfinite(mixed)
        & jnp.isfinite(separation)
        & jnp.isfinite(solve_residual)
        & jnp.isfinite(reference_result.condition_estimate)
    )
    status = jnp.where(
        finite,
        int(NormalFormStatus.SUCCESS),
        int(NormalFormStatus.NONFINITE),
    )
    status = jnp.where(
        ~nullspace.source_success,
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        (nullspace.right_residual_norm > policy_.spectral_residual_tolerance)
        | (nullspace.left_residual_norm > policy_.spectral_residual_tolerance),
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        ~reference_result.successful,
        int(NormalFormStatus.LINEAR_SOLVE_FAILED),
        status,
    )
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & (solve_residual > policy_.linear_residual_tolerance),
        int(NormalFormStatus.LINEAR_SOLVE_RESIDUAL_TOO_LARGE),
        status,
    )
    maximum_condition = jnp.maximum(
        nullspace.eigenvalue_condition,
        reference_result.condition_estimate,
    )
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & (
            (jnp.abs(pairing) <= policy_.overlap_tolerance)
            | (maximum_condition > policy_.maximum_condition)
        ),
        int(NormalFormStatus.ILL_CONDITIONED),
        status,
    )
    diagnostics = NormalFormDiagnostics(
        mode_overlap=jnp.abs(pairing),
        spectral_residuals=jnp.asarray(
            [nullspace.right_residual_norm, nullspace.left_residual_norm]
        ),
        eigenvalue_condition=nullspace.eigenvalue_condition,
        linear_residuals=jnp.asarray([solve_residual]),
        linear_condition_estimates=jnp.asarray([reference_result.condition_estimate]),
        derivative_evaluations=6,
        finite=finite,
    )
    return TranscriticalNormalFormResult(
        quadratic_coefficient=quadratic,
        mixed_coefficient=mixed,
        tangent_separation=separation,
        reference_tangent=geometry.state_tangent_from_execution(
            state_,
            reference_result.solution,
        ),
        reference_solve=reference_result,
        diagnostics=diagnostics,
        status=status,
        provenance=NormalFormProvenance(
            problem_id=problem.problem_id,
            formula_id="transcritical-lyapunov-schmidt",
            linear_solver_id=linear_solver.solver_id,
        ),
    )


def hopf_first_lyapunov(
    problem: ContinuationCurveProblem,
    candidate: HopfState,
    geometry: ContinuationGeometry,
    adjoint_mode_real: PyTree[Any],
    adjoint_mode_imaginary: PyTree[Any],
    linear_solver: AbstractNormalFormLinearSolver,
    /,
    *,
    policy: NormalFormPolicy | None = None,
    args: Any = None,
) -> HopfNormalFormResult:
    """Compute the first Lyapunov coefficient with two explicit shifted solves."""
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(candidate, HopfState):
        raise TypeError("candidate must be a HopfState.")
    if not isinstance(geometry, ContinuationGeometry):
        raise TypeError("geometry must be a ContinuationGeometry.")
    if not geometry.execution_state_space.compatible(geometry.execution_residual_space):
        raise ValueError("Hopf normal forms require an execution-space endomorphism.")
    if geometry.representation.state_coordinates is not None:
        raise ValueError(
            "Hopf normal forms for mapped public states require a dedicated "
            "higher-complex representation."
        )
    if not isinstance(linear_solver, AbstractNormalFormLinearSolver):
        raise TypeError("linear_solver must be an AbstractNormalFormLinearSolver.")
    policy_ = NormalFormPolicy() if policy is None else policy
    if not isinstance(policy_, NormalFormPolicy):
        raise TypeError("policy must be NormalFormPolicy or None.")
    public_problem = problem
    public_state = candidate.physical_state
    state_space = geometry.execution_state_space
    state = geometry.state_to_execution(public_state)
    adjoint_real = geometry.residual_to_execution(adjoint_mode_real)
    adjoint_imaginary = geometry.residual_to_execution(adjoint_mode_imaginary)
    mode_real = geometry.state_tangent_to_execution(
        public_state,
        candidate.mode_real,
    )
    mode_imaginary = geometry.state_tangent_to_execution(
        public_state,
        candidate.mode_imaginary,
    )
    problem = _ExecutionCurve(public_problem, geometry, args)
    args = None
    mode = _tree_complex(mode_real, mode_imaginary)
    adjoint = _tree_complex(adjoint_real, adjoint_imaginary)
    overlap = _complex_inner(state_space, adjoint, mode)
    normalized_adjoint = _tree_scale(1.0 / jnp.conj(overlap), adjoint)
    mode_norm = _complex_norm(state_space, mode)
    adjoint_norm = _complex_norm(state_space, normalized_adjoint)
    eigenvalue_condition = mode_norm * adjoint_norm
    conjugate_mode = _tree_conjugate(mode)
    b_qq = _bilinear(
        problem,
        state,
        candidate.parameter,
        mode,
        mode,
        args,
    )
    b_q_conjugate = _bilinear(
        problem,
        state,
        candidate.parameter,
        mode,
        conjugate_mode,
        args,
    )
    jacobian_action = lambda direction: _linear_action(
        problem,
        state,
        candidate.parameter,
        direction,
        args,
    )
    equilibrium_residual = tree_norm(problem.residual(state, candidate.parameter, args))
    mode_residual = tree_norm(
        _tree_add(
            jacobian_action(mode),
            _tree_scale(-1j * candidate.frequency, mode),
        )
    )
    normalized_adjoint_real = _tree_real(normalized_adjoint)
    normalized_adjoint_imaginary = _tree_imaginary(normalized_adjoint)
    adjoint_action_real = _adjoint_action_real(
        problem,
        state,
        candidate.parameter,
        state_space,
        normalized_adjoint_real,
        args,
    )
    adjoint_action_imaginary = _adjoint_action_real(
        problem,
        state,
        candidate.parameter,
        state_space,
        normalized_adjoint_imaginary,
        args,
    )
    adjoint_residual = tree_norm(
        _tree_complex(
            _tree_add(
                adjoint_action_real,
                _tree_scale(
                    -candidate.frequency,
                    normalized_adjoint_imaginary,
                ),
            ),
            _tree_add(
                adjoint_action_imaginary,
                _tree_scale(candidate.frequency, normalized_adjoint_real),
            ),
        )
    )
    spectral_residuals = jnp.asarray(
        [equilibrium_residual, mode_residual, adjoint_residual]
    )
    zero_result = linear_solver.solve(
        jacobian_action,
        _tree_scale(-1.0, b_q_conjugate),
        system_id=f"{problem.problem_id}/hopf-zero-harmonic",
    )
    second_action = lambda direction: _tree_add(
        _tree_scale(2j * candidate.frequency, direction),
        _tree_scale(-1.0, jacobian_action(direction)),
    )
    second_result = linear_solver.solve(
        second_action,
        b_qq,
        system_id=f"{problem.problem_id}/hopf-second-harmonic",
    )
    zero_actual = tree_norm(
        _tree_add(jacobian_action(zero_result.solution), b_q_conjugate)
    )
    second_actual = tree_norm(
        _tree_add(
            second_action(second_result.solution),
            _tree_scale(-1.0, b_qq),
        )
    )
    zero_residual = jnp.maximum(zero_result.residual_norm, zero_actual)
    second_residual = jnp.maximum(second_result.residual_norm, second_actual)
    cubic = _trilinear(
        problem,
        state,
        candidate.parameter,
        mode,
        mode,
        conjugate_mode,
        args,
    )
    b_conjugate_h20 = _bilinear(
        problem,
        state,
        candidate.parameter,
        conjugate_mode,
        second_result.solution,
        args,
    )
    b_q_h11 = _bilinear(
        problem,
        state,
        candidate.parameter,
        mode,
        zero_result.solution,
        args,
    )
    g21_vector = _tree_add_many(
        cubic,
        b_conjugate_h20,
        _tree_scale(2.0, b_q_h11),
    )
    g21 = _complex_inner(state_space, normalized_adjoint, g21_vector)
    coefficient = jnp.real(g21) / (2.0 * candidate.frequency)
    conditions = jnp.asarray(
        [zero_result.condition_estimate, second_result.condition_estimate]
    )
    residuals = jnp.asarray([zero_residual, second_residual])
    finite = (
        jnp.isfinite(coefficient)
        & jnp.isfinite(g21)
        & jnp.isfinite(overlap)
        & jnp.isfinite(eigenvalue_condition)
        & jnp.all(jnp.isfinite(residuals))
        & jnp.all(jnp.isfinite(conditions))
        & jnp.all(jnp.isfinite(spectral_residuals))
    )
    both_successful = zero_result.successful & second_result.successful
    status = jnp.where(
        finite,
        int(NormalFormStatus.SUCCESS),
        int(NormalFormStatus.NONFINITE),
    )
    status = jnp.where(
        ~both_successful,
        int(NormalFormStatus.LINEAR_SOLVE_FAILED),
        status,
    )
    status = jnp.where(
        (candidate.frequency <= 0.0)
        | jnp.any(spectral_residuals > policy_.spectral_residual_tolerance),
        int(NormalFormStatus.SPECTRAL_EVIDENCE_INVALID),
        status,
    )
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & jnp.any(residuals > policy_.linear_residual_tolerance),
        int(NormalFormStatus.LINEAR_SOLVE_RESIDUAL_TOO_LARGE),
        status,
    )
    maximum_condition = jnp.maximum(eigenvalue_condition, jnp.max(conditions))
    status = jnp.where(
        (status == int(NormalFormStatus.SUCCESS))
        & (
            (jnp.abs(overlap) <= policy_.overlap_tolerance)
            | (maximum_condition > policy_.maximum_condition)
        ),
        int(NormalFormStatus.ILL_CONDITIONED),
        status,
    )
    diagnostics = NormalFormDiagnostics(
        mode_overlap=jnp.abs(overlap),
        eigenvalue_condition=eigenvalue_condition,
        spectral_residuals=spectral_residuals,
        linear_residuals=residuals,
        linear_condition_estimates=conditions,
        derivative_evaluations=22,
        finite=finite,
    )
    zero_real = jax.tree.map(
        lambda value, template: jnp.asarray(value, dtype=template.dtype),
        _tree_real(zero_result.solution),
        state,
    )
    zero_imaginary = jax.tree.map(
        lambda value, template: jnp.asarray(value, dtype=template.dtype),
        _tree_imaginary(zero_result.solution),
        state,
    )
    second_real = jax.tree.map(
        lambda value, template: jnp.asarray(value, dtype=template.dtype),
        _tree_real(second_result.solution),
        state,
    )
    second_imaginary = jax.tree.map(
        lambda value, template: jnp.asarray(value, dtype=template.dtype),
        _tree_imaginary(second_result.solution),
        state,
    )
    zero_public = _tree_complex(
        geometry.state_tangent_from_execution(state, zero_real),
        geometry.state_tangent_from_execution(state, zero_imaginary),
    )
    second_public = _tree_complex(
        geometry.state_tangent_from_execution(state, second_real),
        geometry.state_tangent_from_execution(state, second_imaginary),
    )
    return HopfNormalFormResult(
        first_lyapunov_coefficient=coefficient,
        g21=g21,
        zero_harmonic_correction=zero_public,
        second_harmonic_correction=second_public,
        zero_harmonic_solve=zero_result,
        second_harmonic_solve=second_result,
        diagnostics=diagnostics,
        status=status,
        provenance=NormalFormProvenance(
            problem_id=problem.problem_id,
            formula_id="hopf-first-lyapunov",
            linear_solver_id=linear_solver.solver_id,
        ),
    )


__all__ = [
    "AbstractNormalFormLinearSolver",
    "CallableNormalFormLinearSolver",
    "FoldNormalFormResult",
    "HopfNormalFormResult",
    "NormalFormDiagnostics",
    "NormalFormLinearSolveResult",
    "NormalFormProvenance",
    "NormalFormPolicy",
    "NormalFormStatus",
    "PitchforkNormalFormResult",
    "TranscriticalNormalFormResult",
    "fold_normal_form",
    "hopf_first_lyapunov",
    "pitchfork_normal_form",
    "transcritical_normal_form",
]
