#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from typing import Any, Literal, Sequence, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._operators import (
    AbstractLinearOperator,
    adjoint,
    DenseLinearOperator,
    IdentityLinearOperator,
)
from ._plans import LinearSolvePlan, plan as plan_linear_solve
from ._policies import FailurePolicy, LinearSolvePolicy
from ._prepared import PreparedLinearSolve
from ._problems import LinearSystem
from ._properties import OperatorCapabilities, OperatorProperties
from ._results import LinearSolveStatus
from ._runtime import (
    prepare as prepare_linear_solve,
    refresh as refresh_linear_solve,
    solve as solve_linear_system,
)
from ._spaces import _coordinate_dtype, AbstractVectorSpace, ArraySpace


MatrixEquationKind: TypeAlias = Literal[
    "generalized",
    "sylvester",
    "continuous-lyapunov",
    "discrete-lyapunov",
]


class MatrixEquationStatus(IntEnum):
    """Portable aggregate status for a linear matrix equation."""

    SUCCESS = 0
    LINEAR_SOLVE_FAILURE = 1
    NONFINITE = 2
    STRUCTURE_TOLERANCE_NOT_MET = 3


class MatrixEquationTerm(StrictModule):
    """One coordinate term ``coefficient * left @ X @ right``."""

    left: AbstractLinearOperator
    right: AbstractLinearOperator
    coefficient: Array

    def __init__(
        self,
        left: AbstractLinearOperator | ArrayLike,
        right: AbstractLinearOperator | ArrayLike,
        /,
        *,
        coefficient: ArrayLike = 1.0,
    ):
        left_ = _coerce_square_operator(left, "left")
        right_ = _coerce_square_operator(right, "right")
        if not left_.capabilities.transpose or not right_.capabilities.transpose:
            raise ValueError(
                "Matrix-equation terms require transpose-capable left and right operators."
            )
        coefficient_ = jnp.asarray(coefficient)
        if coefficient_.ndim != 0:
            raise ValueError("Matrix-equation coefficients must be scalar.")
        if not jnp.issubdtype(coefficient_.dtype, jnp.number) or jnp.issubdtype(
            coefficient_.dtype, jnp.bool_
        ):
            raise TypeError("Matrix-equation coefficients must be real or complex.")
        if not bool(jnp.isfinite(coefficient_)):
            raise ValueError("Matrix-equation coefficients must be finite.")
        self.left = left_
        self.right = right_
        self.coefficient = coefficient_

    @property
    def row_dimension(self) -> int:
        return self.left.source.size

    @property
    def column_dimension(self) -> int:
        return self.right.source.size


class MatrixEquationLinearOperator(AbstractLinearOperator):
    """Matrix-free operator ``X -> sum_i c_i A_i X B_i`` in row-major coordinates."""

    terms: tuple[MatrixEquationTerm, ...]
    row_space: AbstractVectorSpace
    column_space: AbstractVectorSpace
    source: ArraySpace
    target: ArraySpace

    def __init__(
        self,
        terms: Sequence[MatrixEquationTerm],
        /,
        *,
        dtype: Any | None = None,
        operator_id: str | None = None,
    ):
        terms_ = tuple(terms)
        if not terms_ or any(not isinstance(term, MatrixEquationTerm) for term in terms_):
            raise TypeError(
                "terms must be a nonempty sequence of MatrixEquationTerm values."
            )
        row_space = terms_[0].left.source
        column_space = terms_[0].right.source
        for term in terms_:
            if not term.left.source.compatible(row_space):
                raise ValueError("All left terms must act on compatible row spaces.")
            if not term.right.source.compatible(column_space):
                raise ValueError("All right terms must act on compatible column spaces.")
        inferred_dtype = jnp.result_type(
            *(
                tuple(_coordinate_dtype(term.left.source) for term in terms_)
                + tuple(_coordinate_dtype(term.right.source) for term in terms_)
                + tuple(term.coefficient.dtype for term in terms_)
            )
        )
        dtype_ = np.dtype(
            jax.dtypes.canonicalize_dtype(
                inferred_dtype
                if dtype is None
                else jnp.result_type(inferred_dtype, dtype)
            )
        )
        if not np.issubdtype(dtype_, np.inexact):
            dtype_ = np.dtype(float)
        matrix_space = ArraySpace(
            (row_space.size, column_space.size),
            dtype=dtype_,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "matrix-equation-operator",
                    "rows": row_space.size,
                    "columns": column_space.size,
                    "dtype": dtype_.str,
                    "terms": [
                        {
                            "left": term.left.operator_id,
                            "right": term.right.operator_id,
                        }
                        for term in terms_
                    ],
                    "coordinate_order": "row-major",
                }
            )
            if operator_id is None
            else str(operator_id)
        )
        if not identifier:
            raise ValueError("operator_id must be non-empty.")
        self.terms = terms_
        self.row_space = row_space
        self.column_space = column_space
        self.source = matrix_space
        self.target = matrix_space
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
        )
        self.batch_shape = ()
        self.operator_id = identifier

    @property
    def row_dimension(self) -> int:
        return self.row_space.size

    @property
    def column_dimension(self) -> int:
        return self.column_space.size

    def mv(self, vector: ArrayLike, /) -> Array:
        matrix = self.source.validate(vector)
        result = jnp.zeros_like(matrix)
        for term in self.terms:
            left_applied = jax.vmap(
                lambda column: _coordinate_operator_action(term.left, column),
                in_axes=1,
                out_axes=1,
            )(matrix)
            applied = jax.vmap(lambda row: _coordinate_transpose_action(term.right, row))(
                left_applied
            )
            result = result + term.coefficient.astype(result.dtype) * applied
        return result

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        matrix = self.target.validate(vector)
        result = jnp.zeros_like(matrix)
        for term in self.terms:
            left_applied = jax.vmap(
                lambda column: _coordinate_transpose_action(term.left, column),
                in_axes=1,
                out_axes=1,
            )(matrix)
            applied = jax.vmap(lambda row: _coordinate_operator_action(term.right, row))(
                left_applied
            )
            result = result + term.coefficient.astype(result.dtype) * applied
        return result

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        matrix = self.target.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(matrix)))

    def _materialize(self, /) -> Array:
        size = self.source.size
        basis = jnp.eye(size, dtype=self.source.dtype).reshape(
            (size, self.row_dimension, self.column_dimension)
        )
        images = jax.vmap(self.mv)(basis)
        return images.reshape((size, size)).T


class MatrixEquationProblem(StrictModule):
    """A finite linear matrix equation with one declared right-hand side."""

    operator: MatrixEquationLinearOperator
    right_hand_side: Array
    kind: MatrixEquationKind = eqx.field(static=True)
    expected_self_adjoint_solution: bool = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[MatrixEquationTerm],
        right_hand_side: ArrayLike,
        /,
        *,
        kind: MatrixEquationKind = "generalized",
        expected_self_adjoint_solution: bool = False,
        problem_id: str | None = None,
    ):
        if kind not in (
            "generalized",
            "sylvester",
            "continuous-lyapunov",
            "discrete-lyapunov",
        ):
            raise ValueError("Unknown matrix-equation kind.")
        rhs = jnp.asarray(right_hand_side)
        if rhs.ndim != 2:
            raise ValueError("right_hand_side must be one rank-two matrix.")
        if not jnp.issubdtype(rhs.dtype, jnp.number) or jnp.issubdtype(
            rhs.dtype, jnp.bool_
        ):
            raise TypeError("right_hand_side must contain real or complex values.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "matrix-equation-problem",
                    "equation_kind": kind,
                    "rows": int(rhs.shape[0]),
                    "columns": int(rhs.shape[1]),
                    "terms": [
                        {
                            "left": term.left.operator_id,
                            "right": term.right.operator_id,
                        }
                        for term in terms
                    ],
                    "expected_self_adjoint": bool(expected_self_adjoint_solution),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        equation_operator = MatrixEquationLinearOperator(
            terms,
            dtype=rhs.dtype,
            operator_id=f"{identifier}:operator",
        )
        expected_shape = (
            equation_operator.row_dimension,
            equation_operator.column_dimension,
        )
        if rhs.shape != expected_shape:
            raise ValueError(
                f"right_hand_side must have shape {expected_shape}; got {rhs.shape}."
            )
        rhs = rhs.astype(equation_operator.source.dtype)
        finite = jnp.all(jnp.isfinite(rhs))
        if isinstance(finite, jax_core.Tracer):
            rhs = eqx.error_if(
                rhs, ~finite, "Matrix-equation right-hand side must be finite."
            )
        elif not bool(finite):
            raise ValueError("Matrix-equation right-hand side must be finite.")
        expected_structure = bool(expected_self_adjoint_solution)
        if expected_structure:
            if expected_shape[0] != expected_shape[1]:
                raise ValueError("A self-adjoint matrix solution must be square.")
            symmetric_rhs = jnp.allclose(rhs, jnp.conj(rhs.T))
            if not isinstance(symmetric_rhs, jax_core.Tracer) and not bool(symmetric_rhs):
                raise ValueError(
                    "A declared self-adjoint solution requires a self-adjoint right-hand side."
                )
        self.operator = equation_operator
        self.right_hand_side = rhs
        self.kind = kind
        self.expected_self_adjoint_solution = expected_structure
        self.problem_id = identifier

    @property
    def row_dimension(self) -> int:
        return self.operator.row_dimension

    @property
    def column_dimension(self) -> int:
        return self.operator.column_dimension

    @property
    def terms(self) -> tuple[MatrixEquationTerm, ...]:
        return self.operator.terms


class MatrixEquationPolicy(StrictModule):
    """Linear backend, expected-structure tolerance, and aggregate failure policy."""

    linear: LinearSolvePolicy = eqx.field(static=True)
    structure_tolerance: float = eqx.field(static=True)
    failure: FailurePolicy = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear: LinearSolvePolicy | None = None,
        structure_tolerance: float = 1e-8,
        failure: FailurePolicy | None = None,
    ):
        linear_ = LinearSolvePolicy() if linear is None else linear
        tolerance = float(structure_tolerance)
        failure_ = FailurePolicy() if failure is None else failure
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be a LinearSolvePolicy or None.")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("structure_tolerance must be finite and non-negative.")
        if not isinstance(failure_, FailurePolicy):
            raise TypeError("failure must be a FailurePolicy or None.")
        self.linear = linear_
        self.structure_tolerance = tolerance
        self.failure = failure_


class MatrixEquationCostEstimate(StrictModule):
    """Static coefficient, forcing, Kronecker, and primitive-action estimates."""

    row_dimension: int = eqx.field(static=True)
    column_dimension: int = eqx.field(static=True)
    unknown_count: int = eqx.field(static=True)
    num_terms: int = eqx.field(static=True)
    primitive_actions_per_matvec: int = eqx.field(static=True)
    coefficient_storage_bytes: int = eqx.field(static=True)
    right_hand_side_bytes: int = eqx.field(static=True)
    explicit_kronecker_bytes: int = eqx.field(static=True)
    selected_backend: str = eqx.field(static=True)
    selected_method: str = eqx.field(static=True)


class MatrixEquationPlan(StrictModule):
    """Immutable matrix-equation plan backed by one Phydrax linear solve plan."""

    linear_plan: LinearSolvePlan = eqx.field(static=True)
    policy: MatrixEquationPolicy = eqx.field(static=True)
    cost: MatrixEquationCostEstimate = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedMatrixEquation(StrictModule):
    """Reusable numerical linear state and default matrix right-hand side."""

    problem: MatrixEquationProblem
    linear: PreparedLinearSolve
    plan: MatrixEquationPlan = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def numeric_version(self) -> Array:
        return self.linear.numeric_version


class MatrixEquationDiagnostics(StrictModule):
    """True matrix residual, expected structure, and delegated solve evidence."""

    residual_norm: Array
    relative_residual: Array
    linear_residual_norm: Array
    linear_relative_residual: Array
    self_adjoint_error: Array
    structure_satisfied: Array
    finite: Array
    converged: Array
    linear_status: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    matvec_count: Array
    adjoint_matvec_count: Array
    primitive_actions_per_matvec: int = eqx.field(static=True)


class MatrixEquationProvenance(StrictModule):
    """Equation convention, delegated backend, identities, and numerical version."""

    kind: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    numeric_version: Array


class MatrixEquationResult(StrictModule):
    """Matrix solution with aggregate and delegated numerical evidence."""

    value: Array
    status: Array
    diagnostics: MatrixEquationDiagnostics
    provenance: MatrixEquationProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(MatrixEquationStatus.SUCCESS)


def sylvester_equation(
    left: AbstractLinearOperator | ArrayLike,
    right: AbstractLinearOperator | ArrayLike,
    forcing: ArrayLike,
    /,
    *,
    problem_id: str | None = None,
) -> MatrixEquationProblem:
    """Construct ``A X + X B = C``."""
    left_ = _coerce_square_operator(left, "left")
    right_ = _coerce_square_operator(right, "right")
    terms = (
        MatrixEquationTerm(left_, IdentityLinearOperator(right_.source)),
        MatrixEquationTerm(IdentityLinearOperator(left_.source), right_),
    )
    return MatrixEquationProblem(
        terms,
        forcing,
        kind="sylvester",
        problem_id=problem_id,
    )


def continuous_lyapunov_equation(
    operator: AbstractLinearOperator | ArrayLike,
    forcing: ArrayLike,
    /,
    *,
    problem_id: str | None = None,
) -> MatrixEquationProblem:
    """Construct the pairing-aware equation ``A X + X A* = -Q``."""
    operator_ = _coerce_square_operator(operator, "operator")
    identity = IdentityLinearOperator(operator_.source)
    terms = (
        MatrixEquationTerm(operator_, identity),
        MatrixEquationTerm(identity, adjoint(operator_)),
    )
    return MatrixEquationProblem(
        terms,
        -jnp.asarray(forcing),
        kind="continuous-lyapunov",
        expected_self_adjoint_solution=True,
        problem_id=problem_id,
    )


def discrete_lyapunov_equation(
    operator: AbstractLinearOperator | ArrayLike,
    forcing: ArrayLike,
    /,
    *,
    problem_id: str | None = None,
) -> MatrixEquationProblem:
    """Construct the pairing-aware equation ``X - A X A* = Q``."""
    operator_ = _coerce_square_operator(operator, "operator")
    identity = IdentityLinearOperator(operator_.source)
    terms = (
        MatrixEquationTerm(identity, identity),
        MatrixEquationTerm(operator_, adjoint(operator_), coefficient=-1.0),
    )
    return MatrixEquationProblem(
        terms,
        forcing,
        kind="discrete-lyapunov",
        expected_self_adjoint_solution=True,
        problem_id=problem_id,
    )


def plan_matrix_equation(
    problem: MatrixEquationProblem,
    policy: MatrixEquationPolicy | None = None,
    /,
) -> MatrixEquationPlan:
    """Plan the induced matrix-free or materialized linear solve."""
    if not isinstance(problem, MatrixEquationProblem):
        raise TypeError("problem must be a MatrixEquationProblem.")
    selected = MatrixEquationPolicy() if policy is None else policy
    if not isinstance(selected, MatrixEquationPolicy):
        raise TypeError("policy must be a MatrixEquationPolicy or None.")
    linear_problem = _linear_problem(problem)
    linear_plan = plan_linear_solve(linear_problem, selected.linear)
    cost = _matrix_equation_cost(problem, linear_plan)
    return MatrixEquationPlan(
        linear_plan=linear_plan,
        policy=selected,
        cost=cost,
        problem_id=problem.problem_id,
        operator_id=problem.operator.operator_id,
        kind=problem.kind,
        plan_id=canonical_fingerprint(
            {
                "kind": "matrix-equation-plan",
                "problem": problem.problem_id,
                "operator": problem.operator.operator_id,
                "linear_plan": linear_plan.plan_id,
                "structure_tolerance": selected.structure_tolerance,
            }
        ),
    )


def prepare_matrix_equation(
    problem: MatrixEquationProblem,
    policy: MatrixEquationPolicy | MatrixEquationPlan | None = None,
    /,
) -> PreparedMatrixEquation:
    """Prepare reusable numerical state for a matrix-equation operator."""
    plan = (
        policy
        if isinstance(policy, MatrixEquationPlan)
        else plan_matrix_equation(problem, policy)
    )
    _validate_plan(problem, plan)
    linear = prepare_linear_solve(_linear_problem(problem), plan.linear_plan)
    return PreparedMatrixEquation(
        problem=problem,
        linear=linear,
        plan=plan,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-matrix-equation",
                "plan": plan.plan_id,
                "template": linear.template.template_id,
                "operator": canonical_fingerprint(
                    array_tree_fingerprint(problem.operator)
                ),
            }
        ),
    )


def refresh_matrix_equation(
    prepared: PreparedMatrixEquation,
    problem: MatrixEquationProblem,
    /,
) -> PreparedMatrixEquation:
    """Refresh equation coefficients while preserving the symbolic solve plan."""
    if not isinstance(prepared, PreparedMatrixEquation):
        raise TypeError("prepared must be a PreparedMatrixEquation.")
    _validate_plan(problem, prepared.plan)
    linear = refresh_linear_solve(prepared.linear, _linear_problem(problem))
    return PreparedMatrixEquation(
        problem=problem,
        linear=linear,
        plan=prepared.plan,
        prepared_id=prepared.prepared_id,
    )


def solve_matrix_equation(
    problem_or_prepared: MatrixEquationProblem | PreparedMatrixEquation,
    /,
    *,
    right_hand_side: ArrayLike | None = None,
    policy: MatrixEquationPolicy | MatrixEquationPlan | None = None,
    initial_guess: ArrayLike | None = None,
) -> MatrixEquationResult:
    """Solve a linear matrix equation with true residual and structure checks."""
    if isinstance(problem_or_prepared, PreparedMatrixEquation):
        if policy is not None:
            raise ValueError("policy must be omitted for a prepared matrix equation.")
        prepared = problem_or_prepared
    elif isinstance(problem_or_prepared, MatrixEquationProblem):
        prepared = prepare_matrix_equation(problem_or_prepared, policy)
    else:
        raise TypeError("Expected a MatrixEquationProblem or PreparedMatrixEquation.")
    problem = prepared.problem
    rhs = (
        problem.right_hand_side
        if right_hand_side is None
        else _matrix_value(problem.operator.target, right_hand_side, "right_hand_side")
    )
    guess = (
        None
        if initial_guess is None
        else _matrix_value(problem.operator.source, initial_guess, "initial_guess")
    )
    linear_result = solve_linear_system(prepared.linear, rhs, initial_guess=guess)
    value = jnp.asarray(linear_result.value)
    residual_matrix = problem.operator.mv(value) - rhs
    residual = jnp.linalg.norm(residual_matrix)
    rhs_norm = jnp.linalg.norm(rhs)
    tiny = jnp.asarray(jnp.finfo(residual.real.dtype).tiny)
    relative_residual = residual / jnp.maximum(rhs_norm, tiny)
    if problem.expected_self_adjoint_solution:
        self_adjoint_error = jnp.linalg.norm(value - jnp.conj(value.T)) / jnp.maximum(
            jnp.linalg.norm(value), tiny
        )
    else:
        self_adjoint_error = jnp.asarray(jnp.nan, dtype=residual.real.dtype)
    structure_satisfied = (
        self_adjoint_error <= prepared.plan.policy.structure_tolerance
        if problem.expected_self_adjoint_solution
        else jnp.asarray(True)
    )
    finite = jnp.all(jnp.isfinite(value)) & jnp.isfinite(residual)
    linear_succeeded = linear_result.status == int(LinearSolveStatus.SUCCESS)
    converged = finite & linear_succeeded & structure_satisfied
    status = jnp.where(
        ~finite,
        int(MatrixEquationStatus.NONFINITE),
        jnp.where(
            ~linear_succeeded,
            int(MatrixEquationStatus.LINEAR_SOLVE_FAILURE),
            jnp.where(
                ~structure_satisfied,
                int(MatrixEquationStatus.STRUCTURE_TOLERANCE_NOT_MET),
                int(MatrixEquationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    if prepared.plan.policy.failure.mode == "error":
        value = eqx.error_if(
            value,
            status != int(MatrixEquationStatus.SUCCESS),
            "Matrix equation did not satisfy its numerical contract.",
        )
    diagnostics = MatrixEquationDiagnostics(
        residual_norm=residual,
        relative_residual=relative_residual,
        linear_residual_norm=linear_result.diagnostics.residual_norm,
        linear_relative_residual=linear_result.diagnostics.relative_residual,
        self_adjoint_error=self_adjoint_error,
        structure_satisfied=structure_satisfied,
        finite=finite,
        converged=converged,
        linear_status=linear_result.status,
        iterations=linear_result.diagnostics.iterations,
        rank=linear_result.diagnostics.rank,
        condition_estimate=linear_result.diagnostics.condition_estimate,
        matvec_count=linear_result.diagnostics.matvec_count,
        adjoint_matvec_count=linear_result.diagnostics.adjoint_matvec_count,
        primitive_actions_per_matvec=prepared.plan.cost.primitive_actions_per_matvec,
    )
    return MatrixEquationResult(
        value=value,
        status=status,
        diagnostics=diagnostics,
        provenance=MatrixEquationProvenance(
            kind=problem.kind,
            convention=_equation_convention(problem.kind),
            backend=linear_result.provenance.backend,
            method=linear_result.provenance.method,
            problem_id=problem.problem_id,
            plan_id=prepared.plan.plan_id,
            prepared_id=prepared.prepared_id,
            operator_id=problem.operator.operator_id,
            numeric_version=prepared.numeric_version,
        ),
    )


def _linear_problem(problem: MatrixEquationProblem, /) -> LinearSystem:
    return LinearSystem(problem.operator, problem_id=problem.problem_id)


def _matrix_equation_cost(
    problem: MatrixEquationProblem,
    linear_plan: LinearSolvePlan,
    /,
) -> MatrixEquationCostEstimate:
    arrays = {
        id(leaf): leaf for leaf in jax.tree.leaves(problem.operator) if eqx.is_array(leaf)
    }
    coefficient_bytes = sum(
        int(array.size * array.dtype.itemsize) for array in arrays.values()
    )
    unknowns = problem.row_dimension * problem.column_dimension
    itemsize = problem.operator.source.dtype.itemsize
    return MatrixEquationCostEstimate(
        row_dimension=problem.row_dimension,
        column_dimension=problem.column_dimension,
        unknown_count=unknowns,
        num_terms=len(problem.terms),
        primitive_actions_per_matvec=len(problem.terms)
        * (problem.row_dimension + problem.column_dimension),
        coefficient_storage_bytes=coefficient_bytes,
        right_hand_side_bytes=int(problem.right_hand_side.nbytes),
        explicit_kronecker_bytes=unknowns * unknowns * itemsize,
        selected_backend=linear_plan.backend,
        selected_method=linear_plan.method,
    )


def _validate_plan(problem: MatrixEquationProblem, plan: MatrixEquationPlan, /) -> None:
    if not isinstance(problem, MatrixEquationProblem):
        raise TypeError("problem must be a MatrixEquationProblem.")
    if not isinstance(plan, MatrixEquationPlan):
        raise TypeError("plan must be a MatrixEquationPlan.")
    if (
        problem.problem_id != plan.problem_id
        or problem.operator.operator_id != plan.operator_id
        or problem.kind != plan.kind
    ):
        raise ValueError("Matrix-equation plan belongs to a different symbolic problem.")


def _coerce_square_operator(
    value: AbstractLinearOperator | ArrayLike,
    name: str,
    /,
) -> AbstractLinearOperator:
    operator = (
        value if isinstance(value, AbstractLinearOperator) else DenseLinearOperator(value)
    )
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError(f"{name} must be an unbatched endomorphism.")
    return operator


def _coordinate_operator_action(
    operator: AbstractLinearOperator,
    coordinates: Array,
    /,
) -> Array:
    return _promoted_coordinate_action(operator, coordinates, transpose=False)


def _coordinate_transpose_action(
    operator: AbstractLinearOperator,
    coordinates: Array,
    /,
) -> Array:
    return _promoted_coordinate_action(operator, coordinates, transpose=True)


def _promoted_coordinate_action(
    operator: AbstractLinearOperator,
    coordinates: Array,
    /,
    *,
    transpose: bool,
) -> Array:
    native_dtype = _coordinate_dtype(operator.source)

    def native_action(value):
        native = value.astype(native_dtype)
        if transpose:
            return operator.source.flatten(
                operator.transpose_mv(operator.target.unflatten(native))
            )
        return operator.target.flatten(operator.mv(operator.source.unflatten(native)))

    if jnp.issubdtype(coordinates.dtype, jnp.complexfloating) and not jnp.issubdtype(
        native_dtype, jnp.complexfloating
    ):
        real = native_action(jnp.real(coordinates))
        imaginary = native_action(jnp.imag(coordinates))
        return (real + 1j * imaginary).astype(coordinates.dtype)
    return native_action(coordinates).astype(coordinates.dtype)


def _matrix_value(space: ArraySpace, value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != space.shape:
        raise ValueError(f"{name} must have shape {space.shape}; got {array.shape}.")
    if np.dtype(array.dtype) != space.dtype:
        raise TypeError(f"{name} must have dtype {space.dtype}; got {array.dtype}.")
    return array


def _equation_convention(kind: str, /) -> str:
    return {
        "generalized": "sum_i c_i A_i X B_i = C",
        "sylvester": "A X + X B = C",
        "continuous-lyapunov": "A X + X A* = -Q",
        "discrete-lyapunov": "X - A X A* = Q",
    }[kind]


__all__ = [
    "MatrixEquationCostEstimate",
    "MatrixEquationDiagnostics",
    "MatrixEquationKind",
    "MatrixEquationLinearOperator",
    "MatrixEquationPlan",
    "MatrixEquationPolicy",
    "MatrixEquationProblem",
    "MatrixEquationProvenance",
    "MatrixEquationResult",
    "MatrixEquationStatus",
    "MatrixEquationTerm",
    "PreparedMatrixEquation",
    "continuous_lyapunov_equation",
    "discrete_lyapunov_equation",
    "plan_matrix_equation",
    "prepare_matrix_equation",
    "refresh_matrix_equation",
    "solve_matrix_equation",
    "sylvester_equation",
]
